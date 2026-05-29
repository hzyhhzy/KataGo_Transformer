#!/usr/bin/python3
import argparse
import copy
import logging
from typing import Dict, List, Tuple

import torch

import load_model
import modelconfigs
import train_muon_ki as muon_train
from metrics_pytorch import Metrics, huber_loss


NUM_RULE_HEADS = 6
RULE_HEAD_LOSS_WEIGHTS = [1.0, 0.5, 0.5, 0.5, 0.2, 0.2]
assert len(RULE_HEAD_LOSS_WEIGHTS) == NUM_RULE_HEADS


def make_teacher_global_inputs(row_global: torch.Tensor) -> List[torch.Tensor]:
    assert row_global.shape[1] == 39, f"Expected 39 v102 global channels, got {row_global.shape[1]}"

    rule0 = row_global

    rule1 = row_global.clone()
    rule1[:, 17:30] = 0

    rule2 = rule1.clone()
    rule2[:, 13] = 0.8

    rule3 = rule1.clone()
    rule3[:, 13] = -0.8

    rule4 = row_global.clone()
    rule4[:, 13:15] = 0
    rule4[:, 17:38] = 0
    rule4[:, 22] = 1

    rule5 = row_global.clone()
    rule5[:, 13:15] = 0
    rule5[:, 17:38] = 0
    rule5[:, 27] = 1

    return [rule0, rule1, rule2, rule3, rule4, rule5]


class MultiRuleDistillMetrics(Metrics):
    teacher_checkpoint = None
    teacher_use_swa = False

    def __init__(self, batch_size: int, world_size: int, raw_model):
        super().__init__(batch_size, world_size, raw_model)
        assert self.teacher_checkpoint is not None, "Teacher checkpoint was not configured"
        assert raw_model.config["version"] == 102, "Multi-rule distillation only supports v102 students"
        assert raw_model.get_num_rule_distill_heads() == NUM_RULE_HEADS, (
            f"Student must have {NUM_RULE_HEADS} rule heads, got {raw_model.get_num_rule_distill_heads()}"
        )

        teacher, teacher_swa, _ = load_model.load_model(
            self.teacher_checkpoint,
            use_swa=self.teacher_use_swa,
            device=raw_model.device,
            pos_len=raw_model.pos_len,
        )
        self.teacher = teacher_swa if self.teacher_use_swa else teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        assert self.teacher.config["version"] == 102, "Multi-rule distillation only supports v102 teachers"
        assert self.teacher.get_num_rule_distill_heads() == 1, "Teacher must have the ordinary single v102 output head"
        assert not self.teacher.get_has_metadata_encoder() or raw_model.get_has_metadata_encoder(), (
            "Teacher uses metadata, but the student/data path does not load metadata"
        )

    @staticmethod
    def _kl_from_logits(pred_logits: torch.Tensor, target_logits: torch.Tensor, dim: int) -> torch.Tensor:
        target_log_probs = torch.log_softmax(target_logits.detach(), dim=dim)
        target_probs = torch.exp(target_log_probs)
        pred_log_probs = torch.log_softmax(pred_logits, dim=dim)
        return torch.sum(target_probs * (target_log_probs - pred_log_probs), dim=dim)

    @staticmethod
    def _weighted_sum(samplewise_loss: torch.Tensor, sample_weight: torch.Tensor, global_weight: torch.Tensor) -> torch.Tensor:
        return torch.sum(samplewise_loss * sample_weight * global_weight)

    @staticmethod
    def _huber_mean_samplewise(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        losses = huber_loss(pred, target.detach(), delta=delta)
        return losses.reshape(losses.shape[0], -1).mean(dim=1)

    @staticmethod
    def _masked_mean_samplewise(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_sum = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return (loss * mask).sum(dim=(1, 2)) / mask_sum

    def _masked_mse_samplewise(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = torch.square(pred - target.detach()).mean(dim=1)
        return self._masked_mean_samplewise(loss, mask)

    def _masked_seki_kl_samplewise(self, pred_logits: torch.Tensor, target_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self._kl_from_logits(pred_logits, target_logits, dim=1)
        return self._masked_mean_samplewise(loss, mask)

    @staticmethod
    def _distill_weights(batch, meta_kata_only_soft_policy):
        target_global_nc = batch["globalTargetsNC"]
        weights = {
            "global": target_global_nc[:, 25],
            "policy_player": target_global_nc[:, 26],
            "ownership": target_global_nc[:, 27],
            "policy_opponent": target_global_nc[:, 28],
            "lead": target_global_nc[:, 29],
            "futurepos": target_global_nc[:, 33],
            "scoring": target_global_nc[:, 34],
            "value": 1.0 - target_global_nc[:, 35],
            "td_value": 1.0 - target_global_nc[:, 24],
        }
        weights["policy_player_soft"] = weights["policy_player"]
        weights["policy_opponent_soft"] = weights["policy_opponent"]
        if meta_kata_only_soft_policy:
            metadata_input_nc = batch["metadataInputNC"]
            kata_weight = metadata_input_nc[:, 151]
            weights["policy_player_soft"] = weights["policy_player_soft"] * kata_weight
            weights["policy_opponent_soft"] = weights["policy_opponent_soft"] * kata_weight
        return weights

    @staticmethod
    def _split_head_batch(output_head: Tuple[torch.Tensor, ...]) -> List[Tuple[torch.Tensor, ...]]:
        output_chunks = [tensor.chunk(NUM_RULE_HEADS, dim=0) for tensor in output_head]
        return [tuple(chunks[rule_idx] for chunks in output_chunks) for rule_idx in range(NUM_RULE_HEADS)]

    def _teacher_targets(self, batch) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple[torch.Tensor, ...]]]:
        row_spatial = batch["binaryInputNCHW"]
        row_global = batch["globalInputNC"]
        teacher_global_inputs = make_teacher_global_inputs(row_global)
        teacher_spatial = torch.cat([row_spatial] * NUM_RULE_HEADS, dim=0)
        teacher_global = torch.cat(teacher_global_inputs, dim=0)

        teacher_meta = None
        if self.teacher.get_has_metadata_encoder():
            teacher_meta = torch.cat([batch["metadataInputNC"]] * NUM_RULE_HEADS, dim=0)

        with torch.no_grad():
            teacher_outputs = self.teacher(teacher_spatial, teacher_global, input_meta=teacher_meta)
            teacher_outputs = self.teacher.float32ify_output(teacher_outputs)
            teacher_outputs = self.teacher.postprocess_output(teacher_outputs)

        main_targets = self._split_head_batch(teacher_outputs[0])
        if self.teacher.get_has_intermediate_head():
            intermediate_targets = self._split_head_batch(teacher_outputs[1])
        else:
            intermediate_targets = main_targets
        return main_targets, intermediate_targets

    def _distill_single_output(
        self,
        pred,
        target,
        mask: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        soft_policy_weight_scale: float,
        value_loss_scale: float,
        td_value_loss_scales: List[float],
        seki_loss_scale: float,
        variance_time_loss_scale: float,
    ) -> Dict[str, torch.Tensor]:
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_pretanh,
            pred_scoring,
            futurepos_pretanh,
            seki_logits,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            scorebelief_logits,
        ) = pred
        (
            target_policy_logits,
            target_value_logits,
            target_td_value_logits,
            target_td_score,
            target_ownership_pretanh,
            target_scoring,
            target_futurepos_pretanh,
            target_seki_logits,
            target_scoremean,
            target_scorestdev,
            target_lead,
            target_variance_time,
            target_shortterm_value_error,
            target_shortterm_score_error,
            target_scorebelief_logits,
        ) = target

        global_weight = weights["global"]
        policy_kl = self._kl_from_logits(policy_logits, target_policy_logits, dim=2)
        policy_player_loss = self._weighted_sum(policy_kl[:, 0], weights["policy_player"], global_weight)
        policy_opponent_loss = 0.15 * self._weighted_sum(
            policy_kl[:, 1], weights["policy_opponent"], global_weight
        )
        policy_player_soft_loss = self._weighted_sum(
            policy_kl[:, 2], weights["policy_player_soft"], global_weight
        )
        policy_opponent_soft_loss = 0.15 * self._weighted_sum(
            policy_kl[:, 3], weights["policy_opponent_soft"], global_weight
        )

        value_loss = self._weighted_sum(
            self._kl_from_logits(value_logits, target_value_logits, dim=1),
            weights["value"],
            global_weight,
        )

        td_value_kl = self._kl_from_logits(td_value_logits, target_td_value_logits, dim=2)
        td_value_losses = []
        for td_idx, td_value_loss_scale in enumerate(td_value_loss_scales):
            del td_value_loss_scale
            td_value_losses.append(self._weighted_sum(
                td_value_kl[:, td_idx],
                weights["td_value"],
                global_weight,
            ))

        td_score_loss = self._weighted_sum(
            self._huber_mean_samplewise(pred_td_score, target_td_score, delta=12.0),
            weights["ownership"],
            global_weight,
        )
        ownership_loss = self._weighted_sum(
            self._masked_mse_samplewise(
                torch.tanh(ownership_pretanh),
                torch.tanh(target_ownership_pretanh),
                mask,
            ),
            weights["ownership"],
            global_weight,
        )
        scoring_loss = self._weighted_sum(
            self._masked_mse_samplewise(pred_scoring, target_scoring, mask),
            weights["scoring"],
            global_weight,
        )
        futurepos_loss = self._weighted_sum(
            self._masked_mse_samplewise(
                torch.tanh(futurepos_pretanh),
                torch.tanh(target_futurepos_pretanh),
                mask,
            ),
            weights["futurepos"],
            global_weight,
        )
        seki_loss = self._weighted_sum(
            self._masked_seki_kl_samplewise(seki_logits, target_seki_logits, mask),
            weights["ownership"],
            global_weight,
        )

        scoremean_loss = self._weighted_sum(
            self._huber_mean_samplewise(pred_scoremean, target_scoremean, delta=12.0),
            weights["ownership"],
            global_weight,
        )
        scorestdev_loss = self._weighted_sum(
            self._huber_mean_samplewise(pred_scorestdev, target_scorestdev, delta=10.0),
            torch.ones_like(global_weight),
            global_weight,
        )
        lead_loss = self._weighted_sum(
            self._huber_mean_samplewise(pred_lead, target_lead, delta=8.0),
            weights["lead"],
            global_weight,
        )
        variance_time_loss = self._weighted_sum(
            self._huber_mean_samplewise(pred_variance_time, target_variance_time, delta=50.0),
            weights["ownership"],
            global_weight,
        )
        shortterm_value_error_loss = self._weighted_sum(
            self._huber_mean_samplewise(
                pred_shortterm_value_error,
                target_shortterm_value_error,
                delta=0.4,
            ),
            weights["ownership"],
            global_weight,
        )
        shortterm_score_error_loss = self._weighted_sum(
            self._huber_mean_samplewise(
                pred_shortterm_score_error,
                target_shortterm_score_error,
                delta=100.0,
            ),
            weights["ownership"],
            global_weight,
        )
        scorebelief_loss = self._weighted_sum(
            self._kl_from_logits(scorebelief_logits, target_scorebelief_logits, dim=1),
            weights["ownership"],
            global_weight,
        )

        loss_sum = (
            policy_player_loss
            + policy_opponent_loss
            + policy_player_soft_loss * soft_policy_weight_scale
            + policy_opponent_soft_loss * soft_policy_weight_scale
            + value_loss * value_loss_scale
            + td_value_losses[0] * td_value_loss_scales[0]
            + td_value_losses[1] * td_value_loss_scales[1]
            + td_value_losses[2] * td_value_loss_scales[2]
            + td_score_loss * 0.0004 * 0.0
            + ownership_loss * 0.0
            + scoring_loss * 0.5 * 0.0
            + futurepos_loss * 0.25
            + seki_loss * seki_loss_scale * 0.0
            + scoremean_loss * 0.0015 * 0.0
            + scorebelief_loss * 0.02 * 0.0
            + scorestdev_loss * 0.001 * 0.0
            + lead_loss * 0.006 * 0.0
            + variance_time_loss * variance_time_loss_scale * 0.0003
            + shortterm_value_error_loss * 2.0
            + shortterm_score_error_loss * 0.00002 * 0.0
        )
        return {
            "p0loss_sum": policy_player_loss,
            "p1loss_sum": policy_opponent_loss,
            "p0softloss_sum": policy_player_soft_loss,
            "p1softloss_sum": policy_opponent_soft_loss,
            "vloss_sum": value_loss,
            "tdvloss1_sum": td_value_losses[0],
            "tdvloss2_sum": td_value_losses[1],
            "tdvloss3_sum": td_value_losses[2],
            "tdsloss_sum": td_score_loss,
            "oloss_sum": ownership_loss,
            "sloss_sum": scoring_loss,
            "fploss_sum": futurepos_loss,
            "skloss_sum": seki_loss,
            "smloss_sum": scoremean_loss,
            "sbloss_sum": scorebelief_loss,
            "sdregloss_sum": scorestdev_loss,
            "leadloss_sum": lead_loss,
            "vtimeloss_sum": variance_time_loss,
            "evstloss_sum": shortterm_value_error_loss,
            "esstloss_sum": shortterm_score_error_loss,
            "loss_sum": loss_sum,
        }

    @staticmethod
    def _accumulate_metrics(
        results: Dict[str, torch.Tensor],
        head_results: Dict[str, torch.Tensor],
        scale: float,
        prefix: str,
    ):
        for key, value in head_results.items():
            results[key] = results.get(key, 0.0) + value * scale
            results[prefix + key] = value

    def metrics_dict_batchwise(
        self,
        raw_model,
        model_output_postprocessed_byheads,
        extra_outputs,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        seki_loss_scale,
        variance_time_loss_scale,
        main_loss_scale,
        intermediate_loss_scale,
    ):
        del extra_outputs
        del is_training
        del disable_optimistic_policy

        assert raw_model.config["version"] == 102, "Multi-rule distillation only supports v102 students"
        assert raw_model.get_num_rule_distill_heads() == NUM_RULE_HEADS
        expected_outputs = NUM_RULE_HEADS * (2 if raw_model.get_has_intermediate_head() else 1)
        assert len(model_output_postprocessed_byheads) == expected_outputs, (
            f"Expected {expected_outputs} student output head sets, got {len(model_output_postprocessed_byheads)}"
        )

        mask = batch["binaryInputNCHW"][:, 0, :, :].contiguous()
        weights = self._distill_weights(batch, meta_kata_only_soft_policy)
        main_targets, intermediate_targets = self._teacher_targets(batch)
        results = {}
        main_scale = 1.0 if main_loss_scale is None else main_loss_scale
        intermediate_scale = 1.0 if intermediate_loss_scale is None else intermediate_loss_scale

        for rule_idx in range(NUM_RULE_HEADS):
            rule_scale = RULE_HEAD_LOSS_WEIGHTS[rule_idx]
            head_results = self._distill_single_output(
                model_output_postprocessed_byheads[rule_idx],
                main_targets[rule_idx],
                mask,
                weights,
                soft_policy_weight_scale,
                value_loss_scale,
                td_value_loss_scales,
                seki_loss_scale,
                variance_time_loss_scale,
            )
            self._accumulate_metrics(results, head_results, main_scale * rule_scale, f"r{rule_idx}_")

            if raw_model.get_has_intermediate_head():
                intermediate_results = self._distill_single_output(
                    model_output_postprocessed_byheads[NUM_RULE_HEADS + rule_idx],
                    intermediate_targets[rule_idx],
                    mask,
                    weights,
                    soft_policy_weight_scale,
                    value_loss_scale,
                    td_value_loss_scales,
                    seki_loss_scale,
                    variance_time_loss_scale,
                )
                self._accumulate_metrics(
                    results,
                    intermediate_results,
                    intermediate_scale * rule_scale,
                    f"Ir{rule_idx}_",
                )

        (
            modelnorm_normal,
            modelnorm_normal_gamma,
            modelnorm_normal_attn,
            modelnorm_output,
            modelnorm_noreg,
            modelnorm_output_noreg,
        ) = self.get_model_norms(raw_model)
        results.update({
            "wsum": weights["global"].sum() * self.world_size,
            "nsamp": mask.shape[0] * self.world_size,
            "norm_normal_batch": modelnorm_normal,
            "norm_normal_gamma_batch": modelnorm_normal_gamma,
            "norm_normal_attn_batch": modelnorm_normal_attn,
            "norm_output_batch": modelnorm_output,
            "norm_noreg_batch": modelnorm_noreg,
            "norm_output_noreg_batch": modelnorm_output_noreg,
        })
        return results


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a six-rule v102 output-head student from a v102 teacher.",
        add_help=False,
    )
    required_args = parser.add_argument_group("required arguments")
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS)

    required_args.add_argument("-traindir", required=True)
    required_args.add_argument("-datadir", required=True)
    required_args.add_argument("-teacher-checkpoint", required=True)
    optional_args.add_argument("-teacher-use-swa", required=False, action="store_true")
    optional_args.add_argument("-exportdir", required=False)
    optional_args.add_argument("-exportprefix", required=False)
    optional_args.add_argument("-initial-checkpoint", required=False)

    required_args.add_argument("-pos-len", type=int, required=True)
    required_args.add_argument("-batch-size", type=int, required=True)
    optional_args.add_argument("-samples-per-epoch", type=int, required=False)
    optional_args.add_argument("-history-matrices-type", type=str, default="", required=False)
    optional_args.add_argument("-symmetry-type", type=str, default="xyt", required=False)

    optional_args.add_argument("-model-kind", required=False)
    optional_args.add_argument("-lr-base", type=float, default=6e-6, required=False)
    optional_args.add_argument("-lr-scale", type=float, required=False)
    optional_args.add_argument("-lr-scale-auto-type", type=str, required=False, default="")
    optional_args.add_argument("-wd-scale", type=float, default=1.0, required=False)
    optional_args.add_argument("-muon-momentum", type=float, default=0.95, required=False)
    optional_args.add_argument("-gnorm-clip-scale", type=float, required=False)
    optional_args.add_argument("-sub-epochs", type=int, default=1, required=False)
    optional_args.add_argument("-swa-period-samples", type=float, required=False)
    optional_args.add_argument("-swa-scales", type=str, required=False)

    optional_args.add_argument("-multi-gpus", required=False)
    optional_args.add_argument("-use-fp16", required=False, action="store_true")
    optional_args.add_argument("-qat-int8", required=False, action="store_true")
    optional_args.add_argument("-master-port", default=23456, type=int, required=False)
    optional_args.add_argument("-no-compile", required=False, action="store_true")

    optional_args.add_argument("-epochs-per-export", type=int, required=False)
    optional_args.add_argument("-export-prob", type=float, required=False)
    optional_args.add_argument("-max-epochs-this-instance", type=int, required=False)
    optional_args.add_argument("-max-training-samples", type=int, required=False)
    optional_args.add_argument("-sleep-seconds-per-epoch", type=int, required=False)
    optional_args.add_argument("-max-train-bucket-per-new-data", type=float, required=False)
    optional_args.add_argument("-max-train-bucket-size", type=float, required=False)
    optional_args.add_argument("-max-train-steps-since-last-reload", type=float, required=False)
    optional_args.add_argument("-stop-when-train-bucket-limited", required=False, action="store_true")
    optional_args.add_argument("-max-val-samples", type=int, required=False)
    optional_args.add_argument("-randomize-val", required=False, action="store_true")
    optional_args.add_argument("-no-export", required=False, action="store_true")
    optional_args.add_argument("-no-repeat-files", required=False, action="store_true")
    optional_args.add_argument("-quit-if-no-data", required=False, action="store_true")
    optional_args.add_argument("-gnorm-stats-debug", required=False, action="store_true")

    optional_args.add_argument("-lookahead-k", type=int, default=6, required=False)
    optional_args.add_argument("-lookahead-alpha", type=float, default=1.0, required=False)
    optional_args.add_argument("-lookahead-print", required=False, action="store_true")
    optional_args.add_argument("-brenorm-avg-momentum", type=float, required=False)
    optional_args.add_argument("-brenorm-target-rmax", type=float, required=False)
    optional_args.add_argument("-brenorm-target-dmax", type=float, required=False)
    optional_args.add_argument("-brenorm-adjustment-scale", type=float, required=False)

    optional_args.add_argument("-soft-policy-weight-scale", type=float, default=8.0, required=False)
    optional_args.add_argument("-disable-optimistic-policy", required=False, action="store_true")
    optional_args.add_argument("-meta-kata-only-soft-policy", required=False, action="store_true")
    optional_args.add_argument("-value-loss-scale", type=float, default=0.6, required=False)
    optional_args.add_argument("-td-value-loss-scales", type=str, default="0.6,0.6,0.6", required=False)
    optional_args.add_argument("-seki-loss-scale", type=float, default=1.0, required=False)
    optional_args.add_argument("-variance-time-loss-scale", type=float, default=1.0, required=False)
    optional_args.add_argument("-main-loss-scale", type=float, required=False)
    optional_args.add_argument("-intermediate-loss-scale", type=float, required=False)
    return parser


def prepare_student_config(model_config):
    student_config = copy.deepcopy(model_config)
    assert student_config["version"] == 102, "Multi-rule distillation only supports v102 students"
    student_config["num_rule_distill_heads"] = NUM_RULE_HEADS
    return student_config


def prepare_initial_state_dict(state_dict, is_initial_checkpoint):
    if not is_initial_checkpoint:
        return state_dict

    config = state_dict.get("config")
    if config is None:
        return state_dict
    assert config["version"] == 102, "Multi-rule distillation only supports v102 initial checkpoints"

    checkpoint_num_heads = config.get("num_rule_distill_heads", 1)
    if checkpoint_num_heads == NUM_RULE_HEADS:
        return state_dict
    assert checkpoint_num_heads == 1, (
        f"Can only expand a single-head initial checkpoint to {NUM_RULE_HEADS} heads, got {checkpoint_num_heads}"
    )

    state_dict = dict(state_dict)
    state_dict["config"] = prepare_student_config(config)
    state_dict.pop("optimizer", None)
    for key in list(state_dict.keys()):
        if key == "swa_model" or key.startswith("swa_model_"):
            state_dict.pop(key)
    logging.info(
        f"Expanding single-head v102 initial checkpoint to {NUM_RULE_HEADS} distillation heads; "
        "starting optimizer and SWA state fresh."
    )
    return state_dict


def _duplicate_head_params(model_state_dict, source_prefix, dest_prefixes):
    for key, value in list(model_state_dict.items()):
        if not key.startswith(source_prefix):
            continue
        suffix = key[len(source_prefix):]
        for dest_prefix in dest_prefixes:
            model_state_dict.setdefault(dest_prefix + suffix, value)


def expand_initial_head_state_dict(raw_model, model_state_dict, is_initial_checkpoint):
    if not is_initial_checkpoint:
        return model_state_dict
    if raw_model.get_num_rule_distill_heads() != NUM_RULE_HEADS:
        return model_state_dict
    if any(key.startswith("rule_policy_heads.") for key in model_state_dict):
        return model_state_dict

    model_state_dict = dict(model_state_dict)
    extra_head_ids = range(NUM_RULE_HEADS - 1)
    _duplicate_head_params(
        model_state_dict,
        "policy_head.",
        [f"rule_policy_heads.{head_idx}." for head_idx in extra_head_ids],
    )
    _duplicate_head_params(
        model_state_dict,
        "value_head.",
        [f"rule_value_heads.{head_idx}." for head_idx in extra_head_ids],
    )
    if raw_model.get_has_intermediate_head():
        _duplicate_head_params(
            model_state_dict,
            "intermediate_policy_head.",
            [f"intermediate_rule_policy_heads.{head_idx}." for head_idx in extra_head_ids],
        )
        _duplicate_head_params(
            model_state_dict,
            "intermediate_value_head.",
            [f"intermediate_rule_value_heads.{head_idx}." for head_idx in extra_head_ids],
        )
    return model_state_dict


def configure_muon_training(args):
    MultiRuleDistillMetrics.teacher_checkpoint = args["teacher_checkpoint"]
    MultiRuleDistillMetrics.teacher_use_swa = args["teacher_use_swa"]
    muon_train.Metrics = MultiRuleDistillMetrics
    muon_train.prepare_model_config_for_training = prepare_student_config
    muon_train.prepare_loaded_state_dict_for_training = prepare_initial_state_dict
    muon_train.prepare_loaded_model_state_dict_for_training = expand_initial_head_state_dict

    model_kind = args["model_kind"]
    if model_kind is not None:
        assert model_kind in modelconfigs.config_of_name, f"Unknown model kind {model_kind}"
        assert modelconfigs.config_of_name[model_kind]["version"] == 102, (
            "Multi-rule distillation only supports v102 model kinds"
        )


def main(rank: int, world_size: int, args, multi_gpu_device_ids, readpipes, writepipes, barrier):
    configure_muon_training(args)
    muon_train.main(rank, world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier)


def run(args):
    configure_muon_training(args)

    multi_gpu_device_ids = []
    if args["multi_gpus"] is not None:
        for piece in args["multi_gpus"].split(","):
            multi_gpu_device_ids.append(int(piece.strip()))
        num_gpus_used = len(multi_gpu_device_ids)
    else:
        multi_gpu_device_ids = [0]
        num_gpus_used = 1

    muon_train.make_dirs(args)
    readpipes = []
    writepipes = []

    if num_gpus_used > 1:
        torch.multiprocessing.set_start_method("spawn")
        world_size = num_gpus_used
        barrier = torch.multiprocessing.Barrier(num_gpus_used)
        for _ in range(world_size - 1):
            rpipe, wpipe = torch.multiprocessing.Pipe()
            readpipes.append(rpipe)
            writepipes.append(wpipe)
        torch.multiprocessing.spawn(
            main,
            nprocs=num_gpus_used,
            args=(world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier),
        )
    else:
        main(0, 1, args, multi_gpu_device_ids, readpipes, writepipes, None)


if __name__ == "__main__":
    run(vars(build_parser().parse_args()))
