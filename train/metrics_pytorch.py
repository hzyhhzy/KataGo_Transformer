from typing import Any, Dict, List
import math

from model_pytorch import Model, compute_gain, ExtraOutputs, MetadataEncoder

import torch
import torch.nn
import torch.nn.functional

def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * torch.nn.functional.log_softmax(pred_logits, dim=dim), dim=dim)

def huber_loss(x, y, delta):
    abs_diff = torch.abs(x - y)
    return torch.where(
        abs_diff > delta,
        (0.5 * delta * delta) + delta * (abs_diff - delta),
        0.5 * abs_diff * abs_diff,
    )

def constant_like(data, other_tensor):
    return torch.tensor(data, dtype=other_tensor.dtype, device=other_tensor.device, requires_grad=False)

class Metrics:
    def __init__(self, batch_size: int, world_size: int, raw_model: Model):
        self.n = batch_size
        self.world_size = world_size
        self.pos_len = raw_model.pos_len
        self.pos_area = raw_model.pos_len * raw_model.pos_len
        self.policy_len = raw_model.pos_len * raw_model.pos_len + 1
        self.value_len = 3
        self.num_td_values = 3
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def state_dict(self):
        return dict(
            moving_unowned_proportion_sum = self.moving_unowned_proportion_sum,
            moving_unowned_proportion_weight = self.moving_unowned_proportion_weight,
        )
    def load_state_dict(self, state_dict: Dict[str,Any]):
        if isinstance(state_dict["moving_unowned_proportion_sum"],torch.Tensor):
            self.moving_unowned_proportion_sum = state_dict["moving_unowned_proportion_sum"].item()
        else:
            self.moving_unowned_proportion_sum = state_dict["moving_unowned_proportion_sum"]
        self.moving_unowned_proportion_weight = state_dict["moving_unowned_proportion_weight"]

    def loss_policy_player_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n, self.policy_len)
        assert target_probs.shape == (self.n, self.policy_len)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return global_weight * weight * loss

    def loss_policy_opponent_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n, self.policy_len)
        assert target_probs.shape == (self.n, self.policy_len)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.15 * global_weight * weight * loss


    def loss_value_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n, self.value_len)
        assert target_probs.shape == (self.n, self.value_len)
        #target_probs = torch.pow(target_probs, 0.3)
        #target_probs = target_probs / (torch.sum(target_probs, dim=1, keepdim=True) + 1e-10)
        assert weight.shape == (self.n,)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 1.20 * global_weight * weight * loss

    def loss_td_value_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n, self.num_td_values, self.value_len)
        assert target_probs.shape == (self.n, self.num_td_values, self.value_len)
        #target_probs = torch.pow(target_probs, 0.3)
        #target_probs = target_probs / (torch.sum(target_probs, dim=2, keepdim=True) + 1e-10)
        assert weight.shape == (self.n,)
        assert global_weight.shape == (self.n,)
        loss = cross_entropy(pred_logits, target_probs, dim=2) - cross_entropy(torch.log(target_probs + 1.0e-30), target_probs, dim=2)
        return 1.20 * global_weight.unsqueeze(1) * weight.unsqueeze(1) * loss

    def loss_td_score_samplewise(self, pred, target, weight, global_weight):
        assert pred.shape == (self.n, self.num_td_values)
        assert target.shape == (self.n, self.num_td_values)
        loss = torch.sum(huber_loss(pred, target, delta = 12.0), dim=1)
        return 0.0004 * global_weight * weight * loss

    def loss_zero_mse_samplewise(self, pred, weight, global_weight, scale):
        assert pred.shape == (self.n,)
        assert weight.shape == (self.n,)
        assert global_weight.shape == (self.n,)
        loss = torch.square(pred)
        return scale * global_weight * weight * loss

    def loss_scoremean_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == (self.n,)
        assert target.shape == (self.n,)
        loss = huber_loss(pred, target, delta = 12.0)
        return 0.0015 * global_weight * weight * loss

    def loss_lead_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == (self.n,)
        assert target.shape == (self.n,)
        loss = huber_loss(pred, target, delta = 8.0)
        return 0.0060 * global_weight * weight * loss

    def loss_variance_time_samplewise(self, pred, target, weight, global_weight):
        assert pred.shape == (self.n,)
        assert target.shape == (self.n,)
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction.
        loss = huber_loss(pred, target + 1.0e-5, delta = 50.0)
        return 0.0003 * global_weight * weight * loss


    def loss_shortterm_value_error_samplewise(self, pred, td_value_pred_logits, td_value_target_probs, weight, global_weight):
        td_value_pred_probs = torch.softmax(td_value_pred_logits[:,2,:],axis=1)
        predvalue = (td_value_pred_probs[:,0] - td_value_pred_probs[:,1]).detach()
        realvalue = td_value_target_probs[:,2,0] - td_value_target_probs[:,2,1]
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction, 0.01%.
        sqerror = torch.square(predvalue-realvalue) + 1.0e-8
        loss = huber_loss(pred, sqerror, delta = 0.4)
        return 2.0 * global_weight * weight * loss

    def loss_shortterm_score_error_samplewise(self, pred, td_score_pred, td_score_target, weight, global_weight):
        predscore = td_score_pred[:,2].detach()
        realscore = td_score_target[:,2]
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction, one hundredth of a point.
        sqerror = torch.square(predscore-realscore) + 1.0e-4
        loss = huber_loss(pred, sqerror, delta = 100.0)
        return 0.00002 * global_weight * weight * loss

    def accuracy1(self, pred_logits, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * (torch.argmax(pred_logits,dim=1) == torch.argmax(target_probs,dim=1)))

    def target_entropy(self, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * -torch.sum(target_probs * torch.log(target_probs + 1e-30), dim=-1))

    def square_value(self, value_logits, global_weight):
        return torch.sum(global_weight * torch.square(torch.sum(torch.softmax(value_logits,dim=1) * constant_like([1,-1,0],global_weight), dim=1)))

    # Returns 0.5 times the sum of squared model weights, for each reg group of model weights
    @staticmethod
    def get_model_norms(raw_model):
        reg_dict : Dict[str,List] = {}
        raw_model.add_reg_dict(reg_dict)

        device = reg_dict["normal"][0].device
        dtype = torch.float32

        modelnorm_normal = torch.zeros([],device=device,dtype=dtype)
        modelnorm_normal_gamma = torch.zeros([],device=device,dtype=dtype)
        modelnorm_normal_attn = torch.zeros([],device=device,dtype=dtype)
        modelnorm_output = torch.zeros([],device=device,dtype=dtype)
        modelnorm_noreg = torch.zeros([],device=device,dtype=dtype)
        modelnorm_output_noreg = torch.zeros([],device=device,dtype=dtype)

        #for tensor in reg_dict["normal"]:
        #    print(tensor.shape,torch.mean(tensor * tensor))
        
        for tensor in reg_dict["normal"]:
            modelnorm_normal += torch.sum(tensor * tensor)
        for tensor in reg_dict["normal_gamma"]:
            modelnorm_normal_gamma += torch.sum(tensor * tensor)
        for tensor in reg_dict["normal_attn"]:
            modelnorm_normal_attn += torch.sum(tensor * tensor)
        for tensor in reg_dict["output"]:
            modelnorm_output += torch.sum(tensor * tensor)
        for tensor in reg_dict["noreg"]:
            modelnorm_noreg += torch.sum(tensor * tensor)
        for tensor in reg_dict["output_noreg"]:
            modelnorm_output_noreg += torch.sum(tensor * tensor)
        modelnorm_normal *= 0.5
        modelnorm_normal_gamma *= 0.5
        modelnorm_normal_attn *= 0.5
        modelnorm_output *= 0.5
        modelnorm_noreg *= 0.5
        modelnorm_output_noreg *= 0.5
        return (modelnorm_normal, modelnorm_normal_gamma,  modelnorm_normal_attn, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg)

    def get_specific_norms_and_gradient_stats(self,raw_model):
        with torch.no_grad():
            params = {}
            for name, param in raw_model.named_parameters():
                params[name] = param

            stats = {}
            def add_norm_and_grad_stats(name):
                param = params[name]
                if name.endswith(".weight"):
                    fanin = param.shape[1]
                elif name.endswith(".gamma"):
                    fanin = 1
                elif name.endwith(".beta"):
                    fanin = 1
                else:
                    assert False, "unimplemented case to compute stats on parameter"

                # 1.0 means that the average squared magnitude of a parameter in this tensor is around where
                # it would be at initialization, assuming it uses the activation that the model generally
                # uses (e.g. relu or mish)
                param_scale = torch.sqrt(torch.mean(torch.square(param))) / compute_gain(raw_model.activation) * math.sqrt(fanin)
                stats[f"{name}.SCALE_batch"] = param_scale

                # How large is the gradient, on the same scale?
                stats[f"{name}.GRADSC_batch"] = torch.sqrt(torch.mean(torch.square(param.grad))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

                # And how large is the component of the gradient that is orthogonal to the overall magnitude of the parameters?
                orthograd = param.grad - param * (torch.sum(param.grad * param) / (1e-20 + torch.sum(torch.square(param))))
                stats[f"{name}.OGRADSC_batch"] = torch.sqrt(torch.mean(torch.square(orthograd))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

            add_norm_and_grad_stats("blocks.1.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.1.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.1.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.6.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.6.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.6.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.10.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.10.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.10.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.16.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.16.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.16.normactconvq.norm.gamma")

            add_norm_and_grad_stats("policy_head.conv1p.weight")
            add_norm_and_grad_stats("value_head.conv1.weight")
            add_norm_and_grad_stats("intermediate_policy_head.conv1p.weight")
            add_norm_and_grad_stats("intermediate_value_head.conv1.weight")

        return stats

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
        variance_time_loss_scale,
        main_loss_scale,
        intermediate_loss_scale,
        seki_loss_scale=None,
    ):
        results = self.metrics_dict_batchwise_single_heads_output(
            raw_model,
            model_output_postprocessed_byheads[0],
            batch,
            is_training=is_training,
            soft_policy_weight_scale=soft_policy_weight_scale,
            disable_optimistic_policy=disable_optimistic_policy,
            meta_kata_only_soft_policy=meta_kata_only_soft_policy,
            value_loss_scale=value_loss_scale,
            td_value_loss_scales=td_value_loss_scales,
            seki_loss_scale=seki_loss_scale,
            variance_time_loss_scale=variance_time_loss_scale,
            is_intermediate=False,
        )
        if main_loss_scale is not None:
            results["loss_sum"] = main_loss_scale * results["loss_sum"]

        if raw_model.get_has_intermediate_head():
            assert len(model_output_postprocessed_byheads) > 1
            if raw_model.training:
                assert intermediate_loss_scale is not None
            else:
                if intermediate_loss_scale is None:
                    intermediate_loss_scale = 1.0

            if intermediate_loss_scale is not None:
                iresults = self.metrics_dict_batchwise_single_heads_output(
                    raw_model,
                    model_output_postprocessed_byheads[1],
                    batch,
                    is_training=is_training,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    disable_optimistic_policy=disable_optimistic_policy,
                    meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=seki_loss_scale,
                    variance_time_loss_scale=variance_time_loss_scale,
                    is_intermediate=True,
                )
                for key,value in iresults.items():
                    if key != "loss_sum":
                        results["I"+key] = value
                results["loss_sum"] = results["loss_sum"] + intermediate_loss_scale * iresults["loss_sum"]

        return results

    def metrics_dict_batchwise_single_heads_output(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        variance_time_loss_scale,
        is_intermediate,
        seki_loss_scale=None,
    ):
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
        ) = model_output_postprocessed

        input_binary_nchw = batch["binaryInputNCHW"]
        input_global_nc = batch["globalInputNC"]
        target_policy_ncmove = batch["policyTargetsNCMove"]
        target_global_nc = batch["globalTargetsNC"]
        mask = input_binary_nchw[:, 0, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(1,2))

        n = input_binary_nchw.shape[0]
        h = input_binary_nchw.shape[2]
        w = input_binary_nchw.shape[3]

        policymask = torch.cat((mask.view(n,h*w),mask.new_ones((n,1))),dim=1)

        target_policy_player = target_policy_ncmove[:, 0, :]
        target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
        target_policy_opponent = target_policy_ncmove[:, 1, :]
        target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)
        target_policy_player_soft = (target_policy_player + 1e-7) * policymask
        target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
        target_policy_player_soft /= torch.sum(target_policy_player_soft, dim=1, keepdim=True)
        target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
        target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
        target_policy_opponent_soft /= torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_policy_opponent = target_global_nc[:, 28]

        target_value = target_global_nc[:, 0:3]
        target_scoremean = target_global_nc[:, 3]
        target_td_value = torch.stack(
            (target_global_nc[:, 4:7], target_global_nc[:, 8:11], target_global_nc[:, 12:15]), dim=1
        )
        target_td_score = torch.cat(
            (target_global_nc[:, 7:8], target_global_nc[:, 11:12], target_global_nc[:, 15:16]), dim=1
        )
        target_lead = target_global_nc[:, 21]
        target_variance_time = target_global_nc[:, 22]
        global_weight = target_global_nc[:, 25]
        target_weight_ownership = target_global_nc[:, 27]
        target_weight_lead = target_global_nc[:, 29]
        target_weight_value = 1.0 - target_global_nc[:, 35]
        target_weight_td_value = 1.0 - target_global_nc[:, 24]

        if raw_model.config["version"] <= 11 or (raw_model.config["version"] >= 101 and raw_model.config["version"] <= 199):
            assert raw_model.policy_head.num_policy_outputs == 4
            policy_opt_loss_scale = 1.000
            long_policy_opt_loss_scale = 0.0
            short_policy_opt_loss_scale = 0.0
        else:
            assert raw_model.policy_head.num_policy_outputs == 6
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :],
            target_policy_opponent,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        target_weight_policy_player_soft = target_weight_policy_player
        target_weight_policy_opponent_soft = target_weight_policy_opponent
        if meta_kata_only_soft_policy:
            metadata_input_nc = batch["metadataInputNC"]
            assert metadata_input_nc.shape[0] == target_weight_policy_player_soft.shape[0]
            # 151 indicates source 0 = katago
            target_weight_policy_player_soft = target_weight_policy_player_soft * metadata_input_nc[:,151]
            target_weight_policy_opponent_soft = target_weight_policy_opponent_soft * metadata_input_nc[:,151]

        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :],
            target_policy_player_soft,
            target_weight_policy_player_soft,
            global_weight,
        ).sum()
        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :],
            target_policy_opponent_soft,
            target_weight_policy_opponent_soft,
            global_weight,
        ).sum()

        if raw_model.config["version"] <= 11 or (raw_model.config["version"] >= 101 and raw_model.config["version"] <= 199):
            target_weight_longoptimistic_policy = torch.zeros_like(global_weight)
            loss_longoptimistic_policy = torch.zeros_like(loss_policy_player)
        elif disable_optimistic_policy:
            target_weight_longoptimistic_policy = target_weight_policy_player * 0.5
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()
        else:
            # Long-term optimistic policy. Weight policy by:
            # Final game win squared (squaring discourages draws)
            win_squared = torch.square(
                target_global_nc[:, 0] # win (or draw, weighted by draw utility)
                + 0.5 * target_global_nc[:, 2] # noresult
            )
            # Or the score outcome of the game being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            longterm_score_stdevs_excess = (target_global_nc[:, 3] - pred_scoremean.detach()) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
            target_weight_longoptimistic_policy = torch.clamp(
                win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_longoptimistic_policy = (
                target_weight_longoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_weight_ownership # and also actually ended in full ownership and score, not a sidepos
            )
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()

        assert len(loss_longoptimistic_policy.shape) == 0
        assert len(target_weight_longoptimistic_policy.shape) == 1
        assert target_weight_longoptimistic_policy.shape[0] == n
        target_weight_longoptimistic_policy_sum = (global_weight * target_weight_longoptimistic_policy).sum()

        if raw_model.config["version"] <= 11 or (raw_model.config["version"] >= 101 and raw_model.config["version"] <= 199):
            target_weight_shortoptimistic_policy = torch.zeros_like(global_weight)
            loss_shortoptimistic_policy = torch.zeros_like(loss_policy_player)
        elif disable_optimistic_policy:
            target_weight_shortoptimistic_policy = target_weight_policy_player * 0.5
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()
        else:
            # Short-term optimistic policy. Weight policy by:
            # The shortterm value outcome being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            shortterm_value_actual = target_global_nc[:, 12] - target_global_nc[:, 13]
            shortterm_value_pred = torch.nn.functional.softmax(td_value_logits[:, 2, :].detach(), dim=1)
            shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
            shortterm_value_stdevs_excess = (shortterm_value_actual - shortterm_value_pred) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
            # Or the shortterm score outcome being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            shortterm_score_stdevs_excess = (target_global_nc[:, 15] - pred_td_score[:,2].detach()) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
            target_weight_shortoptimistic_policy = torch.clamp(
                torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0) + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_shortoptimistic_policy = (
                target_weight_shortoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_weight_ownership # and also actually ended in full ownership and score, not a sidepos
            )
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()

        assert len(loss_shortoptimistic_policy.shape) == 0
        assert len(target_weight_shortoptimistic_policy.shape) == 1
        assert target_weight_shortoptimistic_policy.shape[0] == n
        target_weight_shortoptimistic_policy_sum = (global_weight * target_weight_shortoptimistic_policy).sum()


        loss_value = self.loss_value_samplewise(
            value_logits, target_value, target_weight_value, global_weight
        ).sum()

        loss_td_value_unsummed = self.loss_td_value_samplewise(
            td_value_logits, target_td_value, target_weight_td_value, global_weight
        )
        assert self.num_td_values == 3
        loss_td_value1 = loss_td_value_unsummed[:,0].sum()
        loss_td_value2 = loss_td_value_unsummed[:,1].sum()
        loss_td_value3 = loss_td_value_unsummed[:,2].sum()

        loss_td_score = self.loss_td_score_samplewise(
            pred_td_score, target_td_score, target_weight_ownership, global_weight
        ).sum()
        loss_scoremean = self.loss_scoremean_samplewise(
            pred_scoremean,
            target_scoremean,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorestdev_zero = self.loss_zero_mse_samplewise(
            pred_scorestdev,
            target_weight_ownership,
            global_weight,
            scale=0.001,
        ).sum()
        loss_lead = self.loss_lead_samplewise(
            pred_lead,
            target_lead,
            target_weight_lead,
            global_weight,
        ).sum()
        loss_variance_time = self.loss_variance_time_samplewise(
            pred_variance_time,
            target_variance_time,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_value_error = self.loss_shortterm_value_error_samplewise(
            pred_shortterm_value_error,
            td_value_logits,
            target_td_value,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_score_error = self.loss_shortterm_score_error_samplewise(
            pred_shortterm_score_error,
            pred_td_score,
            target_td_score,
            target_weight_ownership,
            global_weight,
        ).sum()

        loss_sum = (
            loss_policy_player * policy_opt_loss_scale
            + loss_policy_opponent
            + loss_policy_player_soft * soft_policy_weight_scale
            + loss_policy_opponent_soft * soft_policy_weight_scale
            + loss_longoptimistic_policy * long_policy_opt_loss_scale
            + loss_shortoptimistic_policy * short_policy_opt_loss_scale
            + loss_value * value_loss_scale
            + loss_td_value1 * td_value_loss_scales[0]
            + loss_td_value2 * td_value_loss_scales[1]
            + loss_td_value3 * td_value_loss_scales[2]
            + loss_td_score
            + loss_scoremean
            + loss_scorestdev_zero
            + loss_lead
            + loss_variance_time * variance_time_loss_scale
            + loss_shortterm_value_error
            + loss_shortterm_score_error
        )

        policy_acc1 = self.accuracy1(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        square_value = self.square_value(value_logits, global_weight)

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "p0lopt_sum": loss_longoptimistic_policy,
            "p0loptw_sum": target_weight_longoptimistic_policy_sum,
            "p0sopt_sum": loss_shortoptimistic_policy,
            "p0soptw_sum": target_weight_shortoptimistic_policy_sum,
            "vloss_sum": loss_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdsloss_sum": loss_td_score,
            "smloss_sum": loss_scoremean,
            "sdregloss_sum": loss_scorestdev_zero,
            "leadloss_sum": loss_lead,
            "vtimeloss_sum": loss_variance_time,
            "evstloss_sum": loss_shortterm_value_error,
            "esstloss_sum": loss_shortterm_score_error,
            "loss_sum": loss_sum,
            "pacc1_sum": policy_acc1,
            "vsquare_sum": square_value,
        }

        if is_intermediate:
            return results
        else:
            weight = global_weight.sum()
            nsamples = int(global_weight.shape[0])
            policy_target_entropy = self.target_entropy(
                target_policy_player,
                target_weight_policy_player,
                global_weight,
            )
            soft_policy_target_entropy = self.target_entropy(
                target_policy_player_soft,
                target_weight_policy_player,
                global_weight,
            )

            (modelnorm_normal, modelnorm_normal_gamma, modelnorm_normal_attn, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = self.get_model_norms(raw_model)

            extra_results = {
                "wsum": weight * self.world_size,
                "nsamp": nsamples * self.world_size,
                "ptentr_sum": policy_target_entropy,
                "ptsoftentr_sum": soft_policy_target_entropy,
                "norm_normal_batch": modelnorm_normal,
                "norm_normal_gamma_batch": modelnorm_normal_gamma,
                "norm_normal_attn_batch": modelnorm_normal_attn,
                "norm_output_batch": modelnorm_output,
                "norm_noreg_batch": modelnorm_noreg,
                "norm_output_noreg_batch": modelnorm_output_noreg,
            }
            for key,value in extra_results.items():
                results[key] = value
            return results
