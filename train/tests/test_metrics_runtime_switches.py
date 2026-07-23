import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import metrics_pytorch


class _FakeRawModel:
    def __init__(self, pos_len=2):
        scorebelief_len = 2 * (
            pos_len * pos_len + metrics_pytorch.EXTRA_SCORE_DISTR_RADIUS
        )
        self.pos_len = pos_len
        self.value_head = SimpleNamespace(
            score_belief_offset_vector=torch.linspace(-1.0, 1.0, scorebelief_len)
        )
        self.policy_head = SimpleNamespace(num_policy_outputs=4)
        self.config = {"version": 102}
        self.training = True
        self._normal_parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))

    def add_reg_dict(self, reg_dict):
        reg_dict.update(
            normal=[self._normal_parameter],
            normal_gamma=[],
            normal_attn=[],
            output=[],
            noreg=[],
            output_noreg=[],
        )

    def get_has_intermediate_head(self):
        return False


def _make_metrics(seki_ema_on_device):
    env = {
        "KATAGO_SEKI_EMA_ON_DEVICE": "1" if seki_ema_on_device else "0"
    }
    with mock.patch.dict(os.environ, env, clear=False):
        raw_model = _FakeRawModel()
        return metrics_pytorch.Metrics(batch_size=4, world_size=1, raw_model=raw_model)


class MetricsRuntimeSwitchTests(unittest.TestCase):
    def test_device_seki_ema_is_enabled_by_default(self):
        env = dict(os.environ)
        env.pop("KATAGO_SEKI_EMA_ON_DEVICE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            metrics = metrics_pytorch.Metrics(
                batch_size=4,
                world_size=1,
                raw_model=_FakeRawModel(),
            )

        self.assertTrue(metrics.seki_ema_on_device)
        self.assertIsInstance(metrics.moving_unowned_proportion_sum, torch.Tensor)

    def test_seki_ema_checkpoint_loads_in_both_runtime_modes(self):
        on_device = _make_metrics(seki_ema_on_device=True)
        on_device.load_state_dict({
            "moving_unowned_proportion_sum": 1.25,
            "moving_unowned_proportion_weight": 3.5,
        })
        self.assertIsInstance(on_device.moving_unowned_proportion_sum, torch.Tensor)
        self.assertEqual(on_device.moving_unowned_proportion_sum.item(), 1.25)
        self.assertEqual(on_device.moving_unowned_proportion_weight.item(), 3.5)

        legacy = _make_metrics(seki_ema_on_device=False)
        legacy.load_state_dict(on_device.state_dict())
        self.assertIsInstance(legacy.moving_unowned_proportion_sum, float)
        self.assertIsInstance(legacy.moving_unowned_proportion_weight, float)
        self.assertEqual(legacy.moving_unowned_proportion_sum, 1.25)
        self.assertEqual(legacy.moving_unowned_proportion_weight, 3.5)

    def test_model_norms_can_be_omitted_from_batch_metrics(self):
        n = 4
        pos_len = 2
        pos_area = pos_len * pos_len
        raw_model = _FakeRawModel(pos_len=pos_len)
        metrics = metrics_pytorch.Metrics(batch_size=n, world_size=1, raw_model=raw_model)
        scorebelief_len = metrics.scorebelief_len

        outputs = (
            torch.zeros(n, 4, pos_area + 1),
            torch.zeros(n, 3),
            torch.zeros(n, 3, 3),
            torch.zeros(n, 3),
            torch.zeros(n, 1, pos_len, pos_len),
            torch.zeros(n, 1, pos_len, pos_len),
            torch.zeros(n, 2, pos_len, pos_len),
            torch.zeros(n, 4, pos_len, pos_len),
            torch.zeros(n),
            torch.ones(n),
            torch.zeros(n),
            torch.ones(n),
            torch.ones(n),
            torch.ones(n),
            torch.zeros(n, scorebelief_len),
        )

        global_targets = torch.zeros(n, 39)
        global_targets[:, 0] = 1.0
        global_targets[:, 4] = 1.0
        global_targets[:, 8] = 1.0
        global_targets[:, 12] = 1.0
        global_targets[:, 25:30] = 1.0
        global_targets[:, 33:35] = 1.0
        batch = {
            "binaryInputNCHW": torch.ones(n, 1, pos_len, pos_len),
            "globalInputNC": torch.zeros(n, 39),
            "policyTargetsNCMove": torch.full((n, 2, pos_area + 1), 1.0 / (pos_area + 1)),
            "globalTargetsNC": global_targets,
            "scoreDistrN": torch.full((n, scorebelief_len), 100.0 / scorebelief_len),
            "valueTargetsNCHW": torch.zeros(n, 5, pos_len, pos_len),
        }

        common_args = dict(
            raw_model=raw_model,
            model_output_postprocessed_byheads=(outputs,),
            extra_outputs=None,
            batch=batch,
            is_training=False,
            soft_policy_weight_scale=8.0,
            disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False,
            value_loss_scale=0.6,
            td_value_loss_scales=(0.6, 0.6, 0.6),
            seki_loss_scale=1.0,
            variance_time_loss_scale=1.0,
            main_loss_scale=None,
            intermediate_loss_scale=None,
        )
        skipped = metrics.metrics_dict_batchwise(
            **common_args,
            include_model_norms=False,
        )
        included = metrics.metrics_dict_batchwise(
            **common_args,
            include_model_norms=True,
        )

        norm_keys = {
            "norm_normal_batch",
            "norm_normal_gamma_batch",
            "norm_normal_attn_batch",
            "norm_output_batch",
            "norm_noreg_batch",
            "norm_output_noreg_batch",
        }
        self.assertTrue(norm_keys.isdisjoint(skipped))
        self.assertTrue(norm_keys.issubset(included))
        self.assertAlmostEqual(included["norm_normal_batch"].item(), 6.5)

    def test_device_seki_ema_tracks_legacy_formula(self):
        legacy = _make_metrics(seki_ema_on_device=False)
        on_device = _make_metrics(seki_ema_on_device=True)

        generator = torch.Generator().manual_seed(1234)
        mask = torch.ones(4, 2, 2)
        mask_sum = mask.sum(dim=(1, 2))
        weight = torch.ones(4)
        global_weight = torch.ones(4)

        legacy_scales = []
        device_scales = []
        for _ in range(256):
            logits = torch.randn(4, 4, 2, 2, generator=generator)
            target = torch.empty(4, 2, 2).uniform_(-1.0, 1.0, generator=generator)
            ownership = torch.empty(4, 2, 2).uniform_(-1.0, 1.0, generator=generator)
            _, legacy_scale = legacy.loss_seki_samplewise(
                logits,
                target,
                ownership,
                weight,
                mask,
                mask_sum,
                global_weight,
                is_training=True,
                skip_moving_update=False,
            )
            _, device_scale = on_device.loss_seki_samplewise(
                logits,
                target,
                ownership,
                weight,
                mask,
                mask_sum,
                global_weight,
                is_training=True,
                skip_moving_update=False,
            )
            legacy_scales.append(float(legacy_scale))
            device_scales.append(float(device_scale))

        self.assertIsInstance(legacy.moving_unowned_proportion_sum, float)
        self.assertIsInstance(on_device.moving_unowned_proportion_sum, torch.Tensor)
        self.assertIsInstance(on_device.moving_unowned_proportion_weight, torch.Tensor)
        self.assertFalse(on_device.moving_unowned_proportion_sum.requires_grad)
        self.assertFalse(on_device.moving_unowned_proportion_weight.requires_grad)
        torch.testing.assert_close(
            torch.tensor(device_scales),
            torch.tensor(legacy_scales),
            rtol=2e-6,
            atol=2e-7,
        )

    def test_device_seki_ema_is_stable_under_torch_compile(self):
        eager = _make_metrics(seki_ema_on_device=True)
        compiled = _make_metrics(seki_ema_on_device=True)
        mask = torch.ones(4, 2, 2)
        mask_sum = mask.sum(dim=(1, 2))
        weight = torch.ones(4)
        global_weight = torch.ones(4)
        target = torch.linspace(-0.8, 0.9, 16).reshape(4, 2, 2)
        ownership = torch.linspace(0.7, -0.6, 16).reshape(4, 2, 2)

        def compiled_call(logits):
            return compiled.loss_seki_samplewise(
                logits,
                target,
                ownership,
                weight,
                mask,
                mask_sum,
                global_weight,
                is_training=True,
                skip_moving_update=False,
            )

        compiled_call = torch.compile(compiled_call, backend="eager", dynamic=False)
        generator = torch.Generator().manual_seed(5678)
        for _ in range(4):
            logits_base = torch.randn(4, 4, 2, 2, generator=generator)
            eager_logits = logits_base.detach().clone().requires_grad_(True)
            compiled_logits = logits_base.detach().clone().requires_grad_(True)
            eager_loss, eager_scale = eager.loss_seki_samplewise(
                eager_logits,
                target,
                ownership,
                weight,
                mask,
                mask_sum,
                global_weight,
                is_training=True,
                skip_moving_update=False,
            )
            compiled_loss, compiled_scale = compiled_call(compiled_logits)
            eager_loss.sum().backward()
            compiled_loss.sum().backward()
            torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(compiled_scale, eager_scale, rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(
                compiled_logits.grad, eager_logits.grad, rtol=2e-6, atol=2e-7
            )
            torch.testing.assert_close(
                compiled.moving_unowned_proportion_sum,
                eager.moving_unowned_proportion_sum,
                rtol=2e-6,
                atol=2e-7,
            )
            torch.testing.assert_close(
                compiled.moving_unowned_proportion_weight,
                eager.moving_unowned_proportion_weight,
                rtol=2e-6,
                atol=2e-7,
            )


if __name__ == "__main__":
    unittest.main()
