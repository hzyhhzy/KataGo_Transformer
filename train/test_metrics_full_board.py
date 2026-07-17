import os
import sys
import unittest
from types import SimpleNamespace

import torch


sys.path.insert(0, os.path.dirname(__file__))
import metrics_pytorch


class _FakeRawModel:
    def __init__(self, pos_len=3, has_intermediate_head=True):
        scorebelief_len = 2 * (
            pos_len * pos_len + metrics_pytorch.EXTRA_SCORE_DISTR_RADIUS
        )
        self.pos_len = pos_len
        self.value_head = SimpleNamespace(
            score_belief_offset_vector=torch.linspace(-1.0, 1.0, scorebelief_len)
        )
        self.policy_head = SimpleNamespace(num_policy_outputs=4)
        self.config = {"version": 102}
        self.training = False
        self._has_intermediate_head = has_intermediate_head

    def get_has_intermediate_head(self):
        return self._has_intermediate_head


class _RecordingMetrics(metrics_pytorch.Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_masks = []

    def loss_ownership_samplewise(self, *args, **kwargs):
        self.spatial_masks.append(("ownership", args[3] is None))
        return super().loss_ownership_samplewise(*args, **kwargs)

    def loss_scoring_samplewise(self, *args, **kwargs):
        self.spatial_masks.append(("scoring", args[3] is None))
        return super().loss_scoring_samplewise(*args, **kwargs)

    def loss_futurepos_samplewise(self, *args, **kwargs):
        self.spatial_masks.append(("futurepos", args[3] is None))
        return super().loss_futurepos_samplewise(*args, **kwargs)

    def loss_seki_samplewise(self, *args, **kwargs):
        self.spatial_masks.append(("seki", args[4] is None))
        return super().loss_seki_samplewise(*args, **kwargs)


def _clone_output(output):
    return tuple(value.detach().clone().requires_grad_(True) for value in output)


def _make_output(n, pos_len, scorebelief_len, generator):
    pos_area = pos_len * pos_len
    shapes = (
        (n, 4, pos_area + 1),
        (n, 3),
        (n, 3, 3),
        (n, 3),
        (n, 1, pos_len, pos_len),
        (n, 1, pos_len, pos_len),
        (n, 2, pos_len, pos_len),
        (n, 4, pos_len, pos_len),
        (n,),
        (n,),
        (n,),
        (n,),
        (n,),
        (n,),
        (n, scorebelief_len),
    )
    return tuple(torch.randn(shape, generator=generator) for shape in shapes)


def _make_batch(n, pos_len, scorebelief_len, generator):
    pos_area = pos_len * pos_len
    policy_targets = torch.rand(n, 2, pos_area + 1, generator=generator) + 0.1
    global_targets = torch.zeros(n, 39)
    global_targets[:, 0:3] = torch.tensor([0.55, 0.35, 0.10])
    global_targets[:, 4:7] = torch.tensor([0.50, 0.30, 0.20])
    global_targets[:, 8:11] = torch.tensor([0.45, 0.40, 0.15])
    global_targets[:, 12:15] = torch.tensor([0.40, 0.45, 0.15])
    global_targets[:, 3] = torch.linspace(-1.0, 1.0, n)
    global_targets[:, 7] = 0.25
    global_targets[:, 11] = -0.50
    global_targets[:, 15] = 0.75
    global_targets[:, 21] = 0.5
    global_targets[:, 22] = 2.0
    global_targets[:, 25] = 1.0
    global_targets[:, 26] = 0.8
    global_targets[:, 27] = 0.9
    global_targets[:, 28] = 0.7
    global_targets[:, 29] = 0.6
    global_targets[:, 33] = 0.5
    global_targets[:, 34] = 0.4

    return {
        "binaryInputNCHW": torch.ones(n, 3, pos_len, pos_len),
        "globalInputNC": torch.zeros(n, 39),
        "policyTargetsNCMove": policy_targets,
        "globalTargetsNC": global_targets,
        "scoreDistrN": torch.full(
            (n, scorebelief_len), 100.0 / scorebelief_len
        ),
        "valueTargetsNCHW": torch.empty(
            n, 5, pos_len, pos_len
        ).uniform_(-0.8, 0.8, generator=generator),
    }


class MetricsFullBoardTests(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.pos_len = 3
        self.raw_model = _FakeRawModel(
            pos_len=self.pos_len, has_intermediate_head=True
        )
        self.metrics = metrics_pytorch.Metrics(
            batch_size=self.n, world_size=1, raw_model=self.raw_model
        )
        self.mask = torch.ones(self.n, self.pos_len, self.pos_len)
        self.mask_sum = self.mask.sum(dim=(1, 2))
        self.weight = torch.tensor([0.4, 0.7, 1.0])
        self.global_weight = torch.tensor([1.0, 0.8, 0.6])
        self.generator = torch.Generator().manual_seed(12345)

    def _assert_prediction_loss_and_gradient_match(
        self, prediction, masked_call, full_board_call
    ):
        masked_prediction = prediction.detach().clone().requires_grad_(True)
        full_board_prediction = prediction.detach().clone().requires_grad_(True)
        masked_loss = masked_call(masked_prediction)
        full_board_loss = full_board_call(full_board_prediction)
        torch.testing.assert_close(full_board_loss, masked_loss, rtol=1e-6, atol=1e-7)
        masked_loss.sum().backward()
        full_board_loss.sum().backward()
        torch.testing.assert_close(
            full_board_prediction.grad,
            masked_prediction.grad,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_spatial_losses_accept_none_mask_and_match_all_ones(self):
        ownership_target = torch.empty(
            self.n, self.pos_len, self.pos_len
        ).uniform_(-1.0, 1.0, generator=self.generator)
        ownership_pred = torch.randn(
            self.n, 1, self.pos_len, self.pos_len, generator=self.generator
        )
        self._assert_prediction_loss_and_gradient_match(
            ownership_pred,
            lambda pred: self.metrics.loss_ownership_samplewise(
                pred,
                ownership_target,
                self.weight,
                self.mask,
                self.mask_sum,
                self.global_weight,
            ),
            lambda pred: self.metrics.loss_ownership_samplewise(
                pred,
                ownership_target,
                self.weight,
                None,
                None,
                self.global_weight,
            ),
        )

        scoring_target = torch.randn(
            self.n, self.pos_len, self.pos_len, generator=self.generator
        )
        scoring_pred = torch.randn(
            self.n, 1, self.pos_len, self.pos_len, generator=self.generator
        )
        self._assert_prediction_loss_and_gradient_match(
            scoring_pred,
            lambda pred: self.metrics.loss_scoring_samplewise(
                pred,
                scoring_target,
                self.weight,
                self.mask,
                self.mask_sum,
                self.global_weight,
            ),
            lambda pred: self.metrics.loss_scoring_samplewise(
                pred,
                scoring_target,
                self.weight,
                None,
                None,
                self.global_weight,
            ),
        )

        futurepos_target = torch.empty(
            self.n, 2, self.pos_len, self.pos_len
        ).uniform_(-1.0, 1.0, generator=self.generator)
        futurepos_pred = torch.randn(
            self.n, 2, self.pos_len, self.pos_len, generator=self.generator
        )
        self._assert_prediction_loss_and_gradient_match(
            futurepos_pred,
            lambda pred: self.metrics.loss_futurepos_samplewise(
                pred,
                futurepos_target,
                self.weight,
                self.mask,
                self.mask_sum,
                self.global_weight,
            ),
            lambda pred: self.metrics.loss_futurepos_samplewise(
                pred,
                futurepos_target,
                self.weight,
                None,
                None,
                self.global_weight,
            ),
        )

        seki_target = torch.empty(
            self.n, self.pos_len, self.pos_len
        ).uniform_(-1.0, 1.0, generator=self.generator)
        seki_ownership = torch.empty(
            self.n, self.pos_len, self.pos_len
        ).uniform_(-1.0, 1.0, generator=self.generator)
        seki_pred = torch.randn(
            self.n, 4, self.pos_len, self.pos_len, generator=self.generator
        )
        masked_pred = seki_pred.detach().clone().requires_grad_(True)
        full_board_pred = seki_pred.detach().clone().requires_grad_(True)
        masked_loss, masked_scale = self.metrics.loss_seki_samplewise(
            masked_pred,
            seki_target,
            seki_ownership,
            self.weight,
            self.mask,
            self.mask_sum,
            self.global_weight,
            is_training=False,
            skip_moving_update=False,
        )
        full_board_loss, full_board_scale = self.metrics.loss_seki_samplewise(
            full_board_pred,
            seki_target,
            seki_ownership,
            self.weight,
            None,
            None,
            self.global_weight,
            is_training=False,
            skip_moving_update=False,
        )
        torch.testing.assert_close(full_board_loss, masked_loss, rtol=1e-6, atol=1e-7)
        self.assertEqual(full_board_scale, masked_scale)
        masked_loss.sum().backward()
        full_board_loss.sum().backward()
        torch.testing.assert_close(
            full_board_pred.grad, masked_pred.grad, rtol=1e-6, atol=1e-7
        )

    def test_seki_training_ema_matches_full_board_fast_path(self):
        target = torch.empty(
            self.n, self.pos_len, self.pos_len
        ).uniform_(-1.0, 1.0, generator=self.generator)
        ownership = torch.empty(
            self.n, self.pos_len, self.pos_len
        ).uniform_(-1.0, 1.0, generator=self.generator)
        prediction = torch.randn(
            self.n, 4, self.pos_len, self.pos_len, generator=self.generator
        )

        for skip_moving_update in (False, True):
            with self.subTest(skip_moving_update=skip_moving_update):
                masked_metrics = metrics_pytorch.Metrics(
                    batch_size=self.n, world_size=1, raw_model=self.raw_model
                )
                full_board_metrics = metrics_pytorch.Metrics(
                    batch_size=self.n, world_size=1, raw_model=self.raw_model
                )
                for metrics in (masked_metrics, full_board_metrics):
                    metrics.moving_unowned_proportion_sum = 0.2
                    metrics.moving_unowned_proportion_weight = 1.0
                masked_loss, masked_scale = masked_metrics.loss_seki_samplewise(
                    prediction,
                    target,
                    ownership,
                    self.weight,
                    self.mask,
                    self.mask_sum,
                    self.global_weight,
                    is_training=True,
                    skip_moving_update=skip_moving_update,
                )
                full_board_loss, full_board_scale = (
                    full_board_metrics.loss_seki_samplewise(
                        prediction,
                        target,
                        ownership,
                        self.weight,
                        None,
                        None,
                        self.global_weight,
                        is_training=True,
                        skip_moving_update=skip_moving_update,
                    )
                )
                torch.testing.assert_close(full_board_loss, masked_loss)
                torch.testing.assert_close(
                    torch.as_tensor(full_board_scale), torch.as_tensor(masked_scale)
                )
                torch.testing.assert_close(
                    torch.as_tensor(full_board_metrics.moving_unowned_proportion_sum),
                    torch.as_tensor(masked_metrics.moving_unowned_proportion_sum),
                )
                torch.testing.assert_close(
                    torch.as_tensor(full_board_metrics.moving_unowned_proportion_weight),
                    torch.as_tensor(masked_metrics.moving_unowned_proportion_weight),
                )

    def test_batch_metrics_propagates_full_board_to_main_and_intermediate(self):
        generator = torch.Generator().manual_seed(67890)
        scorebelief_len = 2 * (
            self.pos_len * self.pos_len + metrics_pytorch.EXTRA_SCORE_DISTR_RADIUS
        )
        output_bases = (
            _make_output(self.n, self.pos_len, scorebelief_len, generator),
            _make_output(self.n, self.pos_len, scorebelief_len, generator),
        )
        masked_outputs = tuple(_clone_output(output) for output in output_bases)
        full_board_outputs = tuple(_clone_output(output) for output in output_bases)
        batch = _make_batch(
            self.n, self.pos_len, scorebelief_len, generator
        )

        common_args = dict(
            raw_model=self.raw_model,
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
            main_loss_scale=1.1,
            intermediate_loss_scale=0.3,
            include_model_norms=False,
        )
        masked_metrics = _RecordingMetrics(
            batch_size=self.n, world_size=1, raw_model=self.raw_model
        )
        full_board_metrics = _RecordingMetrics(
            batch_size=self.n, world_size=1, raw_model=self.raw_model
        )

        # Omitting assume_full_board exercises and protects the default False path.
        masked_results = masked_metrics.metrics_dict_batchwise(
            model_output_postprocessed_byheads=masked_outputs,
            **common_args,
        )
        full_board_results = full_board_metrics.metrics_dict_batchwise(
            model_output_postprocessed_byheads=full_board_outputs,
            assume_full_board=True,
            **common_args,
        )

        expected_names = ["ownership", "scoring", "futurepos", "seki"] * 2
        self.assertEqual(
            [name for name, _ in masked_metrics.spatial_masks], expected_names
        )
        self.assertEqual(
            [name for name, _ in full_board_metrics.spatial_masks], expected_names
        )
        self.assertEqual(
            [is_none for _, is_none in masked_metrics.spatial_masks], [False] * 8
        )
        self.assertEqual(
            [is_none for _, is_none in full_board_metrics.spatial_masks], [True] * 8
        )

        self.assertEqual(masked_results.keys(), full_board_results.keys())
        for name in masked_results:
            masked_value = masked_results[name]
            full_board_value = full_board_results[name]
            if isinstance(masked_value, torch.Tensor):
                torch.testing.assert_close(
                    full_board_value, masked_value, rtol=2e-6, atol=2e-6
                )
            else:
                self.assertEqual(full_board_value, masked_value)

        masked_results["loss_sum"].backward()
        full_board_results["loss_sum"].backward()
        for masked_output, full_board_output in zip(
            masked_outputs, full_board_outputs
        ):
            for masked_value, full_board_value in zip(
                masked_output, full_board_output
            ):
                if masked_value.grad is None or full_board_value.grad is None:
                    self.assertIsNone(masked_value.grad)
                    self.assertIsNone(full_board_value.grad)
                else:
                    torch.testing.assert_close(
                        full_board_value.grad,
                        masked_value.grad,
                        rtol=2e-6,
                        atol=2e-6,
                    )


if __name__ == "__main__":
    unittest.main()
