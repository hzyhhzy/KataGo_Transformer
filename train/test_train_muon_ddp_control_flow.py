import os
import sys
import unittest
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(__file__))
import train_muon_ki


class DdpControlFlowTests(unittest.TestCase):
    def test_compile_training_loss_honors_no_compile_and_qat(self):
        resolve = train_muon_ki.resolve_compile_training_loss
        self.assertTrue(resolve(True, no_compile=False, qat_int8=False))
        self.assertFalse(resolve(True, no_compile=True, qat_int8=False))
        self.assertFalse(resolve(True, no_compile=False, qat_int8=True))
        self.assertFalse(resolve(False, no_compile=False, qat_int8=False))

    def test_snapshot_metrics_replace_zero_weight_lookahead_entries(self):
        sums = {"norm_normal_batch": 0.0}
        weights = {"norm_normal_batch": 0.0}
        metrics = {"norm_normal_batch": 123.5}

        train_muon_ki.set_snapshot_metrics(
            sums,
            weights,
            metrics,
            ("norm_normal_batch",),
        )

        self.assertEqual(sums["norm_normal_batch"], 123.5)
        self.assertEqual(weights["norm_normal_batch"], 1.0)

    def test_single_process_action_does_not_call_distributed(self):
        with mock.patch.object(train_muon_ki.torch.distributed, "broadcast") as broadcast:
            action = train_muon_ki.broadcast_rank0_action(
                train_muon_ki._RANK0_ACTION_RETRY,
                rank=0,
                world_size=1,
                device=torch.device("cpu"),
            )
        self.assertEqual(action, train_muon_ki._RANK0_ACTION_RETRY)
        broadcast.assert_not_called()

    def test_rank0_action_is_broadcast_to_worker(self):
        def receive_stop(tensor, src):
            self.assertEqual(src, 0)
            tensor.fill_(train_muon_ki._RANK0_ACTION_STOP)

        with mock.patch.object(
            train_muon_ki.torch.distributed,
            "broadcast",
            side_effect=receive_stop,
        ) as broadcast:
            action = train_muon_ki.broadcast_rank0_action(
                None,
                rank=1,
                world_size=2,
                device=torch.device("cpu"),
            )
        self.assertEqual(action, train_muon_ki._RANK0_ACTION_STOP)
        broadcast.assert_called_once()

    def test_rank0_sends_its_control_action(self):
        observed = []

        def observe(tensor, src):
            observed.append((int(tensor.item()), src))

        with mock.patch.object(
            train_muon_ki.torch.distributed,
            "broadcast",
            side_effect=observe,
        ):
            action = train_muon_ki.broadcast_rank0_action(
                train_muon_ki._RANK0_ACTION_RETRY,
                rank=0,
                world_size=2,
                device=torch.device("cpu"),
            )
        self.assertEqual(action, train_muon_ki._RANK0_ACTION_RETRY)
        self.assertEqual(observed, [(train_muon_ki._RANK0_ACTION_RETRY, 0)])

    def test_validation_uses_compiled_local_module_inside_ddp(self):
        class FakeDdp:
            def __init__(self, module):
                self.module = module

        raw_model = object()
        compiled_local_model = object()
        with mock.patch.object(train_muon_ki, "DistributedDataParallel", FakeDdp):
            result = train_muon_ki.get_local_validation_model(
                FakeDdp(compiled_local_model),
                raw_model,
                world_size=2,
            )
        self.assertIs(result, compiled_local_model)

    def test_validation_unwraps_compiled_ddp_wrapper_without_calling_it(self):
        class FakeDdp:
            def __init__(self, module):
                self.module = module

        class FakeCompiledWrapper:
            def __init__(self, original):
                self._orig_mod = original

        raw_model = object()
        outer_compiled_ddp = FakeCompiledWrapper(FakeDdp(raw_model))
        with mock.patch.object(train_muon_ki, "DistributedDataParallel", FakeDdp):
            result = train_muon_ki.get_local_validation_model(
                outer_compiled_ddp,
                raw_model,
                world_size=2,
            )
        self.assertIs(result, raw_model)

    def test_unknown_distributed_wrapper_falls_back_to_raw_model(self):
        raw_model = object()
        result = train_muon_ki.get_local_validation_model(
            object(),
            raw_model,
            world_size=2,
        )
        self.assertIs(result, raw_model)


if __name__ == "__main__":
    unittest.main()
