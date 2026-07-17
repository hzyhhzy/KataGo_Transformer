import argparse
import contextlib
import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(__file__))
import train_muon_ki


class Bf16AmpTests(unittest.TestCase):
    def _parse_amp_args(self, argv):
        parser = argparse.ArgumentParser()
        train_muon_ki.add_amp_arguments(parser.add_argument_group("precision"))
        return vars(parser.parse_args(argv))

    def test_amp_cli_defaults_to_fp32(self):
        args = self._parse_amp_args([])
        self.assertFalse(args["use_fp16"])
        self.assertFalse(args["use_bf16"])

    def test_amp_cli_accepts_each_precision_independently(self):
        fp16_args = self._parse_amp_args(["-use-fp16"])
        self.assertTrue(fp16_args["use_fp16"])
        self.assertFalse(fp16_args["use_bf16"])

        bf16_args = self._parse_amp_args(["-use-bf16"])
        self.assertFalse(bf16_args["use_fp16"])
        self.assertTrue(bf16_args["use_bf16"])

    def test_fp16_and_bf16_cli_are_mutually_exclusive(self):
        with mock.patch.object(argparse.ArgumentParser, "error", side_effect=ValueError):
            with self.assertRaises(ValueError):
                self._parse_amp_args(["-use-fp16", "-use-bf16"])

    def test_qat_rejects_both_amp_modes(self):
        with self.assertRaisesRegex(AssertionError, "FP16/AMP"):
            train_muon_ki.validate_amp_qat_options(True, False, True)
        with self.assertRaisesRegex(AssertionError, "BF16/AMP"):
            train_muon_ki.validate_amp_qat_options(False, True, True)

        train_muon_ki.validate_amp_qat_options(False, False, True)
        train_muon_ki.validate_amp_qat_options(False, True, False)

    def test_bf16_autocast_uses_explicit_bfloat16_dtype(self):
        context = object()
        with mock.patch.object(
            train_muon_ki.torch.amp, "autocast", return_value=context
        ) as autocast:
            result = train_muon_ki.amp_autocast_context(False, True)

        self.assertIs(result, context)
        autocast.assert_called_once_with(device_type="cuda", dtype=train_muon_ki.torch.bfloat16)

    def test_fp16_autocast_keeps_legacy_default_dtype_call(self):
        context = object()
        with mock.patch.object(
            train_muon_ki.torch.amp, "autocast", return_value=context
        ) as autocast:
            result = train_muon_ki.amp_autocast_context(True, False)

        self.assertIs(result, context)
        autocast.assert_called_once_with(device_type="cuda")

    def test_fp32_does_not_enter_autocast(self):
        with mock.patch.object(train_muon_ki.torch.amp, "autocast") as autocast:
            context = train_muon_ki.amp_autocast_context(False, False)

        self.assertIsInstance(context, contextlib.nullcontext)
        autocast.assert_not_called()

    def test_grad_scaler_is_created_only_for_fp16(self):
        scaler = object()
        with mock.patch.object(
            train_muon_ki.torch.amp, "GradScaler", return_value=scaler
        ) as factory:
            self.assertIs(train_muon_ki.create_grad_scaler(True, False), scaler)
            factory.assert_called_once_with("cuda")

        with mock.patch.object(train_muon_ki.torch.amp, "GradScaler") as factory:
            self.assertIsNone(train_muon_ki.create_grad_scaler(False, True))
            self.assertIsNone(train_muon_ki.create_grad_scaler(False, False))
            factory.assert_not_called()

    def test_bf16_and_fp32_use_unscaled_backward_and_step(self):
        for precision in ("bf16", "fp32"):
            with self.subTest(precision=precision):
                loss = mock.Mock()
                optimizer = mock.Mock()
                scaler = train_muon_ki.create_grad_scaler(False, precision == "bf16")

                train_muon_ki.backward_and_unscale(loss, optimizer, scaler)
                train_muon_ki.optimizer_step(optimizer, scaler)

                loss.backward.assert_called_once_with()
                optimizer.step.assert_called_once_with()

    def test_fp16_keeps_scaled_backward_and_step_order(self):
        loss = object()
        scaled_loss = mock.Mock()
        optimizer = object()
        scaler = mock.Mock()
        scaler.scale.return_value = scaled_loss

        train_muon_ki.backward_and_unscale(loss, optimizer, scaler)
        train_muon_ki.optimizer_step(optimizer, scaler)

        scaler.scale.assert_called_once_with(loss)
        scaled_loss.backward.assert_called_once_with()
        scaler.unscale_.assert_called_once_with(optimizer)
        scaler.step.assert_called_once_with(optimizer)
        scaler.update.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
