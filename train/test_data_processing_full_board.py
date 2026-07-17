import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(__file__))
import data_processing_pytorch
import modelconfigs


class FullBoardDataValidationTests(unittest.TestCase):
    pos_len = 15
    config = modelconfigs.config_of_name["b24c256h8tflrs"]

    def _write_data(self, path: Path, full_board: bool) -> None:
        area = self.pos_len * self.pos_len
        packed_bytes = (area + 7) // 8
        num_bin = modelconfigs.get_num_bin_input_features(self.config)
        num_global = modelconfigs.get_num_global_input_features(self.config)
        binary = np.zeros((1, num_bin, packed_bytes), dtype=np.uint8)
        binary[0, 0, :] = np.uint8(0xFF)
        if not full_board:
            binary[0, 0, 7] = np.uint8(0xFE)

        with path.open("wb") as out:
            np.savez_compressed(
                out,
                binaryInputNCHWPacked=binary,
                globalInputNC=np.zeros((1, num_global), dtype=np.float32),
                policyTargetsNCMove=np.zeros((1, 2, area + 1), dtype=np.int16),
                globalTargetsNC=np.zeros((1, 64), dtype=np.float32),
                scoreDistrN=np.zeros((1, area * 2 + 120), dtype=np.int8),
                valueTargetsNCHW=np.zeros(
                    (1, 5, self.pos_len, self.pos_len), dtype=np.int8
                ),
            )

    def _read_one(self, path: Path, *, channels_last: bool = False):
        return next(
            data_processing_pytorch.read_npz_training_data(
                [str(path)],
                batch_size=1,
                world_size=1,
                rank=0,
                pos_len=self.pos_len,
                device=torch.device("cpu"),
                symmetry_type="none",
                include_meta=False,
                history_matrices_type="none",
                model_config=self.config,
                require_full_board=True,
                binary_input_channels_last=channels_last,
            )
        )

    def test_full_board_validation_ignores_padding_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full.npz"
            self._write_data(path, full_board=True)
            batch = self._read_one(path)
            self.assertTrue(bool(torch.all(batch["binaryInputNCHW"][:, 0])))

    def test_binary_input_can_use_channels_last_memory_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full.npz"
            self._write_data(path, full_board=True)
            batch = self._read_one(path, channels_last=True)
            binary_input = batch["binaryInputNCHW"]
            self.assertTrue(
                binary_input.is_contiguous(memory_format=torch.channels_last)
            )
            self.assertFalse(binary_input.is_contiguous())

    def test_non_full_board_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.npz"
            self._write_data(path, full_board=False)
            with self.assertRaisesRegex(ValueError, "first row index 0"):
                self._read_one(path)


if __name__ == "__main__":
    unittest.main()
