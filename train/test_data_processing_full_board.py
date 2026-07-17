import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(__file__))
import data_processing_pytorch
import modelconfigs


class FullBoardDataValidationTests(unittest.TestCase):
    pos_len = 15
    config = modelconfigs.config_of_name["b24c256h8tflrs"]

    def _write_data(self, path: Path, full_board) -> None:
        full_board_values = (
            [full_board] if isinstance(full_board, (bool, np.bool_)) else list(full_board)
        )
        num_rows = len(full_board_values)
        area = self.pos_len * self.pos_len
        packed_bytes = (area + 7) // 8
        num_bin = modelconfigs.get_num_bin_input_features(self.config)
        num_global = modelconfigs.get_num_global_input_features(self.config)
        binary = np.zeros((num_rows, num_bin, packed_bytes), dtype=np.uint8)
        binary[:, 0, :] = np.uint8(0xFF)
        for row, is_full_board in enumerate(full_board_values):
            if not is_full_board:
                binary[row, 0, 7] = np.uint8(0xFE)

        global_input = np.zeros((num_rows, num_global), dtype=np.float32)
        global_input[:, 0] = np.arange(num_rows, dtype=np.float32)

        with path.open("wb") as out:
            np.savez_compressed(
                out,
                binaryInputNCHWPacked=binary,
                globalInputNC=global_input,
                policyTargetsNCMove=np.zeros(
                    (num_rows, 2, area + 1), dtype=np.int16
                ),
                globalTargetsNC=np.zeros((num_rows, 64), dtype=np.float32),
                scoreDistrN=np.zeros((num_rows, area * 2 + 120), dtype=np.int8),
                valueTargetsNCHW=np.zeros(
                    (num_rows, 5, self.pos_len, self.pos_len), dtype=np.int8
                ),
            )

    def _reader(
        self,
        path: Path,
        *,
        batch_size: int = 1,
        world_size: int = 1,
        rank: int = 0,
        require_full_board: bool = True,
        filter_full_board_on_load: bool = False,
        channels_last: bool = False,
    ):
        return data_processing_pytorch.read_npz_training_data(
            [str(path)],
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            pos_len=self.pos_len,
            device=torch.device("cpu"),
            symmetry_type="none",
            include_meta=False,
            history_matrices_type="none",
            model_config=self.config,
            require_full_board=require_full_board,
            filter_full_board_on_load=filter_full_board_on_load,
            binary_input_channels_last=channels_last,
        )

    def _read_one(self, path: Path, **kwargs):
        return next(self._reader(path, **kwargs))

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

    def test_non_full_board_rows_can_be_filtered_on_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.npz"
            self._write_data(path, full_board=[False, True, False, True])

            with mock.patch.object(data_processing_pytorch.logging, "info") as log_info:
                batch = self._read_one(
                    path,
                    batch_size=2,
                    filter_full_board_on_load=True,
                )

            log_info.assert_not_called()
            self.assertTrue(bool(torch.all(batch["binaryInputNCHW"][:, 0])))
            torch.testing.assert_close(
                batch["globalInputNC"][:, 0],
                torch.tensor([1.0, 3.0]),
            )

    def test_filter_warns_when_file_cannot_fill_one_global_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.npz"
            self._write_data(path, full_board=[True, False])

            with self.assertLogs(level="WARNING") as captured:
                batches = list(
                    self._reader(
                        path,
                        batch_size=1,
                        world_size=2,
                        filter_full_board_on_load=True,
                    )
                )

            self.assertEqual(batches, [])
            warning_text = "\n".join(captured.output)
            self.assertIn("fewer than one global batch", warning_text)
            self.assertIn("batch_size 1 * world_size 2", warning_text)

    def test_filter_warns_and_yields_nothing_when_no_rows_are_full_board(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "no-full-boards.npz"
            self._write_data(path, full_board=[False, False, False])

            with self.assertLogs(level="WARNING") as captured:
                batches = list(
                    self._reader(
                        path,
                        batch_size=1,
                        filter_full_board_on_load=True,
                    )
                )

            self.assertEqual(batches, [])
            warning_text = "\n".join(captured.output)
            self.assertIn("retained 0/3 rows", warning_text)
            self.assertIn("this file will yield no training batches", warning_text)

    def test_filter_requires_full_board_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full.npz"
            self._write_data(path, full_board=True)

            with self.assertRaisesRegex(
                ValueError, "requires require_full_board=True"
            ):
                list(
                    self._reader(
                        path,
                        require_full_board=False,
                        filter_full_board_on_load=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
