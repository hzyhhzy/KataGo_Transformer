import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from filter_full_board_npz import (
    BINARY_INPUT_KEY,
    DatasetFormatError,
    MANIFEST_FILENAME,
    filter_full_board_dataset,
    full_board_row_mask,
    verify_filtered_dataset,
)


POS_LEN = 15
PACKED_BYTES = (POS_LEN * POS_LEN + 7) // 8


def _packed_inputs(num_rows):
    packed = np.zeros((num_rows, 3, PACKED_BYTES), dtype=np.uint8)
    if num_rows >= 1:
        packed[0, 0, :28] = 0xFF
        packed[0, 0, 28] = 0x80
    if num_rows >= 2:
        # Padding bits do not belong to the board and may be either value.
        packed[1, 0, :] = 0xFF
    if num_rows >= 3:
        packed[2, 0, :28] = 0xFF
        packed[2, 0, 28] = 0x00
    if num_rows >= 4:
        packed[3, 0, :28] = 0xFF
        packed[3, 0, 0] = 0x7F
        packed[3, 0, 28] = 0x80
    if num_rows >= 5:
        packed[4, 0, :27] = 0xFF
        packed[4, 0, 27] = 0xFE
        packed[4, 0, 28] = 0x80
    # Make the non-mask channels distinct so row alignment is testable.
    packed[:, 1, 0] = np.arange(num_rows, dtype=np.uint8)
    packed[:, 2, -1] = np.arange(num_rows, dtype=np.uint8) + 10
    return packed


def _arrays(num_rows):
    row = np.arange(num_rows)
    return {
        BINARY_INPUT_KEY: _packed_inputs(num_rows),
        "globalInputNC": np.stack((row, row + 100), axis=1).astype(np.float32),
        "policyTargetsNCMove": np.stack((row, row + 10, row + 20), axis=1).astype(np.float16),
        "globalTargetsNC": np.stack((row + 30, row + 40), axis=1).astype(np.float32),
        "scoreDistrN": np.stack((row + 50, row + 60), axis=1).astype(np.int16),
        "valueTargetsNCHW": row.reshape(num_rows, 1, 1, 1).astype(np.float16),
        "metadataInputNC": np.stack((row + 70, row + 80), axis=1).astype(np.float32),
        # An unknown future row field must be filtered too.
        "futureRowField": np.stack((row + 90, row + 100), axis=1).astype(np.int32),
    }


def _write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        np.savez_compressed(out, **arrays)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        while True:
            chunk = inp.read(65536)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


class FilterFullBoardNpzTests(unittest.TestCase):
    def _make_source(self, parent):
        source = parent / "source"
        train = source / "train"
        val = source / "val"
        train.mkdir(parents=True)
        val.mkdir()
        train_json_bytes = b'{"range":[123,456789],"preserve":"exactly"}\n'
        (source / "train.json").write_bytes(train_json_bytes)
        (source / "val.json").write_text('{"range":[10,20]}\n', encoding="utf-8")

        train_arrays = _arrays(5)
        val_arrays = _arrays(4)
        _write_npz(train / "data0.npz", train_arrays)
        _write_npz(val / "data1.npz", val_arrays)
        (train / "data0.json").write_text(
            json.dumps({"num_rows": 5, "num_batches": 5, "custom": "kept"}),
            encoding="utf-8",
        )
        return source, train_json_bytes, train_arrays, val_arrays

    def test_fast_mask_ignores_padding_and_matches_unpackbits(self):
        packed = _packed_inputs(5)
        fast = full_board_row_mask(packed, POS_LEN)
        reference = np.unpackbits(
            packed[:, 0, :], axis=1, count=POS_LEN * POS_LEN, bitorder="big"
        ).all(axis=1)
        np.testing.assert_array_equal(fast, np.array([True, True, False, False, False]))
        np.testing.assert_array_equal(fast, reference)

    def test_filter_is_atomic_preserves_source_and_filters_every_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, train_json_bytes, train_arrays, val_arrays = self._make_source(parent)
            destination = parent / "full15"

            source_hashes = {
                path.relative_to(source).as_posix(): _sha256(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            # Exercise compatibility with a read-only source tree. Restore modes
            # in finally so TemporaryDirectory can clean up on every platform.
            source_paths = sorted(source.rglob("*"), key=lambda p: len(p.parts), reverse=True)
            for path in source_paths:
                path.chmod(stat.S_IREAD | (stat.S_IEXEC if path.is_dir() else 0))
            source.chmod(stat.S_IREAD | stat.S_IEXEC)
            try:
                manifest = filter_full_board_dataset(
                    source, destination, pos_len=POS_LEN, workers=1
                )
            finally:
                source.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
                for path in reversed(source_paths):
                    if path.is_dir():
                        path.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
                    else:
                        path.chmod(stat.S_IREAD | stat.S_IWRITE)

            self.assertTrue(destination.is_dir())
            self.assertEqual((destination / "train.json").read_bytes(), train_json_bytes)
            self.assertEqual(manifest["total_input_rows"], 9)
            self.assertEqual(manifest["total_output_rows"], 4)
            self.assertEqual(manifest["total_removed_rows"], 5)
            self.assertEqual(manifest["splits"]["train"]["output_rows"], 2)
            self.assertEqual(manifest["splits"]["val"]["output_rows"], 2)

            with np.load(destination / "train" / "data0.npz", allow_pickle=False) as out:
                self.assertEqual(set(out.files), set(train_arrays))
                for key, source_array in train_arrays.items():
                    self.assertEqual(out[key].dtype, source_array.dtype)
                    self.assertEqual(out[key].shape[1:], source_array.shape[1:])
                    np.testing.assert_array_equal(out[key], source_array[[0, 1]])

            with np.load(destination / "val" / "data1.npz", allow_pickle=False) as out:
                for key, source_array in val_arrays.items():
                    np.testing.assert_array_equal(out[key], source_array[[0, 1]])

            sidecar = json.loads(
                (destination / "train" / "data0.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["num_rows"], 2)
            self.assertEqual(sidecar["num_batches"], 2)
            self.assertEqual(sidecar["custom"], "kept")

            manifest_on_disk = json.loads(
                (destination / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_on_disk["total_output_rows"], 4)
            verification = verify_filtered_dataset(destination, pos_len=POS_LEN)
            self.assertEqual(verification["total_output_rows"], 4)

            after_hashes = {
                path.relative_to(source).as_posix(): _sha256(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after_hashes, source_hashes)
            self.assertEqual(list(parent.glob(".full15.staging-*")), [])

    def test_misaligned_key_fails_without_publishing_or_leaving_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, _, train_arrays, _ = self._make_source(parent)
            bad_arrays = dict(train_arrays)
            bad_arrays["badField"] = np.zeros((4, 2), dtype=np.float32)
            _write_npz(source / "train" / "data0.npz", bad_arrays)
            destination = parent / "full15"

            with self.assertRaises(DatasetFormatError):
                filter_full_board_dataset(source, destination, pos_len=POS_LEN, workers=1)
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.glob(".full15.staging-*")), [])

    def test_existing_destination_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, _, _, _ = self._make_source(parent)
            destination = parent / "full15"
            destination.mkdir()
            marker = destination / "marker"
            marker.write_text("untouched", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                filter_full_board_dataset(source, destination, pos_len=POS_LEN, workers=1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")

    def test_verification_rejects_extra_packed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, _, _, _ = self._make_source(parent)
            destination = parent / "full15"
            filter_full_board_dataset(source, destination, pos_len=POS_LEN, workers=1)

            path = destination / "train" / "data0.npz"
            with np.load(path, allow_pickle=False) as npz:
                arrays = {key: npz[key] for key in npz.files}
            arrays[BINARY_INPUT_KEY] = np.pad(
                arrays[BINARY_INPUT_KEY], ((0, 0), (0, 0), (0, 1))
            )
            _write_npz(path, arrays)

            with self.assertRaisesRegex(DatasetFormatError, "packed area"):
                verify_filtered_dataset(destination, pos_len=POS_LEN)

    def test_zero_retained_training_rows_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, _, train_arrays, _ = self._make_source(parent)
            train_arrays[BINARY_INPUT_KEY][:, 0, :] = 0
            _write_npz(source / "train" / "data0.npz", train_arrays)
            destination = parent / "full15"

            with self.assertRaisesRegex(DatasetFormatError, "zero training rows"):
                filter_full_board_dataset(source, destination, pos_len=POS_LEN, workers=1)
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.glob(".full15.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
