import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from numpy_npz_headers import read_numpy_array_header
from shuffle import compute_num_rows


def _npy_3_0_bytes(shape, dtype):
    header = repr(
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        }
    ).encode("utf-8")
    padding = b" " * ((64 - (12 + len(header) + 1) % 64) % 64) + b"\n"
    return (
        np.lib.format.magic(3, 0)
        + struct.pack("<I", len(header) + len(padding))
        + header
        + padding
    )


class NumpyNpzHeadersTest(unittest.TestCase):
    def _temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent)

    def test_reads_standard_npz_without_loading_array_data(self):
        with self._temporary_directory() as directory:
            path = Path(directory) / "data.npz"
            np.savez(path, binaryInputNCHWPacked=np.zeros((7, 2), dtype=np.uint8))
            self.assertEqual(compute_num_rows(str(path)), (str(path), 7))

    def test_reads_npy_3_0_header(self):
        stream = io.BytesIO(_npy_3_0_bytes((11, 3), np.float32))
        version = np.lib.format.read_magic(stream)
        shape, is_fortran, dtype = read_numpy_array_header(stream, version)
        self.assertEqual(shape, (11, 3))
        self.assertFalse(is_fortran)
        self.assertEqual(dtype, np.dtype(np.float32))

    def test_reads_npy_3_0_header_inside_npz(self):
        with self._temporary_directory() as directory:
            path = Path(directory) / "data.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "binaryInputNCHWPacked.npy",
                    _npy_3_0_bytes((13, 2), np.uint8),
                )
            self.assertEqual(compute_num_rows(str(path)), (str(path), 13))


if __name__ == "__main__":
    unittest.main()
