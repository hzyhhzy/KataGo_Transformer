from pathlib import Path
import sys
import unittest

import torch


TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from calibrate_cpu_ptq_v106 import Moments, gptq_matrix, symmetric_codes


class CpuPtqV106Tests(unittest.TestCase):
    def test_symmetric_codes_are_per_row_and_ties_to_even(self) -> None:
        values = torch.tensor(
            [[0.0, 0.0], [1.0, -0.5], [2.0, 1.0]],
            dtype=torch.float32,
        )
        codes, scales = symmetric_codes(values, 127.0, 1)
        torch.testing.assert_close(
            scales,
            torch.tensor([[1.0], [1.0 / 127.0], [2.0 / 127.0]]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            torch.equal(
                codes,
                torch.tensor(
                    [[0.0, 0.0], [127.0, -64.0], [127.0, 64.0]]
                ),
            )
        )

    def test_gptq_codes_preserve_geometry_and_declared_range(self) -> None:
        moments = Moments(4, torch.device("cpu"))
        moments.observe(
            torch.tensor(
                [
                    [1.0, -2.0, 0.5, 0.25],
                    [-0.5, 0.75, 1.5, -1.0],
                    [2.0, 1.0, -0.25, 0.5],
                ],
                dtype=torch.float32,
            )
        )
        weight = torch.tensor(
            [[0.25, -0.5, 0.75, 1.0], [-1.0, 0.5, 0.125, -0.25]],
            dtype=torch.float32,
        )
        codes, scales = gptq_matrix(
            weight, moments, 63, 0.001, False, 0.0, 1.0
        )
        self.assertEqual(codes.shape, weight.shape)
        self.assertEqual(scales.shape, (2, 1))
        self.assertLessEqual(int(codes.abs().max()), 63)
        torch.testing.assert_close(
            scales[:, 0], weight.abs().amax(dim=1) / 63.0
        )


if __name__ == "__main__":
    unittest.main()
