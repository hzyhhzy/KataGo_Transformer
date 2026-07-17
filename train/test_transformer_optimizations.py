import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.dirname(__file__))
import model_pytorch


class TransformerOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.old_rope_cast = model_pytorch.LEARNED_ROPE_CAST_TO_INPUT_DTYPE

    def tearDown(self):
        model_pytorch.LEARNED_ROPE_CAST_TO_INPUT_DTYPE = self.old_rope_cast

    def test_learned_rope_low_precision_rotation_is_close(self):
        torch.manual_seed(5678)
        batch_size = 2
        seq_len = 9
        num_heads = 4
        head_dim = 8

        q_base = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16
        )
        k_base = torch.randn_like(q_base)
        freqs_base = torch.randn(num_heads, head_dim // 2, 2)
        positions = torch.arange(seq_len, dtype=torch.float32)
        s_y = positions // 3
        s_x = positions % 3

        def run(cast_to_input_dtype):
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            freqs = freqs_base.detach().clone().requires_grad_(True)
            cos, sin = model_pytorch.compute_learnable_rope_cos_sin(
                s_x, s_y, freqs
            )
            model_pytorch.LEARNED_ROPE_CAST_TO_INPUT_DTYPE = cast_to_input_dtype
            q_out, k_out = model_pytorch.apply_learnable_rotary_emb(
                q, k, cos, sin, cos, sin
            )
            loss = q_out.float().square().mean() + k_out.float().square().mean()
            loss.backward()
            return (
                q_out.detach(),
                k_out.detach(),
                q.grad.detach(),
                k.grad.detach(),
                freqs.grad.detach(),
            )

        old_values = run(False)
        new_values = run(True)
        for new_value, old_value in zip(new_values, old_values):
            self.assertTrue(torch.isfinite(new_value).all())
            torch.testing.assert_close(
                new_value, old_value, rtol=3e-2, atol=2e-2
            )


if __name__ == "__main__":
    unittest.main()
