import copy
import os
import sys
import unittest
from unittest import mock

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import model_pytorch
import modelconfigs


class TransformerOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.old_rope_cast = model_pytorch.LEARNED_ROPE_CAST_TO_INPUT_DTYPE

    def tearDown(self):
        model_pytorch.LEARNED_ROPE_CAST_TO_INPUT_DTYPE = self.old_rope_cast

    @staticmethod
    def _small_odd_half_config():
        config = copy.deepcopy(modelconfigs.config_of_name["b11c96h4tflrs-bng-silu"])
        config.update(
            {
                "trunk_num_channels": 16,
                "mid_num_channels": 16,
                "gpool_num_channels": 8,
                "transformer_ffn_channels": 24,
                "transformer_heads": 2,
                "transformer_kv_heads": 2,
                "num_attention_pool_heads": 2,
                "block_kind": [["rconv1", "transformerropesg"]],
                "p1_num_channels": 8,
                "g1_num_channels": 8,
                "v1_num_channels": 8,
                "sbv2_num_channels": 8,
                "v2_size": 16,
                "trunk_odd_half": True,
                "transformer_trunk_nhwc": True,
                "transformer_block_reshape_nchw_to_nlc": False,
            }
        )
        return config

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

    def test_full_mask_and_no_mask_transformer_outputs_and_gradients_match(self):
        torch.manual_seed(1234)
        config = {
            "transformer_ffn_channels": 24,
            "transformer_heads": 2,
            "transformer_kv_heads": 2,
            "learnable_rope": True,
        }
        block_with_mask = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True
        )
        block_without_mask = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True
        )
        block_without_mask.load_state_dict(block_with_mask.state_dict())

        x_with_mask = torch.randn(2, 16, 3, 3, requires_grad=True)
        x_without_mask = x_with_mask.detach().clone().requires_grad_(True)
        mask = torch.ones(2, 1, 3, 3)

        output_with_mask = block_with_mask(
            x_with_mask,
            mask=mask,
            mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True),
            mask_sum=mask.sum(),
        )
        output_without_mask = block_without_mask(
            x_without_mask,
            mask=None,
            mask_sum_hw=None,
            mask_sum=None,
        )
        torch.testing.assert_close(output_without_mask, output_with_mask)

        output_with_mask.square().mean().backward()
        output_without_mask.square().mean().backward()
        torch.testing.assert_close(x_without_mask.grad, x_with_mask.grad)
        for parameter_with_mask, parameter_without_mask in zip(
            block_with_mask.parameters(), block_without_mask.parameters()
        ):
            torch.testing.assert_close(
                parameter_without_mask.grad, parameter_with_mask.grad
            )

    def test_flex_branch_preserves_existing_kv_only_mask_semantics(self):
        torch.manual_seed(1357)
        config = {
            "transformer_ffn_channels": 24,
            "transformer_heads": 2,
            "transformer_kv_heads": 2,
            "learnable_rope": True,
        }
        sdpa_block = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True
        )
        flex_block = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True
        )
        flex_block.load_state_dict(sdpa_block.state_dict())

        mask = torch.ones(2, 1, 3, 3)
        mask[1, :, 2, :] = 0
        mask[1, :, :, 2] = 0
        additive_mask = torch.zeros(2, 1, 1, 9).masked_fill(
            mask.view(2, 1, 1, 9) == 0, float("-inf")
        )
        sentinel_block_mask = object()

        def fake_flex_attention(q, k, v, block_mask):
            self.assertIs(block_mask, sentinel_block_mask)
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask, dropout_p=0.0
            )

        sdpa_input = torch.randn(2, 16, 3, 3, requires_grad=True)
        flex_input = sdpa_input.detach().clone().requires_grad_(True)
        sdpa_output = sdpa_block(
            sdpa_input,
            mask=mask,
            mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True),
            mask_sum=mask.sum(),
        )
        with mock.patch.object(
            model_pytorch, "flex_attention", side_effect=fake_flex_attention
        ):
            flex_output = flex_block(
                flex_input,
                mask=mask,
                mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True),
                mask_sum=mask.sum(),
                attention_block_mask=sentinel_block_mask,
            )

        # Include off-board query outputs in the loss. This catches an
        # accidental change from the existing KV-only mask to Q-and-KV masking.
        torch.testing.assert_close(flex_output, sdpa_output)
        flex_output.square().mean().backward()
        sdpa_output.square().mean().backward()
        torch.testing.assert_close(flex_input.grad, sdpa_input.grad)
        for flex_parameter, sdpa_parameter in zip(
            flex_block.parameters(), sdpa_block.parameters()
        ):
            torch.testing.assert_close(flex_parameter.grad, sdpa_parameter.grad)

    def test_flex_block_mask_depends_only_on_sample_and_kv_position(self):
        mask = torch.tensor(
            [
                [[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
                [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]],
            ]
        )
        block_mask = model_pytorch.create_kv_flex_attention_block_mask(mask)
        expected = mask.reshape(2, 9) != 0

        for batch_idx in range(2):
            for kv_idx in range(9):
                expected_value = bool(expected[batch_idx, kv_idx].item())
                for head_idx, query_idx in ((0, 0), (7, 4), (3, 8)):
                    actual = block_mask.mask_mod(
                        torch.tensor(batch_idx),
                        torch.tensor(head_idx),
                        torch.tensor(query_idx),
                        torch.tensor(kv_idx),
                    )
                    self.assertEqual(bool(actual.item()), expected_value)

    def test_odd_half_disable_mask_keeps_all_masks_out_of_trunk(self):
        torch.manual_seed(8642)
        config = self._small_odd_half_config()
        model = model_pytorch.Model(config, pos_len=3).eval()
        model.configure_flex_attention(enabled=True)
        block = model.blocks[0]

        num_spatial = modelconfigs.get_num_bin_input_features(config)
        num_global = modelconfigs.get_num_global_input_features(config)
        spatial = torch.randn(2, num_spatial, 3, 3)
        global_input = torch.randn(2, num_global)

        with mock.patch.object(
            model_pytorch,
            "create_kv_flex_attention_block_mask",
            wraps=model_pytorch.create_kv_flex_attention_block_mask,
        ) as create_mask_mock, mock.patch.object(
            block, "forward", wraps=block.forward
        ) as block_forward_mock:
            model(spatial, global_input, disable_mask=True)

        create_mask_mock.assert_not_called()
        self.assertEqual(block_forward_mock.call_count, 1)
        trunk_input = block_forward_mock.call_args.args[0]
        self.assertEqual(tuple(trunk_input.shape), (2, 4, 16))
        self.assertIsNone(block_forward_mock.call_args.kwargs["mask"])
        self.assertIsNone(block_forward_mock.call_args.kwargs["mask_sum_hw"])
        self.assertIsNone(block_forward_mock.call_args.kwargs["mask_sum"])
        self.assertIsNone(
            block_forward_mock.call_args.kwargs["attention_block_mask"]
        )

    def test_odd_half_rope_block_full_mask_matches_maskless(self):
        torch.manual_seed(8643)
        config = self._small_odd_half_config()
        block_with_mask = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "silu", pos_len=3, use_swiglu=True
        )
        block_without_mask = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "silu", pos_len=3, use_swiglu=True
        )
        block_without_mask.load_state_dict(block_with_mask.state_dict())

        x_with_mask = torch.randn(2, 4, 16, requires_grad=True)
        x_without_mask = x_with_mask.detach().clone().requires_grad_(True)
        mask = torch.ones(2, 4, 1)
        output_with_mask = block_with_mask(
            x_with_mask,
            mask=mask,
            mask_sum_hw=mask.sum(dim=1, keepdim=True).view(2, 1, 1, 1),
            mask_sum=mask.sum(),
        )
        output_without_mask = block_without_mask(
            x_without_mask,
            mask=None,
            mask_sum_hw=None,
            mask_sum=None,
        )

        torch.testing.assert_close(output_without_mask, output_with_mask)
        output_with_mask.square().mean().backward()
        output_without_mask.square().mean().backward()
        torch.testing.assert_close(x_without_mask.grad, x_with_mask.grad)

    def test_odd_half_flex_mask_is_built_from_cropped_trunk_mask(self):
        torch.manual_seed(9754)
        config = self._small_odd_half_config()
        model = model_pytorch.Model(config, pos_len=3).eval()
        model.configure_flex_attention(enabled=True)
        block = model.blocks[0]

        num_spatial = modelconfigs.get_num_bin_input_features(config)
        num_global = modelconfigs.get_num_global_input_features(config)
        spatial = torch.randn(2, num_spatial, 3, 3)
        spatial[:, 0, :, :] = 1.0
        global_input = torch.randn(2, num_global)
        sentinel_block_mask = object()

        with mock.patch.object(
            model_pytorch,
            "create_kv_flex_attention_block_mask",
            return_value=sentinel_block_mask,
        ) as create_mask_mock, mock.patch.object(
            block, "forward", side_effect=lambda x, **kwargs: x
        ) as block_forward_mock:
            model(spatial, global_input, disable_mask=False)

        create_mask_mock.assert_called_once()
        cropped_mask = create_mask_mock.call_args.args[0]
        self.assertEqual(tuple(cropped_mask.shape), (2, 4, 1))
        self.assertEqual(block_forward_mock.call_count, 1)
        self.assertIs(block_forward_mock.call_args.kwargs["mask"], cropped_mask)
        self.assertIs(
            block_forward_mock.call_args.kwargs["attention_block_mask"],
            sentinel_block_mask,
        )

    def test_transformer_preserves_channels_last_without_changing_values(self):
        torch.manual_seed(4321)
        config = {
            "transformer_ffn_channels": 24,
            "transformer_heads": 2,
            "transformer_kv_heads": 2,
            "learnable_rope": True,
        }
        nchw_block = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True
        )
        channels_last_block = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True
        )
        channels_last_block.load_state_dict(nchw_block.state_dict())

        base = torch.randn(2, 16, 3, 3)
        nchw_input = base.detach().clone().contiguous().requires_grad_(True)
        channels_last_input = (
            base.detach()
            .clone()
            .contiguous(memory_format=torch.channels_last)
            .requires_grad_(True)
        )
        nchw_output = nchw_block(
            nchw_input, mask=None, mask_sum_hw=None, mask_sum=None
        )
        channels_last_output = channels_last_block(
            channels_last_input, mask=None, mask_sum_hw=None, mask_sum=None
        )

        self.assertTrue(nchw_output.is_contiguous())
        self.assertTrue(
            channels_last_output.is_contiguous(memory_format=torch.channels_last)
        )
        torch.testing.assert_close(channels_last_output, nchw_output)

        nchw_output.square().mean().backward()
        channels_last_output.square().mean().backward()
        torch.testing.assert_close(channels_last_input.grad, nchw_input.grad)
        for nchw_parameter, channels_last_parameter in zip(
            nchw_block.parameters(), channels_last_block.parameters()
        ):
            torch.testing.assert_close(
                channels_last_parameter.grad, nchw_parameter.grad
            )

    def test_small_full_model_channels_last_outputs_and_gradients_match(self):
        config = copy.deepcopy(modelconfigs.config_of_name["b11c96h4tflrs-bng-silu"])
        config.update(
            {
                "trunk_num_channels": 16,
                "mid_num_channels": 16,
                "gpool_num_channels": 8,
                "transformer_ffn_channels": 24,
                "transformer_heads": 2,
                "transformer_kv_heads": 2,
                "num_attention_pool_heads": 2,
                "block_kind": [["rconv1", "transformerropesg"]],
                "p1_num_channels": 8,
                "g1_num_channels": 8,
                "v1_num_channels": 8,
                "sbv2_num_channels": 8,
                "v2_size": 16,
            }
        )

        def flatten_tensors(value):
            if isinstance(value, torch.Tensor):
                return [value]
            tensors = []
            for item in value:
                tensors.extend(flatten_tensors(item))
            return tensors

        for disable_mask in (False, True):
            with self.subTest(disable_mask=disable_mask):
                torch.manual_seed(2468)
                nchw_model = model_pytorch.Model(config, pos_len=3).eval()
                channels_last_model = model_pytorch.Model(config, pos_len=3).eval()
                channels_last_model.load_state_dict(nchw_model.state_dict())

                base_spatial = torch.randn(2, 22, 3, 3)
                base_spatial[:, 0, :, :] = 1.0
                nchw_spatial = base_spatial.clone().contiguous().requires_grad_(True)
                channels_last_spatial = (
                    base_spatial.clone()
                    .contiguous(memory_format=torch.channels_last)
                    .requires_grad_(True)
                )
                base_global = torch.randn(2, 19)
                nchw_global = base_global.clone().requires_grad_(True)
                channels_last_global = base_global.clone().requires_grad_(True)

                nchw_outputs = nchw_model(
                    nchw_spatial, nchw_global, disable_mask=disable_mask
                )
                channels_last_outputs = channels_last_model(
                    channels_last_spatial,
                    channels_last_global,
                    disable_mask=disable_mask,
                )
                nchw_tensors = flatten_tensors(nchw_outputs)
                channels_last_tensors = flatten_tensors(channels_last_outputs)
                self.assertEqual(len(channels_last_tensors), len(nchw_tensors))
                for channels_last_tensor, nchw_tensor in zip(
                    channels_last_tensors, nchw_tensors
                ):
                    torch.testing.assert_close(
                        channels_last_tensor, nchw_tensor, rtol=2e-5, atol=2e-6
                    )

                sum(tensor.float().square().mean() for tensor in nchw_tensors).backward()
                sum(
                    tensor.float().square().mean()
                    for tensor in channels_last_tensors
                ).backward()
                torch.testing.assert_close(
                    channels_last_spatial.grad,
                    nchw_spatial.grad,
                    rtol=3e-5,
                    atol=3e-6,
                )
                torch.testing.assert_close(
                    channels_last_global.grad,
                    nchw_global.grad,
                    rtol=3e-5,
                    atol=3e-6,
                )
                for channels_last_parameter, nchw_parameter in zip(
                    channels_last_model.parameters(), nchw_model.parameters()
                ):
                    if nchw_parameter.grad is None:
                        self.assertIsNone(channels_last_parameter.grad)
                    else:
                        torch.testing.assert_close(
                            channels_last_parameter.grad,
                            nchw_parameter.grad,
                            rtol=3e-5,
                            atol=3e-6,
                        )

    def test_attention_pool_accepts_channels_last_maskless_input(self):
        torch.manual_seed(9753)
        config = copy.deepcopy(modelconfigs.b1c6nbt)
        config["num_attention_pool_heads"] = 2
        nchw_pool = model_pytorch.KataConvAndAttentionPool(
            "test", c_in=8, c_out=6, c_gpool=4, config=config, activation="mish"
        ).eval()
        channels_last_pool = model_pytorch.KataConvAndAttentionPool(
            "test", c_in=8, c_out=6, c_gpool=4, config=config, activation="mish"
        ).eval()
        channels_last_pool.load_state_dict(nchw_pool.state_dict())

        base = torch.randn(2, 8, 3, 3)
        nchw_input = base.clone().contiguous().requires_grad_(True)
        channels_last_input = (
            base.clone()
            .contiguous(memory_format=torch.channels_last)
            .requires_grad_(True)
        )
        nchw_output = nchw_pool(
            nchw_input,
            mask=None,
            mask_sum_hw=None,
            mask_sum=None,
            extra_outputs=None,
        )
        channels_last_output = channels_last_pool(
            channels_last_input,
            mask=None,
            mask_sum_hw=None,
            mask_sum=None,
            extra_outputs=None,
        )
        torch.testing.assert_close(channels_last_output, nchw_output)

        nchw_output.square().mean().backward()
        channels_last_output.square().mean().backward()
        torch.testing.assert_close(channels_last_input.grad, nchw_input.grad)
        for channels_last_parameter, nchw_parameter in zip(
            channels_last_pool.parameters(), nchw_pool.parameters()
        ):
            torch.testing.assert_close(
                channels_last_parameter.grad, nchw_parameter.grad
            )


if __name__ == "__main__":
    unittest.main()
