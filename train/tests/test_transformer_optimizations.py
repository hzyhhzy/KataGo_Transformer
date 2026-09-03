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

    def test_qk_norm_model_config_suffix(self):
        base_name = "b11c96h4tflrs-bng-silu"
        qkn_config = modelconfigs.config_of_name[base_name + "-qkn"]

        self.assertTrue(qkn_config["use_qk_norm"])
        self.assertNotIn("use_qk_norm", modelconfigs.config_of_name[base_name])
        self.assertNotIn("b1c6nbt-qkn", modelconfigs.config_of_name)

    def test_swiglu_clip_model_config_suffix(self):
        base_name = "b11c96h4tflrs-bng-silu"
        clip4_config = modelconfigs.config_of_name[base_name + "-clip4"]
        clip_config = modelconfigs.config_of_name[base_name + "-clip7"]
        combined_clip4_config = modelconfigs.config_of_name[
            base_name + "-qkn-clip4"
        ]
        combined_config = modelconfigs.config_of_name[
            base_name + "-qkn-clip7"
        ]

        self.assertEqual(clip4_config["swiglu_clip"], 4.0)
        self.assertEqual(clip_config["swiglu_clip"], 7.0)
        self.assertNotIn("swiglu_clip", modelconfigs.config_of_name[base_name])
        self.assertTrue(combined_clip4_config["use_qk_norm"])
        self.assertEqual(combined_clip4_config["swiglu_clip"], 4.0)
        self.assertTrue(combined_config["use_qk_norm"])
        self.assertEqual(combined_config["swiglu_clip"], 7.0)
        self.assertNotIn("b11c96h3tfr-clip4", modelconfigs.config_of_name)
        self.assertNotIn("b11c96h3tfr-clip7", modelconfigs.config_of_name)

    def test_b36c384h12_transformer_config_and_variants(self):
        base = modelconfigs.config_of_name["b36c384h12tfrs"]
        self.assertEqual(base["trunk_num_channels"], 384)
        self.assertEqual(base["transformer_ffn_channels"], 1024)
        self.assertEqual(base["transformer_heads"], 12)
        self.assertEqual(base["transformer_kv_heads"], 12)
        self.assertEqual(
            base["block_kind"],
            [[f"rconv{i}", "transformerropesg"] for i in range(1, 37)],
        )

        production = modelconfigs.config_of_name[
            "b36c384h12tflrs-bng-silu-v102-qkn-clip4"
        ]
        self.assertTrue(production["learnable_rope"])
        self.assertEqual(production["norm_kind"], "bnorm")
        self.assertTrue(production["bnorm_use_gamma"])
        self.assertEqual(production["activation"], "silu")
        self.assertEqual(production["version"], 102)
        self.assertTrue(production["use_qk_norm"])
        self.assertEqual(production["swiglu_clip"], 4.0)

    def test_cpu_ptq_profile_configs_and_variants(self):
        expected = {
            "b16c128h4tfrs": (16, 128, 4, 384),
            "b24c192h6tfrs": (24, 192, 6, 512),
        }
        for name, (blocks, channels, heads, ffn_channels) in expected.items():
            with self.subTest(name=name):
                base = modelconfigs.config_of_name[name]
                self.assertEqual(base["trunk_num_channels"], channels)
                self.assertEqual(base["transformer_heads"], heads)
                self.assertEqual(base["transformer_kv_heads"], heads)
                self.assertEqual(base["transformer_ffn_channels"], ffn_channels)
                self.assertEqual(base["v2_size"], 96)
                self.assertEqual(
                    base["block_kind"],
                    [
                        [f"rconv{i}", "transformerropesg"]
                        for i in range(1, blocks + 1)
                    ],
                )

                production = modelconfigs.config_of_name[
                    name[:-4] + "tflrs-bng-silu-v102-qkn-clip4"
                ]
                self.assertTrue(production["learnable_rope"])
                self.assertEqual(production["version"], 102)
                self.assertTrue(production["use_qk_norm"])
                self.assertEqual(production["swiglu_clip"], 4.0)

    def test_full_clip_model_config_suffix(self):
        swiglu_name = "b11c96h4tflrs-bng-silu"
        non_swiglu_name = "b11c96h3tfr"

        for clip_value in (4.0, 7.0):
            suffix = f"-fullclip{int(clip_value)}"
            with self.subTest(clip_value=clip_value):
                self.assertEqual(
                    modelconfigs.config_of_name[swiglu_name + suffix][
                        "full_int8_clip"
                    ],
                    clip_value,
                )
                self.assertEqual(
                    modelconfigs.config_of_name[non_swiglu_name + suffix][
                        "full_int8_clip"
                    ],
                    clip_value,
                )
                combined = modelconfigs.config_of_name[
                    swiglu_name + "-qkn" + suffix
                ]
                self.assertTrue(combined["use_qk_norm"])
                self.assertEqual(combined["full_int8_clip"], clip_value)
                self.assertNotIn(
                    swiglu_name + "-clip4" + suffix,
                    modelconfigs.config_of_name,
                )

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

    def test_qk_norm_normalizes_each_head_before_attention(self):
        torch.manual_seed(24601)
        config = {
            "transformer_ffn_channels": 24,
            "transformer_heads": 4,
            "transformer_kv_heads": 2,
            "learnable_rope": False,
            "use_qk_norm": True,
        }
        block = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True, use_rope=False
        )
        captured = {}

        def capture_attention(q, k, v, **kwargs):
            captured["q"] = q.detach()
            captured["k"] = k.detach()
            return torch.zeros_like(q)

        x = torch.randn(2, 16, 3, 3)
        with mock.patch.object(
            torch.nn.functional,
            "scaled_dot_product_attention",
            side_effect=capture_attention,
        ):
            block(x, mask=None, mask_sum_hw=None, mask_sum=None)

        self.assertIsInstance(block.q_norm, torch.nn.RMSNorm)
        self.assertIsInstance(block.k_norm, torch.nn.RMSNorm)
        torch.testing.assert_close(
            captured["q"].float().square().mean(dim=-1),
            torch.ones_like(captured["q"][..., 0], dtype=torch.float32),
            rtol=2e-4,
            atol=2e-4,
        )
        torch.testing.assert_close(
            captured["k"].float().square().mean(dim=-1),
            torch.ones_like(captured["k"][..., 0], dtype=torch.float32),
            rtol=2e-4,
            atol=2e-4,
        )

        reg_dict = {
            "normal": [],
            "normal_gamma": [],
            "normal_attn": [],
            "output": [],
            "noreg": [],
            "output_noreg": [],
        }
        block.add_reg_dict(reg_dict)
        self.assertTrue(any(param is block.q_norm.weight for param in reg_dict["noreg"]))
        self.assertTrue(any(param is block.k_norm.weight for param in reg_dict["noreg"]))

    def test_qk_norm_disabled_keeps_parameter_layout_compatible(self):
        config = {
            "transformer_ffn_channels": 24,
            "transformer_heads": 2,
            "transformer_kv_heads": 2,
            "learnable_rope": False,
        }
        block = model_pytorch.TransformerRoPEGQABlock(
            "test", 16, config, "mish", pos_len=3, use_swiglu=True, use_rope=False
        )
        self.assertIsInstance(block.q_norm, torch.nn.Identity)
        self.assertIsInstance(block.k_norm, torch.nn.Identity)
        self.assertFalse(
            any("q_norm" in key or "k_norm" in key for key in block.state_dict())
        )

    def test_clip_suffixes_clamp_both_swiglu_multipliers(self):
        for clip_value in (4.0, 7.0):
            with self.subTest(clip_value=clip_value):
                self._check_swiglu_multiplier_clipping(clip_value)

    def _check_swiglu_multiplier_clipping(self, clip_value):
        config = {
            "transformer_ffn_channels": 4,
            "transformer_heads": 1,
            "transformer_kv_heads": 1,
            "learnable_rope": False,
            "swiglu_clip": clip_value,
        }
        block = model_pytorch.TransformerRoPEGQABlock(
            "test", 4, config, "mish", pos_len=1, use_swiglu=True, use_rope=False
        )
        with torch.no_grad():
            block.q_proj.weight.zero_()
            block.k_proj.weight.zero_()
            block.v_proj.weight.zero_()
            block.out_proj.weight.zero_()
            block.ffn_linear1.weight.fill_(10.0)
            block.ffn_linear_gate.weight.fill_(-10.0)

        captured = {}

        def capture_linear2_input(module, args):
            captured["swiglu_product"] = args[0].detach().clone()

        handle = block.ffn_linear2.register_forward_pre_hook(capture_linear2_input)
        try:
            block(
                torch.ones(1, 4, 1, 1),
                mask=None,
                mask_sum_hw=None,
                mask_sum=None,
            )
        finally:
            handle.remove()

        torch.testing.assert_close(
            captured["swiglu_product"],
            torch.full_like(captured["swiglu_product"], -(clip_value ** 2)),
        )

    def test_clip7_is_ignored_by_non_swiglu_blocks(self):
        config = {
            "transformer_ffn_channels": 4,
            "transformer_heads": 1,
            "transformer_kv_heads": 1,
            "learnable_rope": False,
            "swiglu_clip": 7.0,
        }
        block = model_pytorch.TransformerRoPEGQABlock(
            "test", 4, config, "mish", pos_len=1, use_swiglu=False, use_rope=False
        )
        self.assertIsNone(block.swiglu_clip)

    def test_full_clip_bounds_transformer_int8_activation_boundaries(self):
        for use_swiglu in (False, True):
            with self.subTest(use_swiglu=use_swiglu):
                self._check_full_clip_boundaries(use_swiglu)

    def _check_full_clip_boundaries(self, use_swiglu):
        config = {
            "transformer_ffn_channels": 4,
            "transformer_heads": 1,
            "transformer_kv_heads": 1,
            "learnable_rope": False,
            "full_int8_clip": 4.0,
        }
        block = model_pytorch.TransformerRoPEGQABlock(
            "test",
            4,
            config,
            "mish",
            pos_len=1,
            use_swiglu=use_swiglu,
            use_rope=False,
        )
        with torch.no_grad():
            for module in block.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(100.0)

        captured = {}

        def capture_input(name):
            def hook(module, args):
                captured[name] = args[0].detach().clone()
            return hook

        handles = [
            block.q_proj.register_forward_pre_hook(capture_input("q_proj")),
            block.k_proj.register_forward_pre_hook(capture_input("k_proj")),
            block.v_proj.register_forward_pre_hook(capture_input("v_proj")),
            block.out_proj.register_forward_pre_hook(capture_input("out_proj")),
            block.ffn_linear1.register_forward_pre_hook(capture_input("ffn_linear1")),
            block.ffn_linear2.register_forward_pre_hook(capture_input("ffn_linear2")),
        ]
        if use_swiglu:
            handles.append(
                block.ffn_linear_gate.register_forward_pre_hook(
                    capture_input("ffn_linear_gate")
                )
            )

        def fake_attention(q, k, v, **kwargs):
            captured["attention_q"] = q.detach().clone()
            captured["attention_k"] = k.detach().clone()
            captured["attention_v"] = v.detach().clone()
            return q * 100.0

        try:
            with mock.patch.object(
                torch.nn.functional,
                "scaled_dot_product_attention",
                side_effect=fake_attention,
            ):
                output = block(
                    torch.full((1, 4, 1, 1), 100.0),
                    mask=None,
                    mask_sum_hw=None,
                    mask_sum=None,
                )
        finally:
            for handle in handles:
                handle.remove()

        for name, tensor in captured.items():
            self.assertLessEqual(
                tensor.abs().max().item(),
                4.0,
                msg=f"{name} exceeded the full INT8 clipping range",
            )
        self.assertLessEqual(output.abs().max().item(), 4.0)

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
