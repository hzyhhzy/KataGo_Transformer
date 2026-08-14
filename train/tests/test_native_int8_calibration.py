import math
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import torch


TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from model_pytorch import Model
from native_int8_calibration import (
    ActivationSample,
    AggressiveInt8WeightQDQ,
    BOUNDARY_FIELDS,
    ProcessedRowHashes,
    TransformerBoundaryHooks,
    activation_qdq_scales_float32,
    canonical_float32,
    dequantize_symmetric_int8_fp16,
    make_activation_samples,
    native_code_domain_swiglu_factor_int8,
    qdq_symmetric_int8_fp16,
    quantize_symmetric_int8_fp32,
    quantize_symmetric_int8_fp16,
    requantize_swiglu_factor_product_int8,
    transformer_blocks_in_wire_order,
)


def _config(**updates):
    config = {
        "version": 102,
        "norm_kind": "bnorm",
        "bnorm_epsilon": 1e-4,
        "bnorm_running_avg_momentum": 0.001,
        "bnorm_use_gamma": True,
        "initial_conv_1x1": False,
        "trunk_num_channels": 8,
        "mid_num_channels": 8,
        "gpool_num_channels": 4,
        "transformer_ffn_channels": 12,
        "transformer_heads": 2,
        "transformer_kv_heads": 2,
        "learnable_rope": True,
        "use_qk_norm": True,
        "use_attention_pool": False,
        "num_attention_pool_heads": 2,
        "block_kind": [["direct", "transformerropesg"]],
        "p1_num_channels": 4,
        "g1_num_channels": 4,
        "v1_num_channels": 4,
        "sbv2_num_channels": 8,
        "num_scorebeliefs": 2,
        "v2_size": 8,
        "activation": "silu",
    }
    config.update(updates)
    return config


class NativeInt8CalibrationTests(unittest.TestCase):
    def test_activation_qdq_uses_independent_float32_dequant_scale(self):
        max_abs = 3.7
        multiplier, dequant = activation_qdq_scales_float32(max_abs)
        expected_multiplier = struct.unpack(
            "<f", struct.pack("<f", max_abs)
        )[0]
        expected_multiplier = struct.unpack(
            "<f", struct.pack("<f", 127.0 / expected_multiplier)
        )[0]
        expected_dequant = struct.unpack(
            "<f", struct.pack("<f", struct.unpack("<f", struct.pack("<f", max_abs))[0] / 127.0)
        )[0]
        reciprocal_shortcut = struct.unpack(
            "<f", struct.pack("<f", 1.0 / multiplier)
        )[0]
        self.assertEqual(multiplier, expected_multiplier)
        self.assertEqual(dequant, expected_dequant)
        self.assertNotEqual(dequant, reciprocal_shortcut)
        dequant_bits = struct.unpack("<I", struct.pack("<f", dequant))[0]
        reciprocal_bits = struct.unpack("<I", struct.pack("<f", reciprocal_shortcut))[0]
        self.assertEqual(abs(dequant_bits - reciprocal_bits), 1)

    def test_fp16_qdq_is_rne_symmetric_and_never_minus_128(self):
        source = torch.tensor(
            [-200.0, -128.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 128.0, 200.0],
            dtype=torch.float32,
        )
        actual = qdq_symmetric_int8_fp16(source, 127.0)
        expected = torch.tensor(
            [-127.0, -127.0, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 127.0, 127.0]
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertGreaterEqual(float(actual.min()), -127.0)

    def test_d4_factorwise_product_matches_cpp_formula_byte_for_byte(self):
        # Fixed regressions that distinguish direct integer-product requant
        # from dequantizing each factor to FP16 and multiplying afterward.
        regressions = (
            (-126, -64, 4.0, 16.0, 63),
            (-126, -64, 7.0, 49.0, 63),
            (-127, -57, 7.0, 37.0, 75),
        )
        for up, gate, clip, product, expected in regressions:
            actual = requantize_swiglu_factor_product_int8(
                torch.tensor([up], dtype=torch.int8),
                torch.tensor([gate], dtype=torch.int8),
                clip,
                product,
            )
            self.assertEqual(int(actual.item()), expected)

        generator = torch.Generator().manual_seed(0xD4)
        endpoints = torch.tensor(
            [-127, -126, -65, -64, -1, 0, 1, 63, 64, 126, 127],
            dtype=torch.int8,
        )
        random_up = torch.randint(-127, 128, (4096,), generator=generator).to(torch.int8)
        random_gate = torch.randint(-127, 128, (4096,), generator=generator).to(torch.int8)
        quantized_up = torch.cat((endpoints, endpoints, random_up))
        quantized_gate = torch.cat((endpoints, endpoints.flip(0), random_gate))

        for product_max_abs in (16.0, 23.5):
            actual = requantize_swiglu_factor_product_int8(
                quantized_up, quantized_gate, 4.0, product_max_abs
            )
            clip = canonical_float32(4.0)
            product = canonical_float32(product_max_abs)
            multiplier = canonical_float32(
                float(clip) * float(clip) / (127.0 * float(product))
            )
            expected = []
            for up, gate in zip(quantized_up.tolist(), quantized_gate.tolist()):
                scaled = canonical_float32(float(up * gate) * multiplier)
                rounded = round(scaled)
                expected.append(max(-127, min(127, rounded)))
            expected_bytes = np.asarray(expected, dtype=np.int8).tobytes()
            self.assertEqual(actual.numpy().tobytes(), expected_bytes)

    def test_native_factor_quantization_does_not_round_through_fp16(self):
        # Fixed C++-audit counterexamples: the native factor epilogue keeps
        # FP32 through clip/quantization, whereas an FP16 surrogate is 1 code
        # lower for each value.
        for value, clip in ((-3.95275569, 4.0), (-6.91732264, 7.0)):
            source = torch.tensor([value], dtype=torch.float32)
            native = quantize_symmetric_int8_fp32(source, clip)
            fp16_surrogate = quantize_symmetric_int8_fp16(source, clip)
            self.assertEqual(int(native.item()), -125)
            self.assertEqual(int(fp16_surrogate.item()), -126)

    def test_native_code_domain_factor_dot_and_gate_are_byte_exact(self):
        generator = torch.Generator().manual_seed(0xC0DE)
        quantized_input = torch.randint(
            -127, 128, (3, 5, 8), generator=generator
        ).to(torch.int8)
        quantized_weight = torch.randint(
            -127, 128, (7, 8), generator=generator
        ).to(torch.int8)
        input_max_abs = 3.7
        weight_scale = 0.013
        clip = 4.0
        actual = native_code_domain_swiglu_factor_int8(
            quantized_input,
            quantized_weight,
            input_max_abs,
            weight_scale,
            clip,
            apply_silu=False,
        )

        input_scale = canonical_float32(
            canonical_float32(input_max_abs) / 127.0
        )
        alpha = canonical_float32(
            input_scale * canonical_float32(weight_scale)
        )
        multiplier = canonical_float32(127.0 / canonical_float32(clip))
        accumulators = np.matmul(
            quantized_input.numpy().astype(np.int32),
            quantized_weight.numpy().astype(np.int32).T,
        )
        expected = []
        for accumulator in accumulators.reshape(-1):
            value = canonical_float32(int(accumulator) * alpha)
            clipped = max(-clip, min(clip, value))
            scaled = canonical_float32(clipped * multiplier)
            expected.append(max(-127, min(127, round(scaled))))
        expected_bytes = np.asarray(expected, dtype=np.int8).tobytes()
        self.assertEqual(actual.numpy().tobytes(), expected_bytes)

    def test_native_code_domain_silu_is_stable_and_matches_scalar_with_tolerance(self):
        generator = torch.Generator().manual_seed(0x51A0)
        quantized_input = torch.randint(
            -127, 128, (4, 9, 8), generator=generator
        ).to(torch.int8)
        quantized_weight = torch.randint(
            -127, 128, (11, 8), generator=generator
        ).to(torch.int8)
        input_max_abs = 3.7
        weight_scale = 0.013
        clip = 4.0
        first = native_code_domain_swiglu_factor_int8(
            quantized_input,
            quantized_weight,
            input_max_abs,
            weight_scale,
            clip,
            apply_silu=True,
        )
        second = native_code_domain_swiglu_factor_int8(
            quantized_input,
            quantized_weight,
            input_max_abs,
            weight_scale,
            clip,
            apply_silu=True,
        )
        self.assertTrue(torch.equal(first, second))

        input_scale = canonical_float32(
            canonical_float32(input_max_abs) / 127.0
        )
        alpha = canonical_float32(
            input_scale * canonical_float32(weight_scale)
        )
        multiplier = canonical_float32(127.0 / canonical_float32(clip))
        accumulators = np.matmul(
            quantized_input.numpy().astype(np.int32),
            quantized_weight.numpy().astype(np.int32).T,
        )
        scalar_codes = []
        for accumulator in accumulators.reshape(-1):
            value = canonical_float32(int(accumulator) * alpha)
            silu = value / (1.0 + math.exp(-value))
            clipped = max(-clip, min(clip, silu))
            scaled = canonical_float32(clipped * multiplier)
            scalar_codes.append(max(-127, min(127, round(scaled))))
        scalar = np.asarray(scalar_codes, dtype=np.int16).reshape(first.shape)
        difference = np.abs(first.numpy().astype(np.int16) - scalar)
        self.assertLessEqual(int(difference.max()), 1)
        self.assertGreater(float(np.mean(difference == 0)), 0.98)

    def test_fp16_histogram_observes_every_value(self):
        sample = ActivationSample()
        values = torch.arange(0, 4096, dtype=torch.float32).remainder(37) / 8.0
        sample.observe(values)
        summary = sample.summary()
        self.assertEqual(summary["observedValues"], values.numel())
        self.assertEqual(summary["sampledValues"], values.numel())
        self.assertEqual(summary["observations"], 1)
        self.assertEqual(sample.threshold(None), float(values.half().max()))
        self.assertGreater(sample.threshold(99.9), 0.0)

    def test_processed_row_hashes_detect_partial_dataset_overlap(self):
        calibration = ProcessedRowHashes()
        validation = ProcessedRowHashes()
        calibration.observe_batch({
            "binaryInputNCHW": torch.tensor([[[[1.0]]], [[[2.0]]]]),
            "globalInputNC": torch.tensor([[3.0], [4.0]]),
        }, include_metadata=False)
        validation.observe_batch({
            "binaryInputNCHW": torch.tensor([[[[9.0]]], [[[2.0]]]]),
            "globalInputNC": torch.tensor([[8.0], [4.0]]),
        }, include_metadata=False)
        self.assertEqual(len(calibration.digests & validation.digests), 1)
        self.assertEqual(calibration.summary()["rows"], 2)
        self.assertEqual(validation.summary()["uniqueRows"], 2)

    def test_aggressive_weight_qdq_shares_qkv_scale_and_restores_bytes(self):
        torch.manual_seed(5)
        model = Model(_config(), pos_len=5).eval()
        model.initialize()
        layers = transformer_blocks_in_wire_order(model)
        self.assertEqual(len(layers), 1)
        _, block = layers[0]
        with torch.no_grad():
            block.q_proj.weight.copy_(torch.linspace(-0.7, 0.7, block.q_proj.weight.numel()).view_as(block.q_proj.weight))
            block.k_proj.weight.copy_(torch.linspace(-1.3, 1.3, block.k_proj.weight.numel()).view_as(block.k_proj.weight))
            block.v_proj.weight.copy_(torch.linspace(-2.1, 2.1, block.v_proj.weight.numel()).view_as(block.v_proj.weight))

        projection_parameters = []
        for _, layer in layers:
            projection_parameters.extend((
                layer.q_proj.weight,
                layer.k_proj.weight,
                layer.v_proj.weight,
                layer.out_proj.weight,
                layer.ffn_linear1.weight,
                layer.ffn_linear_gate.weight,
                layer.ffn_linear2.weight,
            ))
        original_bytes = [
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for parameter in projection_parameters
        ]

        spatial = torch.randn(2, model.bin_input_shape[0], 5, 5)
        spatial[:, 0].fill_(1.0)
        global_input = torch.randn(2, model.global_input_shape[0])
        with torch.inference_mode():
            baseline = model(spatial, global_input)[0][0].clone()

        with AggressiveInt8WeightQDQ(layers) as qdq:
            expected_scale = struct.unpack(
                "<f", struct.pack("<f", 2.1 / 127.0)
            )[0]
            self.assertAlmostEqual(
                qdq.scales["model.blocks.0"]["qkvSharedWeightScale"],
                expected_scale,
                places=8,
            )
            with torch.inference_mode():
                quantized = model(spatial, global_input)[0][0]
            self.assertFalse(torch.equal(baseline, quantized))

        restored_bytes = [
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for parameter in projection_parameters
        ]
        self.assertEqual(restored_bytes, original_bytes)

    def test_no_clip_product_is_observed_without_mutating_model_semantics(self):
        torch.manual_seed(7)
        model = Model(_config(), pos_len=5).eval()
        model.initialize()
        layers = transformer_blocks_in_wire_order(model)
        layer_order = [name for name, _ in layers]
        samples = make_activation_samples(layer_order)
        spatial = torch.randn(1, model.bin_input_shape[0], 5, 5)
        spatial[:, 0].fill_(1.0)
        global_input = torch.randn(1, model.global_input_shape[0])
        self.assertIsNone(layers[0][1].swiglu_clip)
        with torch.inference_mode(), TransformerBoundaryHooks(layers, samples=samples):
            model(spatial, global_input)
        self.assertIsNone(layers[0][1].swiglu_clip)
        product = samples[layer_order[0]]["productQuantMaxAbs"].summary()
        self.assertGreater(product["observedValues"], 0)
        self.assertGreater(product["maxAbs"], 0.0)

    def test_explicit_zero_clip_is_the_same_no_clip_model_semantic(self):
        model = Model(_config(swiglu_clip=0.0), pos_len=5).eval()
        layers = transformer_blocks_in_wire_order(model)
        self.assertIsNone(layers[0][1].swiglu_clip)

    def _assert_candidate_swiglu_path(self, swiglu_clip):
        torch.manual_seed(17)
        updates = {} if swiglu_clip is None else {"swiglu_clip": swiglu_clip}
        model = Model(_config(**updates), pos_len=5).eval()
        model.initialize()
        layers = transformer_blocks_in_wire_order(model)
        layer_name, block = layers[0]
        self.assertEqual(block.swiglu_clip, swiglu_clip)
        spatial = torch.randn(1, model.bin_input_shape[0], 5, 5)
        spatial[:, 0].fill_(1.0)
        global_input = torch.randn(1, model.global_input_shape[0])
        thresholds = {
            layer_name: {
                "attentionInputQuantMaxAbs": 3.7,
                "attentionOutputQuantMaxAbs": 3.7,
                "ffnInputQuantMaxAbs": 3.7,
                "productQuantMaxAbs": 3.7,
            }
        }

        with torch.inference_mode():
            baseline = model(spatial, global_input)[0][0].clone()

        raw_activated_up = []
        raw_gate = []
        raw_ffn_input = []
        down_input = []
        with AggressiveInt8WeightQDQ(layers) as weight_qdq:
            capture_input = block.ffn_linear1.register_forward_pre_hook(
                lambda _module, args: raw_ffn_input.append(args[0].detach().clone())
            )
            capture_up = block.ffn_act.register_forward_hook(
                lambda _module, _args, output: raw_activated_up.append(output.detach().clone())
            )
            capture_gate = block.ffn_linear_gate.register_forward_hook(
                lambda _module, _args, output: raw_gate.append(output.detach().clone())
            )
            corrupt_handles = []
            if swiglu_clip is not None:
                # The native-code-domain factor path must ignore the ordinary
                # PyTorch Linear/SiLU outputs completely.
                corrupt_handles = [
                    block.ffn_act.register_forward_hook(
                        lambda _module, _args, output: torch.full_like(output, 123.0)
                    ),
                    block.ffn_linear_gate.register_forward_hook(
                        lambda _module, _args, output: torch.full_like(output, -57.0)
                    ),
                ]
            try:
                with TransformerBoundaryHooks(
                    layers, thresholds=thresholds, weight_qdq=weight_qdq
                ):
                    capture_down = block.ffn_linear2.register_forward_pre_hook(
                        lambda _module, args: down_input.append(args[0].detach().clone())
                    )
                    try:
                        with torch.inference_mode():
                            candidate = model(spatial, global_input)[0][0].clone()
                    finally:
                        capture_down.remove()
            finally:
                for handle in corrupt_handles:
                    handle.remove()
                capture_gate.remove()
                capture_up.remove()
                capture_input.remove()

            self.assertEqual(len(raw_ffn_input), 1)
            self.assertEqual(len(raw_activated_up), 1)
            self.assertEqual(len(raw_gate), 1)
            self.assertEqual(len(down_input), 1)
            if swiglu_clip is None:
                expected_factors = raw_activated_up[0] * raw_gate[0]
                expected_down_input = qdq_symmetric_int8_fp16(
                    expected_factors, thresholds[layer_name]["productQuantMaxAbs"]
                )
            else:
                quantized_input = quantize_symmetric_int8_fp16(
                    raw_ffn_input[0], thresholds[layer_name]["ffnInputQuantMaxAbs"]
                )
                quantized_up = native_code_domain_swiglu_factor_int8(
                    quantized_input,
                    weight_qdq.quantized_weights[layer_name]["ffnUp"],
                    thresholds[layer_name]["ffnInputQuantMaxAbs"],
                    weight_qdq.scales[layer_name]["ffnUpWeightScale"],
                    swiglu_clip,
                    apply_silu=True,
                )
                quantized_gate = native_code_domain_swiglu_factor_int8(
                    quantized_input,
                    weight_qdq.quantized_weights[layer_name]["ffnGate"],
                    thresholds[layer_name]["ffnInputQuantMaxAbs"],
                    weight_qdq.scales[layer_name]["ffnGateWeightScale"],
                    swiglu_clip,
                    apply_silu=False,
                )
                quantized_product = requantize_swiglu_factor_product_int8(
                    quantized_up,
                    quantized_gate,
                    swiglu_clip,
                    thresholds[layer_name]["productQuantMaxAbs"],
                )
                expected_down_input = dequantize_symmetric_int8_fp16(
                    quantized_product,
                    thresholds[layer_name]["productQuantMaxAbs"],
                    raw_activated_up[0].dtype,
                )
        torch.testing.assert_close(
            down_input[0], expected_down_input, rtol=0.0, atol=0.0
        )
        self.assertFalse(torch.equal(candidate, baseline))
        with torch.inference_mode():
            restored = model(spatial, global_input)[0][0]
        torch.testing.assert_close(restored, baseline, rtol=0.0, atol=0.0)
        self.assertEqual(block.swiglu_clip, swiglu_clip)

    def test_clip4_candidate_requantizes_integer_factor_product_directly(self):
        self._assert_candidate_swiglu_path(4.0)

    def test_no_clip_candidate_keeps_float_factors_and_only_qdqs_product(self):
        self._assert_candidate_swiglu_path(None)


if __name__ == "__main__":
    unittest.main()
