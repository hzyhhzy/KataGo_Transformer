from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np
import torch


TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from calibrate_cpu_ptq_v106 import (
    PROJECTION_ROLES,
    Moments,
    ProjectionQuantController,
    cpu_ptq_model_version,
    gptq_matrix,
    symmetric_codes,
)
from convert_cpu_ptq import (
    FORMAT_BY_BASE_VERSION,
    FP32_MARKER,
    S7_MARKER,
    convert,
    parse_header,
    read_model,
    scan_attention_headers,
    scan_projections,
)


class CpuPtqV106Tests(unittest.TestCase):
    def test_cpu_ptq_wire_version_follows_source_input_version(self) -> None:
        self.assertEqual(cpu_ptq_model_version(102), 106)
        self.assertEqual(cpu_ptq_model_version(11), 206)
        with self.assertRaises(ValueError):
            cpu_ptq_model_version(15)

    def test_projection_controller_accepts_model_sized_overrides(self) -> None:
        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for role in PROJECTION_ROLES:
                    setattr(self, role, torch.nn.Linear(2, 2, bias=False))

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block()])

        model = Model()
        original = model.blocks[0].q_proj(torch.ones(1, 2)).detach()
        overrides = {
            f"blocks.0.{role}": (
                torch.zeros(2, 2, dtype=torch.int8),
                torch.ones(2, dtype=torch.float32),
            )
            for role in PROJECTION_ROLES
        }
        controller = ProjectionQuantController(model, 127, overrides)
        try:
            quantized = model.blocks[0].q_proj(torch.ones(1, 2))
            self.assertTrue(torch.equal(quantized, torch.zeros_like(quantized)))
        finally:
            controller.close()
        restored = model.blocks[0].q_proj(torch.ones(1, 2)).detach()
        torch.testing.assert_close(restored, original)

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

    def test_native_format_pairs_share_projection_encoding(self) -> None:
        self.assertEqual(FORMAT_BY_BASE_VERSION[102].quantized_version, 106)
        self.assertEqual(FORMAT_BY_BASE_VERSION[11].quantized_version, 206)
        self.assertEqual(FORMAT_BY_BASE_VERSION[102].global_inputs, 39)
        self.assertEqual(FORMAT_BY_BASE_VERSION[11].global_inputs, 19)

        values = np.arange(6, dtype="<f4").reshape(2, 3)
        fp32_payload = (
            b"toy\n11\n22\n19\n"
            b"model.blocks.0.attention.q_proj\n2\n3\n"
            + FP32_MARKER
            + values.tobytes(order="C")
            + b"\n"
        )
        header = parse_header(fp32_payload)
        self.assertEqual((header.version, header.spatial_inputs, header.global_inputs),
                         (11, 22, 19))
        projection = scan_projections(fp32_payload)[0]
        self.assertEqual(projection.canonical_name, "blocks.0.q_proj")
        np.testing.assert_array_equal(projection.values_input_major, values)

        scales = np.asarray([0.25, 0.5, 1.0], dtype="<f4")
        codes = np.asarray(
            [[1, 2], [3, 4], [-5, -6]], dtype=np.int8
        )
        s7_payload = (
            b"toy\n206\n22\n19\n"
            b"model.blocks.0.attention.q_proj\n2\n3\n"
            + S7_MARKER
            + scales.tobytes(order="C")
            + codes.tobytes(order="C")
            + b"\n"
        )
        projection = scan_projections(s7_payload)[0]
        self.assertEqual(projection.qmax, 63)
        np.testing.assert_array_equal(projection.scales, scales)
        np.testing.assert_array_equal(projection.codes_output_major, codes)

        attention = scan_attention_headers(
            b"transformer_attention_block\n"
            b"model.blocks.0.attention\n3\n3\n32\n32\n1\n0\n"
        )[0]
        self.assertEqual(
            (attention.block, attention.heads, attention.kv_heads),
            (0, 3, 3),
        )
        self.assertTrue(attention.use_rope)
        self.assertFalse(attention.learnable_rope)

    def test_v106_v206_share_qkn_clip_wire_for_arbitrary_profile(self) -> None:
        blocks = 2
        channels = 64
        heads = 2
        ffn = 80

        def matrix(name: str, inputs: int, outputs: int) -> bytes:
            values = np.linspace(
                -1.0,1.0,inputs * outputs,dtype=np.dtype("<f4")
            )
            return (
                f"{name}\n{inputs}\n{outputs}\n".encode("ascii")
                + FP32_MARKER
                + values.tobytes(order="C")
                + b"\n"
            )

        def norm(name: str, width: int) -> bytes:
            return (
                f"{name}\n{width}\n1e-06\n".encode("ascii")
                + FP32_MARKER
                + np.ones(width,dtype=np.dtype("<f4")).tobytes(order="C")
                + b"\n"
            )

        body = bytearray()
        for block in range(blocks):
            attention = f"model.blocks.{block}.attention"
            body.extend(
                (
                    "transformer_attention_block\n"
                    f"{attention}\n{heads}\n{heads}\n32\n32\n1\n0\n"
                    "@V102_QKN_CLIP@\n1\n"
                ).encode("ascii")
            )
            body.extend(norm(attention + ".norm1",channels))
            for role in ("q_proj","k_proj","v_proj","out_proj"):
                body.extend(matrix(attention + "." + role,channels,channels))
            body.extend(norm(attention + ".q_norm",32))
            body.extend(norm(attention + ".k_norm",32))
            body.extend((attention + ".rope_theta\n100.0\n").encode("ascii"))

            ffn_name = f"model.blocks.{block}.ffn"
            body.extend(
                (
                    "transformer_ffn_block\n"
                    f"{ffn_name}\n{channels}\n{ffn}\n1\n"
                    "@V102_QKN_CLIP@\n4.0\n"
                ).encode("ascii")
            )
            body.extend(norm(ffn_name + ".norm",channels))
            body.extend(matrix(ffn_name + ".ffn_linear1",channels,ffn))
            body.extend(matrix(ffn_name + ".ffn_linear_gate",channels,ffn))
            body.extend(matrix(ffn_name + ".ffn_linear2",ffn,channels))

        temp_dir = TRAIN_DIR / "tests" / ("cpu_ptq_wire_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            outputs = []
            for base_version,global_inputs,target_version in (
                (102,39,106),(11,19,206)
            ):
                source = temp_dir / f"v{base_version}.bin"
                target = temp_dir / f"v{target_version}.bin.gz"
                source.write_bytes(
                    f"toy\n{base_version}\n22\n{global_inputs}\n".encode("ascii")
                    + body
                )
                report = convert(
                    source,target,manifest_path=None,projection_bits=7,
                    force=False,compression_level=1,write_report=False,
                )
                self.assertEqual(report["profile"],"b2c64h2-f80")
                self.assertEqual(report["modelVersion"],target_version)
                payload = read_model(target)
                self.assertEqual(parse_header(payload).version,target_version)
                self.assertEqual(payload.count(b"@V102_QKN_CLIP@\n"),4)
                self.assertTrue(all(
                    projection.marker == S7_MARKER
                    for projection in scan_projections(payload)
                ))
                outputs.append(payload.split(b"\n",4)[4])
            self.assertEqual(outputs[0],outputs[1])
        finally:
            shutil.rmtree(temp_dir,ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
