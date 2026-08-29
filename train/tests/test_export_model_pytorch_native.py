import gzip
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import unittest
import uuid

import torch


TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from model_pytorch import Model
from native_int8_calibration import (
    BOUNDARY_FIELDS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    WIRE_VERSION,
    WEIGHT_SCALE_FIELDS,
    sha256_file,
    transformer_blocks_in_wire_order,
)


def _read_line(data, offset):
    end = data.index(b"\n", offset)
    return data[offset:end].decode("ascii"), end + 1


def _read_binary_floats(data, offset, count):
    marker = b"@BIN@"
    if data[offset:offset + len(marker)] != marker:
        raise AssertionError(f"missing binary marker at byte {offset}")
    offset += len(marker)
    size = count * 4
    values = struct.unpack(f"<{count}f", data[offset:offset + size])
    offset += size
    if data[offset:offset + 1] != b"\n":
        raise AssertionError(f"missing binary payload newline at byte {offset}")
    return values, offset + 1


def _ffn_clip_and_quant_ranges(data):
    result = []
    for descriptor in data.split(b"transformer_ffn_block\n")[1:]:
        fields = descriptor.split(b"\n", 7)
        if len(fields) != 8:
            raise AssertionError("truncated transformer FFN descriptor")
        result.append((float(fields[4]),float(fields[5]),float(fields[6])))
    return result


class NativeTransformerExportTests(unittest.TestCase):
    def _tiny_config(self, **updates):
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
            "use_attention_pool": False,
            "num_attention_pool_heads": 2,
            "block_kind": [
                ["direct", "transformerropesg"],
                ["nested", "bottlenest2transformerropesg"],
            ],
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

    def _export_command(
        self,
        checkpoint,
        export_dir,
        gzip_output,
        int8_pt_clip4=False,
        calibration_json=None,
        cpu_ptq_base=False,
        export_14_as_15=False,
    ):
        command = [
            sys.executable,
            str(TRAIN_DIR / "export_model_pytorch.py"),
            "-checkpoint",
            str(checkpoint),
            "-export-dir",
            str(export_dir),
            "-model-name",
            "native-export-test",
            "-filename-prefix",
            "model",
            "-pos-len",
            "5",
        ]
        if gzip_output:
            command.append("-gzip")
        if int8_pt_clip4:
            command.append("-int8-pt-clip4")
        if calibration_json is not None:
            command.extend(("-int8-calibration-json",str(calibration_json)))
        if cpu_ptq_base:
            command.append("-cpu-ptq-base")
        if export_14_as_15:
            command.append("-export-14-as-15")
        return command

    def _write_calibration_json(self, checkpoint, config, path, values=None):
        model = Model(config, pos_len=5)
        layer_order = [
            name for name, _ in transformer_blocks_in_wire_order(model)
        ]
        candidates = ["p99.9", "p99.99", "p99.999", "minmax"]
        layers = []
        for index, name in enumerate(layer_order):
            selected = (
                values[index]
                if values is not None
                else {
                    "attentionInputQuantMaxAbs": 10.0 + index,
                    "attentionOutputQuantMaxAbs": 20.0 + index,
                    "ffnInputQuantMaxAbs": 30.0 + index,
                    "productQuantMaxAbs": 40.0 + index,
                }
            )
            layer_candidates = {
                candidate: {
                    "thresholds": dict(selected),
                    "calibrationSaturationRates": {
                        field: 0.0 for field in BOUNDARY_FIELDS
                    },
                }
                for candidate in candidates
            }
            layers.append({
                "index": index,
                "name": name,
                **selected,
                "calibrationSample": {
                    field: {
                        "observedValues": 100,
                        "sampledValues": 100,
                        "observations": 1,
                        "maxAbs": selected[field],
                    }
                    for field in BOUNDARY_FIELDS
                },
                "candidates": layer_candidates,
                "validationSaturationRates": {
                    candidate: {field: 0.0 for field in BOUNDARY_FIELDS}
                    for candidate in candidates
                },
                "weightQdqScales": {
                    field: 0.01 + index * 0.001
                    for field in WEIGHT_SCALE_FIELDS
                },
            })
        document = {
            "schema": SCHEMA_NAME,
            "schemaVersion": SCHEMA_VERSION,
            "wireVersion": WIRE_VERSION,
            "source": {
                "checkpoint": {
                    "sha256": sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                },
                "calibrationData": {
                    "sha256": "1" * 64,
                    "files": [{"index": 0, "name": "calib.npz", "bytes": 1, "sha256": "2" * 64}],
                },
                "validationData": {
                    "sha256": "3" * 64,
                    "files": [{"index": 0, "name": "eval.npz", "bytes": 1, "sha256": "4" * 64}],
                },
                "processedRows": {
                    "calibrationRows": 64,
                    "calibrationUniqueRows": 64,
                    "calibrationSetSha256": "5" * 64,
                    "validationRows": 64,
                    "validationUniqueRows": 64,
                    "validationSetSha256": "6" * 64,
                    "overlapRows": 0,
                },
            },
            "evaluation": {"modelState": "raw", "useSwa": False, "posLen": 5},
            "quantization": {
                "dtype": "int8",
                "qmin": -127,
                "qmax": 127,
                "zeroPoint": 0,
                "rounding": "roundTiesToEven",
                "validationArithmetic": {
                    "overall": "pytorchFakeQdq",
                    "fp16BoundaryFields": list(BOUNDARY_FIELDS[:3]),
                    "clippedFactorProduct":
                        "nativeCodeDomainInt8DotFp32ScaleDirectRequant",
                    "clippedSilu": "pytorchFp32SimulationNotBitExactCutlass",
                    "noClipProduct": "modelFloatProductBoundaryQdq",
                    "productDownFeed": "pytorchDequantizedSurrogate",
                },
                "candidates": candidates,
                "weightQdq": {
                    "qmin": -127,
                    "qmax": 127,
                    "zeroPoint": 0,
                    "rounding": "roundTiesToEven",
                    "scale": "float32GroupMaxAbsDiv127",
                    "groups": ["qkvShared", "attentionOut", "ffnUp", "ffnGate", "ffnDown"],
                },
            },
            "layerOrder": layer_order,
            "layers": layers,
            "selection": {
                "metric": "trainingLossPerWeight",
                "baselineLoss": 1.0,
                "weightOnlyLoss": 1.005,
                "candidateLosses": {candidate: 1.01 for candidate in candidates},
                "baselineMetrics": {
                    "trainingLossPerWeight": 1.0,
                    "p0LossPerWeight": 0.5,
                    "valueLossPerWeight": 0.25,
                },
                "weightOnlyMetrics": {
                    "trainingLossPerWeight": 1.005,
                    "p0LossPerWeight": 0.503,
                    "valueLossPerWeight": 0.251,
                    "deltaTrainingLossPerWeight": 0.005,
                    "deltaP0LossPerWeight": 0.003,
                    "deltaValueLossPerWeight": 0.001,
                },
                "candidateMetrics": {
                    candidate: {
                        "trainingLossPerWeight": 1.01,
                        "p0LossPerWeight": 0.506,
                        "valueLossPerWeight": 0.252,
                        "deltaTrainingLossPerWeight": 0.01,
                        "deltaP0LossPerWeight": 0.006,
                        "deltaValueLossPerWeight": 0.002,
                    }
                    for candidate in candidates
                },
                "chosenCandidate": "p99.9",
                "selectedLoss": 1.01,
                "lossDelta": 0.01,
            },
        }
        path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        return document

    def _run_export(self, checkpoint, export_dir, gzip_output):
        command = self._export_command(checkpoint, export_dir, gzip_output)
        completed = subprocess.run(
            command,
            cwd=TRAIN_DIR,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"native exporter failed with code {completed.returncode}:\n"
                f"{completed.stdout}"
            )

    def _save_checkpoint(self, path, config):
        model = Model(config, pos_len=5)
        model.initialize()
        torch.save(
            {
                "config": config,
                "model": model.state_dict(),
                "train_state": {},
            },
            path,
        )

    def test_transformer_descriptors_norm_gamma_and_deterministic_gzip(self):
        # tempfile.TemporaryDirectory can create an unusable ACL under some
        # sandboxed Windows runners, so create an inherited-ACL directory.
        temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            checkpoint = temp_dir / "checkpoint.ckpt"
            export_dir = temp_dir / "export"

            # An explicit checkpoint zero is the same no-clip semantic as a
            # missing field and must not force a legacy export to v105.
            config = self._tiny_config(swiglu_clip=0.0)
            model = Model(config, pos_len=5)
            model.initialize()
            with torch.no_grad():
                model.norm_trunkfinal.gamma.fill_(-0.25)
            torch.save(
                {
                    "config": config,
                    "model": model.state_dict(),
                    "train_state": {},
                },
                checkpoint,
            )

            # The legacy CLI remains the default and still writes an uncompressed
            # .bin file.
            self._run_export(checkpoint, export_dir, gzip_output=False)
            raw = (export_dir / "model.bin").read_bytes()

            trunk_offset = raw.index(b"trunk\n") + len(b"trunk\n")
            trunk_descriptor_count, _ = _read_line(raw, trunk_offset)
            self.assertEqual(int(trunk_descriptor_count), 3)
            self.assertIn(
                b"nested_bottleneck_block\nmodel.blocks.1\n4\n",
                raw,
            )
            self.assertEqual(raw.count(b"transformer_attention_block\n"), 3)
            self.assertEqual(raw.count(b"transformer_ffn_block\n"), 3)
            self.assertIn(b"ACTIVATION_SILU\n", raw)
            self.assertIn(
                b"transformer_attention_block\n"
                b"model.blocks.0.attention\n2\n2\n4\n4\n1\n1\n",
                raw,
            )
            self.assertIn(
                b"transformer_ffn_block\n"
                b"model.blocks.0.ffn\n8\n12\n1\n",
                raw,
            )

            norm_marker = b"model.norm_trunkfinal\n"
            offset = raw.index(norm_marker) + len(norm_marker)
            channels_string, offset = _read_line(raw, offset)
            epsilon_string, offset = _read_line(raw, offset)
            has_gamma_string, offset = _read_line(raw, offset)
            has_beta_string, offset = _read_line(raw, offset)
            channels = int(channels_string)
            self.assertEqual(channels, 8)
            self.assertGreater(float(epsilon_string), 0.0)
            self.assertEqual(int(has_gamma_string), 1)
            self.assertEqual(int(has_beta_string), 1)
            _, offset = _read_binary_floats(raw, offset, channels)
            _, offset = _read_binary_floats(raw, offset, channels)
            exported_gamma, offset = _read_binary_floats(raw, offset, channels)
            _read_binary_floats(raw, offset, channels)
            for value in exported_gamma:
                self.assertAlmostEqual(value, 0.75, places=7)

            self._run_export(checkpoint, export_dir, gzip_output=True)
            compressed_path = export_dir / "model.bin.gz"
            compressed_first = compressed_path.read_bytes()
            self.assertEqual(compressed_first[4:8], b"\x00\x00\x00\x00")
            self.assertEqual(gzip.decompress(compressed_first), raw)

            # A second export from identical inputs has an identical gzip stream,
            # including its header.
            self._run_export(checkpoint, export_dir, gzip_output=True)
            self.assertEqual(compressed_path.read_bytes(), compressed_first)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_int8_v104_flag_rejects_nonproduction_topology(self):
        temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            checkpoint = temp_dir / "checkpoint.ckpt"
            self._save_checkpoint(checkpoint, self._tiny_config())
            completed = subprocess.run(
                self._export_command(
                    checkpoint,
                    temp_dir / "export",
                    gzip_output=False,
                    int8_pt_clip4=True,
                ),
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requires C256/H8 learned-RoPE attention", completed.stdout)
            self.assertFalse((temp_dir / "export" / "model.bin").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cpu_ptq_change_preserves_export_14_as_15_header(self):
        temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            checkpoint = temp_dir / "checkpoint.ckpt"
            self._save_checkpoint(checkpoint, self._tiny_config(version=14))
            completed = subprocess.run(
                self._export_command(
                    checkpoint,
                    temp_dir / "export",
                    gzip_output=False,
                    export_14_as_15=True,
                ),
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            header = (temp_dir / "export" / "model.bin").read_bytes().split(
                b"\n", 3
            )
            self.assertEqual(header[1], b"15")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cpu_ptq_flag_isolated_from_cuda_v104_and_v105(self):
        temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            checkpoint = temp_dir / "checkpoint.ckpt"
            config = self._tiny_config(use_qk_norm=True,swiglu_clip=7.0)
            self._save_checkpoint(checkpoint, config)
            calibration_json = temp_dir / "calibration.json"
            self._write_calibration_json(checkpoint, config, calibration_json)

            cuda_export = subprocess.run(
                self._export_command(
                    checkpoint,
                    temp_dir / "cuda-v105",
                    gzip_output=False,
                    calibration_json=calibration_json,
                ),
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(cuda_export.returncode, 0, cuda_export.stdout)

            cpu_staging_export = subprocess.run(
                self._export_command(
                    checkpoint,
                    temp_dir / "cpu-v105",
                    gzip_output=False,
                    calibration_json=calibration_json,
                    cpu_ptq_base=True,
                ),
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(
                cpu_staging_export.returncode, 0, cpu_staging_export.stdout
            )
            self.assertEqual(
                (temp_dir / "cuda-v105" / "model.bin").read_bytes(),
                (temp_dir / "cpu-v105" / "model.bin").read_bytes(),
            )

            mixed = subprocess.run(
                self._export_command(
                    checkpoint,
                    temp_dir / "mixed",
                    gzip_output=False,
                    int8_pt_clip4=True,
                    cpu_ptq_base=True,
                ),
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(mixed.returncode, 0)
            self.assertIn("mutually exclusive", mixed.stdout)
            self.assertFalse((temp_dir / "mixed" / "model.bin").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_qknorm_clip7_exports_v105_under_python_optimize(self):
        temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            checkpoint = temp_dir / "checkpoint.ckpt"
            self._save_checkpoint(
                checkpoint,
                self._tiny_config(use_qk_norm=True,swiglu_clip=7.0),
            )
            command = [
                sys.executable,
                "-O",
                str(TRAIN_DIR / "export_model_pytorch.py"),
                "-checkpoint",
                str(checkpoint),
                "-export-dir",
                str(temp_dir / "export"),
                "-model-name",
                "native-v105-test",
                "-filename-prefix",
                "model",
                "-pos-len",
                "5",
            ]
            rejected = subprocess.run(
                command,
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn(
                "requires -int8-calibration-json",
                rejected.stdout,
            )
            self.assertFalse((temp_dir / "export" / "model.bin").exists())

            calibration_json = temp_dir / "calibration.json"
            config = self._tiny_config(use_qk_norm=True,swiglu_clip=7.0)
            self._write_calibration_json(checkpoint, config, calibration_json)
            command.extend(("-int8-calibration-json",str(calibration_json)))
            completed = subprocess.run(
                command,
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(completed.returncode,0,completed.stdout)
            raw = (temp_dir / "export" / "model.bin").read_bytes()
            self.assertTrue(raw.startswith(b"native-v105-test\n105\n"))
            self.assertIn(
                b"transformer_attention_block\n"
                b"model.blocks.0.attention\n2\n2\n4\n4\n1\n1\n1\n10.0\n20.0\n",
                raw,
            )
            self.assertIn(b"model.blocks.0.attention.q_norm\n4\n",raw)
            self.assertIn(b"model.blocks.0.attention.k_norm\n4\n",raw)
            self.assertIn(
                b"transformer_ffn_block\n"
                b"model.blocks.0.ffn\n8\n12\n1\n7.0\n30.0\n40.0\n",
                raw,
            )
            self.assertEqual(raw.count(b".attention.q_norm\n"),3)
            self.assertEqual(raw.count(b".attention.k_norm\n"),3)
            self.assertEqual(
                _ffn_clip_and_quant_ranges(raw),
                [(7.0,30.0,40.0),(7.0,31.0,41.0),(7.0,32.0,42.0)],
            )

            # A calibration record binds each field to each layer in recursive
            # native wire order.
            per_layer_dir = temp_dir / "per-layer-export"
            per_layer_json = temp_dir / "per-layer.json"
            per_layer_values = [
                {
                    "attentionInputQuantMaxAbs": 1.0 + index,
                    "attentionOutputQuantMaxAbs": 11.0 + index,
                    "ffnInputQuantMaxAbs": 21.0 + index,
                    "productQuantMaxAbs": 41.0 + index,
                }
                for index in range(3)
            ]
            self._write_calibration_json(
                checkpoint, config, per_layer_json, values=per_layer_values
            )
            per_layer_command = self._export_command(
                checkpoint,
                per_layer_dir,
                gzip_output=False,
                calibration_json=per_layer_json,
            )
            per_layer = subprocess.run(
                per_layer_command,
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(per_layer.returncode,0,per_layer.stdout)
            self.assertEqual(
                _ffn_clip_and_quant_ranges(
                    (per_layer_dir / "model.bin").read_bytes()
                ),
                [(7.0,21.0,41.0),(7.0,22.0,42.0),(7.0,23.0,43.0)],
            )

            deterministic_dir = temp_dir / "deterministic"
            deterministic_command = self._export_command(
                checkpoint,
                deterministic_dir,
                gzip_output=True,
                calibration_json=calibration_json,
            )
            first = subprocess.run(
                deterministic_command,
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            first_bytes = (deterministic_dir / "model.bin.gz").read_bytes()
            second = subprocess.run(
                deterministic_command,
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual((deterministic_dir / "model.bin.gz").read_bytes(), first_bytes)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_v105_calibration_fail_closed_and_preserves_no_clip(self):
        temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            checkpoint = temp_dir / "checkpoint.ckpt"
            # Explicit zero and a missing field are both checkpoint-native
            # spellings of no SwiGLU factor clipping.
            config = self._tiny_config(use_qk_norm=True,swiglu_clip=0.0)
            self._save_checkpoint(checkpoint, config)
            good_json = temp_dir / "good.json"
            good_document = self._write_calibration_json(
                checkpoint, config, good_json
            )

            good = subprocess.run(
                self._export_command(
                    checkpoint,
                    temp_dir / "good-export",
                    gzip_output=False,
                    calibration_json=good_json,
                ),
                cwd=TRAIN_DIR,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(good.returncode, 0, good.stdout)
            raw = (temp_dir / "good-export" / "model.bin").read_bytes()
            self.assertEqual(
                _ffn_clip_and_quant_ranges(raw),
                [(0.0,30.0,40.0),(0.0,31.0,41.0),(0.0,32.0,42.0)],
            )

            invalid_documents = []

            wrong_sha = json.loads(json.dumps(good_document))
            wrong_sha["source"]["checkpoint"]["sha256"] = "f" * 64
            invalid_documents.append((wrong_sha, "checkpoint SHA256 mismatch"))

            wrong_bytes = json.loads(json.dumps(good_document))
            wrong_bytes["source"]["checkpoint"]["bytes"] += 1
            invalid_documents.append((wrong_bytes, "checkpoint byte size mismatch"))

            missing_field = json.loads(json.dumps(good_document))
            del missing_field["layers"][1]["ffnInputQuantMaxAbs"]
            invalid_documents.append((missing_field, "keys mismatch"))

            wrong_order = json.loads(json.dumps(good_document))
            wrong_order["layerOrder"][0], wrong_order["layerOrder"][1] = (
                wrong_order["layerOrder"][1], wrong_order["layerOrder"][0]
            )
            invalid_documents.append((wrong_order, "layerOrder"))

            clip_override = json.loads(json.dumps(good_document))
            clip_override["layers"][0]["swigluClip"] = 4.0
            invalid_documents.append((clip_override, "SwiGLU clip is read only"))

            row_overlap = json.loads(json.dumps(good_document))
            row_overlap["source"]["processedRows"]["overlapRows"] = 1
            invalid_documents.append((row_overlap, "processed row overlap"))

            wrong_winner = json.loads(json.dumps(good_document))
            wrong_winner["selection"]["chosenCandidate"] = "p99.999"
            invalid_documents.append((wrong_winner, "minimum validation-loss"))

            for index, (document, expected_error) in enumerate(invalid_documents):
                path = temp_dir / f"invalid-{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                rejected = subprocess.run(
                    self._export_command(
                        checkpoint,
                        temp_dir / f"invalid-export-{index}",
                        gzip_output=False,
                        calibration_json=path,
                    ),
                    cwd=TRAIN_DIR,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(expected_error, rejected.stdout)
                self.assertFalse(
                    (temp_dir / f"invalid-export-{index}" / "model.bin").exists()
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
