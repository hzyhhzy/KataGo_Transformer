import gzip
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

    def _export_command(self, checkpoint, export_dir, gzip_output, int8_pt_clip4=False):
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
        return command

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

            config = self._tiny_config()
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

    def test_unsupported_transformer_semantics_fail_closed_under_python_optimize(self):
        for suffix, config_update, expected_message in (
            ("qkn", {"use_qk_norm": True}, "QK norm is not supported"),
            ("clip", {"swiglu_clip": 7.0}, "swiglu_clip is not supported"),
        ):
            with self.subTest(suffix=suffix):
                temp_dir = TRAIN_DIR / "tests" / ("native_export_" + uuid.uuid4().hex)
                temp_dir.mkdir()
                try:
                    checkpoint = temp_dir / "checkpoint.ckpt"
                    self._save_checkpoint(checkpoint, self._tiny_config(**config_update))
                    command = [
                        sys.executable,
                        "-O",
                        str(TRAIN_DIR / "export_model_pytorch.py"),
                        "-checkpoint",
                        str(checkpoint),
                        "-export-dir",
                        str(temp_dir / "export"),
                        "-model-name",
                        "unsupported-test",
                        "-filename-prefix",
                        "model",
                        "-pos-len",
                        "5",
                    ]
                    completed = subprocess.run(
                        command,
                        cwd=TRAIN_DIR,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(expected_message, completed.stdout)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
