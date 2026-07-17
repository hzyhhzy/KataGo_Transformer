import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(__file__))
import benchmark_tensorrt


class _FakeRuntime:
    def __init__(self, logger):
        self.logger = logger

    def deserialize_cuda_engine(self, serialized):
        return {"serialized": serialized}


class _FakeTRT:
    Runtime = _FakeRuntime


class TensorRTProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.onnx_identity = {"sha256": "a" * 64, "size_bytes": 123}
        self.engine_identity = {"sha256": "b" * 64, "size_bytes": 456}
        self.build_config = {
            "builder_optimization_level": 3,
            "fp16": True,
            "network_creation_flags": [],
            "static_shapes_only": True,
            "workspace_bytes": 8 << 30,
            "workspace_gib": 8.0,
        }
        self.environment = {
            "cuda_compute_capability": [8, 9],
            "cuda_device_name": "Fake RTX 4090",
            "pytorch_cuda_version": "13.0",
            "pytorch_version": "2.12.0",
            "tensorrt_version": "10.13.3",
        }
        self.manifest = benchmark_tensorrt._make_engine_manifest(
            onnx_identity=self.onnx_identity,
            engine_identity=self.engine_identity,
            build_config=self.build_config,
            build_environment=self.environment,
        )

    def test_matching_manifest_has_no_rebuild_reasons(self):
        reasons = benchmark_tensorrt._manifest_mismatch_reasons(
            manifest=self.manifest,
            engine_identity=self.engine_identity,
            current_environment=self.environment,
            onnx_identity=self.onnx_identity,
            requested_build_config=self.build_config,
        )
        self.assertEqual(reasons, [])

    def test_changed_onnx_engine_config_and_environment_are_detected(self):
        changed_engine = dict(self.engine_identity, sha256="c" * 64)
        changed_onnx = dict(self.onnx_identity, sha256="d" * 64)
        changed_config = dict(self.build_config, builder_optimization_level=5)
        changed_environment = dict(self.environment, tensorrt_version="10.14.0")

        reasons = benchmark_tensorrt._manifest_mismatch_reasons(
            manifest=self.manifest,
            engine_identity=changed_engine,
            current_environment=changed_environment,
            onnx_identity=changed_onnx,
            requested_build_config=changed_config,
        )
        detail = "\n".join(reasons)
        self.assertIn("engine sha256 differs", detail)
        self.assertIn("ONNX sha256 differs", detail)
        self.assertIn("builder_optimization_level differs", detail)
        self.assertIn("tensorrt_version differs", detail)

    def test_engine_only_validation_uses_recorded_build_inputs(self):
        # With no ONNX there is nothing to rebuild and command-line builder
        # options are irrelevant. The stored identities/configuration must still
        # be structurally valid, while plan compatibility is validated.
        manifest = dict(self.manifest)
        manifest["onnx"] = {"sha256": "e" * 64, "size_bytes": 999}
        manifest["build_config"] = dict(self.build_config, fp16=False)

        reasons = benchmark_tensorrt._manifest_mismatch_reasons(
            manifest=manifest,
            engine_identity=self.engine_identity,
            current_environment=self.environment,
        )
        self.assertEqual(reasons, [])

    def test_missing_manifest_sections_are_rejected(self):
        reasons = benchmark_tensorrt._manifest_mismatch_reasons(
            manifest={"schema_version": benchmark_tensorrt.ENGINE_MANIFEST_SCHEMA_VERSION},
            engine_identity=self.engine_identity,
            current_environment=self.environment,
            onnx_identity=self.onnx_identity,
            requested_build_config=self.build_config,
        )
        detail = "\n".join(reasons)
        self.assertIn("no valid engine identity", detail)
        self.assertIn("no valid ONNX identity", detail)
        self.assertIn("no valid build configuration", detail)
        self.assertIn("no valid build environment", detail)

    def test_hash_and_atomic_manifest_round_trip(self):
        data = b"deterministic test data\x00\xff"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "model.onnx"
            data_path.write_bytes(data)
            identity = benchmark_tensorrt._file_identity(data_path, "test data")
            self.assertEqual(identity["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(identity["size_bytes"], len(data))

            manifest_path = root / "model.plan.build.json"
            benchmark_tensorrt._write_json_atomic(
                manifest_path, self.manifest, "test manifest"
            )
            loaded, error = benchmark_tensorrt._read_engine_manifest(manifest_path)
            self.assertIsNone(error)
            self.assertEqual(loaded, self.manifest)
            self.assertEqual(json.loads(manifest_path.read_text()), self.manifest)

    def test_requested_build_config_records_effective_workspace_bytes(self):
        args = argparse.Namespace(
            workspace_gib=12.5,
            fp16=False,
            builder_optimization_level=5,
        )
        config = benchmark_tensorrt._requested_build_config(args)
        self.assertEqual(config["workspace_bytes"], int(12.5 * (1 << 30)))
        self.assertEqual(config["workspace_gib"], 12.5)
        self.assertFalse(config["fp16"])
        self.assertEqual(config["builder_optimization_level"], 5)

    def test_mismatched_onnx_automatically_rebuilds_and_rewrites_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            onnx_path = root / "model.onnx"
            engine_path = root / "model.plan"
            manifest_path = benchmark_tensorrt._engine_manifest_path(engine_path)
            onnx_path.write_bytes(b"new ONNX")
            engine_path.write_bytes(b"old engine")
            benchmark_tensorrt._write_json_atomic(
                manifest_path, self.manifest, "test manifest"
            )

            new_engine = b"rebuilt engine"

            def fake_build(**kwargs):
                kwargs["engine_path"].write_bytes(new_engine)
                return (
                    new_engine,
                    benchmark_tensorrt._bytes_identity(onnx_path.read_bytes()),
                )

            args = argparse.Namespace(rebuild=False)
            with mock.patch.object(
                benchmark_tensorrt,
                "_build_serialized_engine",
                side_effect=fake_build,
            ) as build_mock:
                _, engine, built, returned_manifest_path, manifest = (
                    benchmark_tensorrt._load_or_build_engine(
                        trt=_FakeTRT,
                        logger=object(),
                        args=args,
                        onnx_path=onnx_path,
                        engine_path=engine_path,
                        build_config=self.build_config,
                        build_environment=self.environment,
                    )
                )

            build_mock.assert_called_once()
            self.assertTrue(built)
            self.assertEqual(engine["serialized"], new_engine)
            self.assertEqual(returned_manifest_path, manifest_path)
            self.assertEqual(
                manifest["onnx"],
                benchmark_tensorrt._bytes_identity(onnx_path.read_bytes()),
            )
            self.assertEqual(
                manifest["engine"], benchmark_tensorrt._bytes_identity(new_engine)
            )
            persisted, error = benchmark_tensorrt._read_engine_manifest(manifest_path)
            self.assertIsNone(error)
            self.assertEqual(persisted, manifest)

    def test_engine_only_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine_path = Path(temp_dir) / "model.plan"
            engine_path.write_bytes(b"untracked engine")
            with self.assertRaisesRegex(
                benchmark_tensorrt.BenchmarkError,
                "could not be validated.*without its ONNX model",
            ):
                benchmark_tensorrt._load_or_build_engine(
                    trt=_FakeTRT,
                    logger=object(),
                    args=argparse.Namespace(rebuild=False),
                    onnx_path=None,
                    engine_path=engine_path,
                    build_config=self.build_config,
                    build_environment=self.environment,
                )


if __name__ == "__main__":
    unittest.main()
