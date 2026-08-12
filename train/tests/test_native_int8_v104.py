import gzip
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np


TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from native_int8_v104 import (
    ENTRY_COUNT,
    Matrix,
    ROLE_FFN_GATE,
    ROLE_FFN_UP,
    ROLE_QK,
    Target,
    _quantize,
    build_payload,
    upgrade_v102_bytes,
)


class NativeInt8V104Tests(unittest.TestCase):
    def test_quantization_is_ties_even_symmetric_and_output_major(self):
        values = np.asarray(
            [[127.0, 126.5], [-126.5, 0.5]],
            dtype=np.float32,
        )
        target = Target(2, ROLE_FFN_UP, (Matrix("layer", 2, 2, values),))
        scale, _, packed_sha, packed = _quantize(target)
        self.assertEqual(scale.tobytes(), np.float32(1.0).tobytes())
        self.assertEqual(
            np.frombuffer(packed, dtype=np.int8).tolist(),
            [127, -126, 126, 0],
        )
        self.assertNotIn(b"\x80", packed)
        self.assertEqual(packed_sha.hex(), hashlib.sha256(packed).hexdigest())

    def test_payload_is_deterministic_and_requires_72_entries(self):
        targets = []
        values = np.asarray([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        for layer in range(24):
            q = Matrix(f"block{layer}.q", 2, 1, values[:, :1].copy())
            k = Matrix(f"block{layer}.k", 2, 1, values[:, 1:].copy())
            up = Matrix(f"block{layer}.up", 2, 2, values.copy())
            gate = Matrix(f"block{layer}.gate", 2, 2, -values.copy())
            topology = 2 + 2 * layer
            targets.extend((
                Target(topology, ROLE_QK, (q, k)),
                Target(topology + 1, ROLE_FFN_UP, (up,)),
                Target(topology + 1, ROLE_FFN_GATE, (gate,)),
            ))

        payload1, entries1 = build_payload(targets)
        payload2, entries2 = build_payload(targets)
        self.assertEqual(len(entries1), ENTRY_COUNT)
        self.assertEqual(payload1, payload2)
        self.assertEqual(entries1, entries2)
        self.assertEqual(
            hashlib.sha256(payload1).hexdigest(),
            "147dcf6d255d76cb66945812e0c5c2d5c81899c4fd7a6edb842a9d535ec8ead9",
        )
        with self.assertRaisesRegex(ValueError, "exactly 72"):
            build_payload(targets[:-1])

    @unittest.skipUnless(
        os.environ.get("KATAGO_V102_GOLD_MODEL"),
        "set KATAGO_V102_GOLD_MODEL to the reviewed native v102 .bin.gz fixture",
    )
    def test_reviewed_model_matches_v104_byte_golds(self):
        source_artifact = Path(os.environ["KATAGO_V102_GOLD_MODEL"]).read_bytes()
        source = gzip.decompress(source_artifact)
        upgraded = upgrade_v102_bytes(source)
        self.assertEqual(
            hashlib.sha256(upgraded.data).hexdigest(),
            "e0de7e93e0a40fab1ef3288022c3da857e4bcc898eaa7237dd2f2d1a601756d6",
        )
        self.assertEqual(
            upgraded.payload_sha256,
            "ccdaef6ff8d83ac588c35630419c0eb40bd3e9dba6b27ef34fc1e8c18ef2267c",
        )
        compressed = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as out:
            out.write(upgraded.data)
        self.assertEqual(
            hashlib.sha256(compressed.getvalue()).hexdigest(),
            "4d82a780523924f2143434a66305b8a696536fc310a3f36b1826c08edd27ef0c",
        )

    @unittest.skipUnless(
        os.environ.get("KATAGO_V104_GOLD_CHECKPOINT"),
        "set KATAGO_V104_GOLD_CHECKPOINT to the reviewed training checkpoint",
    )
    def test_training_exporter_matches_v104_byte_golds(self):
        temp_dir = TRAIN_DIR / "tests" / ("v104_gold_" + uuid.uuid4().hex)
        temp_dir.mkdir()
        try:
            base_command = [
                sys.executable,
                str(TRAIN_DIR / "export_model_pytorch.py"),
                "-checkpoint", os.environ["KATAGO_V104_GOLD_CHECKPOINT"],
                "-export-dir", str(temp_dir),
                "-model-name", "b24c256h8tflrs-renju15-swa",
                "-filename-prefix", "reviewed",
                "-use-swa",
                "-pos-len", "15",
                "-int8-pt-clip4",
            ]
            subprocess.run(base_command, cwd=TRAIN_DIR, check=True)
            raw = (temp_dir / "reviewed.bin").read_bytes()
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(),
                "e0de7e93e0a40fab1ef3288022c3da857e4bcc898eaa7237dd2f2d1a601756d6",
            )

            subprocess.run(base_command + ["-gzip"], cwd=TRAIN_DIR, check=True)
            compressed = (temp_dir / "reviewed.bin.gz").read_bytes()
            self.assertEqual(gzip.decompress(compressed), raw)
            self.assertEqual(
                hashlib.sha256(compressed).hexdigest(),
                "4d82a780523924f2143434a66305b8a696536fc310a3f36b1826c08edd27ef0c",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
