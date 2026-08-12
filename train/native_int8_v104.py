"""Build the explicit clip4/per-tensor INT8 trailer for native model v104.

The native v102 body, including every FP32 master weight, is preserved byte for
byte except for the fixed-width version token.  This module is intentionally
strict: only the reviewed 24-layer C256/H8/F768 topology can be upgraded, and
all 72 packed matrices are derived deterministically from the FP32 masters in
the just-exported body.
"""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from typing import BinaryIO, Optional

import numpy as np


TRAILER_MARKER = b"@KATAGO_QUANT_TRAILER@"
BINARY_MARKER = b"@BIN@"
HEADER_SCHEMA = 1
PAYLOAD_MAGIC = b"KQPT104\0"
PAYLOAD_SCHEMA = 1
ENTRY_SCHEMA = 1
ENTRY_COUNT = 72
ROLE_QK = 1
ROLE_FFN_UP = 2
ROLE_FFN_GATE = 3
LAYOUT_OUTPUT_MAJOR_K_CONTIGUOUS = 1
QUANT_SYMMETRIC_SIGNED_INT8_PER_TENSOR = 1
ROUND_TIES_TO_EVEN_SATURATE_127 = 1
CLIP = np.float32(4.0)
ACTIVATION_SCALE = np.float32(CLIP / np.float32(127.0))


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _i32(value: int) -> bytes:
    return struct.pack("<i", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _f32_bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _name(value: str) -> bytes:
    raw = value.encode("utf-8")
    if not raw or len(raw) > 4096 or b"\0" in raw:
        raise ValueError(f"invalid native layer name {value!r}")
    return _u32(len(raw)) + raw


@dataclass(frozen=True)
class Matrix:
    name: str
    k: int
    n: int
    values: Optional[np.ndarray]


@dataclass(frozen=True)
class Target:
    topology_index: int
    role: int
    matrices: tuple[Matrix, ...]


@dataclass(frozen=True)
class V104Upgrade:
    data: bytes
    payload: bytes
    payload_sha256: str
    entries: tuple[dict[str, object], ...]


class NativeV102Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.stream: BinaryIO = io.BytesIO(data)
        self.version_span = (-1, -1)
        self.targets: list[Target] = []

    def token_with_span(self) -> tuple[str, int, int]:
        first = self.stream.read(1)
        while first and first.isspace():
            first = self.stream.read(1)
        if not first:
            raise EOFError("unexpected EOF while reading native model token")
        start = self.stream.tell() - 1
        raw = bytearray(first)
        byte = self.stream.read(1)
        while byte and not byte.isspace():
            raw.extend(byte)
            byte = self.stream.read(1)
        end = start + len(raw)
        try:
            return raw.decode("ascii"), start, end
        except UnicodeDecodeError as exc:
            raise ValueError("non-ASCII native model token") from exc

    def token(self) -> str:
        return self.token_with_span()[0]

    def integer(self) -> int:
        return int(self.token())

    def floating(self) -> float:
        value = float(self.token())
        if not np.isfinite(value):
            raise ValueError("non-finite scalar in native model")
        return value

    def weights(self, count: int, retain: bool) -> Optional[np.ndarray]:
        if count <= 0:
            raise ValueError("nonpositive native weight count")
        byte = self.stream.read(1)
        whitespace = 0
        while byte != b"@":
            if not byte or not byte.isspace() or whitespace >= 100:
                raise ValueError("missing @BIN@ native float marker")
            whitespace += 1
            byte = self.stream.read(1)
        if self.stream.read(4) != b"BIN@":
            raise ValueError("invalid @BIN@ native float marker")
        raw = self.stream.read(count * 4)
        if len(raw) != count * 4:
            raise EOFError("truncated native float block")
        values = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("non-finite native FP32 master")
        return values.copy() if retain else None

    def conv(self) -> None:
        self.token()
        y, x, k, n, dy, dx = [self.integer() for _ in range(6)]
        if min(y, x, k, n, dy, dx) <= 0 or y % 2 == 0 or x % 2 == 0:
            raise ValueError("invalid native convolution descriptor")
        self.weights(y * x * k * n, False)

    def matmul(self, retain: bool = False) -> Matrix:
        name = self.token()
        k, n = self.integer(), self.integer()
        if k <= 0 or n <= 0:
            raise ValueError(f"{name}: invalid native matmul descriptor")
        values = self.weights(k * n, retain)
        if values is not None:
            values = values.reshape(k, n)
        return Matrix(name, k, n, values)

    def matbias(self) -> None:
        self.token()
        channels = self.integer()
        self.weights(channels, False)

    def batch_norm(self) -> None:
        self.token()
        channels = self.integer()
        epsilon = self.floating()
        has_scale, has_bias = self.integer(), self.integer()
        if channels <= 0 or epsilon <= 0 or has_scale not in (0, 1) or has_bias not in (0, 1):
            raise ValueError("invalid native batch-normalization descriptor")
        self.weights(channels, False)
        self.weights(channels, False)
        if has_scale:
            self.weights(channels, False)
        if has_bias:
            self.weights(channels, False)

    def activation(self) -> None:
        self.token()
        kind = self.token()
        if kind not in {
            "ACTIVATION_IDENTITY", "ACTIVATION_RELU",
            "ACTIVATION_MISH", "ACTIVATION_SILU",
        }:
            raise ValueError(f"unsupported native activation {kind!r}")

    def transformer_norm(self) -> int:
        self.token()
        channels = self.integer()
        epsilon = self.floating()
        if channels <= 0 or epsilon <= 0:
            raise ValueError("invalid transformer RMSNorm descriptor")
        self.weights(channels, False)
        return channels

    def attention(self, topology_index: int) -> None:
        self.token()
        heads, kv_heads, q_dim, v_dim = [self.integer() for _ in range(4)]
        use_rope, learned_rope = self.integer(), self.integer()
        norm_channels = self.transformer_norm()
        q = self.matmul(True)
        k = self.matmul(True)
        v = self.matmul(False)
        out = self.matmul(False)
        if (
            heads != 8 or kv_heads != 8 or q_dim != 32 or v_dim != 32
            or use_rope != 1 or learned_rope != 1 or norm_channels != 256
            or (q.k, q.n) != (256, 256) or (k.k, k.n) != (256, 256)
            or (v.k, v.n) != (256, 256) or (out.k, out.n) != (256, 256)
        ):
            raise ValueError("v104 explicit INT8 exporter requires C256/H8 learned-RoPE attention")
        rope_name = self.token()
        rope_heads, rope_pairs, coordinates = [self.integer() for _ in range(3)]
        if (rope_heads, rope_pairs, coordinates) != (8, 16, 2) or not rope_name:
            raise ValueError("invalid learned-RoPE tensor for v104")
        self.weights(rope_heads * rope_pairs * coordinates, False)
        self.targets.append(Target(topology_index, ROLE_QK, (q, k)))

    def ffn(self, topology_index: int) -> None:
        self.token()
        channels, ffn_channels, swiglu = [self.integer() for _ in range(3)]
        norm_channels = self.transformer_norm()
        up = self.matmul(True)
        gate = self.matmul(True) if swiglu else None
        down = self.matmul(False)
        if (
            channels != 256 or ffn_channels != 768 or swiglu != 1
            or norm_channels != 256 or (up.k, up.n) != (256, 768)
            or gate is None or (gate.k, gate.n) != (256, 768)
            or (down.k, down.n) != (768, 256)
        ):
            raise ValueError("v104 explicit INT8 exporter requires C256/F768 SwiGLU FFN")
        self.targets.append(Target(topology_index, ROLE_FFN_UP, (up,)))
        self.targets.append(Target(topology_index, ROLE_FFN_GATE, (gate,)))

    def parse(self) -> tuple[bytes, list[Target]]:
        self.token()
        version, start, end = self.token_with_span()
        self.version_span = (start, end)
        if version != "102" or end - start != 3:
            raise ValueError("input must be a native v102 model")
        self.integer()
        self.integer()

        if self.token() != "trunk":
            raise ValueError("unexpected native trunk name")
        block_count = self.integer()
        self.integer()
        self.integer()
        self.integer()
        self.integer()
        self.integer()
        self.conv()
        self.matmul()
        topology_index = 2
        attention_count = 0
        ffn_count = 0
        for _ in range(block_count):
            kind = self.token()
            if kind == "transformer_attention_block":
                self.attention(topology_index)
                attention_count += 1
            elif kind == "transformer_ffn_block":
                self.ffn(topology_index)
                ffn_count += 1
            else:
                raise ValueError(f"v104 exporter rejects unsupported trunk block {kind!r}")
            topology_index += 1
        self.batch_norm()
        self.activation()

        self.token()
        self.conv()
        self.conv()
        self.batch_norm()
        self.activation()
        self.matmul()
        self.batch_norm()
        self.activation()
        self.conv()
        self.matmul()

        self.token()
        self.conv()
        self.batch_norm()
        self.activation()
        self.matmul()
        self.matbias()
        self.activation()
        self.matmul()
        self.matbias()
        self.matmul()
        self.matbias()
        self.conv()

        tail = self.stream.read()
        if not tail or tail.strip():
            raise ValueError("native v102 body must end in whitespace and strict EOF")
        if attention_count != 24 or ffn_count != 24 or len(self.targets) != ENTRY_COUNT:
            raise ValueError("v104 explicit INT8 requires 24 attention + 24 FFN blocks and 72 entries")
        upgraded = bytearray(self.data)
        upgraded[start:end] = b"104"
        return bytes(upgraded), self.targets


def _canonical_master(target: Target) -> np.ndarray:
    if any(matrix.values is None for matrix in target.matrices):
        raise AssertionError("retained target matrix has no values")
    if len(target.matrices) == 1:
        return np.asarray(target.matrices[0].values, dtype="<f4", order="C")
    if len(target.matrices) == 2 and target.matrices[0].k == target.matrices[1].k:
        return np.ascontiguousarray(
            np.concatenate((target.matrices[0].values, target.matrices[1].values), axis=1),
            dtype="<f4",
        )
    raise ValueError("invalid QK canonical concatenation")


def _quantize(target: Target) -> tuple[np.float32, bytes, bytes, bytes]:
    master = _canonical_master(target)
    max_abs = np.max(np.abs(master), initial=np.float32(0.0)).astype(np.float32)
    scale = np.float32(max_abs / np.float32(127.0))
    scale = np.maximum(scale, np.finfo(np.float32).tiny).astype(np.float32)
    if not np.isfinite(scale) or not scale > 0:
        raise ValueError("invalid per-tensor INT8 weight scale")
    scaled = np.asarray(master / scale, dtype=np.float32)
    quantized = np.rint(scaled)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    packed = np.ascontiguousarray(quantized.T).tobytes(order="C")
    if b"\x80" in packed:
        raise AssertionError("symmetric signed-int8 contract emitted -128")
    master_bytes = np.ascontiguousarray(master, dtype="<f4").tobytes(order="C")
    return scale, hashlib.sha256(master_bytes).digest(), hashlib.sha256(packed).digest(), packed


def build_payload(targets: list[Target]) -> tuple[bytes, list[dict[str, object]]]:
    if len(targets) != ENTRY_COUNT:
        raise ValueError("v104 payload requires exactly 72 targets")
    payload = bytearray(PAYLOAD_MAGIC)
    payload += _u32(PAYLOAD_SCHEMA)
    payload += _u32(len(targets))
    payload += _u32(QUANT_SYMMETRIC_SIGNED_INT8_PER_TENSOR)
    payload += _u32(QUANT_SYMMETRIC_SIGNED_INT8_PER_TENSOR)
    payload += _u32(ROUND_TIES_TO_EVEN_SATURATE_127)
    payload += _i32(0)
    payload += _u32(_f32_bits(CLIP))
    payload += _u32(_f32_bits(ACTIVATION_SCALE))
    payload += _u32(0) + _u32(0)
    manifest_entries: list[dict[str, object]] = []

    for target in targets:
        master = _canonical_master(target)
        k, n = master.shape
        scale, master_sha, packed_sha, packed = _quantize(target)
        names = [matrix.name for matrix in target.matrices]
        record = bytearray()
        record += _u32(ENTRY_SCHEMA)
        record += _u32(target.topology_index)
        record += _u32(target.role)
        record += _u32(LAYOUT_OUTPUT_MAJOR_K_CONTIGUOUS)
        record += _u32(k) + _u32(n)
        record += _i32(0)
        record += _u32(_f32_bits(ACTIVATION_SCALE))
        record += _u32(_f32_bits(scale))
        record += _u32(len(names))
        record += _u64(len(packed))
        for name in names:
            record += _name(name)
        record += master_sha + packed_sha + packed
        payload += _u64(len(record)) + record
        manifest_entries.append({
            "topology_index": target.topology_index,
            "role": {ROLE_QK: "qk", ROLE_FFN_UP: "ffn_up", ROLE_FFN_GATE: "ffn_gate"}[target.role],
            "layer_names": names,
            "k": k,
            "n": n,
            "layout": "output-major-k-contiguous",
            "zero_point": 0,
            "activation_clip": float(CLIP),
            "activation_scale_bits": f"0x{_f32_bits(ACTIVATION_SCALE):08x}",
            "weight_scale_bits": f"0x{_f32_bits(scale):08x}",
            "master_sha256": master_sha.hex(),
            "packed_sha256": packed_sha.hex(),
            "packed_bytes": len(packed),
        })
    return bytes(payload), manifest_entries


def upgrade_v102_bytes(data: bytes) -> V104Upgrade:
    """Return a complete deterministic native v104 artifact (uncompressed)."""
    body_v104, targets = NativeV102Reader(data).parse()
    payload, entries = build_payload(targets)
    payload_sha = hashlib.sha256(payload).hexdigest()
    header = (
        TRAILER_MARKER + b" " + str(HEADER_SCHEMA).encode("ascii") + b" "
        + str(len(payload)).encode("ascii") + b" " + payload_sha.encode("ascii")
        + b" " + BINARY_MARKER
    )
    return V104Upgrade(
        data=body_v104 + header + payload,
        payload=payload,
        payload_sha256=payload_sha,
        entries=tuple(entries),
    )
