#!/usr/bin/env python3
"""Convert canonical native CPU-PTQ base models to S7/S8 models.

The two supported format pairs intentionally share one projection encoding:

* v105 -> v106 for source checkpoint version 102
* v205 -> v206 for source checkpoint version 11

Every non-projection byte is preserved.  Each Transformer Q/K/V/O and SwiGLU
up/gate/down ``@BIN@`` FP32 block is replaced by an ``@S7P@`` or ``@S8P@``
block containing one little-endian FP32 scale per output followed by canonical
output-major signed codes.  The marker is the wire-level qmax field: S7 means
63 and S8 means 127.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np


FP32_MARKER = b"@BIN@"
S7_MARKER = b"@S7P@"
S8_MARKER = b"@S8P@"


@dataclass(frozen=True)
class FormatSpec:
    base_version: int
    quantized_version: int
    source_checkpoint_version: int
    spatial_inputs: int
    global_inputs: int
    profiles: dict[tuple[int, int, int, int], str]


FORMAT_BY_BASE_VERSION = {
    105: FormatSpec(
        105,
        106,
        102,
        22,
        39,
        {
            (11, 96, 3, 256): "b11c96h3-f256",
            (16, 128, 4, 384): "b16c128h4-f384",
            (24, 192, 6, 512): "b24c192h6-f512",
        },
    ),
    205: FormatSpec(
        205,
        206,
        11,
        22,
        19,
        {
            (11, 96, 3, 256): "b11c96h3-f256",
            (16, 128, 4, 384): "b16c128h4-f384",
        },
    ),
}
FORMAT_BY_QUANTIZED_VERSION = {
    spec.quantized_version: spec for spec in FORMAT_BY_BASE_VERSION.values()
}


PROJECTION_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "ffn_linear1",
    "ffn_linear_gate",
    "ffn_linear2",
)


_HEADER = re.compile(
    rb"\A([^\s]+)(\s+)([0-9]+)(\s+)([0-9]+)(\s+)([0-9]+)(\s+)"
)
_PROJECTION = re.compile(
    rb"(?m)^(model\.blocks\.([0-9]+)\."
    rb"(?:attention\.(q_proj|k_proj|v_proj|out_proj)|"
    rb"ffn\.(ffn_linear1|ffn_linear_gate|ffn_linear2)))"
    rb"\r?\n([0-9]+)\r?\n([0-9]+)\r?\n(@BIN@|@S7P@|@S8P@)"
)
_ATTENTION_HEADER = re.compile(
    rb"(?m)^transformer_attention_block\r?\n"
    rb"model\.blocks\.([0-9]+)\.attention\r?\n"
    rb"([0-9]+)\r?\n([0-9]+)\r?\n([0-9]+)\r?\n([0-9]+)\r?\n"
    rb"([01])\r?\n([01])\r?\n"
)


@dataclass(frozen=True)
class ModelHeader:
    name: str
    version: int
    spatial_inputs: int
    global_inputs: int
    version_span: tuple[int, int]


@dataclass(frozen=True)
class Projection:
    native_name: str
    canonical_name: str
    block: int
    role: str
    inputs: int
    outputs: int
    marker: bytes
    payload_span: tuple[int, int]
    values_input_major: np.ndarray | None
    scales: np.ndarray | None
    codes_output_major: np.ndarray | None
    qmax: int | None


@dataclass(frozen=True)
class AttentionHeader:
    block: int
    heads: int
    kv_heads: int
    query_dim: int
    value_dim: int
    use_rope: bool
    learnable_rope: bool


@dataclass(frozen=True)
class Manifest:
    path: Path
    qmax: int
    recipe: str
    activation_quantizer: str
    activation_scale_factor: float
    matrices: dict[str, tuple[np.ndarray, np.ndarray, str]]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_model(path: Path) -> bytes:
    if path.name.endswith(".bin.gz"):
        with gzip.open(path, "rb") as handle:
            return handle.read()
    if path.suffix == ".bin":
        return path.read_bytes()
    raise ValueError("native model input must end in .bin or .bin.gz")


def parse_header(payload: bytes) -> ModelHeader:
    match = _HEADER.match(payload)
    if match is None:
        raise ValueError("native model header is malformed")
    try:
        name = match.group(1).decode("ascii")
        version = int(match.group(3))
        spatial_inputs = int(match.group(5))
        global_inputs = int(match.group(7))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("native model header is not canonical ASCII") from exc
    return ModelHeader(
        name,
        version,
        spatial_inputs,
        global_inputs,
        match.span(3),
    )


def _projection_role(match: re.Match[bytes]) -> str:
    raw = match.group(3) if match.group(3) is not None else match.group(4)
    return raw.decode("ascii")


def scan_projections(payload: bytes) -> list[Projection]:
    projections: list[Projection] = []
    for match in _PROJECTION.finditer(payload):
        native_name = match.group(1).decode("ascii")
        block = int(match.group(2))
        role = _projection_role(match)
        inputs = int(match.group(5))
        outputs = int(match.group(6))
        marker = match.group(7)
        if inputs <= 0 or outputs <= 0:
            raise ValueError(f"{native_name}: nonpositive projection shape")
        elements = inputs * outputs
        payload_start = match.end(7)
        values: np.ndarray | None = None
        scales: np.ndarray | None = None
        codes: np.ndarray | None = None
        qmax: int | None = None
        if marker == FP32_MARKER:
            payload_end = payload_start + elements * 4
            if payload_end > len(payload):
                raise ValueError(f"{native_name}: truncated FP32 projection")
            values = np.frombuffer(
                payload, dtype="<f4", count=elements, offset=payload_start
            ).reshape(inputs, outputs).copy()
            if not np.isfinite(values).all():
                raise ValueError(f"{native_name}: non-finite FP32 projection")
        else:
            qmax = 63 if marker == S7_MARKER else 127
            scale_end = payload_start + outputs * 4
            payload_end = scale_end + elements
            if payload_end > len(payload):
                raise ValueError(f"{native_name}: truncated quantized projection")
            scales = np.frombuffer(
                payload, dtype="<f4", count=outputs, offset=payload_start
            ).copy()
            codes = np.frombuffer(
                payload, dtype=np.int8, count=elements, offset=scale_end
            ).reshape(outputs, inputs).copy()
            if not np.isfinite(scales).all() or not np.all(scales > 0.0):
                raise ValueError(f"{native_name}: invalid projection scales")
            maximum = int(np.max(np.abs(codes.astype(np.int16))))
            if maximum > qmax:
                raise ValueError(
                    f"{native_name}: code {maximum} exceeds declared qmax {qmax}"
                )
        if payload_end < len(payload) and not bytes(
            (payload[payload_end],)
        ).isspace():
            raise ValueError(f"{native_name}: projection payload has no delimiter")
        projections.append(
            Projection(
                native_name,
                f"blocks.{block}.{role}",
                block,
                role,
                inputs,
                outputs,
                marker,
                (match.start(7), payload_end),
                values,
                scales,
                codes,
                qmax,
            )
        )
    names = [projection.canonical_name for projection in projections]
    if len(names) != len(set(names)):
        raise ValueError("native model contains duplicate Transformer projections")
    return projections


def scan_attention_headers(payload: bytes) -> list[AttentionHeader]:
    headers = [
        AttentionHeader(
            block=int(match.group(1)),
            heads=int(match.group(2)),
            kv_heads=int(match.group(3)),
            query_dim=int(match.group(4)),
            value_dim=int(match.group(5)),
            use_rope=match.group(6) == b"1",
            learnable_rope=match.group(7) == b"1",
        )
        for match in _ATTENTION_HEADER.finditer(payload)
    ]
    blocks = [header.block for header in headers]
    if len(blocks) != len(set(blocks)):
        raise ValueError("native model contains duplicate attention headers")
    return headers


def validate_geometry(
    header: ModelHeader,
    projections: list[Projection],
    attention_headers: list[AttentionHeader],
    spec: FormatSpec,
    *,
    quantized: bool,
) -> str:
    expected_version = (
        spec.quantized_version if quantized else spec.base_version
    )
    if header.version != expected_version:
        raise ValueError(
            f"expected native v{expected_version}, got v{header.version}"
        )
    if (
        header.spatial_inputs != spec.spatial_inputs
        or header.global_inputs != spec.global_inputs
    ):
        raise ValueError(
            f"v{expected_version} requires input ABI "
            f"{spec.spatial_inputs}/{spec.global_inputs}, got "
            f"{header.spatial_inputs}/{header.global_inputs}"
        )
    expected_marker = (S7_MARKER, S8_MARKER) if quantized else (FP32_MARKER,)
    if not projections or any(
        projection.marker not in expected_marker for projection in projections
    ):
        raise ValueError(
            f"v{expected_version} has an unexpected projection storage marker"
        )
    block_ids = sorted({projection.block for projection in projections})
    if block_ids != list(range(len(block_ids))):
        raise ValueError("Transformer block indices must be contiguous from zero")
    by_name = {projection.canonical_name: projection for projection in projections}
    expected_names = {
        f"blocks.{block}.{role}"
        for block in block_ids
        for role in PROJECTION_ROLES
    }
    if set(by_name) != expected_names:
        raise ValueError(
            "projection set mismatch: "
            f"missing={sorted(expected_names - set(by_name))}, "
            f"extra={sorted(set(by_name) - expected_names)}"
        )
    attention_by_block = {
        attention.block: attention for attention in attention_headers
    }
    if set(attention_by_block) != set(block_ids):
        raise ValueError(
            "attention header set does not match projection blocks"
        )
    first = by_name["blocks.0.q_proj"]
    channels = first.inputs
    ffn = by_name["blocks.0.ffn_linear1"].outputs
    heads = attention_by_block[0].heads
    profile_key = (len(block_ids), channels, heads, ffn)
    if profile_key not in spec.profiles:
        raise ValueError(
            f"unsupported v{expected_version} CPU-PTQ profile {profile_key}"
        )
    for block in block_ids:
        attention = attention_by_block[block]
        if (
            attention.heads != heads
            or attention.kv_heads != heads
            or attention.query_dim != 32
            or attention.value_dim != 32
            or channels != heads * 32
            or not attention.use_rope
        ):
            raise ValueError(
                f"block {block}: unsupported attention geometry"
            )
        for role in ("q_proj", "k_proj", "v_proj", "out_proj"):
            projection = by_name[f"blocks.{block}.{role}"]
            if (projection.inputs, projection.outputs) != (channels, channels):
                raise ValueError(f"{projection.native_name}: channel mismatch")
        for role in ("ffn_linear1", "ffn_linear_gate"):
            projection = by_name[f"blocks.{block}.{role}"]
            if (projection.inputs, projection.outputs) != (channels, ffn):
                raise ValueError(f"{projection.native_name}: FFN up mismatch")
        down = by_name[f"blocks.{block}.ffn_linear2"]
        if (down.inputs, down.outputs) != (ffn, channels):
            raise ValueError(f"{down.native_name}: FFN down mismatch")
    if quantized:
        qmaxes = {projection.qmax for projection in projections}
        if len(qmaxes) != 1:
            raise ValueError("one model may not mix S7 and S8 projection blocks")
    return spec.profiles[profile_key]


def load_manifest(path: Path) -> Manifest:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"CPU-PTQ manifest does not exist: {path}")
    matrices: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["format"].item()) != "cpuptq-gptq-v1":
            raise ValueError("unsupported CPU-PTQ manifest format")
        qmax = int(archive["qmax"].item())
        if qmax not in (63, 127):
            raise ValueError(f"manifest qmax must be 63 or 127, got {qmax}")
        activation_quantizer = str(archive["activation_quantizer"].item())
        activation_scale_factor = float(
            archive["activation_scale_factor"].item()
        )
        if activation_quantizer != "row-sym" or activation_scale_factor != 1.0:
            raise ValueError(
                "CPU-PTQ requires row-sym activation quantization with scale factor 1"
            )
        recipe = str(archive["recipe"].item())
        names = [str(item) for item in archive["names"].tolist()]
        if len(names) != len(set(names)):
            raise ValueError("CPU-PTQ manifest contains duplicate names")
        for index, name in enumerate(names):
            codes = np.asarray(archive[f"codes_{index}"])
            scales = np.asarray(archive[f"scales_{index}"], dtype="<f4")
            source_sha256 = str(archive[f"source_sha256_{index}"].item())
            if codes.dtype != np.int8 or codes.ndim != 2:
                raise ValueError(f"{name}: codes must be a rank-2 int8 array")
            if scales.shape != (codes.shape[0],):
                raise ValueError(f"{name}: scale shape does not match outputs")
            if not np.isfinite(scales).all() or not np.all(scales > 0.0):
                raise ValueError(f"{name}: scales must be finite and positive")
            maximum = int(np.max(np.abs(codes.astype(np.int16))))
            if maximum > qmax:
                raise ValueError(f"{name}: code exceeds manifest qmax {qmax}")
            matrices[name] = (
                np.ascontiguousarray(codes),
                np.ascontiguousarray(scales),
                source_sha256,
            )
    return Manifest(
        path,
        qmax,
        recipe,
        activation_quantizer,
        activation_scale_factor,
        matrices,
    )


def source_weight_sha256(projection: Projection) -> str:
    if projection.values_input_major is None:
        raise AssertionError("source projection has no FP32 values")
    output_major = np.ascontiguousarray(
        projection.values_input_major.T, dtype="<f4"
    )
    return sha256_bytes(output_major.tobytes(order="C"))


def _maxabs_quantize(
    projection: Projection, qmax: int
) -> tuple[np.ndarray, np.ndarray]:
    if projection.values_input_major is None:
        raise AssertionError("source projection has no FP32 values")
    values = projection.values_input_major
    maximum = np.max(np.abs(values), axis=0).astype(np.float32)
    scales = np.asarray(maximum / np.float32(qmax), dtype="<f4")
    scales[maximum == np.float32(0.0)] = np.float32(1.0)
    scaled = np.asarray(values / scales.reshape(1, -1), dtype=np.float32)
    input_major_codes = np.clip(np.rint(scaled), -qmax, qmax).astype(np.int8)
    return (
        np.ascontiguousarray(input_major_codes.T),
        np.ascontiguousarray(scales),
    )


def _replace(payload: bytes, replacements: Iterable[tuple[int, int, bytes]]) -> bytes:
    output = bytearray()
    cursor = 0
    for start, end, replacement in sorted(replacements):
        if start < cursor or end < start or end > len(payload):
            raise ValueError("overlapping or invalid native payload replacement")
        output += payload[cursor:start]
        output += replacement
        cursor = end
    output += payload[cursor:]
    return bytes(output)


def write_atomic_gzip(
    destination: Path, payload: bytes, *, compression_level: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=destination.name + ".",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
                compresslevel=compression_level,
            ) as zipped:
                zipped.write(payload)
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_atomic_json(destination: Path, document: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=destination.name + ".",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert(
    source: Path,
    destination: Path,
    *,
    manifest_path: Path | None,
    projection_bits: int | None,
    force: bool,
    compression_level: int,
    write_report: bool,
) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise ValueError(f"source model does not exist: {source}")
    if source == destination:
        raise ValueError("source and destination must differ")
    if not destination.name.endswith(".bin.gz"):
        raise ValueError("destination must end in .bin.gz")
    if destination.exists() and not force:
        raise ValueError(f"refusing to overwrite existing output: {destination}")
    if compression_level < 1 or compression_level > 9:
        raise ValueError("gzip compression level must be in 1..9")
    if projection_bits not in (None, 7, 8):
        raise ValueError("projection_bits must be 7 or 8")

    source_payload = read_model(source)
    header = parse_header(source_payload)
    spec = FORMAT_BY_BASE_VERSION.get(header.version)
    if spec is None:
        raise ValueError(
            f"source must be a CPU-PTQ FP32 base v105 or v205, got v{header.version}"
        )
    projections = scan_projections(source_payload)
    attention_headers = scan_attention_headers(source_payload)
    profile = validate_geometry(
        header, projections, attention_headers, spec, quantized=False
    )

    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    if manifest is not None:
        qmax = manifest.qmax
        if projection_bits is not None and qmax != (63 if projection_bits == 7 else 127):
            raise ValueError(
                f"manifest qmax {qmax} disagrees with --projection-bits {projection_bits}"
            )
        expected = {projection.canonical_name for projection in projections}
        if set(manifest.matrices) != expected:
            raise ValueError(
                "manifest projection set mismatch: "
                f"missing={sorted(expected - set(manifest.matrices))}, "
                f"extra={sorted(set(manifest.matrices) - expected)}"
            )
    else:
        qmax = 63 if projection_bits == 7 else 127

    marker = S7_MARKER if qmax == 63 else S8_MARKER
    replacements: list[tuple[int, int, bytes]] = [
        (*header.version_span, str(spec.quantized_version).encode("ascii"))
    ]
    quantized_bytes = 0
    scale_bytes = 0
    fp32_projection_bytes = 0
    for projection in projections:
        if manifest is None:
            codes, scales = _maxabs_quantize(projection, qmax)
        else:
            codes, scales, recorded_sha256 = manifest.matrices[
                projection.canonical_name
            ]
            if codes.shape != (projection.outputs, projection.inputs):
                raise ValueError(
                    f"{projection.canonical_name}: manifest shape {codes.shape} "
                    f"does not match {(projection.outputs, projection.inputs)}"
                )
            if recorded_sha256 != source_weight_sha256(projection):
                raise ValueError(
                    f"{projection.canonical_name}: manifest belongs to another model"
                )
        replacement = (
            marker
            + np.ascontiguousarray(scales, dtype="<f4").tobytes(order="C")
            + np.ascontiguousarray(codes, dtype=np.int8).tobytes(order="C")
        )
        replacements.append((*projection.payload_span, replacement))
        quantized_bytes += codes.size
        scale_bytes += scales.size * 4
        fp32_projection_bytes += projection.inputs * projection.outputs * 4

    target_payload = _replace(source_payload, replacements)
    target_header = parse_header(target_payload)
    target_projections = scan_projections(target_payload)
    target_attention_headers = scan_attention_headers(target_payload)
    verified_profile = validate_geometry(
        target_header,
        target_projections,
        target_attention_headers,
        spec,
        quantized=True,
    )
    if verified_profile != profile:
        raise AssertionError("round-trip verification selected another profile")
    if {projection.qmax for projection in target_projections} != {qmax}:
        raise AssertionError("round-trip verification changed projection qmax")

    write_atomic_gzip(
        destination, target_payload, compression_level=compression_level
    )
    round_trip = read_model(destination)
    if round_trip != target_payload:
        destination.unlink(missing_ok=True)
        raise RuntimeError("deterministic gzip round-trip verification failed")

    report = {
        "kind": "katago-cpu-ptq-native-export-v2",
        "source": str(source),
        "sourceFileSha256": sha256_file(source),
        "sourceUncompressedSha256": sha256_bytes(source_payload),
        "sourceBaseVersion": spec.base_version,
        "sourceCheckpointVersion": spec.source_checkpoint_version,
        "modelName": header.name,
        "modelVersion": spec.quantized_version,
        "profile": profile,
        "projectionCount": len(projections),
        "projectionBits": 7 if qmax == 63 else 8,
        "qmax": qmax,
        "projectionQuantizer": manifest.recipe if manifest else "maxabs-rne",
        "activationQuantizer": (
            manifest.activation_quantizer if manifest else "row-sym"
        ),
        "manifest": str(manifest.path) if manifest else None,
        "manifestSha256": sha256_file(manifest.path) if manifest else None,
        "projectionFp32Bytes": fp32_projection_bytes,
        "projectionQuantizedBytes": quantized_bytes,
        "projectionScaleBytes": scale_bytes,
        "output": str(destination),
        "outputBytes": destination.stat().st_size,
        "outputSha256": sha256_file(destination),
        "outputUncompressedBytes": len(target_payload),
        "outputUncompressedSha256": sha256_bytes(target_payload),
        "gzipCompressionLevel": compression_level,
    }
    if write_report:
        write_atomic_json(destination.with_name(destination.name + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert native CPU-PTQ base v105/v205 to v106/v206 using the "
            "same canonical S7/S8 projection format"
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--manifest",
        "--gptq-overrides",
        dest="manifest",
        type=Path,
        help="cpuptq-gptq-v1 NPZ; its qmax is authoritative",
    )
    parser.add_argument(
        "--projection-bits",
        type=int,
        choices=(7, 8),
        help="required only for maxabs conversion; verifies manifest qmax if present",
    )
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    report = convert(
        args.source,
        args.destination,
        manifest_path=args.manifest,
        projection_bits=args.projection_bits,
        force=args.force,
        compression_level=args.gzip_level,
        write_report=not args.no_report,
    )
    for key in (
        "profile",
        "modelVersion",
        "projectionCount",
        "projectionBits",
        "projectionQuantizer",
        "outputSha256",
        "output",
    ):
        print(f"{key}={report[key]}")


if __name__ == "__main__":
    main()
