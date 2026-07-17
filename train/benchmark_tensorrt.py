#!/usr/bin/env python3
"""Build/load a TensorRT engine and benchmark it with device-resident tensors.

This utility intentionally supports only static-shape engines. TensorRT is invoked
through the TensorRT 10.x name-based I/O API and ``execute_async_v3``. PyTorch is
used solely for CUDA memory, streams, and timing events; no input or output is
copied during the measured region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class BenchmarkError(RuntimeError):
    """An expected configuration or runtime error with a user-facing message."""


ENGINE_MANIFEST_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 8 << 20
REQUIRED_BUILD_CONFIG_KEYS = (
    "builder_optimization_level",
    "fp16",
    "network_creation_flags",
    "static_shapes_only",
    "workspace_bytes",
    "workspace_gib",
)


def _engine_manifest_path(engine_path: Path) -> Path:
    return engine_path.with_name(engine_path.name + ".build.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path, description: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
    except OSError as exc:
        raise BenchmarkError(f"Could not hash {description} {path}: {exc}") from exc
    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes}


def _bytes_identity(data: bytes) -> Dict[str, Any]:
    return {"sha256": _sha256_bytes(data), "size_bytes": len(data)}


def _requested_build_config(args: argparse.Namespace) -> Dict[str, Any]:
    workspace_bytes = int(args.workspace_gib * (1 << 30))
    return {
        "builder_optimization_level": int(args.builder_optimization_level),
        "fp16": bool(args.fp16),
        "network_creation_flags": [],
        "static_shapes_only": True,
        "workspace_bytes": workspace_bytes,
        "workspace_gib": float(args.workspace_gib),
    }


def _current_build_environment(
    trt: Any, torch: Any, device: Any
) -> Dict[str, Any]:
    return {
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device_name": str(torch.cuda.get_device_name(device)),
        "pytorch_cuda_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        "pytorch_version": str(torch.__version__),
        "tensorrt_version": str(trt.__version__),
    }


def _make_engine_manifest(
    onnx_identity: Mapping[str, Any],
    engine_identity: Mapping[str, Any],
    build_config: Mapping[str, Any],
    build_environment: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": ENGINE_MANIFEST_SCHEMA_VERSION,
        "onnx": dict(onnx_identity),
        "engine": dict(engine_identity),
        "build_config": dict(build_config),
        "build_environment": dict(build_environment),
    }


def _read_engine_manifest(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.is_file():
        return None, f"build manifest is missing: {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"could not read build manifest {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"build manifest root is not an object: {path}"
    return payload, None


def _manifest_mismatch_reasons(
    manifest: Mapping[str, Any],
    engine_identity: Mapping[str, Any],
    current_environment: Mapping[str, Any],
    onnx_identity: Optional[Mapping[str, Any]] = None,
    requested_build_config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return reasons an existing engine is not valid for this invocation."""
    reasons: List[str] = []
    if manifest.get("schema_version") != ENGINE_MANIFEST_SCHEMA_VERSION:
        reasons.append(
            "manifest schema version differs "
            f"(stored={manifest.get('schema_version')!r}, "
            f"required={ENGINE_MANIFEST_SCHEMA_VERSION})"
        )

    stored_engine = manifest.get("engine")
    if not isinstance(stored_engine, dict):
        reasons.append("manifest has no valid engine identity")
    else:
        for key in ("sha256", "size_bytes"):
            if stored_engine.get(key) != engine_identity.get(key):
                reasons.append(
                    f"engine {key} differs "
                    f"(stored={stored_engine.get(key)!r}, "
                    f"current={engine_identity.get(key)!r})"
                )

    stored_environment = manifest.get("build_environment")
    if not isinstance(stored_environment, dict):
        reasons.append("manifest has no valid build environment")
    else:
        # These fields affect plan compatibility. PyTorch fields are recorded
        # for reproducibility but do not influence TensorRT engine generation.
        for key in (
            "tensorrt_version",
            "cuda_device_name",
            "cuda_compute_capability",
        ):
            if stored_environment.get(key) != current_environment.get(key):
                reasons.append(
                    f"build environment {key} differs "
                    f"(stored={stored_environment.get(key)!r}, "
                    f"current={current_environment.get(key)!r})"
                )

    stored_onnx = manifest.get("onnx")
    if not isinstance(stored_onnx, dict):
        reasons.append("manifest has no valid ONNX identity")
    else:
        for key in ("sha256", "size_bytes"):
            if key not in stored_onnx:
                reasons.append(f"manifest ONNX identity is missing {key}")
            elif (
                onnx_identity is not None
                and stored_onnx.get(key) != onnx_identity.get(key)
            ):
                reasons.append(
                    f"ONNX {key} differs "
                    f"(stored={stored_onnx.get(key)!r}, "
                    f"current={onnx_identity.get(key)!r})"
                )

    stored_config = manifest.get("build_config")
    if not isinstance(stored_config, dict):
        reasons.append("manifest has no valid build configuration")
    else:
        for key in REQUIRED_BUILD_CONFIG_KEYS:
            if key not in stored_config:
                reasons.append(f"manifest build configuration is missing {key}")
        if (
            requested_build_config is not None
            and stored_config != dict(requested_build_config)
        ):
            all_keys = sorted(set(stored_config) | set(requested_build_config))
            for key in all_keys:
                if stored_config.get(key) != requested_build_config.get(key):
                    reasons.append(
                        f"build option {key} differs "
                        f"(stored={stored_config.get(key)!r}, "
                        f"requested={requested_build_config.get(key)!r})"
                    )
    return reasons


def _write_json_atomic(
    path: Path, payload: Mapping[str, Any], description: str
) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise BenchmarkError(f"Could not write {description} {path}: {exc}") from exc


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a static ONNX model with TensorRT 10.x at its exported "
            "fixed batch size. The first run builds an engine; later runs load it."
        )
    )
    parser.add_argument(
        "onnx",
        nargs="?",
        type=Path,
        help="Static-shape ONNX file. Required when the engine must be built.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help=(
            "Serialized engine path. Defaults to ONNX_NAME.fp16.plan. An existing "
            "engine is loaded only when its build manifest still matches."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild and overwrite the serialized engine from the ONNX file.",
    )
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=8.0,
        help="TensorRT builder workspace limit in GiB (default: 8).",
    )
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=True,
        help="Allow FP16 builder tactics (default).",
    )
    precision.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Do not enable FP16 builder tactics.",
    )
    parser.add_argument(
        "--builder-optimization-level",
        type=int,
        choices=range(0, 6),
        default=3,
        metavar="0..5",
        help="TensorRT builder optimization level (default: 3).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index after CUDA_VISIBLE_DEVICES filtering (default: 0).",
    )
    parser.add_argument(
        "--expected-batch-size",
        type=int,
        help="Optional guard: fail unless the fixed engine batch equals this value.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Unmeasured warmup launches (default: 20).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Launches measured in each repeat (default: 100).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of measured repeats (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed used to initialize floating-point inputs once (default: 12345).",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "After ordinary TensorRT warmup, capture one execute_async_v3 launch "
            "as a CUDA Graph and time graph replay instead. Disabled by default."
        ),
    )
    parser.add_argument(
        "--verify-onnxruntime",
        action="store_true",
        help=(
            "Before timing, run TensorRT once and compare every output against "
            "ONNX Runtime CPU using the same inputs. Requires an ONNX path and "
            "the onnxruntime Python package. Disabled by default."
        ),
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1.0e-2,
        help="Absolute tolerance for --verify-onnxruntime (default: 1e-2).",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1.0e-2,
        help="Relative tolerance for --verify-onnxruntime (default: 1e-2).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally write engine metadata and results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use TensorRT's verbose logger while parsing/building.",
    )
    return parser.parse_args(argv)


def _load_dependencies() -> Tuple[Any, Any]:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise BenchmarkError(
            "TensorRT's Python module is not installed. Install a TensorRT 10.13 "
            "Python package that matches this machine's CUDA runtime, then rerun "
            "this command."
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise BenchmarkError(
            "PyTorch is not installed. A CUDA-enabled PyTorch build is required "
            "for device buffers and CUDA-event timing."
        ) from exc

    try:
        major_version = int(str(trt.__version__).split(".", maxsplit=1)[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise BenchmarkError("Could not determine the installed TensorRT version.") from exc
    if major_version < 10:
        raise BenchmarkError(
            f"TensorRT {trt.__version__} is installed, but this tool requires the "
            "TensorRT 10.x name-based I/O API."
        )
    return trt, torch


def _resolve_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Path]:
    onnx_path = args.onnx.resolve() if args.onnx is not None else None
    if args.engine is not None:
        engine_path = args.engine.resolve()
    elif onnx_path is not None:
        engine_path = onnx_path.with_suffix(".fp16.plan" if args.fp16 else ".fp32.plan")
    else:
        raise BenchmarkError("Give an ONNX file or an existing --engine path.")

    if onnx_path is not None and engine_path == onnx_path:
        raise BenchmarkError("The ONNX and serialized engine paths must be different.")

    if onnx_path is not None and not onnx_path.is_file():
        raise BenchmarkError(f"ONNX file does not exist: {onnx_path}")
    if (args.rebuild or not engine_path.is_file()) and onnx_path is None:
        raise BenchmarkError(
            "The engine must be built or rebuilt, but no ONNX file was supplied: "
            f"{engine_path}"
        )
    return onnx_path, engine_path


def _validate_positive_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.workspace_gib) or args.workspace_gib <= 0.0:
        raise BenchmarkError("--workspace-gib must be a finite value greater than zero.")
    for name in ("warmup", "iterations", "repeats"):
        value = getattr(args, name)
        if name == "warmup":
            if value < 0:
                raise BenchmarkError("--warmup cannot be negative.")
        elif value <= 0:
            raise BenchmarkError(f"--{name} must be greater than zero.")
    if args.expected_batch_size is not None and args.expected_batch_size <= 0:
        raise BenchmarkError("--expected-batch-size must be greater than zero.")
    if args.cuda_graph and args.warmup == 0:
        raise BenchmarkError(
            "--cuda-graph requires --warmup greater than zero so TensorRT can "
            "complete lazy setup before stream capture."
        )
    for name in ("verify_atol", "verify_rtol"):
        value = getattr(args, name)
        option = "--" + name.replace("_", "-")
        if not math.isfinite(value) or value < 0.0:
            raise BenchmarkError(f"{option} must be a finite, non-negative value.")


def _validate_verification_args(
    args: argparse.Namespace, onnx_path: Optional[Path]
) -> None:
    if not args.verify_onnxruntime:
        return
    if onnx_path is None:
        raise BenchmarkError(
            "--verify-onnxruntime requires the ONNX path, even when loading an "
            "existing --engine plan."
        )
    if not onnx_path.is_file():
        raise BenchmarkError(
            f"--verify-onnxruntime ONNX file does not exist: {onnx_path}"
        )


def _network_static_input_shapes(network: Any) -> List[Tuple[str, Tuple[int, ...]]]:
    shapes: List[Tuple[str, Tuple[int, ...]]] = []
    dynamic: List[str] = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dim) for dim in tensor.shape)
        shapes.append((tensor.name, shape))
        if any(dim < 0 for dim in shape):
            dynamic.append(f"{tensor.name}={shape}")
    if dynamic:
        raise BenchmarkError(
            "Only fixed-shape ONNX inputs are supported; re-export with a fixed "
            f"batch size. Dynamic inputs: {', '.join(dynamic)}"
        )
    return shapes


def _build_serialized_engine(
    trt: Any,
    logger: Any,
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: float,
    fp16: bool,
    optimization_level: int,
) -> Tuple[bytes, Dict[str, Any]]:
    builder = trt.Builder(logger)
    # TensorRT 10.x networks are always explicit batch. Passing the old
    # EXPLICIT_BATCH flag is supported but deprecated and has no effect.
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    try:
        onnx_bytes = onnx_path.read_bytes()
    except OSError as exc:
        raise BenchmarkError(f"Could not read ONNX file {onnx_path}: {exc}") from exc
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        detail = "\n  ".join(errors) if errors else "TensorRT returned no parser detail."
        raise BenchmarkError(f"TensorRT could not parse {onnx_path}:\n  {detail}")

    input_shapes = _network_static_input_shapes(network)
    print("Building TensorRT engine from static ONNX inputs:")
    for name, shape in input_shapes:
        print(f"  {name}: {shape}")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gib * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = optimization_level
    if fp16:
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is None:
            raise BenchmarkError(
                "This TensorRT build does not expose BuilderFlag.FP16; use the "
                "requested TensorRT 10.13 runtime or pass --no-fp16."
            )
        if hasattr(builder, "platform_has_fast_fp16") and not builder.platform_has_fast_fp16:
            print("Warning: TensorRT reports that this platform lacks fast FP16 support.")
        config.set_flag(fp16_flag)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise BenchmarkError("TensorRT failed to build the serialized engine.")
    serialized_bytes = bytes(serialized)

    try:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = engine_path.with_name(
            f".{engine_path.name}.tmp-{os.getpid()}"
        )
        temporary_path.write_bytes(serialized_bytes)
        os.replace(temporary_path, engine_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
        raise BenchmarkError(f"Could not save engine to {engine_path}: {exc}") from exc
    print(f"Saved serialized engine: {engine_path}")
    return serialized_bytes, _bytes_identity(onnx_bytes)


def _load_or_build_engine(
    trt: Any,
    logger: Any,
    args: argparse.Namespace,
    onnx_path: Optional[Path],
    engine_path: Path,
    build_config: Mapping[str, Any],
    build_environment: Mapping[str, Any],
) -> Tuple[Any, Any, bool, Path, Dict[str, Any]]:
    manifest_path = _engine_manifest_path(engine_path)
    built = False
    serialized: Optional[bytes] = None
    manifest: Optional[Dict[str, Any]] = None
    rebuild_reasons: List[str] = []

    if args.rebuild:
        rebuild_reasons.append("--rebuild was requested")
    elif not engine_path.is_file():
        rebuild_reasons.append(f"serialized engine is missing: {engine_path}")
    else:
        print(f"Checking serialized engine provenance: {engine_path}")
        try:
            serialized = engine_path.read_bytes()
        except OSError as exc:
            rebuild_reasons.append(f"could not read serialized engine: {exc}")

        manifest, manifest_error = _read_engine_manifest(manifest_path)
        if manifest_error is not None:
            rebuild_reasons.append(manifest_error)
        elif serialized is not None:
            assert manifest is not None
            onnx_identity = (
                _file_identity(onnx_path, "ONNX model")
                if onnx_path is not None
                else None
            )
            rebuild_reasons.extend(
                _manifest_mismatch_reasons(
                    manifest=manifest,
                    engine_identity=_bytes_identity(serialized),
                    current_environment=build_environment,
                    onnx_identity=onnx_identity,
                    requested_build_config=(
                        build_config if onnx_path is not None else None
                    ),
                )
            )

    if rebuild_reasons:
        if onnx_path is None:
            detail = "\n  ".join(rebuild_reasons)
            raise BenchmarkError(
                "The serialized engine could not be validated and cannot be "
                "rebuilt without its ONNX model:\n  "
                f"{detail}\nSupply the matching ONNX path to rebuild it."
            )
        print("Rebuilding TensorRT engine because:")
        for reason in rebuild_reasons:
            print(f"  - {reason}")
        assert onnx_path is not None
        serialized, built_onnx_identity = _build_serialized_engine(
            trt=trt,
            logger=logger,
            onnx_path=onnx_path,
            engine_path=engine_path,
            workspace_gib=float(build_config["workspace_gib"]),
            fp16=bool(build_config["fp16"]),
            optimization_level=int(
                build_config["builder_optimization_level"]
            ),
        )
        manifest = _make_engine_manifest(
            onnx_identity=built_onnx_identity,
            engine_identity=_bytes_identity(serialized),
            build_config=build_config,
            build_environment=build_environment,
        )
        _write_json_atomic(manifest_path, manifest, "engine build manifest")
        print(f"Saved engine build manifest: {manifest_path}")
        built = True
    else:
        assert serialized is not None
        assert manifest is not None
        print(f"Loading serialized engine: {engine_path}")
        print(f"Validated engine build manifest: {manifest_path}")

    assert serialized is not None
    assert manifest is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise BenchmarkError(
            f"TensorRT could not deserialize {engine_path}. Engine plans depend on "
            "the TensorRT version and GPU; rebuild it on this machine with --rebuild."
        )
    return runtime, engine, built, manifest_path, manifest


def _torch_dtype_for_trt(trt: Any, torch: Any, trt_dtype: Any) -> Any:
    candidates = (
        ("FLOAT", torch.float32),
        ("HALF", torch.float16),
        ("INT8", torch.int8),
        ("INT32", torch.int32),
        ("BOOL", torch.bool),
        ("UINT8", torch.uint8),
        ("INT64", torch.int64),
        ("BF16", torch.bfloat16),
    )
    for trt_name, torch_dtype in candidates:
        enum_value = getattr(trt.DataType, trt_name, None)
        if enum_value is not None and trt_dtype == enum_value:
            return torch_dtype
    raise BenchmarkError(f"Unsupported TensorRT I/O dtype: {trt_dtype}")


def _engine_io_metadata(trt: Any, engine: Any) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    dynamic: List[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            dynamic.append(f"{name}={shape}")
        mode = engine.get_tensor_mode(name)
        location = engine.get_tensor_location(name)
        tensor_format = engine.get_tensor_format(name)
        metadata.append(
            {
                "name": name,
                "shape": shape,
                "dtype_object": engine.get_tensor_dtype(name),
                "dtype": str(engine.get_tensor_dtype(name)),
                "mode_object": mode,
                "mode": "input" if mode == trt.TensorIOMode.INPUT else "output",
                "location_object": location,
                "location": str(location),
                "format_object": tensor_format,
                "format": str(tensor_format),
            }
        )
    if dynamic:
        raise BenchmarkError(
            "Only static engines are supported. Dynamic engine tensors: "
            + ", ".join(dynamic)
        )
    host_tensors = [
        item["name"]
        for item in metadata
        if item["location_object"] != trt.TensorLocation.DEVICE
    ]
    if host_tensors:
        raise BenchmarkError(
            "This benchmark binds every tensor to CUDA memory, but these engine "
            f"I/O tensors are host-resident: {', '.join(host_tensors)}"
        )
    non_linear_tensors = [
        item["name"]
        for item in metadata
        if item["format_object"] != trt.TensorFormat.LINEAR
    ]
    if non_linear_tensors:
        raise BenchmarkError(
            "PyTorch contiguous buffers represent linear TensorRT I/O only. "
            "Rebuild with linear I/O formats; non-linear tensors: "
            + ", ".join(non_linear_tensors)
        )
    return metadata


def _infer_fixed_batch(
    metadata: Sequence[Mapping[str, Any]], trt: Any, expected: Optional[int]
) -> int:
    input_batches = {
        int(item["shape"][0])
        for item in metadata
        if item["mode_object"] == trt.TensorIOMode.INPUT and item["shape"]
    }
    if len(input_batches) != 1:
        raise BenchmarkError(
            "All non-scalar engine inputs must have the same leading batch "
            f"dimension; found {sorted(input_batches)}."
        )
    batch_size = next(iter(input_batches))
    if batch_size <= 0:
        raise BenchmarkError(f"Engine batch size must be positive; found {batch_size}.")
    if expected is not None and batch_size != expected:
        raise BenchmarkError(
            f"Engine batch size is {batch_size}, not --expected-batch-size {expected}."
        )
    return batch_size


def _allocate_and_bind(
    trt: Any,
    torch: Any,
    engine: Any,
    context: Any,
    metadata: Sequence[Mapping[str, Any]],
    device: Any,
    seed: int,
) -> Dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    buffers: Dict[str, Any] = {}
    for item in metadata:
        dtype = _torch_dtype_for_trt(trt, torch, item["dtype_object"])
        tensor = torch.empty(tuple(item["shape"]), dtype=dtype, device=device)
        if item["mode_object"] == trt.TensorIOMode.INPUT:
            if tensor.is_floating_point():
                tensor.uniform_(-1.0, 1.0, generator=generator)
            else:
                tensor.zero_()
        address_set = context.set_tensor_address(item["name"], int(tensor.data_ptr()))
        if address_set is False:
            raise BenchmarkError(f"Could not bind engine tensor {item['name']}.")
        buffers[item["name"]] = tensor
    torch.cuda.synchronize(device)
    return buffers


def _launch(context: Any, stream_handle: int) -> None:
    launched = context.execute_async_v3(stream_handle)
    if launched is False:
        raise BenchmarkError("TensorRT execute_async_v3 returned failure.")


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise BenchmarkError(
            "--verify-onnxruntime requires the 'onnxruntime' Python package. "
            "Install the CPU package in this environment and retry."
        ) from exc

    try:
        providers = ort.get_available_providers()
    except (AttributeError, RuntimeError) as exc:
        raise BenchmarkError(
            "Could not query the installed ONNX Runtime execution providers."
        ) from exc
    if "CPUExecutionProvider" not in providers:
        raise BenchmarkError(
            "--verify-onnxruntime requires ONNX Runtime's CPUExecutionProvider; "
            f"available providers: {providers}."
        )
    return ort


def _tensor_to_numpy(tensor: Any, name: str) -> Any:
    try:
        return tensor.detach().cpu().contiguous().numpy()
    except (RuntimeError, TypeError) as exc:
        raise BenchmarkError(
            f"Could not copy tensor {name!r} to an ONNX Runtime NumPy input/output: "
            f"{exc}"
        ) from exc


def _comparison_metrics(
    actual: Any, reference: Any, atol: float, rtol: float
) -> Dict[str, Any]:
    """Return error statistics using ONNX Runtime as the reference."""
    import numpy as np

    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)
    if actual_array.shape != reference_array.shape:
        raise BenchmarkError(
            "TensorRT and ONNX Runtime output shapes differ: "
            f"{actual_array.shape} versus {reference_array.shape}."
        )

    # Float64 avoids overflowing the error calculation for FP16 tensors. This
    # conversion is validation-only and never occurs in the measured region.
    actual_float = actual_array.astype(np.float64, copy=False)
    reference_float = reference_array.astype(np.float64, copy=False)
    absolute_error = np.abs(actual_float - reference_float)
    if absolute_error.size == 0:
        max_abs = 0.0
        mean_abs = 0.0
        max_rel = 0.0
    else:
        max_abs = float(np.max(absolute_error))
        mean_abs = float(np.mean(absolute_error))
        denominator = np.maximum(
            np.abs(reference_float), np.finfo(np.float64).eps
        )
        max_rel = float(np.max(absolute_error / denominator))

    return {
        "shape": list(actual_array.shape),
        "tensorrt_dtype": str(actual_array.dtype),
        "onnxruntime_dtype": str(reference_array.dtype),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": mean_abs,
        "allclose": bool(
            np.allclose(
                actual_array,
                reference_array,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
        ),
    }


def _verify_with_onnxruntime(
    torch: Any,
    context: Any,
    device: Any,
    metadata: Sequence[Mapping[str, Any]],
    buffers: Mapping[str, Any],
    onnx_path: Path,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """Compare one unmeasured TensorRT launch against ONNX Runtime CPU."""
    ort = _load_onnxruntime()
    try:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise BenchmarkError(
            f"ONNX Runtime could not load {onnx_path}: {exc}"
        ) from exc

    engine_input_names = [
        item["name"] for item in metadata if item["mode"] == "input"
    ]
    engine_output_names = [
        item["name"] for item in metadata if item["mode"] == "output"
    ]
    ort_input_names = [item.name for item in session.get_inputs()]
    ort_output_names = [item.name for item in session.get_outputs()]

    def require_same_names(kind: str, engine_names: List[str], ort_names: List[str]) -> None:
        engine_set = set(engine_names)
        ort_set = set(ort_names)
        if engine_set != ort_set:
            raise BenchmarkError(
                f"TensorRT engine and ONNX Runtime {kind} names differ; "
                f"only in engine: {sorted(engine_set - ort_set)}, "
                f"only in ONNX: {sorted(ort_set - engine_set)}. The engine may "
                "have been built from a different ONNX file."
            )

    require_same_names("input", engine_input_names, ort_input_names)
    require_same_names("output", engine_output_names, ort_output_names)

    # Snapshot the exact already-initialized CUDA inputs. TensorRT and ORT then
    # consume the same values; all copies and both executions are outside timing.
    ort_inputs = {
        name: _tensor_to_numpy(buffers[name], name) for name in ort_input_names
    }
    verification_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(verification_stream):
        _launch(context, int(verification_stream.cuda_stream))
    verification_stream.synchronize()
    trt_outputs = {
        name: _tensor_to_numpy(buffers[name], name) for name in engine_output_names
    }

    try:
        ort_values = session.run(engine_output_names, ort_inputs)
    except Exception as exc:
        raise BenchmarkError(f"ONNX Runtime verification execution failed: {exc}") from exc

    print(
        "ONNX Runtime CPU verification (one TensorRT launch; outside timed region):"
    )
    output_results: Dict[str, Dict[str, Any]] = {}
    failed_outputs: List[str] = []
    for name, ort_value in zip(engine_output_names, ort_values):
        try:
            metrics = _comparison_metrics(trt_outputs[name], ort_value, atol, rtol)
        except BenchmarkError as exc:
            raise BenchmarkError(f"Output {name!r}: {exc}") from exc
        output_results[name] = metrics
        if not metrics["allclose"]:
            failed_outputs.append(name)
        status = "PASS" if metrics["allclose"] else "FAIL"
        print(
            f"  {status:4s} {name}: max_abs={metrics['max_abs']:.6e} "
            f"max_rel={metrics['max_rel']:.6e} "
            f"mean_abs={metrics['mean_abs']:.6e}"
        )

    if failed_outputs:
        raise BenchmarkError(
            "ONNX Runtime verification failed for output(s) "
            f"{', '.join(failed_outputs)} with atol={atol:g}, rtol={rtol:g}. "
            "See the per-output error statistics above."
        )
    print(f"  all outputs pass (atol={atol:g}, rtol={rtol:g})")
    return {
        "provider": "CPUExecutionProvider",
        "atol": atol,
        "rtol": rtol,
        "outputs": output_results,
        "allclose": True,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty sequence.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _capture_cuda_graph(
    torch: Any, context: Any, stream: Any, stream_handle: int
) -> Any:
    """Capture and instantiate one warmed-up TensorRT enqueue."""
    try:
        graph = torch.cuda.CUDAGraph()
        # The static engine's I/O addresses were bound before this function and
        # their owning tensors remain alive through all replays. Warmup has also
        # forced TensorRT's lazy setup to happen before stream capture.
        with torch.cuda.graph(graph, stream=stream):
            _launch(context, stream_handle)
        # Exiting torch.cuda.graph instantiates the executable graph. Keep all
        # capture/instantiation work, plus one first replay, outside timing.
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
    except Exception as exc:
        raise BenchmarkError(
            "CUDA Graph capture/instantiation or first replay failed after "
            "TensorRT warmup. "
            "Graph capture requires a capture-compatible TensorRT engine, static "
            "shapes, and stable device buffer addresses; the benchmark already "
            f"provides the latter two. TensorRT/PyTorch detail: {exc}"
        ) from exc
    return graph


def _benchmark(
    torch: Any,
    context: Any,
    device: Any,
    batch_size: int,
    warmup: int,
    iterations: int,
    repeats: int,
    cuda_graph: bool,
) -> Dict[str, Any]:
    stream = torch.cuda.Stream(device=device)
    stream_handle = int(stream.cuda_stream)

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            _launch(context, stream_handle)
    stream.synchronize()

    captured_graph = None
    if cuda_graph:
        captured_graph = _capture_cuda_graph(
            torch=torch,
            context=context,
            stream=stream,
            stream_handle=stream_handle,
        )
        print(
            "CUDA Graph captured after ordinary warmup; timed launches use replay()."
        )

    repeat_latency_ms: List[float] = []
    for repeat in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(iterations):
                if captured_graph is None:
                    _launch(context, stream_handle)
                else:
                    captured_graph.replay()
            end.record(stream)
        end.synchronize()
        latency_ms = float(start.elapsed_time(end)) / iterations
        repeat_latency_ms.append(latency_ms)
        samples_per_second = batch_size * 1000.0 / latency_ms
        print(
            f"repeat {repeat + 1:02d}/{repeats}: "
            f"{latency_ms:.3f} ms/batch, {samples_per_second:.2f} samples/s"
        )

    aggregate_latency_ms = statistics.fmean(repeat_latency_ms)
    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "launch_mode": "cuda_graph_replay" if cuda_graph else "execute_async_v3",
        "total_measured_iterations": iterations * repeats,
        "repeat_latency_ms": repeat_latency_ms,
        "latency_ms": {
            "mean": aggregate_latency_ms,
            "min": min(repeat_latency_ms),
            "p50": _percentile(repeat_latency_ms, 0.50),
            "p95": _percentile(repeat_latency_ms, 0.95),
            "max": max(repeat_latency_ms),
        },
        "samples_per_second": batch_size * 1000.0 / aggregate_latency_ms,
    }


def _json_metadata(metadata: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "mode": item["mode"],
            "shape": list(item["shape"]),
            "dtype": item["dtype"],
            "location": item["location"],
            "format": item["format"],
        }
        for item in metadata
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json_atomic(path, payload, "JSON result")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_positive_args(args)
        onnx_path, engine_path = _resolve_paths(args)
        manifest_path = _engine_manifest_path(engine_path)
        _validate_verification_args(args, onnx_path)
        if args.json_output is not None:
            json_path = args.json_output.resolve()
            protected_paths = {engine_path, manifest_path}
            if onnx_path is not None:
                protected_paths.add(onnx_path)
            if json_path in protected_paths:
                raise BenchmarkError(
                    "--json-output must not overwrite the ONNX, engine, or "
                    "engine build manifest file."
                )
        trt, torch = _load_dependencies()
        if not torch.cuda.is_available():
            raise BenchmarkError("PyTorch cannot access a CUDA device.")
        if args.device < 0 or args.device >= torch.cuda.device_count():
            raise BenchmarkError(
                f"--device {args.device} is invalid; PyTorch sees "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )

        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        build_config = _requested_build_config(args)
        build_environment = _current_build_environment(trt, torch, device)
        severity = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
        logger = trt.Logger(severity)
        runtime, engine, engine_built, manifest_path, engine_manifest = (
            _load_or_build_engine(
                trt=trt,
                logger=logger,
                args=args,
                onnx_path=onnx_path,
                engine_path=engine_path,
                build_config=build_config,
                build_environment=build_environment,
            )
        )
        # Keep logger/runtime/engine alive until all asynchronous engine work ends.
        _lifetime_guards = (logger, runtime, engine)
        context = engine.create_execution_context()
        if context is None:
            raise BenchmarkError("TensorRT could not create an execution context.")

        metadata = _engine_io_metadata(trt, engine)
        batch_size = _infer_fixed_batch(metadata, trt, args.expected_batch_size)
        print(f"TensorRT: {trt.__version__}")
        print(f"CUDA device: {args.device} ({torch.cuda.get_device_name(device)})")
        print(f"Fixed batch size: {batch_size}")
        print("Engine I/O:")
        for item in metadata:
            print(
                f"  {item['mode']:6s} {item['name']}: shape={item['shape']} "
                f"dtype={item['dtype']} location={item['location']} "
                f"format={item['format']}"
            )

        buffers = _allocate_and_bind(
            trt, torch, engine, context, metadata, device, args.seed
        )
        # Referencing buffers through the last synchronize is intentional.
        assert buffers
        verification_results = None
        if args.verify_onnxruntime:
            assert onnx_path is not None
            verification_results = _verify_with_onnxruntime(
                torch=torch,
                context=context,
                device=device,
                metadata=metadata,
                buffers=buffers,
                onnx_path=onnx_path,
                atol=args.verify_atol,
                rtol=args.verify_rtol,
            )
        results = _benchmark(
            torch=torch,
            context=context,
            device=device,
            batch_size=batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            cuda_graph=args.cuda_graph,
        )

        latency = results["latency_ms"]
        print("Summary (CUDA event time):")
        print(
            "  latency ms/batch: "
            f"mean={latency['mean']:.3f}, min={latency['min']:.3f}, "
            f"p50={latency['p50']:.3f}, p95={latency['p95']:.3f}, "
            f"max={latency['max']:.3f}"
        )
        print(f"  throughput: {results['samples_per_second']:.2f} samples/s")

        if args.json_output is not None:
            payload = {
                "tensorrt_version": trt.__version__,
                "torch_version": torch.__version__,
                "device_index": args.device,
                "device_name": torch.cuda.get_device_name(device),
                "onnx_path": str(onnx_path) if onnx_path is not None else None,
                "onnx_sha256": engine_manifest["onnx"]["sha256"],
                "onnx_size_bytes": engine_manifest["onnx"]["size_bytes"],
                "engine_path": str(engine_path),
                "engine_sha256": engine_manifest["engine"]["sha256"],
                "engine_size_bytes": engine_manifest["engine"]["size_bytes"],
                "engine_manifest_path": str(manifest_path),
                "engine_built": engine_built,
                "build_config": engine_manifest["build_config"],
                "build_environment": engine_manifest["build_environment"],
                "current_environment": build_environment,
                "engine_io": _json_metadata(metadata),
                "results": results,
            }
            if verification_results is not None:
                payload["onnxruntime_verification"] = verification_results
            _write_json(args.json_output.resolve(), payload)
            print(f"Wrote JSON result: {args.json_output.resolve()}")
        return 0
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
