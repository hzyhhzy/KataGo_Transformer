"""Shared contracts for native CUDA INT8 calibration (v105/v205).

The calibration JSON deliberately does not contain SwiGLU clipping. Clipping
is a trained model semantic and is always read from the checkpoint by the
exporter. This module only owns the four symmetric activation ranges consumed
by the native INT8 implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model_pytorch import (
    NestedBottleneckResBlock,
    NestedBottleneckTransformerBlock,
    TransformerRoPEGQABlock,
)


SCHEMA_NAME = "katago.native-int8-calibration"
SCHEMA_VERSION = 1
CUDA_INT8_WIRE_VERSION_BY_FLOAT_VERSION = {
    102: 105,
    11: 205,
}
# Backward-compatible name for callers and fixtures targeting the v102 family.
WIRE_VERSION = CUDA_INT8_WIRE_VERSION_BY_FLOAT_VERSION[102]
QMIN = -127
QMAX = 127
BOUNDARY_FIELDS = (
    "attentionInputQuantMaxAbs",
    "attentionOutputQuantMaxAbs",
    "ffnInputQuantMaxAbs",
    "productQuantMaxAbs",
)
WEIGHT_SCALE_FIELDS = (
    "qkvSharedWeightScale",
    "attentionOutWeightScale",
    "ffnUpWeightScale",
    "ffnGateWeightScale",
    "ffnDownWeightScale",
)
LOSS_METRIC_FIELDS = (
    "trainingLossPerWeight",
    "p0LossPerWeight",
    "valueLossPerWeight",
)
LOSS_DELTA_FIELDS = (
    "deltaTrainingLossPerWeight",
    "deltaP0LossPerWeight",
    "deltaValueLossPerWeight",
)
DEFAULT_CANDIDATES = (
    ("p99.9", 99.9),
    ("p99.99", 99.99),
    ("p99.999", 99.999),
    ("minmax", None),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def cuda_int8_wire_version(float_model_version: int) -> int:
    """Map a floating native model version to its CUDA INT8 wire version."""
    try:
        return CUDA_INT8_WIRE_VERSION_BY_FLOAT_VERSION[int(float_model_version)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "CUDA INT8 export supports floating model versions 102 and 11 only, "
            f"got {float_model_version}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_float32(value: float) -> float:
    """Round a Python number to the exact scalar serialized to the C++ parser."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _finite_positive(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = canonical_float32(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be finite and positive")
    return result


def _finite_nonnegative(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return result


def transformer_blocks_in_wire_order(model) -> List[Tuple[str, TransformerRoPEGQABlock]]:
    """Return combined PyTorch transformer blocks in native descriptor order."""
    result: List[Tuple[str, TransformerRoPEGQABlock]] = []

    def visit(name, block):
        if isinstance(block, TransformerRoPEGQABlock):
            result.append((name, block))
            return
        if isinstance(block, (NestedBottleneckResBlock, NestedBottleneckTransformerBlock)):
            for index, subblock in enumerate(block.blockstack):
                visit(f"{name}.blockstack.{index}", subblock)

    for index, block in enumerate(model.blocks):
        visit(f"model.blocks.{index}", block)
    return result


def expand_npz_paths(path: Path) -> List[Path]:
    path = path.resolve()
    if path.is_dir():
        files = sorted(item.resolve() for item in path.iterdir() if item.suffix.lower() == ".npz")
    elif path.is_file() and path.suffix.lower() == ".npz":
        files = [path]
    else:
        raise ValueError(f"NPZ source does not exist or is not an .npz file/directory: {path}")
    if not files:
        raise ValueError(f"No .npz files found in {path}")
    return files


def dataset_source_record(files: Sequence[Path]) -> Dict:
    """Hash dataset contents without binding the artifact to absolute paths."""
    records = []
    manifest = hashlib.sha256(b"katago-native-int8-dataset-v1\0")
    for index, path in enumerate(files):
        file_sha = sha256_file(path)
        size = path.stat().st_size
        records.append({
            "index": index,
            "name": path.name,
            "bytes": size,
            "sha256": file_sha,
        })
        manifest.update(bytes.fromhex(file_sha))
        manifest.update(struct.pack("<Q", size))
    return {"sha256": manifest.hexdigest(), "files": records}


def require_independent_datasets(
    calibration_files: Sequence[Path], validation_files: Sequence[Path]
) -> None:
    calibration_paths = {path.resolve() for path in calibration_files}
    validation_paths = {path.resolve() for path in validation_files}
    overlap = calibration_paths & validation_paths
    if overlap:
        raise ValueError(
            "Calibration and validation data must be independent; overlapping path: "
            + str(sorted(overlap)[0])
        )
    calibration_hashes = {sha256_file(path) for path in calibration_files}
    validation_hashes = {sha256_file(path) for path in validation_files}
    duplicate_content = calibration_hashes & validation_hashes
    if duplicate_content:
        raise ValueError(
            "Calibration and validation data must be independent; identical NPZ content "
            f"has SHA256 {sorted(duplicate_content)[0]}"
        )


class ProcessedRowHashes:
    """Stable identity set for the actual model inputs after NPZ processing."""

    def __init__(self):
        self.rows = 0
        self.digests = set()

    def observe_batch(self, batch: Mapping[str, torch.Tensor], include_metadata: bool) -> None:
        keys = ["binaryInputNCHW", "globalInputNC"]
        if include_metadata:
            keys.append("metadataInputNC")
        arrays = []
        batch_size = None
        for key in keys:
            if key not in batch:
                raise ValueError(f"processed calibration batch is missing {key}")
            array = batch[key].detach().cpu().contiguous().numpy()
            if batch_size is None:
                batch_size = array.shape[0]
            elif array.shape[0] != batch_size:
                raise ValueError("processed calibration batch fields have inconsistent rows")
            arrays.append((key, array))
        if batch_size is None or batch_size <= 0:
            raise ValueError("processed calibration batch is empty")
        for row in range(batch_size):
            digest = hashlib.sha256(b"katago-native-int8-processed-row-v1\0")
            for key, array in arrays:
                row_array = np.ascontiguousarray(array[row])
                digest.update(key.encode("ascii"))
                digest.update(b"\0")
                digest.update(row_array.dtype.str.encode("ascii"))
                digest.update(struct.pack("<I", row_array.ndim))
                for dimension in row_array.shape:
                    digest.update(struct.pack("<Q", dimension))
                digest.update(row_array.tobytes(order="C"))
            self.digests.add(digest.digest())
            self.rows += 1

    def summary(self) -> Dict:
        if self.rows <= 0 or not self.digests:
            raise ValueError("no processed rows were hashed")
        digest = hashlib.sha256(b"katago-native-int8-processed-row-set-v1\0")
        for row_digest in sorted(self.digests):
            digest.update(row_digest)
        return {
            "rows": self.rows,
            "uniqueRows": len(self.digests),
            "setSha256": digest.hexdigest(),
        }


_FINITE_POSITIVE_FP16_BITS = 0x7C00
_FP16_ABS_VALUES = np.arange(
    _FINITE_POSITIVE_FP16_BITS, dtype=np.uint16
).view(np.float16).astype(np.float32)


class ActivationSample:
    """Exact deterministic histogram over every finite nonnegative FP16 value.

    A million-value reservoir barely contains ten observations above P99.999
    and would cost hundreds of MiB across 144 C384 boundaries. FP16 has only
    31,744 finite nonnegative bit patterns, so an exact uint64 histogram is both
    more accurate and much smaller (about 35 MiB for 36x4 boundaries).
    """

    def __init__(self):
        self.histogram = np.zeros((_FINITE_POSITIVE_FP16_BITS,), dtype=np.uint64)
        self.observed_values = 0
        self.observations = 0
        self.maximum = 0.0

    def observe(self, tensor: torch.Tensor) -> None:
        # Native kernels read FP16 at every calibrated boundary. Observe those
        # exact representable values even if PyTorch's surrounding module kept
        # its public input in FP32 under autocast.
        values = tensor.detach().to(dtype=torch.float16).abs().reshape(-1)
        count = values.numel()
        if count <= 0:
            return
        if not bool(torch.all(torch.isfinite(values)).cpu().item()):
            raise ValueError("activation boundary contains NaN or infinity")
        self.observed_values += count
        self.observations += 1
        self.maximum = max(self.maximum, float(values.max().float().cpu().item()))
        # Absolute FP16 values are nonnegative, so their raw bit patterns are
        # monotonically ordered. Bincount on-device and copy only 31,744 bins.
        bits = values.view(torch.int16).to(dtype=torch.int32)
        counts = torch.bincount(bits, minlength=_FINITE_POSITIVE_FP16_BITS)
        self.histogram += counts[:_FINITE_POSITIVE_FP16_BITS].cpu().numpy().astype(
            np.uint64, copy=False
        )

    def _value_at_rank(self, rank: int) -> float:
        if rank < 0 or rank >= self.observed_values:
            raise ValueError(f"activation order statistic rank {rank} is out of range")
        cumulative = np.cumsum(self.histogram, dtype=np.uint64)
        bits = int(np.searchsorted(cumulative, rank + 1, side="left"))
        return float(_FP16_ABS_VALUES[bits])

    def threshold(self, percentile: Optional[float]) -> float:
        if self.observed_values <= 0:
            raise ValueError("no activation values were observed")
        if percentile is None:
            value = self.maximum
        else:
            if not math.isfinite(percentile) or percentile <= 0.0 or percentile >= 100.0:
                raise ValueError(f"invalid activation percentile {percentile}")
            fractional_rank = percentile / 100.0 * (self.observed_values - 1)
            lower_rank = math.floor(fractional_rank)
            upper_rank = math.ceil(fractional_rank)
            lower = self._value_at_rank(lower_rank)
            upper = self._value_at_rank(upper_rank)
            value = lower + (upper - lower) * (fractional_rank - lower_rank)
        return _finite_positive(value, "calibrated quantization threshold")

    def estimated_saturation_rate(self, threshold: float) -> float:
        threshold = _finite_positive(threshold, "quantization threshold")
        saturated = np.sum(
            self.histogram[_FP16_ABS_VALUES > threshold], dtype=np.uint64
        )
        return float(saturated / self.observed_values)

    def summary(self) -> Dict:
        return {
            "observedValues": self.observed_values,
            "sampledValues": self.observed_values,
            "observations": self.observations,
            "maxAbs": _finite_positive(self.maximum, "observed maxAbs"),
        }


class SaturationCounter:
    def __init__(self):
        self.total = 0
        self.saturated = None

    def observe(self, tensor: torch.Tensor, threshold: float) -> None:
        values = tensor.detach().to(dtype=torch.float16).abs()
        self.total += values.numel()
        count = torch.count_nonzero(values > threshold)
        self._accumulate(count)

    def observe_int8_scaled(self, scaled: torch.Tensor) -> None:
        """Count D4 product values outside the signed symmetric INT8 domain."""
        self.total += scaled.numel()
        count = torch.count_nonzero(scaled.detach().abs() > float(QMAX))
        self._accumulate(count)

    def _accumulate(self, count: torch.Tensor) -> None:
        if self.saturated is None:
            self.saturated = count
        else:
            self.saturated.add_(count)

    def rate(self) -> float:
        if self.total <= 0 or self.saturated is None:
            raise ValueError("no validation activations were observed")
        return int(self.saturated.cpu().item()) / self.total


def activation_qdq_scales_float32(max_abs: float) -> Tuple[float, float]:
    """Return the two independently rounded FP32 activation-QDQ scales.

    The native kernels form the quantization multiplier as ``127 / A`` and
    the dequantization scale as ``A / 127``.  The latter is deliberately not
    computed as the reciprocal of the already-rounded multiplier: for some
    ranges that shortcut differs by one FP32 ULP from the wire contract.
    """
    max_abs = _finite_positive(max_abs, "QDQ maxAbs")
    quant_multiplier = canonical_float32(canonical_float32(127.0) / max_abs)
    dequant_scale = canonical_float32(max_abs / canonical_float32(127.0))
    if (
        not math.isfinite(quant_multiplier)
        or quant_multiplier <= 0.0
        or not math.isfinite(dequant_scale)
        or dequant_scale <= 0.0
    ):
        raise ValueError("QDQ maxAbs produces a non-finite or zero FP32 scale")
    return quant_multiplier, dequant_scale


def quantize_symmetric_int8_fp16(
    tensor: torch.Tensor, max_abs: float
) -> torch.Tensor:
    """Quantize an explicit FP16 boundary to symmetric signed INT8."""
    quant_multiplier, _ = activation_qdq_scales_float32(max_abs)
    fp16 = tensor.detach().to(dtype=torch.float16)
    multiplier_tensor = torch.tensor(
        quant_multiplier, dtype=torch.float32, device=tensor.device
    )
    return torch.round(fp16.float() * multiplier_tensor).clamp_(QMIN, QMAX).to(
        dtype=torch.int8
    )


def dequantize_symmetric_int8_fp16(
    quantized: torch.Tensor, max_abs: float, output_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize with independently rounded A/127 and cross FP16."""
    _, dequant_scale = activation_qdq_scales_float32(max_abs)
    dequant_tensor = torch.tensor(
        dequant_scale, dtype=torch.float32, device=quantized.device
    )
    return (quantized.float() * dequant_tensor).to(dtype=torch.float16).to(
        dtype=output_dtype
    )


def qdq_symmetric_int8_fp16(tensor: torch.Tensor, max_abs: float) -> torch.Tensor:
    """Engine-contract QDQ: FP16 boundary, RNE, symmetric [-127,127]."""
    quantized = quantize_symmetric_int8_fp16(tensor, max_abs)
    return dequantize_symmetric_int8_fp16(quantized, max_abs, tensor.dtype)


def swiglu_product_requant_multiplier_float32(
    swiglu_clip: float, product_max_abs: float
) -> float:
    """Match createDualFfn's double construction followed by one FP32 cast."""
    clip = _finite_positive(swiglu_clip, "SwiGLU clip")
    product = _finite_positive(product_max_abs, "product maxAbs")
    multiplier = canonical_float32(
        float(clip) * float(clip) / (127.0 * float(product))
    )
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("SwiGLU product requantization multiplier is invalid")
    return multiplier


def scaled_swiglu_factor_product_float32(
    quantized_up: torch.Tensor,
    quantized_gate: torch.Tensor,
    swiglu_clip: float,
    product_max_abs: float,
) -> torch.Tensor:
    """Return D4's FP32-scaled integer product immediately before RNE."""
    if quantized_up.shape != quantized_gate.shape:
        raise ValueError("SwiGLU quantized factor shapes do not match")
    multiplier = swiglu_product_requant_multiplier_float32(
        swiglu_clip, product_max_abs
    )
    integer_product = quantized_up.to(torch.int32) * quantized_gate.to(torch.int32)
    return integer_product.float() * torch.tensor(
        multiplier, dtype=torch.float32, device=integer_product.device
    )


def requantize_swiglu_factor_product_int8(
    quantized_up: torch.Tensor,
    quantized_gate: torch.Tensor,
    swiglu_clip: float,
    product_max_abs: float,
) -> torch.Tensor:
    """D4 exact factorwise integer product -> calibrated product INT8."""
    scaled = scaled_swiglu_factor_product_float32(
        quantized_up, quantized_gate, swiglu_clip, product_max_abs
    )
    return torch.round(scaled).clamp_(QMIN, QMAX).to(dtype=torch.int8)


def quantize_symmetric_int8_fp32(
    tensor: torch.Tensor, max_abs: float
) -> torch.Tensor:
    """Native factor epilogue FP32 clamp/multiply/RNE quantization."""
    quant_multiplier, _ = activation_qdq_scales_float32(max_abs)
    maximum = _finite_positive(max_abs, "FP32 QDQ maxAbs")
    values = tensor.detach().float().clamp(-maximum, maximum)
    multiplier_tensor = torch.tensor(
        quant_multiplier, dtype=torch.float32, device=tensor.device
    )
    scaled = (values * multiplier_tensor).clamp_(QMIN, QMAX)
    return torch.round(scaled).to(dtype=torch.int8)


def native_code_domain_swiglu_factor_int8(
    quantized_input: torch.Tensor,
    quantized_weight: torch.Tensor,
    input_max_abs: float,
    weight_scale: float,
    swiglu_clip: float,
    apply_silu: bool,
) -> torch.Tensor:
    """Model the native clipped-factor epilogue from INT8 codes.

    For the supported transformer widths K is at most 384, so every possible
    signed INT8 dot-product and every intermediate partial sum has magnitude
    below 2**24. A true FP32 matmul therefore represents the native INT32
    accumulator exactly, while remaining efficient on calibration GPUs. The
    scale/clamp/RNE stages follow the native FP32 code-domain contract. SiLU is
    PyTorch's FP32 implementation and is not claimed bit-identical to CUTLASS's
    device exponential approximation.
    """
    if quantized_input.dtype != torch.int8 or quantized_weight.dtype != torch.int8:
        raise ValueError("native-code-domain SwiGLU factor inputs must be INT8 codes")
    if quantized_input.device != quantized_weight.device:
        raise ValueError("native-code-domain SwiGLU factor codes must share a device")
    if quantized_input.ndim < 2 or quantized_weight.ndim != 2:
        raise ValueError("native-code-domain SwiGLU factor tensors have invalid ranks")
    inner = quantized_input.shape[-1]
    if quantized_weight.shape[1] != inner or inner <= 0 or inner > 384:
        raise ValueError("native-code-domain SwiGLU factor inner dimension is unsupported")

    _, input_scale = activation_qdq_scales_float32(input_max_abs)
    weight_scale = _finite_positive(weight_scale, "SwiGLU factor weight scale")
    alpha = canonical_float32(input_scale * weight_scale)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("native-code-domain SwiGLU factor alpha is invalid")

    flattened = quantized_input.reshape(-1, inner)
    # The caller sets FP32 matmul precision to highest/TF32 off. Disabling
    # autocast here prevents the enclosing FP16 model pass from changing the
    # code-domain dot product.
    with torch.amp.autocast(device_type=quantized_input.device.type, enabled=False):
        accumulator = torch.matmul(
            flattened.float(), quantized_weight.float().transpose(0, 1)
        )
        values = accumulator * torch.tensor(
            alpha, dtype=torch.float32, device=accumulator.device
        )
        if apply_silu:
            values = F.silu(values)
        quantized = quantize_symmetric_int8_fp32(values, swiglu_clip)
    return quantized.reshape(*quantized_input.shape[:-1], quantized_weight.shape[0])


def _weight_scale(parameters: Sequence[torch.nn.Parameter]) -> float:
    if not parameters:
        raise ValueError("INT8 weight scale group is empty")
    maximum = 0.0
    for parameter in parameters:
        values = parameter.detach()
        if not bool(torch.all(torch.isfinite(values)).cpu().item()):
            raise ValueError("INT8 weight group contains NaN or infinity")
        maximum = max(maximum, float(values.abs().max().float().cpu().item()))
    # Match C++ commonWeightScale: both operands and the division are FP32.
    maximum = canonical_float32(maximum)
    return canonical_float32(max(maximum / canonical_float32(127.0), np.finfo(np.float32).tiny))


def qdq_symmetric_int8_weight(
    weight: torch.Tensor, scale: float
) -> torch.Tensor:
    """C++ commonWeightScale/quantizeWeight equivalent dequantized tensor."""
    quantized = quantize_symmetric_int8_weight(weight, scale)
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=weight.device)
    return (quantized.float() * scale_tensor).to(dtype=weight.dtype)


def quantize_symmetric_int8_weight(
    weight: torch.Tensor, scale: float
) -> torch.Tensor:
    """Return the exact symmetric INT8 codes used by native weight packing."""
    scale = _finite_positive(scale, "weight scale")
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=weight.device)
    quantized = torch.round(weight.detach().float() / scale_tensor).clamp_(QMIN, QMAX)
    return quantized.to(dtype=torch.int8)


class AggressiveInt8WeightQDQ:
    """Temporarily install aggressive-engine weight QDQ and restore bitwise.

    Q/K/V share one max-abs scale, exactly like C++ packProjection in
    aggressive mode. Attention-out, up, gate, and down each use an independent
    per-matrix symmetric max-abs scale.
    """

    def __init__(self, layers: Sequence[Tuple[str, TransformerRoPEGQABlock]]):
        self.layers = list(layers)
        self.scales: Dict[str, Dict[str, float]] = {}
        self.quantized_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        self._originals: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        self._active = False

    def _save(self, parameter: torch.nn.Parameter) -> None:
        if any(saved is parameter for saved, _ in self._originals):
            raise ValueError("INT8 weight QDQ encountered an aliased projection parameter")
        # Keep the restoration image on CPU so C384 candidate validation does
        # not duplicate hundreds of MiB of persistent GPU allocation.
        self._originals.append((parameter, parameter.detach().cpu().clone()))

    def _quantize_group(
        self, parameters: Sequence[torch.nn.Parameter], scale: Optional[float] = None
    ) -> Tuple[float, List[torch.Tensor]]:
        if scale is None:
            scale = _weight_scale(parameters)
        quantized_codes = []
        for parameter in parameters:
            self._save(parameter)
            quantized = quantize_symmetric_int8_weight(parameter, scale)
            quantized_codes.append(quantized)
            scale_tensor = torch.tensor(
                scale, dtype=torch.float32, device=parameter.device
            )
            parameter.copy_((quantized.float() * scale_tensor).to(parameter.dtype))
        return scale, quantized_codes

    def __enter__(self):
        if self._active:
            raise ValueError("INT8 weight QDQ context cannot be re-entered")
        self._active = True
        try:
            with torch.no_grad():
                for layer_name, block in self.layers:
                    if not block.use_swiglu or not hasattr(block, "ffn_linear_gate"):
                        raise ValueError(f"{layer_name}: aggressive weight QDQ requires SwiGLU")
                    qkv = [block.q_proj.weight, block.k_proj.weight, block.v_proj.weight]
                    qkv_scale = _weight_scale(qkv)
                    qkv_scale, _ = self._quantize_group(qkv, qkv_scale)
                    out_scale, _ = self._quantize_group([block.out_proj.weight])
                    up_scale, up_codes = self._quantize_group([block.ffn_linear1.weight])
                    gate_scale, gate_codes = self._quantize_group([block.ffn_linear_gate.weight])
                    down_scale, _ = self._quantize_group([block.ffn_linear2.weight])
                    self.scales[layer_name] = {
                        "qkvSharedWeightScale": qkv_scale,
                        "attentionOutWeightScale": out_scale,
                        "ffnUpWeightScale": up_scale,
                        "ffnGateWeightScale": gate_scale,
                        "ffnDownWeightScale": down_scale,
                    }
                    self.quantized_weights[layer_name] = {
                        "ffnUp": up_codes[0],
                        "ffnGate": gate_codes[0],
                    }
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if not self._active:
            return
        with torch.no_grad():
            for parameter, original in reversed(self._originals):
                parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))
        self._originals.clear()
        self.quantized_weights.clear()
        self._active = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class TransformerBoundaryHooks:
    """Observe or QDQ true Linear inputs without replacing the model forward."""

    def __init__(
        self,
        layers: Sequence[Tuple[str, TransformerRoPEGQABlock]],
        samples: Optional[Mapping[str, Mapping[str, ActivationSample]]] = None,
        thresholds: Optional[Mapping[str, Mapping[str, float]]] = None,
        saturation: Optional[Mapping[str, Mapping[str, SaturationCounter]]] = None,
        weight_qdq: Optional[AggressiveInt8WeightQDQ] = None,
    ):
        if samples is None and thresholds is None:
            raise ValueError("hooks require activation samples or QDQ thresholds")
        self._handles = []

        def install(
            module, layer_name, field, observe, quantized_state=None,
            quantized_state_key=None,
        ):
            def hook(_module, args):
                if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                    raise ValueError(f"unexpected Linear input contract at {layer_name}.{field}")
                value = args[0]
                if observe and samples is not None:
                    samples[layer_name][field].observe(value)
                if thresholds is None:
                    return None
                threshold = thresholds[layer_name][field]
                if observe and saturation is not None:
                    saturation[layer_name][field].observe(value, threshold)
                quantized = quantize_symmetric_int8_fp16(value, threshold)
                if quantized_state is not None:
                    if quantized_state[quantized_state_key] is not None:
                        raise ValueError(
                            f"stale quantized input state at {layer_name}.{field}"
                        )
                    quantized_state[quantized_state_key] = quantized
                return (
                    dequantize_symmetric_int8_fp16(
                        quantized, threshold, value.dtype
                    ),
                )

            self._handles.append(module.register_forward_pre_hook(hook))

        def install_swiglu_product_qdq(
            module, layer_name, clip, state, quantized_weights, weight_scales
        ):
            field = BOUNDARY_FIELDS[3]

            def hook(_module, args):
                if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                    raise ValueError(
                        f"unexpected Linear input contract at {layer_name}.{field}"
                    )
                value = args[0]
                quantized_input = state["input"]
                state["input"] = None
                if quantized_input is None:
                    raise ValueError(f"missing quantized FFN input at {layer_name}")
                product_max_abs = thresholds[layer_name][field]
                input_max_abs = thresholds[layer_name][BOUNDARY_FIELDS[2]]
                quantized_up = native_code_domain_swiglu_factor_int8(
                    quantized_input,
                    quantized_weights["ffnUp"],
                    input_max_abs,
                    weight_scales["ffnUpWeightScale"],
                    clip,
                    apply_silu=True,
                )
                quantized_gate = native_code_domain_swiglu_factor_int8(
                    quantized_input,
                    quantized_weights["ffnGate"],
                    input_max_abs,
                    weight_scales["ffnGateWeightScale"],
                    clip,
                    apply_silu=False,
                )
                scaled = scaled_swiglu_factor_product_float32(
                    quantized_up, quantized_gate, clip, product_max_abs
                )
                if saturation is not None:
                    saturation[layer_name][field].observe_int8_scaled(scaled)
                quantized_product = torch.round(scaled).clamp_(QMIN, QMAX).to(
                    dtype=torch.int8
                )
                dequantized = dequantize_symmetric_int8_fp16(
                    quantized_product, product_max_abs, value.dtype
                )
                return (dequantized,)

            self._handles.append(module.register_forward_pre_hook(hook))

        for layer_name, block in layers:
            # Q/K/V and up/gate share an input. Observe it once, but QDQ every
            # consumer so the actual full-network validation follows the native
            # transaction rather than merely comparing a quantizer to itself.
            install(block.q_proj, layer_name, BOUNDARY_FIELDS[0], True)
            install(block.k_proj, layer_name, BOUNDARY_FIELDS[0], False)
            install(block.v_proj, layer_name, BOUNDARY_FIELDS[0], False)
            install(block.out_proj, layer_name, BOUNDARY_FIELDS[1], True)
            if not block.use_swiglu or not hasattr(block, "ffn_linear_gate"):
                raise ValueError(f"{layer_name}: native INT8 calibration requires SwiGLU")
            # D4's clipped path preserves q_up/q_gate and requantizes their
            # integer product directly. The factors themselves are recomputed
            # from the saved input/weight INT8 codes so PyTorch's FP16 Linear
            # outputs cannot alter the native FP32 factor epilogue.
            if thresholds is not None and block.swiglu_clip is not None:
                if weight_qdq is None or not weight_qdq._active:
                    raise ValueError(
                        f"{layer_name}: clipped native factor validation requires "
                        "an active aggressive weight-QDQ context"
                    )
                if (
                    layer_name not in weight_qdq.quantized_weights
                    or layer_name not in weight_qdq.scales
                ):
                    raise ValueError(f"{layer_name}: missing quantized FFN weights")
                factor_clip = _finite_positive(
                    block.swiglu_clip, f"{layer_name}.swigluClip"
                )
                factor_state = {"input": None}
                install(
                    block.ffn_linear1, layer_name, BOUNDARY_FIELDS[2], True,
                    factor_state, "input"
                )
                install(
                    block.ffn_linear_gate, layer_name, BOUNDARY_FIELDS[2], False
                )
                install_swiglu_product_qdq(
                    block.ffn_linear2,
                    layer_name,
                    factor_clip,
                    factor_state,
                    weight_qdq.quantized_weights[layer_name],
                    weight_qdq.scales[layer_name],
                )
            else:
                install(block.ffn_linear1, layer_name, BOUNDARY_FIELDS[2], True)
                install(block.ffn_linear_gate, layer_name, BOUNDARY_FIELDS[2], False)
                install(block.ffn_linear2, layer_name, BOUNDARY_FIELDS[3], True)

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def make_activation_samples(
    layer_order: Sequence[str],
) -> Dict[str, Dict[str, ActivationSample]]:
    result = {}
    for layer_name in layer_order:
        result[layer_name] = {}
        for field in BOUNDARY_FIELDS:
            result[layer_name][field] = ActivationSample()
    return result


def candidate_thresholds(
    samples: Mapping[str, Mapping[str, ActivationSample]],
    candidates: Sequence[Tuple[str, Optional[float]]],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    thresholds = {}
    saturation = {}
    for candidate_name, percentile in candidates:
        thresholds[candidate_name] = {}
        saturation[candidate_name] = {}
        for layer_name, layer_samples in samples.items():
            thresholds[candidate_name][layer_name] = {}
            saturation[candidate_name][layer_name] = {}
            for field in BOUNDARY_FIELDS:
                value = layer_samples[field].threshold(percentile)
                thresholds[candidate_name][layer_name][field] = value
                saturation[candidate_name][layer_name][field] = \
                    layer_samples[field].estimated_saturation_rate(value)
    return thresholds, saturation


def make_saturation_counters(layer_order: Sequence[str]):
    return {
        layer_name: {field: SaturationCounter() for field in BOUNDARY_FIELDS}
        for layer_name in layer_order
    }


def _reject_clip_keys(value, path="root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).replace("_", "").lower()
            if normalized == "swigluclip":
                raise ValueError(
                    f"{path}.{key} is forbidden: SwiGLU clip is read only from the checkpoint"
                )
            _reject_clip_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_clip_keys(child, f"{path}[{index}]")


def _require_exact_keys(value: Mapping, expected: Iterable[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    expected = set(expected)
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{context} keys mismatch: missing={missing}, extra={extra}")


def _validate_sha(value, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256")
    return value


def _validate_source_record(value, context: str, dataset: bool) -> None:
    if dataset:
        _require_exact_keys(value, ("sha256", "files"), context)
        _validate_sha(value["sha256"], f"{context}.sha256")
        if not isinstance(value["files"], list) or not value["files"]:
            raise ValueError(f"{context}.files must be a nonempty array")
        for index, record in enumerate(value["files"]):
            item_context = f"{context}.files[{index}]"
            _require_exact_keys(record, ("index", "name", "bytes", "sha256"), item_context)
            if record["index"] != index:
                raise ValueError(f"{item_context}.index is out of order")
            if not isinstance(record["name"], str) or not record["name"]:
                raise ValueError(f"{item_context}.name must be nonempty")
            if isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int) or record["bytes"] <= 0:
                raise ValueError(f"{item_context}.bytes must be positive")
            _validate_sha(record["sha256"], f"{item_context}.sha256")
    else:
        _require_exact_keys(value, ("sha256", "bytes"), context)
        _validate_sha(value["sha256"], f"{context}.sha256")
        if isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] <= 0:
            raise ValueError(f"{context}.bytes must be positive")


def load_calibration_json(path: Path) -> Dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as source:
        document = json.load(source, object_pairs_hook=reject_duplicate_keys)
    _reject_clip_keys(document)
    return document


def validate_calibration_document(
    document: Dict,
    checkpoint_path: Path,
    layer_order: Sequence[str],
    use_swa: bool,
    pos_len: int,
    wire_version: int = WIRE_VERSION,
) -> Dict[str, Dict[str, float]]:
    """Fail closed and return the selected per-layer ranges for export."""
    _reject_clip_keys(document)
    _require_exact_keys(
        document,
        (
            "schema", "schemaVersion", "wireVersion", "source", "evaluation",
            "quantization", "layerOrder", "layers", "selection",
        ),
        "calibration",
    )
    if document["schema"] != SCHEMA_NAME or document["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError("unsupported native INT8 calibration schema")
    if wire_version not in CUDA_INT8_WIRE_VERSION_BY_FLOAT_VERSION.values():
        raise ValueError(f"unsupported CUDA INT8 wire version {wire_version}")
    if document["wireVersion"] != wire_version:
        raise ValueError(f"calibration wireVersion must be {wire_version}")

    source = document["source"]
    _require_exact_keys(
        source,
        ("checkpoint", "calibrationData", "validationData", "processedRows"),
        "source",
    )
    _validate_source_record(source["checkpoint"], "source.checkpoint", dataset=False)
    _validate_source_record(source["calibrationData"], "source.calibrationData", dataset=True)
    _validate_source_record(source["validationData"], "source.validationData", dataset=True)
    checkpoint_sha = sha256_file(checkpoint_path)
    if source["checkpoint"]["sha256"] != checkpoint_sha:
        raise ValueError(
            "calibration checkpoint SHA256 mismatch: expected "
            f"{checkpoint_sha}, got {source['checkpoint']['sha256']}"
        )
    checkpoint_bytes = checkpoint_path.stat().st_size
    if source["checkpoint"]["bytes"] != checkpoint_bytes:
        raise ValueError(
            "calibration checkpoint byte size mismatch: expected "
            f"{checkpoint_bytes}, got {source['checkpoint']['bytes']}"
        )
    processed_rows = source["processedRows"]
    _require_exact_keys(
        processed_rows,
        (
            "calibrationRows", "calibrationUniqueRows", "calibrationSetSha256",
            "validationRows", "validationUniqueRows", "validationSetSha256",
            "overlapRows",
        ),
        "source.processedRows",
    )
    for prefix in ("calibration", "validation"):
        rows = processed_rows[prefix + "Rows"]
        unique = processed_rows[prefix + "UniqueRows"]
        if (
            isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
            or isinstance(unique, bool) or not isinstance(unique, int)
            or unique <= 0 or unique > rows
        ):
            raise ValueError(f"source.processedRows {prefix} counts are invalid")
        _validate_sha(
            processed_rows[prefix + "SetSha256"],
            f"source.processedRows.{prefix}SetSha256",
        )
    if processed_rows["overlapRows"] != 0:
        raise ValueError("calibration and validation processed row overlap must be zero")

    evaluation = document["evaluation"]
    _require_exact_keys(evaluation, ("modelState", "useSwa", "posLen"), "evaluation")
    if not isinstance(evaluation["useSwa"], bool) or evaluation["useSwa"] != use_swa:
        raise ValueError("calibration useSwa does not match exporter -use-swa")
    expected_state = "swa" if use_swa else "raw"
    if evaluation["modelState"] != expected_state:
        raise ValueError(f"calibration modelState must be {expected_state}")
    if evaluation["posLen"] != pos_len:
        raise ValueError("calibration posLen does not match exporter -pos-len")

    quantization = document["quantization"]
    _require_exact_keys(
        quantization,
        (
            "dtype", "qmin", "qmax", "zeroPoint", "rounding",
            "validationArithmetic", "candidates", "weightQdq",
        ),
        "quantization",
    )
    if (
        quantization["dtype"] != "int8"
        or quantization["qmin"] != QMIN
        or quantization["qmax"] != QMAX
        or quantization["zeroPoint"] != 0
        or quantization["rounding"] != "roundTiesToEven"
    ):
        raise ValueError("calibration quantization contract is not symmetric RNE INT8")
    validation_arithmetic = quantization["validationArithmetic"]
    _require_exact_keys(
        validation_arithmetic,
        (
            "overall", "fp16BoundaryFields", "clippedFactorProduct",
            "clippedSilu", "noClipProduct", "productDownFeed",
        ),
        "quantization.validationArithmetic",
    )
    if (
        validation_arithmetic["overall"] != "pytorchFakeQdq"
        or validation_arithmetic["fp16BoundaryFields"] != list(BOUNDARY_FIELDS[:3])
        or validation_arithmetic["clippedFactorProduct"]
        != "nativeCodeDomainInt8DotFp32ScaleDirectRequant"
        or validation_arithmetic["clippedSilu"]
        != "pytorchFp32SimulationNotBitExactCutlass"
        or validation_arithmetic["noClipProduct"]
        != "modelFloatProductBoundaryQdq"
        or validation_arithmetic["productDownFeed"]
        != "pytorchDequantizedSurrogate"
    ):
        raise ValueError("calibration validation arithmetic contract is unsupported")
    candidates = quantization["candidates"]
    expected_candidates = [name for name, _ in DEFAULT_CANDIDATES]
    if candidates != expected_candidates:
        raise ValueError(
            "quantization.candidates must exactly match the deterministic "
            f"candidate order {expected_candidates}"
        )
    weight_qdq = quantization["weightQdq"]
    _require_exact_keys(
        weight_qdq, ("qmin", "qmax", "zeroPoint", "rounding", "scale", "groups"),
        "quantization.weightQdq",
    )
    if (
        weight_qdq["qmin"] != QMIN
        or weight_qdq["qmax"] != QMAX
        or weight_qdq["zeroPoint"] != 0
        or weight_qdq["rounding"] != "roundTiesToEven"
        or weight_qdq["scale"] != "float32GroupMaxAbsDiv127"
        or weight_qdq["groups"] != ["qkvShared", "attentionOut", "ffnUp", "ffnGate", "ffnDown"]
    ):
        raise ValueError("calibration weightQdq contract does not match aggressive engine packing")

    if document["layerOrder"] != list(layer_order):
        raise ValueError("calibration layerOrder does not exactly match checkpoint wire order")
    layers = document["layers"]
    if not isinstance(layers, list) or len(layers) != len(layer_order):
        raise ValueError(
            f"calibration must have exactly {len(layer_order)} layer records"
        )

    selection = document["selection"]
    _require_exact_keys(
        selection,
        (
            "metric", "baselineLoss", "weightOnlyLoss", "candidateLosses",
            "chosenCandidate", "selectedLoss", "lossDelta", "baselineMetrics",
            "weightOnlyMetrics", "candidateMetrics",
        ),
        "selection",
    )
    if selection["metric"] != "trainingLossPerWeight":
        raise ValueError("selection.metric must be trainingLossPerWeight")
    baseline_loss = _finite_nonnegative(selection["baselineLoss"], "selection.baselineLoss")
    weight_only_loss = _finite_nonnegative(selection["weightOnlyLoss"], "selection.weightOnlyLoss")
    _require_exact_keys(selection["baselineMetrics"], LOSS_METRIC_FIELDS, "selection.baselineMetrics")
    for field in LOSS_METRIC_FIELDS:
        _finite_nonnegative(selection["baselineMetrics"][field], f"selection.baselineMetrics.{field}")
    if float(selection["baselineMetrics"]["trainingLossPerWeight"]) != baseline_loss:
        raise ValueError("selection.baselineLoss does not match baselineMetrics")

    _require_exact_keys(
        selection["weightOnlyMetrics"],
        (*LOSS_METRIC_FIELDS, *LOSS_DELTA_FIELDS),
        "selection.weightOnlyMetrics",
    )
    for field in LOSS_METRIC_FIELDS:
        _finite_nonnegative(selection["weightOnlyMetrics"][field], f"selection.weightOnlyMetrics.{field}")
    if float(selection["weightOnlyMetrics"]["trainingLossPerWeight"]) != weight_only_loss:
        raise ValueError("selection.weightOnlyLoss does not match weightOnlyMetrics")
    for field, delta_field in zip(LOSS_METRIC_FIELDS, LOSS_DELTA_FIELDS):
        delta = float(selection["weightOnlyMetrics"][delta_field])
        expected = float(selection["weightOnlyMetrics"][field]) - float(selection["baselineMetrics"][field])
        if not math.isfinite(delta) or not math.isclose(delta, expected, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"selection.weightOnlyMetrics.{delta_field} is inconsistent")

    _require_exact_keys(selection["candidateLosses"], candidates, "selection.candidateLosses")
    _require_exact_keys(selection["candidateMetrics"], candidates, "selection.candidateMetrics")
    for name in candidates:
        _finite_nonnegative(selection["candidateLosses"][name], f"selection.candidateLosses.{name}")
        metrics = selection["candidateMetrics"][name]
        _require_exact_keys(
            metrics, (*LOSS_METRIC_FIELDS, *LOSS_DELTA_FIELDS),
            f"selection.candidateMetrics.{name}",
        )
        for field in LOSS_METRIC_FIELDS:
            _finite_nonnegative(metrics[field], f"selection.candidateMetrics.{name}.{field}")
        if float(metrics["trainingLossPerWeight"]) != float(selection["candidateLosses"][name]):
            raise ValueError(f"selection candidate {name} aggregate loss is inconsistent")
        for field, delta_field in zip(LOSS_METRIC_FIELDS, LOSS_DELTA_FIELDS):
            delta = float(metrics[delta_field])
            expected = float(metrics[field]) - float(selection["baselineMetrics"][field])
            if not math.isfinite(delta) or not math.isclose(delta, expected, rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError(f"selection.candidateMetrics.{name}.{delta_field} is inconsistent")
    chosen = selection["chosenCandidate"]
    if chosen not in candidates:
        raise ValueError("selection.chosenCandidate is not a declared candidate")
    expected_chosen = min(
        candidates,
        key=lambda name: (float(selection["candidateLosses"][name]), candidates.index(name)),
    )
    if chosen != expected_chosen:
        raise ValueError(
            "selection.chosenCandidate is not the minimum validation-loss candidate"
        )
    selected_loss = _finite_nonnegative(selection["selectedLoss"], "selection.selectedLoss")
    if selected_loss != float(selection["candidateLosses"][chosen]):
        raise ValueError("selection.selectedLoss does not match chosen candidate loss")
    loss_delta = float(selection["lossDelta"])
    if not math.isfinite(loss_delta) or not math.isclose(
        loss_delta, selected_loss - baseline_loss, rel_tol=1e-7, abs_tol=1e-10
    ):
        raise ValueError("selection.lossDelta is inconsistent")

    selected = {}
    layer_keys = (
        "index", "name", *BOUNDARY_FIELDS, "calibrationSample", "candidates",
        "validationSaturationRates", "weightQdqScales",
    )
    for index, (expected_name, layer) in enumerate(zip(layer_order, layers)):
        context = f"layers[{index}]"
        _require_exact_keys(layer, layer_keys, context)
        if layer["index"] != index or layer["name"] != expected_name:
            raise ValueError(f"{context} index/name does not match layerOrder")
        _require_exact_keys(layer["weightQdqScales"], WEIGHT_SCALE_FIELDS, f"{context}.weightQdqScales")
        for field in WEIGHT_SCALE_FIELDS:
            _finite_positive(layer["weightQdqScales"][field], f"{context}.weightQdqScales.{field}")
        _require_exact_keys(layer["calibrationSample"], BOUNDARY_FIELDS, f"{context}.calibrationSample")
        for field in BOUNDARY_FIELDS:
            sample = layer["calibrationSample"][field]
            _require_exact_keys(
                sample, ("observedValues", "sampledValues", "observations", "maxAbs"),
                f"{context}.calibrationSample.{field}",
            )
            for count_name in ("observedValues", "sampledValues", "observations"):
                count = sample[count_name]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{context}.calibrationSample.{field}.{count_name} must be positive")
            if sample["sampledValues"] != sample["observedValues"]:
                raise ValueError(
                    f"{context}.calibrationSample.{field} must record the exact "
                    "FP16 histogram without sampling"
                )
            _finite_positive(sample["maxAbs"], f"{context}.calibrationSample.{field}.maxAbs")

        _require_exact_keys(layer["candidates"], candidates, f"{context}.candidates")
        for candidate_name in candidates:
            record = layer["candidates"][candidate_name]
            _require_exact_keys(
                record, ("thresholds", "calibrationSaturationRates"),
                f"{context}.candidates.{candidate_name}",
            )
            _require_exact_keys(record["thresholds"], BOUNDARY_FIELDS, f"{context}.candidates.{candidate_name}.thresholds")
            _require_exact_keys(record["calibrationSaturationRates"], BOUNDARY_FIELDS, f"{context}.candidates.{candidate_name}.calibrationSaturationRates")
            for field in BOUNDARY_FIELDS:
                _finite_positive(record["thresholds"][field], f"{context}.candidates.{candidate_name}.thresholds.{field}")
                rate = float(record["calibrationSaturationRates"][field])
                if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
                    raise ValueError(f"{context} calibration saturation rate must be in [0,1]")

        _require_exact_keys(layer["validationSaturationRates"], candidates, f"{context}.validationSaturationRates")
        for candidate_name in candidates:
            rates = layer["validationSaturationRates"][candidate_name]
            _require_exact_keys(rates, BOUNDARY_FIELDS, f"{context}.validationSaturationRates.{candidate_name}")
            for field in BOUNDARY_FIELDS:
                rate = float(rates[field])
                if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
                    raise ValueError(f"{context} validation saturation rate must be in [0,1]")

        selected[expected_name] = {}
        chosen_thresholds = layer["candidates"][chosen]["thresholds"]
        for field in BOUNDARY_FIELDS:
            selected_value = _finite_positive(layer[field], f"{context}.{field}")
            candidate_value = _finite_positive(chosen_thresholds[field], f"{context}.chosen.{field}")
            if struct.pack("<f", selected_value) != struct.pack("<f", candidate_value):
                raise ValueError(f"{context}.{field} does not match chosen candidate")
            selected[expected_name][field] = selected_value
    return selected


def write_calibration_json(path: Path, document: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(document, destination, sort_keys=True, indent=2, allow_nan=False)
        destination.write("\n")
