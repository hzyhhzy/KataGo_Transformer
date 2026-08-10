#!/usr/bin/env python3
"""Build the validated ORT-CPU dynamic-trunk W8A8 ONNX model.

This is the user-facing INT8 entry point for KataGo transformer ONNX graphs.
Its input is an FP32 graph (normally the output of
``ort_cpu_optimize_fp32.py``), and its output uses ONNX Runtime's dynamic
MatMulInteger path:

* activations are dynamically quantized to uint8 for every inference;
* weights are symmetric, per-tensor QInt8;
* only the seven logical projections in every transformer block are selected;
* the spatial stem, global projection, and all policy/value heads stay FP32.

Discovery is topology-aware and accepts fused or unfused Q/K/V and fused or
unfused SwiGLU up/gate projections.  The command fails closed if the physical
selection cannot be proven to cover exactly the seven logical trunk roles in
every contiguous block. Model and report files may be written to any
user-selected directory. Each file is fully staged beside its own destination
and atomically replaces that destination only after structural audits and
optional validation have succeeded; the two replacements are not a cross-file
transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, defs, helper, numpy_helper

EXPECTED_OUTPUTS = (
    "out_policy",
    "out_value",
    "out_miscvalue",
    "out_moremiscvalue",
    "out_ownership",
)
LOGICAL_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "ffn_linear1",
    "ffn_linear_gate",
    "ffn_linear2",
)
NAMED_ROLE_RE = re.compile(
    r"(?:^|/)blocks\.(\d+)/(q_proj|k_proj|v_proj|out_proj|"
    r"ffn_linear1|ffn_linear_gate|ffn_linear2)(?:/|$)"
)
BLOCK_RE = re.compile(r"(?:^|/)blocks\.(\d+)(?:/|$)")


@dataclass(frozen=True)
class FusedProjection:
    block: int
    kind: str
    roles: tuple[str, ...]
    matmul: str
    split: str
    activation: str
    weight: str
    weight_shape: tuple[int, ...]
    split_widths: tuple[int, ...]
    split_axis: int
    split_outputs: tuple[str, ...]


@dataclass(frozen=True)
class BlockSelection:
    block: int
    logical_to_physical: dict[str, str]
    fused: tuple[FusedProjection, ...]

    @property
    def physical_nodes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.logical_to_physical.values()))


class WorkspaceTemporaryDirectory:
    """A TemporaryDirectory that avoids restrictive Windows temp ACLs."""

    def __init__(self, suffix=None, prefix=None, dir=None, **_kwargs) -> None:
        del suffix, dir
        root = _QUANT_TEMP_ROOT
        if root is None:
            raise RuntimeError("Quantization temporary root was not configured")
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self.name = str(root / f".{prefix or 'ort-quant-'}{uuid.uuid4().hex}")
        Path(self.name).mkdir(parents=False, exist_ok=False)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        self.cleanup()
        return False

    def cleanup(self) -> None:
        path = Path(self.name)
        try:
            path.resolve().relative_to(self._root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to clean unexpected temp path {path}") from exc
        shutil.rmtree(path, ignore_errors=True)


_QUANT_TEMP_ROOT: Path | None = None


def prepare_output_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _register_simplified_layer_norm_schema() -> bool:
    """Teach older ONNX checkers about ORT's default-domain SLN operator."""
    try:
        defs.get_schema("SimplifiedLayerNormalization", 20, "")
        return False
    except onnx.onnx_cpp2py_export.defs.SchemaError:
        pass

    floating_types = [
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(bfloat16)",
    ]
    schema = defs.OpSchema(
        "SimplifiedLayerNormalization",
        "",
        1,
        inputs=[
            defs.OpSchema.FormalParameter("X", "T"),
            defs.OpSchema.FormalParameter("scale", "V"),
        ],
        outputs=[
            defs.OpSchema.FormalParameter("Y", "V"),
            defs.OpSchema.FormalParameter(
                "inv_std_var",
                "U",
                param_option=defs.OpSchema.FormalParameterOption.Optional,
            ),
        ],
        type_constraints=[
            ("T", floating_types, "input floating-point type"),
            ("V", floating_types, "scale/output floating-point type"),
            ("U", ["tensor(float)", "tensor(double)"], "accumulator type"),
        ],
        attributes=[
            defs.OpSchema.Attribute(
                "axis", defs.OpSchema.AttrType.INT, "normalization axis", required=False
            ),
            defs.OpSchema.Attribute(
                "epsilon",
                defs.OpSchema.AttrType.FLOAT,
                "numerical epsilon",
                required=False,
            ),
            defs.OpSchema.Attribute(
                "stash_type",
                defs.OpSchema.AttrType.INT,
                "accumulator type",
                required=False,
            ),
        ],
    )
    defs.register_schema(schema)
    return True


def checker_with_ort_schema(model: onnx.ModelProto) -> str:
    registered = _register_simplified_layer_norm_schema()
    try:
        onnx.checker.check_model(model, full_check=True)
    finally:
        if registered:
            defs.deregister_schema("SimplifiedLayerNormalization", 1, "")
    return "onnx.checker full_check passed with ORT SLN schema shim"


def stage_json(path: Path, value: dict[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_external_data(model: onnx.ModelProto, path: Path) -> None:
    external = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL
        or len(initializer.external_data) != 0
    ]
    if external:
        raise ValueError(
            f"External-data ONNX is not supported because one-file atomic/hash "
            f"guarantees would be incomplete ({path}; initializers={external[:10]})"
        )


def op_counts(model: onnx.ModelProto) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{node.domain or 'ai.onnx'}::{node.op_type}"
                for node in model.graph.node
            ).items()
        )
    )


def block_ids(strings: Iterable[str]) -> set[int]:
    result: set[int] = set()
    for text in strings:
        for match in BLOCK_RE.finditer(text):
            result.add(int(match.group(1)))
    return result


def split_spec(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
    output_width: int,
) -> tuple[tuple[int, ...], int] | None:
    if node.op_type != "Split" or len(node.output) not in (2, 3):
        return None
    axis = 0
    widths: tuple[int, ...] | None = None
    for attribute in node.attribute:
        if attribute.name == "axis":
            axis = int(attribute.i)
        elif attribute.name == "split":
            widths = tuple(int(value) for value in attribute.ints)
    if len(node.input) >= 2 and node.input[1] in initializers:
        values = np.asarray(numpy_helper.to_array(initializers[node.input[1]])).reshape(-1)
        widths = tuple(int(value) for value in values.tolist())
    if widths is None and output_width % len(node.output) == 0:
        widths = (output_width // len(node.output),) * len(node.output)
    if (
        widths is None
        or len(widths) != len(node.output)
        or any(width <= 0 for width in widths)
        or sum(widths) != output_width
    ):
        return None
    return widths, axis


def discover_fused_projections(
    model: onnx.ModelProto,
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
) -> dict[int, dict[str, FusedProjection]]:
    result: dict[int, dict[str, FusedProjection]] = defaultdict(dict)
    for split in model.graph.node:
        if not split.input:
            continue
        projection = producers.get(split.input[0])
        if (
            projection is None
            or projection.op_type != "MatMul"
            or len(projection.input) != 2
            or len(projection.output) != 1
            or projection.input[1] not in initializers
        ):
            continue
        weight = np.asarray(numpy_helper.to_array(initializers[projection.input[1]]))
        if weight.ndim != 2:
            continue
        spec = split_spec(split, initializers, int(weight.shape[1]))
        if spec is None:
            continue
        widths, axis = spec
        if len(split.output) == 3:
            kind = "qkv"
            roles = ("q_proj", "k_proj", "v_proj")
            expected_hint = "/norm1/"
            explicit_hint = "qkv_fused"
            # A learned per-head graph with exactly three heads can also have
            # q_proj -> SplitQHeads with three outputs.  It is not fused QKV:
            # its projection is d->d and each Split segment is one head.  The
            # regular-MHA QKV fusion proven by this tool is d->3d with three
            # d-wide segments (GQA is deliberately outside this contract).
            input_width = int(weight.shape[0])
            if int(weight.shape[1]) != 3 * input_width or widths != (
                input_width,
                input_width,
                input_width,
            ):
                continue
        else:
            kind = "up_gate"
            roles = ("ffn_linear1", "ffn_linear_gate")
            expected_hint = "/norm2/"
            explicit_hint = "up_gate_fused"

        scope_strings = [
            projection.name,
            *projection.input,
            *projection.output,
            split.name,
            *split.output,
        ]
        joined = "\n".join(scope_strings)
        # A Split elsewhere in a block is not sufficient proof.  Generated
        # named fusions identify themselves; anonymous onnxsim fusions retain
        # the norm1/norm2 activation path.
        if explicit_hint not in joined and expected_hint not in projection.input[0]:
            continue
        if axis not in (-1, 2):
            raise ValueError(
                f"Fused {kind} candidate {projection.name!r}/{split.name!r} "
                f"splits axis {axis}, expected the rank-3 feature axis 2/-1"
            )
        ids = block_ids(scope_strings)
        if not ids:
            ids = block_ids(
                consumer.name
                for output in split.output
                for consumer in consumers.get(output, ())
            )
        if len(ids) != 1:
            raise ValueError(
                f"Fused {kind} candidate {projection.name!r}/{split.name!r} "
                f"has ambiguous block scopes {sorted(ids)}"
            )
        block = next(iter(ids))
        if kind in result[block]:
            raise ValueError(f"Block {block} has multiple fused {kind} candidates")
        result[block][kind] = FusedProjection(
            block=block,
            kind=kind,
            roles=roles,
            matmul=projection.name,
            split=split.name,
            activation=projection.input[0],
            weight=projection.input[1],
            weight_shape=tuple(int(value) for value in weight.shape),
            split_widths=widths,
            split_axis=axis,
            split_outputs=tuple(split.output),
        )
    return dict(result)


def discover_named_roles(
    model: onnx.ModelProto,
    initializer_names: set[str],
    fused_node_names: set[str],
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = defaultdict(dict)
    for node in model.graph.node:
        if (
            node.op_type != "MatMul"
            or len(node.input) != 2
            or node.input[1] not in initializer_names
            or node.name in fused_node_names
        ):
            continue
        match = NAMED_ROLE_RE.search(node.name)
        if match is None:
            continue
        block = int(match.group(1))
        role = match.group(2)
        if role in result[block]:
            raise ValueError(f"Duplicate block {block} role {role}: {node.name!r}")
        result[block][role] = node.name
    return dict(result)


def classify_linear_inventory(
    model: onnx.ModelProto,
    initializer_names: set[str],
    selected: set[str],
) -> dict[str, list[dict[str, str]]]:
    inventory: dict[str, list[dict[str, str]]] = {
        "selected_trunk": [],
        "fp32_linear_global": [],
        "fp32_policy_value_heads": [],
        "fp32_unclassified_other": [],
    }
    for node in model.graph.node:
        if (
            node.op_type not in {"MatMul", "Gemm"}
            or len(node.input) < 2
            or node.input[1] not in initializer_names
        ):
            continue
        item = {"name": node.name, "op_type": node.op_type, "weight": node.input[1]}
        if node.name in selected:
            inventory["selected_trunk"].append(item)
        elif "/linear_global/" in node.name:
            inventory["fp32_linear_global"].append(item)
        elif "/policy_head/" in node.name or "/value_head/" in node.name:
            inventory["fp32_policy_value_heads"].append(item)
        else:
            inventory["fp32_unclassified_other"].append(item)
    return inventory


def discover_selection(
    model: onnx.ModelProto,
    expected_blocks: int | None,
    expected_head_linears: int | None,
) -> tuple[list[BlockSelection], dict[str, Any]]:
    names = [node.name for node in model.graph.node]
    if any(not name for name in names):
        raise ValueError("Every graph node must have a nonempty name")
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise ValueError(f"Duplicate node names: {duplicate_names[:10]}")
    if any(node.op_type == "MatMulInteger" for node in model.graph.node):
        raise ValueError("Input already contains MatMulInteger; an FP32 input is required")

    initializers = {item.name: item for item in model.graph.initializer}
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)

    fused = discover_fused_projections(model, initializers, producers, consumers)
    fused_node_names = {
        projection.matmul
        for block_fusions in fused.values()
        for projection in block_fusions.values()
    }
    named = discover_named_roles(model, set(initializers), fused_node_names)
    discovered_ids = sorted(set(fused) | set(named))
    if not discovered_ids:
        raise ValueError("No transformer projection blocks were discovered")
    block_count = expected_blocks if expected_blocks is not None else discovered_ids[-1] + 1
    expected_ids = list(range(block_count))
    if discovered_ids != expected_ids:
        raise ValueError(
            f"Discovered block IDs are not exactly 0..{block_count - 1}: {discovered_ids}"
        )

    selections: list[BlockSelection] = []
    for block in expected_ids:
        mapping: dict[str, str] = {}
        block_fused: list[FusedProjection] = []
        named_roles = named.get(block, {})
        fused_roles = fused.get(block, {})

        qkv = fused_roles.get("qkv")
        named_qkv = [role for role in LOGICAL_ROLES[:3] if role in named_roles]
        if qkv is not None and named_qkv:
            raise ValueError(f"Block {block} has both fused and named QKV projections")
        if qkv is not None:
            block_fused.append(qkv)
            mapping.update({role: qkv.matmul for role in qkv.roles})
        elif len(named_qkv) == 3:
            mapping.update({role: named_roles[role] for role in LOGICAL_ROLES[:3]})
        else:
            raise ValueError(f"Block {block} has incomplete QKV roles: {named_qkv}")

        up_gate = fused_roles.get("up_gate")
        up_gate_roles = ("ffn_linear1", "ffn_linear_gate")
        named_up_gate = [role for role in up_gate_roles if role in named_roles]
        if up_gate is not None and named_up_gate:
            raise ValueError(f"Block {block} has both fused and named FFN up/gate")
        if up_gate is not None:
            block_fused.append(up_gate)
            mapping.update({role: up_gate.matmul for role in up_gate.roles})
        elif len(named_up_gate) == 2:
            mapping.update({role: named_roles[role] for role in up_gate_roles})
        else:
            raise ValueError(f"Block {block} has incomplete FFN up/gate: {named_up_gate}")

        for role in ("out_proj", "ffn_linear2"):
            if role not in named_roles:
                raise ValueError(f"Block {block} is missing {role}")
            mapping[role] = named_roles[role]
        if set(mapping) != set(LOGICAL_ROLES):
            raise ValueError(f"Block {block} logical coverage is {sorted(mapping)}")
        selections.append(BlockSelection(block, mapping, tuple(block_fused)))

    physical = [name for item in selections for name in item.physical_nodes]
    if len(set(physical)) != len(physical):
        raise ValueError("One physical MatMul was selected across multiple blocks")
    inventory = classify_linear_inventory(model, set(initializers), set(physical))
    if len(inventory["selected_trunk"]) != len(physical):
        raise ValueError("Selected physical nodes do not match the constant-B inventory")
    if inventory["fp32_unclassified_other"]:
        raise ValueError(
            "Unclassified constant-weight linears make trunk-only selection unsafe: "
            f"{[item['name'] for item in inventory['fp32_unclassified_other']]}"
        )
    head_count = len(inventory["fp32_policy_value_heads"])
    if expected_head_linears is not None and head_count != expected_head_linears:
        raise ValueError(
            f"Expected {expected_head_linears} FP32 head linears, found {head_count}"
        )

    layout_counts = Counter(
        "+".join(projection.kind for projection in item.fused) or "unfused"
        for item in selections
    )
    audit = {
        "block_count": block_count,
        "logical_projections_covered": 7 * block_count,
        "expected_logical_projections": 7 * block_count,
        "physical_matmuls_selected": len(physical),
        "layout_counts": dict(sorted(layout_counts.items())),
        "linear_inventory": inventory,
        "fp32_preservation": {
            "linear_global_count": len(inventory["fp32_linear_global"]),
            "policy_value_head_linear_count": head_count,
            "conv_nodes": [
                node.name for node in model.graph.node if node.op_type == "Conv"
            ],
        },
        "blocks": [
            {
                "block": item.block,
                "logical_to_physical": item.logical_to_physical,
                "physical_nodes": list(item.physical_nodes),
                "fused_projections": [
                    {
                        "kind": projection.kind,
                        "roles": list(projection.roles),
                        "matmul": projection.matmul,
                        "split": projection.split,
                        "activation": projection.activation,
                        "weight": projection.weight,
                        "weight_shape": list(projection.weight_shape),
                        "split_widths": list(projection.split_widths),
                        "split_axis": projection.split_axis,
                        "split_outputs": list(projection.split_outputs),
                    }
                    for projection in item.fused
                ],
            }
            for item in selections
        ],
    }
    return selections, audit


def quantize(source: Path, destination: Path, selected_nodes: list[str]) -> str:
    global _QUANT_TEMP_ROOT

    original_temp = tempfile.TemporaryDirectory
    original_root = _QUANT_TEMP_ROOT
    _QUANT_TEMP_ROOT = destination.parent.resolve()
    tempfile.TemporaryDirectory = WorkspaceTemporaryDirectory  # type: ignore[assignment]
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(source),
            model_output=str(destination),
            op_types_to_quantize=["MatMul"],
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            nodes_to_quantize=selected_nodes,
            extra_options={"MatMulConstBOnly": True},
        )
        return ort.__version__
    finally:
        tempfile.TemporaryDirectory = original_temp  # type: ignore[assignment]
        _QUANT_TEMP_ROOT = original_root


def audit_quantized_graph(
    source: onnx.ModelProto,
    destination: Path,
    selected_nodes: list[str],
    preserved_linears: list[dict[str, str]],
) -> dict[str, Any]:
    model = onnx.load(str(destination), load_external_data=False)
    reject_external_data(model, destination)
    checker_status = checker_with_ort_schema(model)
    source_quantized_nodes = [
        f"{node.domain or 'ai.onnx'}::{node.op_type}:{node.name}"
        for node in source.graph.node
        if node.op_type in {
            "DynamicQuantizeLinear",
            "QuantizeLinear",
            "DequantizeLinear",
        }
        or node.op_type.startswith("QLinear")
        or node.op_type.endswith("Integer")
    ]
    if source_quantized_nodes:
        raise ValueError(
            "The input must be an FP32 graph without pre-existing quantized "
            f"operators: {source_quantized_nodes}"
        )
    source_outputs = tuple(item.name for item in source.graph.output)
    quantized_outputs = tuple(item.name for item in model.graph.output)
    if source_outputs != EXPECTED_OUTPUTS or quantized_outputs != EXPECTED_OUTPUTS:
        raise ValueError(
            f"Five-output graph contract failed: source={source_outputs}, "
            f"INT8={quantized_outputs}"
        )
    nodes_by_name = {node.name: node for node in model.graph.node}
    expected_integer_names = {f"{name}_quant" for name in selected_nodes}
    actual_integer_names = {
        node.name for node in model.graph.node if node.op_type == "MatMulInteger"
    }
    if actual_integer_names != expected_integer_names:
        raise ValueError(
            "MatMulInteger name audit failed: "
            f"missing={sorted(expected_integer_names - actual_integer_names)}, "
            f"unexpected={sorted(actual_integer_names - expected_integer_names)}"
        )
    initializers = {item.name: item for item in model.graph.initializer}
    source_initializers = {item.name: item for item in source.graph.initializer}
    source_nodes_by_name = {node.name: node for node in source.graph.node}
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for graph_node in model.graph.node:
        for input_name in graph_node.input:
            if input_name:
                consumers[input_name].append(graph_node)

    weight_audit: list[dict[str, Any]] = []
    dynamic_usage: dict[str, dict[str, Any]] = {}
    expected_integer_initializers: set[str] = set()
    for original_name in selected_nodes:
        source_node = source_nodes_by_name.get(original_name)
        if (
            source_node is None
            or source_node.op_type != "MatMul"
            or source_node.domain not in ("", "ai.onnx")
            or len(source_node.input) != 2
            or len(source_node.output) != 1
        ):
            raise ValueError(
                f"{original_name}: selected source node is not a canonical FP32 MatMul"
            )
        source_weight = source_initializers.get(source_node.input[1])
        if (
            source_weight is None
            or source_weight.data_type != TensorProto.FLOAT
            or len(source_weight.dims) != 2
        ):
            raise ValueError(
                f"{original_name}: source weight must be a rank-2 FP32 initializer"
            )

        node = nodes_by_name[f"{original_name}_quant"]
        if (
            node.op_type != "MatMulInteger"
            or node.domain not in ("", "ai.onnx")
            or len(node.input) != 4
            or len(node.output) != 1
        ):
            raise ValueError(f"{node.name}: expected four MatMulInteger inputs")
        activation_producer = producers.get(node.input[0])
        if (
            activation_producer is None
            or activation_producer.op_type != "DynamicQuantizeLinear"
            or activation_producer.domain not in ("", "ai.onnx")
            or list(activation_producer.input) != [source_node.input[0]]
            or len(activation_producer.output) != 3
            or node.input[0] != activation_producer.output[0]
            or node.input[2] != activation_producer.output[2]
        ):
            raise ValueError(f"{node.name}: activation is not dynamically quantized")

        expected_weight_name = f"{source_weight.name}_quantized"
        expected_scale_name = f"{source_weight.name}_scale"
        expected_zero_point_name = f"{source_weight.name}_zero_point"
        if node.input[1] != expected_weight_name or node.input[3] != expected_zero_point_name:
            raise ValueError(
                f"{node.name}: quantized weight/zero-point are not bound to source "
                f"weight {source_weight.name!r}: inputs={list(node.input)}"
            )
        weight = initializers.get(node.input[1])
        zero_point = initializers.get(node.input[3])
        scale = initializers.get(expected_scale_name)
        if (
            weight is None
            or weight.data_type != TensorProto.INT8
            or tuple(weight.dims) != tuple(source_weight.dims)
        ):
            raise ValueError(f"{node.name}: weight is not an INT8 initializer")
        if scale is None or scale.data_type != TensorProto.FLOAT or len(scale.dims) != 0:
            raise ValueError(f"{node.name}: weight scale is not scalar FP32")
        scale_value = np.asarray(numpy_helper.to_array(scale))
        if (
            scale_value.shape != ()
            or not np.isfinite(scale_value).item()
            or float(scale_value) <= 0.0
        ):
            raise ValueError(
                f"{node.name}: weight scale must be finite and positive, got {scale_value}"
            )
        if zero_point is None or zero_point.data_type != TensorProto.INT8:
            raise ValueError(f"{node.name}: weight zero-point is not QInt8")
        zero = np.asarray(numpy_helper.to_array(zero_point))
        if zero.shape != () or int(zero) != 0:
            raise ValueError(f"{node.name}: weight zero-point is not symmetric zero")

        integer_consumers = consumers.get(node.output[0], [])
        if len(integer_consumers) != 1:
            raise ValueError(
                f"{node.name}: integer output must have exactly one Cast consumer, "
                f"got {[item.name for item in integer_consumers]}"
            )
        cast = integer_consumers[0]
        cast_to = next(
            (
                int(helper.get_attribute_value(attribute))
                for attribute in cast.attribute
                if attribute.name == "to"
            ),
            None,
        )
        if (
            cast.op_type != "Cast"
            or cast.domain not in ("", "ai.onnx")
            or list(cast.input) != [node.output[0]]
            or len(cast.output) != 1
            or cast_to != TensorProto.FLOAT
        ):
            raise ValueError(f"{node.name}: integer accumulator is not cast to FP32")

        cast_consumers = consumers.get(cast.output[0], [])
        if len(cast_consumers) != 1:
            raise ValueError(
                f"{node.name}: FP32 accumulator must have one output-scale Mul, "
                f"got {[item.name for item in cast_consumers]}"
            )
        output_scale_mul = cast_consumers[0]
        if (
            output_scale_mul.op_type != "Mul"
            or output_scale_mul.domain not in ("", "ai.onnx")
            or len(output_scale_mul.input) != 2
            or list(output_scale_mul.input).count(cast.output[0]) != 1
            or list(output_scale_mul.output) != list(source_node.output)
        ):
            raise ValueError(
                f"{node.name}: output dequantization does not restore source tensor "
                f"{source_node.output[0]!r}"
            )
        combined_scale_name = (
            output_scale_mul.input[0]
            if output_scale_mul.input[1] == cast.output[0]
            else output_scale_mul.input[1]
        )
        scale_mul = producers.get(combined_scale_name)
        if (
            scale_mul is None
            or scale_mul.op_type != "Mul"
            or scale_mul.domain not in ("", "ai.onnx")
            or len(scale_mul.input) != 2
            or set(scale_mul.input)
            != {activation_producer.output[1], expected_scale_name}
            or list(scale_mul.output) != [combined_scale_name]
            or consumers.get(combined_scale_name, []) != [output_scale_mul]
        ):
            raise ValueError(
                f"{node.name}: activation and weight scales are not connected to "
                "the output dequantization Mul"
            )

        usage = dynamic_usage.setdefault(
            activation_producer.name,
            {
                "node": activation_producer,
                "integer_nodes": set(),
                "scale_nodes": set(),
            },
        )
        if usage["node"] is not activation_producer:
            raise ValueError(
                f"Duplicate DynamicQuantizeLinear node name {activation_producer.name!r}"
            )
        usage["integer_nodes"].add(node.name)
        usage["scale_nodes"].add(scale_mul.name)
        expected_integer_initializers.update((weight.name, zero_point.name))
        weight_audit.append(
            {
                "source_node": original_name,
                "source_activation": source_node.input[0],
                "source_weight": source_weight.name,
                "integer_node": node.name,
                "weight": node.input[1],
                "weight_shape": list(weight.dims),
                "weight_scale": scale.name,
                "weight_scale_value": float(scale_value),
                "weight_zero_point": zero_point.name,
                "dynamic_activation_node": activation_producer.name,
                "activation_scale": activation_producer.output[1],
                "activation_zero_point": activation_producer.output[2],
                "accumulator_cast": cast.name,
                "scale_combine_mul": scale_mul.name,
                "output_scale_mul": output_scale_mul.name,
                "restored_output": source_node.output[0],
            }
        )

    actual_dynamic_nodes = {
        node.name: node
        for node in model.graph.node
        if node.op_type == "DynamicQuantizeLinear"
    }
    if set(actual_dynamic_nodes) != set(dynamic_usage):
        raise ValueError(
            "DynamicQuantizeLinear coverage differs: "
            f"missing={sorted(set(dynamic_usage) - set(actual_dynamic_nodes))}, "
            f"unexpected={sorted(set(actual_dynamic_nodes) - set(dynamic_usage))}"
        )
    dynamic_activation_audit: list[dict[str, Any]] = []
    for name, usage in sorted(dynamic_usage.items()):
        dynamic_node = usage["node"]
        expected_integer_users = usage["integer_nodes"]
        expected_scale_users = usage["scale_nodes"]
        actual_quantized_users = {
            node.name for node in consumers.get(dynamic_node.output[0], [])
        }
        actual_scale_users = {
            node.name for node in consumers.get(dynamic_node.output[1], [])
        }
        actual_zero_point_users = {
            node.name for node in consumers.get(dynamic_node.output[2], [])
        }
        if (
            actual_quantized_users != expected_integer_users
            or actual_zero_point_users != expected_integer_users
            or actual_scale_users != expected_scale_users
        ):
            raise ValueError(
                f"{name}: dynamic activation outputs have unexpected users: "
                f"quantized={sorted(actual_quantized_users)}, "
                f"scale={sorted(actual_scale_users)}, "
                f"zero_point={sorted(actual_zero_point_users)}"
            )
        dynamic_activation_audit.append(
            {
                "node": name,
                "source_activation": dynamic_node.input[0],
                "quantized_output": dynamic_node.output[0],
                "scale_output": dynamic_node.output[1],
                "zero_point_output": dynamic_node.output[2],
                "matmul_integer_users": sorted(expected_integer_users),
                "scale_combine_users": sorted(expected_scale_users),
            }
        )

    unexpected_quantized_nodes = [
        f"{node.domain or 'ai.onnx'}::{node.op_type}:{node.name}"
        for node in model.graph.node
        if (
            node.op_type in {"QuantizeLinear", "DequantizeLinear"}
            or node.op_type.startswith("QLinear")
            or node.op_type.endswith("Integer")
        )
        and node.op_type != "MatMulInteger"
    ]
    if unexpected_quantized_nodes:
        raise ValueError(
            f"Unexpected quantized operators in output: {unexpected_quantized_nodes}"
        )
    integer_initializers = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_type in (TensorProto.INT8, TensorProto.UINT8)
    }
    if integer_initializers != expected_integer_initializers:
        raise ValueError(
            "INT8/UINT8 initializer coverage differs: "
            f"missing={sorted(expected_integer_initializers - integer_initializers)}, "
            f"unexpected={sorted(integer_initializers - expected_integer_initializers)}"
        )

    # ORT's quantization pre-pass is allowed to normalize an unselected Gemm
    # into FP32 MatMul+Add.  Audit preservation by the original FP32 weight,
    # rather than requiring the source node name/op_type to survive verbatim.
    preserved_audit: list[dict[str, Any]] = []
    changed_preserved = {}
    for item in preserved_linears:
        candidates = [
            node
            for node in model.graph.node
            if node.op_type in {"MatMul", "Gemm"} and item["weight"] in node.input
        ]
        quantized_weight_name = f"{item['weight']}_quantized"
        integer_users = [
            node.name
            for node in model.graph.node
            if node.op_type == "MatMulInteger" and quantized_weight_name in node.input
        ]
        if len(candidates) != 1 or integer_users or quantized_weight_name in initializers:
            changed_preserved[item["name"]] = {
                "fp32_candidates": [node.name for node in candidates],
                "integer_users": integer_users,
                "quantized_weight_initializer_present": (
                    quantized_weight_name in initializers
                ),
            }
        else:
            preserved_audit.append(
                {
                    "source_node": item["name"],
                    "source_op_type": item["op_type"],
                    "weight": item["weight"],
                    "output_node": candidates[0].name,
                    "output_op_type": candidates[0].op_type,
                }
            )
    if changed_preserved:
        raise ValueError(f"FP32 preserved linears changed: {changed_preserved}")

    source_convs = [node for node in source.graph.node if node.op_type == "Conv"]
    changed_convs = {
        node.name: (
            None
            if node.name not in nodes_by_name
            else {
                "op_type": nodes_by_name[node.name].op_type,
                "inputs": list(nodes_by_name[node.name].input),
            }
        )
        for node in source_convs
        if node.name not in nodes_by_name
        or nodes_by_name[node.name].op_type != "Conv"
        or list(nodes_by_name[node.name].input) != list(node.input)
    }
    integer_convs = [
        node.name
        for node in model.graph.node
        if node.op_type in {"ConvInteger", "QLinearConv"}
    ]
    if changed_convs or integer_convs:
        raise ValueError(
            f"FP32 Conv preservation failed: changed={changed_convs}, "
            f"integer={integer_convs}"
        )

    before_counts = Counter(node.op_type for node in source.graph.node)
    after_counts = Counter(node.op_type for node in model.graph.node)
    integer_delta = after_counts["MatMulInteger"] - before_counts["MatMulInteger"]
    if integer_delta != len(selected_nodes):
        raise ValueError(
            f"Expected {len(selected_nodes)} new MatMulInteger, found {integer_delta}"
        )
    return {
        "onnx_checker_full_check": checker_status,
        "actual_matmul_integer_count": len(actual_integer_names),
        "expected_matmul_integer_count": len(selected_nodes),
        "actual_matmul_integer_names": sorted(actual_integer_names),
        "dynamic_qint8_weight_audit": weight_audit,
        "dynamic_uint8_activation_count": len(dynamic_activation_audit),
        "dynamic_uint8_activation_audit": dynamic_activation_audit,
        "fp32_preserved_linear_count": len(preserved_linears),
        "fp32_preserved_linear_audit": preserved_audit,
        "fp32_preserved_conv_count": len(source_convs),
        "fp32_preserved_conv_names": [node.name for node in source_convs],
        "graph_outputs": list(quantized_outputs),
        "op_counts": op_counts(model),
    }


def load_validation_arrays(
    path: Path,
    sample_count: int,
    spatial_geometry: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    channels, height, width = spatial_geometry
    with np.load(path) as data:
        if "input_spatial" in data and "input_global" in data:
            spatial = np.asarray(data["input_spatial"][:sample_count], dtype=np.float32)
            global_input = np.asarray(data["input_global"][:sample_count], dtype=np.float32)
        elif "binaryInputNCHWPacked" in data and "globalInputNC" in data:
            packed = np.asarray(data["binaryInputNCHWPacked"][:sample_count])
            global_input = np.asarray(data["globalInputNC"][:sample_count], dtype=np.float32)
            if packed.ndim != 3 or packed.shape[1] < channels:
                raise ValueError(
                    f"Packed spatial shape {packed.shape} cannot supply "
                    f"{channels} channels"
                )
            spatial = np.unpackbits(packed[:, :channels, :], axis=2)
            if spatial.shape[2] < height * width:
                raise ValueError(
                    f"Packed spatial data has {spatial.shape[2]} bits/channel, "
                    f"model needs {height * width}"
                )
            spatial = spatial[:, :, : height * width]
            spatial = spatial.reshape(-1, channels, height, width).astype(np.float32)
        else:
            raise ValueError(
                "Validation NPZ needs input_spatial/input_global or "
                "binaryInputNCHWPacked/globalInputNC"
            )
    if len(spatial) != sample_count or len(global_input) != sample_count:
        raise ValueError(
            f"Requested {sample_count} validation samples, loaded "
            f"spatial={len(spatial)}, global={len(global_input)}"
        )
    if tuple(spatial.shape[1:]) != spatial_geometry:
        raise ValueError(
            f"Validation spatial shape {tuple(spatial.shape[1:])} does not match "
            f"model geometry {spatial_geometry}"
        )
    return {
        "input_spatial": np.ascontiguousarray(spatial),
        "input_global": np.ascontiguousarray(global_input),
    }


def adapt_feed(session: Any, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        if meta.name not in arrays:
            raise ValueError(f"Validation data is missing model input {meta.name!r}")
        value = arrays[meta.name]
        expected = list(meta.shape)
        if value.ndim != len(expected):
            raise ValueError(f"{meta.name}: {value.shape} does not match {expected}")
        slices: list[slice] = []
        for axis, (actual, wanted) in enumerate(zip(value.shape, expected)):
            if not isinstance(wanted, int) or wanted == actual:
                slices.append(slice(None))
            elif meta.name == "input_global" and axis == 1 and 0 < wanted < actual:
                slices.append(slice(0, wanted))
            else:
                raise ValueError(
                    f"{meta.name}: cannot adapt actual shape {list(value.shape)} "
                    f"to model shape {expected}"
                )
        feed[meta.name] = np.ascontiguousarray(value[tuple(slices)])
    return feed


def validate_outputs(
    source: Path, quantized: Path, data: Path, sample_count: int, threads: int
) -> dict[str, Any]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    reference = ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    candidate = ort.InferenceSession(
        str(quantized), sess_options=options, providers=["CPUExecutionProvider"]
    )
    reference_names = tuple(meta.name for meta in reference.get_outputs())
    candidate_names = tuple(meta.name for meta in candidate.get_outputs())
    if reference_names != EXPECTED_OUTPUTS or candidate_names != EXPECTED_OUTPUTS:
        raise ValueError(
            f"Expected all five outputs {EXPECTED_OUTPUTS}; "
            f"source={reference_names}, int8={candidate_names}"
        )
    reference_inputs = {meta.name: meta for meta in reference.get_inputs()}
    candidate_inputs = {meta.name: meta for meta in candidate.get_inputs()}
    if set(reference_inputs) != set(candidate_inputs):
        raise ValueError(
            f"Source and INT8 input names differ: "
            f"source={sorted(reference_inputs)}, INT8={sorted(candidate_inputs)}"
        )
    spatial_meta = reference_inputs.get("input_spatial")
    if spatial_meta is None or len(spatial_meta.shape) != 4:
        raise ValueError("Model input_spatial must have rank 4")
    geometry_values = spatial_meta.shape[1:]
    if any(not isinstance(value, int) or value <= 0 for value in geometry_values):
        raise ValueError(
            f"Model must expose fixed spatial C/H/W for NPZ decoding, got "
            f"{spatial_meta.shape}"
        )
    spatial_geometry = tuple(int(value) for value in geometry_values)
    arrays = load_validation_arrays(data, sample_count, spatial_geometry)

    stats: dict[str, dict[str, Any]] = {
        name: {
            "count": 0,
            "sum_abs": 0.0,
            "sum_squared": 0.0,
            "sum_reference_squared": 0.0,
            "max_abs": 0.0,
            "shape": None,
            "dtype": None,
        }
        for name in EXPECTED_OUTPUTS
    }
    for sample_index in range(sample_count):
        one_sample = {
            name: np.ascontiguousarray(value[sample_index : sample_index + 1])
            for name, value in arrays.items()
        }
        reference_feed = adapt_feed(reference, one_sample)
        candidate_feed = adapt_feed(candidate, one_sample)
        if set(reference_feed) != set(candidate_feed):
            raise ValueError("Source and INT8 model input names differ")
        expected = reference.run(list(EXPECTED_OUTPUTS), reference_feed)
        actual = candidate.run(list(EXPECTED_OUTPUTS), candidate_feed)
        for name, expected_value, actual_value in zip(EXPECTED_OUTPUTS, expected, actual):
            expected_array = np.asarray(expected_value)
            actual_array = np.asarray(actual_value)
            if expected_array.shape != actual_array.shape:
                raise ValueError(
                    f"sample {sample_index} {name}: source shape "
                    f"{expected_array.shape} != INT8 {actual_array.shape}"
                )
            if not np.all(np.isfinite(expected_array)):
                raise ValueError(
                    f"FP32 output {name} sample {sample_index} contains non-finite values"
                )
            if not np.all(np.isfinite(actual_array)):
                raise ValueError(
                    f"INT8 output {name} sample {sample_index} contains non-finite values"
                )
            state = stats[name]
            shape = list(actual_array.shape)
            dtype = str(actual_array.dtype)
            if state["shape"] is None:
                state["shape"] = shape
                state["dtype"] = dtype
            elif state["shape"] != shape or state["dtype"] != dtype:
                raise ValueError(f"{name}: output signature changed between samples")
            delta = actual_array.astype(np.float64) - expected_array.astype(np.float64)
            reference64 = expected_array.astype(np.float64)
            absolute = np.abs(delta)
            state["count"] += int(delta.size)
            state["sum_abs"] += float(np.sum(absolute, dtype=np.float64))
            state["sum_squared"] += float(np.sum(delta * delta, dtype=np.float64))
            state["sum_reference_squared"] += float(
                np.sum(reference64 * reference64, dtype=np.float64)
            )
            state["max_abs"] = max(state["max_abs"], float(np.max(absolute)))

    comparisons: list[dict[str, Any]] = []
    for name in EXPECTED_OUTPUTS:
        state = stats[name]
        count = int(state["count"])
        rmse = float(np.sqrt(state["sum_squared"] / count))
        reference_rms = float(np.sqrt(state["sum_reference_squared"] / count))
        comparisons.append(
            {
                "output": name,
                "per_sample_shape": state["shape"],
                "dtype": state["dtype"],
                "finite": True,
                "element_count_across_samples": count,
                "max_abs": state["max_abs"],
                "mean_abs": state["sum_abs"] / count,
                "rmse": rmse,
                "normalized_rmse": rmse / max(reference_rms, np.finfo(np.float64).tiny),
            }
        )
    return {
        "data": str(data.resolve()),
        "data_sha256": sha256_file(data),
        "requested_samples": sample_count,
        "samples_checked": sample_count,
        "batch_per_run": 1,
        "spatial_geometry_from_model": list(spatial_geometry),
        "outputs": comparisons,
        "all_finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="FP32 ORT-CPU ONNX")
    parser.add_argument("--output", required=True, type=Path, help="INT8 ONNX output")
    parser.add_argument("--report", type=Path, help="JSON report (default: beside output)")
    parser.add_argument(
        "--data",
        type=Path,
        help="optional validation NPZ; never used to calibrate dynamic INT8",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=1,
        help="number of batch-1 samples checked when --data is supplied",
    )
    parser.add_argument(
        "--validation-threads",
        type=int,
        default=1,
        help="ORT intra-op threads used only by optional validation",
    )
    parser.add_argument(
        "--expected-blocks", type=int, help="optional exact block-count assertion"
    )
    parser.add_argument(
        "--expected-head-linears",
        type=int,
        help="optional exact FP32 policy/value-head linear-count assertion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path = prepare_output_path(args.output, "Output")
    report_path = prepare_output_path(
        args.report
        if args.report is not None
        else output_path.with_suffix(output_path.suffix + ".report.json"),
        "Report",
    )
    data_path = args.data.resolve() if args.data is not None else None
    if source_path == output_path or report_path in (source_path, output_path):
        raise ValueError("Input, output, and report must be distinct paths")
    if data_path is not None and data_path in {source_path, output_path, report_path}:
        raise ValueError("Validation data, input, output, and report must be distinct")
    if args.validation_samples <= 0 or args.validation_threads <= 0:
        raise ValueError("Validation sample/thread counts must be positive")
    if data_path is not None and not data_path.is_file():
        raise FileNotFoundError(data_path)

    source = onnx.load(str(source_path), load_external_data=False)
    reject_external_data(source, source_path)
    source_checker_status = checker_with_ort_schema(source)
    selections, discovery = discover_selection(
        source, args.expected_blocks, args.expected_head_linears
    )
    selected_nodes = [name for item in selections for name in item.physical_nodes]
    preserved = [
        item
        for group in ("fp32_linear_global", "fp32_policy_value_heads")
        for item in discovery["linear_inventory"][group]
    ]

    temporary_output = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.partial.onnx"
    )
    temporary_report: Path | None = None
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ort_cpu_dynamic_trunk_w8a8",
        "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "onnx_checker_full_check": source_checker_status,
            "op_counts": op_counts(source),
        },
        "quantization_contract": {
            "backend": "ONNX Runtime CPUExecutionProvider / MLAS",
            "activation": "dynamic uint8 via DynamicQuantizeLinear",
            "weight": "symmetric per-tensor QInt8",
            "operator": "MatMulInteger",
            "reduce_range": False,
            "MatMulConstBOnly": True,
            "scope": "transformer trunk projections only",
            "stem_global_all_heads": "FP32",
        },
        "discovery": discovery,
        "selected_nodes": selected_nodes,
    }
    try:
        report["onnxruntime_version"] = quantize(
            source_path, temporary_output, selected_nodes
        )
        report["quantized_graph"] = audit_quantized_graph(
            source, temporary_output, selected_nodes, preserved
        )
        if data_path is not None:
            report["five_output_validation"] = validate_outputs(
                source_path,
                temporary_output,
                data_path,
                args.validation_samples,
                args.validation_threads,
            )
        report["output"] = {
            "path": str(output_path),
            "bytes": temporary_output.stat().st_size,
            "sha256": sha256_file(temporary_output),
            "atomic_replace": True,
        }
        temporary_report = stage_json(report_path, report)
        os.replace(temporary_output, output_path)
        os.replace(temporary_report, report_path)
    finally:
        temporary_output.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "blocks": discovery["block_count"],
                "logical_projections": discovery["logical_projections_covered"],
                "physical_matmul_integer": len(selected_nodes),
                "layout_counts": discovery["layout_counts"],
                "five_output_validation": args.data is not None,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
