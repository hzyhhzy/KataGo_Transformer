"""Strict RMSNorm and residual normalization fusions for ORT CPU graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, defs, helper, numpy_helper

from .geometry import TransformerGeometry


DEFAULT_NORM_KINDS = ("norm1", "norm2")
_CORE_SUFFIXES = ("Mul", "ReduceMean", "Add", "Sqrt", "Div", "Mul_1")


def _replace_repeated(container: Any, values: Iterable[Any]) -> None:
    del container[:]
    container.extend(values)


def op_counts(model: onnx.ModelProto) -> dict[str, int]:
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def qualified_op_counts(model: onnx.ModelProto) -> dict[str, int]:
    counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node
    )
    return dict(sorted(counts.items()))


def retain_source_shape_information(
    transformed: onnx.ModelProto, inferred_source: onnx.ModelProto
) -> int:
    """Retain source types for names that survive contrib-op rewrites."""
    surviving_values = {
        output_name
        for node in transformed.graph.node
        for output_name in node.output
        if output_name
    }
    retained = [
        value_info
        for value_info in inferred_source.graph.value_info
        if value_info.name in surviving_values
    ]
    _replace_repeated(transformed.graph.value_info, retained)
    return len(retained)


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _constant_array(node: onnx.NodeProto) -> np.ndarray:
    if node.op_type != "Constant":
        raise ValueError(f"Expected Constant at {node.name!r}, got {node.op_type}")
    value = _attribute(node, "value")
    if value is None:
        raise ValueError(f"Constant {node.name!r} does not contain a tensor value")
    return numpy_helper.to_array(value)


def _require_node(
    nodes_by_name: dict[str, onnx.NodeProto], name: str, op_type: str
) -> onnx.NodeProto:
    try:
        node = nodes_by_name[name]
    except KeyError as exc:
        raise ValueError(f"Missing expected node {name!r}") from exc
    if node.op_type != op_type:
        raise ValueError(f"Expected {name!r} to be {op_type}, got {node.op_type}")
    return node


def _require_inputs(node: onnx.NodeProto, expected: list[str]) -> None:
    if list(node.input) != expected:
        raise ValueError(
            f"Unexpected inputs for {node.name!r}: got {list(node.input)}, "
            f"expected {expected}"
        )


def _verify_private_intermediate(
    consumers: dict[str, list[str]], value_name: str, expected_consumer: str
) -> None:
    actual = consumers.get(value_name, [])
    if actual != [expected_consumer]:
        raise ValueError(
            f"Refusing to remove shared/interleaved value {value_name!r}: "
            f"consumers={actual}, expected={[expected_consumer]}"
        )


def _value_array(
    value_name: str,
    *,
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
) -> np.ndarray:
    initializer = initializers.get(value_name)
    if initializer is not None:
        return numpy_helper.to_array(initializer)
    producer = producers.get(value_name)
    if producer is not None and producer.op_type == "Constant":
        return _constant_array(producer)
    raise ValueError(f"Expected constant tensor for value {value_name!r}")


def _other_input(node: onnx.NodeProto, known_value: str) -> str:
    if len(node.input) != 2 or list(node.input).count(known_value) != 1:
        raise ValueError(
            f"Expected exactly one {known_value!r} input at {node.name!r}, "
            f"got {list(node.input)}"
        )
    return node.input[0] if node.input[1] == known_value else node.input[1]


def _match_one_norm(
    *,
    block_index: int,
    norm_kind: str,
    nodes_by_name: dict[str, onnx.NodeProto],
    node_indices: dict[str, int],
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[str]],
    expected_width: int,
) -> tuple[int, set[str], onnx.NodeProto, onnx.TensorProto | None, dict[str, Any]]:
    prefix = f"/model/blocks.{block_index}/{norm_kind}"
    names = {suffix: f"{prefix}/{suffix}" for suffix in _CORE_SUFFIXES}

    square = _require_node(nodes_by_name, names["Mul"], "Mul")
    reduce = _require_node(nodes_by_name, names["ReduceMean"], "ReduceMean")
    add = _require_node(nodes_by_name, names["Add"], "Add")
    sqrt = _require_node(nodes_by_name, names["Sqrt"], "Sqrt")
    div = _require_node(nodes_by_name, names["Div"], "Div")
    normalize = _require_node(nodes_by_name, names["Mul_1"], "Mul")

    if len(square.input) != 2 or square.input[0] != square.input[1]:
        raise ValueError(f"Unexpected square inputs at {square.name!r}: {list(square.input)}")
    x_name = square.input[0]
    cast = nodes_by_name.get(f"{prefix}/Cast")
    if cast is not None:
        if (
            cast.op_type != "Cast"
            or len(cast.input) != 1
            or list(cast.output) != [x_name]
            or _attribute(cast, "to") != TensorProto.FLOAT
        ):
            raise ValueError(f"Unexpected float Cast structure at {cast.name!r}")
    _require_inputs(square, [x_name, x_name])
    if len(square.output) != 1:
        raise ValueError(f"Expected one square output at {square.name!r}")

    if len(reduce.input) != 2 or reduce.input[0] != square.output[0]:
        raise ValueError(f"Unexpected ReduceMean inputs at {reduce.name!r}: {list(reduce.input)}")
    axes_name = reduce.input[1]
    axes_value = np.asarray(
        _value_array(axes_name, initializers=initializers, producers=producers),
        dtype=np.int64,
    ).reshape(-1)
    if not np.array_equal(axes_value, np.asarray([-1], dtype=np.int64)):
        raise ValueError(f"Unexpected RMS axes at {reduce.name!r}: {axes_value.tolist()}")
    if _attribute(reduce, "keepdims", 1) != 1 or len(reduce.output) != 1:
        raise ValueError(f"Unexpected ReduceMean attributes at {reduce.name!r}")

    epsilon_name = _other_input(add, reduce.output[0])
    epsilon_array = np.asarray(
        _value_array(epsilon_name, initializers=initializers, producers=producers),
        dtype=np.float64,
    ).reshape(-1)
    if epsilon_array.size != 1:
        raise ValueError(f"Non-scalar epsilon value {epsilon_name!r}")
    epsilon = float(epsilon_array[0])
    if not np.isclose(epsilon, 1.0e-6, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Unexpected RMS epsilon {epsilon_name!r}: {epsilon}")
    _require_inputs(sqrt, [add.output[0]])

    if len(div.input) != 2 or div.input[1] != sqrt.output[0]:
        raise ValueError(f"Unexpected reciprocal Div at {div.name!r}: {list(div.input)}")
    one_name = div.input[0]
    one_array = np.asarray(
        _value_array(one_name, initializers=initializers, producers=producers),
        dtype=np.float64,
    ).reshape(-1)
    if one_array.size != 1 or float(one_array[0]) != 1.0:
        raise ValueError(f"Unexpected reciprocal numerator {one_name!r}: {one_array}")
    if len(normalize.input) != 2 or set(normalize.input) != {x_name, div.output[0]}:
        raise ValueError(f"Unexpected normalize Mul at {normalize.name!r}: {list(normalize.input)}")

    normalized_name = normalize.output[0]
    scale_mul = nodes_by_name.get(f"{prefix}/Mul_2")
    added_gamma: onnx.TensorProto | None = None
    if scale_mul is not None:
        if scale_mul.op_type != "Mul" or len(scale_mul.output) != 1:
            raise ValueError(f"Unexpected scale Mul at {scale_mul.name!r}")
        gamma_name = _other_input(scale_mul, normalized_name)
        if gamma_name not in initializers:
            raise ValueError(
                f"Gamma {gamma_name!r} at {scale_mul.name!r} is not an initializer"
            )
        gamma_array = numpy_helper.to_array(initializers[gamma_name])
        output_name = scale_mul.output[0]
        gamma_source = "initializer"
    else:
        gamma_name = f"flash_unit_gamma_b{block_index}_{norm_kind}"
        if gamma_name in initializers:
            raise ValueError(f"Generated gamma name already exists: {gamma_name}")
        gamma_array = np.ones((expected_width,), dtype=np.float32)
        added_gamma = numpy_helper.from_array(gamma_array, name=gamma_name)
        output_name = normalized_name
        gamma_source = "materialized_unit_gamma"

    if gamma_array.shape != (expected_width,) or gamma_array.dtype != np.float32:
        raise ValueError(
            f"Unexpected gamma layout for {gamma_name!r}: "
            f"shape={gamma_array.shape}, dtype={gamma_array.dtype}; "
            f"expected shape={(expected_width,)}, dtype=float32"
        )

    chain = [square, reduce, add, sqrt, div, normalize]
    if scale_mul is not None:
        chain.append(scale_mul)
    direct_private_edges = (
        (square.output[0], reduce.name),
        (reduce.output[0], add.name),
        (add.output[0], sqrt.name),
        (sqrt.output[0], div.name),
        (div.output[0], normalize.name),
    )
    for value_name, expected_consumer in direct_private_edges:
        _verify_private_intermediate(consumers, value_name, expected_consumer)
    if scale_mul is not None:
        _verify_private_intermediate(consumers, normalized_name, scale_mul.name)

    removable_constant_nodes: list[onnx.NodeProto] = []
    for value_name, expected_consumer in (
        (axes_name, reduce.name),
        (epsilon_name, add.name),
        (one_name, div.name),
    ):
        producer = producers.get(value_name)
        if producer is not None and producer.op_type == "Constant":
            _verify_private_intermediate(consumers, value_name, expected_consumer)
            removable_constant_nodes.append(producer)

    fused_name = f"{prefix}/SimplifiedLayerNormalization"
    fused = helper.make_node(
        "SimplifiedLayerNormalization",
        [x_name, gamma_name],
        [output_name],
        name=fused_name,
        axis=-1,
        epsilon=epsilon,
        stash_type=TensorProto.FLOAT,
    )
    removed_names = {node.name for node in (*chain, *removable_constant_nodes)}
    details = {
        "block": block_index,
        "norm": norm_kind,
        "input": x_name,
        "gamma": gamma_name,
        "gamma_shape": list(gamma_array.shape),
        "gamma_source": gamma_source,
        "epsilon": epsilon,
        "axis": -1,
        "output": output_name,
        "removed_nodes": sorted(removed_names, key=node_indices.__getitem__),
        "replacement_node": fused_name,
    }
    return node_indices[square.name], removed_names, fused, added_gamma, details


def fuse_rmsnorm(
    model: onnx.ModelProto,
    geometry: TransformerGeometry,
    norm_kinds: tuple[str, ...],
) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    old_nodes = list(model.graph.node)
    nodes_by_name = {node.name: node for node in old_nodes}
    if len(nodes_by_name) != len(old_nodes):
        raise ValueError("Graph contains duplicate node names")
    node_indices = {node.name: index for index, node in enumerate(old_nodes)}
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producers = {
        output_name: node
        for node in old_nodes
        for output_name in node.output
        if output_name
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in old_nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node.name)

    insertions: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    nodes_to_remove: set[str] = set()
    details: list[dict[str, Any]] = []
    added_initializers: list[onnx.TensorProto] = []
    for block_index in range(geometry.blocks):
        for norm_kind in norm_kinds:
            expected_width = (
                geometry.hidden_size if norm_kind in ("norm1", "norm2") else geometry.head_size
            )
            insertion_index, removed, fused, added_gamma, one_detail = _match_one_norm(
                block_index=block_index,
                norm_kind=norm_kind,
                nodes_by_name=nodes_by_name,
                node_indices=node_indices,
                initializers=initializers,
                producers=producers,
                consumers=consumers,
                expected_width=expected_width,
            )
            overlap = nodes_to_remove.intersection(removed)
            if overlap:
                raise ValueError(f"Nodes selected by multiple RMS fusions: {sorted(overlap)}")
            insertions[insertion_index].append(fused)
            nodes_to_remove.update(removed)
            if added_gamma is not None:
                added_initializers.append(added_gamma)
                initializers[added_gamma.name] = added_gamma
            details.append(one_detail)

    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(old_nodes):
        rewritten.extend(insertions.get(index, ()))
        if node.name not in nodes_to_remove:
            rewritten.append(node)
    _replace_repeated(model.graph.node, rewritten)
    model.graph.initializer.extend(added_initializers)
    return model, details


def _register_checker_schema() -> bool:
    try:
        defs.get_schema("SimplifiedLayerNormalization", 20, "")
        return False
    except onnx.onnx_cpp2py_export.defs.SchemaError:
        pass

    fp_types = ["tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"]
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
            ("T", fp_types, "input floating-point type"),
            ("V", fp_types, "scale/output floating-point type"),
            ("U", ["tensor(float)", "tensor(double)"], "accumulator type"),
        ],
        attributes=[
            defs.OpSchema.Attribute(
                "axis", defs.OpSchema.AttrType.INT, "normalization axis", required=False
            ),
            defs.OpSchema.Attribute(
                "epsilon", defs.OpSchema.AttrType.FLOAT, "numerical epsilon", required=False
            ),
            defs.OpSchema.Attribute(
                "stash_type", defs.OpSchema.AttrType.INT, "accumulator type", required=False
            ),
        ],
    )
    defs.register_schema(schema)
    return True


def checker_with_ort_schema(model: onnx.ModelProto) -> str:
    registered = _register_checker_schema()
    try:
        onnx.checker.check_model(model, full_check=True)
    finally:
        if registered:
            defs.deregister_schema("SimplifiedLayerNormalization", 1, "")
    return "onnx.checker full_check passed with ORT SimplifiedLayerNormalization schema shim"


def _make_inputs(
    npz_path: Path,
    global_features: int,
    spatial_shape: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        packed = np.asarray(data["binaryInputNCHWPacked"][:1])
        global_input = np.asarray(data["globalInputNC"][:1, :global_features], dtype=np.float32)
    batch, channels, height, width = spatial_shape
    if batch != 1:
        raise ValueError(f"First-sample smoke expects fixed batch 1, got {spatial_shape}")
    spatial = np.unpackbits(packed, axis=2)[:, :channels, : height * width]
    spatial = spatial.reshape(batch, channels, height, width).astype(np.float32)
    return {
        "input_spatial": np.ascontiguousarray(spatial),
        "input_global": np.ascontiguousarray(global_input),
    }


def compare_outputs(
    reference_path: Path, candidate_path: Path, npz_path: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    reference = ort.InferenceSession(
        str(reference_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    candidate = ort.InferenceSession(
        str(candidate_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    reference_outputs = [output.name for output in reference.get_outputs()]
    candidate_outputs = [output.name for output in candidate.get_outputs()]
    if reference_outputs != candidate_outputs:
        raise ValueError(
            f"Output names changed: reference={reference_outputs}, candidate={candidate_outputs}"
        )
    global_input = next(value for value in reference.get_inputs() if value.name == "input_global")
    if len(global_input.shape) != 2 or not isinstance(global_input.shape[1], int):
        raise ValueError(f"Expected fixed input_global shape, got {global_input.shape}")
    spatial_input = next(value for value in reference.get_inputs() if value.name == "input_spatial")
    if len(spatial_input.shape) != 4 or not all(
        isinstance(dim, int) for dim in spatial_input.shape
    ):
        raise ValueError(f"Expected fixed input_spatial shape, got {spatial_input.shape}")
    feeds = _make_inputs(
        npz_path,
        int(global_input.shape[1]),
        tuple(int(dim) for dim in spatial_input.shape),
    )
    ref_values = reference.run(reference_outputs, feeds)
    cand_values = candidate.run(candidate_outputs, feeds)
    comparisons: list[dict[str, Any]] = []
    for name, reference_value, candidate_value in zip(
        reference_outputs, ref_values, cand_values, strict=True
    ):
        difference = candidate_value.astype(np.float64) - reference_value.astype(np.float64)
        comparisons.append(
            {
                "output": name,
                "shape": list(reference_value.shape),
                "max_abs": float(np.max(np.abs(difference))),
                "mean_abs": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "allclose_rtol_1e-5_atol_1e-6": bool(
                    np.allclose(candidate_value, reference_value, rtol=1.0e-5, atol=1.0e-6)
                ),
            }
        )
    return comparisons, reference_outputs


def qkn_nodes(model: onnx.ModelProto) -> list[dict[str, Any]]:
    return [
        {
            "name": node.name,
            "domain": node.domain or "ai.onnx",
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for node in model.graph.node
        if node.op_type == "SimplifiedLayerNormalization"
        and ("/q_norm/" in node.name or "/k_norm/" in node.name)
    ]


def _value_metadata(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape: list[int | str | None] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        result[value.name] = {"elem_type": int(tensor_type.elem_type), "shape": shape}
    return result


def _require_residual_shape(
    metadata: dict[str, dict[str, Any]],
    names: list[str],
    context: str,
    expected_hidden_size: int,
) -> list[list[int | str | None]]:
    shapes: list[list[int | str | None]] = []
    for name in names:
        if name not in metadata:
            raise ValueError(f"Missing inferred type/shape for {name!r} in {context}")
        one = metadata[name]
        if one["elem_type"] != TensorProto.FLOAT:
            raise ValueError(f"Expected float32 {name!r} in {context}, got {one}")
        shape = one["shape"]
        if len(shape) != 3 or shape[-1] != expected_hidden_size:
            raise ValueError(
                "SkipSimplifiedLayerNormalization requires "
                f"[B,S,{expected_hidden_size}] at {context}; "
                f"got {name!r} shape={shape}"
            )
        shapes.append(shape)
    if not all(shape == shapes[0] for shape in shapes[1:]):
        raise ValueError(f"Residual shapes differ at {context}: {dict(zip(names, shapes))}")
    return shapes


def _ensure_ms_opset(model: onnx.ModelProto) -> None:
    imports = [item for item in model.opset_import if item.domain == "com.microsoft"]
    if not imports:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
        return
    if len(imports) != 1 or imports[0].version < 1:
        raise ValueError(f"Unexpected com.microsoft opset imports: {imports}")


def _trace_private_layout_to_add(
    *,
    norm_input: str,
    cast_name: str,
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[str]],
    metadata: dict[str, dict[str, Any]],
) -> tuple[onnx.NodeProto, list[onnx.NodeProto]]:
    """Trace an optional fixed-shape NCHW round trip back to a residual Add."""
    cursor = norm_input
    expected_consumer = cast_name
    backwards: list[onnx.NodeProto] = []
    while True:
        try:
            producer = producers[cursor]
        except KeyError as exc:
            raise ValueError(f"Norm input {cursor!r} has no producer") from exc
        if producer.op_type == "Add" and producer.domain in ("", "ai.onnx"):
            add = producer
            break
        if producer.op_type not in {"Transpose", "Reshape"} or producer.domain not in (
            "",
            "ai.onnx",
        ):
            raise ValueError(
                f"Expected residual Add through only layout nodes, got "
                f"{producer.domain!r}::{producer.op_type} {producer.name!r}"
            )
        if consumers.get(cursor, []) != [expected_consumer]:
            raise ValueError(
                f"Layout value {cursor!r} is not private: consumers="
                f"{consumers.get(cursor, [])}, expected={[expected_consumer]}"
            )
        if not producer.input:
            raise ValueError(f"Layout node {producer.name!r} has no data input")
        backwards.append(producer)
        expected_consumer = producer.name
        cursor = producer.input[0]
        if len(backwards) > 4:
            raise ValueError(f"Unexpectedly long layout chain before {cast_name!r}")

    layout_nodes = list(reversed(backwards))
    forward_types = [node.op_type for node in layout_nodes]
    if forward_types not in ([], ["Transpose", "Reshape", "Reshape", "Transpose"]):
        raise ValueError(
            f"Unexpected residual layout chain before {cast_name!r}: {forward_types}"
        )
    if layout_nodes:
        transposes = [node for node in layout_nodes if node.op_type == "Transpose"]
        perms = [_attribute(node, "perm") for node in transposes]
        if perms != [[0, 2, 1], [0, 2, 1]]:
            raise ValueError(
                f"Block-boundary Transpose chain is not self-inverse before "
                f"{cast_name!r}: perms={perms}"
            )
        add_shape = metadata.get(add.output[0], {}).get("shape")
        norm_shape = metadata.get(norm_input, {}).get("shape")
        if add_shape != norm_shape:
            raise ValueError(
                f"Layout round trip changes endpoint shape before {cast_name!r}: "
                f"add={add_shape}, norm={norm_shape}"
            )
    return add, layout_nodes


def fuse_skip_sln(
    model: onnx.ModelProto,
    inferred_source: onnx.ModelProto,
    geometry: TransformerGeometry,
) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    """Fuse every eligible residual Add + already-fused SLN pair."""
    old_nodes = list(model.graph.node)
    nodes_by_name = {node.name: node for node in old_nodes}
    if len(nodes_by_name) != len(old_nodes):
        raise ValueError("Graph contains duplicate node names")
    node_indices = {node.name: index for index, node in enumerate(old_nodes)}
    consumers: dict[str, list[str]] = defaultdict(list)
    producers: dict[str, onnx.NodeProto] = {}
    for node in old_nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node.name)
        for output_name in node.output:
            if output_name:
                if output_name in producers:
                    raise ValueError(f"Duplicate producer for {output_name!r}")
                producers[output_name] = node

    metadata = _value_metadata(inferred_source)
    targets = [
        (block, "norm2", "attention_residual") for block in range(geometry.blocks)
    ] + [
        (block, "norm1", "previous_ffn_residual")
        for block in range(1, geometry.blocks)
    ]

    insertions: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    nodes_to_remove: set[str] = set()
    selected_adds: set[str] = set()
    details: list[dict[str, Any]] = []
    for block_index, norm_kind, residual_kind in targets:
        prefix = f"/model/blocks.{block_index}/{norm_kind}"
        sln_name = f"{prefix}/SimplifiedLayerNormalization"
        cast_name = f"{prefix}/Cast"
        try:
            sln = nodes_by_name[sln_name]
        except KeyError as exc:
            raise ValueError(f"Missing source SLN {sln_name!r}") from exc
        cast = nodes_by_name.get(cast_name)
        if sln.op_type != "SimplifiedLayerNormalization" or sln.domain not in (
            "",
            "ai.onnx",
        ):
            raise ValueError(f"Unexpected source SLN at {sln_name!r}")
        if len(sln.input) != 2 or len(sln.output) != 1:
            raise ValueError(f"Unexpected SLN arity at {sln_name!r}")
        if _attribute(sln, "axis") != -1 or _attribute(sln, "stash_type") != 1:
            raise ValueError(f"Unexpected SLN attributes at {sln_name!r}")
        epsilon = float(_attribute(sln, "epsilon"))
        if cast is not None:
            if len(cast.input) != 1 or len(cast.output) != 1:
                raise ValueError(f"Unexpected Cast arity at {cast_name!r}")
            if _attribute(cast, "to") != TensorProto.FLOAT:
                raise ValueError(f"Expected float Cast at {cast_name!r}")
            if sln.input[0] != cast.output[0]:
                raise ValueError(f"SLN does not consume expected Cast at {sln_name!r}")
            residual_sum_name = cast.input[0]
            residual_output_name = cast.output[0]
            endpoint_name = cast.name
            residual_consumers = set(consumers.get(residual_output_name, []))
        else:
            residual_sum_name = sln.input[0]
            residual_output_name = sln.input[0]
            endpoint_name = sln.name
            residual_consumers = set(consumers.get(residual_output_name, []))
        if sln_name not in residual_consumers:
            raise ValueError(f"Residual value is not consumed by SLN at {sln_name!r}")
        downstream_residual_consumers = residual_consumers - {sln_name}
        if not downstream_residual_consumers:
            raise ValueError(f"No downstream residual consumer for {residual_output_name!r}")

        add, layout_nodes = _trace_private_layout_to_add(
            norm_input=residual_sum_name,
            cast_name=endpoint_name,
            producers=producers,
            consumers=consumers,
            metadata=metadata,
        )
        if len(add.input) != 2 or len(add.output) != 1:
            raise ValueError(f"Unexpected residual Add at {add.name!r}")
        if add.name in selected_adds:
            raise ValueError(f"Residual Add selected twice: {add.name!r}")
        selected_adds.add(add.name)

        add_output_name = add.output[0]
        shapes = _require_residual_shape(
            metadata,
            list(
                dict.fromkeys(
                    [
                        add.input[0],
                        add.input[1],
                        add_output_name,
                        residual_sum_name,
                        residual_output_name,
                    ]
                )
            ),
            prefix,
            geometry.hidden_size,
        )
        first_after_add = layout_nodes[0].name if layout_nodes else endpoint_name
        expected_add_consumers = {first_after_add}
        if add_output_name == residual_output_name:
            expected_add_consumers.update(downstream_residual_consumers)
        if set(consumers.get(add_output_name, [])) != expected_add_consumers:
            raise ValueError(
                f"Residual Add output {add_output_name!r} is not private: "
                f"{consumers.get(add_output_name, [])}; expected "
                f"{sorted(expected_add_consumers)}"
            )

        gamma_name = sln.input[1]
        normalized_output = sln.output[0]
        fused_name = f"{prefix}/SkipSimplifiedLayerNormalization"
        fused = helper.make_node(
            "SkipSimplifiedLayerNormalization",
            [add.input[0], add.input[1], gamma_name],
            [normalized_output, "", "", residual_output_name],
            name=fused_name,
            domain="com.microsoft",
            epsilon=epsilon,
        )

        removal = {add.name, sln.name}
        if cast is not None:
            removal.add(cast.name)
        removal.update(node.name for node in layout_nodes)
        overlap = nodes_to_remove.intersection(removal)
        if overlap:
            raise ValueError(f"Nodes selected twice: {sorted(overlap)}")
        nodes_to_remove.update(removal)
        insertions[node_indices[add.name]].append(fused)
        details.append(
            {
                "block": block_index,
                "norm": norm_kind,
                "residual_kind": residual_kind,
                "removed_add": add.name,
                "removed_layout_nodes": [node.name for node in layout_nodes],
                "removed_cast": cast.name if cast is not None else None,
                "removed_sln": sln.name,
                "add_inputs": list(add.input),
                "normalized_output0": normalized_output,
                "residual_sum_output3": residual_output_name,
                "downstream_residual_consumers": sorted(downstream_residual_consumers),
                "gamma": gamma_name,
                "epsilon": epsilon,
                "shape": shapes[0],
                "replacement_node": fused_name,
            }
        )

    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(old_nodes):
        rewritten.extend(insertions.get(index, ()))
        if node.name not in nodes_to_remove:
            rewritten.append(node)
    _replace_repeated(model.graph.node, rewritten)
    _ensure_ms_opset(model)
    return model, details
