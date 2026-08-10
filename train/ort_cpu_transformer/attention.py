"""Strict KataGo attention/RoPE rewrite for ONNX Runtime CPU.

Only the production conversion route is kept here: shared-frequency ``tfrs``
uses two RotaryEmbedding nodes per block, while learned-frequency ``tflrs``
uses one node per Q/K head (the numerically validated candidate-B mapping).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from .geometry import TransformerGeometry, discover_transformer_geometry


def _constant_array(node: onnx.NodeProto) -> np.ndarray:
    if node.op_type != "Constant":
        raise ValueError(f"Expected Constant, got {node.op_type}: {node.name}")
    for attribute in node.attribute:
        if attribute.name == "value":
            return numpy_helper.to_array(attribute.t)
    raise ValueError(f"Constant node has no tensor value: {node.name}")


def _cache_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _extract_fixed_rope_caches(
    model: onnx.ModelProto, geometry: TransformerGeometry
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    nodes = list(model.graph.node)
    constants: dict[str, np.ndarray] = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in nodes:
        if node.op_type == "Constant" and len(node.output) == 1:
            constants[node.output[0]] = _constant_array(node)
    by_name = {node.name: node for node in nodes}

    def cache_input(block: int, mul_suffix: str, legacy_suffix: str) -> np.ndarray:
        prefix = f"/model/blocks.{block}/"
        mul = by_name.get(prefix + mul_suffix)
        if mul is not None:
            matches = [constants[value] for value in mul.input if value in constants]
            if len(matches) == 1:
                return matches[0]
        legacy_name = prefix + legacy_suffix + "_output_0"
        if legacy_name in constants:
            return constants[legacy_name]
        raise ValueError(
            f"Could not find block {block} fixed-RoPE cache feeding {mul_suffix}"
        )

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for block in range(geometry.blocks):
        cos_full = cache_input(block, "Mul", "Constant_4").reshape(
            geometry.sequence_length, geometry.head_size
        )
        sin_full = cache_input(block, "Mul_1", "Constant_5").reshape(
            geometry.sequence_length, geometry.head_size
        )
        if not np.array_equal(cos_full[:, 0::2], cos_full[:, 1::2]):
            raise ValueError(f"Block {block} cosine cache is not pair duplicated")
        if not np.array_equal(sin_full[:, 0::2], sin_full[:, 1::2]):
            raise ValueError(f"Block {block} sine cache is not pair duplicated")
        result[block] = (
            np.ascontiguousarray(cos_full[:, 0::2], dtype=np.float32),
            np.ascontiguousarray(sin_full[:, 0::2], dtype=np.float32),
        )
    return result


def _extract_learned_rope_caches(
    model: onnx.ModelProto, geometry: TransformerGeometry
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    initializer_arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    extractor: onnx.utils.Extractor | None = None
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    expected_shape = (
        geometry.sequence_length,
        geometry.num_heads,
        geometry.head_size // 2,
    )
    for block in range(geometry.blocks):
        prefix = f"/model/blocks.{block}/"
        folded_names = [
            prefix + "Unsqueeze_2_output_0",
            prefix + "Unsqueeze_3_output_0",
        ]
        if all(name in initializer_arrays for name in folded_names):
            cos, sin = (initializer_arrays[name] for name in folded_names)
            if cos.shape[0] != 1 or sin.shape[0] != 1:
                raise ValueError(f"Block {block} folded learned cache is not batch-broadcast")
            cos, sin = cos[0], sin[0]
        else:
            if extractor is None:
                inferred = onnx.shape_inference.infer_shapes(
                    model, check_type=True, strict_mode=True, data_prop=False
                )
                extractor = onnx.utils.Extractor(inferred)
            output_names = [prefix + "Cos_output_0", prefix + "Sin_output_0"]
            constant_model = extractor.extract_model([], output_names)
            cos, sin = ReferenceEvaluator(constant_model).run(None, {})
        cos = np.ascontiguousarray(cos, dtype=np.float32)
        sin = np.ascontiguousarray(sin, dtype=np.float32)
        if cos.shape != expected_shape or sin.shape != expected_shape:
            raise ValueError(
                f"Block {block} learned cache shape differs: "
                f"cos={cos.shape}, sin={sin.shape}, expected={expected_shape}"
            )
        result[block] = (cos, sin)
    return result


def _append_initializer(
    model: onnx.ModelProto, existing: set[str], name: str, value: np.ndarray
) -> None:
    if name in existing:
        raise ValueError(f"Initializer already exists: {name}")
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))
    existing.add(name)


def _mha_node(
    block: int, q: str, k: str, v: str, output: str, heads: int
) -> onnx.NodeProto:
    prefix = f"/model/blocks.{block}/"
    return helper.make_node(
        "MultiHeadAttention",
        [q, k, v],
        [output],
        name=prefix + "MultiHeadAttention",
        domain="com.microsoft",
        num_heads=heads,
        unidirectional=0,
    )


def _reshape_node(name: str, source: str, shape: str, output: str) -> onnx.NodeProto:
    return helper.make_node("Reshape", [source, shape], [output], name=name)


def _fixed_replacement(
    *,
    block: int,
    q_input: str,
    k_input: str,
    v_input: str,
    output: str,
    position_name: str,
    cos_name: str,
    sin_name: str,
    qk_rank3_shape_name: str | None,
    geometry: TransformerGeometry,
) -> tuple[list[onnx.NodeProto], list[str]]:
    prefix = f"/model/blocks.{block}/"
    replacements: list[onnx.NodeProto] = []
    rank3_values: list[str] = []
    if qk_rank3_shape_name is not None:
        q_rank3 = prefix + "QNormRank3_output_0"
        k_rank3 = prefix + "KNormRank3_output_0"
        replacements.extend(
            [
                _reshape_node(prefix + "QNormToRank3", q_input, qk_rank3_shape_name, q_rank3),
                _reshape_node(prefix + "KNormToRank3", k_input, qk_rank3_shape_name, k_rank3),
            ]
        )
        q_input, k_input = q_rank3, k_rank3
        rank3_values.extend([q_rank3, k_rank3])
    q_rot = prefix + "RotaryEmbeddingQ_output_0"
    k_rot = prefix + "RotaryEmbeddingK_output_0"
    replacements.extend(
        [
            helper.make_node(
                "RotaryEmbedding",
                [q_input, position_name, cos_name, sin_name],
                [q_rot],
                name=prefix + "RotaryEmbeddingQ",
                domain="com.microsoft",
                num_heads=geometry.num_heads,
                rotary_embedding_dim=geometry.head_size,
                interleaved=1,
            ),
            helper.make_node(
                "RotaryEmbedding",
                [k_input, position_name, cos_name, sin_name],
                [k_rot],
                name=prefix + "RotaryEmbeddingK",
                domain="com.microsoft",
                num_heads=geometry.num_heads,
                rotary_embedding_dim=geometry.head_size,
                interleaved=1,
            ),
            _mha_node(block, q_rot, k_rot, v_input, output, geometry.num_heads),
        ]
    )
    rank3_values.extend([q_rot, k_rot, output])
    return replacements, rank3_values


def _learned_per_head_replacement(
    *,
    model: onnx.ModelProto,
    existing_initializers: set[str],
    block: int,
    q_input: str,
    k_input: str,
    v_input: str,
    output: str,
    position_name: str,
    split_sizes_name: str,
    caches: tuple[np.ndarray, np.ndarray],
    geometry: TransformerGeometry,
) -> tuple[list[onnx.NodeProto], list[str], list[str]]:
    prefix = f"/model/blocks.{block}/"
    q_split = [prefix + f"QHead{head}_output_0" for head in range(geometry.num_heads)]
    k_split = [prefix + f"KHead{head}_output_0" for head in range(geometry.num_heads)]
    replacements: list[onnx.NodeProto] = [
        helper.make_node(
            "Split", [q_input, split_sizes_name], q_split, name=prefix + "SplitQHeads", axis=2
        ),
        helper.make_node(
            "Split", [k_input, split_sizes_name], k_split, name=prefix + "SplitKHeads", axis=2
        ),
    ]
    cos, sin = caches
    q_rotated: list[str] = []
    k_rotated: list[str] = []
    for head in range(geometry.num_heads):
        cos_name = f"flash_learned_rope_cos_b{block}_h{head}"
        sin_name = f"flash_learned_rope_sin_b{block}_h{head}"
        _append_initializer(model, existing_initializers, cos_name, cos[:, head, :])
        _append_initializer(model, existing_initializers, sin_name, sin[:, head, :])
        q_out = prefix + f"RotaryEmbeddingQHead{head}_output_0"
        k_out = prefix + f"RotaryEmbeddingKHead{head}_output_0"
        q_rotated.append(q_out)
        k_rotated.append(k_out)
        for label, source, target in (
            ("Q", q_split[head], q_out),
            ("K", k_split[head], k_out),
        ):
            replacements.append(
                helper.make_node(
                    "RotaryEmbedding",
                    [source, position_name, cos_name, sin_name],
                    [target],
                    name=prefix + f"RotaryEmbedding{label}Head{head}",
                    domain="com.microsoft",
                    num_heads=1,
                    rotary_embedding_dim=geometry.head_size,
                    interleaved=1,
                )
            )
    q_concat = prefix + "RotaryEmbeddingQConcat_output_0"
    k_concat = prefix + "RotaryEmbeddingKConcat_output_0"
    replacements.extend(
        [
            helper.make_node("Concat", q_rotated, [q_concat], name=prefix + "ConcatQHeads", axis=2),
            helper.make_node("Concat", k_rotated, [k_concat], name=prefix + "ConcatKHeads", axis=2),
            _mha_node(block, q_concat, k_concat, v_input, output, geometry.num_heads),
        ]
    )
    head_values = q_split + k_split + q_rotated + k_rotated
    return replacements, [q_concat, k_concat, output], head_values


def rewrite_attention(
    source: Path, output: Path, require_qk_norm: bool = False
) -> dict[str, Any]:
    """Rewrite attention using the single supported production mapping."""
    model = onnx.load(source)
    geometry = discover_transformer_geometry(model)
    if geometry.batch_size != 1:
        raise ValueError(f"This rewrite requires fixed batch 1, got {geometry.batch_size}")
    nodes = list(model.graph.node)
    original_counts = Counter(node.op_type for node in nodes)
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    if len(by_name) != len(nodes):
        raise ValueError("Graph contains duplicate node names")
    node_names = set(by_name)
    existing_initializers = {initializer.name for initializer in model.graph.initializer}

    learned_by_block = {
        block: (
            (
                f"/model/blocks.{block}/Cos" in node_names
                and f"/model/blocks.{block}/Sin" in node_names
            )
            or (
                f"/model/blocks.{block}/Unsqueeze_2_output_0" in existing_initializers
                and f"/model/blocks.{block}/Unsqueeze_3_output_0" in existing_initializers
            )
        )
        for block in range(geometry.blocks)
    }
    if len(set(learned_by_block.values())) != 1:
        raise ValueError(f"Learned-RoPE presence differs by block: {learned_by_block}")
    learned_rope = learned_by_block[0]
    rope_kind = "learned_per_head" if learned_rope else "fixed_shared_frequency"

    qk_norm_by_block: dict[int, bool] = {}
    for block in range(geometry.blocks):
        prefix = f"/model/blocks.{block}/"
        has_q = prefix + "q_norm/Mul_2" in node_names
        has_k = prefix + "k_norm/Mul_2" in node_names
        if has_q != has_k:
            raise ValueError(f"Block {block} contains only one of q_norm/k_norm")
        qk_norm_by_block[block] = has_q
    if len(set(qk_norm_by_block.values())) != 1:
        raise ValueError(f"Q/K Norm presence differs by block: {qk_norm_by_block}")
    preserve_qk_norm = qk_norm_by_block[0]
    if require_qk_norm and not preserve_qk_norm:
        raise ValueError("--require-qk-norm was set, but the model has no Q/K Norm")
    if learned_rope and preserve_qk_norm:
        raise ValueError("Learned-RoPE + Q/K Norm is not supported by this rewrite")

    fixed_caches = None if learned_rope else _extract_fixed_rope_caches(model, geometry)
    learned_caches = _extract_learned_rope_caches(model, geometry) if learned_rope else None

    position_name = "flash_rope_position_offset"
    rank3_shape_name = "flash_attention_rank3_shape"
    _append_initializer(model, existing_initializers, position_name, np.asarray([0], dtype=np.int64))
    _append_initializer(
        model,
        existing_initializers,
        rank3_shape_name,
        np.asarray(
            [geometry.batch_size, geometry.sequence_length, geometry.hidden_size],
            dtype=np.int64,
        ),
    )
    split_sizes_name = "flash_attention_head_split_sizes"
    if learned_rope:
        _append_initializer(
            model,
            existing_initializers,
            split_sizes_name,
            np.full(geometry.num_heads, geometry.head_size, dtype=np.int64),
        )

    remove_indices: set[int] = set()
    replacements_by_index: dict[int, list[onnx.NodeProto]] = {}
    removed_by_block: dict[int, list[str]] = {}
    preserved_qkn_by_block: dict[int, list[str]] = {}
    rank3_value_names: set[str] = set()
    head_value_names: set[str] = set()

    for block in range(geometry.blocks):
        prefix = f"/model/blocks.{block}/"
        if prefix + "q_proj/MatMul" in by_name:
            q_input = by_name[prefix + "q_proj/MatMul"][1].output[0]
            k_input = by_name[prefix + "k_proj/MatMul"][1].output[0]
            v_input = by_name[prefix + "v_proj/MatMul"][1].output[0]
        else:
            if learned_rope:
                q_input = by_name[prefix + "Reshape_4"][1].input[0]
                k_input = by_name[prefix + "Reshape_6"][1].input[0]
                v_input = by_name[prefix + "Reshape_3"][1].input[0]
            else:
                q_input = by_name[prefix + "Reshape_1"][1].input[0]
                k_input = by_name[prefix + "Reshape_2"][1].input[0]
                v_input = by_name[prefix + "Reshape_3"][1].input[0]
        out_node = by_name[prefix + "out_proj/MatMul"][1]
        old_output_node = by_name[prefix + "Reshape_8"][1]
        attention_output = old_output_node.output[0]
        if attention_output not in out_node.input:
            raise ValueError(f"Attention output does not feed out_proj in block {block}")

        preserved_qkn_names: list[str] = []
        if learned_rope:
            assert learned_caches is not None
            start_index = (
                by_name[prefix + "Reshape_1"][0]
                if prefix + "Reshape_1" in by_name
                else min(
                    by_name[prefix + "Reshape_3"][0],
                    by_name[prefix + "Reshape_4"][0],
                    by_name[prefix + "Reshape_6"][0],
                )
            )
            end_index = by_name[prefix + "Reshape_8"][0]
            replacements, rank3_values, head_values = _learned_per_head_replacement(
                model=model,
                existing_initializers=existing_initializers,
                block=block,
                q_input=q_input,
                k_input=k_input,
                v_input=v_input,
                output=attention_output,
                position_name=position_name,
                split_sizes_name=split_sizes_name,
                caches=learned_caches[block],
                geometry=geometry,
            )
            block_remove = set(range(start_index, end_index + 1))
        else:
            assert fixed_caches is not None
            start_index = (
                by_name[prefix + "Reshape_1"][0]
                if prefix + "Reshape_1" in by_name
                else min(
                    by_name[prefix + "Reshape_3"][0],
                    by_name[prefix + "Reshape_4"][0],
                    by_name[prefix + "Reshape_6"][0],
                )
            )
            end_index = by_name[prefix + "Reshape_8"][0]
            block_remove = set(range(start_index, end_index + 1))
            qkn_shape = rank3_shape_name if preserve_qk_norm else None
            if preserve_qk_norm:
                q_reshape = by_name[prefix + "Reshape_1"][1]
                k_reshape = by_name[prefix + "Reshape_2"][1]
                q_norm = by_name[prefix + "q_norm/Mul_2"][1]
                k_norm = by_name[prefix + "k_norm/Mul_2"][1]
                if q_reshape.output[0] not in by_name[prefix + "q_norm/Cast"][1].input:
                    raise ValueError(f"q_norm does not consume Q reshape in block {block}")
                if k_reshape.output[0] not in by_name[prefix + "k_norm/Cast"][1].input:
                    raise ValueError(f"k_norm does not consume K reshape in block {block}")
                preserved_qkn_names = [
                    node.name
                    for node in nodes[start_index : end_index + 1]
                    if node.name in {prefix + "Reshape_1", prefix + "Reshape_2"}
                    or node.name.startswith(prefix + "q_norm/")
                    or node.name.startswith(prefix + "k_norm/")
                ]
                preserved_set = set(preserved_qkn_names)
                block_remove = {
                    index for index in block_remove if nodes[index].name not in preserved_set
                }
                block_remove.add(by_name[prefix + "Constant_3"][0])
                start_index = by_name[prefix + "Constant_4"][0]
                q_input, k_input = q_norm.output[0], k_norm.output[0]
            cos_name = f"flash_fixed_rope_cos_b{block}"
            sin_name = f"flash_fixed_rope_sin_b{block}"
            _append_initializer(model, existing_initializers, cos_name, fixed_caches[block][0])
            _append_initializer(model, existing_initializers, sin_name, fixed_caches[block][1])
            replacements, rank3_values = _fixed_replacement(
                block=block,
                q_input=q_input,
                k_input=k_input,
                v_input=v_input,
                output=attention_output,
                position_name=position_name,
                cos_name=cos_name,
                sin_name=sin_name,
                qk_rank3_shape_name=qkn_shape,
                geometry=geometry,
            )
            head_values = []

        removed_nodes = [nodes[index] for index in sorted(block_remove)]
        removed_outputs = {name for node in removed_nodes for name in node.output}
        outside = [
            (node.name, value)
            for index, node in enumerate(nodes)
            if index not in block_remove
            for value in node.input
            if value in removed_outputs and value != attention_output
        ]
        if outside:
            raise ValueError(f"Block {block} removed values have outside consumers: {outside}")
        remove_indices.update(block_remove)
        replacements_by_index[start_index] = replacements
        removed_by_block[block] = [node.name for node in removed_nodes]
        preserved_qkn_by_block[block] = preserved_qkn_names
        rank3_value_names.update(rank3_values)
        head_value_names.update(head_values)

    rewritten_nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        if index in replacements_by_index:
            rewritten_nodes.extend(replacements_by_index[index])
        if index not in remove_indices:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    live_values = {value.name for value in model.graph.input}
    live_values.update(value.name for value in model.graph.output)
    live_values.update(initializer.name for initializer in model.graph.initializer)
    for node in model.graph.node:
        live_values.update(node.output)
    kept_value_info = [value for value in model.graph.value_info if value.name in live_values]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    existing_value_info = {value.name for value in model.graph.value_info}
    for name in sorted(rank3_value_names):
        if name not in existing_value_info:
            model.graph.value_info.append(
                helper.make_tensor_value_info(
                    name,
                    TensorProto.FLOAT,
                    [geometry.batch_size, geometry.sequence_length, geometry.hidden_size],
                )
            )
    for name in sorted(head_value_names):
        if name not in existing_value_info:
            model.graph.value_info.append(
                helper.make_tensor_value_info(
                    name,
                    TensorProto.FLOAT,
                    [geometry.batch_size, geometry.sequence_length, geometry.head_size],
                )
            )

    ms_import = next((item for item in model.opset_import if item.domain == "com.microsoft"), None)
    if ms_import is None:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    elif ms_import.version < 1:
        ms_import.version = 1

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["attention_rewrite"] = "com.microsoft.MultiHeadAttention"
    metadata["attention_rope_kind"] = rope_kind
    metadata["attention_learned_rope_mode"] = "per-head"
    metadata["attention_rewrite_fixed_batch"] = str(geometry.batch_size)
    metadata["attention_rewrite_preserves_qk_norm"] = str(preserve_qk_norm).lower()
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    onnx.checker.check_model(onnx.load(output), full_check=True)

    cache_source = learned_caches if learned_caches is not None else fixed_caches
    return {
        "kind": "rope_mha_rewrite",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "geometry": geometry.to_dict(),
        "rope_kind": rope_kind,
        "learned_rope_mode": "per-head",
        "qk_norm_preserved": preserve_qk_norm,
        "checker": "onnx.checker full_check passed before and after serialization",
        "original_op_counts": dict(sorted(original_counts.items())),
        "rewritten_op_counts": dict(
            sorted(Counter(node.op_type for node in model.graph.node).items())
        ),
        "removed_nodes_per_block": {
            str(block): len(names) for block, names in removed_by_block.items()
        },
        "preserved_qk_norm_nodes_per_block": {
            str(block): len(names) for block, names in preserved_qkn_by_block.items()
        },
        "cache_digests_by_block": {
            str(block): {"cos": _cache_digest(pair[0]), "sin": _cache_digest(pair[1])}
            for block, pair in cache_source.items()
        },
    }
