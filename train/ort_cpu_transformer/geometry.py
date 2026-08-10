"""Strict geometry discovery for fixed-batch KataGo transformer ONNX exports."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper


_BLOCK_NODE = re.compile(r"^/model/blocks\.(\d+)/")


@dataclass(frozen=True)
class TransformerGeometry:
    blocks: int
    batch_size: int
    sequence_length: int
    num_heads: int
    head_size: int
    hidden_size: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _constant_array(node: onnx.NodeProto) -> np.ndarray:
    if node.op_type != "Constant":
        raise ValueError(f"Expected Constant, got {node.op_type}: {node.name}")
    for attribute in node.attribute:
        if attribute.name == "value":
            return numpy_helper.to_array(helper.get_attribute_value(attribute))
    raise ValueError(f"Constant node has no tensor value: {node.name}")


def discover_transformer_geometry(model: onnx.ModelProto) -> TransformerGeometry:
    block_indices = sorted(
        {
            int(match.group(1))
            for node in model.graph.node
            if (match := _BLOCK_NODE.match(node.name)) is not None
        }
    )
    if not block_indices or block_indices != list(range(len(block_indices))):
        raise ValueError(f"Transformer blocks are missing/non-contiguous: {block_indices}")

    spatial_input = next(
        (value for value in model.graph.input if value.name == "input_spatial"), None
    )
    if spatial_input is None:
        raise ValueError("Missing input_spatial geometry")
    spatial_dims = spatial_input.type.tensor_type.shape.dim
    if len(spatial_dims) != 4 or any(not dim.HasField("dim_value") for dim in spatial_dims):
        raise ValueError("Expected fully fixed input_spatial shape")
    fixed_batch = int(spatial_dims[0].dim_value)
    sequence_length = int(spatial_dims[2].dim_value) * int(spatial_dims[3].dim_value)

    constant_values: dict[str, np.ndarray] = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            constant_values[node.output[0]] = _constant_array(node)

    fixed_value_shapes: dict[str, tuple[int, ...]] = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            fixed_value_shapes[value.name] = tuple(int(dim.dim_value) for dim in dims)

    geometries: list[tuple[int, int, int, int]] = []
    for block in block_indices:
        prefix = f"/model/blocks.{block}/"
        candidates: set[tuple[int, int, int, int]] = set()
        for node in model.graph.node:
            if not node.name.startswith(prefix) or node.op_type != "Reshape" or len(node.input) != 2:
                continue
            value = constant_values.get(node.input[1])
            if value is None:
                continue
            shape = np.asarray(value, dtype=np.int64).reshape(-1)
            if (
                shape.size == 4
                and np.all(shape > 0)
                and int(shape[0]) == fixed_batch
                and int(shape[1]) == sequence_length
            ):
                candidates.add(tuple(int(item) for item in shape))

        # A per-head RotaryEmbedding rewrite removes the rank-4 attention
        # reshapes. This fallback lets later stages rediscover the same geometry.
        mha_nodes = [
            node
            for node in model.graph.node
            if node.name.startswith(prefix)
            and node.domain == "com.microsoft"
            and node.op_type == "MultiHeadAttention"
        ]
        for mha in mha_nodes:
            heads = next(
                (
                    int(helper.get_attribute_value(attribute))
                    for attribute in mha.attribute
                    if attribute.name == "num_heads"
                ),
                None,
            )
            if heads is None or heads <= 0:
                raise ValueError(f"Missing/invalid num_heads on {mha.name}")
            for input_name in mha.input[:3]:
                shape = fixed_value_shapes.get(input_name)
                if (
                    shape is not None
                    and len(shape) == 3
                    and shape[0] == fixed_batch
                    and shape[1] == sequence_length
                    and shape[2] > 0
                    and shape[2] % heads == 0
                ):
                    candidates.add((fixed_batch, sequence_length, heads, shape[2] // heads))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one unique fixed [B,S,H,D] reshape geometry in block {block}, "
                f"got {sorted(candidates)}"
            )
        batch, sequence, heads, head_size = next(iter(candidates))
        geometries.append((batch, sequence, heads, head_size))

    if len(set(geometries)) != 1:
        raise ValueError(f"Transformer geometry differs by block: {geometries}")
    batch, sequence, heads, head_size = geometries[0]
    norm_gamma = next(
        (
            numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
            if initializer.name == "model.blocks.0.norm1.weight"
        ),
        None,
    )
    hidden_size = heads * head_size
    if norm_gamma is not None and norm_gamma.shape != (hidden_size,):
        raise ValueError(
            f"Canonical norm1 gamma does not match H*D={hidden_size}: {norm_gamma.shape}"
        )
    return TransformerGeometry(
        blocks=len(block_indices),
        batch_size=batch,
        sequence_length=sequence,
        num_heads=heads,
        head_size=head_size,
        hidden_size=hidden_size,
    )


def resolve_expected_blocks(
    geometry: TransformerGeometry, expected_blocks: int | None
) -> int:
    if expected_blocks is not None and expected_blocks != geometry.blocks:
        raise ValueError(
            f"--expected-blocks={expected_blocks} disagrees with ONNX geometry "
            f"({geometry.blocks})"
        )
    return geometry.blocks
