#!/usr/bin/env python3
"""Build a strictly audited ORT-CPU-optimized FP32 ONNX model.

The input must be an onnxsim-simplified, fixed-batch-1 KataGo transformer
export produced by ``train/export_onnx.py -disable-mask``. The converter uses
ONNX Runtime CPU contrib operators to rewrite RoPE/attention, fuse RMSNorm,
and fuse residual Add + RMSNorm. The output is intentionally ORT-specific and
is not the portable ONNX/TensorRT export path.

Supported source structures are fixed-frequency ``tfrs``, learned-frequency
``tflrs`` (the final per-head mapping), old fixed-RoPE QK-Norm/no-QK-Norm, and
ordinary SwiGLU-only ``clip4``/``clip7``. ``fullclip4``/``fullclip7`` and
learned-RoPE + QK-Norm are rejected rather than converted ambiguously.

The destination is replaced only after all graph rewrites, strict structural
audits, serialization checks, and an optional five-output numerical check
succeed. Output paths are unrestricted; temporary files are kept next to the
destination so final replacement is atomic on the destination filesystem.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import onnx

from ort_cpu_transformer.attention import rewrite_attention
from ort_cpu_transformer.geometry import (
    TransformerGeometry,
    discover_transformer_geometry,
    resolve_expected_blocks,
)
from ort_cpu_transformer.norms import (
    DEFAULT_NORM_KINDS,
    checker_with_ort_schema,
    compare_outputs,
    fuse_rmsnorm,
    fuse_skip_sln,
    op_counts,
    qkn_nodes,
    qualified_op_counts,
    retain_source_shape_information,
)


PIPELINE_VERSION = 2
FP32_MAX_ABS_LIMIT = 5.0e-5
FP32_MEAN_ABS_LIMIT = 5.0e-6
EXPECTED_OUTPUTS = (
    "out_policy",
    "out_value",
    "out_miscvalue",
    "out_moremiscvalue",
    "out_ownership",
)


def _require_unique_node_names(model: onnx.ModelProto, stage: str) -> None:
    names = [node.name for node in model.graph.node]
    unnamed = sum(not name for name in names)
    if unnamed:
        raise ValueError(f"{stage}: graph has {unnamed} unnamed nodes")
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    if duplicates:
        raise ValueError(f"{stage}: duplicate node names: {duplicates[:20]}")


def _source_audit(
    source: Path, model: onnx.ModelProto, expected_blocks: int | None
) -> tuple[TransformerGeometry, dict[str, Any]]:
    _require_unique_node_names(model, "source")
    geometry = discover_transformer_geometry(model)
    blocks = resolve_expected_blocks(geometry, expected_blocks)
    if geometry.batch_size != 1:
        raise ValueError(
            "ORT FP32 optimization requires a fixed-batch-1 source; "
            f"discovered batch={geometry.batch_size}"
        )
    output_names = tuple(value.name for value in model.graph.output)
    if output_names != EXPECTED_OUTPUTS:
        raise ValueError(
            "Expected the five KataGo outputs in their canonical order, got "
            f"{output_names}; expected {EXPECTED_OUTPUTS}"
        )
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if metadata.get("has_mask", "").strip().lower() != "false":
        raise ValueError(
            "Input contract violation: the source must be exported with "
            "train/export_onnx.py -disable-mask and carry metadata "
            f"has_mask=false; got {metadata.get('has_mask')!r}"
        )
    node_names = {node.name for node in model.graph.node}
    initializer_names = {value.name for value in model.graph.initializer}
    learned_by_block = {
        block: (
            (
                f"/model/blocks.{block}/Cos" in node_names
                and f"/model/blocks.{block}/Sin" in node_names
            )
            or (
                f"/model/blocks.{block}/Unsqueeze_2_output_0" in initializer_names
                and f"/model/blocks.{block}/Unsqueeze_3_output_0" in initializer_names
            )
        )
        for block in range(blocks)
    }
    qkn_pair_by_block = {
        block: (
            f"/model/blocks.{block}/q_norm/Mul_2" in node_names,
            f"/model/blocks.{block}/k_norm/Mul_2" in node_names,
        )
        for block in range(blocks)
    }
    mismatched_qkn = {
        block: pair for block, pair in qkn_pair_by_block.items() if pair[0] != pair[1]
    }
    if mismatched_qkn:
        raise ValueError(f"Only one of Q/K Norm is present: {mismatched_qkn}")
    qkn_by_block = {block: pair[0] for block, pair in qkn_pair_by_block.items()}
    if len(set(learned_by_block.values())) != 1:
        raise ValueError(f"Learned-RoPE presence differs by block: {learned_by_block}")
    if len(set(qkn_by_block.values())) != 1:
        raise ValueError(f"Q/K Norm presence differs by block: {qkn_by_block}")
    learned_rope = learned_by_block[0]
    qk_norm = qkn_by_block[0]
    if learned_rope and qk_norm:
        raise ValueError(
            "Learned-RoPE + Q/K Norm is not supported by the final per-head ORT "
            "attention rewrite"
        )

    clip_counts = {
        block: sum(
            node.op_type == "Clip" and node.name.startswith(f"/model/blocks.{block}/")
            for node in model.graph.node
        )
        for block in range(blocks)
    }
    if any(count not in (0, 2) for count in clip_counts.values()):
        raise ValueError(
            "Detected an unsupported/fullclip-style transformer topology (a block "
            "has neither zero nor exactly two ordinary SwiGLU clip nodes). "
            "fullclip4/fullclip7 are not supported by this ORT attention rewrite: "
            f"{clip_counts}"
        )
    onnx.checker.check_model(model, full_check=True)
    return geometry, {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "checker": "onnx.checker full_check passed",
        "geometry": geometry.to_dict(),
        "output_names": list(output_names),
        "node_count": len(model.graph.node),
        "op_counts": op_counts(model),
        "expected_blocks": blocks,
        "input_contract": {
            "fixed_batch_1": True,
            "disable_mask": True,
            "metadata_has_mask": metadata["has_mask"],
            "onnxsim_simplified_topology": True,
        },
        "topology_preflight": {
            "learned_rope": learned_rope,
            "qk_norm": qk_norm,
            "clip_nodes_per_block": clip_counts,
            "fullclip_supported": False,
        },
    }


def _nodes_of(
    model: onnx.ModelProto, op_type: str, domain: str
) -> list[onnx.NodeProto]:
    return [
        node
        for node in model.graph.node
        if node.op_type == op_type and (node.domain or "") == domain
    ]


def _audit_attention(
    model: onnx.ModelProto,
    geometry: TransformerGeometry,
    attention_report: dict[str, Any],
) -> dict[str, Any]:
    _require_unique_node_names(model, "attention")
    blocks = geometry.blocks
    mha = _nodes_of(model, "MultiHeadAttention", "com.microsoft")
    expected_mha_names = {
        f"/model/blocks.{block}/MultiHeadAttention" for block in range(blocks)
    }
    if len(mha) != blocks or {node.name for node in mha} != expected_mha_names:
        raise ValueError(
            "Expected exactly one ORT MultiHeadAttention per block; got "
            f"{[node.name for node in mha]}"
        )

    rope_kind = attention_report["rope_kind"]
    learned_mode = attention_report["learned_rope_mode"]
    if rope_kind == "learned_per_head":
        if learned_mode != "per-head":
            raise ValueError(f"Learned RoPE did not use final per-head mapping: {learned_mode}")
        expected_rotary = 2 * blocks * geometry.num_heads
    elif rope_kind == "fixed_shared_frequency":
        expected_rotary = 2 * blocks
    else:
        raise ValueError(f"Unknown RoPE kind reported by attention rewrite: {rope_kind!r}")
    rotary = _nodes_of(model, "RotaryEmbedding", "com.microsoft")
    if len(rotary) != expected_rotary:
        raise ValueError(
            f"Expected {expected_rotary} ORT RotaryEmbedding nodes for {rope_kind}, "
            f"got {len(rotary)}"
        )
    removed = attention_report.get("removed_nodes_per_block", {})
    if set(removed) != {str(block) for block in range(blocks)} or any(
        not isinstance(count, int) or count <= 0 for count in removed.values()
    ):
        raise ValueError(f"Attention rewrite did not audit every block: {removed}")
    checker = checker_with_ort_schema(model)
    return {
        "checker": checker,
        "rope_kind": rope_kind,
        "learned_rope_mode": learned_mode,
        "qk_norm_preserved": bool(attention_report["qk_norm_preserved"]),
        "mha_nodes": len(mha),
        "rotary_embedding_nodes": len(rotary),
        "removed_nodes_per_block": removed,
        "cache_digests_by_block": attention_report.get("cache_digests_by_block"),
        "op_counts": op_counts(model),
    }


def _audit_rmsnorm(
    before_counts: dict[str, int],
    model: onnx.ModelProto,
    details: list[dict[str, Any]],
    geometry: TransformerGeometry,
    norm_kinds: tuple[str, ...],
) -> dict[str, Any]:
    _require_unique_node_names(model, "rmsnorm")
    expected = geometry.blocks * len(norm_kinds)
    expected_pairs = {
        (block, norm_kind)
        for block in range(geometry.blocks)
        for norm_kind in norm_kinds
    }
    actual_pairs = {(item["block"], item["norm"]) for item in details}
    if len(details) != expected or actual_pairs != expected_pairs:
        raise ValueError(
            f"RMSNorm fusion coverage differs: expected={sorted(expected_pairs)}, "
            f"actual={sorted(actual_pairs)}"
        )
    after_counts = op_counts(model)
    if after_counts.get("SimplifiedLayerNormalization", 0) != (
        before_counts.get("SimplifiedLayerNormalization", 0) + expected
    ):
        raise ValueError(f"Expected {expected} new SLN nodes: {after_counts}")
    if after_counts.get("ReduceMean", 0) != before_counts.get("ReduceMean", 0) - expected:
        raise ValueError("Unexpected ReduceMean count after RMSNorm fusion")
    checker = checker_with_ort_schema(model)
    return {
        "checker": checker,
        "norm_kinds": list(norm_kinds),
        "fusions": len(details),
        "op_counts_before": before_counts,
        "op_counts_after": after_counts,
        "details": details,
    }


def _audit_skip_sln(
    before_counts: dict[str, int],
    qkn_before: list[dict[str, Any]],
    model: onnx.ModelProto,
    details: list[dict[str, Any]],
    geometry: TransformerGeometry,
) -> dict[str, Any]:
    _require_unique_node_names(model, "skip_sln")
    expected = 2 * geometry.blocks - 1
    expected_pairs = {
        *((block, "norm2") for block in range(geometry.blocks)),
        *((block, "norm1") for block in range(1, geometry.blocks)),
    }
    actual_pairs = {(item["block"], item["norm"]) for item in details}
    if len(details) != expected or actual_pairs != expected_pairs:
        raise ValueError(
            f"SkipSLN fusion coverage differs: expected={sorted(expected_pairs)}, "
            f"actual={sorted(actual_pairs)}"
        )
    skip_nodes = _nodes_of(model, "SkipSimplifiedLayerNormalization", "com.microsoft")
    if len(skip_nodes) != expected:
        raise ValueError(f"Expected {expected} ORT SkipSLN nodes, got {len(skip_nodes)}")
    qkn_after = qkn_nodes(model)
    if qkn_after != qkn_before:
        raise ValueError("q_norm/k_norm SLN nodes changed during SkipSLN rewrite")
    after_counts = op_counts(model)
    if after_counts.get("SimplifiedLayerNormalization", 0) != 1 + len(qkn_before):
        raise ValueError(f"Unexpected remaining SLN count: {after_counts}")
    if after_counts.get("Add", 0) != before_counts.get("Add", 0) - expected:
        raise ValueError("Unexpected Add count after SkipSLN fusion")
    checker = checker_with_ort_schema(model)
    return {
        "checker": checker,
        "fusions": len(details),
        "qkn_nodes_preserved": len(qkn_after),
        "op_counts_before": before_counts,
        "op_counts_after": after_counts,
        "qualified_op_counts_after": qualified_op_counts(model),
        "details": details,
    }


def _stage_json(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(staged.read_text(encoding="utf-8"))
    return staged


@contextmanager
def _work_directory(parent: Path, output_name: str) -> Iterator[Path]:
    """Use an ACL-inheriting, same-filesystem directory for atomic output."""
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    working = parent / f".{output_name}.ort_fp32_{uuid.uuid4().hex}"
    working.mkdir(parents=False, exist_ok=False, mode=0o777)
    try:
        yield working
    finally:
        # Cleanup only the exact child generated above.
        if working.parent == parent and working.name.startswith(f".{output_name}.ort_fp32_"):
            shutil.rmtree(working, ignore_errors=True)


def optimize(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.resolve()
    output = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output.with_suffix(output.suffix + ".ort_cpu_fp32_report.json")
    )
    data = args.data.resolve() if args.data is not None else None
    if source.suffix.lower() != ".onnx" or output.suffix.lower() != ".onnx":
        raise ValueError("--input and --output must both have the .onnx suffix")
    if report_path.suffix.lower() != ".json":
        raise ValueError("--report must have the .json suffix")
    named_paths = {"input": source, "output": output, "report": report_path}
    if data is not None:
        named_paths["data"] = data
    duplicate_groups: dict[Path, list[str]] = {}
    for label, path in named_paths.items():
        duplicate_groups.setdefault(path, []).append(label)
    collisions = {
        str(path): labels for path, labels in duplicate_groups.items() if len(labels) > 1
    }
    if collisions:
        raise ValueError(f"Input/output/report/data paths must be pairwise distinct: {collisions}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if data is not None and not data.is_file():
        raise FileNotFoundError(data)

    source_model = onnx.load(source)
    geometry, source_report = _source_audit(source, source_model, args.expected_blocks)

    output.parent.mkdir(parents=True, exist_ok=True)
    with _work_directory(output.parent, output.name) as temporary:
        attention_path = temporary / "01_attention.onnx"
        rmsnorm_path = temporary / "02_rmsnorm.onnx"
        final_path = temporary / output.name

        attention_raw = rewrite_attention(
            source=source,
            output=attention_path,
            require_qk_norm=args.require_qk_norm,
        )
        attention_model = onnx.load(attention_path)
        attention_audit = _audit_attention(attention_model, geometry, attention_raw)
        has_qkn = attention_audit["qk_norm_preserved"]
        if args.require_no_qk_norm and has_qkn:
            raise ValueError("--require-no-qk-norm was set, but Q/K Norm was detected")

        norm_kinds = tuple(DEFAULT_NORM_KINDS) + (("q_norm", "k_norm") if has_qkn else ())
        rms_before_counts = op_counts(attention_model)
        inferred_attention = onnx.shape_inference.infer_shapes(
            attention_model, check_type=True, strict_mode=True, data_prop=False
        )
        rms_model, rms_details = fuse_rmsnorm(attention_model, geometry, norm_kinds)
        retained_rms_shapes = retain_source_shape_information(rms_model, inferred_attention)
        rms_audit = _audit_rmsnorm(
            rms_before_counts, rms_model, rms_details, geometry, norm_kinds
        )
        rms_audit["retained_source_value_info"] = retained_rms_shapes
        onnx.save(rms_model, rmsnorm_path)
        rms_reloaded = onnx.load(rmsnorm_path)
        rms_audit["saved_model_checker"] = checker_with_ort_schema(rms_reloaded)

        skip_before_counts = op_counts(rms_reloaded)
        qkn_before = qkn_nodes(rms_reloaded)
        expected_qkn = 2 * geometry.blocks if has_qkn else 0
        if len(qkn_before) != expected_qkn:
            raise ValueError(
                f"Expected {expected_qkn} q/k SLN nodes before SkipSLN, "
                f"got {len(qkn_before)}"
            )
        inferred_rms = onnx.shape_inference.infer_shapes(
            rms_reloaded, check_type=True, strict_mode=True, data_prop=False
        )
        final_model, skip_details = fuse_skip_sln(rms_reloaded, inferred_rms, geometry)
        retained_skip_shapes = retain_source_shape_information(final_model, inferred_rms)
        skip_audit = _audit_skip_sln(
            skip_before_counts, qkn_before, final_model, skip_details, geometry
        )
        skip_audit["retained_source_value_info"] = retained_skip_shapes
        onnx.save(final_model, final_path)
        saved_final = onnx.load(final_path)
        skip_audit["saved_model_checker"] = checker_with_ort_schema(saved_final)

        final_outputs = [value.name for value in saved_final.graph.output]
        if tuple(final_outputs) != EXPECTED_OUTPUTS:
            raise ValueError(
                f"Five model outputs changed: source={source_report['output_names']}, "
                f"final={final_outputs}, expected={EXPECTED_OUTPUTS}"
            )

        numerical: dict[str, Any] | None = None
        if data is not None:
            comparisons, compared_names = compare_outputs(source, final_path, data)
            if compared_names != final_outputs or len(comparisons) != 5:
                raise ValueError(
                    "Numerical check did not compare exactly five outputs: "
                    f"names={compared_names}, comparisons={len(comparisons)}"
                )
            passed = all(
                item["max_abs"] <= FP32_MAX_ABS_LIMIT
                and item["mean_abs"] <= FP32_MEAN_ABS_LIMIT
                for item in comparisons
            )
            numerical = {
                "data": str(data),
                "reference": str(source),
                "output_names": compared_names,
                "max_abs_limit": FP32_MAX_ABS_LIMIT,
                "mean_abs_limit": FP32_MEAN_ABS_LIMIT,
                "passed": passed,
                "comparisons": comparisons,
            }
            if not passed:
                raise ValueError(f"FP32 five-output equivalence gate failed: {comparisons}")

        report: dict[str, Any] = {
            "kind": "ort_cpu_fp32_transformer_optimization",
            "pipeline_version": PIPELINE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "specialization": {
                "runtime": "ONNX Runtime",
                "execution_provider": "CPUExecutionProvider",
                "portable_onnx_or_tensorrt_path": False,
            },
            "input_model": str(source),
            "output_model": str(output),
            "report": str(report_path),
            "atomic_output": True,
            "atomic_report": True,
            "onnx_version": onnx.__version__,
            "source": source_report,
            "detected": {
                "rope_kind": attention_audit["rope_kind"],
                "qk_norm": has_qkn,
                "learned_rope_mapping": (
                    "per-head-final-b"
                    if attention_audit["rope_kind"] == "learned_per_head"
                    else None
                ),
            },
            "stages": {
                "attention_rope_to_ort_mha": attention_audit,
                "rmsnorm_to_sln": rms_audit,
                "residual_sln_to_skip_sln": skip_audit,
            },
            "five_output_numerical_equivalence": numerical,
            "final": {
                "bytes": final_path.stat().st_size,
                "node_count": len(saved_final.graph.node),
                "output_names": final_outputs,
                "checker": checker_with_ort_schema(saved_final),
                "op_counts": op_counts(saved_final),
                "qualified_op_counts": qualified_op_counts(saved_final),
            },
        }
        staged_report = _stage_json(report_path, report)
        try:
            os.replace(final_path, output)
            os.replace(staged_report, report_path)
        finally:
            staged_report.unlink(missing_ok=True)

    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="fixed-batch-1, onnxsim-simplified KataGo transformer ONNX",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination ORT-CPU-specialized FP32 ONNX",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="aggregate JSON (default: <output>.ort_cpu_fp32_report.json)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="optional validation NPZ; gates source/final alignment for all five outputs",
    )
    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=None,
        help="optional assertion; otherwise inferred strictly from graph structure",
    )
    qkn = parser.add_mutually_exclusive_group()
    qkn.add_argument(
        "--require-qk-norm",
        action="store_true",
        help="fail unless every block contains Q/K Norm",
    )
    qkn.add_argument(
        "--require-no-qk-norm",
        action="store_true",
        help="fail if Q/K Norm is detected",
    )
    return parser.parse_args()


def main() -> None:
    optimize(parse_args())


if __name__ == "__main__":
    main()
