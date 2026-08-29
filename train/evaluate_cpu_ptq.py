#!/usr/bin/env python3
"""Measure FP32 and fake-quantized CPU-PTQ loss on training NPZ data.

This evaluator is deliberately independent of a native inference engine. It
uses the checkpoint's model configuration and KataGo's normal data loader and
Metrics implementation, so model/input v102 (CPU-PTQ v106) and v11 (CPU-PTQ
v206) share one loss-validation path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import data_processing_pytorch
from calibrate_cpu_ptq_v106 import (
    PROJECTION_ROLES,
    ProjectionQuantController,
    cpu_ptq_model_version,
)
from export_onnx import load_model_for_export
from metrics_pytorch import Metrics


@dataclass(frozen=True)
class QuantizedManifest:
    path: Path
    qmax: int
    recipe: str
    overrides: dict[str, tuple[torch.Tensor, torch.Tensor]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_weight_sha256(weight: torch.Tensor) -> str:
    values = np.asarray(
        weight.detach().cpu(), dtype="<f4", order="C"
    )
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def load_manifest(path: Path, model: Any) -> QuantizedManifest:
    path = path.resolve()
    expected = {
        f"blocks.{block_index}.{role}"
        for block_index in range(len(model.blocks))
        for role in PROJECTION_ROLES
    }
    overrides: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["format"].item()) != "cpuptq-gptq-v1":
            raise ValueError(f"{path}: unsupported CPU-PTQ manifest format")
        qmax = int(archive["qmax"].item())
        if qmax not in (63, 127):
            raise ValueError(f"{path}: qmax must be 63 or 127")
        if str(archive["activation_quantizer"].item()) != "row-sym":
            raise ValueError(f"{path}: only row-sym activations are supported")
        if float(archive["activation_scale_factor"].item()) != 1.0:
            raise ValueError(f"{path}: activation scale factor must be 1")
        recipe = str(archive["recipe"].item())
        names = [str(name) for name in archive["names"].tolist()]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(
                f"{path}: projection set mismatch: "
                f"missing={sorted(expected - set(names))}, "
                f"extra={sorted(set(names) - expected)}"
            )
        for index, name in enumerate(names):
            _, block_text, role = name.split(".")
            weight = getattr(model.blocks[int(block_text)], role).weight
            codes = np.asarray(archive[f"codes_{index}"])
            scales = np.asarray(archive[f"scales_{index}"])
            recorded_hash = str(archive[f"source_sha256_{index}"].item())
            if codes.dtype != np.int8 or codes.shape != tuple(weight.shape):
                raise ValueError(
                    f"{path}: {name} has invalid code dtype/shape "
                    f"{codes.dtype}/{codes.shape}, expected int8/{tuple(weight.shape)}"
                )
            if scales.shape != (weight.shape[0],):
                raise ValueError(f"{path}: {name} has invalid scale shape")
            if not np.isfinite(scales).all() or not np.all(scales > 0.0):
                raise ValueError(f"{path}: {name} has invalid scales")
            if int(np.max(np.abs(codes.astype(np.int16)))) > qmax:
                raise ValueError(f"{path}: {name} contains a code beyond qmax")
            if recorded_hash != source_weight_sha256(weight):
                raise ValueError(
                    f"{path}: {name} was calibrated from another checkpoint"
                )
            overrides[name] = (
                torch.from_numpy(np.array(codes, copy=True)),
                torch.from_numpy(
                    np.asarray(scales, dtype=np.float32).copy()
                ),
            )
    return QuantizedManifest(path, qmax, recipe, overrides)


def expand_data_paths(path: Path) -> list[Path]:
    path = path.resolve()
    files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not files or any(not item.is_file() for item in files):
        raise ValueError(f"no NPZ data found at {path}")
    return files


def data_loader(
    files: list[Path],
    model: Any,
    board_size: int,
    batch_size: int,
    device: torch.device,
):
    return data_processing_pytorch.read_npz_training_data(
        npz_files=[str(path) for path in files],
        batch_size=batch_size,
        world_size=1,
        rank=0,
        pos_len=board_size,
        device=device,
        symmetry_type="none",
        include_meta=model.get_has_metadata_encoder(),
        history_matrices_type="",
        model_config=model.config,
        require_full_board=True,
        binary_input_nhwc=False,
        filter_full_board_on_load=False,
    )


def evaluate(
    model: Any,
    files: list[Path],
    board_size: int,
    batch_size: int,
    samples: int,
    device: torch.device,
) -> dict[str, float | int]:
    if samples % batch_size != 0:
        raise ValueError("--samples must be divisible by --batch-size")
    metrics_object = Metrics(batch_size, 1, model)
    totals = {"loss": 0.0, "p0": 0.0, "v": 0.0, "weight": 0.0}
    processed = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in data_loader(files, model, board_size, batch_size, device):
            if processed >= samples:
                break
            output = model(
                batch["binaryInputNCHW"],
                batch["globalInputNC"],
                input_meta=(
                    batch["metadataInputNC"]
                    if model.get_has_metadata_encoder()
                    else None
                ),
                disable_mask=True,
            )
            metrics = metrics_object.metrics_dict_batchwise(
                model,
                model.postprocess_output(output),
                None,
                batch,
                is_training=False,
                soft_policy_weight_scale=8.0,
                disable_optimistic_policy=False,
                meta_kata_only_soft_policy=False,
                value_loss_scale=0.6,
                td_value_loss_scales=(0.6, 0.6, 0.6),
                seki_loss_scale=1.0,
                variance_time_loss_scale=1.0,
                main_loss_scale=1.0,
                intermediate_loss_scale=1.0,
                include_model_norms=False,
                assume_full_board=True,
            )
            totals["loss"] += float(metrics["loss_sum"].cpu().item())
            totals["p0"] += float(metrics["p0loss_sum"].cpu().item())
            totals["v"] += float(metrics["vloss_sum"].cpu().item())
            totals["weight"] += float(metrics["wsum"].cpu().item())
            processed += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if processed != samples or totals["weight"] <= 0.0:
        raise ValueError(
            f"requested {samples} rows but evaluated {processed} with "
            f"weight {totals['weight']}"
        )
    return {
        "samples": processed,
        "weightSum": totals["weight"],
        "trainingLossPerWeight": totals["loss"] / totals["weight"],
        "p0LossPerWeight": totals["p0"] / totals["weight"],
        "valueLossPerWeight": totals["v"] / totals["weight"],
        "wallSeconds": elapsed,
    }


def parse_manifest_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("manifest must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("manifest must use NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", action="append", type=parse_manifest_argument,
        default=[], metavar="NAME=PATH",
    )
    parser.add_argument("--board-size", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.board_size <= 0 or args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("board size, samples, and batch size must be positive")
    labels = [name for name, _ in args.manifest]
    if len(labels) != len(set(labels)):
        raise ValueError("manifest labels must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(20260829)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    checkpoint = args.checkpoint.resolve()
    files = expand_data_paths(args.data)
    regular, swa, _, _ = load_model_for_export(
        str(checkpoint), use_swa=True, device=device, pos_len=args.board_size
    )
    del regular
    if swa is None:
        raise ValueError("checkpoint has no SWA model")
    model = swa.to(device).eval()
    model.configure_flex_attention(False)
    source_version = int(model.config["version"])
    target_version = cpu_ptq_model_version(source_version)
    manifests = {
        name: load_manifest(path, model) for name, path in args.manifest
    }

    results: dict[str, dict[str, Any]] = {}
    results["fp32"] = evaluate(
        model, files, args.board_size, args.batch_size, args.samples, device
    )
    reference = results["fp32"]
    print(json.dumps({"fp32": results["fp32"]}, sort_keys=True), flush=True)
    for name, manifest in manifests.items():
        controller = ProjectionQuantController(
            model, manifest.qmax, manifest.overrides
        )
        try:
            measured = evaluate(
                model,
                files,
                args.board_size,
                args.batch_size,
                args.samples,
                device,
            )
        finally:
            controller.close()
        measured.update(
            {
                "qmax": manifest.qmax,
                "recipe": manifest.recipe,
                "manifest": str(manifest.path),
                "manifestSha256": sha256_file(manifest.path),
                "minusFp32": {
                    key: float(measured[key]) - float(reference[key])
                    for key in (
                        "trainingLossPerWeight",
                        "p0LossPerWeight",
                        "valueLossPerWeight",
                    )
                },
            }
        )
        results[name] = measured
        print(json.dumps({name: measured}, sort_keys=True), flush=True)

    report = {
        "kind": "cpuptq-fake-quant-loss-v1",
        "sourceModelVersion": source_version,
        "cpuPtqModelVersion": target_version,
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256_file(checkpoint),
        "boardSize": args.board_size,
        "data": [str(path) for path in files],
        "dataSha256": [sha256_file(path) for path in files],
        "batchSize": args.batch_size,
        "results": results,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
