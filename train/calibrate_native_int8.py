#!/usr/bin/python3
"""Calibrate native CUDA INT8 v105/v205 ranges from training NPZ rows."""

import argparse
import logging
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

import data_processing_pytorch
from load_model import load_model
from metrics_pytorch import Metrics
from native_int8_calibration import (
    AggressiveInt8WeightQDQ,
    BOUNDARY_FIELDS,
    DEFAULT_CANDIDATES,
    LOSS_DELTA_FIELDS,
    LOSS_METRIC_FIELDS,
    ProcessedRowHashes,
    QMAX,
    QMIN,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TransformerBoundaryHooks,
    candidate_thresholds,
    cuda_int8_wire_version,
    dataset_source_record,
    expand_npz_paths,
    make_activation_samples,
    make_saturation_counters,
    require_independent_datasets,
    sha256_file,
    transformer_blocks_in_wire_order,
    validate_calibration_document,
    write_calibration_json,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect FP16 transformer boundary distributions on one NPZ split, "
            "select a percentile policy using real training loss on an independent "
            "validation split, and emit strict native CUDA INT8 calibration JSON."
        )
    )
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-calibration-data", required=True)
    parser.add_argument("-validation-data", required=True)
    parser.add_argument("-output", required=True)
    parser.add_argument("-pos-len", type=int, required=True)
    parser.add_argument("-batch-size", type=int, default=32)
    parser.add_argument("-calibration-max-batches", type=int, default=8)
    parser.add_argument("-validation-max-batches", type=int, default=256)
    parser.add_argument("-use-swa", action="store_true")
    parser.add_argument("-device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("-history-matrices-type", default="")
    parser.add_argument(
        "-require-full-board",
        action="store_true",
        help="Reject non-full-board rows and run the model/metric exact-board paths",
    )
    parser.add_argument("-soft-policy-weight-scale", type=float, default=8.0)
    parser.add_argument("-value-loss-scale", type=float, default=0.6)
    parser.add_argument("-td-value-loss-scales", default="0.6,0.6,0.6")
    parser.add_argument("-seki-loss-scale", type=float, default=1.0)
    parser.add_argument("-variance-time-loss-scale", type=float, default=1.0)
    parser.add_argument("-disable-optimistic-policy", action="store_true")
    parser.add_argument("-meta-kata-only-soft-policy", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    if args.pos_len <= 0 or args.batch_size <= 0:
        raise ValueError("-pos-len and -batch-size must be positive")
    for name in ("calibration_max_batches", "validation_max_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"-{name.replace('_','-')} must be positive")
    td_scales = tuple(float(value) for value in args.td_value_loss_scales.split(","))
    if len(td_scales) != 3 or any(not math.isfinite(value) or value < 0.0 for value in td_scales):
        raise ValueError("-td-value-loss-scales must contain three finite nonnegative values")
    return td_scales


def _data_loader(files, args, model, device):
    return data_processing_pytorch.read_npz_training_data(
        npz_files=[str(path) for path in files],
        batch_size=args.batch_size,
        world_size=1,
        rank=0,
        pos_len=args.pos_len,
        device=device,
        symmetry_type="none",
        include_meta=model.get_has_metadata_encoder(),
        history_matrices_type=args.history_matrices_type,
        model_config=model.config,
        require_full_board=args.require_full_board,
        binary_input_nhwc=False,
        filter_full_board_on_load=False,
    )


def _forward(model, batch, device, require_full_board):
    autocast_enabled = device.type == "cuda"
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16 if autocast_enabled else torch.bfloat16,
        enabled=autocast_enabled,
    ):
        output = model(
            batch["binaryInputNCHW"],
            batch["globalInputNC"],
            input_meta=(
                batch["metadataInputNC"]
                if model.get_has_metadata_encoder()
                else None
            ),
            disable_mask=require_full_board,
        )
    return model.float32ify_output(output) if autocast_enabled else output


def _reset_data_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _run_calibration_collection(
    model, layers, files, args, device, samples, row_hashes
):
    batches = 0
    _reset_data_seed(0xCA11B, device)
    with torch.inference_mode(), TransformerBoundaryHooks(layers, samples=samples):
        for batch in _data_loader(files, args, model, device):
            if args.calibration_max_batches is not None and batches >= args.calibration_max_batches:
                break
            row_hashes.observe_batch(batch, model.get_has_metadata_encoder())
            _forward(model, batch, device, args.require_full_board)
            batches += 1
    if batches <= 0:
        raise ValueError("calibration data yielded no complete batches")
    return batches


def _run_validation(
    model,
    layers,
    files,
    args,
    device,
    td_value_loss_scales,
    thresholds=None,
    saturation=None,
    row_hashes=None,
    weight_qdq=None,
):
    metrics_obj = Metrics(args.batch_size, 1, model)
    loss_sum = 0.0
    p0_loss_sum = 0.0
    value_loss_sum = 0.0
    weight_sum = 0.0
    batches = 0
    hooks = (
        TransformerBoundaryHooks(
            layers,
            thresholds=thresholds,
            saturation=saturation,
            weight_qdq=weight_qdq,
        )
        if thresholds is not None
        else None
    )
    _reset_data_seed(0xEA11, device)
    try:
        with torch.inference_mode():
            for batch in _data_loader(files, args, model, device):
                if args.validation_max_batches is not None and batches >= args.validation_max_batches:
                    break
                if row_hashes is not None:
                    row_hashes.observe_batch(batch, model.get_has_metadata_encoder())
                output = _forward(model, batch, device, args.require_full_board)
                postprocessed = model.postprocess_output(output)
                metrics = metrics_obj.metrics_dict_batchwise(
                    model,
                    postprocessed,
                    None,
                    batch,
                    is_training=False,
                    soft_policy_weight_scale=args.soft_policy_weight_scale,
                    disable_optimistic_policy=args.disable_optimistic_policy,
                    meta_kata_only_soft_policy=args.meta_kata_only_soft_policy,
                    value_loss_scale=args.value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=args.seki_loss_scale,
                    variance_time_loss_scale=args.variance_time_loss_scale,
                    main_loss_scale=1.0,
                    intermediate_loss_scale=1.0,
                    include_model_norms=False,
                    assume_full_board=args.require_full_board,
                )
                loss_sum += float(metrics["loss_sum"].detach().cpu().item())
                p0_loss_sum += float(metrics["p0loss_sum"].detach().cpu().item())
                value_loss_sum += float(metrics["vloss_sum"].detach().cpu().item())
                weight_sum += float(metrics["wsum"].detach().cpu().item())
                batches += 1
    finally:
        if hooks is not None:
            hooks.close()
    if batches <= 0 or weight_sum <= 0.0:
        raise ValueError("validation data yielded no positive-weight complete batches")
    normalized = {
        "trainingLossPerWeight": loss_sum / weight_sum,
        "p0LossPerWeight": p0_loss_sum / weight_sum,
        "valueLossPerWeight": value_loss_sum / weight_sum,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
        raise ValueError(f"validation produced invalid normalized metrics {normalized}")
    return normalized, batches, weight_sum


def calibrate(args):
    td_value_loss_scales = _validate_args(args)
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    calibration_files = expand_npz_paths(Path(args.calibration_data))
    validation_files = expand_npz_paths(Path(args.validation_data))
    require_independent_datasets(calibration_files, validation_files)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    # The clipped SwiGLU subpath uses FP32 matmul over integer-valued codes to
    # reproduce INT32 accumulation exactly. TF32 would violate that contract.
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    raw_model, swa_model, _ = load_model(
        str(checkpoint),
        args.use_swa,
        device=device,
        pos_len=args.pos_len,
        verbose=True,
    )
    model = swa_model if args.use_swa else raw_model
    wire_version = cuda_int8_wire_version(int(model.config["version"]))
    model.eval()
    model.configure_flex_attention(False)
    layers = transformer_blocks_in_wire_order(model)
    if not layers:
        raise ValueError("checkpoint has no native TransformerRoPEGQABlock layers")
    layer_order = [name for name, _ in layers]
    for name, block in layers:
        if not block.use_swiglu:
            raise ValueError(f"{name}: native INT8 PTQ requires SwiGLU")

    logging.info(
        "Collecting FP16 boundary distributions from %d layer(s) on %s",
        len(layers),
        device,
    )
    samples = make_activation_samples(layer_order)
    calibration_row_hashes = ProcessedRowHashes()
    calibration_batches = _run_calibration_collection(
        model, layers, calibration_files, args, device, samples,
        calibration_row_hashes,
    )
    thresholds_by_candidate, calibration_saturation = candidate_thresholds(
        samples, DEFAULT_CANDIDATES
    )

    validation_row_hashes = ProcessedRowHashes()
    baseline_metrics, validation_batches, validation_weight = _run_validation(
        model,
        layers,
        validation_files,
        args,
        device,
        td_value_loss_scales,
        row_hashes=validation_row_hashes,
    )
    overlap_rows = len(
        calibration_row_hashes.digests & validation_row_hashes.digests
    )
    if overlap_rows != 0:
        raise ValueError(
            "Calibration and validation processed model inputs overlap: "
            f"{overlap_rows} unique row(s)"
        )
    baseline_loss = baseline_metrics["trainingLossPerWeight"]
    candidate_losses = {}
    candidate_metrics = {}
    validation_saturation = {}
    with AggressiveInt8WeightQDQ(layers) as weight_qdq:
        weight_only_metrics, weight_batches, weight_total = _run_validation(
            model,
            layers,
            validation_files,
            args,
            device,
            td_value_loss_scales,
        )
        if weight_batches != validation_batches or weight_total != validation_weight:
            raise ValueError("validation data traversal changed for weight-only QDQ")
        weight_only_loss = weight_only_metrics["trainingLossPerWeight"]
        weight_scales = {
            layer_name: dict(scales)
            for layer_name, scales in weight_qdq.scales.items()
        }
        for candidate_name, _ in DEFAULT_CANDIDATES:
            counters = make_saturation_counters(layer_order)
            metrics, candidate_batches, candidate_weight = _run_validation(
                model,
                layers,
                validation_files,
                args,
                device,
                td_value_loss_scales,
                thresholds=thresholds_by_candidate[candidate_name],
                saturation=counters,
                weight_qdq=weight_qdq,
            )
            if candidate_batches != validation_batches or candidate_weight != validation_weight:
                raise ValueError("validation data traversal changed between percentile candidates")
            candidate_metrics[candidate_name] = metrics
            candidate_losses[candidate_name] = metrics["trainingLossPerWeight"]
            validation_saturation[candidate_name] = {
                layer_name: {
                    field: counters[layer_name][field].rate()
                    for field in BOUNDARY_FIELDS
                }
                for layer_name in layer_order
            }
            logging.info(
                "candidate=%s validation_loss=%.10g p0loss=%.10g vloss=%.10g",
                candidate_name,
                metrics["trainingLossPerWeight"],
                metrics["p0LossPerWeight"],
                metrics["valueLossPerWeight"],
            )

    candidate_names = [name for name, _ in DEFAULT_CANDIDATES]
    chosen_candidate = min(
        candidate_names, key=lambda name: (candidate_losses[name], candidate_names.index(name))
    )
    layers_json = []
    for index, layer_name in enumerate(layer_order):
        selected = thresholds_by_candidate[chosen_candidate][layer_name]
        candidate_json = {}
        for candidate_name in candidate_names:
            candidate_json[candidate_name] = {
                "thresholds": thresholds_by_candidate[candidate_name][layer_name],
                "calibrationSaturationRates": calibration_saturation[candidate_name][layer_name],
            }
        layer_json = {
            "index": index,
            "name": layer_name,
            **selected,
            "calibrationSample": {
                field: samples[layer_name][field].summary()
                for field in BOUNDARY_FIELDS
            },
            "candidates": candidate_json,
            "validationSaturationRates": {
                candidate_name: validation_saturation[candidate_name][layer_name]
                for candidate_name in candidate_names
            },
            "weightQdqScales": weight_scales[layer_name],
        }
        layers_json.append(layer_json)

    checkpoint_record = {
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }
    selected_loss = candidate_losses[chosen_candidate]

    def with_deltas(metrics):
        result = dict(metrics)
        for field, delta_field in zip(LOSS_METRIC_FIELDS, LOSS_DELTA_FIELDS):
            result[delta_field] = metrics[field] - baseline_metrics[field]
        return result

    calibration_rows = calibration_row_hashes.summary()
    validation_rows = validation_row_hashes.summary()
    document = {
        "schema": SCHEMA_NAME,
        "schemaVersion": SCHEMA_VERSION,
        "wireVersion": wire_version,
        "source": {
            "checkpoint": checkpoint_record,
            "calibrationData": dataset_source_record(calibration_files),
            "validationData": dataset_source_record(validation_files),
            "processedRows": {
                "calibrationRows": calibration_rows["rows"],
                "calibrationUniqueRows": calibration_rows["uniqueRows"],
                "calibrationSetSha256": calibration_rows["setSha256"],
                "validationRows": validation_rows["rows"],
                "validationUniqueRows": validation_rows["uniqueRows"],
                "validationSetSha256": validation_rows["setSha256"],
                "overlapRows": overlap_rows,
            },
        },
        "evaluation": {
            "modelState": "swa" if args.use_swa else "raw",
            "useSwa": bool(args.use_swa),
            "posLen": args.pos_len,
        },
        "quantization": {
            "dtype": "int8",
            "qmin": QMIN,
            "qmax": QMAX,
            "zeroPoint": 0,
            "rounding": "roundTiesToEven",
            "validationArithmetic": {
                "overall": "pytorchFakeQdq",
                "fp16BoundaryFields": list(BOUNDARY_FIELDS[:3]),
                "clippedFactorProduct":
                    "nativeCodeDomainInt8DotFp32ScaleDirectRequant",
                "clippedSilu": "pytorchFp32SimulationNotBitExactCutlass",
                "noClipProduct": "modelFloatProductBoundaryQdq",
                "productDownFeed": "pytorchDequantizedSurrogate",
            },
            "candidates": candidate_names,
            "weightQdq": {
                "qmin": QMIN,
                "qmax": QMAX,
                "zeroPoint": 0,
                "rounding": "roundTiesToEven",
                "scale": "float32GroupMaxAbsDiv127",
                "groups": ["qkvShared", "attentionOut", "ffnUp", "ffnGate", "ffnDown"],
            },
        },
        "layerOrder": layer_order,
        "layers": layers_json,
        "selection": {
            "metric": "trainingLossPerWeight",
            "baselineLoss": baseline_loss,
            "weightOnlyLoss": weight_only_loss,
            "candidateLosses": candidate_losses,
            "baselineMetrics": baseline_metrics,
            "weightOnlyMetrics": with_deltas(weight_only_metrics),
            "candidateMetrics": {
                name: with_deltas(candidate_metrics[name])
                for name in candidate_names
            },
            "chosenCandidate": chosen_candidate,
            "selectedLoss": selected_loss,
            "lossDelta": selected_loss - baseline_loss,
        },
    }
    validate_calibration_document(
        document,
        checkpoint_path=checkpoint,
        layer_order=layer_order,
        use_swa=args.use_swa,
        pos_len=args.pos_len,
        wire_version=wire_version,
    )
    write_calibration_json(output, document)
    logging.info(
        "Wrote %s: calibration_batches=%d validation_batches=%d chosen=%s "
        "baseline_loss=%.10g selected_loss=%.10g",
        output,
        calibration_batches,
        validation_batches,
        chosen_candidate,
        baseline_loss,
        selected_loss,
    )
    return document


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    calibrate(_parse_args())


if __name__ == "__main__":
    main()
