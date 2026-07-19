#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import contextlib
import json
import datetime
from datetime import timezone
import gc
import shutil
import glob
import numpy as np
import itertools
import copy
import atexit
from collections import defaultdict
from typing import Dict, List

import torch
import torch._dynamo
torch._dynamo.config.recompile_limit = 32
import torch.ao.quantization
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
#torch.autograd.set_detect_anomaly(True) # Should set GradScaler(init_scale=2.0)

import modelconfigs
from model_pytorch import Model, ExtraOutputs, MetadataEncoder
from metrics_pytorch import Metrics
import load_model
import data_processing_pytorch
from metrics_logging import accumulate_metrics, log_metrics, clear_metric_nonfinite
from muon_kissin import MuonWithAuxAdamKimi
torch.set_float32_matmul_precision('high')


_ALLOWED_COMPILE_MODES = (
    "default",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
_ALLOWED_SDPA_BACKENDS = ("auto", "flash", "cudnn", "efficient", "math")
_ALLOWED_INPUT_MEMORY_FORMATS = ("nhwc", "nchw")


def add_amp_arguments(argument_container):
    """Add mutually exclusive mixed-precision command-line options."""
    amp_args = argument_container.add_mutually_exclusive_group()
    amp_args.add_argument(
        '-use-fp16', help='Use FP16 AMP training', required=False, action='store_true'
    )
    amp_args.add_argument(
        '-use-bf16', help='Use BF16 AMP training', required=False, action='store_true'
    )


def validate_amp_qat_options(use_fp16: bool, use_bf16: bool, qat_int8: bool):
    if use_fp16 and use_bf16:
        raise ValueError("FP16 AMP and BF16 AMP are mutually exclusive")
    if qat_int8:
        assert not use_fp16, "QAT INT8 enabled. FP16/AMP is not supported. Remove this if it not report any error."
        assert not use_bf16, "QAT INT8 enabled. BF16/AMP is not supported. Remove this if it not report any error."


def validate_full_board_filter_options(
    disable_mask: bool, filter_full_board_on_load: bool
):
    if filter_full_board_on_load and not disable_mask:
        raise ValueError(
            "-filter-full-board-on-load requires -disable-mask because filtering "
            "is only needed for the mask-free training path"
        )


def validate_flex_attention_options(
    enabled: bool,
    disable_mask: bool,
    no_compile: bool,
    qat_int8: bool,
):
    if not enabled:
        return
    if disable_mask:
        raise ValueError("-use-flex-attention cannot be combined with -disable-mask")
    if no_compile:
        raise ValueError("-use-flex-attention requires torch.compile; remove -no-compile")
    if qat_int8:
        raise ValueError("-use-flex-attention is not supported with -qat-int8")


def amp_autocast_context(use_fp16: bool, use_bf16: bool):
    """Return the requested CUDA autocast context without changing legacy defaults."""
    if use_fp16:
        return torch.amp.autocast(device_type='cuda')
    if use_bf16:
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def create_grad_scaler(use_fp16: bool, use_bf16: bool):
    """Grad scaling is needed for FP16 only; BF16 uses unscaled gradients."""
    if use_fp16 and use_bf16:
        raise ValueError("FP16 AMP and BF16 AMP are mutually exclusive")
    return torch.amp.GradScaler("cuda") if use_fp16 else None


def backward_and_unscale(loss, optimizer, scaler):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)


def optimizer_step(optimizer, scaler):
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()


# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

    description = """
    Train neural net on Go positions from npz files of batches from selfplay.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )

    required_args.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
    required_args.add_argument('-datadir', help='Directory with a train and val subdir of npz data, output by shuffle.py', required=True)
    optional_args.add_argument('-exportdir', help='Directory to export models periodically', required=False)
    optional_args.add_argument('-exportprefix', help='Prefix to append to names of models', required=False)
    optional_args.add_argument('-initial-checkpoint', help='If no training checkpoint exists, initialize from this checkpoint', required=False)

    required_args.add_argument('-pos-len', help='Spatial edge length of expected training data, e.g. 19 for 19x19 Go', type=int, required=True)
    required_args.add_argument('-batch-size', help='Per-GPU batch size to use for training', type=int, required=True)
    optional_args.add_argument('-samples-per-epoch', help='Number of data samples to consider as one epoch', type=int, required=False)
    optional_args.add_argument('-history-matrices-type', help='History matrices mode: "go", "gomoku", "none", or empty', type=str, default="", required=False)
    optional_args.add_argument('-symmetry-type', help='Data symmetry type. "none" to disable, "xyt" for Go/Gomoku, "x+y" for Hex, "x" for chess, "t" for tiaoqi', type=str, default="xyt", required=False)

    
    optional_args.add_argument('-model-kind', help='String name for what model config to use', required=False)
    optional_args.add_argument('-lr-base', help='LR base', type=float, default=6e-6, required=False)
    optional_args.add_argument('-lr-scale', help='LR multiplier on the hardcoded schedule', type=float, required=False)
    optional_args.add_argument('-lr-scale-auto-type', help='LR auto scaling type',type=str, required=False, default="")
    optional_args.add_argument('-wd-scale', help='Weight decay scale', type=float, default=1.0, required=False)
    
    optional_args.add_argument('-muon-momentum', type=float, help='momentum of Muon optimizer', default=0.95, required=False)
    
    optional_args.add_argument('-gnorm-clip-scale', help='Multiplier on gradient clipping threshold', type=float, required=False)
    optional_args.add_argument('-sub-epochs', help='Reload training data up to this many times per epoch', type=int, default=1, required=False)
    optional_args.add_argument('-swa-period-samples', help='How frequently to average an SWA sample, in samples', type=float, required=False)
    optional_args.add_argument('-swa-scales', help='Number of samples to average in expectation together for SWA', type=str, required=False)

    optional_args.add_argument('-multi-gpus', help='Use multiple gpus, comma-separated device ids', required=False)
    add_amp_arguments(optional_args)
    optional_args.add_argument('-qat-int8', help='Enable INT8 QAT', required=False, action='store_true')    
    optional_args.add_argument('-master-port', help='Localhost port', default=23456, type=int, required=False)
    optional_args.add_argument('-no-compile', help='Do not torch.compile', required=False, action='store_true')
    optional_args.add_argument(
        '-compile-mode',
        choices=_ALLOWED_COMPILE_MODES,
        default='default',
        help='torch.compile mode (default: default)',
        required=False,
    )
    optional_args.add_argument(
        '-sdpa-backend',
        choices=_ALLOWED_SDPA_BACKENDS,
        default='auto',
        help='CUDA SDPA backend selection (default: auto)',
        required=False,
    )
    optional_args.add_argument(
        '-input-memory-format',
        choices=_ALLOWED_INPUT_MEMORY_FORMATS,
        default='nhwc',
        help='Training input memory format: nhwc or nchw (default: nhwc)',
        required=False,
    )
    optional_args.add_argument(
        '-disable-mask',
        help=(
            'Assume every training sample uses the full pos-len by pos-len board, '
            'validate that assumption while loading, and omit model masks'
        ),
        required=False,
        action='store_true',
    )
    optional_args.add_argument(
        '-filter-full-board-on-load',
        help=(
            'With -disable-mask, discard non-full-board training rows while loading '
            'instead of rejecting the entire NPZ file'
        ),
        required=False,
        action='store_true',
    )
    optional_args.add_argument(
        '-use-flex-attention',
        help='Use a per-sample FlexAttention BlockMask for masked transformer attention',
        required=False,
        action='store_true',
    )

    optional_args.add_argument('-epochs-per-export', help='Export model once every this many epochs', type=int, required=False)
    optional_args.add_argument('-export-prob', help='Export model with this probablity', type=float, required=False)
    optional_args.add_argument('-max-epochs-this-instance', help='Terminate training after this many more epochs', type=int, required=False)
    optional_args.add_argument('-max-training-samples', help='Terminate training after about this many training steps in samples', type=int, required=False)
    optional_args.add_argument('-sleep-seconds-per-epoch', help='Sleep this long between epochs', type=int, required=False)
    optional_args.add_argument('-max-train-bucket-per-new-data', help='When data added, add this many train rows per data row to bucket', type=float, required=False)
    optional_args.add_argument('-max-train-bucket-size', help='Approx total number of train rows allowed if data stops', type=float, required=False)
    optional_args.add_argument('-max-train-steps-since-last-reload', help='Approx total of training allowed if shuffling stops', type=float, required=False)
    optional_args.add_argument('-stop-when-train-bucket-limited', help='Terminate due to train bucket rather than waiting for more', required=False, action='store_true')
    optional_args.add_argument('-max-val-samples', help='Approx max of validation samples per epoch', type=int, required=False)
    optional_args.add_argument('-randomize-val', help='Randomize order of validation files', required=False, action='store_true')
    optional_args.add_argument('-no-export', help='Do not export models', required=False, action='store_true')
    optional_args.add_argument('-no-repeat-files', help='Track what shuffled data was used and do not repeat, even when killed and resumed', required=False, action='store_true')
    optional_args.add_argument('-quit-if-no-data', help='If no data, quit instead of waiting for data', required=False, action='store_true')

    optional_args.add_argument('-gnorm-stats-debug', required=False, action='store_true')

    optional_args.add_argument('-lookahead-k', help='Use lookahead optimizer', type=int, default=6, required=False)
    optional_args.add_argument('-lookahead-alpha', help='Use lookahead optimizer, 1.0 to disable', type=float, default=1.0, required=False)
    optional_args.add_argument('-lookahead-print', help='Only print on lookahead syncs', required=False, action='store_true')
    optional_args.add_argument('-brenorm-avg-momentum', type=float, help='Set brenorm running avg rate to this value', required=False)
    optional_args.add_argument('-brenorm-target-rmax', type=float, help='Gradually adjust brenorm rmax to this value', required=False)
    optional_args.add_argument('-brenorm-target-dmax', type=float, help='Gradually adjust brenorm dmax to this value', required=False)
    optional_args.add_argument('-brenorm-adjustment-scale', type=float, help='How many samples to adjust brenorm params all but 1/e of the way to target', required=False)

    optional_args.add_argument('-soft-policy-weight-scale', type=float, default=8.0, help='Soft policy loss coeff', required=False)
    optional_args.add_argument('-disable-optimistic-policy', help='Disable optimistic policy', required=False, action='store_true')
    optional_args.add_argument('-meta-kata-only-soft-policy', help='Mask soft policy on non-kata rows using sgfmeta', required=False, action='store_true')
    optional_args.add_argument('-value-loss-scale', type=float, default=0.6, help='Additional value loss coeff', required=False)
    optional_args.add_argument('-td-value-loss-scales', type=str, default="0.6,0.6,0.6", help='Additional td value loss coeffs, 3 comma separated values', required=False)
    optional_args.add_argument('-seki-loss-scale', type=float, default=1.0, help='Additional seki loss coeff', required=False)
    optional_args.add_argument('-variance-time-loss-scale', type=float, default=1.0, help='Additional variance time loss coeff', required=False)

    optional_args.add_argument('-main-loss-scale', type=float, help='Loss factor scale for main head', required=False)
    optional_args.add_argument('-intermediate-loss-scale', type=float, help='Loss factor scale for intermediate head', required=False)

    args = vars(parser.parse_args())
    try:
        validate_full_board_filter_options(
            args["disable_mask"], args["filter_full_board_on_load"]
        )
    except ValueError as error:
        parser.error(str(error))


def get_longterm_checkpoints_dir(traindir):
    return os.path.join(traindir,"longterm_checkpoints")

def make_dirs(args):
    traindir = args["traindir"]
    exportdir = args["exportdir"]

    if not os.path.exists(traindir):
        os.makedirs(traindir)
    if exportdir is not None and not os.path.exists(exportdir):
        os.makedirs(exportdir)

    longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)
    if not os.path.exists(longterm_checkpoints_dir):
        os.makedirs(longterm_checkpoints_dir)

def multiprocessing_setup(rank: int, world_size: int, master_port: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{master_port}'
    logging.info("Running torch.distributed.init_process_group")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Returned from torch.distributed.init_process_group, my rank = {rank}, world_size={world_size}")

def multiprocessing_cleanup():
    torch.distributed.destroy_process_group()


_DDP_STATIC_GRAPH_ENV = "KATAGO_DDP_STATIC_GRAPH"
_DDP_GRADIENT_AS_BUCKET_VIEW_ENV = "KATAGO_DDP_GRADIENT_AS_BUCKET_VIEW"
_DDP_BROADCAST_BUFFERS_ENV = "KATAGO_DDP_BROADCAST_BUFFERS"
_DDP_ALIGN_CONV1X1_WEIGHT_STRIDES_ENV = "KATAGO_DDP_ALIGN_CONV1X1_WEIGHT_STRIDES"

_RANK0_ACTION_PROCEED = 0
_RANK0_ACTION_RETRY = 1
_RANK0_ACTION_STOP = 2


def _get_strict_env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag, accepting only the exact values 0 and 1."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(
        f"Environment variable {name} must be exactly '0' or '1', got {value!r}"
    )


def resolve_input_nhwc(input_memory_format: str) -> bool:
    """Return whether training inputs should use NHWC memory layout."""
    if input_memory_format not in _ALLOWED_INPUT_MEMORY_FORMATS:
        allowed = "|".join(_ALLOWED_INPUT_MEMORY_FORMATS)
        raise ValueError(
            f"-input-memory-format must be one of {allowed}, "
            f"got {input_memory_format!r}"
        )
    return input_memory_format == "nhwc"


def validate_compile_mode(mode: str) -> str:
    if mode in _ALLOWED_COMPILE_MODES:
        return mode
    allowed = "|".join(_ALLOWED_COMPILE_MODES)
    raise ValueError(
        f"-compile-mode must be one of {allowed}, got {mode!r}"
    )


def configure_sdpa_backend(backend: str) -> str:
    """Select one CUDA SDPA backend, or leave all backends enabled for auto."""
    if backend not in _ALLOWED_SDPA_BACKENDS:
        raise ValueError(
            f"-sdpa-backend must be one of "
            f"{'|'.join(_ALLOWED_SDPA_BACKENDS)}, got {backend!r}"
        )

    enabled = (
        set(_ALLOWED_SDPA_BACKENDS[1:]) if backend == "auto" else {backend}
    )
    torch.backends.cuda.enable_flash_sdp("flash" in enabled)
    torch.backends.cuda.enable_cudnn_sdp("cudnn" in enabled)
    torch.backends.cuda.enable_mem_efficient_sdp("efficient" in enabled)
    torch.backends.cuda.enable_math_sdp("math" in enabled)
    return backend


def resolve_compile_training_loss(
    requested: bool,
    no_compile: bool,
    qat_int8: bool,
) -> bool:
    """Honor global compile/QAT constraints for the separately compiled loss."""
    return requested and not no_compile and not qat_int8


def set_snapshot_metrics(metric_sums, metric_weights, metrics, keys):
    """Store point-in-time metrics without depending on moving-average weight."""
    for key in keys:
        metric_sums[key] = metrics[key]
        metric_weights[key] = 1.0


def broadcast_rank0_action(local_action, rank: int, world_size: int, device) -> int:
    """Broadcast a low-frequency control-flow decision made by rank 0."""
    if world_size <= 1:
        if local_action is None:
            raise ValueError("rank 0 action must be provided for single-process training")
        return int(local_action)

    action_value = int(local_action) if rank == 0 else _RANK0_ACTION_PROCEED
    action_tensor = torch.tensor(
        [action_value],
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.broadcast(action_tensor, src=0)
    return int(action_tensor.item())


def get_local_validation_model(training_model, raw_model, world_size: int):
    """Return a local forward module that cannot initiate DDP collectives."""
    if world_size <= 1:
        return training_model

    current = training_model
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, DistributedDataParallel):
            # The common compile-before-DDP path preserves the compiled local
            # module here, while bypassing DDP's forward-time buffer broadcast.
            return current.module
        original_module = getattr(current, "_orig_mod", None)
        if original_module is None or original_module is current:
            break
        current = original_module

    # Unknown wrappers are not safe to call from rank 0 alone.
    return raw_model


@torch.no_grad()
def align_conv1x1_weight_strides_for_ddp(raw_model) -> int:
    """Match the layout produced by convolution backward for 1x1 weights.

    CUDA convolution backward commonly returns a logically contiguous
    ``[out, in, 1, 1]`` gradient with strides ``[in, 1, in, in]``. PyTorch's
    default parameter allocation uses ``[in, 1, 1, 1]`` instead. Both layouts
    address exactly the same storage because the last dimensions have size 1,
    but DDP otherwise warns and copies into a differently-strided bucket view.
    """
    aligned_count = 0
    for module in raw_model.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        weight = module.weight
        if (
            weight is None
            or weight.ndim != 4
            or tuple(weight.shape[2:]) != (1, 1)
        ):
            continue
        in_channels_per_group = weight.shape[1]
        desired_stride = (
            in_channels_per_group,
            1,
            in_channels_per_group,
            in_channels_per_group,
        )
        if weight.stride() == desired_stride:
            continue
        aligned_weight = torch.empty_strided(
            weight.shape,
            desired_stride,
            dtype=weight.dtype,
            device=weight.device,
        )
        aligned_weight.copy_(weight)
        module.weight = torch.nn.Parameter(
            aligned_weight,
            requires_grad=weight.requires_grad,
        )
        aligned_count += 1
    return aligned_count


def wrap_model_for_training(
    raw_model,
    device,
    world_size: int,
    no_compile: bool,
    compile_mode: str = "default",
    qat_int8: bool = False,
):
    """Apply torch.compile and DDP without changing ownership of ``raw_model``."""
    compile_mode = None if no_compile else validate_compile_mode(compile_mode)
    if world_size <= 1:
        logging.info(
            "Training model wrapper: single GPU, compile=%s, compile_mode=%s; "
            "DDP environment flags ignored",
            not no_compile,
            compile_mode,
        )
        if no_compile:
            return raw_model
        return torch.compile(raw_model, mode=compile_mode)

    static_graph = _get_strict_env_bool(_DDP_STATIC_GRAPH_ENV, default=True)
    gradient_as_bucket_view = _get_strict_env_bool(
        _DDP_GRADIENT_AS_BUCKET_VIEW_ENV, default=True
    )
    align_conv1x1_weight_strides = _get_strict_env_bool(
        _DDP_ALIGN_CONV1X1_WEIGHT_STRIDES_ENV, default=True
    )
    aligned_conv1x1_count = (
        align_conv1x1_weight_strides_for_ddp(raw_model)
        if align_conv1x1_weight_strides
        else 0
    )
    # Plain batch norm uses the current batch during training, so synchronizing
    # its running statistics before every forward does not affect gradients or
    # rank 0's checkpointed statistics. Batch renorm does consume its running
    # statistics in the training forward and therefore keeps DDP's behavior.
    # QAT also keeps broadcasts because observer/FakeQuant buffers affect
    # later forwards.
    norm_kind = raw_model.get_norm_kind()
    broadcast_buffers = _get_strict_env_bool(
        _DDP_BROADCAST_BUFFERS_ENV,
        default=qat_int8 or norm_kind in ("brenorm", "fixbrenorm"),
    )
    logging.info(
        "Training model wrapper: DDP world_size=%d, compile=%s, "
        "compile_mode=%s, static_graph=%s, gradient_as_bucket_view=%s, "
        "broadcast_buffers=%s, aligned_conv1x1_weights=%d",
        world_size,
        not no_compile,
        compile_mode,
        static_graph,
        gradient_as_bucket_view,
        broadcast_buffers,
        aligned_conv1x1_count,
    )

    ddp_kwargs = {
        "device_ids": [device],
        "broadcast_buffers": broadcast_buffers,
    }
    if static_graph:
        ddp_kwargs["static_graph"] = True
    if gradient_as_bucket_view:
        ddp_kwargs["gradient_as_bucket_view"] = True

    if no_compile:
        return DistributedDataParallel(raw_model, **ddp_kwargs)

    compiled_model = torch.compile(raw_model, mode=compile_mode)
    return DistributedDataParallel(compiled_model, **ddp_kwargs)


import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm  # 覆盖所有 BatchNorm 类型


def reset_nan_batchnorm(model, verbose=True):
    """
    Reset NaN/Inf in BatchNorm layers
    """
    has_nan = False
    
    for module in model.modules():
        for name, param in module.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                if verbose:
                    logging.info(f"Reset {name} in {module.__class__.__name__} (include NaN/Inf)")
                if "running_mean" in name:
                    nn.init.zeros_(param)  # weight 初始化为 1
                elif "running_var" in name:
                    nn.init.ones_(param)  # bias 初始化为 0
                elif "running_std" in name:
                    nn.init.ones_(param)  # bias 初始化为 0
                else:
                    logging.info("Unrecoverable NAN")
                    assert(False)  # bias 初始化为 0
                has_nan = True
        
        for name, buf in module.named_buffers():
            if torch.isnan(buf).any() or torch.isinf(buf).any():
                if verbose:
                    logging.info(f"Reset {name} in {module.__class__.__name__} (include NaN/Inf)")
                if "running_mean" in name:
                    nn.init.zeros_(buf)  # weight 初始化为 1
                elif "running_var" in name:
                    nn.init.ones_(buf)  # bias 初始化为 0
                elif "running_std" in name:
                    nn.init.ones_(buf)  # bias 初始化为 0
                else:
                    logging.info("Unrecoverable NAN")
                    assert(False)  # bias 初始化为 0
                has_nan = True
    
    if verbose and not has_nan:
        logging.info("No NaN/Inf in BatchNorm layers")
    
    return has_nan

#def sync_swa_buffers_shape(swa_model, raw_model):
    # swa_model is AveragedModel
    # raw_model is the training model
    
#    if not hasattr(swa_model, "module"):
#        assert(False)
    # logging.info("Debug: Comparing SWA buffers with Raw buffers")
    # Iterate over the underlying model's buffers
#    for (name_swa, buf_swa), (name_raw, buf_raw) in zip(swa_model.module.named_buffers(), raw_model.named_buffers()):
#        logging.info(f"Checking buffer {name_swa} {name_raw}: SWA shape {buf_swa.shape}, Raw shape {buf_raw.shape}")

#def fix_qat_zero_points(model):
    # Iterate over all modules and find those that have zero_point
    # Round zero_point to nearest integer and clamp to quant_min/quant_max
#    for m in model.modules():
        # Check if it's a FakeQuantize module (which has zero_point, scale, quant_min, quant_max)
#        if hasattr(m, "zero_point") and hasattr(m, "quant_min") and hasattr(m, "quant_max"):
#            with torch.no_grad():
#                m.zero_point.copy_(m.zero_point.round().clamp(m.quant_min, m.quant_max))



def main(rank: int, world_size: int, args, multi_gpu_device_ids, readpipes, writepipes, barrier):
    traindir = args["traindir"]
    datadir = args["datadir"]
    exportdir = args["exportdir"]
    exportprefix = args["exportprefix"]
    initial_checkpoint = args["initial_checkpoint"]

    pos_len = args["pos_len"]
    batch_size = args["batch_size"]
    samples_per_epoch = args["samples_per_epoch"]
    symmetry_type=args["symmetry_type"]
    model_kind = args["model_kind"]
    lr_base = args["lr_base"]
    lr_scale = args["lr_scale"]
    wd_scale = args["wd_scale"]
    muon_momentum = args["muon_momentum"]
    lr_scale_auto_type = args["lr_scale_auto_type"]
    gnorm_clip_scale = args["gnorm_clip_scale"]
    sub_epochs = args["sub_epochs"]
    swa_period_samples = args["swa_period_samples"]
    swa_scales = [float(x) for x in args["swa_scales"].split(",")]
    lookahead_k = args["lookahead_k"]
    lookahead_alpha = args["lookahead_alpha"]
    lookahead_print = args["lookahead_print"]
    history_matrices_type = args["history_matrices_type"]

    use_fp16 = args["use_fp16"]
    use_bf16 = args["use_bf16"]
    master_port = args["master_port"]
    no_compile = args["no_compile"]
    compile_mode = args["compile_mode"]
    sdpa_backend = args["sdpa_backend"]
    input_memory_format = args["input_memory_format"]
    disable_mask = args["disable_mask"]
    filter_full_board_on_load = args["filter_full_board_on_load"]
    validate_full_board_filter_options(disable_mask, filter_full_board_on_load)
    use_flex_attention = args["use_flex_attention"]
    input_nhwc = resolve_input_nhwc(input_memory_format)
    
    epochs_per_export = args["epochs_per_export"]
    export_prob = args["export_prob"]
    max_epochs_this_instance = args["max_epochs_this_instance"]
    max_training_samples = args["max_training_samples"]
    sleep_seconds_per_epoch = args["sleep_seconds_per_epoch"]
    max_train_bucket_per_new_data = args["max_train_bucket_per_new_data"]
    max_train_bucket_size = args["max_train_bucket_size"]
    max_train_steps_since_last_reload = args["max_train_steps_since_last_reload"]
    stop_when_train_bucket_limited = args["stop_when_train_bucket_limited"]
    max_val_samples = args["max_val_samples"]
    randomize_val = args["randomize_val"]
    no_export = args["no_export"]
    no_repeat_files = args["no_repeat_files"]
    quit_if_no_data = args["quit_if_no_data"]
    qat_int8 = args["qat_int8"]

    validate_amp_qat_options(use_fp16, use_bf16, qat_int8)
    validate_flex_attention_options(
        use_flex_attention,
        disable_mask,
        no_compile,
        qat_int8,
    )
    if qat_int8:
        assert no_compile, "QAT INT8 enabled. Compilation is not supported. Remove this if it not report any error."

    gnorm_stats_debug = args["gnorm_stats_debug"]
    model_norms_only_at_print = _get_strict_env_bool(
        "KATAGO_MODEL_NORMS_ONLY_AT_PRINT", default=True
    )
    compile_training_loss_requested = _get_strict_env_bool(
        "KATAGO_COMPILE_TRAINING_LOSS", default=True
    )
    compile_training_loss = resolve_compile_training_loss(
        compile_training_loss_requested,
        no_compile,
        qat_int8,
    )
    if compile_training_loss_requested and not compile_training_loss:
        logging.info(
            "Disabling compiled training loss because no_compile=%s and qat_int8=%s",
            no_compile,
            qat_int8,
        )

    brenorm_target_rmax = args["brenorm_target_rmax"]
    brenorm_target_dmax = args["brenorm_target_dmax"]
    brenorm_avg_momentum = args["brenorm_avg_momentum"]
    brenorm_adjustment_scale = args["brenorm_adjustment_scale"]

    soft_policy_weight_scale = args["soft_policy_weight_scale"]
    disable_optimistic_policy = args["disable_optimistic_policy"]
    meta_kata_only_soft_policy = args["meta_kata_only_soft_policy"]
    value_loss_scale = args["value_loss_scale"]
    td_value_loss_scales = [float(x) for x in args["td_value_loss_scales"].split(",")]
    seki_loss_scale = args["seki_loss_scale"]
    variance_time_loss_scale = args["variance_time_loss_scale"]

    main_loss_scale = args["main_loss_scale"]
    intermediate_loss_scale = args["intermediate_loss_scale"]

    if lr_scale is None:
        lr_scale = 1.0
    if lr_scale_auto_type != "":
        assert lr_scale == 1.0, "Cannot specify both lr_scale and lr_scale_auto_type"
        
    if samples_per_epoch is None:
        samples_per_epoch = 1000000
    if max_train_bucket_size is None:
        max_train_bucket_size = 1.0e30
    if epochs_per_export is None:
        epochs_per_export = 1
    if swa_period_samples is None:
        swa_period_samples = max(1, samples_per_epoch // 2)
    if swa_scales is None:
        swa_scales = [8,]

    assert lookahead_alpha > 0.0 and lookahead_alpha <= 1.0
    if lookahead_alpha >= 1.0:  # 1.0 means to disable lookahead optimizer
        lookahead_alpha = None
        lookahead_k = None

    longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)

    assert (swa_period_samples is not None) 
    assert (swa_scales is not None) 
    assert (len(swa_scales)>0 ) 
    assert (lookahead_k is None) == (lookahead_alpha is None)

    from qat_helper import get_tensorrt_qat_qconfig, is_qat_checkpoint, disable_qat_for_unsupported_modules
    #if qat_int8:
    #    from qat_helper import get_tensorrt_qat_qconfig, is_qat_checkpoint, disable_qat_for_unsupported_modules
    # SET UP LOGGING -------------------------------------------------------------

    logging.root.handlers = []
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(traindir,f"train{rank}.log"), mode="a"),
                logging.StreamHandler()
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(traindir,f"train{rank}.log"), mode="a"),
            ],
        )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # FIGURE OUT MULTIGPU ------------------------------------------------------------
    if world_size > 1:
        multiprocessing_setup(rank, world_size, master_port)
        atexit.register(multiprocessing_cleanup)
        assert torch.cuda.is_available()

    if True or torch.cuda.is_available():
        my_gpu_id = multi_gpu_device_ids[rank]
        torch.cuda.set_device(my_gpu_id)
        logging.info("Using GPU device: " + torch.cuda.get_device_name())
        device = torch.device("cuda", my_gpu_id)
    else:
        logging.warning("WARNING: No GPU, using CPU")
        device = torch.device("cpu")

    sdpa_backend = configure_sdpa_backend(sdpa_backend)
    logging.info(f"SDPA backend selection: {sdpa_backend}")
    logging.info("FlexAttention: enabled=%s block_size=128", use_flex_attention)

    seed = int.from_bytes(os.urandom(7), sys.byteorder)
    logging.info(f"Seeding torch with {seed}")
    torch.manual_seed(seed)

    # LOAD MODEL ---------------------------------------------------------------------

    def lr_scale_auto_factor_custom(train_state):
        
        x = train_state["global_step_samples"]
        lr0 = 1.0
        s0 = 1e8
        t = x/s0

        # time ~ wdtc (weight decay time constant)
        if t < 2.0**-2:
            return lr0 * 2 ** (1.0)
        if t < 2.0**-1:
            return lr0 * 2 ** (0.5)
        if t < 2.0**0:
            return lr0 * 2 ** (-0.0)
        if t < 2.0**1:
            return lr0 * 2 ** (-0.5)
        if t < 2.0**2:
            return lr0 * 2 ** (-1.0)
        if t < 2.0**3:
            return lr0 * 2 ** (-1.5)
        if t < 2.0**4:
            return lr0 * 2 ** (-2.0)
        if t < 22:
            return lr0 * 2 ** (-2.5)
        # final drop
        if t < 24:
            return lr0 * 2 ** (-3.0)
        if t < 26:
            return lr0 * 2 ** (-3.5)
        if t < 28:
            return lr0 * 2 ** (-4.0)
        if t < 30:
            return lr0 * 2 ** (-5.0)
        if t < 32:
            return lr0 * 2 ** (-6.0)
        return lr0 * 2 ** (-7.0)
        
        
    def lr_scale_auto_factor(train_state):
        if lr_scale_auto_type == "":
            return 1.0
        elif lr_scale_auto_type == "custom":
            return lr_scale_auto_factor_custom(train_state)
        #elif lr_scale_auto_type == "1b":
        #    return lr_scale_auto_factor_1b(train_state)
        #elif lr_scale_auto_type == "2":
        #    return lr_scale_auto_factor_2(train_state)
        assert False, f"Unknown lr_scale_auto_type: {lr_scale_auto_type}"

        return 1.0

    def get_checkpoint_path():
        return os.path.join(traindir,"checkpoint.ckpt")
    def get_checkpoint_prev_path(i):
        return os.path.join(traindir,f"checkpoint_prev{i}.ckpt")
        
    def _recursive_check_for_nan(data, path=""):
        """
        Recursively checks for NaNs in nested data structures.
        If a NaN is found, prints the path and exits the program.
        This is a helper function.
    
        Args:
            data: The data to check (can be a tensor, dict, list, float, etc.).
            path (str): The current path in the original structure, for tracking and reporting.
        """
        has_nan=False
        if torch.is_tensor(data):
            if torch.isnan(data).any():
                #print(f"ERROR: NaN found in tensor at path: '{path}'")
                has_nan=True
        # elif isinstance(data, np.ndarray): # Uncomment this block if using numpy arrays
        #     if np.isnan(data).any():
        #         print(f"ERROR: NaN found in NumPy array at path: '{path}'")
        #         sys.exit(1)
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if(_recursive_check_for_nan(value, new_path)):
                    has_nan=True
                    
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                if(_recursive_check_for_nan(item, new_path)):
                    has_nan=True
        elif isinstance(data, float):
            if math.isnan(data):
                print(f"ERROR: NaN found in float at path: '{path}'")
                has_nan=True
        return has_nan
        # Other scalar types (like int) are generally not NaN, so no explicit check needed.
    
    def check_state_dict_for_nan_and_exit(state_dict_to_check):
        """
        Checks a state dictionary for NaNs in any of its components.
        If a NaN is found, it prints the location and exits the program.
    
        Args:
            state_dict_to_check (dict): The dictionary containing model state,
                                         optimizer state, metrics, etc.
        """
        if qat_int8:
            logging.info("Skipping nan checking because the model is QAT for int8")
            return False
        print("Starting NaN check in state_dict...")
        assert(isinstance(state_dict_to_check, dict))
        
        has_nan=False
        for component_name, component_data in state_dict_to_check.items():
            # print(f"Checking component: {component_name}") # Optional: for verbose logging
            if _recursive_check_for_nan(component_data, component_name):
                has_nan=True

        return has_nan

            
    NUM_SHORTTERM_CHECKPOINTS_TO_KEEP = 4
    def save(raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics, path=None, skip_optimizer=False):
        if gnorm_stats_debug:
            logging.warning("Skipping save since debugging gnorm stats")
            return
        # 若skip_optimizer==False，必须所有 rank 都调用：内部使用了 dist.gather_object 这一 collective，
        # 若只让 rank0 进入会导致其余 rank 与 rank0 通信不同步并卡住。
        optimizer_state_dict = None
        if not skip_optimizer:
            optimizer_state_dict = optimizer.state_dict_for_checkpoint()
        if rank == 0:
            state_dict = {}
            state_dict["model"] = raw_model.state_dict()
            if not skip_optimizer:
                state_dict["optimizer"] = optimizer_state_dict
            state_dict["metrics"] = metrics_obj.state_dict()
            state_dict["running_metrics"] = running_metrics
            
            if(check_state_dict_for_nan_and_exit(state_dict)):
                print("Detect NANs in state dict, exiting")
                raise Exception(f"ERROR: NaN found in state dict, exiting")
            
            state_dict["train_state"] = train_state
            state_dict["last_val_metrics"] = last_val_metrics
            state_dict["config"] = model_config

            if swa_models is not None and len(swa_models)>0:
                for i in range(len(swa_models)):
                    if swa_models[i] is not None:
                        state_dict[f"swa_model_{i}"] = swa_models[i].state_dict()
                    else:
                        assert qat_int8, f"swa_model_{i} is None but qat_int8 is False"
                        logging.warning(f"Skipping swa_model_{i} because it is None")

            if path is not None:
                logging.info("Saving checkpoint: " + path)
                torch.save(state_dict, path + ".tmp")
                time.sleep(1)
                os.replace(path + ".tmp", path)
            else:
                logging.info("Saving checkpoint: " + get_checkpoint_path())
                for i in reversed(range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP-1)):
                    if os.path.exists(get_checkpoint_prev_path(i)):
                        os.replace(get_checkpoint_prev_path(i), get_checkpoint_prev_path(i+1))
                if os.path.exists(get_checkpoint_path()):
                    shutil.copy(get_checkpoint_path(), get_checkpoint_prev_path(0))
                torch.save(state_dict, get_checkpoint_path() + ".tmp")
                os.replace(get_checkpoint_path() + ".tmp", get_checkpoint_path())

    def pslr_func(batchsize,lrscale):
        pslr = lr_base * lrscale * (batchsize/1.0)**0.5
        return pslr

        
    def wd_factor(batchsize,lrscale):
        pslr = pslr_func(batchsize,lrscale)
        tc=1e7*lrscale**(-2.0) # 1.75 is good for CNN. For transformer probably 1.8 ~ 1.85. But 2.0 is more beautiful
        #tc=batchsize/(wd*pslr)
        wd=batchsize/(pslr*tc)
        return wd

        
    def get_weight_decay(raw_model, lr_scale, warmup_scale, train_state, running_metrics, group_name, wd_scale):
        lr_scale_with_auto = lr_scale * lr_scale_auto_factor(train_state)
        if raw_model.get_norm_kind() == "fixup" or raw_model.get_norm_kind() == "fixscale":
            if group_name == "normal" or group_name == "normal_gamma" or group_name == "normal_attn" or group_name == "output":
                return 0.00000003 * world_size * batch_size / 256.0 * wd_scale
            elif group_name == "noreg":
                return 0.0000000003 * world_size * batch_size / 256.0 * wd_scale
            elif group_name == "output_noreg":
                return 0.0000000003 * world_size * batch_size / 256.0 * wd_scale
            else:
                assert False
        elif (
            raw_model.get_norm_kind() == "bnorm" or
            raw_model.get_norm_kind() == "brenorm" or
            raw_model.get_norm_kind() == "fixbrenorm" or
            raw_model.get_norm_kind() == "fixscaleonenorm"
        ):
            
            wd0 = wd_factor(batch_size * world_size,lr_scale * lr_scale_auto_factor(train_state))
            
            
            if group_name == "normal":
                group_factor = 1.0
            elif group_name == "normal_attn":
                group_factor = 0.5
            elif group_name == "normal_gamma":
                group_factor = 0.1 
            elif group_name == "output":
                group_factor = 0.25 
            elif group_name == "noreg":
                group_factor = 0.001
            elif group_name == "output_noreg":
                group_factor = 0.001
            else:
                assert False
                
            return group_factor * wd0 * wd_scale
        else:
            assert False

    def get_param_groups(raw_model,train_state,running_metrics):
        reg_dict : Dict[str,List] = {}
        raw_model.add_reg_dict(reg_dict)
        param_groups = []
        param_groups.append({
            "params": reg_dict["normal"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="normal", wd_scale=wd_scale),
            "group_name": "normal",
        })
        if len(reg_dict["normal_gamma"]) > 0:
            param_groups.append({
                "params": reg_dict["normal_gamma"],
                "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="normal_gamma", wd_scale=wd_scale),
                "group_name": "normal_gamma",
            })
        if len(reg_dict["normal_attn"]) > 0:
            param_groups.append({
                "params": reg_dict["normal_attn"],
                "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="normal_attn", wd_scale=wd_scale),
                "group_name": "normal_attn",
            })
        param_groups.append({
            "params": reg_dict["output"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="output", wd_scale=wd_scale),
            "group_name": "output",
        })
        param_groups.append({
            "params": reg_dict["noreg"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="noreg", wd_scale=wd_scale),
            "group_name": "noreg",
        })
        param_groups.append({
            "params": reg_dict["output_noreg"],
            "weight_decay": get_weight_decay(raw_model, lr_scale, warmup_scale=1.0, train_state=train_state, running_metrics=running_metrics, group_name="output_noreg", wd_scale=wd_scale),
            "group_name": "output_noreg",
        })
        num_params = len(list(raw_model.parameters()))
        num_reg_dict_params = len(reg_dict["normal"]) + len(reg_dict["normal_gamma"])  + len(reg_dict["normal_attn"]) + len(reg_dict["output"]) + len(reg_dict["noreg"]) + len(reg_dict["output_noreg"])
        assert num_params == num_reg_dict_params, "Reg dict does not have entries for all params in model"
        return param_groups

    def make_ema_avg(factor):
        def ema_avg(avg_param, cur_param, num_averaged):
            return avg_param + factor * (cur_param - avg_param)
        return ema_avg

    def make_swa_model(raw_model, factor):
        swa_model = AveragedModel(raw_model, avg_fn=make_ema_avg(factor))
        if use_flex_attention:
            # AveragedModel is evaluated eagerly rather than through the
            # compiled DDP training wrapper. Eager FlexAttention materializes
            # the full score matrix, so use the semantically equivalent fused
            # SDPA path for SWA validation. This runtime flag is not checkpointed
            # and does not affect which parameters are averaged or exported.
            swa_model.module.configure_flex_attention(enabled=False)
            logging.info(
                "SWA validation will use masked SDPA instead of eager FlexAttention"
            )
        return swa_model

    def configure_raw_model_runtime(raw_model):
        raw_model.configure_flex_attention(enabled=use_flex_attention)

    def load():
        if not os.path.exists(get_checkpoint_path()):
            logging.info("No preexisting checkpoint found at: " + get_checkpoint_path())
            for i in range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP):
                if os.path.exists(get_checkpoint_prev_path(i)):
                    raise Exception(f"No preexisting checkpoint found, but {get_checkpoint_prev_path(i)} exists, something is wrong with the training dir")

            if initial_checkpoint is not None:
                if os.path.exists(initial_checkpoint):
                    logging.info("Using initial checkpoint: {initial_checkpoint}")
                    path_to_load_from = initial_checkpoint
                else:
                    raise Exception("No preexisting checkpoint found, initial checkpoint provided is invalid: {initial_checkpoint}")
            else:
                path_to_load_from = None
        else:
            path_to_load_from = get_checkpoint_path()

            
        if path_to_load_from is None:
            logging.info("Initializing new model!")
            assert model_kind is not None, "Model kind is none or unspecified but the model is being created fresh"
            model_config = modelconfigs.config_of_name[model_kind]
            logging.info(str(model_config))
            raw_model = Model(model_config,pos_len)
            raw_model.initialize()
            
            if qat_int8:
                logging.info("Preparing model for INT8 QAT (TensorRT compatible)...")
                # raw_model = QATModelWrapper(raw_model)
                raw_model.qconfig = get_tensorrt_qat_qconfig()

                disable_qat_for_unsupported_modules(raw_model)

                torch.ao.quantization.prepare_qat(raw_model, inplace=True)
                #fix_qat_zero_points(raw_model)
                logging.info("Model prepared for QAT (inputs and heads excluded).")

            configure_raw_model_runtime(raw_model)
            raw_model.to(device)
            #raw_model_compiled=torch.compile(raw_model,mode="max-autotune-no-cudagraphs")
            ddp_model = wrap_model_for_training(
                raw_model,
                device,
                world_size,
                no_compile,
                compile_mode=compile_mode,
                qat_int8=qat_int8,
            )

            swa_models = []
            if rank == 0 and len(swa_scales)>0:
                for i in range(len(swa_scales)):
                    swa_scale=swa_scales[i]
                    new_factor = 1.0 / swa_scale
                    #ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (cur_param - avg_param)
                    if qat_int8:
                        swa_models.append(None) # init it when accumulating swa for the first time
                        continue
                    swa_model = make_swa_model(raw_model, new_factor)
                    swa_models.append(swa_model)

            metrics_obj = Metrics(batch_size,world_size,raw_model)
            running_metrics = {}
            train_state = {}
            last_val_metrics = {}

            train_state["global_step_samples"] = 0

            with torch.no_grad():
                (modelnorm_normal, modelnorm_normal_gamma, modelnorm_normal_attn, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = Metrics.get_model_norms(raw_model)
                modelnorm_normal_baseline = modelnorm_normal.detach().cpu().item()
                train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline
                logging.info(f"Model norm normal baseline computed: {modelnorm_normal_baseline}")

            optimizer = MuonWithAuxAdamKimi(get_param_groups(raw_model,train_state,running_metrics),muon_momentum)

            return (model_config, ddp_model, raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics)
        else:
            state_dict = torch.load(path_to_load_from, map_location=device, weights_only=False)
            model_config = state_dict["config"] if "config" in state_dict else modelconfigs.config_of_name[model_kind]
            logging.info(str(model_config))
            raw_model = Model(model_config,pos_len)
            raw_model.initialize()

            train_state = {}
            if "train_state" in state_dict:
                train_state = state_dict["train_state"]
            else:
                logging.info("WARNING: Train state not found in state dict, using fresh train state")

            # Do this before loading the state dict, while the model is initialized to fresh values, to get a good baseline
            if "modelnorm_normal_baseline" not in train_state:
                logging.info("Computing modelnorm_normal_baseline since not in train state")
                with torch.no_grad():
                    (modelnorm_normal, modelnorm_normal_gamma,  modelnorm_normal_attn,modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = Metrics.get_model_norms(raw_model)
                    modelnorm_normal_baseline = modelnorm_normal.detach().cpu().item()
                    train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline
                    logging.info(f"Model norm normal baseline computed: {modelnorm_normal_baseline}")
            
            model_state_dict = load_model.load_model_state_dict(state_dict)
            checkpoint_is_qat_like = is_qat_checkpoint(model_state_dict)
            
            if(qat_int8 and not checkpoint_is_qat_like):
                logging.info("QAT_Int8 is enabled but checkpoint is not QAT format, converting it to QAT format")

            if not qat_int8 or not checkpoint_is_qat_like: #if qat and checkpoint is qat, then load it later, not here
                raw_model.load_state_dict(model_state_dict)



            if qat_int8:
                logging.info("Preparing model for INT8 QAT (TensorRT compatible)...")
                # raw_model = QATModelWrapper(raw_model)
                raw_model.qconfig = get_tensorrt_qat_qconfig()

                disable_qat_for_unsupported_modules(raw_model)

                torch.ao.quantization.prepare_qat(raw_model, inplace=True)
                #fix_qat_zero_points(raw_model)
                logging.info("Model prepared for QAT (inputs and heads excluded).")

            # Strip off any "module." from when the model was saved with DDP or other things
            
            if qat_int8 and checkpoint_is_qat_like:
                logging.info("Loading QAT checkpoint (native) into QAT model")
                import dummy_input
                logging.info("Running dummy forward pass to determine QAT shapes...")
                raw_model.eval()
                dummy_binary, dummy_global, dummy_meta = dummy_input.generate_dummy_inputs(model_config, 1, pos_len, device="cpu")
                with torch.no_grad():
                    raw_model(dummy_binary, dummy_global, input_meta=dummy_meta)
                raw_model.train()
                raw_model.load_state_dict(model_state_dict)
                
            
            if not qat_int8:
                reset_nan_batchnorm(raw_model, verbose=True)
            
            configure_raw_model_runtime(raw_model)
            raw_model.to(device)
            #raw_model_compiled=torch.compile(raw_model,mode="max-autotune-no-cudagraphs")
            ddp_model = wrap_model_for_training(
                raw_model,
                device,
                world_size,
                no_compile,
                compile_mode=compile_mode,
                qat_int8=qat_int8,
            )
                
            swa_models = []
            if rank == 0 and len(swa_scales)>0:
                for i in range(len(swa_scales)):
                    if qat_int8 and not checkpoint_is_qat_like:
                        logging.info(f"Swa model {i} in state_dict is not QAT like, not loading it")
                        swa_models.append(None) # init it when accumulating swa for the first time
                        continue
                    swa_scale=swa_scales[i]
                    new_factor = 1.0 / swa_scale
                    #ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (cur_param - avg_param)
                    swa_model = make_swa_model(raw_model, new_factor)
                    swa_model_state_dict = load_model.load_swa_model_state_dict(state_dict,idx=i)
                    if swa_model_state_dict is not None:
                        logging.info(f"Load swa model {i}")
                        swa_model.load_state_dict(swa_model_state_dict)
                    else:
                        logging.info(f"Swa model {i} not found in state_dict")
                        if qat_int8:
                            swa_models.append(None) # init it when accumulating swa for the first time
                            continue
                    swa_models.append(swa_model)
                    
                    
                    

            metrics_obj = Metrics(batch_size,world_size,raw_model)
            if "metrics" in state_dict:
                metrics_obj.load_state_dict(state_dict["metrics"])
            else:
                logging.info("WARNING: Metrics not found in state dict, using fresh metrics")

            running_metrics = {}
            if "running_metrics" in state_dict:
                running_metrics = state_dict["running_metrics"]
            else:
                logging.info("WARNING: Running metrics not found in state dict, using fresh running metrics")

            last_val_metrics = {}
            if "last_val_metrics" in state_dict:
                last_val_metrics = state_dict["last_val_metrics"]
            else:
                logging.info("WARNING: Running metrics not found in state dict, using fresh last val metrics")

            optimizer = MuonWithAuxAdamKimi(get_param_groups(raw_model,train_state,running_metrics),muon_momentum)
            if "optimizer" in state_dict:
                optimizer.load_state_dict_for_checkpoint(state_dict["optimizer"])
            else:
                logging.info("WARNING: Optimizer not found in state dict, using fresh optimizer")

            return (model_config, ddp_model, raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics)

    (model_config, ddp_model, raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics) = load()


    if "global_step_samples" not in train_state:
        train_state["global_step_samples"] = 0
    if max_train_bucket_per_new_data is not None and "train_bucket_level" not in train_state:
        train_state["train_bucket_level"] = samples_per_epoch
    if "train_steps_since_last_reload" not in train_state:
        train_state["train_steps_since_last_reload"] = 0
    if "export_cycle_counter" not in train_state:
        train_state["export_cycle_counter"] = 0
    if "total_num_data_rows" not in train_state:
        train_state["total_num_data_rows"] = 0
    if "old_train_data_dirs" not in train_state:
        train_state["old_train_data_dirs"] = []
    #if "data_files_used" not in train_state:
    train_state["data_files_used"] = set()
    if "swa_sample_accum" not in train_state:
        train_state["swa_sample_accum"] = 0.0


    if intermediate_loss_scale is not None:
        assert raw_model.get_has_intermediate_head(), "Model must have intermediate head to use intermediate loss"

    # If the user specified an intermediate head but no loss scale, pick something reasonable by default
    if raw_model.get_has_intermediate_head():
        if intermediate_loss_scale is None and main_loss_scale is None:
            if model_config["trunk_normless"]:
                # fson-bnh default
                assert model_config["intermediate_head_blocks"] == len(model_config["block_kind"]), "If these are unequal, don't know what you intend, please specify intermediate_loss_scale"
                intermediate_loss_scale = 0.8
                main_loss_scale = 0.2
            else:
                # Intermediate head in the middle of the trunk
                intermediate_loss_scale = 0.5
                main_loss_scale = 0.5
        elif intermediate_loss_scale is None:
            assert False, "Please specify both of main_loss_scale and intermediate_loss_scale or neither when using an architecture with an intermediate head."

    logging.info(f"swa_period_samples {swa_period_samples}")
    logging.info(f"swa_scales {swa_scales}")
    logging.info(f"lookahead_alpha {lookahead_alpha}")
    logging.info(f"lookahead_k {lookahead_k}")
    logging.info(f"soft_policy_weight_scale {soft_policy_weight_scale}")
    logging.info(f"disable_optimistic_policy {disable_optimistic_policy}")
    logging.info(f"meta_kata_only_soft_policy {meta_kata_only_soft_policy}")
    logging.info(f"value_loss_scale {value_loss_scale}")
    logging.info(f"td_value_loss_scales {td_value_loss_scales}")
    logging.info(f"seki_loss_scale {seki_loss_scale}")
    logging.info(f"variance_time_loss_scale {variance_time_loss_scale}")
    logging.info(f"main_loss_scale {main_loss_scale}")
    logging.info(f"intermediate_loss_scale {intermediate_loss_scale}")
    logging.info(f"model_norms_only_at_print {model_norms_only_at_print}")
    logging.info(f"compile_training_loss {compile_training_loss}")
    logging.info(f"disable_mask {disable_mask}")
    logging.info(f"filter_full_board_on_load {filter_full_board_on_load}")
    logging.info(f"input_memory_format {input_memory_format}")

    training_metrics_fn = metrics_obj.metrics_dict_batchwise
    if compile_training_loss:
        if not model_norms_only_at_print:
            raise ValueError(
                "KATAGO_COMPILE_TRAINING_LOSS=1 requires "
                "KATAGO_MODEL_NORMS_ONLY_AT_PRINT=1 so the compiled result structure is fixed"
            )
        if not metrics_obj.seki_ema_on_device:
            raise ValueError(
                "KATAGO_COMPILE_TRAINING_LOSS=1 requires KATAGO_SEKI_EMA_ON_DEVICE=1 "
                "to avoid a per-step Python scalar guard"
            )
        loss_compile_mode = validate_compile_mode(compile_mode)
        training_metrics_fn = torch.compile(
            training_metrics_fn,
            mode=loss_compile_mode,
            dynamic=False,
        )

    # Print all model parameters just to get a summary
    total_num_params = 0
    total_trainable_params = 0
    logging.info("Parameters in model:")
    for name, param in raw_model.named_parameters():
        product = 1
        for dim in param.shape:
            product *= int(dim)
        if param.requires_grad:
            total_trainable_params += product
        total_num_params += product
        #logging.info(f"{name}, {list(param.shape)}, {product} params")
    logging.info(f"Total num params: {total_num_params}")
    logging.info(f"Total trainable params: {total_trainable_params}")

    lookahead_cache = {}
    if lookahead_k is not None:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                lookahead_cache[param] = torch.zeros_like(param.data)
                lookahead_cache[param] = lookahead_cache[param].copy_(param.data)
        logging.info(f"Using lookahead optimizer {lookahead_alpha} {lookahead_k}")

    # EPOCHS AND LR ---------------------------------------------------------------------

    
    def update_and_return_lr_and_wd():
        per_sample_lr = pslr_func(batch_size * world_size,lr_scale * lr_scale_auto_factor(train_state))

        # Warmup for initial training
        warmup_scale = 1.0
        if model_config["norm_kind"] == "fixup" or model_config["norm_kind"] == "fixscale" or model_config["norm_kind"] == "fixscaleonenorm":
            if train_state["global_step_samples"] < 1000000:
                warmup_scale = 1.0 / 5.0
            elif train_state["global_step_samples"] < 2000000:
                warmup_scale = 1.0 / 3.0
            elif train_state["global_step_samples"] < 4000000:
                warmup_scale = 1.0 / 2.0
            elif train_state["global_step_samples"] < 6000000:
                warmup_scale = 1.0 / 1.4
        elif model_config["norm_kind"] == "bnorm" or model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
            if train_state["global_step_samples"] < 250000:
                warmup_scale = 1.0 / 20.0
            elif train_state["global_step_samples"] < 500000:
                warmup_scale = 1.0 / 14.0
            elif train_state["global_step_samples"] < 750000:
                warmup_scale = 1.0 / 10.0
            elif train_state["global_step_samples"] < 1000000:
                warmup_scale = 1.0 / 7.0
            elif train_state["global_step_samples"] < 1250000:
                warmup_scale = 1.0 / 5.0
            elif train_state["global_step_samples"] < 1500000:
                warmup_scale = 1.0 / 3.0
            elif train_state["global_step_samples"] < 1750000:
                warmup_scale = 1.0 / 2.0
            elif train_state["global_step_samples"] < 2000000:
                warmup_scale = 1.0 / 1.4
            else:
                warmup_scale = 1.0 / 1.0
        else:
            assert False

        normal_weight_decay = 0.0

        for param_group in optimizer.param_groups:
            group_name = param_group["group_name"]
            if group_name == "normal":
                group_scale = 1.0
            elif group_name == "normal_gamma":
                group_scale = 1.0
            elif group_name == "normal_attn":
                group_scale = 1.0
            elif group_name == "output":
                group_scale = 0.5
            elif group_name == "noreg":
                group_scale = 0.2
            elif group_name == "output_noreg":
                group_scale = 0.2
            else:
                assert False

            changed = False

            param_group["eps"] = 1e-6
            # For lookahead optimizer, use weight decay appropriate for lr scale,
            # but tell optimizer to take larger steps so as to maintain the same
            # effective learning rate after lookahead averaging.
            if lookahead_alpha is not None:
                new_lr_this_group = per_sample_lr * warmup_scale * group_scale / lookahead_alpha
            else:
                new_lr_this_group = per_sample_lr * warmup_scale * group_scale 

            #new_lr_this_group*=0.125
            if("muon_lr_multiplier" in param_group):
                param_group["muon_lr_multiplier"] = 8.0
            
            if param_group["lr"] != new_lr_this_group:
                param_group["lr"] = new_lr_this_group
                changed = True

            new_weight_decay_this_group = get_weight_decay(
                raw_model,
                lr_scale,
                warmup_scale=warmup_scale,
                train_state=train_state,
                running_metrics=running_metrics,
                group_name=group_name,
                wd_scale=wd_scale
            )
            if param_group["weight_decay"] != new_weight_decay_this_group:
                param_group["weight_decay"] = new_weight_decay_this_group
                changed = True

            if group_name == "normal":
                normal_weight_decay = param_group["weight_decay"]

            #if changed:
            #    logging.info(f"Param group {param_group['group_name']} lr {param_group['lr']} weight_decay {param_group['weight_decay']}")

        return per_sample_lr * warmup_scale, normal_weight_decay

    last_brenorm_update_samples_this_instance = train_state["global_step_samples"]
    def maybe_update_brenorm_params(force_update=False):
        nonlocal last_brenorm_update_samples_this_instance
        should_update=force_update
        if model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
            if "brenorm_rmax" not in train_state:
                train_state["brenorm_rmax"] = 1.0
                should_update=True
            if "brenorm_dmax" not in train_state:
                train_state["brenorm_dmax"] = 0.0
                should_update=True

            #num_samples_elapsed = train_state["global_step_samples"] - last_brenorm_update_samples_this_instance
            #factor = math.exp(-num_samples_elapsed / brenorm_adjustment_scale)
            #train_state["brenorm_rmax"] = train_state["brenorm_rmax"] + (1.0 - factor) * (brenorm_target_rmax - train_state["brenorm_rmax"])
            #train_state["brenorm_dmax"] = train_state["brenorm_dmax"] + (1.0 - factor) * (brenorm_target_dmax - train_state["brenorm_dmax"])

            factor = math.exp(-train_state["global_step_samples"] / brenorm_adjustment_scale)
            rmax=1.0 + (1.0 - factor) * (brenorm_target_rmax - 1.0)
            dmax=0.0 + (1.0 - factor) * (brenorm_target_dmax - 0.0)

            delta_threhold=0.1
            if(should_update or train_state["brenorm_rmax"]-rmax>delta_threhold or train_state["brenorm_rmax"]-rmax < -delta_threhold or train_state["brenorm_dmax"]-dmax>delta_threhold or train_state["brenorm_dmax"]-dmax < -delta_threhold):
                train_state["brenorm_rmax"]=rmax
                train_state["brenorm_dmax"]=dmax
                logging.info(f"update brenorm params: rmax {train_state["brenorm_rmax"]}, dmax {train_state["brenorm_dmax"]}")
                raw_model.set_brenorm_params(brenorm_avg_momentum, train_state["brenorm_rmax"], train_state["brenorm_dmax"])
                last_brenorm_update_samples_this_instance = train_state["global_step_samples"]

    # DATA RELOADING GENERATOR ------------------------------------------------------------

    # Some globals
    last_curdatadir = None
    trainfilegenerator = None
    vdatadir = None

    def maybe_reload_training_data():
        nonlocal last_curdatadir
        nonlocal trainfilegenerator
        nonlocal vdatadir

        assert rank == 0, "Helper ddp training processes should not call maybe_reload_training_data"

        while True:
            curdatadir = os.path.realpath(datadir)

            # Different directory - new shuffle
            if curdatadir != last_curdatadir:
                if not os.path.exists(curdatadir):
                    if quit_if_no_data:
                        logging.info("Shuffled data path does not exist, there seems to be no data or not enough data yet, qutting: %s" % curdatadir)
                        sys.exit(0)
                    logging.info("Shuffled data path does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % curdatadir)
                    time.sleep(30)
                    continue

                trainjsonpath = os.path.join(curdatadir,"train.json")
                if not os.path.exists(trainjsonpath):
                    if quit_if_no_data:
                        logging.info("Shuffled data train.json file does not exist, there seems to be no data or not enough data yet, qutting: %s" % trainjsonpath)
                        sys.exit(0)
                    logging.info("Shuffled data train.json file does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s" % trainjsonpath)
                    time.sleep(30)
                    continue

                logging.info("Updated training data: " + curdatadir)
                last_curdatadir = curdatadir

                with open(trainjsonpath) as f:
                    datainfo = json.load(f)
                    train_state["total_num_data_rows"] = datainfo["range"][1]

                # Fill the buckets
                if max_train_bucket_per_new_data is not None:
                    if "train_bucket_level_at_row" not in train_state:
                        train_state["train_bucket_level_at_row"] = train_state["total_num_data_rows"]
                    if train_state["total_num_data_rows"] > train_state["train_bucket_level_at_row"]:
                        new_row_count = train_state["total_num_data_rows"] - train_state["train_bucket_level_at_row"]
                        logging.info("Advancing trainbucket row %.0f to %.0f, %.0f new rows" % (
                            train_state["train_bucket_level_at_row"], train_state["total_num_data_rows"], new_row_count
                        ))
                        train_state["train_bucket_level_at_row"] = train_state["total_num_data_rows"]
                        logging.info("Fill per data %.3f, Max bucket size %.0f" % (max_train_bucket_per_new_data, max_train_bucket_size))
                        logging.info("Old rows in bucket: %.0f" % train_state["train_bucket_level"])
                        train_state["train_bucket_level"] += new_row_count * max_train_bucket_per_new_data
                        cap = max(max_train_bucket_size, samples_per_epoch)
                        if train_state["train_bucket_level"] > cap:
                            train_state["train_bucket_level"] = cap
                        logging.info("New rows in bucket: %.0f" % train_state["train_bucket_level"])
                    if train_state["total_num_data_rows"] < train_state["train_bucket_level_at_row"]:
                        # Bucket went backward! This must be a network imported from a different run, reset the train bucket level
                        logging.warning("Train bucket last filled at %d rows but now there are only %d rows!" % (
                            train_state["train_bucket_level_at_row"], train_state["total_num_data_rows"]
                        ))
                        logging.warning("Data was deleted or this network was transplanted into a new run, resetting the train bucket fill rows")
                        train_state["train_bucket_level_at_row"] = train_state["total_num_data_rows"]

                logging.info("Train steps since last reload: %.0f -> 0" % train_state["train_steps_since_last_reload"])
                train_state["train_steps_since_last_reload"] = 0

                # Load training data files

                # Load training data files
                tdatadir = os.path.join(curdatadir,"train")
                #train_files = [os.path.join(tdatadir,fname) for fname in os.listdir(tdatadir) if fname.endswith(".npz")]
                train_files = []
                for root, dirs, files in os.walk(tdatadir):
                    for fname in files:
                        if fname.endswith(".npz"):
                            train_files.append(os.path.join(root, fname))

                epoch0_train_files = [path for path in train_files if path not in train_state["data_files_used"]]
                if no_repeat_files:
                    logging.info(f"Dropping {len(train_files)-len(epoch0_train_files)}/{len(train_files)} files in: {tdatadir} as already used")
                else:
                    logging.info(f"Skipping {len(train_files)-len(epoch0_train_files)}/{len(train_files)} files in: {tdatadir} as already used first pass")

                if len(train_files) <= 0 or (no_repeat_files and len(epoch0_train_files) <= 0):
                    if quit_if_no_data:
                        logging.info(f"No new training files found in: {tdatadir}, quitting")
                        sys.exit(0)
                    logging.info(f"No new training files found in: {tdatadir}, waiting 30s and trying again")
                    time.sleep(30)
                    continue

                # Update history of what training data we used
                if tdatadir not in train_state["old_train_data_dirs"]:
                    train_state["old_train_data_dirs"].append(tdatadir)
                # Clear out tracking of sufficiently old files
                while len(train_state["old_train_data_dirs"]) > 20:
                    old_dir = train_state["old_train_data_dirs"][0]
                    train_state["old_train_data_dirs"] = train_state["old_train_data_dirs"][1:]
                    for filename in list(train_state["data_files_used"]):
                        if filename.startswith(old_dir):
                            train_state["data_files_used"].remove(filename)

                def train_files_gen():
                    train_files_shuffled = epoch0_train_files.copy()
                    while True:
                        random.shuffle(train_files_shuffled)
                        for filename in train_files_shuffled:
                            #logging.info("Yielding training file for dataset: " + filename)
                            train_state["data_files_used"].add(filename)
                            yield filename
                        if no_repeat_files:
                            break
                        else:
                            train_files_shuffled = train_files.copy()
                            train_state["data_files_used"] = set()

                trainfilegenerator = train_files_gen()
                vdatadir = os.path.join(curdatadir,"val")

            # Same directory as before, no new shuffle
            else:
                if max_train_steps_since_last_reload is not None:
                    if train_state["train_steps_since_last_reload"] + 0.99 * samples_per_epoch/sub_epochs > max_train_steps_since_last_reload:
                        logging.info(
                            "Too many train steps since last reload, waiting 5m and retrying (current %f)" %
                            train_state["train_steps_since_last_reload"]
                        )
                        time.sleep(300)
                        continue

            break

    # Load all the files we should train on during a subepoch
    def get_files_for_subepoch():
        nonlocal trainfilegenerator

        assert rank == 0, "Helper ddp training processes should not call get_files_for_subepoch"

        num_batches_per_epoch = int(round(samples_per_epoch / batch_size))
        num_batches_per_subepoch = num_batches_per_epoch / sub_epochs

        # Pick enough files to get the number of batches we want
        train_files_to_use = []
        batches_to_use_so_far = 0
        found_enough = False
        for filename in trainfilegenerator:
            jsonfilename = os.path.splitext(filename)[0] + ".json"
            with open(jsonfilename) as f:
                trainfileinfo = json.load(f)

            num_batches_this_file = trainfileinfo["num_rows"] // batch_size
            if num_batches_this_file <= 0:
                continue

            if batches_to_use_so_far + num_batches_this_file > num_batches_per_subepoch:
                # If we're going over the desired amount, randomly skip the file with probability equal to the
                # proportion of batches over - this makes it so that in expectation, we have the desired number of batches
                if batches_to_use_so_far > 0 and random.random() >= (batches_to_use_so_far + num_batches_this_file - num_batches_per_subepoch) / num_batches_this_file:
                    found_enough = True
                    break

            train_files_to_use.append(filename)
            batches_to_use_so_far += num_batches_this_file

            #Sanity check - load a max of 100000 files.
            if batches_to_use_so_far >= num_batches_per_subepoch or len(train_files_to_use) > 100000:
                found_enough = True
                break

        if found_enough:
            return train_files_to_use
        return None

    # METRICS -----------------------------------------------------------------------------------
    def detensorify_metrics(metrics):
        ret = {}
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                ret[key] = metrics[key].detach().cpu().item()
            else:
                ret[key] = metrics[key]
        return ret

    if rank == 0:
        train_metrics_out = open(os.path.join(traindir,"metrics_train.json"),"a")
        val_metrics_out = open(os.path.join(traindir,"metrics_val.json"),"a")
        val_swa_metrics_outs=[]
        for i in range(len(swa_scales)):
            val_swa_metrics_outs.append( open(os.path.join(traindir,f"metrics_val_swa{i}.json"),"a"))
    else:
        train_metrics_out = open(os.path.join(traindir,f"metrics_train_rank{rank}.json"),"a")
        val_metrics_out = open(os.path.join(traindir,f"metrics_val_rank{rank}.json"),"a")

    # TRAIN! -----------------------------------------------------------------------------------

    last_longterm_checkpoint_save_time = datetime.datetime.now()
    num_epochs_this_instance = 0
    print_train_loss_every_batches = 100 if not gnorm_stats_debug else 1000
    model_norm_metric_keys = (
        "norm_normal_batch",
        "norm_normal_gamma_batch",
        "norm_normal_attn_batch",
        "norm_output_batch",
        "norm_noreg_batch",
        "norm_output_noreg_batch",
    )

    if "sums" not in running_metrics:
        running_metrics["sums"] = defaultdict(float)
    else:
        running_metrics["sums"] = defaultdict(float,running_metrics["sums"])
    if "weights" not in running_metrics:
        running_metrics["weights"] = defaultdict(float)
    else:
        running_metrics["weights"] = defaultdict(float,running_metrics["weights"])

    torch.backends.cudnn.benchmark = True

    scaler = create_grad_scaler(use_fp16, use_bf16)
    if use_fp16:
        logging.info("Training in FP16! Creating scaler")
    elif use_bf16:
        logging.info("Training in BF16 AMP without gradient scaling.")
    else:
        logging.info("Training in FP32.")

    # All ddp threads should be lined up at this point before continuing
    if barrier is not None:
        barrier.wait()

    maybe_update_brenorm_params(force_update=True)
    
    while True:
        if max_epochs_this_instance is not None and max_epochs_this_instance >= 0 and num_epochs_this_instance >= max_epochs_this_instance:
            logging.info("Hit max epochs this instance, done")
            break
        if max_training_samples is not None and train_state["global_step_samples"] >= max_training_samples:
            logging.info("Hit max training samples, done")
            break

        epoch_action = _RANK0_ACTION_PROCEED
        if rank == 0:
            maybe_reload_training_data()

            if max_train_bucket_per_new_data is not None:
                if train_state["train_bucket_level"] > 0.99 * samples_per_epoch:
                    logging.info("Consuming %.0f rows from train bucket (%.0f -> %.0f)" % (
                        samples_per_epoch, train_state["train_bucket_level"], train_state["train_bucket_level"]-samples_per_epoch
                    ))
                    train_state["train_bucket_level"] -= samples_per_epoch
                else:
                    if stop_when_train_bucket_limited:
                        logging.info(
                            "Exceeding train bucket, not enough new data rows, terminating (current level %f)" %
                            train_state["train_bucket_level"]
                        )
                        epoch_action = _RANK0_ACTION_STOP
                    else:
                        logging.info(
                            "Exceeding train bucket, not enough new data rows, waiting 5m and retrying (current level %f)" %
                            train_state["train_bucket_level"]
                        )
                        epoch_action = _RANK0_ACTION_RETRY

        epoch_action = broadcast_rank0_action(
            epoch_action if rank == 0 else None,
            rank,
            world_size,
            device,
        )
        if epoch_action == _RANK0_ACTION_STOP:
            break
        if epoch_action == _RANK0_ACTION_RETRY:
            # Complete the collective before waiting so NCCL cannot time out
            # while rank 0 sleeps for new data.
            time.sleep(300)
            continue
        if epoch_action != _RANK0_ACTION_PROCEED:
            raise RuntimeError(f"Unknown rank 0 epoch action: {epoch_action}")

        # DDP need to wait on the main process after reloading data and/or training bucket waiting
        if barrier is not None:
            barrier.wait()

        logging.info("GC collect")
        gc.collect()

        clear_metric_nonfinite(running_metrics["sums"], running_metrics["weights"])

        logging.info("=========================================================================")
        logging.info("BEGINNING NEXT EPOCH " + str(num_epochs_this_instance))
        logging.info("=========================================================================")
        logging.info("Current time: " + str(datetime.datetime.now()))
        logging.info("Global step: %d samples" % (train_state["global_step_samples"]))
        logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
        logging.info(f"Training dir: {traindir}")
        logging.info(f"Export dir: {exportdir}")
        if use_fp16:
            logging.info(f"Current grad scale: {scaler.get_scale()}")

        lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()

        # SUB EPOCH LOOP -----------
        batch_count_this_epoch = 0
        last_train_stats_time = time.perf_counter()
        quit_due_to_no_data = False
        for i in range(sub_epochs):
            data_attempt = 0
            while True:
                no_data_action = _RANK0_ACTION_PROCEED
                if rank == 0:
                    if i != 0 or data_attempt > 0:
                        maybe_reload_training_data()
                    train_files_to_use = get_files_for_subepoch()
                    if train_files_to_use is None or len(train_files_to_use) <= 0:
                        if quit_if_no_data:
                            logging.info("Not enough data files to fill a subepoch! Quitting.")
                            no_data_action = _RANK0_ACTION_STOP
                        else:
                            logging.info("Not enough data files to fill a subepoch! Waiting 5m before retrying.")
                            no_data_action = _RANK0_ACTION_RETRY

                no_data_action = broadcast_rank0_action(
                    no_data_action if rank == 0 else None,
                    rank,
                    world_size,
                    device,
                )
                if no_data_action == _RANK0_ACTION_RETRY:
                    # All ranks leave the collective before sleeping, avoiding
                    # a process-group timeout during an arbitrarily long wait.
                    time.sleep(300)
                    data_attempt += 1
                    continue
                if no_data_action == _RANK0_ACTION_STOP:
                    quit_due_to_no_data = True
                    break
                if no_data_action != _RANK0_ACTION_PROCEED:
                    raise RuntimeError(f"Unknown rank 0 no-data action: {no_data_action}")
                break

            if quit_due_to_no_data:
                break

            if rank == 0:
                if barrier is not None:
                    barrier.wait()
                for wpipe in writepipes:
                    wpipe.send(train_files_to_use)
                # Wait briefly just in case to reduce chance of races with filesystem or anything else
                time.sleep(5)
            else:
                if barrier is not None:
                    barrier.wait()
                train_files_to_use = readpipes[rank-1].recv()

            # DDP need to wait on the main process after reloading data and sending files to train with
            if barrier is not None:
                barrier.wait()

            logging.info("Beginning training subepoch!")
            #logging.info("This subepoch, using files: " + str(train_files_to_use))
            logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
            lookahead_counter = 0
            for batch in data_processing_pytorch.read_npz_training_data(
                train_files_to_use,
                batch_size,
                world_size,
                rank,
                pos_len=pos_len,
                device=device,
                symmetry_type=symmetry_type,
                include_meta=raw_model.get_has_metadata_encoder(),
                history_matrices_type=history_matrices_type,
                model_config=model_config,
                require_full_board=disable_mask,
                filter_full_board_on_load=filter_full_board_on_load,
                binary_input_nhwc=input_nhwc,
            ):
                optimizer.zero_grad(set_to_none=True)
                extra_outputs = None
                # if raw_model.get_has_metadata_encoder():
                #     extra_outputs = ExtraOutputs([MetadataEncoder.OUTMEAN_KEY,MetadataEncoder.OUTLOGVAR_KEY])

                if use_fp16 or use_bf16:
                    with amp_autocast_context(use_fp16, use_bf16):
                        model_outputs = ddp_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                            extra_outputs=extra_outputs,
                            disable_mask=disable_mask,
                        )
                    model_outputs = raw_model.float32ify_output(model_outputs)
                else:
                    model_outputs = ddp_model(
                        batch["binaryInputNCHW"],
                        batch["globalInputNC"],
                        input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                        extra_outputs=extra_outputs,
                        disable_mask=disable_mask,
                    )

                postprocessed = raw_model.postprocess_output(model_outputs)
                metrics = training_metrics_fn(
                    raw_model,
                    postprocessed,
                    extra_outputs,
                    batch,
                    is_training=True,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    disable_optimistic_policy=disable_optimistic_policy,
                    meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=seki_loss_scale,
                    variance_time_loss_scale=variance_time_loss_scale,
                    main_loss_scale=main_loss_scale,
                    intermediate_loss_scale=intermediate_loss_scale,
                    include_model_norms=not model_norms_only_at_print,
                    assume_full_board=disable_mask,
                )
                if (
                    model_norms_only_at_print
                    and (batch_count_this_epoch + 1) % print_train_loss_every_batches == 0
                ):
                    metrics.update(metrics_obj.get_model_norm_metrics(raw_model))

                # DDP averages loss across instances, so to preserve LR as per-sample lr, we scale by world size.
                loss = metrics["loss_sum"] * world_size

                # Reduce gradients across DDP
                backward_and_unscale(loss, optimizer, scaler)

                if model_config["norm_kind"] == "fixup" or model_config["norm_kind"] == "fixscale" or model_config["norm_kind"] == "fixscaleonenorm":
                    gnorm_cap = 20000.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
                elif model_config["norm_kind"] == "bnorm" or model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
                    gnorm_cap = 50000.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
                else:
                    assert False

                if gnorm_stats_debug:
                    stats = metrics_obj.get_specific_norms_and_gradient_stats(raw_model)
                    for stat, value in stats.items():
                        metrics[stat] = value

                if "use_repvgg_learning_rate" in model_config and model_config["use_repvgg_learning_rate"]:
                    gradscale_constant = torch.tensor([[1.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,1.0]],dtype=torch.float32,device=device,requires_grad=False).view(1,1,3,3)
                    for name, param in ddp_model.named_parameters():
                        if "normactconv" in name and ".conv.weight" in name and len(param.shape) == 4 and param.shape[2] == 3 and param.shape[3] == 3:
                            param.grad *= gradscale_constant

                # Loosen gradient clipping as we shift to smaller learning rates
                gnorm_cap = gnorm_cap / math.sqrt(max(0.0000001,lr_scale * lr_scale_auto_factor(train_state)))

                gnorm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gnorm_cap).detach().cpu().item()

                if math.isfinite(gnorm) and abs(gnorm < 1e30):
                    metrics["gnorm_batch"] = gnorm
                    exgnorm = max(0.0, gnorm - gnorm_cap)
                    metrics["exgnorm_sum"] = exgnorm * batch_size

                metrics["pslr_batch"] = lr_right_now
                metrics["wdnormal_batch"] = normal_weight_decay_right_now
                metrics["gnorm_cap_batch"] = gnorm_cap
                metrics["batch_size_batch"] = batch_size * world_size
                metrics["world_size_batch"] = world_size

                optimizer_step(optimizer, scaler)

                batch_count_this_epoch += 1
                train_state["train_steps_since_last_reload"] += batch_size * world_size
                train_state["global_step_samples"] += batch_size * world_size

                metrics = detensorify_metrics(metrics)

                if model_norms_only_at_print and batch_count_this_epoch % print_train_loss_every_batches == 0:
                    missing_model_norm_metrics = [key for key in model_norm_metric_keys if key not in metrics]
                    if missing_model_norm_metrics:
                        raise RuntimeError(
                            "Model norm metrics were requested for logging but are missing: "
                            + ", ".join(missing_model_norm_metrics)
                        )

                if lookahead_k is not None and lookahead_print:
                    # Only accumulate metrics when lookahead is synced if lookahead_print is True
                    if lookahead_counter == 0:
                        accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=math.exp(-0.01 * lookahead_k), new_weight=1.0)
                    else:
                        accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=1.0, new_weight=0.0)
                else:
                    accumulate_metrics(running_metrics["sums"], running_metrics["weights"], metrics, batch_size, decay=0.99, new_weight=1.0)


                if batch_count_this_epoch % print_train_loss_every_batches == 0:

                    if model_norms_only_at_print:
                        # Norms are computed only for this print batch. Treat
                        # them as a snapshot, including when lookahead_print
                        # gives the surrounding metrics zero accumulation
                        # weight, so logging cannot divide 0 by 0.
                        set_snapshot_metrics(
                            running_metrics["sums"],
                            running_metrics["weights"],
                            metrics,
                            model_norm_metric_keys,
                        )

                    if model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
                        metrics["brn_rmax"] = train_state["brenorm_rmax"]
                        metrics["brn_dmax"] = train_state["brenorm_dmax"]
                        metrics["brn_mmnt"] = brenorm_avg_momentum
                        upper_rclippage = []
                        lower_rclippage = []
                        dclippage = []
                        raw_model.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
                        metrics["brn_ruclip"] = sum(upper_rclippage) / max(len(upper_rclippage),1.0)
                        metrics["brn_rlclip"] = sum(lower_rclippage) / max(len(lower_rclippage),1.0)
                        metrics["brn_dclip"] = sum(dclippage) / max(len(dclippage),1.0)

                    t1 = time.perf_counter()
                    timediff = t1 - last_train_stats_time
                    last_train_stats_time = t1
                    metrics["time_since_last_print"] = timediff
                    log_metrics(running_metrics["sums"], running_metrics["weights"], metrics, train_metrics_out, exportprefix)

                # Update LR more frequently at the start for smoother warmup ramp and wd adjustment
                if train_state["global_step_samples"] <= 350000000 and batch_count_this_epoch % 50 == 0:
                    lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()

                # Update batch renorm parameters
                if batch_count_this_epoch % 100 == 0:
                    maybe_update_brenorm_params()

                # Perform lookahead
                in_between_lookaheads = False
                if lookahead_k is not None:
                    lookahead_counter += 1
                    if lookahead_counter >= lookahead_k:
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                slow_param_data = lookahead_cache[param]
                                slow_param_data.add_(param.data.detach() - slow_param_data, alpha=lookahead_alpha)
                                param.data.copy_(slow_param_data)
                        lookahead_counter = 0
                        in_between_lookaheads = False
                    else:
                        in_between_lookaheads = True

                # Perform SWA
                if len(swa_models)>0:
                    assert(len(swa_scales)==len(swa_models))
                    train_state["swa_sample_accum"] += batch_size * world_size
                    # Only snap SWA when lookahead slow params are in sync.
                    if train_state["swa_sample_accum"] >= swa_period_samples and not in_between_lookaheads:
                        train_state["swa_sample_accum"] = 0
                        #logging.info("Accumulating SWA")
                        for i in range(len(swa_models)):
                            #sync_swa_buffers_shape(swa_models[i],raw_model)
                            #logging.info(f"Accumulating swa_scale={1/swa_models[i].avg_fn(0,1,0)}")
                            if swa_models[i] is None:
                                assert(qat_int8)
                                logging.info(f"Initializing qat swa_model[{i}] with swa_scale={1/swa_scales[i]}")
                                swa_models[i] = make_swa_model(
                                    raw_model, 1 / swa_scales[i]
                                )
                            swa_models[i].update_parameters(raw_model)

            logging.info("Finished training subepoch!")

        # END SUB EPOCH LOOP ------------

        if quit_due_to_no_data:
            break

        # Discard the gradient updates from the leftover batches in the sub epoch from lookahead.
        # This wastes a very tiny bit, but makes it so that we can be in sync and deterministic on ends of subepochs/epochs.
        if lookahead_k is not None:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    slow_param_data = lookahead_cache[param]
                    param.data.copy_(slow_param_data)

        if rank == 0:
            train_state["export_cycle_counter"] += 1

        save(raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics)

        num_epochs_this_instance += 1

        # Validate
        if rank == 0:
            logging.info("Beginning validation after epoch!")
            val_files = []
            if os.path.exists(vdatadir):
                val_files = [os.path.join(vdatadir,fname) for fname in os.listdir(vdatadir) if fname.endswith(".npz")]
            if randomize_val:
                random.shuffle(val_files)
            else:
                # Sort to ensure deterministic order to validation files in case we use only a subset
                val_files = sorted(val_files)
            if len(val_files) == 0:
                logging.info("No validation files, skipping validation step")
            else:
                validation_model = get_local_validation_model(
                    ddp_model,
                    raw_model,
                    world_size,
                )
                with torch.no_grad():
                    validation_model.eval()
                    val_metric_sums = defaultdict(float)
                    val_metric_weights = defaultdict(float)
                    val_samples = 0
                    t0 = time.perf_counter()
                    for batch in data_processing_pytorch.read_npz_training_data(
                        val_files,
                        batch_size,
                        world_size=1,  # Only the main process validates
                        rank=0,        # Only the main process validates
                        pos_len=pos_len,
                        device=device,
                        symmetry_type=symmetry_type,
                        include_meta=raw_model.get_has_metadata_encoder(),
                        history_matrices_type=history_matrices_type,
                        model_config=model_config,
                    ):
                        model_outputs = validation_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                        )
                        postprocessed = raw_model.postprocess_output(model_outputs)
                        extra_outputs = None
                        metrics = metrics_obj.metrics_dict_batchwise(
                            raw_model,
                            postprocessed,
                            extra_outputs,
                            batch,
                            is_training=False,
                            soft_policy_weight_scale=soft_policy_weight_scale,
                            disable_optimistic_policy=disable_optimistic_policy,
                            meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                            value_loss_scale=value_loss_scale,
                            td_value_loss_scales=td_value_loss_scales,
                            seki_loss_scale=seki_loss_scale,
                            variance_time_loss_scale=variance_time_loss_scale,
                            main_loss_scale=main_loss_scale,
                            intermediate_loss_scale=intermediate_loss_scale,
                        )
                        metrics = detensorify_metrics(metrics)
                        accumulate_metrics(val_metric_sums, val_metric_weights, metrics, batch_size, decay=1.0, new_weight=1.0)
                        val_samples += batch_size
                        if max_val_samples is not None and val_samples > max_val_samples:
                            break
                        val_metric_sums["nsamp_train"] = running_metrics["sums"]["nsamp"]
                        val_metric_weights["nsamp_train"] = running_metrics["weights"]["nsamp"]
                        val_metric_sums["wsum_train"] = running_metrics["sums"]["wsum"]
                        val_metric_weights["wsum_train"] = running_metrics["weights"]["wsum"]
                    last_val_metrics["sums"] = val_metric_sums
                    last_val_metrics["weights"] = val_metric_weights
                    log_metrics(val_metric_sums, val_metric_weights, metrics, val_metrics_out, exportprefix)
                    t1 = time.perf_counter()
                    logging.info(f"Validation took {t1-t0} seconds")
                    validation_model.train()

                for swa_idx in range(len(swa_models)):
                    swa_model=swa_models[swa_idx]
                    if swa_model is None:
                        assert qat_int8, f"swa_model_{swa_idx} is None but qat_int8 is False"
                        logging.warning(f"Skipping validating swa_model_{swa_idx} because it is None")
                        continue
                    
                    logging.info(f"Validating swa_scale={1/swa_model.avg_fn(0,1,0)}")
                    with torch.no_grad():
                        swa_model.eval()
                        val_metric_sums = defaultdict(float)
                        val_metric_weights = defaultdict(float)
                        val_samples = 0
                        t0 = time.perf_counter()
                        for batch in data_processing_pytorch.read_npz_training_data(
                            val_files,
                            batch_size,
                            world_size=1,  # Only the main process validates
                            rank=0,        # Only the main process validates
                            pos_len=pos_len,
                            device=device,
                            symmetry_type=symmetry_type,
                            include_meta=raw_model.get_has_metadata_encoder(),
                            history_matrices_type=history_matrices_type,
                            model_config=model_config,
                        ):
                            model_outputs = swa_model(
                                batch["binaryInputNCHW"],
                                batch["globalInputNC"],
                                input_meta=(batch["metadataInputNC"] if raw_model.get_has_metadata_encoder() else None),
                            )
                            postprocessed = swa_model.module.postprocess_output(model_outputs)
                            extra_outputs = None
                            metrics = metrics_obj.metrics_dict_batchwise(
                                swa_model.module,
                                postprocessed,
                                extra_outputs,
                                batch,
                                is_training=False,
                                soft_policy_weight_scale=soft_policy_weight_scale,
                                disable_optimistic_policy=disable_optimistic_policy,
                                meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                                value_loss_scale=value_loss_scale,
                                td_value_loss_scales=td_value_loss_scales,
                                seki_loss_scale=seki_loss_scale,
                                variance_time_loss_scale=variance_time_loss_scale,
                                main_loss_scale=main_loss_scale,
                                intermediate_loss_scale=intermediate_loss_scale,
                            )
                            metrics = detensorify_metrics(metrics)
                            accumulate_metrics(val_metric_sums, val_metric_weights, metrics, batch_size, decay=1.0, new_weight=1.0)
                            val_samples += batch_size
                            if max_val_samples is not None and val_samples > max_val_samples:
                                break
                            val_metric_sums["nsamp_train"] = running_metrics["sums"]["nsamp"]
                            val_metric_weights["nsamp_train"] = running_metrics["weights"]["nsamp"]
                            val_metric_sums["wsum_train"] = running_metrics["sums"]["wsum"]
                            val_metric_weights["wsum_train"] = running_metrics["weights"]["wsum"]
                        
                        log_metrics(val_metric_sums, val_metric_weights, metrics,  val_swa_metrics_outs[swa_idx], exportprefix)
                        t1 = time.perf_counter()
                        logging.info(f"Validation swa took {t1-t0} seconds")
                        swa_model.train()

        if rank == 0:
            logging.info("Export cycle counter = " + str(train_state["export_cycle_counter"]))

            is_time_to_export = False
            if train_state["export_cycle_counter"] >= epochs_per_export:
                if no_export:
                    train_state["export_cycle_counter"] = epochs_per_export
                else:
                    train_state["export_cycle_counter"] = 0
                    is_time_to_export = True

            skip_export_this_time = False
            if export_prob is not None:
                if random.random() > export_prob:
                    skip_export_this_time = True
                    logging.info("Skipping export model this time")

            if not no_export and is_time_to_export and not skip_export_this_time and exportdir is not None and not gnorm_stats_debug:
                # Export a model for testing, unless somehow it already exists
                modelname = "%s-s%d-d%d" % (
                    exportprefix,
                    train_state["global_step_samples"],
                    train_state["total_num_data_rows"],
                )
                savepath = os.path.join(exportdir,modelname)
                savepathtmp = os.path.join(exportdir,modelname+".tmp")
                if os.path.exists(savepath):
                    logging.info("NOT saving model, already exists at: " + savepath)
                else:
                    os.mkdir(savepathtmp)
                    logging.info("SAVING MODEL FOR EXPORT TO: " + savepath)
                    save(raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics, path=os.path.join(savepathtmp,"model.ckpt"), skip_optimizer=True)
                    time.sleep(2)
                    os.rename(savepathtmp,savepath)


        if sleep_seconds_per_epoch is None:
            time.sleep(1)
        else:
            time.sleep(sleep_seconds_per_epoch)

        if rank == 0:
            now = datetime.datetime.now()
            if now - last_longterm_checkpoint_save_time >= datetime.timedelta(hours=12):
                last_longterm_checkpoint_save_time = now
                dated_name = datetime.datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                save(raw_model, swa_models, optimizer, metrics_obj, running_metrics, train_state, last_val_metrics, path=os.path.join(longterm_checkpoints_dir,f"{dated_name}.ckpt"), skip_optimizer=True)

        # Rank 0 performs validation and export locally. Keep peers alive until
        # that work is complete, including when the next loop iteration exits
        # immediately due to an epoch/sample limit.
        if barrier is not None:
            barrier.wait()

    train_metrics_out.close()
    val_metrics_out.close()


if __name__ == "__main__":
    multi_gpus = args["multi_gpus"]
    num_gpus_used = 1
    multi_gpu_device_ids = []
    if multi_gpus is not None:
        for piece in multi_gpus.split(","):
            piece = piece.strip()
            multi_gpu_device_ids.append(int(piece))
        num_gpus_used = len(multi_gpu_device_ids)
    else:
        multi_gpu_device_ids = [0]

    make_dirs(args)

    readpipes = []
    writepipes = []

    if num_gpus_used > 1:
        torch.multiprocessing.set_start_method("spawn")

        world_size = num_gpus_used
        barrier = torch.multiprocessing.Barrier(num_gpus_used)

        for i in range(world_size - 1):
            rpipe, wpipe = torch.multiprocessing.Pipe()
            readpipes.append(rpipe)
            writepipes.append(wpipe)

        torch.multiprocessing.spawn(
            main,
            nprocs=num_gpus_used,
            args=(world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier)
        )
    else:
        rank = 0
        world_size = 1
        barrier = None
        main(rank, world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier)
