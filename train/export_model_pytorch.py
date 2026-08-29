#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import struct
import json
import datetime
import logging
import gzip
import io
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn
from torch.optim.swa_utils import AveragedModel

import modelconfigs
from model_pytorch import (
    Model,
    ResBlock,
    NestedBottleneckResBlock,
    NestedBottleneckTransformerBlock,
    TransformerRoPEGQABlock,
)
from load_model import load_model
from native_int8_v104 import upgrade_v102_bytes
from native_int8_calibration import (
    load_calibration_json,
    transformer_blocks_in_wire_order,
    validate_calibration_document,
)

#Command and args-------------------------------------------------------------------

description = """
Export neural net weights to file for KataGo engine.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint to test', required=True)
parser.add_argument('-export-dir', help='model file dir to save to', required=True)
parser.add_argument('-model-name', help='name to record in model file', required=True)
parser.add_argument('-filename-prefix', help='filename prefix to save to within dir', required=True)
parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
parser.add_argument('-export-14-as-15', help='Export model version 14 as 15', action="store_true", required=False)
parser.add_argument('-pos-len', help='Board side length used to construct the model', type=int, default=19, required=False)
parser.add_argument('-gzip', help='Write a deterministic .bin.gz file instead of .bin', action="store_true", required=False)
parser.add_argument(
    '-cpu-ptq-base',
    help=(
        'Export the FP32 staging format used by the unified CPU-PTQ exporter: '
        'checkpoint v102 becomes native v105 and checkpoint v11 becomes native v205.'
    ),
    action="store_true",
    required=False,
)
parser.add_argument(
    '-int8-pt-clip4',
    help='Explicitly export native v104 with embedded clip4/per-tensor INT8 weights',
    action="store_true",
    required=False,
)
parser.add_argument(
    '-int8-calibration-json',
    dest='int8_calibration_json',
    help=(
        'Strict PTQ calibration JSON. Supplying it upgrades a supported '
        'SwiGLU Transformer export to the extended native v105 wire format.'
    ),
    required=False,
)
args = vars(parser.parse_args())


def main(args):
    checkpoint_file = args["checkpoint"]
    export_dir = args["export_dir"]
    model_name = args["model_name"]
    filename_prefix = args["filename_prefix"]
    use_swa = args["use_swa"]
    export_14_as_15 = args["export_14_as_15"]
    pos_len = args["pos_len"]
    gzip_output = args["gzip"]
    cpu_ptq_base = args["cpu_ptq_base"]
    int8_pt_clip4 = args["int8_pt_clip4"]
    calibration_json_path = args["int8_calibration_json"]

    if pos_len <= 0:
        raise ValueError("-pos-len must be positive")

    os.makedirs(export_dir,exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(export_dir + "/log.txt"),
        ],
    )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # LOAD MODEL ---------------------------------------------------------------------
    model, swa_model, other_state_dict = load_model(
        checkpoint_file,
        use_swa,
        device="cpu",
        pos_len=pos_len,
        verbose=True,
    )
    model_config = model.config
    model_to_export = swa_model if swa_model is not None else model
    transformer_layers = transformer_blocks_in_wire_order(model_to_export)
    transformer_layer_order = [name for name, _ in transformer_layers]

    if int8_pt_clip4 and calibration_json_path is not None:
        raise ValueError("-int8-pt-clip4 and -int8-calibration-json are mutually exclusive")
    if calibration_json_path is not None:
        if not transformer_layers:
            raise ValueError("-int8-calibration-json requires at least one Transformer layer")
        calibration_document = load_calibration_json(
            Path(calibration_json_path).resolve()
        )
        int8_calibration = validate_calibration_document(
            calibration_document,
            checkpoint_path=Path(checkpoint_file).resolve(),
            layer_order=transformer_layer_order,
            use_swa=use_swa,
            pos_len=pos_len,
        )
    else:
        int8_calibration = None

    source_checkpoint_version = int(model_config["version"])
    if cpu_ptq_base and source_checkpoint_version not in (11, 102):
        raise ValueError(
            "-cpu-ptq-base supports checkpoint versions 11 and 102 only"
        )
    if cpu_ptq_base and int8_pt_clip4:
        raise ValueError("-cpu-ptq-base and -int8-pt-clip4 are mutually exclusive")
    if (
        cpu_ptq_base
        and source_checkpoint_version == 11
        and int8_calibration is not None
    ):
        raise ValueError(
            "checkpoint v11 uses the v205 staging schema and does not accept "
            "native-v105 static activation calibration"
        )
    if cpu_ptq_base and source_checkpoint_version == 102 and int8_calibration is None:
        # v105 has four legacy static-activation fields per Transformer layer.
        # v106 performs dynamic rowwise activation quantization and deliberately
        # does not consume them, but the staging body must remain structurally
        # valid. Positive unit placeholders avoid coupling CPU-PTQ export to the
        # unrelated native-v105 static-activation calibration workflow.
        int8_calibration = {
            layer_name: {
                "attentionInputQuantMaxAbs": 1.0,
                "attentionOutputQuantMaxAbs": 1.0,
                "ffnInputQuantMaxAbs": 1.0,
                "productQuantMaxAbs": 1.0,
            }
            for layer_name in transformer_layer_order
        }

    # Ignore what's in the config if less than 11 since a lot of testing models
    # are on old version but actually have various new architectures.
    # Native v105 extends the transformer wire format with per-head Q/K
    # RMSNorm, a SwiGLU operand-clipping scalar, and a mandatory calibrated
    # product range on every FFN. Old versions retain their original layout.
    needs_native_v105 = bool(model_config.get("use_qk_norm",False)) or \
        any(block.swiglu_clip is not None for _, block in transformer_layers) or \
        int8_calibration is not None or \
        (cpu_ptq_base and source_checkpoint_version == 102)
    version = max(model_config["version"],105 if needs_native_v105 else 11)
    true_version = version
    # Hack to be able to export version 14 as version 15
    if version == 14 and export_14_as_15:
        version = 15
    output_version = version
    if cpu_ptq_base:
        if version == 11:
            output_version = 205
        elif version != 105:
            raise ValueError(
                f"-cpu-ptq-base expected a v105 or v11 native schema, got v{version}"
            )
    if int8_pt_clip4 and version != 102:
        raise ValueError("-int8-pt-clip4 requires a native v102 model")

    if version >= 105 and transformer_layers and int8_calibration is None:
        raise ValueError(
            "extended native v105 Transformer export requires -int8-calibration-json"
        )

    # WRITING MODEL ----------------------------------------------------------------
    extension = ".bin.gz" if gzip_output else ".bin"
    mode = "wb"
    output_path = os.path.join(export_dir, filename_prefix + extension)
    raw_f = None
    if int8_pt_clip4:
        # Preserve the ordinary v102 serialization exactly, then derive all
        # explicit INT8 bytes from those serialized FP32 masters in one strict
        # in-memory transaction. The default v102 path below remains untouched.
        f = io.BytesIO()
    else:
        raw_f = open(output_path, mode)
    if gzip_output and not int8_pt_clip4:
        # Do not embed a timestamp or source filename. Given the same checkpoint,
        # CLI arguments, and zlib runtime, this makes the compressed stream stable.
        f = gzip.GzipFile(filename="", mode=mode, fileobj=raw_f, mtime=0)
    elif not int8_pt_clip4:
        f = raw_f
    def writeln(s):
        f.write((str(s)+"\n").encode(encoding="ascii",errors="backslashreplace"))
    def writestr(s):
        f.write(s.encode(encoding="ascii",errors="backslashreplace"))

    writeln(model_name)
    writeln(output_version)
    writeln(modelconfigs.get_num_bin_input_features(model_config))
    writeln(modelconfigs.get_num_global_input_features(model_config))

    if version <= 12 or (version >= 100 and version <= 199):
        assert model.td_score_multiplier == 20.0
        assert model.scoremean_multiplier == 20.0
        assert model.scorestdev_multiplier == 20.0
        assert model.lead_multiplier == 20.0
        assert model.variance_time_multiplier == 40.0
        assert model.shortterm_value_error_multiplier == 0.25
        assert model.shortterm_score_error_multiplier == 30.0
    else:
        writeln(model.td_score_multiplier)
        writeln(model.scoremean_multiplier)
        writeln(model.scorestdev_multiplier)
        writeln(model.lead_multiplier)
        writeln(model.variance_time_multiplier)
        writeln(model.shortterm_value_error_multiplier)
        writeln(model.shortterm_score_error_multiplier)

    if version >= 15 and version < 100:
        if model.metadata_encoder is not None:
            writeln(model.metadata_encoder.meta_encoder_version)
        else:
            writeln(0)

        # Write some dummy placeholders for future features
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)


    def write_weights(weights):
        # Little endian
        reshaped = np.reshape(weights.detach().numpy(),[-1])
        num_weights = len(reshaped)
        writestr("@BIN@")
        f.write(struct.pack(f'<{num_weights}f',*reshaped))
        writestr("\n")

    def write_conv_weight(name,convweight):
        (out_channels, in_channels, diamy, diamx) = convweight.shape
        dilation = 1
        writeln(name)
        writeln(diamy) #y
        writeln(diamx) #x
        writeln(in_channels)
        writeln(out_channels)
        writeln(dilation) #y
        writeln(dilation) #x
        # Torch order is oc,ic,y,x
        # Desired output order is y,x,ic,oc
        write_weights(torch.permute(convweight,(2,3,1,0)))

    def write_conv(name,conv):
        assert conv.bias is None
        write_conv_weight(name, conv.weight)

    def write_bn(name,normmask):
        writeln(name)

        writeln(normmask.c_in)
        epsilon = 1e-20
        writeln(epsilon)
        has_gamma_or_scale = normmask.scale is not None or normmask.gamma is not None
        has_beta = True
        writeln(1 if has_gamma_or_scale else 0)
        writeln(1 if has_beta else 0)

        if hasattr(normmask,"running_mean") and normmask.running_mean is not None:
            assert normmask.is_using_batchnorm
            assert normmask.running_mean.shape == (normmask.c_in,)
            write_weights(normmask.running_mean)
        else:
            assert not normmask.is_using_batchnorm
            write_weights(torch.zeros(normmask.c_in, dtype=torch.float))

        if hasattr(normmask,"running_std") and normmask.running_std is not None:
            assert normmask.is_using_batchnorm
            assert normmask.running_std.shape == (normmask.c_in,)
            write_weights(torch.maximum(torch.tensor(1e-20), normmask.running_std * normmask.running_std - epsilon))
        else:
            assert not normmask.is_using_batchnorm
            write_weights((1.0-epsilon) * torch.ones(normmask.c_in, dtype=torch.float))

        if normmask.scale is not None:
            if normmask.gamma is not None:
                assert normmask.gamma.shape == (1, normmask.c_in, 1, 1)
                assert has_gamma_or_scale
                write_weights(normmask.scale * (normmask.gamma + 1.0))
            else:
                assert has_gamma_or_scale
                write_weights(normmask.scale * torch.ones(normmask.c_in, dtype=torch.float, device="cpu"))
        else:
            if normmask.gamma is not None:
                assert normmask.gamma.shape == (1, normmask.c_in, 1, 1)
                assert has_gamma_or_scale
                write_weights(normmask.gamma + 1.0)
            else:
                assert not has_gamma_or_scale
                pass

        assert normmask.beta.shape == (1, normmask.c_in, 1, 1)
        write_weights(normmask.beta)

    def write_biasmask(name,biasmask):
        writeln(name)

        writeln(biasmask.c_in)
        epsilon = 1e-20
        writeln(epsilon)
        has_gamma_or_scale = biasmask.scale is not None
        has_beta = True
        writeln(1 if has_gamma_or_scale else 0)
        writeln(1 if has_beta else 0)

        write_weights(torch.zeros(biasmask.c_in, dtype=torch.float))
        write_weights((1.0-epsilon) * torch.ones(biasmask.c_in, dtype=torch.float))

        if biasmask.scale is not None:
            write_weights(biasmask.scale * torch.ones(biasmask.c_in, dtype=torch.float, device="cpu"))

        assert biasmask.beta.shape == (1, biasmask.c_in, 1, 1)
        write_weights(biasmask.beta)

    def write_activation(name, activation):
        writeln(name)
        if isinstance(activation,torch.nn.ReLU):
            writeln("ACTIVATION_RELU")
        elif isinstance(activation,torch.nn.Mish):
            writeln("ACTIVATION_MISH")
        elif isinstance(activation,torch.nn.SiLU):
            writeln("ACTIVATION_SILU")
        elif isinstance(activation,torch.nn.Identity):
            writeln("ACTIVATION_IDENTITY")
        else:
            assert False, f"Activation not supported for export: {activation}"


    def write_matmul(name,linearweight):
        writeln(name)
        (out_channels,in_channels) = linearweight.shape
        writeln(in_channels)
        writeln(out_channels)
        # Torch order is oc,ic
        # Desired output order is ic,oc
        write_weights(torch.permute(linearweight,(1,0)))

    def write_matbias(name,linearbias):
        writeln(name)
        (out_channels,) = linearbias.shape
        writeln(out_channels)
        write_weights(linearbias)

    def write_normactconv(name,normactconv):
        if normactconv.c_gpool is None:
            assert normactconv.convpool is None
            if normactconv.conv1x1 is None:
                write_bn(name+".norm", normactconv.norm)
                write_activation(name+".act", normactconv.act)
                write_conv(name+".conv", normactconv.conv)
            else:
                write_bn(name+".norm", normactconv.norm)
                write_activation(name+".act", normactconv.act)
                # Torch conv order is oc,ic,h,w
                # We want to add the 1x1 conv to the center of the h,w
                h,w = (normactconv.conv.weight.shape[2],normactconv.conv.weight.shape[3])
                assert h % 2 == 1, "Conv1x1 can't be merged with even-sized convolution kernel"
                assert w % 2 == 1, "Conv1x1 can't be merged with even-sized convolution kernel"
                combined_conv = normactconv.conv.weight.detach().clone()
                combined_conv[:,:,h//2:h//2+1,w//2:w//2+1] += normactconv.conv1x1.weight
                assert normactconv.conv.bias is None
                assert normactconv.conv1x1.bias is None
                write_conv_weight(name+".conv", combined_conv)
        else:
            assert normactconv.convpool is not None
            assert normactconv.conv1x1 is None
            write_bn(name+".norm", normactconv.norm)
            write_activation(name+".act", normactconv.act)
            write_conv(name+".convpool.conv1r", normactconv.convpool.conv1r)
            write_conv(name+".convpool.conv1g", normactconv.convpool.conv1g)
            write_bn(name+".convpool.normg", normactconv.convpool.normg)
            write_activation(name+".convpool.actg", normactconv.convpool.actg)
            write_matmul(name+".convpool.linear_g", normactconv.convpool.linear_g.weight)
            assert normactconv.convpool.linear_g.bias is None

    def write_transformer_norm(name,rmsnorm):
        if not isinstance(rmsnorm, torch.nn.RMSNorm):
            raise ValueError(f"{name}: native transformer format requires RMSNorm")
        if not rmsnorm.elementwise_affine or rmsnorm.weight is None or rmsnorm.weight.ndim != 1:
            raise ValueError(f"{name}: native transformer RMSNorm requires one-dimensional affine weights")
        if not isinstance(rmsnorm.eps, float) or not math.isfinite(rmsnorm.eps) or rmsnorm.eps <= 0.0 or rmsnorm.eps > 1.0:
            raise ValueError(f"{name}: RMSNorm epsilon must be finite and in (0, 1]")
        writeln(name)
        writeln(rmsnorm.weight.shape[0])
        writeln(rmsnorm.eps)
        write_weights(rmsnorm.weight)

    def validate_combined_transformer(name,block):
        if not isinstance(block, TransformerRoPEGQABlock):
            raise ValueError(f"{name}: expected TransformerRoPEGQABlock, got {type(block)}")
        # These are semantic compatibility gates rather than internal
        # invariants. Keep them active under ``python -O`` so an exporter can
        # never silently discard a trained operation.
        if block.full_int8_clip is not None:
            raise ValueError(f"{name}: full_int8_clip is not supported by the native model format")
        if block.swiglu_clip is not None:
            if not isinstance(block.swiglu_clip,(int,float)) or \
               not math.isfinite(float(block.swiglu_clip)) or \
               float(block.swiglu_clip) <= 0.0:
                raise ValueError(f"{name}: swiglu_clip must be finite and positive")
        if block.use_qk_norm:
            if not isinstance(block.q_norm, torch.nn.RMSNorm) or \
               not isinstance(block.k_norm, torch.nn.RMSNorm):
                raise ValueError(f"{name}: enabled QK norm requires q/k RMSNorm modules")
            if block.q_norm.normalized_shape != (block.head_dim,) or \
               block.k_norm.normalized_shape != (block.head_dim,):
                raise ValueError(f"{name}: q/k RMSNorm shape must equal head_dim")
        elif not isinstance(block.q_norm, torch.nn.Identity) or \
             not isinstance(block.k_norm, torch.nn.Identity):
            raise ValueError(f"{name}: disabled QK norm must use identity modules")
        if not block.use_swiglu:
            raise ValueError(f"{name}: the native CUDA transformer backend requires SwiGLU")
        if block.num_heads <= 0 or block.num_kv_heads <= 0 or block.num_heads % block.num_kv_heads != 0:
            raise ValueError(f"{name}: invalid query/KV head counts")
        if block.head_dim <= 0 or block.head_dim % 2 != 0:
            raise ValueError(f"{name}: head_dim must be positive and even")
        trunk_channels = block.q_proj.in_features
        projection_shapes_match = (
            block.q_proj.out_features == block.num_heads * block.head_dim and
            block.k_proj.in_features == trunk_channels and
            block.k_proj.out_features == block.num_kv_heads * block.head_dim and
            block.v_proj.in_features == trunk_channels and
            block.v_proj.out_features == block.num_kv_heads * block.head_dim and
            block.out_proj.in_features == block.num_heads * block.head_dim and
            block.out_proj.out_features == trunk_channels
        )
        if block.q_proj.out_features != trunk_channels or not projection_shapes_match:
            raise ValueError(f"{name}: attention projection shapes do not match the declared geometry")
        ffn_shapes_match = (
            block.ffn_dim > 0 and
            block.ffn_linear1.in_features == trunk_channels and
            block.ffn_linear1.out_features == block.ffn_dim and
            block.ffn_linear_gate.in_features == trunk_channels and
            block.ffn_linear_gate.out_features == block.ffn_dim and
            block.ffn_linear2.in_features == block.ffn_dim and
            block.ffn_linear2.out_features == trunk_channels
        )
        if not ffn_shapes_match:
            raise ValueError(f"{name}: FFN projection shapes do not match the declared geometry")
        if any(layer.bias is not None for layer in (
            block.q_proj,
            block.k_proj,
            block.v_proj,
            block.out_proj,
            block.ffn_linear1,
            block.ffn_linear_gate,
            block.ffn_linear2,
        )):
            raise ValueError(f"{name}: native transformer projections do not support bias")
        if not isinstance(block.norm1, torch.nn.RMSNorm) or not isinstance(block.norm2, torch.nn.RMSNorm):
            raise ValueError(f"{name}: native transformer format requires RMSNorm")
        if block.use_rope:
            if block.learnable_rope:
                expected_shape = (block.num_kv_heads, block.head_dim // 2, 2)
                if tuple(block.rope_freqs.shape) != expected_shape:
                    raise ValueError(f"{name}: learnable RoPE shape must be {expected_shape}")
            else:
                if block.head_dim % 4 != 0:
                    raise ValueError(f"{name}: fixed 2D RoPE requires head_dim divisible by 4")
                if not math.isfinite(block.rope_theta) or block.rope_theta <= 0.0:
                    raise ValueError(f"{name}: fixed RoPE theta must be finite and positive")
        else:
            if block.learnable_rope:
                raise ValueError(f"{name}: learnable_rope requires use_rope")

    def write_transformer_attention_block(name,block):
        validate_combined_transformer(name, block)
        writeln("transformer_attention_block")
        writeln(name)
        writeln(block.num_heads)
        writeln(block.num_kv_heads)
        writeln(block.head_dim)
        writeln(block.head_dim)
        writeln(1 if block.use_rope else 0)
        writeln(1 if block.learnable_rope else 0)
        if version >= 105:
            writeln(1 if block.use_qk_norm else 0)
            layer_name = name[:-len(".attention")]
            if int8_calibration is None or layer_name not in int8_calibration:
                raise ValueError(f"{name}: missing INT8 calibration")
            writeln(int8_calibration[layer_name]["attentionInputQuantMaxAbs"])
            writeln(int8_calibration[layer_name]["attentionOutputQuantMaxAbs"])

        write_transformer_norm(name+".norm1", block.norm1)
        write_matmul(name+".q_proj", block.q_proj.weight)
        write_matmul(name+".k_proj", block.k_proj.weight)
        write_matmul(name+".v_proj", block.v_proj.weight)
        write_matmul(name+".out_proj", block.out_proj.weight)
        if version >= 105 and block.use_qk_norm:
            write_transformer_norm(name+".q_norm",block.q_norm)
            write_transformer_norm(name+".k_norm",block.k_norm)

        if block.use_rope:
            if block.learnable_rope:
                freqs = block.rope_freqs.detach()
                writeln(name+".rope_freqs")
                writeln(freqs.shape[0])
                writeln(freqs.shape[1])
                writeln(freqs.shape[2])
                write_weights(freqs)
            else:
                writeln(name+".rope_theta")
                writeln(block.rope_theta)

    used_int8_calibration_layers = set()

    def write_transformer_ffn_block(name,block):
        validate_combined_transformer(name, block)
        writeln("transformer_ffn_block")
        writeln(name)
        writeln(block.q_proj.in_features)
        writeln(block.ffn_dim)
        writeln(1 if block.use_swiglu else 0)
        if version >= 105:
            # This is a model semantic, never a PTQ override. Zero means the
            # checkpoint has no SwiGLU operand clipping.
            writeln(0.0 if block.swiglu_clip is None else float(block.swiglu_clip))
            layer_name = name[:-len(".ffn")]
            if int8_calibration is None or layer_name not in int8_calibration:
                raise ValueError(f"{name}: missing INT8 calibration")
            writeln(int8_calibration[layer_name]["ffnInputQuantMaxAbs"])
            writeln(int8_calibration[layer_name]["productQuantMaxAbs"])
            used_int8_calibration_layers.add(layer_name)

        write_transformer_norm(name+".norm", block.norm2)
        write_matmul(name+".ffn_linear1", block.ffn_linear1.weight)
        if block.use_swiglu:
            write_matmul(name+".ffn_linear_gate", block.ffn_linear_gate.weight)
        write_matmul(name+".ffn_linear2", block.ffn_linear2.weight)

    def logical_block_count(block):
        # A combined PyTorch transformer block has two residual operations on
        # the wire. Other block kinds, including nested blocks, have one outer
        # descriptor (their own inner count is written inside that descriptor).
        return 2 if isinstance(block, TransformerRoPEGQABlock) else 1

    def write_block(name,block):
        if isinstance(block,ResBlock) and block.normactconv1.c_gpool is None:
            assert block.normactconv2.c_gpool is None
            writeln("ordinary_block")
            writeln(name)
            write_normactconv(name+".normactconv1", block.normactconv1)
            write_normactconv(name+".normactconv2", block.normactconv2)
        elif isinstance(block,ResBlock) and block.normactconv1.c_gpool is not None:
            assert block.normactconv2.c_gpool is None
            writeln("gpool_block")
            writeln(name)
            write_normactconv(name+".normactconv1", block.normactconv1)
            write_normactconv(name+".normactconv2", block.normactconv2)
        elif isinstance(block,(NestedBottleneckResBlock,NestedBottleneckTransformerBlock)):
            writeln("nested_bottleneck_block")
            writeln(name)
            if block.internal_length != len(block.blockstack):
                raise ValueError(f"{name}: nested block length does not match its block stack")
            writeln(sum(logical_block_count(subblock) for subblock in block.blockstack))
            write_normactconv(name+".normactconvp", block.normactconvp)
            for i,subblock in enumerate(block.blockstack):
                write_block(name+".blockstack."+str(i),subblock)
            write_normactconv(name+".normactconvq", block.normactconvq)
        elif isinstance(block,TransformerRoPEGQABlock):
            # Training combines attention and FFN in one module, while the
            # native engine format stores them as two consecutive descriptors.
            write_transformer_attention_block(name+".attention", block)
            write_transformer_ffn_block(name+".ffn", block)
        else:
            raise ValueError(f"This kind of block is not supported for export right now: {type(block)}")

    def write_metadata_encoder(name,encoder):
        writeln(name)
        writeln(encoder.c_input)
        # Torch order is oc,ic. Flatten feature mask into the first mul
        write_matmul(name+".mul1", encoder.linear1.weight * encoder.feature_mask.reshape((1,-1)))
        write_matbias(name+".bias1", encoder.linear1.bias)
        write_activation(name+".act1", encoder.act1)
        write_matmul(name+".mul2", encoder.linear2.weight)
        write_matbias(name+".bias2", encoder.linear2.bias)
        write_activation(name+".act2", encoder.act2)
        write_matmul(name+".mul3", encoder.out_scale * encoder.linear_output_to_trunk.weight)
        assert encoder.linear_output_to_trunk.bias is None

    def write_trunk(name,model):
        writeln("trunk")
        writeln(sum(logical_block_count(block) for block in model.blocks))
        writeln(model.c_trunk)
        writeln(model.c_mid)
        writeln(model.c_mid-model.c_gpool)
        writeln(model.c_gpool)
        writeln(model.c_gpool)
        if version >= 15 and version < 100:
            # Write some dummy placeholders for future features
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)

        write_conv("model.conv_spatial", model.conv_spatial)
        write_matmul("model.linear_global", model.linear_global.weight)
        assert model.linear_global.bias is None
        if model.metadata_encoder is not None:
            assert version >= 15 and version < 100
            write_metadata_encoder("model.sgf_metadata_encoder",model.metadata_encoder)

        for i,block in enumerate(model.blocks):
            write_block("model.blocks."+str(i), block)
        if model.trunk_normless:
            write_biasmask("model.norm_trunkfinal", model.norm_trunkfinal)
        else:
            write_bn("model.norm_trunkfinal", model.norm_trunkfinal)
        write_activation("model.act_trunkfinal", model.act_trunkfinal)

    def write_policy_head(name,policyhead):
        writeln(name)
        write_conv(name+".conv1p", policyhead.conv1p)
        write_conv(name+".conv1g", policyhead.conv1g)
        write_biasmask(name+".biasg", policyhead.biasg)
        write_activation(name+".actg", policyhead.actg)
        write_matmul(name+".linear_g", policyhead.linear_g.weight)
        assert policyhead.linear_g.bias is None
        write_biasmask(name+".bias2", policyhead.bias2)
        write_activation(name+".act2", policyhead.act2)

        # Write the this-move prediction and the optimistic policy prediction
        if version <= 11 or (version >= 100 and version <= 199):
            assert policyhead.conv2p.weight.shape[0] == 4
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0],), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 4
            write_matmul(name+".linear_pass", torch.stack((policyhead.linear_pass.weight[0],), dim=0))
            assert policyhead.linear_pass.bias is None
        elif version <= 14:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 6
            write_matmul(name+".linear_pass", torch.stack((policyhead.linear_pass.weight[0], policyhead.linear_pass.weight[5]), dim=0))
            assert policyhead.linear_pass.bias is None
        elif version == 15 and true_version == 14:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 6
            linear_pass_stack = [policyhead.linear_pass.weight[0], policyhead.linear_pass.weight[5]]
            c_p1 = int(policyhead.linear_g.weight.shape[0])
            for _ in range(c_p1-2):
                linear_pass_stack.append(torch.zeros_like(linear_pass_stack[0]))
            write_matmul(name+".linear_pass", torch.stack(linear_pass_stack, dim=0))
            assert policyhead.linear_pass.bias is None
            write_matbias(name+".linear_pass_bias", torch.tensor([0.0]*c_p1,dtype=torch.float32,device="cpu"))
            write_activation(name+".act_pass", torch.nn.Identity())
            write_matmul(name+".linear_pass2", torch.tensor([[1.0,0.0]+[0.0]*(c_p1-2),[0.0,1.0]+[0.0]*(c_p1-2)],dtype=torch.float32,device="cpu"))
        else:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            write_matmul(name+".linear_pass", policyhead.linear_pass.weight)
            write_matbias(name+".linear_pass_bias", policyhead.linear_pass.bias)
            write_activation(name+".act_pass", policyhead.act_pass)
            assert policyhead.linear_pass2.weight.shape[0] == 6
            write_matmul(name+".linear_pass2", torch.stack((policyhead.linear_pass2.weight[0], policyhead.linear_pass2.weight[5]), dim=0))
            assert policyhead.linear_pass2.bias is None

        assert policyhead.conv2p.bias is None


    def write_value_head(name, valuehead):
        writeln(name)
        write_conv(name+".conv1", valuehead.conv1)
        write_biasmask(name+".bias1", valuehead.bias1)
        write_activation(name+".act1", valuehead.act1)
        write_matmul(name+".linear2", valuehead.linear2.weight)
        write_matbias(name+".bias2", valuehead.linear2.bias)
        write_activation(name+".act2", valuehead.act2)
        write_matmul(name+".linear_valuehead", valuehead.linear_valuehead.weight)
        write_matbias(name+".bias_valuehead", valuehead.linear_valuehead.bias)
        #write_matmul(name+".linear_valuehead", 2*valuehead.linear_moremiscvaluehead.weight[2:5])
        #write_matbias(name+".bias_valuehead", 2*valuehead.linear_moremiscvaluehead.bias[2:5])

        # For now, only output the scoremean and scorestdev and lead and vtime channels
        w = valuehead.linear_miscvaluehead.weight[0:4]
        b = valuehead.linear_miscvaluehead.bias[0:4]
        # Grab the shortterm channels
        w2 = valuehead.linear_moremiscvaluehead.weight[0:2]
        b2 = valuehead.linear_moremiscvaluehead.bias[0:2]
        w = torch.cat((w,w2),dim=0)
        b = torch.cat((b,b2),dim=0)
        write_matmul(name+".linear_miscvaluehead", w)
        write_matbias(name+".bias_miscvaluehead", b)

        write_conv(name+".conv_ownership",valuehead.conv_ownership)

    def write_model(model):
        write_trunk("model",model)
        write_policy_head("model.policy_head",model.policy_head)
        write_value_head("model.value_head",model.value_head)

    if swa_model is not None:
        logging.info("Writing SWA model")
        write_model(swa_model.module if hasattr(swa_model, "module") else swa_model)
    else:
        logging.info("Writing model")
        write_model(model)
    if int8_calibration is not None and used_int8_calibration_layers != set(transformer_layer_order):
        raise ValueError(
            "INT8 calibration consumption did not match Transformer wire order"
        )
    if int8_pt_clip4:
        v102_body = f.getvalue()
        f.close()
        upgraded = upgrade_v102_bytes(v102_body)
        raw_f = open(output_path, mode)
        if gzip_output:
            with gzip.GzipFile(filename="", mode=mode, fileobj=raw_f, mtime=0) as gzip_f:
                gzip_f.write(upgraded.data)
        else:
            raw_f.write(upgraded.data)
        raw_f.close()
        logging.info(
            "Embedded native v104 INT8 trailer: entries=%d payload_bytes=%d payload_sha256=%s",
            len(upgraded.entries),
            len(upgraded.payload),
            upgraded.payload_sha256,
        )
    else:
        f.close()
        if gzip_output:
            raw_f.close()

    with open(os.path.join(export_dir,"metadata.json"),"w") as f:
        train_state = other_state_dict["train_state"]
        data = {}
        if "global_step_samples" in train_state:
            data["global_step_samples"] = train_state["global_step_samples"]
        if "total_num_data_rows" in train_state:
            data["total_num_data_rows"] = train_state["total_num_data_rows"]
        if "running_metrics" in other_state_dict:
            assert sorted(list(other_state_dict["running_metrics"].keys())) == ["sums", "weights"]
            data["extra_stats"] = {
                "sums": { key: value for (key,value) in other_state_dict["running_metrics"]["sums"].items() if "sopt" not in key and "lopt" not in key },
                "weights": { key: value for (key,value) in other_state_dict["running_metrics"]["weights"].items() if "sopt" not in key and "lopt" not in key },
            }
            if "last_val_metrics" in other_state_dict and "sums" in other_state_dict["last_val_metrics"] and "weights" in other_state_dict["last_val_metrics"]:
                data["extra_stats"]["last_val_metrics"] = {
                    "sums": { key: value for (key,value) in other_state_dict["last_val_metrics"]["sums"].items() if "sopt" not in key and "lopt" not in key },
                    "weights": { key: value for (key,value) in other_state_dict["last_val_metrics"]["weights"].items() if "sopt" not in key and "lopt" not in key },
                }
        json.dump(data,f)


    logging.info("Exported at: ")
    logging.info(str(datetime.datetime.utcnow()) + " UTC")

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main(args)
