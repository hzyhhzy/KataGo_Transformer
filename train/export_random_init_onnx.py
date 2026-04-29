#!/usr/bin/python3
import argparse
import datetime
import logging
import os
import sys
from typing import Optional, Tuple

import torch
import torch.onnx

import modelconfigs
from model_pytorch import Model


def manual_rms_norm_forward(self, x):
    x_f32 = x.float()
    mean_square = (x_f32 * x_f32).mean(-1, keepdim=True)
    eps_tensor = torch.tensor([self.eps], dtype=x_f32.dtype, device=x_f32.device)
    inv_rms = torch.rsqrt(mean_square + eps_tensor)
    return self.weight * (x_f32 * inv_rms).type_as(x)


if hasattr(torch.nn, "RMSNorm"):
    torch.nn.RMSNorm.forward = manual_rms_norm_forward


class ONNXExportWrapper(torch.nn.Module):
    def __init__(self, model: Model, disable_mask: bool):
        super().__init__()
        self.model = model
        self.disable_mask = disable_mask
        self.has_metadata_encoder = model.get_has_metadata_encoder()

    def forward(
        self,
        input_spatial: torch.Tensor,
        input_global: torch.Tensor,
        input_meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if self.has_metadata_encoder:
            outputs = self.model(input_spatial, input_global, input_meta, disable_mask=self.disable_mask)
        else:
            outputs = self.model(input_spatial, input_global, disable_mask=self.disable_mask)
        main_outputs = outputs[0]
        return (
            main_outputs[0],
            main_outputs[1],
            main_outputs[2],
            main_outputs[3],
        )


def build_dummy_inputs(model: Model, batch_size: int, pos_len: int):
    pos_volume = pos_len * pos_len * pos_len
    num_spatial_inputs = modelconfigs.get_num_bin_input_features(model.config)
    num_global_inputs = modelconfigs.get_num_global_input_features(model.config)

    input_spatial = torch.randn(batch_size, num_spatial_inputs, pos_volume, dtype=torch.float32)
    input_spatial[:, 0, :] = 1.0
    input_global = torch.randn(batch_size, num_global_inputs, dtype=torch.float32)

    input_meta = None
    if model.get_has_metadata_encoder():
        num_meta_inputs = modelconfigs.get_num_meta_encoder_input_features(model.config)
        input_meta = torch.randn(batch_size, num_meta_inputs, dtype=torch.float32)

    return input_spatial, input_global, input_meta


def add_metadata(
    export_path: str,
    model_config_name: str,
    model: Model,
    pos_len: int,
    disable_mask: bool,
    opset_version: int,
):
    try:
        import onnx
    except ImportError:
        logging.warning("onnx package not installed, skipping metadata writing")
        return

    onnx_model = onnx.load(export_path)
    if hasattr(onnx_model, "metadata_props"):
        del onnx_model.metadata_props[:]

    num_spatial_inputs = modelconfigs.get_num_bin_input_features(model.config)
    num_global_inputs = modelconfigs.get_num_global_input_features(model.config)
    metadata = {
        "name": model_config_name,
        "modelVersion": str(model.config["version"]),
        "exported_at": datetime.datetime.now().isoformat(),
        "model_config_name": model_config_name,
        "model_version": str(model.config["version"]),
        "opset_version": str(opset_version),
        "num_spatial_inputs": str(num_spatial_inputs),
        "num_global_inputs": str(num_global_inputs),
        "pos_len": str(pos_len),
        "pos_len_x": str(pos_len),
        "pos_len_y": str(pos_len),
        "pos_len_z": str(pos_len),
        "has_mask": "true" if not disable_mask else "false",
        "has_metadata_encoder": "true" if model.get_has_metadata_encoder() else "false",
        "auto_fp16_already": "false",
        "exported_with_dynamo": "false",
        "is_simplified": "false",
        "is_int8": "false",
        "random_initialized": "true",
        "model_config": str(model.config),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, export_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export a randomly initialized model from modelconfigs.py to ONNX."
    )
    parser.add_argument("-model-config", required=True, help="Name in modelconfigs.config_of_name")
    parser.add_argument("-pos-len", type=int, required=True, help="Spatial edge length")
    parser.add_argument("-output", help="Output ONNX path")
    parser.add_argument("-batch-size", type=int, default=1, help="Dummy batch size for export")
    parser.add_argument("-disable-mask", action="store_true", help="Disable model mask logic")
    parser.add_argument("-opset-version", type=int, default=20, help="ONNX opset version")
    parser.add_argument("-fix-batchsize", action="store_true", help="Disable dynamic batch axis")
    parser.add_argument("-verbose", action="store_true", help="Enable verbose ONNX export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])

    if args.model_config not in modelconfigs.config_of_name:
        available = ", ".join(sorted(modelconfigs.config_of_name.keys())[:20])
        raise KeyError(f"Unknown model config: {args.model_config}. Examples: {available}")

    export_path = args.output
    if export_path is None:
        mask_suffix = "nomask" if args.disable_mask else "mask"
        export_path = os.path.abspath(f"{args.model_config}_random_pos{args.pos_len}_{mask_suffix}.onnx")
    else:
        export_path = os.path.abspath(export_path)

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    model_config = modelconfigs.config_of_name[args.model_config].copy()
    model = Model(model_config, args.pos_len)
    model.initialize()
    model.eval()

    wrapper = ONNXExportWrapper(model, disable_mask=args.disable_mask)
    wrapper.eval()

    input_spatial, input_global, input_meta = build_dummy_inputs(model, args.batch_size, args.pos_len)
    inputs = [input_spatial, input_global]
    input_names = ["input_spatial", "input_global"]
    if input_meta is not None:
        inputs.append(input_meta)
        input_names.append("input_meta")

    output_names = [
        "out_policy",
        "out_value",
        "out_miscvalue",
        "out_moremiscvalue",
    ]

    dynamic_axes = None
    if not args.fix_batchsize:
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    logging.info(f"Exporting random initialized model: {args.model_config}")
    logging.info(f"pos_len={args.pos_len}, disable_mask={args.disable_mask}, batch_size={args.batch_size}")
    logging.info(f"output={export_path}")
    logging.info(f"input_spatial shape={tuple(input_spatial.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            export_path,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=args.verbose,
        )

    add_metadata(
        export_path=export_path,
        model_config_name=args.model_config,
        model=model,
        pos_len=args.pos_len,
        disable_mask=args.disable_mask,
        opset_version=args.opset_version,
    )
    logging.info("Export completed successfully")


if __name__ == "__main__":
    main()
