#!/usr/bin/env python3
"""Unified checkpoint/native exporter for CPU-PTQ v106 and v206.

Checkpoint v102 is serialized through FP32 native v105 and becomes v106.
Checkpoint v11 is serialized through FP32 native v205 and becomes v206.
Callers that already have a v105/v205 model can pass it directly instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from convert_cpu_ptq import (
    convert,
    sha256_file,
    write_atomic_json,
)


def _copy_base(source: Path, destination: Path, force: bool) -> None:
    destination = destination.resolve()
    if not destination.name.endswith(".bin.gz"):
        raise ValueError("--base-output must end in .bin.gz")
    if destination.exists() and not force:
        raise ValueError(f"refusing to overwrite existing base model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _export_checkpoint_base(
    checkpoint: Path,
    destination: Path,
    *,
    model_name: str,
    pos_len: int,
    use_swa: bool,
    native_calibration_json: Path | None,
) -> None:
    script = Path(__file__).with_name("export_model_pytorch.py")
    command = [
        sys.executable,
        str(script),
        "-checkpoint",
        str(checkpoint),
        "-export-dir",
        str(destination.parent),
        "-filename-prefix",
        destination.name[: -len(".bin.gz")],
        "-model-name",
        model_name,
        "-pos-len",
        str(pos_len),
        "-gzip",
        "-cpu-ptq-base",
    ]
    if use_swa:
        command.append("-use-swa")
    if native_calibration_json is not None:
        command.extend(
            ["-int8-calibration-json", str(native_calibration_json.resolve())]
        )
    subprocess.run(command, check=True)
    if not destination.is_file():
        raise RuntimeError("native base exporter did not create its requested output")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export CPU-PTQ from checkpoint v102/v11 or native base v105/v205; "
            "the target is inferred as v106/v206"
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="native FP32 v105/v205 .bin(.gz)")
    inputs.add_argument("--checkpoint", type=Path, help="checkpoint v102 or v11")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--pos-len", type=int, default=7)
    swa = parser.add_mutually_exclusive_group()
    swa.add_argument("--use-swa", dest="use_swa", action="store_true")
    swa.add_argument("--no-swa", dest="use_swa", action="store_false")
    parser.set_defaults(use_swa=True)
    parser.add_argument(
        "--native-calibration-json",
        type=Path,
        help=(
            "optional true v105 static-activation calibration; CPU-PTQ v106 "
            "otherwise uses structurally valid placeholders because its runtime "
            "activation quantization is dynamic"
        ),
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        help="optionally retain the generated v105/v205 FP32 staging model",
    )
    parser.add_argument("--projection-bits", type=int, choices=(7, 8))
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.pos_len <= 0:
        raise ValueError("--pos-len must be positive")
    if args.checkpoint is None:
        if args.model_name is not None:
            raise ValueError("--model-name is only valid with --checkpoint")
        if args.native_calibration_json is not None:
            raise ValueError(
                "--native-calibration-json is only valid with --checkpoint"
            )
        if args.base_output is not None:
            raise ValueError("--base-output is only valid with --checkpoint")
        report = convert(
            args.source,
            args.output,
            manifest_path=args.manifest,
            projection_bits=args.projection_bits,
            force=args.force,
            compression_level=args.gzip_level,
            write_report=not args.no_report,
        )
    else:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint does not exist: {checkpoint}")
        if not args.model_name:
            raise ValueError("--model-name is required with --checkpoint")
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # tempfile.TemporaryDirectory uses a restrictive Windows ACL that is
        # not inherited by the child exporter in some sandboxed environments.
        # A unique ordinary directory remains private by name and works on both
        # Windows and Linux; the resolved parent and exact child are known here.
        staging = output.parent / (".cpu-ptq-export-" + uuid.uuid4().hex)
        staging.mkdir()
        try:
            base = staging / "cpu-ptq-base.bin.gz"
            _export_checkpoint_base(
                checkpoint,
                base,
                model_name=args.model_name,
                pos_len=args.pos_len,
                use_swa=args.use_swa,
                native_calibration_json=args.native_calibration_json,
            )
            base_sha256 = sha256_file(base)
            if args.base_output is not None:
                _copy_base(base, args.base_output, args.force)
            report = convert(
                base,
                output,
                manifest_path=args.manifest,
                projection_bits=args.projection_bits,
                force=args.force,
                compression_level=args.gzip_level,
                write_report=False,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        report["checkpoint"] = str(checkpoint)
        report["checkpointSha256"] = sha256_file(checkpoint)
        report["useSwa"] = args.use_swa
        report["generatedBaseSha256"] = base_sha256
        report["baseOutput"] = (
            str(args.base_output.resolve()) if args.base_output is not None else None
        )
        # Do not leave a now-deleted temporary path in the durable report.
        report["source"] = report["baseOutput"]
        report["sourceFileSha256"] = base_sha256
        if not args.no_report:
            write_atomic_json(output.with_name(output.name + ".json"), report)

    for key in (
        "profile",
        "modelVersion",
        "projectionCount",
        "projectionBits",
        "projectionQuantizer",
        "outputSha256",
        "output",
    ):
        print(f"{key}={report[key]}")


if __name__ == "__main__":
    main()
