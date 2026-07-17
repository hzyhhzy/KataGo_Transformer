#!/usr/bin/env python3
"""Filter shuffled KataGo NPZ data down to full-size boards.

The source dataset is only ever opened for reading. Output is assembled in a
sibling staging directory, verified, and then atomically renamed to the final
destination. This makes it safe to point training at the destination as soon as
it appears.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


BINARY_INPUT_KEY = "binaryInputNCHWPacked"
REQUIRED_ROW_KEYS = frozenset(
    {
        BINARY_INPUT_KEY,
        "globalInputNC",
        "policyTargetsNCMove",
        "globalTargetsNC",
        "scoreDistrN",
        "valueTargetsNCHW",
    }
)
MANIFEST_FILENAME = "full_board_filter_manifest.json"
MANIFEST_VERSION = 1


class DatasetFormatError(ValueError):
    """Raised when an input or filtered dataset violates the NPZ contract."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def full_board_row_mask(binary_input_packed: np.ndarray, pos_len: int) -> np.ndarray:
    """Return the rows whose spatial mask (channel zero) is entirely one.

    ``np.unpackbits`` defaults to big-endian bit order, as does KataGo's data
    loader. The final byte may contain padding bits, which are deliberately
    ignored here.
    """

    if pos_len <= 0:
        raise ValueError(f"pos_len must be positive, got {pos_len}")
    if binary_input_packed.dtype != np.uint8:
        raise DatasetFormatError(
            f"{BINARY_INPUT_KEY} must have dtype uint8, got {binary_input_packed.dtype}"
        )
    if binary_input_packed.ndim != 3 or binary_input_packed.shape[1] < 1:
        raise DatasetFormatError(
            f"{BINARY_INPUT_KEY} must have shape [N,C,packed_area] with C >= 1, "
            f"got {binary_input_packed.shape}"
        )

    area = pos_len * pos_len
    required_bytes = (area + 7) // 8
    if binary_input_packed.shape[2] != required_bytes:
        raise DatasetFormatError(
            f"{BINARY_INPUT_KEY} packed area is {binary_input_packed.shape[2]} bytes, "
            f"expected {required_bytes} for {pos_len}x{pos_len}"
        )

    mask_bytes = binary_input_packed[:, 0, :]
    whole_bytes, remaining_bits = divmod(area, 8)
    if whole_bytes == 0:
        is_full = np.ones(mask_bytes.shape[0], dtype=np.bool_)
    else:
        is_full = np.all(mask_bytes[:, :whole_bytes] == np.uint8(0xFF), axis=1)

    if remaining_bits:
        # unpackbits uses the most significant bit first.
        required = np.uint8(((1 << remaining_bits) - 1) << (8 - remaining_bits))
        is_full &= (mask_bytes[:, whole_bytes] & required) == required
    return is_full


def _load_and_filter_npz(source_path: Path, pos_len: int) -> Tuple[Dict[str, np.ndarray], int, int]:
    with np.load(source_path, allow_pickle=False) as npz:
        keys = tuple(npz.files)
        missing = REQUIRED_ROW_KEYS.difference(keys)
        if missing:
            raise DatasetFormatError(
                f"{source_path} is missing required keys: {sorted(missing)}"
            )

        binary_input = npz[BINARY_INPUT_KEY]
        keep = full_board_row_mask(binary_input, pos_len)
        input_rows = int(binary_input.shape[0])
        output_rows = int(np.count_nonzero(keep))

        arrays: Dict[str, np.ndarray] = {}
        for key in keys:
            array = binary_input if key == BINARY_INPUT_KEY else npz[key]
            if array.ndim < 1 or array.shape[0] != input_rows:
                raise DatasetFormatError(
                    f"{source_path}:{key} is not row-aligned with {BINARY_INPUT_KEY}: "
                    f"shape {array.shape}, expected first dimension {input_rows}"
                )
            # A single selection is applied to every row-aligned key, including
            # optional and future fields not known to this script.
            arrays[key] = array[keep]

    return arrays, input_rows, output_rows


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as out:
            np.savez_compressed(out, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _updated_sidecar(
    sidecar_path: Path,
    input_rows: int,
    output_rows: int,
) -> Dict[str, Any]:
    with sidecar_path.open("r", encoding="utf-8") as inp:
        value = json.load(inp)
    if not isinstance(value, dict):
        raise DatasetFormatError(f"Sidecar {sidecar_path} must contain a JSON object")

    value["num_rows"] = output_rows
    if "num_batches" in value:
        old_num_batches = value["num_batches"]
        if isinstance(old_num_batches, bool) or not isinstance(old_num_batches, int):
            raise DatasetFormatError(
                f"Sidecar {sidecar_path} num_batches must be an integer"
            )
        if old_num_batches < 0:
            raise DatasetFormatError(
                f"Sidecar {sidecar_path} num_batches must be nonnegative"
            )
        if old_num_batches == 0:
            value["num_batches"] = 0
        else:
            # shuffle.py writes exactly num_batches * shuffle_batch_size rows to
            # each NPZ. Infer that historical batch size without requiring an
            # extra command-line argument.
            if input_rows % old_num_batches != 0:
                raise DatasetFormatError(
                    f"Cannot infer shuffle batch size for {sidecar_path}: "
                    f"NPZ has {input_rows} rows and sidecar has {old_num_batches} batches"
                )
            shuffle_batch_size = input_rows // old_num_batches
            if shuffle_batch_size <= 0:
                raise DatasetFormatError(
                    f"Invalid inferred shuffle batch size for {sidecar_path}"
                )
            value["num_batches"] = output_rows // shuffle_batch_size
    return value


def _verify_npz(path: Path, pos_len: int, expected_rows: int | None = None) -> int:
    with np.load(path, allow_pickle=False) as npz:
        missing = REQUIRED_ROW_KEYS.difference(npz.files)
        if missing:
            raise DatasetFormatError(f"{path} is missing required keys: {sorted(missing)}")
        packed = npz[BINARY_INPUT_KEY]
        rows = int(packed.shape[0])
        if expected_rows is not None and rows != expected_rows:
            raise DatasetFormatError(
                f"{path} has {rows} rows after writing, expected {expected_rows}"
            )
        for key in npz.files:
            array = packed if key == BINARY_INPUT_KEY else npz[key]
            if array.ndim < 1 or array.shape[0] != rows:
                raise DatasetFormatError(
                    f"{path}:{key} is not row-aligned after filtering: {array.shape}"
                )

        # Enforce the same packed-array contract as the runtime loader. The
        # independent unpackbits check below verifies the actual mask contents.
        full_board_row_mask(packed, pos_len)

        # Use unpackbits rather than the optimized byte predicate so verification
        # is independent of the filtering implementation.
        unpacked_mask = np.unpackbits(
            packed[:, 0, :], axis=1, count=pos_len * pos_len, bitorder="big"
        )
        if not bool(np.all(unpacked_mask)):
            bad_rows = np.flatnonzero(~np.all(unpacked_mask, axis=1))
            raise DatasetFormatError(
                f"{path} contains {bad_rows.size} non-full-board rows after filtering"
            )
    return rows


def _filter_one_file(job: Tuple[str, str, str, int, bool]) -> Dict[str, Any]:
    source_string, destination_string, relative_string, pos_len, require_sidecar = job
    source_path = Path(source_string)
    destination_path = Path(destination_string)
    relative_path = Path(relative_string)

    arrays, input_rows, output_rows = _load_and_filter_npz(source_path, pos_len)
    _atomic_write_npz(destination_path, arrays)
    del arrays
    _verify_npz(destination_path, pos_len, expected_rows=output_rows)

    source_sidecar = source_path.with_suffix(".json")
    destination_sidecar = destination_path.with_suffix(".json")
    if source_sidecar.is_file():
        sidecar = _updated_sidecar(source_sidecar, input_rows, output_rows)
        _atomic_write_json(destination_sidecar, sidecar)
    elif require_sidecar:
        raise DatasetFormatError(f"Training NPZ is missing sidecar: {source_sidecar}")

    return {
        "path": relative_path.as_posix(),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "removed_rows": input_rows - output_rows,
    }


def _discover_npz_files(source_root: Path) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    train_dir = source_root / "train"
    if not train_dir.is_dir():
        raise DatasetFormatError(f"Source dataset has no train directory: {train_dir}")
    for split in ("train", "val"):
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        if not split_dir.is_dir():
            raise DatasetFormatError(f"Expected directory, got: {split_dir}")
        files.extend((split, path) for path in split_dir.rglob("*.npz") if path.is_file())
    files.sort(key=lambda item: item[1].relative_to(source_root).as_posix())
    if not any(split == "train" for split, _ in files):
        raise DatasetFormatError(f"No training NPZ files found under {train_dir}")
    return files


def verify_filtered_dataset(dataset_root: os.PathLike[str] | str, pos_len: int = 15) -> Dict[str, Any]:
    """Verify the manifest, row alignment, sidecars, and full-board invariant."""

    root = Path(dataset_root).resolve()
    train_json = root / "train.json"
    if not train_json.is_file():
        raise DatasetFormatError(f"Filtered dataset has no train.json: {train_json}")
    with train_json.open("r", encoding="utf-8") as inp:
        train_info = json.load(inp)
    if not isinstance(train_info, dict) or "range" not in train_info:
        raise DatasetFormatError(f"Invalid train.json: {train_json}")

    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DatasetFormatError(f"Filtered dataset has no manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as inp:
        manifest = json.load(inp)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise DatasetFormatError(
            f"Unsupported manifest version in {manifest_path}: "
            f"{manifest.get('manifest_version')}"
        )
    if manifest.get("pos_len") != pos_len:
        raise DatasetFormatError(
            f"Manifest pos_len is {manifest.get('pos_len')}, expected {pos_len}"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise DatasetFormatError(f"Manifest files must be a list: {manifest_path}")
    expected_by_path: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DatasetFormatError(f"Invalid file entry in {manifest_path}: {entry!r}")
        relative = entry["path"]
        if relative in expected_by_path:
            raise DatasetFormatError(f"Duplicate manifest path: {relative}")
        expected_by_path[relative] = entry

    discovered = _discover_npz_files(root)
    discovered_paths = {
        path.relative_to(root).as_posix(): (split, path) for split, path in discovered
    }
    if set(discovered_paths) != set(expected_by_path):
        missing = sorted(set(expected_by_path) - set(discovered_paths))
        extra = sorted(set(discovered_paths) - set(expected_by_path))
        raise DatasetFormatError(
            f"Manifest/file mismatch, missing={missing}, extra={extra}"
        )

    total_rows = 0
    split_rows: Dict[str, int] = {"train": 0, "val": 0}
    for relative in sorted(discovered_paths):
        split, path = discovered_paths[relative]
        entry = expected_by_path[relative]
        expected_rows = entry.get("output_rows")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
            raise DatasetFormatError(f"Invalid output_rows for {relative}: {expected_rows!r}")
        rows = _verify_npz(path, pos_len, expected_rows=expected_rows)
        if split == "train":
            sidecar_path = path.with_suffix(".json")
            if not sidecar_path.is_file():
                raise DatasetFormatError(f"Training NPZ is missing sidecar: {sidecar_path}")
            with sidecar_path.open("r", encoding="utf-8") as inp:
                sidecar = json.load(inp)
            if not isinstance(sidecar, dict) or sidecar.get("num_rows") != rows:
                raise DatasetFormatError(
                    f"Sidecar row count does not match {path}: {sidecar!r}"
                )
        total_rows += rows
        split_rows[split] += rows

    if manifest.get("total_output_rows") != total_rows:
        raise DatasetFormatError(
            f"Manifest total_output_rows is {manifest.get('total_output_rows')}, "
            f"actual total is {total_rows}"
        )
    return {
        "num_files": len(discovered_paths),
        "total_output_rows": total_rows,
        "split_output_rows": split_rows,
    }


def filter_full_board_dataset(
    source_root: os.PathLike[str] | str,
    destination_root: os.PathLike[str] | str,
    *,
    pos_len: int = 15,
    workers: int = 1,
) -> Dict[str, Any]:
    """Filter a shuffled dataset and atomically publish the verified result."""

    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if _is_relative_to(destination, source):
        raise ValueError(
            f"Destination must be independent of the source tree: {destination}"
        )

    train_json = source / "train.json"
    if not train_json.is_file():
        raise DatasetFormatError(f"Source dataset has no train.json: {train_json}")
    with train_json.open("r", encoding="utf-8") as inp:
        train_info = json.load(inp)
    if not isinstance(train_info, dict) or "range" not in train_info:
        raise DatasetFormatError(f"Invalid source train.json: {train_json}")

    source_files = _discover_npz_files(source)
    # Preflight sidecars before creating any output.
    for split, source_path in source_files:
        if split == "train" and not source_path.with_suffix(".json").is_file():
            raise DatasetFormatError(
                f"Training NPZ is missing sidecar: {source_path.with_suffix('.json')}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()

    try:
        # copyfile copies contents but not source permissions, so a read-only
        # train.json cannot make staging cleanup fail on Windows.
        shutil.copyfile(train_json, staging / "train.json")
        val_json = source / "val.json"
        if val_json.is_file():
            shutil.copyfile(val_json, staging / "val.json")
        (staging / "train").mkdir()
        if (source / "val").is_dir():
            (staging / "val").mkdir()

        jobs: List[Tuple[str, str, str, int, bool]] = []
        for split, source_path in source_files:
            relative = source_path.relative_to(source)
            jobs.append(
                (
                    str(source_path),
                    str(staging / relative),
                    relative.as_posix(),
                    pos_len,
                    split == "train",
                )
            )

        if workers == 1:
            results = [_filter_one_file(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_filter_one_file, jobs))
        results.sort(key=lambda entry: entry["path"])

        split_stats: Dict[str, Dict[str, int]] = {
            "train": {"num_files": 0, "input_rows": 0, "output_rows": 0},
            "val": {"num_files": 0, "input_rows": 0, "output_rows": 0},
        }
        for entry in results:
            split = Path(entry["path"]).parts[0]
            stats = split_stats[split]
            stats["num_files"] += 1
            stats["input_rows"] += entry["input_rows"]
            stats["output_rows"] += entry["output_rows"]

        if split_stats["train"]["output_rows"] == 0:
            raise DatasetFormatError(
                "Filtering retained zero training rows; refusing to publish an "
                "unusable dataset"
            )

        total_input_rows = sum(entry["input_rows"] for entry in results)
        total_output_rows = sum(entry["output_rows"] for entry in results)
        manifest: Dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_root": str(source),
            "destination_root": str(destination),
            "pos_len": pos_len,
            "num_files": len(results),
            "total_input_rows": total_input_rows,
            "total_output_rows": total_output_rows,
            "total_removed_rows": total_input_rows - total_output_rows,
            "splits": split_stats,
            "files": results,
        }
        _atomic_write_json(staging / MANIFEST_FILENAME, manifest)

        verification = verify_filtered_dataset(staging, pos_len=pos_len)
        manifest["verification"] = verification
        _atomic_write_json(staging / MANIFEST_FILENAME, manifest)

        if destination.exists():
            raise FileExistsError(
                f"Destination appeared while filtering; refusing to replace it: {destination}"
            )
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", help="Shuffled dataset root to read")
    parser.add_argument("destination_root", nargs="?", help="New filtered dataset root")
    parser.add_argument("--pos-len", type=int, default=15, help="Board side length (default: 15)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel NPZ workers (default: 1)")
    parser.add_argument(
        "--verify-only",
        metavar="DATASET_ROOT",
        help="Verify an existing filtered dataset instead of filtering",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.verify_only is not None:
        if args.source_root is not None or args.destination_root is not None:
            raise SystemExit("Do not pass source/destination with --verify-only")
        result = verify_filtered_dataset(args.verify_only, pos_len=args.pos_len)
    else:
        if args.source_root is None or args.destination_root is None:
            raise SystemExit("source_root and destination_root are required")
        result = filter_full_board_dataset(
            args.source_root,
            args.destination_root,
            pos_len=args.pos_len,
            workers=args.workers,
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
