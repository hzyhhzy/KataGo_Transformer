"""Compatibility helpers for reading array metadata from ``.npz`` files."""

import ast
import struct

import numpy as np


_MAX_HEADER_SIZE = 10_000
_EXPECTED_HEADER_KEYS = {"descr", "fortran_order", "shape"}


def _read_exact(fileobj, size, description):
    chunks = []
    remaining = size
    while remaining:
        chunk = fileobj.read(remaining)
        if not chunk:
            raise ValueError(
                "EOF while reading %s: expected %d more bytes"
                % (description, remaining)
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_array_header_3_0(fileobj):
    """Read an NPY 3.0 header without depending on a NumPy private API."""
    header_length = struct.unpack(
        "<I", _read_exact(fileobj, 4, "array header length")
    )[0]
    if header_length > _MAX_HEADER_SIZE:
        raise ValueError(
            "Array header is too large (%d bytes; maximum is %d)"
            % (header_length, _MAX_HEADER_SIZE)
        )

    header_text = _read_exact(
        fileobj, header_length, "array header"
    ).decode("utf-8")
    try:
        header = ast.literal_eval(header_text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Cannot parse NPY 3.0 array header") from exc

    if not isinstance(header, dict) or set(header) != _EXPECTED_HEADER_KEYS:
        raise ValueError("Array header does not contain the expected keys")

    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
    ):
        raise ValueError("Invalid array shape in NPY header: %r" % (shape,))
    if not isinstance(header["fortran_order"], bool):
        raise ValueError("Invalid fortran_order value in NPY header")

    try:
        dtype = np.lib.format.descr_to_dtype(header["descr"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid dtype descriptor in NPY header") from exc
    return shape, header["fortran_order"], dtype


def read_numpy_array_header(fileobj, version):
    """Read an NPY header using stable NumPy APIs for every supported version."""
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fileobj)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(fileobj)
    if version == (3, 0):
        return _read_array_header_3_0(fileobj)
    raise ValueError("Unsupported NPY file format version: %r" % (version,))
