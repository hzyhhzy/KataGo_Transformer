#!/usr/bin/env python3
"""Compatibility entry point for the unified CPU-PTQ exporter.

New code should call :mod:`export_cpu_ptq` directly. This filename remains so
existing v206 commands keep working while sharing exactly one implementation.
"""

from export_cpu_ptq import main


if __name__ == "__main__":
    main()
