"""CLI entry point for HDF5 → .mat conversion.

Usage:
    python -m slac_measurements.wires.analysis <h5_path> [mat_path]

Converts a wire scan HDF5 file to a MATLAB .mat file compatible with
wirescan_gui and emittance_gui.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="slac_measurements.wires.analysis",
        description="Convert wire scan HDF5 to MATLAB .mat format",
    )
    parser.add_argument(
        "h5_path",
        help="Path to the wire scan HDF5 file",
    )
    parser.add_argument(
        "mat_path",
        nargs="?",
        default=None,
        help="Output .mat file path (default: same name with .mat extension)",
    )
    args = parser.parse_args(argv)

    from slac_measurements.wires.analysis.results import load_from_h5
    from slac_measurements.wires.analysis import mat_export
    from slac_measurements.wires.analysis.mat_export import analysis_result_to_mat

    h5_path = pathlib.Path(args.h5_path)
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found", file=sys.stderr)
        sys.exit(1)

    mat_path = args.mat_path or str(h5_path.with_suffix(".mat"))

    result = load_from_h5(str(h5_path))

    try:
        saved = analysis_result_to_mat(result, mat_path)
    except Exception:
        warnings.warn(
            "Name lookup from slac_db failed; using MAD names in .mat file.",
            stacklevel=2,
        )
        _orig_build = mat_export._build_name_map
        mat_export._build_name_map = lambda names: {}
        try:
            saved = analysis_result_to_mat(result, mat_path, name_map={})
        finally:
            mat_export._build_name_map = _orig_build

    print(f"MAT_PATH:{saved}")


if __name__ == "__main__":
    main()
