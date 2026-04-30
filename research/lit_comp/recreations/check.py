# check.py
"""Compare original and recreated DESC output files for any paper recreation."""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from argparse import ArgumentParser
from pathlib import Path
import csv
import math
import sys

import numpy as np

from desc.io import load










#========================================================================================================================================
# SETTINGS
#========================================================================================================================================
CHECK_SUFFIX = "_CHECK"
ORIGINAL_SUFFIX = "_output"

SUMMARY_KEYS = [
    "value",
    "max_abs",
    "mean_abs",
    "rms",
    "min",
    "max",
    "size",
]

QUANTITY_CANDIDATES = [
    #-----------------------------------
    # Main equilibrium / objective-like quantities
    #-----------------------------------
    "F_rho",
    "F_theta",
    "F_zeta",
    "|F|",
    "force",
    "p",
    "iota",
    "current",
    "lambda",

    #-----------------------------------
    # Common natural DESC scalar outputs
    #-----------------------------------
    "R0",
    "a",
    "A",
    "V",
    "S",
    "R0/a",
    "<R>",
    "<Z>",
    "<B>",
    "<B^2>",
    "B0",
    "W",
    "W_p",
    "W_B",
    "beta",
    "<beta>",
    "beta_a",
    "beta_p",
    "beta_t",
    "shear",
    "well",
    "magnetic well",

    #-----------------------------------
    # Stability / optimization metrics used across papers
    #-----------------------------------
    "D_Mercier",
    "Mercier",
    "ideal ballooning lambda",
    "ideal ballooning",
    "QS error",
    "quasisymmetry",
    "mirror ratio",
    "elongation",
    "curvature",
    "curvature_k1_rho",
    "curvature_k2_rho",
]










#========================================================================================================================================
# PATH HELPERS
#========================================================================================================================================
def normalize_case_name(name):
    """Normalize a passed filename/case stem by removing .h5, _CHECK, and _output."""

    path = Path(name)
    stem = path.stem

    if stem.endswith(CHECK_SUFFIX):
        stem = stem[: -len(CHECK_SUFFIX)]

    if stem.endswith(ORIGINAL_SUFFIX):
        stem = stem[: -len(ORIGINAL_SUFFIX)]

    return stem










def get_recreations_dir():
    """Return the recreations directory containing this script."""

    return Path(__file__).resolve().parent










def get_output_dir(paper):
    """Return the output directory for a paper recreation."""

    return get_recreations_dir() / paper / "output"










def find_matching_files(output_dir, file_name):
    """Find the original and CHECK files matching the requested case."""

    case_name = normalize_case_name(file_name)

    h5_files = sorted(output_dir.glob("*.h5"))

    original_matches = []
    check_matches = []

    for path in h5_files:
        normalized = normalize_case_name(path.name)

        if normalized != case_name:
            continue

        if path.stem.endswith(CHECK_SUFFIX):
            check_matches.append(path)
        else:
            original_matches.append(path)

    if len(original_matches) == 0:
        raise FileNotFoundError(
            "Could not find original file for case '{}' in {}".format(
                case_name,
                output_dir,
            )
        )

    if len(check_matches) == 0:
        raise FileNotFoundError(
            "Could not find CHECK file for case '{}' in {}".format(
                case_name,
                output_dir,
            )
        )

    if len(original_matches) > 1:
        raise RuntimeError(
            "Found multiple original files for case '{}':\n{}".format(
                case_name,
                "\n".join(str(path) for path in original_matches),
            )
        )

    if len(check_matches) > 1:
        raise RuntimeError(
            "Found multiple CHECK files for case '{}':\n{}".format(
                case_name,
                "\n".join(str(path) for path in check_matches),
            )
        )

    return original_matches[0], check_matches[0], case_name










#========================================================================================================================================
# DESC LOAD HELPERS
#========================================================================================================================================
def load_latest_equilibrium(path):
    """Load a DESC output file and return the final equilibrium."""

    obj = load(str(path))

    if hasattr(obj, "equilibria"):
        return obj.equilibria[-1]

    if hasattr(obj, "__getitem__") and not hasattr(obj, "compute"):
        return obj[-1]

    if isinstance(obj, list) or isinstance(obj, tuple):
        return obj[-1]

    return obj










def safe_compute(eq, quantity):
    """Compute a DESC quantity if available."""

    try:
        data = eq.compute(quantity)
    except Exception:
        return None

    if isinstance(data, dict):
        if quantity in data:
            return data[quantity]

        if len(data) == 1:
            return next(iter(data.values()))

        return data

    return data










#========================================================================================================================================
# NUMERIC HELPERS
#========================================================================================================================================
def to_numeric_array(value):
    """Convert a DESC compute output to a finite numeric numpy array."""

    if isinstance(value, dict):
        return None

    try:
        array = np.asarray(value)
    except Exception:
        return None

    if array.dtype == object:
        return None

    try:
        array = array.astype(float)
    except Exception:
        return None

    array = array[np.isfinite(array)]

    if array.size == 0:
        return None

    return array










def summarize_value(value):
    """Summarize scalar or array-valued DESC output."""

    array = to_numeric_array(value)

    if array is None:
        return None

    flat = array.reshape(-1)

    if flat.size == 1:
        scalar = float(flat[0])

        return {
            "value": scalar,
            "max_abs": abs(scalar),
            "mean_abs": abs(scalar),
            "rms": abs(scalar),
            "min": scalar,
            "max": scalar,
            "size": 1,
        }

    return {
        "value": np.nan,
        "max_abs": float(np.max(np.abs(flat))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "size": int(flat.size),
    }










def relative_difference(original, check):
    """Return relative difference using the original value as reference."""

    if not np.isfinite(original) or not np.isfinite(check):
        return np.nan

    scale = max(abs(original), 1.0e-30)

    return abs(check - original) / scale










def format_float(value):
    """Format floats for terminal output."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        if not np.isfinite(value):
            return "nan"

        return "{:.6e}".format(value)
    except Exception:
        return str(value)










#========================================================================================================================================
# COMPARISON
#========================================================================================================================================
def compare_equilibria(eq_original, eq_check):
    """Compare all supported DESC quantities between original and CHECK equilibria."""

    rows = []

    for quantity in QUANTITY_CANDIDATES:
        original_value = safe_compute(eq_original, quantity)
        check_value = safe_compute(eq_check, quantity)

        if original_value is None and check_value is None:
            continue

        original_summary = summarize_value(original_value)
        check_summary = summarize_value(check_value)

        if original_summary is None or check_summary is None:
            rows.append(
                {
                    "quantity": quantity,
                    "statistic": "status",
                    "original": "computed" if original_value is not None else "missing",
                    "check": "computed" if check_value is not None else "missing",
                    "absolute_difference": "",
                    "relative_difference": "",
                }
            )

            continue

        for key in SUMMARY_KEYS:
            original_stat = original_summary[key]
            check_stat = check_summary[key]

            if isinstance(original_stat, int) or isinstance(check_stat, int):
                absolute = check_stat - original_stat
                relative = 0.0 if absolute == 0 else np.nan
            else:
                absolute = abs(check_stat - original_stat)
                relative = relative_difference(original_stat, check_stat)

            rows.append(
                {
                    "quantity": quantity,
                    "statistic": key,
                    "original": original_stat,
                    "check": check_stat,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
            )

    return rows










def print_rows(rows):
    """Print comparison rows to terminal."""

    if len(rows) == 0:
        print("")
        print("No comparable DESC quantities were found.")
        return

    print("")
    print("================================================================")
    print("Comparison")
    print("================================================================")
    print(
        "{:<28} {:<14} {:>16} {:>16} {:>16} {:>16}".format(
            "quantity",
            "statistic",
            "original",
            "CHECK",
            "abs diff",
            "rel diff",
        )
    )
    print("-" * 112)

    for row in rows:
        print(
            "{:<28} {:<14} {:>16} {:>16} {:>16} {:>16}".format(
                row["quantity"][:28],
                row["statistic"][:14],
                format_float(row["original"]),
                format_float(row["check"]),
                format_float(row["absolute_difference"]),
                format_float(row["relative_difference"]),
            )
        )










def save_rows(rows, path):
    """Save comparison rows to CSV."""

    with open(path, "w", newline = "") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames = [
                "quantity",
                "statistic",
                "original",
                "check",
                "absolute_difference",
                "relative_difference",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)










#========================================================================================================================================
# CLI
#========================================================================================================================================
def parse_args():
    """Parse command-line arguments."""

    parser = ArgumentParser(
        description = "Compare original and CHECK DESC output files for a paper recreation.",
    )

    parser.add_argument(
        "--paper",
        required = True,
        help = "Paper recreation directory name, e.g. panici2023.",
    )

    parser.add_argument(
        "--file",
        required = True,
        help = "Case name or HDF5 file name to compare.",
    )

    parser.add_argument(
        "--save",
        action = "store_true",
        help = "Save comparison CSV in the paper output directory.",
    )

    return parser.parse_args()










def main():
    """Run comparison."""

    args = parse_args()

    output_dir = get_output_dir(args.paper)

    if not output_dir.exists():
        raise FileNotFoundError(
            "Output directory does not exist: {}".format(output_dir)
        )

    original_file, check_file, case_name = find_matching_files(
        output_dir = output_dir,
        file_name = args.file,
    )

    print("")
    print("================================================================")
    print("DESC recreation check")
    print("================================================================")
    print("Paper:")
    print(args.paper)
    print("")
    print("Case:")
    print(case_name)
    print("")
    print("Original:")
    print(original_file)
    print("")
    print("CHECK:")
    print(check_file)

    eq_original = load_latest_equilibrium(original_file)
    eq_check = load_latest_equilibrium(check_file)

    rows = compare_equilibria(
        eq_original = eq_original,
        eq_check = eq_check,
    )

    print_rows(rows)

    if args.save:
        csv_path = output_dir / "{}_comparison.csv".format(case_name)

        save_rows(
            rows = rows,
            path = csv_path,
        )

        print("")
        print("Saved comparison CSV:")
        print(csv_path)










#========================================================================================================================================
# RUN
#========================================================================================================================================
if __name__ == "__main__":
    main()