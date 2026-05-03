# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC refs recreation file.
#
# This driver always runs:
#   1. the recreation file exactly as-is
#   2. the same recreation file with the FREE pressure-profile patch
#   3. the comparison CSV generation
#   4. the comparison plot generation
#
#==============================================================================================================

import os
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "JAX_PLATFORM_NAME" in os.environ:
    del os.environ["JAX_PLATFORM_NAME"]

from desc import set_device
set_device("gpu")

import argparse

try:
    from .helper import (
        get_case_dir,
        normalize_case_name,
        run_file,
    )

    from .compare import (
        case_obj,
        plot_case,
    )

except ImportError:
    from helper import (
        get_case_dir,
        normalize_case_name,
        run_file,
    )

    from compare import (
        case_obj,
        plot_case,
    )










#==============================================================================================================
# Source File Helpers
#==============================================================================================================

def get_source_file_from_refs_case(
        refs,
        case,
    ):
    """
    Resolve the recreation source file from --refs and --case.

    Expected layout:
        research/lit_comp/refs/<refs>/<case>/<case>.py
    """

    case_name = normalize_case_name(
        name = case,
    )

    case_dir = get_case_dir(
        refs = refs,
        case = case_name,
    )

    source_file = case_dir / f"{case_name}.py"

    if not case_dir.exists():
        raise FileNotFoundError(
            f"Case directory does not exist: {case_dir}"
        )

    if not source_file.exists():
        raise FileNotFoundError(
            "Could not find recreation source file:\n"
            f"{source_file}\n\n"
            "Expected layout:\n"
            f"research/lit_comp/refs/{refs}/{case_name}/{case_name}.py"
        )

    return source_file, case_name










#==============================================================================================================
# Compare / Plot Helpers
#==============================================================================================================

def run_comparison_outputs(
        refs,
        case,
    ):
    """
    Run compare.py logic after the OG/FXD and FREE runs.
    """

    print("")
    print("================================================================================================================")
    print("Running comparison CSV generation")
    print("================================================================================================================")
    print("")

    rows, csv_path = case_obj(
        refs = refs,
        case = case,
    )

    print("")
    print("Finished case-objective comparison.")
    print(f"CSV written to: {csv_path}")

    print("")
    print("================================================================================================================")
    print("Running comparison plot generation")
    print("================================================================================================================")
    print("")

    paths = plot_case(
        refs = refs,
        case = case,
        plot_folder = "pressure",
    )

    if len(paths) == 0:
        print("No plots written.")

    for path in paths:
        print(f"Plot written to: {path}")

    return rows, csv_path, paths










#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description = "Run one DESC refs recreation case as-is and with the FREE variant.",
    )

    parser.add_argument(
        "--refs",
        type = str,
        required = True,
        help = "Paper directory name, e.g. dudt2024.",
    )

    parser.add_argument(
        "--case",
        type = str,
        required = True,
        help = "Case name, e.g. helical_qs.",
    )

    return parser.parse_args()










#==============================================================================================================
# Main
#==============================================================================================================

def main():
    """
    Command-line entry point.
    """

    args = parse_args()

    source_file, case_name = get_source_file_from_refs_case(
        ref = args.ref,
        case = args.case,
    )

    print("")
    print("================================================================================================================")
    print("DESC refs recreation driver")
    print("================================================================================================================")
    print("")
    print("Paper:")
    print(args.refs)
    print("")
    print("Case:")
    print(case_name)
    print("")
    print("Source file:")
    print(source_file)

    print("")
    print("================================================================================================================")
    print("Running unmodified recreation")
    print("================================================================================================================")
    print("")

    run_file(
        source_file = source_file,
        free = False,
    )

    print("")
    print("================================================================================================================")
    print("Running FREE constrained-pressure recreation")
    print("================================================================================================================")
    print("")

    free_output = run_file(
        source_file = source_file,
        free = True,
    )

    print("")
    print("Completed runs:")
    print("================================================================================================================")
    print("Unmodified source file ran as-is.")
    print(f"FREE constrained-pressure output: {free_output}")

    run_comparison_outputs(
        refs = args.refs,
        case = case_name,
    )










if __name__ == "__main__":
    main()