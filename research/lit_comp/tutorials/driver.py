# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC tutorial file.
#
# This driver always runs:
#   1. the tutorial file exactly as-is
#   2. the same tutorial file with the FREE pressure-profile patch
#   3. the tutorial-specific comparison CSV generation
#   4. the comparison plot generation
#
# Usage:
#   cd research/lit_comp/tutorials
#   python3 driver.py --tutorial balloon
#
# Or from DESC root:
#   python3 research/lit_comp/tutorials/driver.py --tutorial balloon
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
        get_source_file_from_tutorial,
        run_file,
    )

    from .compare import (
        tutorial_obj,
        plot_tutorial,
    )

except ImportError:
    from helper import (
        get_source_file_from_tutorial,
        run_file,
    )

    from compare import (
        tutorial_obj,
        plot_tutorial,
    )










#==============================================================================================================
# Compare / Plot Helpers
#==============================================================================================================

def run_comparison_outputs(
        tutorial,
    ):
    """
    Run compare.py logic after the OG/CHECK/FXD and FREE runs.
    """

    print("")
    print("================================================================================================================")
    print("Running tutorial comparison CSV generation")
    print("================================================================================================================")
    print("")

    rows, csv_path = tutorial_obj(
        tutorial = tutorial,
    )

    print("")
    print("Finished tutorial-objective comparison.")
    print(f"CSV written to: {csv_path}")

    print("")
    print("================================================================================================================")
    print("Running comparison plot generation")
    print("================================================================================================================")
    print("")

    paths = plot_tutorial(
        tutorial = tutorial,
        plot_folder = "profiles",
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
        description = "Run one DESC tutorial case as-is and with the FREE variant.",
    )

    parser.add_argument(
        "--tutorial",
        type = str,
        required = True,
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
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

    source_file, tutorial_name = get_source_file_from_tutorial(
        tutorial = args.tutorial,
    )

    print("")
    print("================================================================================================================")
    print("DESC tutorial driver")
    print("================================================================================================================")
    print("")
    print("Tutorial:")
    print(tutorial_name)
    print("")
    print("Source file:")
    print(source_file)

    print("")
    print("================================================================================================================")
    print("Running unmodified tutorial")
    print("================================================================================================================")
    print("")

    run_file(
        source_file = source_file,
        free = False,
    )

    print("")
    print("================================================================================================================")
    print("Running FREE constrained-pressure tutorial")
    print("================================================================================================================")
    print("")

    free_output = run_file(
        source_file = source_file,
        free = True,
    )

    print("")
    print("Completed runs:")
    print("================================================================================================================")
    print("Unmodified tutorial source file ran as-is.")
    print(f"FREE constrained-pressure output: {free_output}")

    run_comparison_outputs(
        tutorial = tutorial_name,
    )










if __name__ == "__main__":
    main()