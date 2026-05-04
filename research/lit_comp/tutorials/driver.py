# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC tutorial file.
#
# Each tutorial file now directly runs and saves both:
#   1. FXD optimization with FixPressure
#   2. FREE optimization with FREE pressure objectives/constraints replacing FixPressure
#
# Usage:
#   cd research/lit_comp/tutorials
#   python3 driver.py --tutorial basic_qs
#
# Or from DESC root:
#   python3 research/lit_comp/tutorials/driver.py --tutorial basic_qs
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
    )

except ImportError:
    from helper import (
        get_source_file_from_tutorial,
        run_file,
    )

    from compare import (
        tutorial_obj,
    )










#==============================================================================================================
# Compare Helpers
#==============================================================================================================

def run_comparison_outputs(
        tutorial,
    ):
    """
    Run compare.py logic after the tutorial writes FXD and FREE files.
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

    return rows, csv_path










#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description = "Run one DESC tutorial case. Tutorial file handles both FXD and FREE outputs.",
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
    print("Running tutorial file")
    print("================================================================================================================")
    print("")

    output = run_file(
        source_file = source_file,
    )

    print("")
    print("Completed tutorial run:")
    print("================================================================================================================")
    print(f"Output: {output}")

    run_comparison_outputs(
        tutorial = tutorial_name,
    )










if __name__ == "__main__":
    main()