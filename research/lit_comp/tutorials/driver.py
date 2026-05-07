# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC tutorial file on CPU.
#
# Each tutorial file handles its own tolerance sweep and saves FXD/FREE outputs into
# one numbered output folder per optimization and tolerance combination.
#
# Examples:
#   cd research/lit_comp/tutorials
#   python3 driver.py --tutorial basic_qs
#   python3 driver.py --tutorial basic_qs --sweep-index 1
#
# Or from DESC root:
#   python3 research/lit_comp/tutorials/driver.py --tutorial basic_qs
#
#==============================================================================================================

import os
os.environ["JAX_PLATFORMS"] = "cpu"

if "JAX_PLATFORM_NAME" in os.environ:
    del os.environ["JAX_PLATFORM_NAME"]

from desc import set_device
set_device("cpu")

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
# Environment Helpers
#==============================================================================================================
def resolve_sweep_index(
        args,
    ):
    """
    Resolve the requested tolerance-sweep index.

    Priority:
        1. explicit --sweep-index
        2. existing DESC_SWEEP_INDEX
        3. None, meaning run every sweep case serially
    """

    if args.sweep_index is not None:
        return int(args.sweep_index)

    requested_sweep_index = os.environ.get(
        "DESC_SWEEP_INDEX",
        None,
    )

    if requested_sweep_index is not None:
        return int(requested_sweep_index)

    return None





def configure_cpu_environment(
        args,
    ):
    """
    Export environment variables consumed by the tutorial files.
    """

    os.environ["JAX_PLATFORMS"] = "cpu"

    sweep_index = resolve_sweep_index(
        args = args,
    )

    if sweep_index is not None:
        os.environ["DESC_SWEEP_INDEX"] = str(sweep_index)

    elif "DESC_SWEEP_INDEX" in os.environ:
        del os.environ["DESC_SWEEP_INDEX"]

    if args.case_start_index is not None:
        os.environ["DESC_CASE_START_INDEX"] = str(args.case_start_index)

    return sweep_index










#==============================================================================================================
# Compare Helpers
#==============================================================================================================
def run_comparison_outputs(
        tutorial,
        case = None,
    ):
    """
    Run compare.py logic after the tutorial writes FXD and FREE files.
    """

    print("")
    print("================================================================================================================")

    if case is None:
        print("Running tutorial comparison CSV generation")

    else:
        print(f"Running tutorial comparison CSV generation for case {case}")

    print("================================================================================================================")
    print("")

    rows, csv_paths = tutorial_obj(
        tutorial = tutorial,
        case = case,
    )

    print("")
    print("Finished tutorial-objective comparison.")
    print("CSV files written to:")
    print("")

    for case_label, csv_path in csv_paths.items():
        print(f"{case_label}: {csv_path}")

    print("")

    return rows, csv_paths










#==============================================================================================================
# CLI
#==============================================================================================================
def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tutorial",
        required = True,
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
    )

    parser.add_argument(
        "--sweep-index",
        type = int,
        default = None,
        help = "Optional one-based tolerance-sweep index. If omitted, every sweep case runs serially.",
    )

    parser.add_argument(
        "--case-start-index",
        type = int,
        default = None,
        help = "Optional first numbered output folder index for deterministic case labels.",
    )

    parser.add_argument(
        "--skip-compare",
        action = "store_true",
        help = "Skip comparison CSV generation after the tutorial run.",
    )

    return parser.parse_args()





def main():
    """
    Run one tutorial file and generate comparison CSV files.
    """

    args = parse_args()

    sweep_index = configure_cpu_environment(
        args = args,
    )

    source_file, tutorial_name = get_source_file_from_tutorial(
        tutorial = args.tutorial,
    )

    print("")
    print("================================================================================================================")
    print(f"Running tutorial = {tutorial_name}")
    print(f"Source file: {source_file}")

    if sweep_index is None:
        print("Sweep mode: all tolerance cases serially")

    else:
        print(f"Sweep mode: single tolerance case, sweep index = {sweep_index}")

    print("Backend: CPU")
    print("================================================================================================================")
    print("")

    previous_skip_compare = os.environ.get(
        "DESC_SKIP_TUTORIAL_COMPARE",
        None,
    )

    os.environ["DESC_SKIP_TUTORIAL_COMPARE"] = "1"

    try:
        run_file(
            source_file,
        )

    finally:
        if previous_skip_compare is None:
            del os.environ["DESC_SKIP_TUTORIAL_COMPARE"]

        else:
            os.environ["DESC_SKIP_TUTORIAL_COMPARE"] = previous_skip_compare

    if not args.skip_compare:
        run_comparison_outputs(
            tutorial = tutorial_name,
        )





if __name__ == "__main__":
    main()
