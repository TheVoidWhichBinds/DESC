# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC tutorial file.
#
# Each tutorial file handles its own optimization sweep and saves:
#   1. FXD optimization outputs
#   2. FREE optimization outputs
#   3. one numbered output folder per tolerance combination:
#
#        001/
#        002/
#        003/
#        ...
#
# Serial usage:
#   cd research/lit_comp/tutorials
#   python3 driver.py --tutorial basic_qs
#
# Single sweep-case usage:
#   cd research/lit_comp/tutorials
#   python3 driver.py --tutorial basic_qs --sweep-index 1
#
# SLURM array usage:
#   sbatch basic_qs_gpu_array.sbatch
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
# Environment Helpers
#==============================================================================================================
def resolve_sweep_index(
        args,
    ):
    """
    Resolve the requested tolerance-sweep index.

    Priority:
        1. explicit --sweep-index
        2. SLURM_ARRAY_TASK_ID
        3. None, meaning run every sweep case serially
    """

    if args.sweep_index is not None:
        return int(args.sweep_index)

    slurm_array_task_id = os.environ.get(
        "SLURM_ARRAY_TASK_ID",
        None,
    )

    if slurm_array_task_id is not None:
        return int(slurm_array_task_id)

    return None





def configure_parallel_environment(
        args,
    ):
    """
    Export environment variables consumed by the tutorial files.
    """

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

    If case is given, compare.py generates only that numbered folder CSV.
    This is the safe mode for SLURM array jobs, because each array task only
    compares the folder it just finished.
    """

    print("")
    print("================================================================================================================")

    if case is None:
        print("Running tutorial comparison CSV generation")

    else:
        print(f"Running tutorial comparison CSV generation for case {case}")

    print("================================================================================================================")
    print("")

    rows, csv_path = tutorial_obj(
        tutorial = tutorial,
        case = case,
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
        description = "Run one DESC tutorial tolerance sweep. Tutorial file handles FXD and FREE outputs.",
    )

    parser.add_argument(
        "--tutorial",
        type = str,
        required = True,
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
    )

    parser.add_argument(
        "--sweep-index",
        type = int,
        default = None,
        help = "Optional one-based tolerance-sweep index. If omitted, SLURM_ARRAY_TASK_ID is used when present.",
    )

    parser.add_argument(
        "--case-start-index",
        type = int,
        default = None,
        help = "Optional first numbered output folder for sweep-index 1. Mostly used by the GPU-array submit wrapper.",
    )

    parser.add_argument(
        "--skip-compare",
        action = "store_true",
        help = "Skip case_obj.csv generation after optimization.",
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

    sweep_index = configure_parallel_environment(
        args = args,
    )

    source_file, tutorial_name = get_source_file_from_tutorial(
        tutorial = args.tutorial,
    )

    print("")
    print("================================================================================================================")
    print("DESC tutorial tolerance-sweep driver")
    print("================================================================================================================")
    print("")
    print("Tutorial:")
    print(tutorial_name)
    print("")
    print("Source file:")
    print(source_file)

    if sweep_index is not None:
        print("")
        print("Parallel sweep mode:")
        print(f"sweep_index: {sweep_index}")
        print(f"case_start_index: {os.environ.get('DESC_CASE_START_INDEX', 'atomic-next-folder')}")

    print("")
    print("================================================================================================================")
    print("Running tutorial file")
    print("================================================================================================================")
    print("")

    output = run_file(
        source_file = source_file,
    )

    print("")
    print("Completed tutorial sweep:")
    print("================================================================================================================")
    print(f"Output: {output}")

    if args.skip_compare:
        print("")
        print("Skipping comparison CSV generation.")
        print("")
        return

    for case_label in output.keys():
        comparison_case = case_label if str(case_label).isdigit() else None

        run_comparison_outputs(
            tutorial = tutorial_name,
            case = comparison_case,
        )




if __name__ == "__main__":
    main()
