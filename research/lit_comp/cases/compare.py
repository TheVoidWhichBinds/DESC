# compare.py
#==============================================================================================================
#
# Compare case objective outputs between FXD and FREE pressure optimizations.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 compare.py --case ARIES-CS
#   python3 compare.py --case ARIES-CS --obj qs3
#   python3 compare.py --case ARIES-CS --obj balloon
#   python3 compare.py --case ARIES-CS --obj force
#
#==============================================================================================================

import argparse

from desc.objectives import ForceBalance

try:
    from .driver import (
        build_balloon_objective,
        build_force_balance_grid,
        build_qs3_objective,
    )

    from .helper import (
        compare_objective_set,
        find_h5_files,
        get_existing_objective_dirs,
        get_existing_run_dirs,
        normalize_case_name,
        objective_spec,
        write_table_csv,
    )

except ImportError:
    from driver import (
        build_balloon_objective,
        build_force_balance_grid,
        build_qs3_objective,
    )

    from helper import (
        compare_objective_set,
        find_h5_files,
        get_existing_objective_dirs,
        get_existing_run_dirs,
        normalize_case_name,
        objective_spec,
        write_table_csv,
    )










#==============================================================================================================
# Objective Builders
#==============================================================================================================

def build_force_balance_objective(
        eq,
    ):
    """
    Build the ForceBalance objective for comparison.
    """

    return ForceBalance(
        eq = eq,
        grid = build_force_balance_grid(
            eq = eq,
        ),
        normalize = True,
    )





def qs3_objectives(
        eq,
    ):
    """
    Return comparison rows for the qs3 folder.
    """

    return [
        objective_spec(
            name = "ForceBalance",
            objective = build_force_balance_objective(
                eq = eq,
            ),
        ),
        objective_spec(
            name = "QuasisymmetryTripleProduct",
            objective = build_qs3_objective(
                eq = eq,
            ),
        ),
    ]





def balloon_objectives(
        eq,
    ):
    """
    Return comparison rows for the balloon folder.
    """

    return [
        objective_spec(
            name = "ForceBalance",
            objective = build_force_balance_objective(
                eq = eq,
            ),
        ),
        objective_spec(
            name = "BallooningStability",
            objective = build_balloon_objective(
                eq = eq,
            ),
        ),
    ]





def force_objectives(
        eq,
    ):
    """
    Return comparison rows for the force folder.
    """

    return [
        objective_spec(
            name = "ForceBalance",
            objective = build_force_balance_objective(
                eq = eq,
            ),
        ),
    ]










#==============================================================================================================
# Objective Registry
#==============================================================================================================

OBJECTIVE_BUILDERS = {
    "qs3": qs3_objectives,
    "balloon": balloon_objectives,
    "force": force_objectives,
}





def get_objective_builder(
        obj,
    ):
    """
    Return the comparison objective builder for an objective folder.
    """

    objective_builder = OBJECTIVE_BUILDERS.get(
        obj,
        None,
    )

    if objective_builder is None:
        raise ValueError(
            f"No objective registry entry found for obj = {obj}."
        )

    return objective_builder








#==============================================================================================================
# Case Comparison
#==============================================================================================================

def make_case_csv_path(
        case,
        obj,
        output_dir,
    ):
    """
    Build the comparison CSV path for one output folder.
    """

    case_name = normalize_case_name(
        case = case,
    )

    output_dir = output_dir.resolve()

    if output_dir.name.isdigit():
        return output_dir / f"{case_name}_{obj}_{output_dir.name}_case_obj.csv"

    return output_dir / f"{case_name}_{obj}_case_obj.csv"





def get_comparison_output_dirs(
        objective_dir,
    ):
    """
    Return numbered run folders when present, otherwise the objective folder itself.
    """

    run_dirs = get_existing_run_dirs(
        objective_dir = objective_dir,
    )

    if len(run_dirs) > 0:
        return run_dirs

    return (
        objective_dir,
    )





def case_obj(
        case,
        obj = None,
    ):
    """
    Compare objective values between FXD and FREE files.

    If obj is None, every configured objective folder is compared.
    """

    objective_dirs = get_existing_objective_dirs(
        case = case,
        obj = obj,
    )

    rows_by_case = {}
    csv_paths = {}

    for objective_dir in objective_dirs:
        objective_label = objective_dir.name

        objective_builder = get_objective_builder(
            obj = objective_label,
        )

        output_dirs = get_comparison_output_dirs(
            objective_dir = objective_dir,
        )

        for output_dir in output_dirs:
            files = find_h5_files(
                case_dir = output_dir,
            )

            output_label = objective_label

            if output_dir.name.isdigit():
                output_label = f"{objective_label}/{output_dir.name}"

            if len(files) == 0:
                print("")
                print(f"No *_FXD.h5 or *_FREE.h5 files were found in: {output_dir}")
                print("")
                continue

            rows = compare_objective_set(
                files = files,
                objective_getter = objective_builder,
            )

            csv_path = make_case_csv_path(
                case = case,
                obj = objective_label,
                output_dir = output_dir,
            )

            write_table_csv(
                rows = rows,
                path = csv_path,
            )

            rows_by_case[output_label] = rows
            csv_paths[output_label] = csv_path

    return rows_by_case, csv_paths








#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        required = True,
        help = "Case name, e.g. ARIES-CS, HELIOTRON, W7-X.",
    )

    parser.add_argument(
        "--obj",
        default = None,
        choices = [
            "qs3",
            "balloon",
            "force",
        ],
        help = "Optional objective folder to compare. If omitted, qs3, balloon, and force are compared.",
    )

    return parser.parse_args()





def main():
    """
    Runs case-specific objective comparison.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Comparing case objectives for case = {args.case}")

    if args.obj is not None:
        print(f"Objective folder = {args.obj}")

    print("=" * 120 + "\n")

    rows_by_case, csv_paths = case_obj(
        case = args.case,
        obj = args.obj,
    )

    print("")
    print("Finished case-objective comparison.")
    print("CSV files written to:")
    print("")

    for case_label, csv_path in csv_paths.items():
        print(f"{case_label}: {csv_path}")

    print("")





if __name__ == "__main__":
    main()
