# compare.py
#==============================================================================================================
#
# Compare tutorial-specific objective outputs between FXD, and FREE files.
#
# Usage:
#   python3 research/lit_comp/tutorials/compare.py --tutorial basic_qs
#   python3 research/lit_comp/tutorials/compare.py --tutorial balloon
#   python3 research/lit_comp/tutorials/compare.py --tutorial basic_qs --case 001
#
#==============================================================================================================

import argparse
import numpy as np
from desc.grid import LinearGrid
from helper import (
    compare_objective_set,
    find_h5_files,
    get_output_case_dirs,
    get_tutorial_dir,
    write_table_csv,
)










#==============================================================================================================
# Tutorial Objective Helpers
#==============================================================================================================

def objective_spec(
        name,
        objective,
        thing = None,
    ):
    """
    Build one objective comparison spec.
    """

    spec = {
        "name": name,
        "objective": objective,
    }

    if thing is not None:
        spec["thing"] = thing

    return spec





def basic_qs_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the basic_qs tutorial.
    """

    from desc.objectives import (
        ForceBalance,
        QuasisymmetryBoozer,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    boozer_grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = False,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "QuasisymmetryBoozer",
            objective = QuasisymmetryBoozer(
                eq = eq,
                helicity = (1, eq.NFP),
                M_booz = 2 * eq.M,
                N_booz = 2 * eq.N,
                grid = boozer_grid,
            ),
        ),
    ]





def adv_qs_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the adv_qs tutorial.
    """

    from desc.objectives import (
        ForceBalance,
        QuasisymmetryBoozer,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    boozer_grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = False,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "QuasisymmetryBoozer",
            objective = QuasisymmetryBoozer(
                eq = eq,
                helicity = (1, eq.NFP),
                M_booz = 2 * eq.M,
                N_booz = 2 * eq.N,
                grid = boozer_grid,
            ),
        ),
    ]





def balloon_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the ballooning tutorial.

    FixIota, FixPressure, FixPsi, FixBoundaryR, and FixBoundaryZ are intentionally
    omitted because they are linear constraints and should not be compared.
    """

    from desc.objectives import (
        ForceBalance,
        AspectRatio,
        BallooningStability,
        GenericObjective,
    )

    alpha = np.linspace(
        0,
        np.pi,
        8,
        endpoint = False,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    curvature_grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        rho = np.array(
            [
                1.0,
            ]
        ),
        NFP = eq.NFP,
        sym = eq.sym,
        axis = False,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "BallooningStability",
            objective = BallooningStability(
                eq = eq,
                rho = np.array(
                    [
                        0.8,
                    ]
                ),
                alpha = alpha,
                nturns = 3,
                nzetaperturn = 200,
                weight = 2,
            ),
        ),
        objective_spec(
            name = "AspectRatio",
            objective = AspectRatio(
                eq = eq,
                bounds = (
                    8,
                    11,
                ),
                weight = 1,
            ),
        ),
        objective_spec(
            name = "GenericObjective curvature_k2_rho",
            objective = GenericObjective(
                f = "curvature_k2_rho",
                thing = eq,
                grid = curvature_grid,
                bounds = (
                    -np.inf,
                    0,
                ),
                weight = 2,
            ),
        ),
    ]










def neoclassical_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the neoclassical tutorial.
    """

    from desc.objectives import (
        ForceBalance,
        EffectiveRipple,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    neoclassical_grid = LinearGrid(
        rho = np.linspace(0.1, 1.0, 10),
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "EffectiveRipple",
            objective = EffectiveRipple(
                eq = eq,
                grid = neoclassical_grid,
            ),
        ),
    ]










#==============================================================================================================
# Tutorial Objective Registry
#==============================================================================================================

TUTORIAL_OBJECTIVES = {
    "basic_qs": basic_qs_objectives,
    "adv_qs": adv_qs_objectives,
    "balloon": balloon_objectives,
    "neoclassical": neoclassical_objectives,
}










#==============================================================================================================
# Multiple-Optimization File Helpers
#==============================================================================================================

def flatten_h5_file_map(
        files,
    ):
    """
    Flatten the h5 file map.

    New helper.find_h5_files format:

        {
            "C": {
                "FXD": Path(...),
                "FREE": Path(...),
            },
            "T": {
                "FXD": Path(...),
                "FREE": Path(...),
            },
        }

    Flattened format expected by compare_objective_set:

        {
            "C_FXD": Path(...),
            "C_FREE": Path(...),
            "T_FXD": Path(...),
            "T_FREE": Path(...),
        }

    Also supports the older flat format:

        {
            "FXD": Path(...),
            "FREE": Path(...),
        }
    """

    if len(files) == 0:
        return {}

    first_value = next(iter(files.values()))

    if not isinstance(first_value, dict):
        return files

    flattened_files = {}

    for optimization_name, variant_files in files.items():
        for variant_name, path in variant_files.items():
            flattened_label = f"{optimization_name}_{variant_name}"

            flattened_files[flattened_label] = path

    return flattened_files










def normalize_case_label(
        case,
    ):
    """
    Normalize a requested output folder label.

    Examples:
        1   -> 001
        001 -> 001
    """

    if case is None:
        return None

    case = str(case)

    if case.isdigit():
        return f"{int(case):03d}"

    return case





def get_requested_case_dirs(
        tutorial_dir,
        case = None,
    ):
    """
    Return either all output folders or one requested numbered output folder.
    """

    if case is None:
        return get_output_case_dirs(
            tutorial_dir = tutorial_dir,
        )

    case_label = normalize_case_label(
        case = case,
    )

    case_dir = tutorial_dir / case_label

    if not case_dir.exists():
        raise FileNotFoundError(
            f"Requested output folder does not exist: {case_dir}"
        )

    if not case_dir.is_dir():
        raise NotADirectoryError(
            f"Requested output folder is not a directory: {case_dir}"
        )

    return (
        case_dir,
    )










#==============================================================================================================
# Tutorial Objective Comparison
#==============================================================================================================

def tutorial_obj(
        tutorial : str,
        case = None,
    ):
    """
    Compare tutorial-specific objective values between FXD and FREE files.

    If case is None, write one comparison CSV per numbered output folder.
    If case is given, write only the CSV for that requested folder.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    objective_getter = TUTORIAL_OBJECTIVES.get(
        tutorial,
        None,
    )

    if objective_getter is None:
        raise ValueError(
            f"No tutorial objective registry entry found for tutorial = {tutorial}."
        )

    case_dirs = get_requested_case_dirs(
        tutorial_dir = tutorial_dir,
        case = case,
    )

    rows_by_case = {}
    csv_paths = {}

    for case_dir in case_dirs:

        files = find_h5_files(
            case_dir = case_dir,
        )

        files = flatten_h5_file_map(
            files = files,
        )

        rows = compare_objective_set(
            files = files,
            objective_getter = objective_getter,
        )

        if case_dir == tutorial_dir:
            csv_path = tutorial_dir / f"{tutorial}_case_obj.csv"

        else:
            csv_path = case_dir / f"{tutorial}_{case_dir.name}_case_obj.csv"

        write_table_csv(
            rows = rows,
            path = csv_path,
        )

        rows_by_case[case_dir.name] = rows
        csv_paths[case_dir.name] = csv_path

    return rows_by_case, csv_paths










#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tutorial",
        required = True,
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
    )

    parser.add_argument(
        "--case",
        default = None,
        help = "Optional numbered output folder to compare, e.g. 001 or 1. If omitted, all numbered folders are compared.",
    )

    return parser.parse_args()





def main():
    """
    Runs tutorial-specific objective comparison.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Comparing tutorial objectives for tutorial = {args.tutorial}")
    print("=" * 120 + "\n")

    rows_by_case, csv_paths = tutorial_obj(
        tutorial = args.tutorial,
        case = args.case,
    )

    print("")
    print("Finished tutorial-objective comparison.")
    print("CSV files written to:")
    print("")

    for case_label, csv_path in csv_paths.items():
        print(f"{case_label}: {csv_path}")

    print("")





if __name__ == "__main__":
    main()