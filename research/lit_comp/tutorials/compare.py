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
from desc.integrals import surface_max, surface_min
from helper import (
    compare_objective_set,
    find_h5_files,
    get_output_case_dirs,
    get_tutorial_dir,
    write_table_csv,
)










#==============================================================================================================
# Shared Objective Helpers
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





def build_force_balance_grid(
        eq,
    ):
    """
    Build the shared force-balance comparison grid.
    """

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return grid





def build_surface_grid(
        eq,
        rho,
        sym = None,
        axis = None,
    ):
    """
    Build a tutorial comparison grid on one or more flux surfaces.
    """

    if sym is None:
        sym = eq.sym

    kwargs = {
        "M": eq.M_grid,
        "N": eq.N_grid,
        "NFP": eq.NFP,
        "rho": np.atleast_1d(rho),
        "sym": sym,
    }

    if axis is not None:
        kwargs["axis"] = axis

    grid = LinearGrid(
        **kwargs,
    )

    return grid





def get_reference_volume(
        eq,
        reference_eq = None,
    ):
    """
    Return the run-reference volume when available.
    """

    if reference_eq is None:
        reference_eq = eq

    return reference_eq.compute("V")["V"]





def find_reference_h5_file(
        case_dir,
    ):
    """
    Find the run-local initial equilibrium file when it exists.
    """

    candidates = sorted(
        case_dir.glob("*_initial.h5")
    )

    if len(candidates) == 0:
        return None

    return candidates[0]





def normalize_loaded_equilibrium(
        loaded,
    ):
    """
    Convert a loaded DESC object into one representative equilibrium.
    """

    if hasattr(
            loaded,
            "equilibria",
        ):
        equilibria = loaded.equilibria

        if len(equilibria) > 0:
            return equilibria[-1]

    try:
        return loaded[-1]

    except Exception:
        return loaded





def load_reference_equilibrium(
        case_dir,
    ):
    """
    Load the run-local initial equilibrium when available.
    """

    reference_path = find_reference_h5_file(
        case_dir = case_dir,
    )

    if reference_path is None:
        return None

    try:
        from desc.equilibrium import Equilibrium

        loaded = Equilibrium.load(
            str(reference_path),
        )

        return normalize_loaded_equilibrium(
            loaded = loaded,
        )

    except Exception:
        pass

    try:
        from desc.equilibrium import EquilibriaFamily

        loaded = EquilibriaFamily.load(
            str(reference_path),
        )

        return normalize_loaded_equilibrium(
            loaded = loaded,
        )

    except Exception:
        return None





def wrap_objective_getter(
        objective_builder,
        case_dir,
    ):
    """
    Build the one-argument getter expected by helper.compare_objective_set.
    """

    reference_eq = load_reference_equilibrium(
        case_dir = case_dir,
    )

    def objective_getter(
            eq,
        ):
        return objective_builder(
            eq = eq,
            reference_eq = reference_eq,
        )

    return objective_getter










#==============================================================================================================
# User Objective Helpers
#==============================================================================================================

def fun_mirror_ratio(
        grid,
        data,
    ):
    """
    Compute the mirror ratio from |B|.
    """

    max_tz_B = surface_max(
        grid = grid,
        x = data["|B|"],
        surface_label = "rho",
    )

    min_tz_B = surface_min(
        grid = grid,
        x = data["|B|"],
        surface_label = "rho",
    )

    max_tz_B = grid.compress(
        max_tz_B,
        surface_label = "rho",
    )

    min_tz_B = grid.compress(
        min_tz_B,
        surface_label = "rho",
    )

    mirror_ratio = (max_tz_B - min_tz_B) / (min_tz_B + max_tz_B)

    return mirror_ratio










#==============================================================================================================
# Basic QS Objective Builders
#==============================================================================================================

def basic_qs_tripleqs_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Return comparison rows for the basic_qs/tripleQS optimization.
    """

    from desc.objectives import (
        ForceBalance,
        QuasisymmetryTripleProduct,
    )

    grid = build_force_balance_grid(
        eq = eq,
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
            name = "QuasisymmetryTripleProduct",
            objective = QuasisymmetryTripleProduct(
                eq = eq,
                helicity = (1, eq.NFP),
                grid = grid,
            ),
        ),
    ]





def basic_qs_twotermqh_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Return comparison rows for the basic_qs/twotermQH optimization.
    """

    from desc.objectives import (
        ForceBalance,
        QuasisymmetryTwoTerm,
    )

    grid = build_force_balance_grid(
        eq = eq,
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
            name = "QuasisymmetryTwoTerm",
            objective = QuasisymmetryTwoTerm(
                eq = eq,
                helicity = (1, eq.NFP),
                grid = grid,
            ),
        ),
    ]





def basic_qs_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Backward-compatible basic_qs objective list for old direct-run folders.
    """

    return basic_qs_tripleqs_objectives(
        eq = eq,
        reference_eq = reference_eq,
    )










#==============================================================================================================
# Advanced QS Objective Builders
#==============================================================================================================

def adv_qs_multigrid_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Return comparison rows for the adv_qs/multigrid optimization.
    """

    from desc.objectives import (
        AspectRatio,
        ForceBalance,
        QuasisymmetryTwoTerm,
    )

    force_grid = build_force_balance_grid(
        eq = eq,
    )

    qh_grid = build_surface_grid(
        eq = eq,
        rho = np.array(
            [
                0.6,
                0.8,
                1.0,
            ]
        ),
        sym = True,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = force_grid,
            ),
        ),
        objective_spec(
            name = "QuasisymmetryTwoTerm",
            objective = QuasisymmetryTwoTerm(
                eq = eq,
                helicity = (1, eq.NFP),
                grid = qh_grid,
            ),
        ),
        objective_spec(
            name = "AspectRatio",
            objective = AspectRatio(
                eq = eq,
                target = 8,
                weight = 100,
            ),
        ),
    ]





def adv_qs_auglag_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Return comparison rows for the adv_qs/auglag optimization.
    """

    from desc.objectives import (
        AspectRatio,
        Elongation,
        ForceBalance,
        GenericObjective,
        ObjectiveFromUser,
        RotationalTransform,
        Volume,
    )

    force_grid = build_force_balance_grid(
        eq = eq,
    )

    qh_grid = build_surface_grid(
        eq = eq,
        rho = np.array(
            [
                0.6,
                0.8,
                1.0,
            ]
        ),
        sym = True,
    )

    mirror_grid = LinearGrid(
        rho = 1.0,
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = force_grid,
            ),
        ),
        objective_spec(
            name = "QS Two-Term",
            objective = GenericObjective(
                f = "f_C",
                thing = eq,
                grid = qh_grid,
                compute_kwargs = {
                    "helicity": (1, eq.NFP),
                },
                name = "QS Two-Term",
            ),
        ),
        objective_spec(
            name = "AspectRatio",
            objective = AspectRatio(
                eq = eq,
                bounds = (
                    7,
                    9,
                ),
            ),
        ),
        objective_spec(
            name = "Elongation",
            objective = Elongation(
                eq = eq,
                bounds = (
                    0,
                    3,
                ),
            ),
        ),
        objective_spec(
            name = "Volume",
            objective = Volume(
                eq = eq,
                target = get_reference_volume(
                    eq = eq,
                    reference_eq = reference_eq,
                ),
            ),
        ),
        objective_spec(
            name = "RotationalTransform",
            objective = RotationalTransform(
                eq = eq,
                target = 1.1,
                loss_function = "mean",
            ),
        ),
        objective_spec(
            name = "MirrorRatio",
            objective = ObjectiveFromUser(
                fun = fun_mirror_ratio,
                thing = eq,
                grid = mirror_grid,
                bounds = (
                    0.18,
                    0.22,
                ),
                name = "MirrorRatio",
            ),
        ),
    ]





def adv_qs_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Backward-compatible adv_qs objective list for old direct-run folders.
    """

    return adv_qs_multigrid_objectives(
        eq = eq,
        reference_eq = reference_eq,
    )










#==============================================================================================================
# Ballooning Objective Builders
#==============================================================================================================

def balloon_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Return comparison rows for the ballooning optimization.

    FixIota, FixPressure, FixPsi, FixBoundaryR, and FixBoundaryZ are intentionally
    omitted because they are linear constraints and should not be compared.
    """

    from desc.objectives import (
        AspectRatio,
        BallooningStability,
        ForceBalance,
        GenericObjective,
    )

    alpha = np.linspace(
        0,
        np.pi,
        8,
        endpoint = False,
    )

    force_grid = build_force_balance_grid(
        eq = eq,
    )

    curvature_grid = build_surface_grid(
        eq = eq,
        rho = np.array(
            [
                1.0,
            ]
        ),
        sym = eq.sym,
        axis = False,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = force_grid,
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










#==============================================================================================================
# Neoclassical Objective Builders
#==============================================================================================================

def neoclassical_objectives(
        eq,
        reference_eq = None,
    ):
    """
    Return comparison rows for the neoclassical optimization.
    """

    from desc.objectives import (
        EffectiveRipple,
        ForceBalance,
    )

    force_grid = build_force_balance_grid(
        eq = eq,
    )

    neoclassical_grid = LinearGrid(
        rho = np.linspace(
            0.1,
            1.0,
            10,
        ),
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
                grid = force_grid,
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

TUTORIAL_OPTIMIZATION_OBJECTIVES = {
    ("basic_qs", "main"): basic_qs_objectives,
    ("basic_qs", "tripleQS"): basic_qs_tripleqs_objectives,
    ("basic_qs", "twotermQH"): basic_qs_twotermqh_objectives,
    ("adv_qs", "main"): adv_qs_objectives,
    ("adv_qs", "multigrid"): adv_qs_multigrid_objectives,
    ("adv_qs", "auglag"): adv_qs_auglag_objectives,
    ("balloon", "main"): balloon_objectives,
    ("balloon", "balloon"): balloon_objectives,
    ("neoclassical", "main"): neoclassical_objectives,
    ("neoclassical", "neoclassical"): neoclassical_objectives,
}

TUTORIAL_DEFAULT_OBJECTIVES = {
    "basic_qs": basic_qs_objectives,
    "adv_qs": adv_qs_objectives,
    "balloon": balloon_objectives,
    "neoclassical": neoclassical_objectives,
}





def get_objective_builder(
        tutorial,
        optimization_label,
    ):
    """
    Return the objective builder for a tutorial/optimization folder pair.
    """

    key = (
        tutorial,
        optimization_label,
    )

    objective_builder = TUTORIAL_OPTIMIZATION_OBJECTIVES.get(
        key,
        None,
    )

    if objective_builder is not None:
        return objective_builder

    objective_builder = TUTORIAL_DEFAULT_OBJECTIVES.get(
        tutorial,
        None,
    )

    if objective_builder is not None:
        return objective_builder

    raise ValueError(
        f"No tutorial objective registry entry found for tutorial = {tutorial}."
    )










#==============================================================================================================
# Nested Output Folder Helpers
#==============================================================================================================

def is_numbered_case_dir(
        path,
    ):
    """
    Return True for output folders named 001, 002, 003, ...
    """

    return path.is_dir() and path.name.isdigit()





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





def get_optimization_dirs(
        tutorial_dir,
    ):
    """
    Return tutorial-local optimization folders.

    New layout:
        basic_qs/tripleQS/001
        basic_qs/twotermQH/001
        adv_qs/multigrid/001
        adv_qs/auglag/001
        balloon/balloon/001
        neoclassical/neoclassical/001
    """

    optimization_dirs = []

    for path in sorted(tutorial_dir.iterdir()):
        if not path.is_dir():
            continue

        if path.name.startswith("__"):
            continue

        numbered_children = [
            child
            for child in path.iterdir()
            if is_numbered_case_dir(
                path = child,
            )
        ]

        if len(numbered_children) > 0:
            optimization_dirs.append(path)

    return tuple(optimization_dirs)





def get_nested_output_case_dirs(
        tutorial_dir,
    ):
    """
    Return all numbered case folders in the new optimization-subfolder layout.
    """

    case_dirs = []

    for optimization_dir in get_optimization_dirs(
            tutorial_dir = tutorial_dir,
        ):

        case_dirs.extend(
            sorted(
                path
                for path in optimization_dir.iterdir()
                if is_numbered_case_dir(
                    path = path,
                )
            )
        )

    return tuple(case_dirs)





def get_all_output_case_dirs(
        tutorial_dir,
    ):
    """
    Return output case folders for either the new or old layout.
    """

    nested_case_dirs = get_nested_output_case_dirs(
        tutorial_dir = tutorial_dir,
    )

    if len(nested_case_dirs) > 0:
        return nested_case_dirs

    direct_case_dirs = get_output_case_dirs(
        tutorial_dir = tutorial_dir,
    )

    return direct_case_dirs





def get_requested_case_dirs(
        tutorial_dir,
        case = None,
    ):
    """
    Return either all output folders or every optimization folder for one case label.
    """

    if case is None:
        return get_all_output_case_dirs(
            tutorial_dir = tutorial_dir,
        )

    case_label = normalize_case_label(
        case = case,
    )

    requested_case_dirs = []

    direct_case_dir = tutorial_dir / case_label

    if direct_case_dir.exists() and direct_case_dir.is_dir():
        requested_case_dirs.append(direct_case_dir)

    for optimization_dir in get_optimization_dirs(
            tutorial_dir = tutorial_dir,
        ):

        case_dir = optimization_dir / case_label

        if case_dir.exists() and case_dir.is_dir():
            requested_case_dirs.append(case_dir)

    if len(requested_case_dirs) == 0:
        raise FileNotFoundError(
            f"Requested output folder does not exist for case {case_label} under {tutorial_dir}."
        )

    return tuple(requested_case_dirs)





def get_optimization_label_from_case_dir(
        tutorial_dir,
        case_dir,
    ):
    """
    Return the optimization folder label for a case directory.
    """

    tutorial_dir = tutorial_dir.resolve()
    case_dir = case_dir.resolve()

    if case_dir.parent == tutorial_dir:
        return "main"

    return case_dir.parent.name





def make_csv_case_key(
        tutorial_dir,
        case_dir,
    ):
    """
    Return a stable key for rows_by_case and csv_paths.
    """

    optimization_label = get_optimization_label_from_case_dir(
        tutorial_dir = tutorial_dir,
        case_dir = case_dir,
    )

    if optimization_label == "main":
        return case_dir.name

    return f"{optimization_label}/{case_dir.name}"





def flatten_h5_file_map(
        files,
    ):
    """
    Flatten helper.find_h5_files output for folders containing multiple optimizations.
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





def collapse_single_optimization_h5_file_map(
        files,
    ):
    """
    Collapse a one-optimization folder into FXD/FREE columns.
    """

    if len(files) != 1:
        return flatten_h5_file_map(
            files = files,
        )

    first_value = next(iter(files.values()))

    if not isinstance(first_value, dict):
        return files

    return first_value





def make_case_csv_path(
        tutorial,
        tutorial_dir,
        case_dir,
    ):
    """
    Build the comparison CSV path for one output case folder.
    """

    optimization_label = get_optimization_label_from_case_dir(
        tutorial_dir = tutorial_dir,
        case_dir = case_dir,
    )

    if case_dir == tutorial_dir:
        return tutorial_dir / f"{tutorial}_case_obj.csv"

    if optimization_label == "main":
        return case_dir / f"{tutorial}_{case_dir.name}_case_obj.csv"

    return case_dir / f"{tutorial}_{optimization_label}_{case_dir.name}_case_obj.csv"










#==============================================================================================================
# Tutorial Objective Comparison
#==============================================================================================================

def tutorial_obj(
        tutorial : str,
        case = None,
    ):
    """
    Compare tutorial-specific objective values between FXD and FREE files.

    New layout writes one CSV inside each optimization/case folder.
    If case is None, every numbered folder is compared.
    If case is given, every optimization folder with that case label is compared.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    if tutorial not in TUTORIAL_DEFAULT_OBJECTIVES:
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

        optimization_label = get_optimization_label_from_case_dir(
            tutorial_dir = tutorial_dir,
            case_dir = case_dir,
        )

        objective_builder = get_objective_builder(
            tutorial = tutorial,
            optimization_label = optimization_label,
        )

        objective_getter = wrap_objective_getter(
            objective_builder = objective_builder,
            case_dir = case_dir,
        )

        files = find_h5_files(
            case_dir = case_dir,
        )

        files = collapse_single_optimization_h5_file_map(
            files = files,
        )

        if len(files) == 0:
            print("")
            print(f"No *_FXD.h5 or *_FREE.h5 files were found in: {case_dir}")
            print("")
            continue

        rows = compare_objective_set(
            files = files,
            objective_getter = objective_getter,
        )

        csv_path = make_case_csv_path(
            tutorial = tutorial,
            tutorial_dir = tutorial_dir,
            case_dir = case_dir,
        )

        write_table_csv(
            rows = rows,
            path = csv_path,
        )

        case_key = make_csv_case_key(
            tutorial_dir = tutorial_dir,
            case_dir = case_dir,
        )

        rows_by_case[case_key] = rows
        csv_paths[case_key] = csv_path

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
