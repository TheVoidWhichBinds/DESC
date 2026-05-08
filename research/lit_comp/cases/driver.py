# driver.py
#==============================================================================================================
#
# Top-level CLI for running DESC case optimizations with FXD/FREE pressure pairs.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 driver.py --case ATF
#   python3 driver.py --case ATF --obj qs3
#   python3 driver.py --case ATF --obj iso
#   python3 driver.py --case ATF --obj balloon
#   python3 driver.py --case ATF --obj all
#
#==============================================================================================================

import os
os.environ["JAX_PLATFORMS"] = "cpu"

if "JAX_PLATFORM_NAME" in os.environ:
    del os.environ["JAX_PLATFORM_NAME"]

from desc import set_device
set_device("cpu")

import argparse

import numpy as np

import desc.examples
from desc.grid import ConcentricGrid, LinearGrid
from desc.objectives import (
    BallooningStability,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    Isodynamicity,
    ObjectiveFunction,
    QuasisymmetryTripleProduct,
)

try:
    from .helper import (
        clear_objective_folder,
        ensure_case_layout,
        get_initial_equilibrium_path,
        get_objective_dir,
        get_output_path,
        optimize_save_report,
        remove_file_if_present,
        resolve_requested_objectives,
        build_free_extension,
    )

except ImportError:
    from helper import (
        clear_objective_folder,
        ensure_case_layout,
        get_initial_equilibrium_path,
        get_objective_dir,
        get_output_path,
        optimize_save_report,
        remove_file_if_present,
        resolve_requested_objectives,
        build_free_extension,
    )










#==============================================================================================================
# HYPERPARAMETERS
#==============================================================================================================

OPTIMIZER = "proximal-lsq-exact"

FTOL = 1e-6
XTOL = 1e-6
GTOL = 1e-6

MAXITER = 100
MAX_NFEV = None

X_SCALE = "auto"

ISO_RHO = np.array(
    [
        0.25,
        0.50,
        0.75,
        1.00,
    ]
)

BALLOON_RHO = np.array(
    [
        0.30,
        0.45,
        0.60,
        0.75,
    ]
)

BALLOON_ALPHA = np.linspace(
    0.0,
    np.pi,
    8,
    endpoint = False,
)

BALLOON_NTURNS = 3
BALLOON_NZETA_PER_TURN = 200










#==============================================================================================================
# Resolution / Grid Helpers
#==============================================================================================================

def get_grid_resolution(
        eq,
    ):
    """
    Return grid resolution equal to twice the incoming spectral resolution.
    """

    return {
        "L": 2 * int(eq.L),
        "M": 2 * int(eq.M),
        "N": 2 * int(eq.N),
    }





def print_hyperparameter_report(
        case,
        eq,
    ):
    """
    Print optimizer hyperparameters and incoming equilibrium resolution.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    print("")
    print("================================================================================================================")
    print("HYPERPARAMETERS")
    print("================================================================================================================")
    print("")
    print(f"case: {case}")
    print(f"optimizer: {OPTIMIZER}")
    print(f"ftol: {FTOL}")
    print(f"xtol: {XTOL}")
    print(f"gtol: {GTOL}")
    print(f"maxiter: {MAXITER}")
    print(f"max_nfev: {MAX_NFEV}")
    print(f"x_scale: {X_SCALE}")
    print("")
    print("Incoming equilibrium resolution:")
    print(f"L      = {eq.L}")
    print(f"M      = {eq.M}")
    print(f"N      = {eq.N}")
    print(f"L_grid = {eq.L_grid}")
    print(f"M_grid = {eq.M_grid}")
    print(f"N_grid = {eq.N_grid}")
    print("")
    print("Optimization/comparison grid resolution:")
    print(f"L_grid_objectives = {grid_resolution['L']}")
    print(f"M_grid_objectives = {grid_resolution['M']}")
    print(f"N_grid_objectives = {grid_resolution['N']}")
    print("")
    print("Isodynamicity surfaces:")
    print(f"ISO_RHO = {ISO_RHO}")
    print("")
    print("Ballooning settings:")
    print(f"BALLOON_RHO = {BALLOON_RHO}")
    print(f"BALLOON_ALPHA = {BALLOON_ALPHA}")
    print(f"BALLOON_NTURNS = {BALLOON_NTURNS}")
    print(f"BALLOON_NZETA_PER_TURN = {BALLOON_NZETA_PER_TURN}")
    print("================================================================================================================")
    print("")





def build_force_balance_grid(
        eq,
    ):
    """
    Build the force-balance grid.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    return LinearGrid(
        M = grid_resolution["M"],
        N = grid_resolution["N"],
        NFP = eq.NFP,
        sym = eq.sym,
    )





def build_qs3_grid(
        eq,
    ):
    """
    Build the quasi-symmetry triple-product grid.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    return ConcentricGrid(
        L = grid_resolution["L"],
        M = grid_resolution["M"],
        N = grid_resolution["N"],
        NFP = eq.NFP,
        sym = eq.sym,
    )





def build_iso_grid(
        eq,
    ):
    """
    Build the isodynamicity grid.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    return LinearGrid(
        rho = ISO_RHO,
        M = grid_resolution["M"],
        N = grid_resolution["N"],
        NFP = eq.NFP,
        sym = eq.sym,
    )










#==============================================================================================================
# Objective Builders
#==============================================================================================================

def build_qs3_objective(
        eq,
    ):
    """
    Build the quasi-symmetry triple-product objective.
    """

    return QuasisymmetryTripleProduct(
        eq = eq,
        grid = build_qs3_grid(
            eq = eq,
        ),
        normalize = True,
    )





def build_iso_objective(
        eq,
    ):
    """
    Build the isodynamicity objective.
    """

    return Isodynamicity(
        eq = eq,
        grid = build_iso_grid(
            eq = eq,
        ),
        normalize = True,
    )





def build_balloon_objective(
        eq,
    ):
    """
    Build the ideal ballooning stability objective.
    """

    return BallooningStability(
        eq = eq,
        rho = BALLOON_RHO,
        alpha = BALLOON_ALPHA,
        nturns = BALLOON_NTURNS,
        nzetaperturn = BALLOON_NZETA_PER_TURN,
        normalize = True,
    )





def build_primary_objectives(
        eq,
        obj,
    ):
    """
    Build the requested objective tuple for one optimization folder.
    """

    if obj == "qs3":
        return (
            build_qs3_objective(
                eq = eq,
            ),
        )

    if obj == "iso":
        return (
            build_iso_objective(
                eq = eq,
            ),
        )

    if obj == "balloon":
        return (
            build_balloon_objective(
                eq = eq,
            ),
        )

    if obj == "all":
        return (
            build_qs3_objective(
                eq = eq,
            ),
            build_iso_objective(
                eq = eq,
            ),
            build_balloon_objective(
                eq = eq,
            ),
        )

    raise ValueError(
        f"Unknown objective folder: {obj}"
    )





def build_core_constraints(
        eq,
    ):
    """
    Build constraints shared by FXD and FREE pressure optimizations.
    """

    return (
        FixBoundaryZ(
            eq = eq,
        ),
        FixBoundaryR(
            eq = eq,
        ),
        ForceBalance(
            eq = eq,
            grid = build_force_balance_grid(
                eq = eq,
            ),
            normalize = True,
        ),
        FixIota(
            eq = eq,
        ),
        FixCurrent(
            eq = eq,
        ),
        FixPsi(
            eq = eq,
        ),
    )





def build_optimization_problem(
        eq,
        eq_initial,
        obj,
        variant,
    ):
    """
    Build the ObjectiveFunction objects for one FXD or FREE optimization.
    """

    primary_objectives = build_primary_objectives(
        eq = eq,
        obj = obj,
    )

    constraints = build_core_constraints(
        eq = eq,
    )

    variant = str(variant).upper()

    if variant == "FXD":
        constraints = constraints + (
            FixPressure(
                eq = eq,
            ),
        )

    elif variant == "FREE":
        free_objectives, free_constraints = build_free_extension(
            eq = eq,
            eq_initial = eq_initial,
        )

        primary_objectives = primary_objectives + tuple(free_objectives)
        constraints = constraints + tuple(free_constraints)

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    objective = ObjectiveFunction(
        objectives = primary_objectives,
    )

    constraint_objective = ObjectiveFunction(
        objectives = constraints,
    )

    return objective, constraint_objective










#==============================================================================================================
# Optimization Helpers
#==============================================================================================================

def make_optimizer_options():
    """
    Build the optimizer options dictionary.
    """

    options = {}

    if MAX_NFEV is not None:
        options["max_nfev"] = MAX_NFEV

    if len(options) == 0:
        return None

    return options





def load_initial_equilibrium(
        case,
    ):
    """
    Load the initial equilibrium from DESC examples.
    """

    print("")
    print("================================================================================================================")
    print(f"Loading DESC example equilibrium: {case}")
    print("================================================================================================================")
    print("")

    eq = desc.examples.get(
        case,
    )

    return eq





def save_initial_equilibrium(
        case,
        eq,
    ):
    """
    Save the incoming DESC example equilibrium inside the case folder.
    """

    initial_path = get_initial_equilibrium_path(
        case = case,
    )

    remove_file_if_present(
        path = initial_path,
    )

    eq.save(
        str(initial_path)
    )

    print("")
    print(f"Saved initial equilibrium: {initial_path}")
    print("")

    return initial_path





def run_one_variant(
        case,
        obj,
        variant,
        eq_initial,
    ):
    """
    Run one FXD or FREE pressure optimization.
    """

    variant = str(variant).upper()

    eq = eq_initial.copy()

    objective, constraints = build_optimization_problem(
        eq = eq,
        eq_initial = eq_initial,
        obj = obj,
        variant = variant,
    )

    output_path = get_output_path(
        case = case,
        obj = obj,
        variant = variant,
    )

    label = f"{case}_{obj}_{variant}"

    print("")
    print("################################################################################################################")
    print(f"Starting optimization: {label}")
    print(f"Output path: {output_path}")
    print("################################################################################################################")
    print("")

    eq, result = optimize_save_report(
        eq = eq,
        objective = objective,
        constraints = constraints,
        optimizer = OPTIMIZER,
        output_path = output_path,
        label = label,
        ftol = FTOL,
        xtol = XTOL,
        gtol = GTOL,
        maxiter = MAXITER,
        options = make_optimizer_options(),
        copy = False,
        x_scale = X_SCALE,
    )

    return eq, result





def run_objective_pair(
        case,
        obj,
        eq_initial,
    ):
    """
    Run the FXD/FREE pair for one objective folder.
    """

    objective_dir = get_objective_dir(
        case = case,
        obj = obj,
    )

    clear_objective_folder(
        folder = objective_dir,
    )

    run_one_variant(
        case = case,
        obj = obj,
        variant = "FXD",
        eq_initial = eq_initial,
    )

    run_one_variant(
        case = case,
        obj = obj,
        variant = "FREE",
        eq_initial = eq_initial,
    )





def run_case(
        case,
        obj = None,
    ):
    """
    Run requested optimization pairs for one DESC example case.
    """

    objective_names = resolve_requested_objectives(
        obj = obj,
    )

    ensure_case_layout(
        case = case,
    )

    eq_initial = load_initial_equilibrium(
        case = case,
    )

    print_hyperparameter_report(
        case = case,
        eq = eq_initial,
    )

    save_initial_equilibrium(
        case = case,
        eq = eq_initial,
    )

    for objective_name in objective_names:
        run_objective_pair(
            case = case,
            obj = objective_name,
            eq_initial = eq_initial,
        )

    return objective_names





def run_comparison_outputs(
        case,
        obj = None,
    ):
    """
    Run compare.py logic after optimization outputs are written.
    """

    try:
        from .compare import case_obj

    except ImportError:
        from compare import case_obj

    print("")
    print("================================================================================================================")
    print("Running case comparison CSV generation")
    print("================================================================================================================")
    print("")

    rows, csv_paths = case_obj(
        case = case,
        obj = obj,
    )

    print("")
    print("Finished case-objective comparison.")
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
        "--case",
        required = True,
        help = "DESC example/case name, e.g. ATF, HELIOTRON, NCSX, W7-X.",
    )

    parser.add_argument(
        "--obj",
        default = None,
        choices = [
            "qs3",
            "iso",
            "balloon",
            "all",
        ],
        help = "Optional objective pair to run. If omitted, all four pairs run.",
    )

    parser.add_argument(
        "--skip-compare",
        action = "store_true",
        help = "Skip comparison CSV generation after the case run.",
    )

    return parser.parse_args()





def main():
    """
    Run one DESC example case.
    """

    args = parse_args()

    print("")
    print("================================================================================================================")
    print(f"Running case = {args.case}")

    if args.obj is None:
        print("Objective mode: all objective pairs")

    else:
        print(f"Objective mode: {args.obj}")

    print("Backend: CPU")
    print("Overwrite mode: enabled")
    print("================================================================================================================")
    print("")

    run_case(
        case = args.case,
        obj = args.obj,
    )

    if not args.skip_compare:
        run_comparison_outputs(
            case = args.case,
            obj = args.obj,
        )





if __name__ == "__main__":
    main()
