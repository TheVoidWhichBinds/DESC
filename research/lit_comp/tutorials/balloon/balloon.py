# balloon.py
"""Infinite-n ideal ballooning stability optimization.

Write-up of the DESC ideal ballooning stability tutorial, minus plotting.
"""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from desc import set_device
set_device("gpu")

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

import numpy as np

import desc

from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    BallooningStability,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    GenericObjective,
    ObjectiveFunction,
)
from desc.optimize import Optimizer

from helper import (
    append_free_objectives,
    build_free_extension,
    iter_tolerance_cases,
    optimize_save_report,
    prepare_tolerance_case_dir,
    print_tolerance_case_header,
    should_run_tolerance_case,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "balloon"

OUTPUT_DIR = Path(__file__).resolve().parent










#========================================================================================================================================
# INITIAL EQUILIBRIUM
#========================================================================================================================================
eq0 = desc.examples.get("HELIOTRON")

eq0.change_resolution(
    L = 12,
    M = 6,
    N = 2,
    L_grid = 18,
    M_grid = 12,
    N_grid = 4,
)

eq0.surface = eq0.get_surface_at(
    rho = 1,
)










#========================================================================================================================================
# BALLOONING GRID
#========================================================================================================================================
alpha = np.linspace(
    0,
    np.pi,
    8,
    endpoint = False,
)

nturns = 3










#========================================================================================================================================
# SHARED OPTIMIZATION SETUP
#========================================================================================================================================
nzetaperturn = 200
mode_cutoff = 2
optimizer = Optimizer("proximal-lsq-exact")
maxiter = 500
x_scale = "auto"
BASE_OPTIONS = {
    "max_nfev": 500,
}


OUTER_FTOL_ORDERS = [2,4,6]

OUTER_XTOL_ORDERS = [6]

OUTER_GTOL_ORDERS = [3,6]

INNER_FTOL_ORDERS = [4,6]

INNER_XTOL_ORDERS = [6]

INNER_GTOL_ORDERS = [6]

INITIAL_TRUST_RATIO_ORDERS = [3]













#========================================================================================================================================
# OBJECTIVE / CONSTRAINT HELPERS
#========================================================================================================================================
def build_balloon_modes(
        eq,
    ):
    """
    Build boundary modes fixed during the ballooning optimization.
    """

    modes_R = np.vstack(
        (
            [
                0,
                0,
                0,
            ],
            eq.surface.R_basis.modes[
                np.max(
                    np.abs(eq.surface.R_basis.modes),
                    1,
                )
                > mode_cutoff,
                :,
            ],
        )
    )

    modes_Z = eq.surface.Z_basis.modes[
        np.max(
            np.abs(eq.surface.Z_basis.modes),
            1,
        )
        > mode_cutoff,
        :,
    ]

    return modes_R, modes_Z





def build_curvature_grid(
        eq,
    ):
    """
    Build the curvature grid used by the ballooning tutorial.
    """

    grid = LinearGrid(
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

    return grid





def build_balloon_objective(
        eq,
        free_objectives = (),
    ):
    """
    Build the ballooning objective function.
    """

    curvature_grid = build_curvature_grid(
        eq = eq,
    )

    objective = ObjectiveFunction(
        (
            BallooningStability(
                eq = eq,
                rho = np.array(
                    [
                        0.8,
                    ]
                ),
                alpha = alpha,
                nturns = nturns,
                nzetaperturn = nzetaperturn,
                weight = 2,
            ),
            AspectRatio(
                eq = eq,
                bounds = (
                    8,
                    11,
                ),
                weight = 1,
            ),
            GenericObjective(
                f = "curvature_k2_rho",
                thing = eq,
                grid = curvature_grid,
                bounds = (
                    -np.inf,
                    0,
                ),
                weight = 2,
            ),
        )
    )

    objective = append_free_objectives(
        objective = objective,
        free_objectives = free_objectives,
    )

    return objective





def build_balloon_constraints(
        eq,
        free_pressure = False,
    ):
    """
    Build FXD or FREE pressure constraints for the ballooning optimization.
    """

    modes_R, modes_Z = build_balloon_modes(
        eq = eq,
    )

    if free_pressure:
        free_objectives, free_constraints = build_free_extension(
            eq = eq,
            eq_initial = eq.copy(),
        )

        constraints = (
            ForceBalance(eq = eq),
            FixBoundaryR(
                eq = eq,
                modes = modes_R,
            ),
            FixBoundaryZ(
                eq = eq,
                modes = modes_Z,
            ),
            FixIota(eq = eq),
            FixPsi(eq = eq),
        ) + tuple(free_constraints)

    else:
        free_objectives = ()

        constraints = (
            ForceBalance(eq = eq),
            FixBoundaryR(
                eq = eq,
                modes = modes_R,
            ),
            FixBoundaryZ(
                eq = eq,
                modes = modes_Z,
            ),
            FixPressure(eq = eq),
            FixIota(eq = eq),
            FixPsi(eq = eq),
        )

    return constraints, free_objectives





def run_balloon_optimization(
        eq_initial,
        output_path,
        label,
        tolerance_case,
        free_pressure = False,
    ):
    """
    Run one ballooning optimization for one tolerance case.
    """

    eq = eq_initial.copy()

    constraints, free_objectives = build_balloon_constraints(
        eq = eq,
        free_pressure = free_pressure,
    )

    objective = build_balloon_objective(
        eq = eq,
        free_objectives = free_objectives,
    )

    tolerances = tolerance_case["tolerances"]

    eq, result = optimize_save_report(
        eq = eq,
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        output_path = output_path,
        label = label,
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = maxiter,
        options = tolerance_case["options"],
        copy = False,
        verbose = 3,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
    )

    return eq, result










#========================================================================================================================================
# TOLERANCE SWEEP
#========================================================================================================================================
ran_tolerance_case = False

for tolerance_case in iter_tolerance_cases(
        ftol_orders = OUTER_FTOL_ORDERS,
        xtol_orders = OUTER_XTOL_ORDERS,
        gtol_orders = OUTER_GTOL_ORDERS,
        initial_trust_ratio_orders = INITIAL_TRUST_RATIO_ORDERS,
        inner_ftol_orders = INNER_FTOL_ORDERS,
        inner_xtol_orders = INNER_XTOL_ORDERS,
        inner_gtol_orders = INNER_GTOL_ORDERS,
        base_options = BASE_OPTIONS,
    ):

    if not should_run_tolerance_case(
            tolerance_case = tolerance_case,
        ):
        continue

    ran_tolerance_case = True

    case_dir, tolerance_report_path = prepare_tolerance_case_dir(
        output_dir = OUTPUT_DIR,
        tolerance_case = tolerance_case,
    )

    print_tolerance_case_header(
        tolerance_case = tolerance_case,
        case_dir = case_dir,
    )

    INITIAL_PATH = case_dir / f"{fname}_initial.h5"

    BALLOON_FXD_PATH = case_dir / f"{fname}_optimized_FXD.h5"

    BALLOON_FREE_PATH = case_dir / f"{fname}_optimized_FREE.h5"

    eq0.save(
        str(INITIAL_PATH)
    )

    print("")
    print(f"Tolerance report written to: {tolerance_report_path}")
    print("")




    #========================================================================================================================================
    # OPTIMIZING BALLOONING STABILITY - FXD PRESSURE
    #========================================================================================================================================
    eq_balloon_FXD, result_FXD = run_balloon_optimization(
        eq_initial = eq0,
        output_path = BALLOON_FXD_PATH,
        label = "balloon_optimized_FXD",
        tolerance_case = tolerance_case,
        free_pressure = False,
    )




    #========================================================================================================================================
    # OPTIMIZING BALLOONING STABILITY - FREE PRESSURE
    #========================================================================================================================================
    eq_balloon_FREE, result_FREE = run_balloon_optimization(
        eq_initial = eq0,
        output_path = BALLOON_FREE_PATH,
        label = "balloon_optimized_FREE",
        tolerance_case = tolerance_case,
        free_pressure = True,
    )




if not ran_tolerance_case:
    requested_sweep_index = os.environ.get(
        "DESC_SWEEP_INDEX",
        None,
    )

    raise ValueError(
        f"No balloon tolerance case matched DESC_SWEEP_INDEX = {requested_sweep_index}."
    )
