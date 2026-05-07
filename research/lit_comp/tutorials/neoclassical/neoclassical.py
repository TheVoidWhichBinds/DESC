# neoclassical.py
"""Neoclassical transport optimization.

Write-up of the DESC neoclassical transport and fast ions tutorial, minus plotting.
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

from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    EffectiveRipple,
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
fname = "neoclassical"

OUTPUT_DIR = Path(__file__).resolve().parent










#========================================================================================================================================
# INITIAL EQUILIBRIUM
#========================================================================================================================================
eq0 = get("HELIOTRON")










#========================================================================================================================================
# SHARED OPTIMIZATION SETUP
#========================================================================================================================================
def build_boundary_modes(
        eq,
        k,
    ):
    """
    Build boundary modes fixed during the neoclassical optimization.
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
                > k,
                :,
            ],
        )
    )

    modes_Z = eq.surface.Z_basis.modes[
        np.max(
            np.abs(eq.surface.Z_basis.modes),
            1,
        )
        > k,
        :,
    ]

    return modes_R, modes_Z





def build_curvature_grid(
        eq,
    ):
    """
    Build the curvature constraint grid.
    """

    grid = LinearGrid(
        rho = np.array(
            [
                1.0,
            ]
        ),
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return grid





def build_ripple_grid(
        eq,
    ):
    """
    Build the effective ripple objective grid.
    """

    grid = LinearGrid(
        rho = np.linspace(
            0.2,
            1,
            3,
        ),
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = False,
    )

    return grid





def build_neoclassical_objective(
        eq,
        free_objectives = (),
    ):
    """
    Build the neoclassical objective function.
    """

    objective = ObjectiveFunction(
        (
            EffectiveRipple(
                eq,
                grid = build_ripple_grid(
                    eq = eq,
                ),
                X = 16,
                Y = 32,
                Y_B = 133,
                num_transit = 10,
                num_well = 25 * 10,
                num_quad = 32,
                num_pitch = 45,
                jac_chunk_size = 1,
            ),
            AspectRatio(
                eq,
                bounds = (
                    8,
                    11,
                ),
                weight = 1e3,
            ),
            GenericObjective(
                "curvature_k2_rho",
                eq,
                grid = build_curvature_grid(
                    eq = eq,
                ),
                bounds = (
                    -128,
                    10,
                ),
                weight = 2e3,
            ),
        )
    )

    objective = append_free_objectives(
        objective = objective,
        free_objectives = free_objectives,
    )

    return objective





def build_neoclassical_constraints(
        eq,
        k,
        free_pressure = False,
    ):
    """
    Build constraints for the neoclassical optimization.
    """

    modes_R, modes_Z = build_boundary_modes(
        eq = eq,
        k = k,
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





k = 1

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
# OPTIMIZATION HELPER
#========================================================================================================================================
def run_neoclassical_optimization(
        eq_initial,
        output_path,
        label,
        tolerance_case,
        free_pressure = False,
    ):
    """
    Run one neoclassical optimization for one tolerance case.
    """

    eq = eq_initial.copy()

    constraints, free_objectives = build_neoclassical_constraints(
        eq = eq,
        k = k,
        free_pressure = free_pressure,
    )

    objective = build_neoclassical_objective(
        eq = eq,
        free_objectives = free_objectives,
    )

    tolerances = tolerance_case["outer_tolerances"]

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

    NEOCLASSICAL_FXD_PATH = case_dir / f"{fname}_optimized_FXD.h5"

    NEOCLASSICAL_FREE_PATH = case_dir / f"{fname}_optimized_FREE.h5"

    eq0.save(
        str(INITIAL_PATH)
    )

    print("")
    print(f"Tolerance report written to: {tolerance_report_path}")
    print("")




    #========================================================================================================================================
    # RUN NEOCLASSICAL OPTIMIZATION - FXD PRESSURE
    #========================================================================================================================================
    eq_neoclassical_FXD, result_FXD = run_neoclassical_optimization(
        eq_initial = eq0,
        output_path = NEOCLASSICAL_FXD_PATH,
        label = "neoclassical_optimized_FXD",
        tolerance_case = tolerance_case,
        free_pressure = False,
    )




    #========================================================================================================================================
    # RUN NEOCLASSICAL OPTIMIZATION - FREE PRESSURE
    #========================================================================================================================================
    eq_neoclassical_FREE, result_FREE = run_neoclassical_optimization(
        eq_initial = eq0,
        output_path = NEOCLASSICAL_FREE_PATH,
        label = "neoclassical_optimized_FREE",
        tolerance_case = tolerance_case,
        free_pressure = True,
    )




if not ran_tolerance_case:
    requested_sweep_index = os.environ.get(
        "DESC_SWEEP_INDEX",
        None,
    )

    raise ValueError(
        f"No neoclassical tolerance case matched DESC_SWEEP_INDEX = {requested_sweep_index}."
    )
