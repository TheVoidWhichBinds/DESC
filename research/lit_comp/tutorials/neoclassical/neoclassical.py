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
    optimize_save_report,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "neoclassical"

OUTPUT_DIR = Path(__file__).resolve().parent

INITIAL_PATH = OUTPUT_DIR / f"{fname}_initial.h5"

NEOCLASSICAL_FXD_PATH = OUTPUT_DIR / f"{fname}_optimized_FXD.h5"

NEOCLASSICAL_FREE_PATH = OUTPUT_DIR / f"{fname}_optimized_FREE.h5"










#========================================================================================================================================
# INITIAL EQUILIBRIUM
#========================================================================================================================================
eq0 = get("HELIOTRON")

eq0.save(str(INITIAL_PATH))










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
maxiter = 5
x_scale = "auto"

ftol = 1e-4
xtol = 1e-6
gtol = 1e-6
options = {
    "initial_trust_ratio": 2e-3,
}










#========================================================================================================================================
# RUN NEOCLASSICAL OPTIMIZATION - FXD PRESSURE
#========================================================================================================================================
eq_neoclassical_FXD = eq0.copy()

constraints, free_objectives = build_neoclassical_constraints(
    eq = eq_neoclassical_FXD,
    k = k,
    free_pressure = False,
)

objective_FXD = build_neoclassical_objective(
    eq = eq_neoclassical_FXD,
    free_objectives = free_objectives,
)

eq_neoclassical_FXD, result_FXD = optimize_save_report(
    eq = eq_neoclassical_FXD,
    objective = objective_FXD,
    constraints = constraints,
    optimizer = optimizer,
    output_path = NEOCLASSICAL_FXD_PATH,
    label = "neoclassical_optimized_FXD",
    ftol = ftol,
    xtol = xtol,
    gtol = gtol,
    maxiter = maxiter,
    options = options,
    copy = False,
    verbose = 3,
    x_scale = x_scale,
)










#========================================================================================================================================
# RUN NEOCLASSICAL OPTIMIZATION - FREE PRESSURE
#========================================================================================================================================
eq_neoclassical_FREE = eq0.copy()

constraints, free_objectives = build_neoclassical_constraints(
    eq = eq_neoclassical_FREE,
    k = k,
    free_pressure = True,
)

objective_FREE = build_neoclassical_objective(
    eq = eq_neoclassical_FREE,
    free_objectives = free_objectives,
)

eq_neoclassical_FREE, result_FREE = optimize_save_report(
    eq = eq_neoclassical_FREE,
    objective = objective_FREE,
    constraints = constraints,
    optimizer = optimizer,
    output_path = NEOCLASSICAL_FREE_PATH,
    label = "neoclassical_optimized_FREE",
    ftol = ftol,
    xtol = xtol,
    gtol = gtol,
    maxiter = maxiter,
    options = options,
    copy = False,
    verbose = 3,
    x_scale = x_scale,
)
