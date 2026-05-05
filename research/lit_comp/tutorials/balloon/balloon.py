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

from desc.grid import Grid, LinearGrid
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
    optimize_save_report,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "balloon"

OUTPUT_DIR = Path(__file__).resolve().parent

INITIAL_PATH = OUTPUT_DIR / f"{fname}_initial.h5"

BALLOON_FXD_PATH = OUTPUT_DIR / f"{fname}_optimized_FXD.h5"

BALLOON_FREE_PATH = OUTPUT_DIR / f"{fname}_optimized_FREE.h5"










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

eq0.save(str(INITIAL_PATH))










#========================================================================================================================================
# BALLOONING GRID
#========================================================================================================================================
surfaces = np.array(
    [
        0.01,
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
)

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

k = 2

optimizer = Optimizer("proximal-lsq-exact")
maxiter = 200
x_scale = "auto"

ftol = 1e-4
xtol = 1e-6
gtol = 1e-6
options = {
    "initial_trust_ratio": 2e-3,
    "max_nfev": 200,
    "solve_options": {
        "ftol": 1e-2,
        "xtol": 1e-3,
        "gtol": 1e-4,
        "verbose": 0,
    },
}










#========================================================================================================================================
# OPTIMIZING BALLOONING STABILITY - FXD PRESSURE
#========================================================================================================================================
eq_balloon_FXD = eq0.copy()

modes_R = np.vstack(
    (
        [
            0,
            0,
            0,
        ],
        eq_balloon_FXD.surface.R_basis.modes[
            np.max(
                np.abs(eq_balloon_FXD.surface.R_basis.modes),
                1,
            )
            > k,
            :,
        ],
    )
)

modes_Z = eq_balloon_FXD.surface.Z_basis.modes[
    np.max(
        np.abs(eq_balloon_FXD.surface.Z_basis.modes),
        1,
    )
    > k,
    :,
]

constraints = (
    ForceBalance(eq = eq_balloon_FXD),
    FixBoundaryR(
        eq = eq_balloon_FXD,
        modes = modes_R,
    ),
    FixBoundaryZ(
        eq = eq_balloon_FXD,
        modes = modes_Z,
    ),
    FixPressure(eq = eq_balloon_FXD),
    FixIota(eq = eq_balloon_FXD),
    FixPsi(eq = eq_balloon_FXD),
)

Curvature_grid = LinearGrid(
    M = eq_balloon_FXD.M_grid,
    N = eq_balloon_FXD.N_grid,
    rho = np.array(
        [
            1.0,
        ]
    ),
    NFP = eq_balloon_FXD.NFP,
    sym = eq_balloon_FXD.sym,
    axis = False,
)

objective_FXD = ObjectiveFunction(
    (
        BallooningStability(
            eq = eq_balloon_FXD,
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
            eq = eq_balloon_FXD,
            bounds = (
                8,
                11,
            ),
            weight = 1,
        ),
        GenericObjective(
            f = "curvature_k2_rho",
            thing = eq_balloon_FXD,
            grid = Curvature_grid,
            bounds = (
                -np.inf,
                0,
            ),
            weight = 2,
        ),
    )
)

eq_balloon_FXD, result_FXD = optimize_save_report(
    eq = eq_balloon_FXD,
    objective = objective_FXD,
    constraints = constraints,
    optimizer = optimizer,
    output_path = BALLOON_FXD_PATH,
    label = "balloon_optimized_FXD",
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
# OPTIMIZING BALLOONING STABILITY - FREE PRESSURE
#========================================================================================================================================
eq_balloon_FREE = eq0.copy()

free_objectives, free_constraints = build_free_extension(
    eq = eq_balloon_FREE,
    eq_initial = eq_balloon_FREE.copy(),
)

modes_R = np.vstack(
    (
        [
            0,
            0,
            0,
        ],
        eq_balloon_FREE.surface.R_basis.modes[
            np.max(
                np.abs(eq_balloon_FREE.surface.R_basis.modes),
                1,
            )
            > k,
            :,
        ],
    )
)

modes_Z = eq_balloon_FREE.surface.Z_basis.modes[
    np.max(
        np.abs(eq_balloon_FREE.surface.Z_basis.modes),
        1,
    )
    > k,
    :,
]

constraints = (
    ForceBalance(eq = eq_balloon_FREE),
    FixBoundaryR(
        eq = eq_balloon_FREE,
        modes = modes_R,
    ),
    FixBoundaryZ(
        eq = eq_balloon_FREE,
        modes = modes_Z,
    ),
    FixIota(eq = eq_balloon_FREE),
    FixPsi(eq = eq_balloon_FREE),
) + tuple(free_constraints)

Curvature_grid = LinearGrid(
    M = eq_balloon_FREE.M_grid,
    N = eq_balloon_FREE.N_grid,
    rho = np.array(
        [
            1.0,
        ]
    ),
    NFP = eq_balloon_FREE.NFP,
    sym = eq_balloon_FREE.sym,
    axis = False,
)

objective_FREE = ObjectiveFunction(
    (
        BallooningStability(
            eq = eq_balloon_FREE,
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
            eq = eq_balloon_FREE,
            bounds = (
                8,
                11,
            ),
            weight = 1,
        ),
        GenericObjective(
            f = "curvature_k2_rho",
            thing = eq_balloon_FREE,
            grid = Curvature_grid,
            bounds = (
                -np.inf,
                0,
            ),
            weight = 2,
        ),
    )
)

objective_FREE = append_free_objectives(
    objective = objective_FREE,
    free_objectives = free_objectives,
)

(eq_balloon_FREE,), result_FREE = optimizer.optimize(
    eq_balloon_FREE,
    objective_FREE,
    constraints,
    ftol = 1e-4,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 200,
    verbose = 3,
    options = {
        "initial_trust_ratio": 2e-3,
        "max_nfev": 200,
        "solve_options": {
            "ftol": 1e-2,
            "xtol": 1e-3,
            "gtol": 1e-4,
            "verbose": 0,
        },
    },
)

eq_balloon_FREE.save(str(BALLOON_FREE_PATH))