# balloon.py
"""Infinite-n ideal ballooning stability optimization.

Write-up of the DESC ideal ballooning stability tutorial, minus plotting.
"""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
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










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "balloon"

OUTPUT_DIR = Path(__file__).resolve().parent

INITIAL_PATH = OUTPUT_DIR / f"{fname}_initial.h5"
OPTIMIZED_PATH = OUTPUT_DIR / f"{fname}_optimized.h5"










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
# OPTIMIZATION SETUP
#========================================================================================================================================
eq1 = eq0.copy()

nzetaperturn = 200

k = 2

modes_R = np.vstack(
    (
        [
            0,
            0,
            0,
        ],
        eq1.surface.R_basis.modes[
            np.max(
                np.abs(eq1.surface.R_basis.modes),
                1,
            )
            > k,
            :,
        ],
    )
)

modes_Z = eq1.surface.Z_basis.modes[
    np.max(
        np.abs(eq1.surface.Z_basis.modes),
        1,
    )
    > k,
    :,
]

constraints = (
    ForceBalance(eq = eq1),
    FixBoundaryR(
        eq = eq1,
        modes = modes_R,
    ),
    FixBoundaryZ(
        eq = eq1,
        modes = modes_Z,
    ),
    FixPressure(eq = eq1),
    FixIota(eq = eq1),
    FixPsi(eq = eq1),
)

Curvature_grid = LinearGrid(
    M = eq1.M_grid,
    N = eq1.N_grid,
    rho = np.array(
        [
            1.0,
        ]
    ),
    NFP = eq1.NFP,
    sym = eq1.sym,
    axis = False,
)

objective = ObjectiveFunction(
    (
        BallooningStability(
            eq = eq1,
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
            eq = eq1,
            bounds = (
                8,
                11,
            ),
            weight = 1,
        ),
        GenericObjective(
            f = "curvature_k2_rho",
            thing = eq1,
            grid = Curvature_grid,
            bounds = (
                -np.inf,
                0,
            ),
            weight = 2,
        ),
    )
)

optimizer = Optimizer("proximal-lsq-exact")










#========================================================================================================================================
# RUN BALLOONING OPTIMIZATION
#========================================================================================================================================
(eq1,), result = optimizer.optimize(
    eq1,
    objective,
    constraints,
    ftol = 1e-4,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 5,
    verbose = 3,
    options = {
        "initial_trust_ratio": 2e-3,
    },
)

eq1.save(str(OPTIMIZED_PATH))