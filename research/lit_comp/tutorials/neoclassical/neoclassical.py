# neoclassical.py
"""Neoclassical transport optimization.

Write-up of the DESC neoclassical transport and fast ions tutorial, minus plotting.
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










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "neoclassical"

OUTPUT_DIR = Path(__file__).resolve().parent

INITIAL_PATH = OUTPUT_DIR / f"{fname}_initial.h5"
OPTIMIZED_PATH = OUTPUT_DIR / f"{fname}_optimized.h5"










#========================================================================================================================================
# INITIAL EQUILIBRIUM
#========================================================================================================================================
eq0 = get("HELIOTRON")

eq0.save(str(INITIAL_PATH))










#========================================================================================================================================
# OPTIMIZATION SETUP
#========================================================================================================================================
eq1 = eq0.copy()

k = 1

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

curvature_grid = LinearGrid(
    rho = np.array(
        [
            1.0,
        ]
    ),
    M = eq1.M_grid,
    N = eq1.N_grid,
    NFP = eq1.NFP,
    sym = eq1.sym,
)

ripple_grid = LinearGrid(
    rho = np.linspace(
        0.2,
        1,
        3,
    ),
    M = eq1.M_grid,
    N = eq1.N_grid,
    NFP = eq1.NFP,
    sym = False,
)

objective = ObjectiveFunction(
    (
        EffectiveRipple(
            eq1,
            grid = ripple_grid,
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
            eq1,
            bounds = (
                8,
                11,
            ),
            weight = 1e3,
        ),
        GenericObjective(
            "curvature_k2_rho",
            eq1,
            grid = curvature_grid,
            bounds = (
                -128,
                10,
            ),
            weight = 2e3,
        ),
    )
)

optimizer = Optimizer("proximal-lsq-exact")










#========================================================================================================================================
# RUN NEOCLASSICAL OPTIMIZATION
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