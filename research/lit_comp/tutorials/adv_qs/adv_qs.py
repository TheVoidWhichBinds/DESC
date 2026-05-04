# adv_qs.py
"""Advanced quasi-symmetry optimization.

Write-up of the DESC Advanced QS Optimization tutorial, minus plotting.
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

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.integrals import surface_max, surface_min
from desc.objectives import (
    AspectRatio,
    Elongation,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    GenericObjective,
    ObjectiveFromUser,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    RotationalTransform,
    Volume,
)
from desc.optimize import Optimizer










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "adv_qs"

OUTPUT_DIR = Path(__file__).resolve().parent

MULTIGRID_PATH = OUTPUT_DIR / f"{fname}_multigrid.h5"
AUGLAG_PATH = OUTPUT_DIR / f"{fname}_auglag.h5"










#========================================================================================================================================
# INITIAL GUESS
#========================================================================================================================================
surf = FourierRZToroidalSurface(
    R_lmn = [1, 0.125, 0.1],
    Z_lmn = [-0.125, -0.1],
    modes_R = [[0, 0], [1, 0], [0, 1]],
    modes_Z = [[-1, 0], [0, -1]],
    NFP = 4,
)

eq = Equilibrium(
    M = 4,
    N = 4,
    Psi = 0.04,
    surface = surf,
)

eq0 = solve_continuation_automatic(
    eq,
    verbose = 0,
)[-1]

eqfam = EquilibriaFamily(eq0)










#========================================================================================================================================
# MULTIGRID METHOD WITH PROXIMAL OPTIMIZER
#========================================================================================================================================
def run_qh_step(k, eq):
    """Run a step of the precise QH optimization example from Landreman & Paul."""

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        rho = np.array([0.6, 0.8, 1.0]),
        sym = True,
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(
                eq = eq,
                helicity = (1, eq.NFP),
                grid = grid,
            ),
            AspectRatio(
                eq = eq,
                target = 8,
                weight = 100,
            ),
        ),
    )

    R_modes = np.vstack(
        (
            [0, 0, 0],
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

    Z_modes = eq.surface.Z_basis.modes[
        np.max(
            np.abs(eq.surface.Z_basis.modes),
            1,
        )
        > k,
        :,
    ]

    constraints = (
        ForceBalance(eq = eq),
        FixBoundaryR(
            eq = eq,
            modes = R_modes,
        ),
        FixBoundaryZ(
            eq = eq,
            modes = Z_modes,
        ),
        FixPressure(eq = eq),
        FixCurrent(eq = eq),
        FixPsi(eq = eq),
    )

    optimizer = Optimizer("proximal-lsq-exact")

    eq_new, history = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        maxiter = 20,
        verbose = 3,
        copy = True,
        options = {
            "initial_trust_ratio": 0.1,
        },
    )

    return eq_new










#========================================================================================================================================
# RUN MULTIGRID STEPS
#========================================================================================================================================
eq1 = run_qh_step(
    1,
    eq0,
)

eqfam.append(eq1)

eq2 = run_qh_step(
    2,
    eq1,
)

eqfam.append(eq2)

eq3 = run_qh_step(
    3,
    eq2,
)

eqfam.append(eq3)

eqfam.save(str(MULTIGRID_PATH))










#========================================================================================================================================
# CONSTRAINED OPTIMIZATION OBJECTIVE
#========================================================================================================================================
grid = LinearGrid(
    M = eq0.M_grid,
    N = eq0.N_grid,
    NFP = eq0.NFP,
    rho = np.array([0.6, 0.8, 1.0]),
    sym = True,
)

objective = ObjectiveFunction(
    (
        GenericObjective(
            f = "f_C",
            thing = eq0,
            grid = grid,
            compute_kwargs = {
                "helicity": (1, eq.NFP),
            },
            name = "QS Two-Term",
        ),
    ),
)










#========================================================================================================================================
# CONSTRAINED OPTIMIZATION CONSTRAINTS
#========================================================================================================================================
def fun_mirror_ratio(grid, data):
    """Compute the mirror ratio from |B|."""

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





obj_mirror_ratio = ObjectiveFromUser(
    fun = fun_mirror_ratio,
    thing = eq0,
    grid = LinearGrid(
        rho = 1.0,
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
    ),
    bounds = (0.18, 0.22),
    name = "my mirror ratio",
)

constraints = (
    ForceBalance(eq = eq0),
    AspectRatio(
        eq = eq0,
        bounds = (7, 9),
    ),
    Elongation(
        eq = eq0,
        bounds = (0, 3),
    ),
    Volume(
        eq = eq0,
        target = eq0.compute("V")["V"],
    ),
    RotationalTransform(
        eq = eq0,
        target = 1.1,
        loss_function = "mean",
    ),
    obj_mirror_ratio,
    FixBoundaryR(
        eq = eq0,
        modes = [0, 0, 0],
    ),
    FixPressure(eq = eq0),
    FixCurrent(eq = eq0),
    FixPsi(eq = eq0),
)










#========================================================================================================================================
# RUN CONSTRAINED OPTIMIZATION
#========================================================================================================================================
optimizer = Optimizer("lsq-auglag")

eqa, history = eq0.optimize(
    objective = objective,
    constraints = constraints,
    optimizer = optimizer,
    maxiter = 200,
    copy = True,
    verbose = 3,
    options = {},
)

eqa.solve()

eqa.save(str(AUGLAG_PATH))