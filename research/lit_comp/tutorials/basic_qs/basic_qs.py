# basic_qs.py
"""Basic quasi-symmetry optimization.

Write-up of the DESC Basic QS Optimization tutorial, minus plotting.
"""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from desc import set_device
set_device("gpu")
import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))
import numpy as np
import desc.io
from desc.grid import LinearGrid, ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
    ForceBalance,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
)
from desc.optimize import Optimizer

from helper import (
    append_free_objectives,
    build_free_extension,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "basic_qs"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRIPLE_PRODUCT_FXD_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_T_FXD.h5",
)

TRIPLE_PRODUCT_FREE_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_T_FREE.h5",
)

TWO_TERM_FXD_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_C_FXD.h5",
)

TWO_TERM_FREE_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_C_FREE.h5",
)










#========================================================================================================================================
# INITIAL GUESS
#========================================================================================================================================
DESC_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../..",
    )
)

QS_INITIAL_GUESS_PATH = os.path.join(
    DESC_ROOT,
    "docs",
    "notebooks",
    "tutorials",
    "qs_initial_guess.h5",
)

eq_init = desc.io.load(QS_INITIAL_GUESS_PATH)










#========================================================================================================================================
# OPTIMIZER
#========================================================================================================================================
optimizer = Optimizer("proximal-lsq-exact")










#========================================================================================================================================
# SPECIFYING CONSTRAINTS
#========================================================================================================================================
idx_Rcc = eq_init.surface.R_basis.get_idx(M = 1, N = 2)
idx_Rss = eq_init.surface.R_basis.get_idx(M = -1, N = -2)
idx_Zsc = eq_init.surface.Z_basis.get_idx(M = -1, N = 2)
idx_Zcs = eq_init.surface.Z_basis.get_idx(M = 1, N = -2)

R_modes = np.delete(
    eq_init.surface.R_basis.modes,
    [
        idx_Rcc,
        idx_Rss,
    ],
    axis = 0,
)

Z_modes = np.delete(
    eq_init.surface.Z_basis.modes,
    [
        idx_Zsc,
        idx_Zcs,
    ],
    axis = 0,
)










#========================================================================================================================================
# SHARED GRIDS
#========================================================================================================================================
grid_vol = ConcentricGrid(
    L = eq_init.L_grid,
    M = eq_init.M_grid,
    N = eq_init.N_grid,
    NFP = eq_init.NFP,
    sym = eq_init.sym,
)

grid_rho1 = LinearGrid(
    M = eq_init.M_grid,
    N = eq_init.N_grid,
    NFP = eq_init.NFP,
    sym = eq_init.sym,
    rho = np.array(1.0),
)










#========================================================================================================================================
# OPTIMIZING FOR TRIPLE PRODUCT QS IN VOLUME - FXD PRESSURE
#========================================================================================================================================
eq_qs_T_FXD = eq_init.copy()

constraints = (
    ForceBalance(eq = eq_qs_T_FXD),
    FixBoundaryR(eq = eq_qs_T_FXD, modes = R_modes),
    FixBoundaryZ(eq = eq_qs_T_FXD, modes = Z_modes),
    FixPressure(eq = eq_qs_T_FXD),
    FixIota(eq = eq_qs_T_FXD),
    FixPsi(eq = eq_qs_T_FXD),
)

objective_fT = ObjectiveFunction(
    QuasisymmetryTripleProduct(
        eq = eq_qs_T_FXD,
        grid = grid_vol,
    )
)

eq_qs_T_FXD, result_T_FXD = eq_qs_T_FXD.optimize(
    objective = objective_fT,
    constraints = constraints,
    optimizer = optimizer,
    ftol = 1e-4,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 50,
    options = {
        "perturb_options": {
            "order": 2,
            "verbose": 0,
        },
        "solve_options": {
            "ftol": 1e-1,
            "xtol": 1e-2,
            "gtol": 1e-2,
            "verbose": 0,
        },
    },
    copy = False,
    verbose = 3,
)

eq_qs_T_FXD.save(TRIPLE_PRODUCT_FXD_PATH)










#========================================================================================================================================
# OPTIMIZING FOR TRIPLE PRODUCT QS IN VOLUME - FREE PRESSURE
#========================================================================================================================================
eq_qs_T_FREE = eq_init.copy()

free_objectives_T, free_constraints_T = build_free_extension(
    eq = eq_qs_T_FREE,
    eq_initial = eq_qs_T_FREE.copy(),
)

constraints = (
    ForceBalance(eq = eq_qs_T_FREE),
    FixBoundaryR(eq = eq_qs_T_FREE, modes = R_modes),
    FixBoundaryZ(eq = eq_qs_T_FREE, modes = Z_modes),
    FixIota(eq = eq_qs_T_FREE),
    FixPsi(eq = eq_qs_T_FREE),
) + tuple(free_constraints_T)

objective_fT_FREE = ObjectiveFunction(
    QuasisymmetryTripleProduct(
        eq = eq_qs_T_FREE,
        grid = grid_vol,
    )
)

objective_fT_FREE = append_free_objectives(
    objective = objective_fT_FREE,
    free_objectives = free_objectives_T,
)

eq_qs_T_FREE, result_T_FREE = eq_qs_T_FREE.optimize(
    objective = objective_fT_FREE,
    constraints = constraints,
    optimizer = optimizer,
    ftol = 1e-4,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 50,
    options = {
        "perturb_options": {
            "order": 2,
            "verbose": 0,
        },
        "solve_options": {
            "ftol": 1e-1,
            "xtol": 1e-2,
            "gtol": 1e-2,
            "verbose": 0,
        },
    },
    copy = False,
    verbose = 3,
)

eq_qs_T_FREE.save(TRIPLE_PRODUCT_FREE_PATH)










#========================================================================================================================================
# OPTIMIZING FOR TWO-TERM QH AT BOUNDARY SURFACE - FXD PRESSURE
#========================================================================================================================================
eq_qs_C_FXD = eq_init.copy()

constraints = (
    ForceBalance(eq = eq_qs_C_FXD),
    FixBoundaryR(eq = eq_qs_C_FXD, modes = R_modes),
    FixBoundaryZ(eq = eq_qs_C_FXD, modes = Z_modes),
    FixPressure(eq = eq_qs_C_FXD),
    FixIota(eq = eq_qs_C_FXD),
    FixPsi(eq = eq_qs_C_FXD),
)

objective_fC = ObjectiveFunction(
    QuasisymmetryTwoTerm(
        eq = eq_qs_C_FXD,
        grid = grid_rho1,
        helicity = (1, eq_init.NFP),
    )
)

eq_qs_C_FXD, result_C_FXD = eq_qs_C_FXD.optimize(
    objective = objective_fC,
    constraints = constraints,
    optimizer = optimizer,
    ftol = 1e-4,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 50,
    options = {
        "perturb_options": {
            "order": 2,
            "verbose": 0,
        },
        "solve_options": {
            "ftol": 1e-1,
            "xtol": 1e-2,
            "gtol": 1e-2,
            "verbose": 0,
        },
    },
    copy = False,
    verbose = 3,
)

eq_qs_C_FXD.save(TWO_TERM_FXD_PATH)










#========================================================================================================================================
# OPTIMIZING FOR TWO-TERM QH AT BOUNDARY SURFACE - FREE PRESSURE
#========================================================================================================================================
eq_qs_C_FREE = eq_init.copy()

free_objectives_C, free_constraints_C = build_free_extension(
    eq = eq_qs_C_FREE,
    eq_initial = eq_qs_C_FREE.copy(),
)

constraints = (
    ForceBalance(eq = eq_qs_C_FREE),
    FixBoundaryR(eq = eq_qs_C_FREE, modes = R_modes),
    FixBoundaryZ(eq = eq_qs_C_FREE, modes = Z_modes),
    FixIota(eq = eq_qs_C_FREE),
    FixPsi(eq = eq_qs_C_FREE),
) + tuple(free_constraints_C)

objective_fC_FREE = ObjectiveFunction(
    QuasisymmetryTwoTerm(
        eq = eq_qs_C_FREE,
        grid = grid_rho1,
        helicity = (1, eq_init.NFP),
    )
)

objective_fC_FREE = append_free_objectives(
    objective = objective_fC_FREE,
    free_objectives = free_objectives_C,
)

eq_qs_C_FREE, result_C_FREE = eq_qs_C_FREE.optimize(
    objective = objective_fC_FREE,
    constraints = constraints,
    optimizer = optimizer,
    ftol = 1e-4,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 50,
    options = {
        "perturb_options": {
            "order": 2,
            "verbose": 0,
        },
        "solve_options": {
            "ftol": 1e-1,
            "xtol": 1e-2,
            "gtol": 1e-2,
            "verbose": 0,
        },
    },
    copy = False,
    verbose = 3,
)

eq_qs_C_FREE.save(TWO_TERM_FREE_PATH)