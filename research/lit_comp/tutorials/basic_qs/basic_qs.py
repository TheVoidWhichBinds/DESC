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










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "basic_qs"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRIPLE_PRODUCT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_triple_product.h5",
)

TWO_TERM_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_two_term.h5",
)










#========================================================================================================================================
# INITIAL GUESS
#========================================================================================================================================
eq_init = desc.io.load("qs_initial_guess.h5")










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
# OPTIMIZING FOR TRIPLE PRODUCT QS IN VOLUME
#========================================================================================================================================
eq_qs_T = eq_init.copy()

constraints = (
    ForceBalance(eq = eq_qs_T),
    FixBoundaryR(eq = eq_qs_T, modes = R_modes),
    FixBoundaryZ(eq = eq_qs_T, modes = Z_modes),
    FixPressure(eq = eq_qs_T),
    FixIota(eq = eq_qs_T),
    FixPsi(eq = eq_qs_T),
)

grid_vol = ConcentricGrid(
    L = eq_init.L_grid,
    M = eq_init.M_grid,
    N = eq_init.N_grid,
    NFP = eq_init.NFP,
    sym = eq_init.sym,
)

objective_fT = ObjectiveFunction(
    QuasisymmetryTripleProduct(
        eq = eq_qs_T,
        grid = grid_vol,
    )
)

eq_qs_T, result_T = eq_qs_T.optimize(
    objective = objective_fT,
    constraints = constraints,
    optimizer = optimizer,
    ftol = 5e-2,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 50,
    options = {
        "perturb_options": {
            "order": 2,
            "verbose": 0,
        },
        "solve_options": {
            "ftol": 5e-3,
            "xtol": 1e-6,
            "gtol": 1e-6,
            "verbose": 0,
        },
    },
    copy = False,
    verbose = 3,
)

eq_qs_T.save(TRIPLE_PRODUCT_PATH)










#========================================================================================================================================
# OPTIMIZING FOR TWO-TERM QH AT BOUNDARY SURFACE
#========================================================================================================================================
eq_qs_C = eq_init.copy()

constraints = (
    ForceBalance(eq = eq_qs_C),
    FixBoundaryR(eq = eq_qs_C, modes = R_modes),
    FixBoundaryZ(eq = eq_qs_C, modes = Z_modes),
    FixPressure(eq = eq_qs_C),
    FixIota(eq = eq_qs_C),
    FixPsi(eq = eq_qs_C),
)

grid_rho1 = LinearGrid(
    M = eq_init.M_grid,
    N = eq_init.N_grid,
    NFP = eq_init.NFP,
    sym = eq_init.sym,
    rho = np.array(1.0),
)

objective_fC = ObjectiveFunction(
    QuasisymmetryTwoTerm(
        eq = eq_qs_C,
        grid = grid_rho1,
        helicity = (1, eq_init.NFP),
    )
)

eq_qs_C, result_C = eq_qs_C.optimize(
    objective = objective_fC,
    constraints = constraints,
    optimizer = optimizer,
    ftol = 1e-2,
    xtol = 1e-6,
    gtol = 1e-6,
    maxiter = 50,
    options = {
        "perturb_options": {
            "order": 2,
            "verbose": 0,
        },
        "solve_options": {
            "ftol": 1e-2,
            "xtol": 1e-6,
            "gtol": 1e-6,
            "verbose": 0,
        },
    },
    copy = False,
    verbose = 3,
)

eq_qs_C.save(TWO_TERM_PATH)