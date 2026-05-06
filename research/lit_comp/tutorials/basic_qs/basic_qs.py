# basic_qs.py
"""Basic quasi-symmetry optimization.

Write-up of the DESC Basic QS Optimization tutorial, minus plotting.
"""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from copy import deepcopy
from pathlib import Path
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
    iter_tolerance_cases,
    optimize_save_report,
    prepare_tolerance_case_dir,
    print_tolerance_case_header,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "basic_qs"

OUTPUT_DIR = Path(__file__).resolve().parent










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










#===========================================================================================================================
# CHECK INITIAL PRESSURE
#=============================================================================================================================
rho = np.linspace(
    0.0,
    1.0,
    11,
)

grid_pressure = LinearGrid(
    rho = rho,
    M = 0,
    N = 0,
    NFP = eq_init.NFP,
    sym = eq_init.sym,
)

print("")
print("Initial pressure object:")
print(eq_init.pressure)

print("")
print("Initial pressure coefficients p_l:")
print(np.asarray(eq_init.params_dict["p_l"]))

print("")
print("Initial pressure P(rho):")
p_rho = np.asarray(eq_init.pressure.compute(grid_pressure))
for rho_i, p_i in zip(grid_pressure.nodes[:, 0], p_rho):
    print(f"rho = {rho_i:.3f}, P = {p_i:.8e}")

print("")
print("Initial pressure gradient dP/drho:")
dp_drho = np.asarray(eq_init.pressure.compute(grid_pressure, dr = 1))
for rho_i, dp_i in zip(grid_pressure.nodes[:, 0], dp_drho):
    print(f"rho = {rho_i:.3f}, dP/drho = {dp_i:.8e}")










#========================================================================================================================================
# OPTIMIZER
#========================================================================================================================================
optimizer = Optimizer("proximal-lsq-exact")
maxiter = 100
x_scale = "auto"

FTOL_ORDERS = range(
    4,
    7,
)

XTOL_ORDERS = range(
    5,
    7,
)

GTOL_ORDERS = range(
    5,
    7,
)

INITIAL_TRUST_RATIO_ORDERS = range(
    1,
    3,
)

BASE_OPTIONS_T = {
    "perturb_options": {
        "order": 2,
        "verbose": 0,
    },
    "max_nfev": 100,
    "solve_options": {
        "maxiter": maxiter,
        "verbose": 0,
    },
}

BASE_OPTIONS_C = {
    "perturb_options": {
        "order": 2,
        "verbose": 0,
    },
    "max_nfev": 100,
    "solve_options": {
        "maxiter": maxiter,
        "verbose": 0,
    },
}










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
# TOLERANCE HELPERS
#========================================================================================================================================
def build_basic_qs_options(
        base_options,
        tolerance_case,
    ):
    """
    Build DESC optimizer options for one basic_qs tolerance case.
    """

    options = deepcopy(base_options)

    tolerances = tolerance_case["tolerances"]

    options["initial_trust_ratio"] = tolerances["initial_trust_ratio"]

    options["solve_options"]["ftol"] = tolerances["ftol"]
    options["solve_options"]["xtol"] = tolerances["xtol"]
    options["solve_options"]["gtol"] = tolerances["gtol"]

    return options










#========================================================================================================================================
# TOLERANCE SWEEP
#========================================================================================================================================
for tolerance_case in iter_tolerance_cases(
        ftol_orders = FTOL_ORDERS,
        xtol_orders = XTOL_ORDERS,
        gtol_orders = GTOL_ORDERS,
        initial_trust_ratio_orders = INITIAL_TRUST_RATIO_ORDERS,
        base_options = {},
    ):

    case_dir, tolerance_report_path = prepare_tolerance_case_dir(
        output_dir = OUTPUT_DIR,
        tolerance_case = tolerance_case,
    )

    print_tolerance_case_header(
        tolerance_case = tolerance_case,
        case_dir = case_dir,
    )

    INITIAL_PATH = case_dir / f"{fname}_initial.h5"

    TRIPLE_PRODUCT_FXD_PATH = case_dir / f"{fname}_T_FXD.h5"

    TRIPLE_PRODUCT_FREE_PATH = case_dir / f"{fname}_T_FREE.h5"

    TWO_TERM_FXD_PATH = case_dir / f"{fname}_C_FXD.h5"

    TWO_TERM_FREE_PATH = case_dir / f"{fname}_C_FREE.h5"

    eq_init.save(
        str(INITIAL_PATH)
    )

    print("")
    print(f"Tolerance report written to: {tolerance_report_path}")
    print("")

    tolerances = tolerance_case["tolerances"]

    options_T = build_basic_qs_options(
        base_options = BASE_OPTIONS_T,
        tolerance_case = tolerance_case,
    )

    options_C = build_basic_qs_options(
        base_options = BASE_OPTIONS_C,
        tolerance_case = tolerance_case,
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

    eq_qs_T_FXD, result_T_FXD = optimize_save_report(
        eq = eq_qs_T_FXD,
        objective = objective_fT,
        constraints = constraints,
        optimizer = optimizer,
        output_path = TRIPLE_PRODUCT_FXD_PATH,
        label = "basic_qs_T_FXD",
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = maxiter,
        options = options_T,
        copy = False,
        verbose = 3,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
    )




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

    eq_qs_T_FREE, result_T_FREE = optimize_save_report(
        eq = eq_qs_T_FREE,
        objective = objective_fT_FREE,
        constraints = constraints,
        optimizer = optimizer,
        output_path = TRIPLE_PRODUCT_FREE_PATH,
        label = "basic_qs_T_FREE",
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = maxiter,
        options = options_T,
        copy = False,
        verbose = 3,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
    )




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

    eq_qs_C_FXD, result_C_FXD = optimize_save_report(
        eq = eq_qs_C_FXD,
        objective = objective_fC,
        constraints = constraints,
        optimizer = optimizer,
        output_path = TWO_TERM_FXD_PATH,
        label = "basic_qs_C_FXD",
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = maxiter,
        options = options_C,
        copy = False,
        verbose = 3,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
    )




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

    eq_qs_C_FREE, result_C_FREE = optimize_save_report(
        eq = eq_qs_C_FREE,
        objective = objective_fC_FREE,
        constraints = constraints,
        optimizer = optimizer,
        output_path = TWO_TERM_FREE_PATH,
        label = "basic_qs_C_FREE",
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = maxiter,
        options = options_C,
        copy = False,
        verbose = 3,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
    )