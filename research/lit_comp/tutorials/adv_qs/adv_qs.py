# adv_qs.py
"""Advanced quasi-symmetry optimization.

Write-up of the DESC Advanced QS Optimization tutorial, minus plotting.
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

from helper import (
    append_free_objectives,
    build_free_extension,
    get_final_cost,
    get_result_value,
    save_optimization_result,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "adv_qs"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MULTIGRID_FXD_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_multigrid_FXD.h5",
)

MULTIGRID_FREE_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_multigrid_FREE.h5",
)

AUGLAG_FXD_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_auglag_FXD.h5",
)

AUGLAG_FREE_PATH = os.path.join(
    OUTPUT_DIR,
    f"{fname}_auglag_FREE.h5",
)










#========================================================================================================================================
# INITIAL GUESS
#========================================================================================================================================
surf = FourierRZToroidalSurface(
    R_lmn = [1, 0.125, 0.1],
    Z_lmn = [-0.125, -0.1],
    modes_R = [
        [0, 0],
        [1, 0],
        [0, 1],
    ],
    modes_Z = [
        [-1, 0],
        [0, -1],
    ],
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











#========================================================================================================================================
# OPTIMIZATION HYPERPARAMETERS
#========================================================================================================================================
MULTIGRID_OPTIMIZER = "proximal-lsq-exact"
MULTIGRID_MAXITER = 20
MULTIGRID_FTOL = None
MULTIGRID_XTOL = None
MULTIGRID_GTOL = None
MULTIGRID_X_SCALE = "auto"
MULTIGRID_OPTIONS = {
    "initial_trust_ratio": 0.1,
}

AUGLAG_OPTIMIZER = "lsq-auglag"
AUGLAG_MAXITER = 200
AUGLAG_FTOL = None
AUGLAG_XTOL = None
AUGLAG_GTOL = None
AUGLAG_X_SCALE = "auto"
AUGLAG_OPTIONS = {}










#========================================================================================================================================
# RESULT HELPERS
#========================================================================================================================================
def sum_history_value(
        histories,
        key,
    ):
    """
    Sum scalar optimization-result counters across a multigrid sequence.
    """

    values = []

    for history in histories:
        value = get_result_value(
            result = history,
            key = key,
        )

        if value is not None:
            values.append(value)

    if len(values) == 0:
        return None

    try:
        return int(
            np.sum(values)
        )

    except Exception:
        return None





def bundle_optimization_histories(
        histories,
    ):
    """
    Bundle multigrid step histories into one save-compatible result object.
    """

    histories = tuple(histories)
    final_history = histories[-1]

    return {
        "cost": get_final_cost(
            result = final_history,
        ),
        "message": get_result_value(
            result = final_history,
            key = "message",
        ),
        "termination_message": get_result_value(
            result = final_history,
            key = "termination_message",
        ),
        "success": get_result_value(
            result = final_history,
            key = "success",
        ),
        "status": get_result_value(
            result = final_history,
            key = "status",
        ),
        "nfev": sum_history_value(
            histories = histories,
            key = "nfev",
        ),
        "njev": sum_history_value(
            histories = histories,
            key = "njev",
        ),
        "nit": sum_history_value(
            histories = histories,
            key = "nit",
        ),
        "optimality": get_result_value(
            result = final_history,
            key = "optimality",
        ),
        "steps": histories,
    }





def save_multigrid_result(
        histories,
        output_path,
        label,
    ):
    """
    Save multigrid histories using the same sidecar files as the other tutorials.
    """

    result = bundle_optimization_histories(
        histories = histories,
    )

    return save_optimization_result(
        result = result,
        output_path = output_path,
        label = label,
        optimizer = MULTIGRID_OPTIMIZER,
        ftol = MULTIGRID_FTOL,
        xtol = MULTIGRID_XTOL,
        gtol = MULTIGRID_GTOL,
        maxiter = MULTIGRID_MAXITER,
        options = MULTIGRID_OPTIONS,
        x_scale = MULTIGRID_X_SCALE,
    )










#========================================================================================================================================
# MULTIGRID HELPERS
#========================================================================================================================================
def build_qh_grid(
        eq,
    ):
    """
    Build the shared QH objective grid for the multigrid optimization.
    """

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        rho = np.array(
            [
                0.6,
                0.8,
                1.0,
            ]
        ),
        sym = True,
    )

    return grid





def build_qh_modes(
        k,
        eq,
    ):
    """
    Build boundary modes fixed during a multigrid step.
    """

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

    return R_modes, Z_modes





def build_qh_objective(
        eq,
        free_objectives = (),
    ):
    """
    Build the QH objective for one multigrid step.
    """

    grid = build_qh_grid(
        eq = eq,
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

    objective = append_free_objectives(
        objective = objective,
        free_objectives = free_objectives,
    )

    return objective





def build_qh_constraints(
        k,
        eq,
        free_pressure = False,
    ):
    """
    Build constraints for one multigrid step.
    """

    R_modes, Z_modes = build_qh_modes(
        k = k,
        eq = eq,
    )

    if free_pressure:
        free_objectives, free_constraints = build_free_extension(
            eq = eq,
            eq_initial = eq.copy(),
        )

        constraints = (
            ForceBalance(eq = eq),
            FixBoundaryR(eq = eq, modes = R_modes),
            FixBoundaryZ(eq = eq, modes = Z_modes),
            FixCurrent(eq = eq),
            FixPsi(eq = eq),
        ) + tuple(free_constraints)

    else:
        free_objectives = ()

        constraints = (
            ForceBalance(eq = eq),
            FixBoundaryR(eq = eq, modes = R_modes),
            FixBoundaryZ(eq = eq, modes = Z_modes),
            FixPressure(eq = eq),
            FixCurrent(eq = eq),
            FixPsi(eq = eq),
        )

    return constraints, free_objectives





def run_qh_step(
        k,
        eq,
        free_pressure = False,
    ):
    """
    Run one step of the precise QH optimization example from Landreman & Paul.
    """

    constraints, free_objectives = build_qh_constraints(
        k = k,
        eq = eq,
        free_pressure = free_pressure,
    )

    objective = build_qh_objective(
        eq = eq,
        free_objectives = free_objectives,
    )

    optimizer = Optimizer(MULTIGRID_OPTIMIZER)

    eq_new, history = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        maxiter = MULTIGRID_MAXITER,
        verbose = 3,
        copy = True,
        options = MULTIGRID_OPTIONS,
        x_scale = MULTIGRID_X_SCALE,
    )

    return eq_new, history










#========================================================================================================================================
# RUN MULTIGRID STEPS - FXD PRESSURE
#========================================================================================================================================
eq_multigrid_FXD_0 = eq0.copy()
eqfam_multigrid_FXD = EquilibriaFamily(eq_multigrid_FXD_0)

eq_multigrid_FXD_1, history_multigrid_FXD_1 = run_qh_step(
    1,
    eq_multigrid_FXD_0,
    free_pressure = False,
)

eqfam_multigrid_FXD.append(eq_multigrid_FXD_1)

eq_multigrid_FXD_2, history_multigrid_FXD_2 = run_qh_step(
    2,
    eq_multigrid_FXD_1,
    free_pressure = False,
)

eqfam_multigrid_FXD.append(eq_multigrid_FXD_2)

eq_multigrid_FXD_3, history_multigrid_FXD_3 = run_qh_step(
    3,
    eq_multigrid_FXD_2,
    free_pressure = False,
)

eqfam_multigrid_FXD.append(eq_multigrid_FXD_3)

eqfam_multigrid_FXD.save(MULTIGRID_FXD_PATH)

save_multigrid_result(
    histories = (
        history_multigrid_FXD_1,
        history_multigrid_FXD_2,
        history_multigrid_FXD_3,
    ),
    output_path = MULTIGRID_FXD_PATH,
    label = "adv_qs_multigrid_FXD",
)










#========================================================================================================================================
# RUN MULTIGRID STEPS - FREE PRESSURE
#========================================================================================================================================
eq_multigrid_FREE_0 = eq0.copy()
eqfam_multigrid_FREE = EquilibriaFamily(eq_multigrid_FREE_0)

eq_multigrid_FREE_1, history_multigrid_FREE_1 = run_qh_step(
    1,
    eq_multigrid_FREE_0,
    free_pressure = True,
)

eqfam_multigrid_FREE.append(eq_multigrid_FREE_1)

eq_multigrid_FREE_2, history_multigrid_FREE_2 = run_qh_step(
    2,
    eq_multigrid_FREE_1,
    free_pressure = True,
)

eqfam_multigrid_FREE.append(eq_multigrid_FREE_2)

eq_multigrid_FREE_3, history_multigrid_FREE_3 = run_qh_step(
    3,
    eq_multigrid_FREE_2,
    free_pressure = True,
)

eqfam_multigrid_FREE.append(eq_multigrid_FREE_3)

eqfam_multigrid_FREE.save(MULTIGRID_FREE_PATH)

save_multigrid_result(
    histories = (
        history_multigrid_FREE_1,
        history_multigrid_FREE_2,
        history_multigrid_FREE_3,
    ),
    output_path = MULTIGRID_FREE_PATH,
    label = "adv_qs_multigrid_FREE",
)










#========================================================================================================================================
# CONSTRAINED OPTIMIZATION HELPERS
#========================================================================================================================================
def build_constrained_grid(
        eq,
    ):
    """
    Build the shared QH objective grid for the constrained optimization.
    """

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        rho = np.array(
            [
                0.6,
                0.8,
                1.0,
            ]
        ),
        sym = True,
    )

    return grid





def build_constrained_objective(
        eq,
        free_objectives = (),
    ):
    """
    Build the constrained-optimization QH objective.
    """

    grid = build_constrained_grid(
        eq = eq,
    )

    objective = ObjectiveFunction(
        (
            GenericObjective(
                f = "f_C",
                thing = eq,
                grid = grid,
                compute_kwargs = {
                    "helicity": (1, eq.NFP),
                },
                name = "QS Two-Term",
            ),
        ),
    )

    objective = append_free_objectives(
        objective = objective,
        free_objectives = free_objectives,
    )

    return objective





def fun_mirror_ratio(
        grid,
        data,
    ):
    """
    Compute the mirror ratio from |B|.
    """

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





def build_mirror_ratio_objective(
        eq,
    ):
    """
    Build the mirror-ratio user objective for the constrained optimization.
    """

    obj_mirror_ratio = ObjectiveFromUser(
        fun = fun_mirror_ratio,
        thing = eq,
        grid = LinearGrid(
            rho = 1.0,
            M = eq.M_grid,
            N = eq.N_grid,
            NFP = eq.NFP,
        ),
        bounds = (0.18, 0.22),
        name = "my mirror ratio",
    )

    return obj_mirror_ratio





def build_constrained_constraints(
        eq,
        free_pressure = False,
    ):
    """
    Build the constrained-optimization constraints.
    """

    obj_mirror_ratio = build_mirror_ratio_objective(
        eq = eq,
    )

    if free_pressure:
        free_objectives, free_constraints = build_free_extension(
            eq = eq,
            eq_initial = eq.copy(),
        )

        constraints = (
            ForceBalance(eq = eq),
            AspectRatio(
                eq = eq,
                bounds = (7, 9),
            ),
            Elongation(
                eq = eq,
                bounds = (0, 3),
            ),
            Volume(
                eq = eq,
                target = eq.compute("V")["V"],
            ),
            RotationalTransform(
                eq = eq,
                target = 1.1,
                loss_function = "mean",
            ),
            obj_mirror_ratio,
            FixBoundaryR(
                eq = eq,
                modes = [0, 0, 0],
            ),
            FixCurrent(eq = eq),
            FixPsi(eq = eq),
        ) + tuple(free_constraints)

    else:
        free_objectives = ()

        constraints = (
            ForceBalance(eq = eq),
            AspectRatio(
                eq = eq,
                bounds = (7, 9),
            ),
            Elongation(
                eq = eq,
                bounds = (0, 3),
            ),
            Volume(
                eq = eq,
                target = eq.compute("V")["V"],
            ),
            RotationalTransform(
                eq = eq,
                target = 1.1,
                loss_function = "mean",
            ),
            obj_mirror_ratio,
            FixBoundaryR(
                eq = eq,
                modes = [0, 0, 0],
            ),
            FixPressure(eq = eq),
            FixCurrent(eq = eq),
            FixPsi(eq = eq),
        )

    return constraints, free_objectives





def run_constrained_optimization(
        eq,
        free_pressure = False,
    ):
    """
    Run the augmented-Lagrangian constrained optimization.
    """

    constraints, free_objectives = build_constrained_constraints(
        eq = eq,
        free_pressure = free_pressure,
    )

    objective = build_constrained_objective(
        eq = eq,
        free_objectives = free_objectives,
    )

    optimizer = Optimizer(AUGLAG_OPTIMIZER)

    eq_new, history = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        maxiter = AUGLAG_MAXITER,
        copy = True,
        verbose = 3,
        options = AUGLAG_OPTIONS,
        x_scale = AUGLAG_X_SCALE,
    )

    eq_new.solve()

    return eq_new, history










#========================================================================================================================================
# RUN CONSTRAINED OPTIMIZATION - FXD PRESSURE
#========================================================================================================================================
eq_auglag_FXD = eq0.copy()

eq_auglag_FXD, history_auglag_FXD = run_constrained_optimization(
    eq = eq_auglag_FXD,
    free_pressure = False,
)

eq_auglag_FXD.save(AUGLAG_FXD_PATH)

save_optimization_result(
    result = history_auglag_FXD,
    output_path = AUGLAG_FXD_PATH,
    label = "adv_qs_auglag_FXD",
    optimizer = AUGLAG_OPTIMIZER,
    ftol = AUGLAG_FTOL,
    xtol = AUGLAG_XTOL,
    gtol = AUGLAG_GTOL,
    maxiter = AUGLAG_MAXITER,
    options = AUGLAG_OPTIONS,
    x_scale = AUGLAG_X_SCALE,
)










#========================================================================================================================================
# RUN CONSTRAINED OPTIMIZATION - FREE PRESSURE
#========================================================================================================================================
eq_auglag_FREE = eq0.copy()

eq_auglag_FREE, history_auglag_FREE = run_constrained_optimization(
    eq = eq_auglag_FREE,
    free_pressure = True,
)

eq_auglag_FREE.save(AUGLAG_FREE_PATH)

save_optimization_result(
    result = history_auglag_FREE,
    output_path = AUGLAG_FREE_PATH,
    label = "adv_qs_auglag_FREE",
    optimizer = AUGLAG_OPTIMIZER,
    ftol = AUGLAG_FTOL,
    xtol = AUGLAG_XTOL,
    gtol = AUGLAG_GTOL,
    maxiter = AUGLAG_MAXITER,
    options = AUGLAG_OPTIONS,
    x_scale = AUGLAG_X_SCALE,
)
