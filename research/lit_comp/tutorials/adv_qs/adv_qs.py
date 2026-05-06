# adv_qs.py
"""Advanced quasi-symmetry optimization.

Write-up of the DESC Advanced QS Optimization tutorial, minus plotting.
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
    iter_tolerance_cases,
    prepare_tolerance_case_dir,
    print_tolerance_case_header,
    save_optimization_result,
    should_run_tolerance_case,
)










#========================================================================================================================================
# PATHS
#========================================================================================================================================
fname = "adv_qs"

OUTPUT_DIR = Path(__file__).resolve().parent










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
MULTIGRID_MAXITER = 200
MULTIGRID_X_SCALE = "auto"
MULTIGRID_BASE_OPTIONS = {}

AUGLAG_OPTIMIZER = "lsq-auglag"
AUGLAG_MAXITER = 200
AUGLAG_X_SCALE = "auto"
AUGLAG_BASE_OPTIONS = {}

OUTER_FTOL_ORDERS = [2,4,6]

OUTER_XTOL_ORDERS = [6]

OUTER_GTOL_ORDERS = [3,6]

INNER_FTOL_ORDERS = [4,6]

INNER_XTOL_ORDERS = [6]

INNER_GTOL_ORDERS = [6]

INITIAL_TRUST_RATIO_ORDERS = [3]










#========================================================================================================================================
# TOLERANCE HELPERS
#========================================================================================================================================
def build_multigrid_options(
        tolerance_case,
    ):
    """
    Build proximal-lsq-exact options for one tolerance case.
    """

    options = deepcopy(MULTIGRID_BASE_OPTIONS)

    outer_tolerances = tolerance_case["outer_tolerances"]
    inner_tolerances = tolerance_case["inner_tolerances"]

    options["initial_trust_ratio"] = outer_tolerances["initial_trust_ratio"]

    solve_options = options.get(
        "solve_options",
        {},
    )

    if solve_options is None:
        solve_options = {}

    else:
        solve_options = deepcopy(solve_options)

    solve_options["ftol"] = inner_tolerances["ftol"]
    solve_options["xtol"] = inner_tolerances["xtol"]
    solve_options["gtol"] = inner_tolerances["gtol"]

    options["solve_options"] = solve_options

    return options





def build_auglag_options(
        tolerance_case,
    ):
    """
    Build lsq-auglag options for one tolerance case.

    The tolerance case is still recorded in the result summary, but
    initial_trust_ratio is not passed to lsq-auglag options.
    """

    options = deepcopy(AUGLAG_BASE_OPTIONS)

    return options










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
        tolerance_case,
        options,
    ):
    """
    Save multigrid histories using the same sidecar files as the other tutorials.
    """

    result = bundle_optimization_histories(
        histories = histories,
    )

    tolerances = tolerance_case["tolerances"]

    return save_optimization_result(
        result = result,
        output_path = output_path,
        label = label,
        optimizer = MULTIGRID_OPTIMIZER,
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = MULTIGRID_MAXITER,
        options = options,
        x_scale = MULTIGRID_X_SCALE,
        tolerance_case = tolerance_case,
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
        tolerance_case,
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

    tolerances = tolerance_case["tolerances"]

    options = build_multigrid_options(
        tolerance_case = tolerance_case,
    )

    eq_new, history = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = MULTIGRID_MAXITER,
        verbose = 3,
        copy = True,
        options = options,
        x_scale = MULTIGRID_X_SCALE,
    )

    return eq_new, history





def run_multigrid_sequence(
        eq_initial,
        output_path,
        label,
        tolerance_case,
        free_pressure = False,
    ):
    """
    Run the three-step multigrid QH optimization.
    """

    eq_multigrid_0 = eq_initial.copy()
    eqfam_multigrid = EquilibriaFamily(eq_multigrid_0)

    eq_multigrid_1, history_multigrid_1 = run_qh_step(
        1,
        eq_multigrid_0,
        tolerance_case = tolerance_case,
        free_pressure = free_pressure,
    )

    eqfam_multigrid.append(eq_multigrid_1)

    eq_multigrid_2, history_multigrid_2 = run_qh_step(
        2,
        eq_multigrid_1,
        tolerance_case = tolerance_case,
        free_pressure = free_pressure,
    )

    eqfam_multigrid.append(eq_multigrid_2)

    eq_multigrid_3, history_multigrid_3 = run_qh_step(
        3,
        eq_multigrid_2,
        tolerance_case = tolerance_case,
        free_pressure = free_pressure,
    )

    eqfam_multigrid.append(eq_multigrid_3)

    eqfam_multigrid.save(
        str(output_path)
    )

    options = build_multigrid_options(
        tolerance_case = tolerance_case,
    )

    save_multigrid_result(
        histories = (
            history_multigrid_1,
            history_multigrid_2,
            history_multigrid_3,
        ),
        output_path = output_path,
        label = label,
        tolerance_case = tolerance_case,
        options = options,
    )

    return eqfam_multigrid, (
        history_multigrid_1,
        history_multigrid_2,
        history_multigrid_3,
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
        bounds = (
            0.18,
            0.22,
        ),
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
                bounds = (
                    7,
                    9,
                ),
            ),
            Elongation(
                eq = eq,
                bounds = (
                    0,
                    3,
                ),
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
                modes = [
                    0,
                    0,
                    0,
                ],
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
                bounds = (
                    7,
                    9,
                ),
            ),
            Elongation(
                eq = eq,
                bounds = (
                    0,
                    3,
                ),
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
                modes = [
                    0,
                    0,
                    0,
                ],
            ),
            FixPressure(eq = eq),
            FixCurrent(eq = eq),
            FixPsi(eq = eq),
        )

    return constraints, free_objectives





def run_constrained_optimization(
        eq,
        tolerance_case,
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

    tolerances = tolerance_case["tolerances"]

    options = build_auglag_options(
        tolerance_case = tolerance_case,
    )

    eq_new, history = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = AUGLAG_MAXITER,
        copy = True,
        verbose = 3,
        options = options,
        x_scale = AUGLAG_X_SCALE,
    )

    eq_new.solve()

    return eq_new, history, options










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
        base_options = {},
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

    MULTIGRID_FXD_PATH = case_dir / f"{fname}_multigrid_FXD.h5"

    MULTIGRID_FREE_PATH = case_dir / f"{fname}_multigrid_FREE.h5"

    AUGLAG_FXD_PATH = case_dir / f"{fname}_auglag_FXD.h5"

    AUGLAG_FREE_PATH = case_dir / f"{fname}_auglag_FREE.h5"

    eq0.save(
        str(INITIAL_PATH)
    )

    print("")
    print(f"Tolerance report written to: {tolerance_report_path}")
    print("")

    tolerances = tolerance_case["tolerances"]




    #========================================================================================================================================
    # RUN MULTIGRID STEPS - FXD PRESSURE
    #========================================================================================================================================
    eqfam_multigrid_FXD, histories_multigrid_FXD = run_multigrid_sequence(
        eq_initial = eq0,
        output_path = MULTIGRID_FXD_PATH,
        label = "adv_qs_multigrid_FXD",
        tolerance_case = tolerance_case,
        free_pressure = False,
    )




    #========================================================================================================================================
    # RUN MULTIGRID STEPS - FREE PRESSURE
    #========================================================================================================================================
    eqfam_multigrid_FREE, histories_multigrid_FREE = run_multigrid_sequence(
        eq_initial = eq0,
        output_path = MULTIGRID_FREE_PATH,
        label = "adv_qs_multigrid_FREE",
        tolerance_case = tolerance_case,
        free_pressure = True,
    )




    #========================================================================================================================================
    # RUN CONSTRAINED OPTIMIZATION - FXD PRESSURE
    #========================================================================================================================================
    eq_auglag_FXD = eq0.copy()

    eq_auglag_FXD, history_auglag_FXD, options_auglag_FXD = run_constrained_optimization(
        eq = eq_auglag_FXD,
        tolerance_case = tolerance_case,
        free_pressure = False,
    )

    eq_auglag_FXD.save(
        str(AUGLAG_FXD_PATH)
    )

    save_optimization_result(
        result = history_auglag_FXD,
        output_path = AUGLAG_FXD_PATH,
        label = "adv_qs_auglag_FXD",
        optimizer = AUGLAG_OPTIMIZER,
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = AUGLAG_MAXITER,
        options = options_auglag_FXD,
        x_scale = AUGLAG_X_SCALE,
        tolerance_case = tolerance_case,
    )




    #========================================================================================================================================
    # RUN CONSTRAINED OPTIMIZATION - FREE PRESSURE
    #========================================================================================================================================
    eq_auglag_FREE = eq0.copy()

    eq_auglag_FREE, history_auglag_FREE, options_auglag_FREE = run_constrained_optimization(
        eq = eq_auglag_FREE,
        tolerance_case = tolerance_case,
        free_pressure = True,
    )

    eq_auglag_FREE.save(
        str(AUGLAG_FREE_PATH)
    )

    save_optimization_result(
        result = history_auglag_FREE,
        output_path = AUGLAG_FREE_PATH,
        label = "adv_qs_auglag_FREE",
        optimizer = AUGLAG_OPTIMIZER,
        ftol = tolerances["ftol"],
        xtol = tolerances["xtol"],
        gtol = tolerances["gtol"],
        maxiter = AUGLAG_MAXITER,
        options = options_auglag_FREE,
        x_scale = AUGLAG_X_SCALE,
        tolerance_case = tolerance_case,
    )




if not ran_tolerance_case:
    requested_sweep_index = os.environ.get(
        "DESC_SWEEP_INDEX",
        None,
    )

    raise ValueError(
        f"No adv_qs tolerance case matched DESC_SWEEP_INDEX = {requested_sweep_index}."
    )
