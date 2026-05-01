# opt.py

#===================================================================================================================================================
import contextlib
import io
import time
import traceback
import warnings
import sys
from desc.objectives import (
    BallooningStability,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    LinearObjectiveFromUser,
    MercierStability,
    ObjectiveFromUser,
    ObjectiveFunction,
    QuasisymmetryBoozer,
)

from .helper import (
    Tee,
    _build_terms,
    _eq,
    _extract_result_iterations,
    _extract_result_message,
    _has_automatic_continuation_failure,
    _has_bad_approximation_failure,
)
#===================================================================================================================================================











#============== REGISTRIES =========================================================================================================================
#===========================
CORE_OBJECTIVE_REGISTRY = {
    "forcebalance_obj": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "qs": {
        "wrapper": QuasisymmetryBoozer,
        "defaults": {
            "eq": _eq,
        },
    },
    "ballooning": {
        "wrapper": BallooningStability,
        "defaults": {
            "eq": _eq,
        },
    },
    "mercier": {
        "wrapper": MercierStability,
        "defaults": {
            "eq": _eq,
        },
    },
}
#===========================




#============================
CORE_CONSTRAINT_REGISTRY = {
    "forcebalance_con": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_pressure": {
        "wrapper": FixPressure,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_iota": {
        "wrapper": FixIota,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_psi": {
        "wrapper": FixPsi,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_boundary_R": {
        "wrapper": FixBoundaryR,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_boundary_Z": {
        "wrapper": FixBoundaryZ,
        "defaults": {
            "eq": _eq,
        },
    },
}
#============================




#===========================
# FLO custom registries:
FLO_OBJECTIVE_REGISTRY = {
    "FLO_pressure_axis": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_pressure_shape": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_iota_axis": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_iota_edge": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}

FLO_CONSTRAINT_REGISTRY = {
    "FLO_pressure_octic": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_pressure_DOF": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_pressure_edge": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_grad_pressure_edge": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_iota_quadratic": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#===========================




#===========================
# FNO custom registries:
FNO_OBJECTIVE_REGISTRY = {
    "FNO_pressure_axis": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FNO_pressure_edge": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    # "FNO_pressure_positive": {
    #     "wrapper": ObjectiveFromUser,
    #     "defaults": {
    #         "thing": _eq,
    #     },
    # },
    # "FNO_pressure_monotonic": {
    #     "wrapper": ObjectiveFromUser,
    #     "defaults": {
    #         "thing": _eq,
    #     },
    # },
    "FNO_grad_pressure_edge": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FNO_iota": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}

FNO_CONSTRAINT_REGISTRY = {
    "FLO_pressure_octic": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_pressure_DOF": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FLO_iota_quadratic": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#===========================
#===================================================================================================================================================











#============== OPTIMIZER FUNCTION =================================================================================================================
def run_optimization(
        eq_0,
        optimizer,
        opt_config,
        formulation,
    ):
    """
    Run one optimization using:
        - shared core toggle dictionary
        - either FLO or FNO toggle dictionary

    Returns:
        eq_opt,
        opt_result,
        run_status,
        optimization_log
    """
    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]
    max_nfev = opt_config["max_nfev"]
    x_scale = opt_config["x_scale"]

    objectives_list, constraints_list = _build_terms(
        eq_0 = eq_0,
        optimizer = optimizer,
        opt_config = opt_config,
        formulation = formulation,
        core_objective_registry = CORE_OBJECTIVE_REGISTRY,
        core_constraint_registry = CORE_CONSTRAINT_REGISTRY,
        FLO_objective_registry = FLO_OBJECTIVE_REGISTRY,
        FLO_constraint_registry = FLO_CONSTRAINT_REGISTRY,
        FNO_objective_registry = FNO_OBJECTIVE_REGISTRY,
        FNO_constraint_registry = FNO_CONSTRAINT_REGISTRY,
    )

    constraints = tuple(constraints_list)
    objectives = ObjectiveFunction(objectives_list)

    run_status = {
        "formulation": formulation,
        "optimizer": optimizer,
        "equilibrium_returned": False,
        "exception_raised": False,
        "bad_approximation_failure": False,
        "automatic_continuation_failure": False,
        "plottable": False,
        "plotted": False,
        "failure_stage": None,
        "message": None,
        "final_iterations": None,
        "runtime_seconds": None,
    }

    optimization_log_buffer = io.StringIO()
    eq_opt = None
    opt_result = None
    caught_warnings = []

    try:
        t0 = time.perf_counter()

        with warnings.catch_warnings(record = True) as caught_warnings:
            warnings.simplefilter("always")

            with contextlib.redirect_stdout(Tee(sys.stdout, optimization_log_buffer)), contextlib.redirect_stderr(Tee(sys.stderr, optimization_log_buffer)):
                eq_opt, opt_result = eq_0.optimize(
                    objective = objectives,
                    constraints = constraints,
                    optimizer = optimizer,
                    ftol = ftol,
                    xtol = xtol,
                    gtol = gtol,
                    maxiter = maxiter,
                    options = {"max_nfev": max_nfev},
                    x_scale = x_scale,
                    copy = True,
                    verbose = 3,
                )

        run_status["runtime_seconds"] = time.perf_counter() - t0

        optimization_log = optimization_log_buffer.getvalue()

        if len(caught_warnings) > 0:
            optimization_log += "\n\n# Python warnings captured during optimization\n"
            for warning_item in caught_warnings:
                optimization_log += (
                    f"{warning_item.category.__name__}: "
                    f"{warning_item.message}\n"
                )

        result_message = _extract_result_message(opt_result)
        result_iterations = _extract_result_iterations(opt_result)
        combined_message = "\n".join(
            [
                "" if result_message is None else str(result_message),
                optimization_log,
            ]
        )

        run_status["equilibrium_returned"] = eq_opt is not None
        run_status["message"] = result_message
        run_status["final_iterations"] = result_iterations
        run_status["bad_approximation_failure"] = _has_bad_approximation_failure(
            combined_message
        )
        run_status["automatic_continuation_failure"] = _has_automatic_continuation_failure(
            combined_message
        )
        run_status["plottable"] = (
            run_status["equilibrium_returned"]
            and not run_status["bad_approximation_failure"]
            and not run_status["automatic_continuation_failure"]
        )

        if run_status["automatic_continuation_failure"]:
            run_status["failure_stage"] = "optimizer_log"

        elif run_status["bad_approximation_failure"]:
            run_status["failure_stage"] = "optimizer_result"

        return eq_opt, opt_result, run_status, optimization_log

    except Exception:
        run_status["runtime_seconds"] = time.perf_counter() - t0
        run_status["exception_raised"] = True
        run_status["failure_stage"] = "optimizer_exception"

        optimization_log = optimization_log_buffer.getvalue()

        if len(caught_warnings) > 0:
            optimization_log += "\n\n# Python warnings captured during optimization\n"
            for warning_item in caught_warnings:
                optimization_log += (
                    f"{warning_item.category.__name__}: "
                    f"{warning_item.message}\n"
                )

        error_trace = traceback.format_exc()
        optimization_log += "\n\n# Exception traceback\n"
        optimization_log += error_trace

        run_status["message"] = error_trace.strip().splitlines()[-1]
        run_status["bad_approximation_failure"] = _has_bad_approximation_failure(
            optimization_log
        )
        run_status["automatic_continuation_failure"] = _has_automatic_continuation_failure(
            optimization_log
        )
        run_status["plottable"] = False

        return None, None, run_status, optimization_log
#===================================================================================================================================================