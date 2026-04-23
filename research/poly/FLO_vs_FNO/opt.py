#===================================================================================================================================================
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
    _build_terms,
    _eq,
    _extract_result_message,
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
    "FNO_pressure": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "FNO_pressure_monotonic": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
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

FNO_CONSTRAINT_REGISTRY = {}
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
        run_status
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
        "plottable": False,
        "plotted": False,
        "failure_stage": None,
        "message": None,
    }

    try:
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

        result_message = _extract_result_message(opt_result)

        run_status["equilibrium_returned"] = eq_opt is not None
        run_status["message"] = result_message
        run_status["bad_approximation_failure"] = _has_bad_approximation_failure(
            result_message
        )
        run_status["plottable"] = (
            run_status["equilibrium_returned"]
            and not run_status["bad_approximation_failure"]
        )

        if run_status["bad_approximation_failure"]:
            run_status["failure_stage"] = "optimizer_result"

        return eq_opt, opt_result, run_status

    except Exception as e:
        error_message = repr(e)

        run_status["exception_raised"] = True
        run_status["message"] = error_message
        run_status["bad_approximation_failure"] = _has_bad_approximation_failure(
            error_message
        )
        run_status["plottable"] = False
        run_status["failure_stage"] = "optimizer_exception"

        return None, None, run_status
#===================================================================================================================================================