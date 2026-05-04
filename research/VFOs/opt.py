# opt.py
#===================================================================================================================================================

from desc.objectives import (
    ObjectiveFunction,
    FixIota,
    FixPsi,
    FixPressure,
    FixBoundaryR,
    FixBoundaryZ,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    BallooningStability,
    MercierStability,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)

from helper import (
    _eq,
    _append_terms,
)

#===================================================================================================================================================










#===========================================================
# CORE OBJECTIVES
#===========================================================

OBJECTIVE_REGISTRY_CORE = {
    "forcebalance_obj": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "aspect_ratio": {
        "wrapper": AspectRatio,
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

#===================================================================================================================================================










#===========================================================
# CORE CONSTRAINTS
#===========================================================

CONSTRAINT_REGISTRY_CORE = {
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
    "fix_surfaceR": {
        "wrapper": FixBoundaryR,
        "defaults": {
            "eq": _eq,
        }
    },
    "fix_surfaceZ": {
        "wrapper": FixBoundaryZ,
        "defaults": {
            "eq": _eq,
        }
    },
}

#===================================================================================================================================================










#===========================================================
# FLO OBJECTIVES
#===========================================================

OBJECTIVE_REGISTRY_FLO = {
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

#===================================================================================================================================================










#===========================================================
# FLO CONSTRAINTS
#===========================================================

CONSTRAINT_REGISTRY_FLO = {
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

#===================================================================================================================================================










#===========================================================
# FNO OBJECTIVES
#===========================================================

OBJECTIVE_REGISTRY_FNO = {
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
    "FNO_pressure_positive": {
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

#===================================================================================================================================================










#===========================================================
# FNO CONSTRAINTS
#===========================================================

CONSTRAINT_REGISTRY_FNO = {
}

#===================================================================================================================================================










#===========================================================
# OBJECTIVE / CONSTRAINT BUILDER
#===========================================================

def build_objective_and_constraints(
        eq_0,
        optimizer,
        opt_config,
        formulation,
    ):
    """
    Build objective and constraint objects from config toggles.
    """

    context = {
        "eq_0": eq_0,
        "optimizer": optimizer,
        "formulation": formulation,
    }

    objectives_list = []
    constraints_list = []

    _append_terms(
        term_list = objectives_list,
        toggle = opt_config["opt_toggles_core"],
        registry = OBJECTIVE_REGISTRY_CORE,
        context = context,
        kind = "objective",
    )

    _append_terms(
        term_list = constraints_list,
        toggle = opt_config["opt_toggles_core"],
        registry = CONSTRAINT_REGISTRY_CORE,
        context = context,
        kind = "constraint",
    )

    if formulation == "FLO":
        _append_terms(
            term_list = objectives_list,
            toggle = opt_config["opt_toggles_FLO"],
            registry = OBJECTIVE_REGISTRY_FLO,
            context = context,
            kind = "objective",
        )

        _append_terms(
            term_list = constraints_list,
            toggle = opt_config["opt_toggles_FLO"],
            registry = CONSTRAINT_REGISTRY_FLO,
            context = context,
            kind = "constraint",
        )

    elif formulation == "FNO":
        _append_terms(
            term_list = objectives_list,
            toggle = opt_config["opt_toggles_FNO"],
            registry = OBJECTIVE_REGISTRY_FNO,
            context = context,
            kind = "objective",
        )

        _append_terms(
            term_list = constraints_list,
            toggle = opt_config["opt_toggles_FNO"],
            registry = CONSTRAINT_REGISTRY_FNO,
            context = context,
            kind = "constraint",
        )

    else:
        raise ValueError(
            f"Unknown formulation '{formulation}'. Expected 'FLO' or 'FNO'."
        )

    objective = ObjectiveFunction(objectives_list)
    constraints = tuple(constraints_list)

    return objective, constraints

#===================================================================================================================================================










#===========================================================
# OPTIMIZATION RUNNER
#===========================================================

def run_optimization(
        eq_0,
        optimizer,
        opt_config,
        formulation,
    ):
    """
    Run one DESC optimization.

    formulation = "FLO" runs core objectives/constraints plus FLO terms.
    formulation = "FNO" runs core objectives/constraints plus FNO terms.
    """

    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]
    max_nfev = opt_config["max_nfev"]
    x_scale = opt_config["x_scale"]

    objective, constraints = build_objective_and_constraints(
        eq_0 = eq_0,
        optimizer = optimizer,
        opt_config = opt_config,
        formulation = formulation,
    )

    eq_opt, result = eq_0.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        ftol = ftol,
        xtol = xtol,
        gtol = gtol,
        maxiter = maxiter,
        options = {
            "max_nfev": max_nfev,
        },
        x_scale = x_scale,
        copy = True,
        verbose = 3,
    )

    return eq_opt, result

#===================================================================================================================================================