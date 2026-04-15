import sys
sys.path.append("/Users/macdaddi/DESC")

from desc.objectives import (
    ObjectiveFunction,
    FixPsi,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    BallooningStability,
    MercierStability,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)

from .helper import (
    _eq,
    _append_terms,
)










#============== REGISTRIES ============================================================================================================================
#======================
OBJECTIVE_REGISTRY = {
        # Standard:
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

        # Custom:
    "pressure_axis_range": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_shape": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_axis_range": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_edge_range": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_monotone_obj": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_positive_obj": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_edge_obj": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_pressure_edge_obj": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#----------------------


#=======================
CONSTRAINT_REGISTRY = {
        # Standard:
    "forcebalance_con": {
        "wrapper": ForceBalance,
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

        # Custom:
    "pressure_octic_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_DOF_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_edge_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_pressure_edge_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#======================
#==============================================================================================================================================================










#============== OPTIMIZER FUNCTION ============================================================================================================================
def run_optimization(eq_0, optimizer, opt_config):
    """
    Run one optimization using a single shared opt_toggles dict.
    """

    #=============================================
    # Unpacking optimization configuration values:
    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]
    max_nfev = opt_config["max_nfev"]
    x_scale = opt_config["x_scale"]
    opt_toggles = opt_config["opt_toggles"]
    #=============================================

    #=========================================
    # Runtime values available to config dict:
    context = {
        "eq_0": eq_0,
        "optimizer": optimizer,
    }
    #=========================================

    constraints_list = []
    objectives_list = []

    #-------------
    _append_terms(
        term_list=objectives_list,
        toggle=opt_toggles,
        registry=OBJECTIVE_REGISTRY,
        context=context,
        kind="objective",
    )
    #--------------------

    #----------------
    # All constraints
    _append_terms(
        term_list=constraints_list,
        toggle=opt_toggles,
        registry=CONSTRAINT_REGISTRY,
        context=context,
        kind="constraint",
    )
    #---------------------

    #---------------------------------------
    # Finalizing optimization objects/setup:
    constraints = tuple(constraints_list)
    objectives = ObjectiveFunction(objectives_list)
    #----------------------------------------------
    #==============================================

    #======================
    # Running optimization:
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
    #===============

    return eq_opt, opt_result
#==============================================================================================================================================================