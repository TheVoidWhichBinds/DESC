
import sys
import numpy as np
sys.path.append("/Users/macdaddi/DESC")
from desc.objectives import (
    ObjectiveFunction,
    FixIota,
    FixPsi,
    FixPressure,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    BallooningStability,
    MercierStability,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
    CurrentDensity
)
from .helper import (
    _eq,
    _merge_toggles,
    _append_terms,
)










#============== REGISTRIES ============================================================================================================================
#========
# Shared:
#-------------------------
OBJECTIVE_REGISTRY_BOTH = {
        # Standard:
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
#---------------------

#---------------------------
CONSTRAINT_REGISTRY_BOTH = {
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
}
#---------------------
#=====================




#================
# Fixed profiles:
#--------------------------
OBJECTIVE_REGISTRY_FXD = {
    "aspect_ratio": {
        "wrapper": AspectRatio,
        "defaults": {
            "eq": _eq,
        },
    },
}
#--------------------------

#--------------------------
CONSTRAINT_REGISTRY_FXD = {
        # Standard:
    "fix_iota": {
        "wrapper": FixIota,
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
}
#---------------------
#=====================




#=========================
# Free profiles:
#-------------------------
OBJECTIVE_REGISTRY_FREE = {
        # Custom:
    "aspect_ratio_range": {
        "wrapper": AspectRatio,
        "defaults": {
            "eq": _eq,
        },
    },
    "pressure_axis_range": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_monotonicity": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_positive": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_range": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "current_range": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#------------------------

#--------------------------
CONSTRAINT_REGISTRY_FREE = {
        # Standard:
    "fix_iota": {
        "wrapper": FixIota,
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

        # Custom:
    "pressure_edge": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_pressure_axis": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_pressure_edge": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_iota_axis": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_edge": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_axis": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#------------------------
#========================
#==============================================================================================================================================================










#============== OPTIMIZER FUNCTION ============================================================================================================================
def run_optimization(eq_0, optimizer, opt_config, FXD: bool):
    """
    Run optimization with objectives/constraints controlled by config dict.

    FXD = True  -> fixed-pressure family
    FXD = False -> constrained/optimized-pressure family
    """

    #=============================================
    # Unpacking optimization configuration values:
    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]
    max_nfev = opt_config["max_nfev"]
    x_scale = opt_config["x_scale"]
    toggle_BOTH = opt_config["toggle_BOTH"]
    toggle_FXD = opt_config["toggle_FXD"]
    toggle_FREE = opt_config["toggle_FREE"]
    toggle = _merge_toggles(
        toggle_BOTH,
        toggle_FXD if FXD else toggle_FREE,
    )
    #=====================================


    #=========================================
    # Runtime values available to config dict:
    context = {
        "eq_0": eq_0,
        "FXD": FXD,
        "optimizer": optimizer,
    }
    #=========================================


    constraints_list = []
    objectives_list = []

    #-------------
    _append_terms(
        term_list=objectives_list,
        toggle=toggle,
        registry=OBJECTIVE_REGISTRY_BOTH,
        context=context,
        kind="objective",
    )

    _append_terms(
        term_list=constraints_list,
        toggle=toggle,
        registry=CONSTRAINT_REGISTRY_BOTH,
        context=context,
        kind="constraint",
    )
    #---------------------

    #-----------------
    if FXD:
        _append_terms(
            term_list=objectives_list,
            toggle=toggle,
            registry=OBJECTIVE_REGISTRY_FXD,
            context=context,
            kind="objective",
        )

        _append_terms(
            term_list=constraints_list,
            toggle=toggle,
            registry=CONSTRAINT_REGISTRY_FXD,
            context=context,
            kind="constraint",
        )
    #-------------------------

    #-----------------
    else:
        _append_terms(
            term_list=objectives_list,
            toggle=toggle,
            registry=OBJECTIVE_REGISTRY_FREE,
            context=context,
            kind="objective",
        )

        _append_terms(
            term_list=constraints_list,
            toggle=toggle,
            registry=CONSTRAINT_REGISTRY_FREE,
            context=context,
            kind="constraint",
        )
    #-------------------------

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
    #======================

    return eq_opt, opt_result
#==============================================================================================================================================================