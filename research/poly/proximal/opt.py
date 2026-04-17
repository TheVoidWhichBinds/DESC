import sys
sys.path.append("/Users/macdaddi/DESC")

from desc.objectives import (
    AspectRatio,
    BallooningStability,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    LinearObjectiveFromUser,
    MercierStability,
    ObjectiveFunction,
    QuasisymmetryBoozer,
)

from .helper import (
    _append_terms,
    _eq,
)










#============== REGISTRIES ========================================================================================================================
#===========================
CORE_OBJECTIVE_REGISTRY = {
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
}
#---------------------------


#============================
CORE_CONSTRAINT_REGISTRY = {
        # Standard:
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
#----------------------------


#==============================
CUSTOM_OBJECTIVE_REGISTRY = {
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
}
#------------------------------


#===============================
CUSTOM_CONSTRAINT_REGISTRY = {
    "pressure_axis": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_octic": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_DOF": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_edge": {
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
    "iota_quadratic": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#===============================
#==================================================================================================================================================










#============== HELPERS ===========================================================================================================================
def _build_terms(eq_0, optimizer, opt_config):
    """
    Build objective and constraint term lists from shared core/custom toggles.
    For auglag, all custom terms are routed into constraints.
    """
    #=========================================
    # Runtime values available to config dict:
    context = {
        "eq_0": eq_0,
        "optimizer": optimizer,
    }
    #=========================================

    opt_toggles_core = opt_config["opt_toggles_core"]
    opt_toggles_custom = opt_config["opt_toggles_custom"]

    objectives_list = []
    constraints_list = []

    #-------------------
    # Core objectives:
    _append_terms(
        term_list=objectives_list,
        toggle=opt_toggles_core,
        registry=CORE_OBJECTIVE_REGISTRY,
        context=context,
        kind="objective",
    )
    #-------------------

    #-------------------
    # Core constraints:
    _append_terms(
        term_list=constraints_list,
        toggle=opt_toggles_core,
        registry=CORE_CONSTRAINT_REGISTRY,
        context=context,
        kind="constraint",
    )
    #-------------------

    #-----------------------------------
    # Custom routing depends on optimizer:
    if optimizer == "proximal-lsq-auglag":
        _append_terms(
            term_list=objectives_list,
            toggle=opt_toggles_custom,
            registry=CUSTOM_OBJECTIVE_REGISTRY,
            context=context,
            kind="objective",
        )

        _append_terms(
            term_list=constraints_list,
            toggle=opt_toggles_custom,
            registry=CUSTOM_CONSTRAINT_REGISTRY,
            context=context,
            kind="constraint",
        )

    elif optimizer == "lsq-auglag":
        _append_terms(
            term_list=constraints_list,
            toggle=opt_toggles_custom,
            registry=CUSTOM_OBJECTIVE_REGISTRY,
            context=context,
            kind="constraint",
        )

        _append_terms(
            term_list=constraints_list,
            toggle=opt_toggles_custom,
            registry=CUSTOM_CONSTRAINT_REGISTRY,
            context=context,
            kind="constraint",
        )

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    #-----------------------------------

    return objectives_list, constraints_list
#==================================================================================================================================================










#============== OPTIMIZER FUNCTION ================================================================================================================
def run_optimization(eq_0, optimizer, opt_config):
    """
    Run one optimization using shared core/custom toggle dictionaries.
    """
    #=============================================
    # Unpacking optimization configuration values:
    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]
    max_nfev = opt_config["max_nfev"]
    x_scale = opt_config["x_scale"]
    #=============================================

    #------------------------
    # Building term objects:
    objectives_list, constraints_list = _build_terms(
        eq_0=eq_0,
        optimizer=optimizer,
        opt_config=opt_config,
    )
    #------------------------

    #---------------------------------------
    # Finalizing optimization objects/setup:

    constraints = tuple(constraints_list)
    objectives = ObjectiveFunction(objectives_list)

    print("constraints:")
    print(type(constraints))
    print("n constraints =", len(constraints))
    for i, con in enumerate(constraints):
        print(i, type(con).__name__, getattr(con, "name", None))
        
    #---------------------------------------
    #=======================================

    #======================
    # Running optimization:
    eq_opt, opt_result = eq_0.optimize(
        objective=objectives,
        constraints=constraints,
        optimizer=optimizer,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        maxiter=maxiter,
        options={"max_nfev": max_nfev},
        x_scale=x_scale,
        copy=True,
        verbose=3,
    )
    #======================

    return eq_opt, opt_result
#==================================================================================================================================================