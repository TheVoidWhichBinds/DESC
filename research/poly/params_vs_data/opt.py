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
    ObjectiveFromUser,
    ObjectiveFunction,
    QuasisymmetryBoozer,
)

from .helper import (
    _append_terms,
    _eq,
)


#============== REGISTRIES ========================================================================================================================
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

        # Custom params-based objectives:
    "pressure_axis_range_params": {
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
    "iota_axis_range_params": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_edge_range_params": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },

        # Custom data-based objectives:
    "pressure_axis_range_data": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_axis_range_data": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_edge_range_data": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "pressure_edge_data": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_pressure_edge_data": {
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

        # Custom params-based constraints:
    "pressure_axis_fxd": {
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
    "pressure_edge_params": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "grad_pressure_edge_params": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#======================
#==================================================================================================================================================




#============== HELPERS ===========================================================================================================================
def _get_toggle_block(opt_config, toggle_group):
    """
    Merge shared toggles with the toggle-group-specific additions.
    """
    shared = opt_config["opt_toggles_all"]
    specific = opt_config[toggle_group]
    merged = dict(shared)
    merged.update(specific)
    return merged
#==================================================================================================================================================




#============== OPTIMIZER FUNCTION ================================================================================================================
def run_optimization(eq_0, optimizer, toggle_group, opt_config):
    """
    Run one optimization using shared + toggle-group-specific entries.
    """

    #=============================================
    # Unpacking optimization configuration values:
    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]
    max_nfev = opt_config["max_nfev"]
    x_scale = opt_config["x_scale"]
    opt_toggles = _get_toggle_block(opt_config, toggle_group)
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

    #-------------------
    _append_terms(
        term_list=objectives_list,
        toggle=opt_toggles,
        registry=OBJECTIVE_REGISTRY,
        context=context,
        kind="objective",
    )
    #-------------------

    #-------------------
    _append_terms(
        term_list=constraints_list,
        toggle=opt_toggles,
        registry=CONSTRAINT_REGISTRY,
        context=context,
        kind="constraint",
    )
    #-------------------

    #---------------------------------------
    # Finalizing optimization objects/setup:
    constraints = tuple(constraints_list)
    objectives = ObjectiveFunction(objectives_list)
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