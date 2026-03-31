import inspect
import os
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
)










#============== HELPER FUNCTIONS ============================================================================================================================
#=========================
def resolve_from_context(func):
    """
    Mark a callable as something that should be resolved
    from the runtime context dict.
    """
    func._resolve_from_context = True
    return func
#-------------------------


#=========================
@resolve_from_context
def _eq(ctx):
    return ctx["eq_0"]
#--------------------------------------------------------------------------------------------------------------------------------------------------


#=========================
def _resolve_value(value, context):
    """
    Resolve a config value.

    Only call callables that were explicitly marked
    as context resolvers.
    """
    if callable(value) and getattr(value, "_resolve_from_context", False):
        return value(context)
    return value
#--------------------------------------------------------------------------------------------------------------------------------------------------


#=========================
def _resolve_kwargs(kwargs, context):
    """
    Resolve all values in a kwargs dict.
    """
    return {
        key: _resolve_value(value, context)
        for key, value in kwargs.items()
    }
#-------------------------


#=========================
def _validate_kwargs(entry, key, wrapper, kind):
    """
    Validate user-provided kwargs against wrapper signature.
    """
    sig = inspect.signature(wrapper)
    valid_params = set(sig.parameters.keys())

    invalid_keys = set(entry.keys()) - valid_params
    if invalid_keys:
        raise TypeError(
            f"Invalid kwarg(s) for {kind} '{key}' using wrapper "
            f"'{wrapper.__name__}': {sorted(invalid_keys)}. "
            f"Valid kwargs are: {sorted(valid_params)}"
        )
#-------------------------


#=========================
def _parse_entry(toggle, key, kind):
    """
    Parse a config entry of the form:
        {
            "use": bool,
            "kwargs": dict,
        }

    Missing entries default to off.
    """
    entry = toggle.get(key, {"use": False, "kwargs": {}})

    if not isinstance(entry, dict):
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' must be a dict with keys "
            f"'use' and 'kwargs'. Got {type(entry).__name__}: {entry!r}"
        )

    allowed_keys = {"use", "kwargs"}
    invalid_keys = set(entry.keys()) - allowed_keys
    if invalid_keys:
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' has invalid top-level key(s): "
            f"{sorted(invalid_keys)}. Allowed keys are: {sorted(allowed_keys)}"
        )

    use = entry.get("use", False)
    kwargs = entry.get("kwargs", {})

    if not isinstance(use, bool):
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' has non-bool 'use': {use!r}"
        )

    if not isinstance(kwargs, dict):
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' has non-dict 'kwargs': {kwargs!r}"
        )

    return use, dict(kwargs)
#-------------------------


#=========================
def _append_term(term_list, wrapper, key, toggle, defaults, context, kind):
    """
    Append objective or constraint from a single config entry.

    User kwargs override defaults.
    """
    use, user_kwargs = _parse_entry(toggle, key, kind)
    if not use:
        return

    _validate_kwargs(user_kwargs, key, wrapper, kind)

    kwargs = {**defaults, **user_kwargs}
    kwargs = _resolve_kwargs(kwargs, context)
    _validate_kwargs(kwargs, key, wrapper, kind)

    term_list.append(wrapper(**kwargs))
#-------------------------


#=========================
def _append_terms(term_list, toggle, registry, context, kind):
    """
    Loop over a registry and append all enabled terms.
    """
    for key, spec in registry.items():
        _append_term(
            term_list = term_list,
            wrapper = spec["wrapper"],
            key = key,
            toggle = toggle,
            defaults = spec["defaults"],
            context = context,
            kind = kind,
        )
#=========================
#==============================================================================================================================================================










#============== REGISTRIES ============================================================================================================================
OBJECTIVE_REGISTRY_FXD = {
    "forcebalance_obj": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "aspect_ratio_obj": {
        "wrapper": AspectRatio,
        "defaults": {
            "eq": _eq,
        },
    },
    "qs_obj": {
        "wrapper": QuasisymmetryBoozer,
        "defaults": {
            "eq": _eq,
        },
    },
    "ballooning_obj": {
        "wrapper": BallooningStability,
        "defaults": {
            "eq": _eq,
        },
    },
    "mercier_obj": {
        "wrapper": MercierStability,
        "defaults": {
            "eq": _eq,
        },
    },
}


OBJECTIVE_REGISTRY_CON = {
    "forcebalance_obj": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "aspect_ratio_obj": {
        "wrapper": AspectRatio,
        "defaults": {
            "eq": _eq,
        },
    },
    "qs_obj": {
        "wrapper": QuasisymmetryBoozer,
        "defaults": {
            "eq": _eq,
        },
    },
    "ballooning_obj": {
        "wrapper": BallooningStability,
        "defaults": {
            "eq": _eq,
        },
    },
    "mercier_obj": {
        "wrapper": MercierStability,
        "defaults": {
            "eq": _eq,
        },
    },
    "pressure_monotonicity_obj": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_positivity_obj": {
        "wrapper": ObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}


CONSTRAINT_REGISTRY_FXD = {
    "forcebalance_con": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_iota_con": {
        "wrapper": FixIota,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_psi_con": {
        "wrapper": FixPsi,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_pressure_con": {
        "wrapper": FixPressure,
        "defaults": {
            "eq": _eq,
        },
    },
}


CONSTRAINT_REGISTRY_CON = {
    "forcebalance_con": {
        "wrapper": ForceBalance,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_iota_con": {
        "wrapper": FixIota,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_psi_con": {
        "wrapper": FixPsi,
        "defaults": {
            "eq": _eq,
        },
    },
    "fix_pressure_con": {
        "wrapper": FixPressure,
        "defaults": {
            "eq": _eq,
        },
    },
    "pressure_axis_con": {
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
    "grad_pressure_axis_con": {
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
    "grad_iota_axis_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_rational_edge_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
    "iota_rational_range_con": {
        "wrapper": LinearObjectiveFromUser,
        "defaults": {
            "thing": _eq,
        },
    },
}
#==============================================================================================================================================================










#============== OPTIMIZER FUNCTION ============================================================================================================================
def run_optimization(eq_0, optimizer, p_scale, out_dir, opt_config, FXD: bool):
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
    toggle_FXD = opt_config["toggle_FXD"]
    toggle_CON = opt_config["toggle_CON"]
    toggle = toggle_FXD if FXD else toggle_CON
    #=============================================


    #=========================================
    # Runtime values available to config dict:
    context = {
        "eq_0": eq_0,
        "p_scale": p_scale,
        "FXD": FXD,
        "optimizer": optimizer,
        "out_dir": out_dir,
    }
    #=========================================


    constraints_list = []
    objectives_list = []

    #=========================
    if FXD:  # (fixed profiles)
        _append_terms(
            term_list = objectives_list,
            toggle = toggle,
            registry = OBJECTIVE_REGISTRY_FXD,
            context = context,
            kind = "objective",
        )

        _append_terms(
            term_list = constraints_list,
            toggle = toggle,
            registry = CONSTRAINT_REGISTRY_FXD,
            context = context,
            kind = "constraint",
        )
    #====================================================================


    #=======================
    else:  # (free profiles)
        _append_terms(
            term_list = objectives_list,
            toggle = toggle,
            registry = OBJECTIVE_REGISTRY_CON,
            context = context,
            kind = "objective",
        )

        _append_terms(
            term_list = constraints_list,
            toggle = toggle,
            registry = CONSTRAINT_REGISTRY_CON,
            context = context,
            kind = "constraint",
        )
    #====================================================================


    #---------------------------------------
    # Finalizing optimization objects/setup:
    constraints = tuple(constraints_list)
    objectives = ObjectiveFunction(objectives_list)
    #---------------------------------------
    #=======================================


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
        copy = True,
        verbose = 3,
    )
    #======================


    save_name = "opt_FXD.h5" if FXD else "opt_CON.h5"
    save_path = os.path.join(out_dir, save_name)
    eq_opt.save(save_path)

    return eq_opt, opt_result
#==============================================================================================================================================================