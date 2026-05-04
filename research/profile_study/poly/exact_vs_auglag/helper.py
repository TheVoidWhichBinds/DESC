import jax.numpy as jnp
from math import comb
import inspect
import re
import os
import numpy as np
from copy import deepcopy




#============== CONFIG ====================================================================================================
#======================
def pressure_generator(
        p_axis,
        width_percentage
    ):
    """
    Generates an 8th order polynomial for pressure that obeys
    boundary relations, DOF constraint, and is within the
    allowed region for c[5] (coefficient of the 8th-order term).
    width_percentage gives how narrow (0%) to how wide (100%) the
    shape is.
    """
    c_0 = 1
    c_4 = -1.3 + width_percentage/100 * (1.8 - 1.3)
    c_3 = -3.45 * c_4
    c_1 = -2 + c_3 + 2*c_4
    c_2 = 1 - 2*c_3 - 3*c_4
    coeffs = jnp.array([c_0, c_1, c_2])

    return list(p_axis * coeffs)
#===================================


#==========================
def iota_between_rationals(
        iota_axis: float,
    ):
    """
    Generates bounds for iota optimizer constraint
    that are between low-order rational surfaces.
    """
    allowed_ranges = [
        (0.0,    0.25),
        (0.25,   0.3333),
        (0.3333, 0.5),
        (0.5,    0.6667),
        (0.6667, 0.75),
        (0.75,   1.0),
        (1.0,    1.25),
        (1.25,   1.3333),
        (1.3333, 1.5),
        (1.5,    1.6667),
        (1.6667, 2.0),
        (2.0,    3.0),
        (3.0,    4.0),
    ]

    matched_lower = None
    matched_upper = None

    for lower, upper in allowed_ranges:
        if lower <= iota_axis <= upper:
            matched_lower = lower
            matched_upper = upper
            break

    if matched_lower is None:
        raise ValueError("iota_axis is outside all allowed rational intervals.")

    lower_bound = matched_lower
    upper_bound = matched_upper

    return lower_bound, upper_bound
#==================================
#===================================================================================================================================










#============== DRIVER ==============================================================================================================================
#=========================
def sci_compact(x, sig=2):
    s = f"{x:.{sig-1}e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"{mant}e{exp}"
#=========================


#================
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan
#====================


#============================
def _extract_f_stats(objval):
    """
    Preps final objective values to be put into comparison table.
    Returns numeric f_min, f_mean, f_max.
    """
    #---------------------------
    if isinstance(objval, list):
        chosen = None
        for item in objval:
            if isinstance(item, dict) and all(k in item for k in ("f_min", "f_mean", "f_max")):
                chosen = item
                break
        if chosen is None and len(objval) > 0:
            chosen = objval[0]
        objval = chosen
    #------------------

    #--------------------------------------------------------------------------------------
    if isinstance(objval, dict) and all(k in objval for k in ("f_min", "f_mean", "f_max")):
        return (
            _to_float(objval["f_min"]),
            _to_float(objval["f_mean"]),
            _to_float(objval["f_max"]),
        )
    #----------------------------------

    #---------------------------
    if isinstance(objval, dict):
        for v in objval.values():
            val = _to_float(v)
            if not np.isnan(val):
                return val, val, val
        return np.nan, np.nan, np.nan
    #--------------------------------

    val = _to_float(objval)
    return val, val, val
#=======================


#============================================
def _safe_extract_from_result(result, label):
    """
    Safely gets objective stats from result dict.
    Returns NaNs if label is absent.
    """
    objvals = result.get("Objective values", {})
    if label not in objvals:
        return np.nan, np.nan, np.nan
    return _extract_f_stats(objvals[label])
#==========================================


#===================================
def _next_run_dir(continuation_dir):
    """
    Create next zero-padded run directory: 001, 002, ...
    """
    os.makedirs(continuation_dir, exist_ok=True)

    run_nums = []
    for name in os.listdir(continuation_dir):
        path = os.path.join(continuation_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"\d{3}", name):
            run_nums.append(int(name))

    next_num = 1 if len(run_nums) == 0 else max(run_nums) + 1
    run_dir = os.path.abspath(os.path.join(continuation_dir, f"{next_num:03d}"))

    if os.path.exists(run_dir):
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    os.makedirs(run_dir)

    return run_dir
#=================


#=======================================
def _write_readme(out_dir, config_path):
    """
    Write config.py contents into README.md, 
    omitting the first 50 lines.
    """
    with open(config_path, "r") as f:
        config_lines = f.readlines()

    config_text = "".join(config_lines[50:])

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("```python\n")
        f.write(config_text)
        if not config_text.endswith("\n"):
            f.write("\n")
        f.write("```\n")
#=======================


#============== OPT ============================================================================================================================
#==============================
def resolve_from_context(func):
    """
    Mark a callable as something that should be resolved
    from the runtime context dict.
    """
    func._resolve_from_context = True
    return func
#==============


#====================
@resolve_from_context
def _eq(ctx):
    return ctx["eq_0"]
#=====================


#==================
def _resolve_value(
        value,
        context
    ):
    """
    Resolve a config value.

    Only call callables that were explicitly marked
    as context resolvers.
    """
    if callable(value) and getattr(value, "_resolve_from_context", False):
        return value(context)
    return value
#===============


#===================
def _resolve_kwargs(
        kwargs,
        context
    ):
    """
    Resolve all values in a kwargs dict.
    """
    return {
        key: _resolve_value(value, context)
        for key, value in kwargs.items()
    }
#=======================================


#====================
def _validate_kwargs(
        entry,
        key,
        wrapper,
        kind
    ):
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
#======================================================


#================
def _parse_entry(
        toggle,
        key,
        kind
    ):
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
#===========================


#=========================
def _append_term(
        term_list,
        wrapper,
        key,
        toggle,
        defaults,
        context,
        kind
    ):
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
def _append_terms(
        term_list,
        toggle,
        registry,
        context,
        kind
    ):
    """
    Loop over a registry and append all enabled terms.
    """
    for key, spec in registry.items():
        _append_term(
            term_list=term_list,
            wrapper=spec["wrapper"],
            key=key,
            toggle=toggle,
            defaults=spec["defaults"],
            context=context,
            kind=kind,
        )
#=========================


#=========================
def _merge_toggle_dicts(*toggle_dicts):
    """
    Merge toggle dicts left-to-right.
    Later dicts override earlier dicts for shared keys.
    """
    merged = {}
    for toggle_dict in toggle_dicts:
        if toggle_dict is None:
            continue
        for key, value in toggle_dict.items():
            merged[key] = deepcopy(value)
    return merged
#=========================


#=========================
def _get_optimizer_toggles(
        opt_config,
        optimizer
    ):
    """
    Build final toggle dict for a given optimizer from:
        - opt_toggles_both
        - optimizer-specific toggles
    """
    opt_toggles_both = opt_config.get("opt_toggles_both", {})

    if optimizer == "proximal-lsq-exact":
        opt_toggles_specific = opt_config.get("opt_toggles_prox", {})
    elif optimizer == "lsq-auglag":
        opt_toggles_specific = opt_config.get("opt_toggles_auglag", {})
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return _merge_toggle_dicts(opt_toggles_both, opt_toggles_specific)
#=========================
#==============================================================================================================================================================