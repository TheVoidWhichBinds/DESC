import jax.numpy as jnp
from math import comb
import inspect





#============== HELPER FUNCTION ====================================================================================================
#============================
def pressure_generator(
        p_axis, 
        n, 
        min_n=2
    ):
    """
    Coeffs for p(rho) = p_scale * (1 - rho^2)^n
    Guarantees: p(0)=p_scale, p(1)=0, p'(0)=0, p'(1)=0 for n>=2
    Also nonnegative + monotone decreasing on [0,1].
    """
    n_eff = max(int(n), int(min_n))
    coeff = [0.0] * (2 * n_eff + 1)
    for k in range(n_eff + 1):
        coeff[2 * k] = p_axis * comb(n_eff, k) * ((-1) ** k)
    return coeff
#===============


#==========================
def iota_between_rationals(
        iota_axis: float,
    ):
    """
    Generates bounds for iota optimizer constraint
    that are between low-order rational surfaces.
    """
    allowed_ranges = [
        (0.25,   0.3333),
        (0.3333, 0.5),
        (0.5,    0.6667),
        (0.6667, 0.75),
        (0.75,   1.0),
        (1.0,    1.3333),
        (1.3333, 1.5),
        (1.5,    2.0),
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










#============== HELPER FUNCTIONS ============================================================================================================================
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


#==================
def _merge_toggles(
        toggle_BOTH,
        toggle_single
    ):
    """
    Merge shared toggle entries with family-specific ones.

    Family-specific entries override shared ones if keys overlap.
    """
    return {
        **toggle_BOTH,
        **toggle_single,
    }
#========================


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
#==============================================================================================================================================================