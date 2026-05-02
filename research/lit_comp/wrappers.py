# wrappers.py
#==============================================================================================================
#
# FREE variant configuration for DESC/research/lit_comp.
#
# FREE objectives and constraints are applied only when free = True.
# They are not active when free = False.
#
# The actual functions live in custom_funcs.py.
#
# Each objective/constraint config entry uses:
#   "name"    : descriptive label
#   "fun"     : function from custom_funcs.py
#   "target"  : target value for equality-style objectives/constraints, OR
#   "bounds"  : lower and upper bounds for bounded objectives
#   "wrapper" : "linear" or "nonlinear"
#   "kwargs"  : keyword arguments passed into the DESC objective wrapper
#
# "thing" should be filled in by helper.py with the active equilibrium object.
# FREE_pressure_axis has target = None here because helper.py injects the initial pressure-axis value.
#
#==============================================================================================================

import jax.numpy as jnp

try:
    from .custom_funcs import (
        FREE_pressure_axis,
        FREE_pressure_edge,
        FREE_pressure_positive,
        FREE_pressure_monotonic,
        FREE_grad_pressure_edge,
    )
except ImportError:
    from custom_funcs import (
        FREE_pressure_axis,
        FREE_pressure_edge,
        FREE_pressure_positive,
        FREE_pressure_monotonic,
        FREE_grad_pressure_edge,
    )










#==============================================================================================================
# FREE Configuration
#==============================================================================================================

FREE_CONFIG = {
    "objectives": (
        {
            "name": "FREE_pressure_positive",
            "fun": FREE_pressure_positive,
            "bounds": (0, jnp.inf),
            "wrapper": "nonlinear",
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FREE_pressure_monotonic",
            "fun": FREE_pressure_monotonic,
            "bounds": (-jnp.inf, 0),
            "wrapper": "nonlinear",
            "kwargs": {
                "thing": None,
            },
        },
    ),

    "constraints": (
        {
            "name": "FREE_pressure_axis",
            "fun": FREE_pressure_axis,
            "target": None,
            "wrapper": "linear",
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FREE_pressure_edge",
            "fun": FREE_pressure_edge,
            "target": 0,
            "wrapper": "linear",
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FREE_grad_pressure_edge",
            "fun": FREE_grad_pressure_edge,
            "target": 0,
            "wrapper": "linear",
            "kwargs": {
                "thing": None,
            },
        },
    ),
}