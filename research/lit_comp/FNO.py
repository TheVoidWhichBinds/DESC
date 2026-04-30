# FNO.py
#==============================================================================================================
#
# FNO variant configuration for DESC/research/final.
#
# FNO objectives are applied only when variant = True.
# They are not active when variant = False.
#
# The actual functions live in custom_funcs.py.
#
# Each objective config entry uses:
#   "name"   : descriptive label
#   "fun"    : function from custom_funcs.py
#   "target" : target value for equality-style objectives, OR
#   "bounds" : lower and upper bounds for bounded objectives
#   "kwargs" : keyword arguments passed into the DESC objective wrapper
#
# "thing" should be filled in by helper.py with the active equilibrium object.
#
#==============================================================================================================

import jax.numpy as jnp

try:
    from .custom_funcs import (
        FNO_pressure_edge,
        FNO_pressure_positive,
        FNO_pressure_monotonic,
        FNO_grad_pressure_edge,
    )
except ImportError:
    from custom_funcs import (
        FNO_pressure_edge,
        FNO_pressure_positive,
        FNO_pressure_monotonic,
        FNO_grad_pressure_edge,
    )










#==============================================================================================================
# FNO Configuration
#==============================================================================================================

FNO_CONFIG = {
    "objectives": (
        {
            "name": "FNO_pressure_edge",
            "fun": FNO_pressure_edge,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FNO_pressure_positive",
            "fun": FNO_pressure_positive,
            "bounds": (0, jnp.inf),
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FNO_pressure_monotonic",
            "fun": FNO_pressure_monotonic,
            "bounds": (-jnp.inf, 0),
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FNO_grad_pressure_edge",
            "fun": FNO_grad_pressure_edge,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
    ),

    "constraints": (),
}