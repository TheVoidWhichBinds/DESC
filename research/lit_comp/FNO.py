# FNO.py
#==============================================================================================================
#
# FNO variant configuration for DESC/research/final.
#
# FNO objectives are applied only when variant = "FNO".
# They are not active during the base optimization.
#
# The actual functions live in VFOs.py.
#
# Each objective config entry uses:
#   "name"   : descriptive label
#   "fun"    : function from VFOs.py
#   "target" : target value for equality-style objectives, OR
#   "bounds" : lower and upper bounds for bounded objectives
#   "kwargs" : keyword arguments passed into the DESC objective wrapper
# 
# "thing" should be filled in by helper.py with the initial equilibrium object.
#
# "lower_rational" and "upper_rational" should be filled in by helper.py
# from the paper-specific initial on-axis iota value.
#
#==============================================================================================================

import jax.numpy as jnp

try:
    from .VFOs import (
        FNO_pressure_axis,
        FNO_pressure_edge,
        FNO_pressure_positive,
        FNO_pressure_monotonic,
        FNO_grad_pressure_edge,
        FNO_iota,
    )
except ImportError:
    from VFOs import (
        FNO_pressure_axis,
        FNO_pressure_edge,
        FNO_pressure_positive,
        FNO_pressure_monotonic,
        FNO_grad_pressure_edge,
        FNO_iota,
    )










#==============================================================================================================
# FNO Configuration
#==============================================================================================================

FNO_CONFIG = {
    "objectives": (
        {
            "name": "FNO_pressure_axis",
            "fun": FNO_pressure_axis,
            "bounds": (1e4, 1e7),
            "kwargs": {
                "thing": None,
            },
        },
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
        {
            "name": "FNO_iota",
            "fun": FNO_iota,
            "bounds": ("lower_rational", "upper_rational"),
            "kwargs": {
                "thing": None,
            },
        },
    ),

    "constraints": (),
}
