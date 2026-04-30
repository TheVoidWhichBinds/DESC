# FLO.py
#==============================================================================================================
#
# FLO variant configuration for DESC/research/final.
#
# FLO objectives and constraints are applied only when variant = "FLO".
# They are not active during the base optimization.
#
# The actual functions live in VFOs.py.
#
# Each objective config entry uses:
#   "name"   : descriptive label
#   "fun"    : function from VFOs.py
#   "bounds" : lower and upper bounds for the objective value
#   "kwargs" : keyword arguments passed into the DESC objective wrapper
#
# Each constraint config entry uses:
#   "name"   : descriptive label
#   "fun"    : function from VFOs.py
#   "target" : target value for the constraint
#   "kwargs" : keyword arguments passed into the DESC objective wrapper
#
# "thing" should be filled in by helper.py with the initial equilibrium object.
#
# "lower_rational" and "upper_rational" should be filled in by helper.py
# from the paper-specific initial on-axis iota value.
#
#==============================================================================================================

try:
    from .custom_funcs import (
        FLO_pressure_axis,
        FLO_pressure_shape,
        FLO_pressure_octic,
        FLO_pressure_DOF,
        FLO_pressure_edge,
        FLO_grad_pressure_edge,
        FLO_iota_axis,
        FLO_iota_edge,
        FLO_iota_quadratic,
    )
except ImportError:
    from research.lit_comp.custom_funcs import (
        FLO_pressure_axis,
        FLO_pressure_shape,
        FLO_pressure_octic,
        FLO_pressure_DOF,
        FLO_pressure_edge,
        FLO_grad_pressure_edge,
        FLO_iota_axis,
        FLO_iota_edge,
        FLO_iota_quadratic,
    )










#==============================================================================================================
# FLO Configuration
#==============================================================================================================

FLO_CONFIG = {
    "objectives": (
        {
            "name": "FLO_pressure_axis",
            "fun": FLO_pressure_axis,
            "bounds": (1e4, 1e7),
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_pressure_shape",
            "fun": FLO_pressure_shape,
            "bounds": (-1.3, 1.8),
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_iota_axis",
            "fun": FLO_iota_axis,
            "bounds": ("lower_rational", "upper_rational"),
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_iota_edge",
            "fun": FLO_iota_edge,
            "bounds": ("lower_rational", "upper_rational"),
            "kwargs": {
                "thing": None,
            },
        },
    ),

    "constraints": (
        {
            "name": "FLO_pressure_octic",
            "fun": FLO_pressure_octic,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_pressure_DOF",
            "fun": FLO_pressure_DOF,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_pressure_edge",
            "fun": FLO_pressure_edge,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_grad_pressure_edge",
            "fun": FLO_grad_pressure_edge,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
        {
            "name": "FLO_iota_quadratic",
            "fun": FLO_iota_quadratic,
            "target": 0,
            "kwargs": {
                "thing": None,
            },
        },
    ),
}
