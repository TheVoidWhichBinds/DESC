import jax.numpy as jnp

from desc.grid import LinearGrid

from poly_constraints import (
    FLO_pressure_axis,
    FLO_pressure_shape,
    FLO_pressure_octic,
    FLO_pressure_DOF,
    FLO_pressure_edge,
    FLO_grad_pressure_edge,
    FLO_iota_axis,
    FLO_iota_edge,
    FLO_iota_quadratic,
    FNO_pressure_axis,
    FNO_pressure_edge,
    FNO_pressure_positive,
    FNO_pressure_monotonic,
    FNO_grad_pressure_edge,
    FNO_iota,
)










#============== SHARED VARIANT SETTINGS ==========================================================================
FLO_pressure_axis_bounds = (1e4, 1e7)
FLO_pressure_shape_bounds = (-1.3, 1.8)

FNO_pressure_axis_bounds = (1e4, 1e7)

iota_bounds = ("iota_between_rationals", "axis")

barrier_weights = 1e5

data_grid = LinearGrid(
    L = 200,
    M = 0,
    N = 0,
)
#==============================================================================================================










#============== BASE CONFIG ======================================================================================
BASE_CONFIG = {
    "variant_id": "base",

    "description": "Original literature setup. Does not modify objectives or constraints.",

    "objective_additions": [],

    "constraint_removals": [],

    "constraint_additions": [],
}
#==============================================================================================================










#============== FLO CONFIG =======================================================================================
FLO_CONFIG = {
    "variant_id": "FLO",

    "description": "Feature-linear-objective variant. Adds FLO pressure/iota objectives and linear profile constraints.",

    "objective_additions": [
        {
            "name": "FLO_pressure_axis",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_pressure_axis",
                "fun": FLO_pressure_axis,
                "bounds": FLO_pressure_axis_bounds,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FLO_pressure_shape",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_pressure_shape",
                "fun": FLO_pressure_shape,
                "bounds": FLO_pressure_shape_bounds,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FLO_iota_axis",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_iota_axis",
                "fun": FLO_iota_axis,
                "bounds": iota_bounds,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FLO_iota_edge",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_iota_edge",
                "fun": FLO_iota_edge,
                "bounds": iota_bounds,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
    ],

    "constraint_removals": [
        "FixedPressure",
        "FixPressure",
        "FixedIota",
        "FixIota",
    ],

    "constraint_additions": [
        {
            "name": "FLO_pressure_octic",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_pressure_octic",
                "fun": FLO_pressure_octic,
                "target": 0.0,
            },
        },
        {
            "name": "FLO_pressure_DOF",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_pressure_DOF",
                "fun": FLO_pressure_DOF,
                "target": 0.0,
            },
        },
        {
            "name": "FLO_pressure_edge",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_pressure_edge",
                "fun": FLO_pressure_edge,
                "target": 0.0,
            },
        },
        {
            "name": "FLO_grad_pressure_edge",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_grad_pressure_edge",
                "fun": FLO_grad_pressure_edge,
                "target": 0.0,
            },
        },
        {
            "name": "FLO_iota_quadratic",
            "class": "LinearObjectiveFromUser",
            "kwargs": {
                "name": "FLO_iota_quadratic",
                "fun": FLO_iota_quadratic,
                "target": 0.0,
            },
        },
    ],
}
#==============================================================================================================










#============== FNO CONFIG =======================================================================================
FNO_CONFIG = {
    "variant_id": "FNO",

    "description": "Feature-nonlinear-objective variant. Adds FNO pressure/iota profile objectives.",

    "objective_additions": [
        {
            "name": "FNO_pressure_axis",
            "class": "ObjectiveFromUser",
            "kwargs": {
                "name": "FNO_pressure_axis",
                "fun": FNO_pressure_axis,
                "grid": data_grid,
                "bounds": FNO_pressure_axis_bounds,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FNO_pressure_edge",
            "class": "ObjectiveFromUser",
            "kwargs": {
                "name": "FNO_pressure_edge",
                "fun": FNO_pressure_edge,
                "grid": data_grid,
                "target": 0.0,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FNO_pressure_positive",
            "class": "ObjectiveFromUser",
            "kwargs": {
                "name": "FNO_pressure_positive",
                "fun": FNO_pressure_positive,
                "grid": data_grid,
                "bounds": (0.0, jnp.inf),
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FNO_pressure_monotonic",
            "class": "ObjectiveFromUser",
            "kwargs": {
                "name": "FNO_pressure_monotonic",
                "fun": FNO_pressure_monotonic,
                "grid": data_grid,
                "bounds": (-jnp.inf, 0.0),
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FNO_grad_pressure_edge",
            "class": "ObjectiveFromUser",
            "kwargs": {
                "name": "FNO_grad_pressure_edge",
                "fun": FNO_grad_pressure_edge,
                "grid": data_grid,
                "target": 0.0,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
        {
            "name": "FNO_iota",
            "class": "ObjectiveFromUser",
            "kwargs": {
                "name": "FNO_iota",
                "fun": FNO_iota,
                "grid": data_grid,
                "bounds": iota_bounds,
                "weight": barrier_weights,
                "normalize": True,
            },
        },
    ],

    "constraint_removals": [
        "FixedPressure",
        "FixPressure",
        "FixedIota",
        "FixIota",
    ],

    "constraint_additions": [],
}
#==============================================================================================================










#============== VARIANT REGISTRY =================================================================================
VARIANT_REGISTRY = {
    "base": BASE_CONFIG,
    "FLO": FLO_CONFIG,
    "FNO": FNO_CONFIG,
}
#==============================================================================================================










#========== get_variant_config ===================================================================================
def get_variant_config(
        variant_id,
    ):
    if variant_id not in VARIANT_REGISTRY:
        available = ", ".join(sorted(VARIANT_REGISTRY.keys()))
        raise ValueError(f"Unknown variant_id '{variant_id}'. Available variants: {available}")

    return VARIANT_REGISTRY[variant_id]
#==============================================================================================================