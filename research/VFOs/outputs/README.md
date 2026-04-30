```python
# config.py
#===================================================================================================================================================

from pathlib import Path

import jax.numpy as jnp

from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile

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
        FNO_pressure_axis,
        FNO_pressure_edge,
        FNO_pressure_positive,
        FNO_pressure_monotonic,
        FNO_grad_pressure_edge,
        FNO_iota,
    )

    from .helper import (
        iota_between_rationals,
        load_surface_pool,
    )

except ImportError:
    from custom_funcs import (
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

    from helper import (
        iota_between_rationals,
        load_surface_pool,
    )

#===================================================================================================================================================










#===========================================================
# PATH CONFIG
#===========================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

#===================================================================================================================================================










#===========================================================
# EQUILIBRIUM INPUTS
#===========================================================

L = 12
M = 12
N = 12
eq_resolution = [
    L,
    M,
    N,
]

NFP = 4

surface_source_config = {
    "dataset_path": str(
        REPO_ROOT
        / "research"
        / "surface_generator"
        / f"NFP_{NFP}"
        / "dataset.pkl"
    ),
    "selection_method": "all_nested",
    "shuffle": True,
    "shuffle_seed": 42,
}

surface_pool = load_surface_pool(
    surface_source_config = surface_source_config,
    NFP = NFP,
    N_eq = 1,
)

surface_init = surface_pool[0]["surface_init"]
selected_point = surface_pool[0]["selected_point"]

p_axis_init = 1.0e5
pressure_init = PowerSeriesProfile(
    [
        p_axis_init,
        -2.0E5,
        1.0E5,
    ],
    sym = True,
)

iota_axis_init = 0.51
iota_init = PowerSeriesProfile(
    [
        iota_axis_init,
        0.14,
    ],
    sym = True,
)

EQ_CONFIG = {
    "NFP":           NFP,
    "surface_init":  surface_init,
    "pressure_init": pressure_init,
    "iota_init":     iota_init,
    "eq_resolution": eq_resolution,
}

#===================================================================================================================================================










#===========================================================
# OPTIMIZATION SCALARS
#===========================================================

FLO_pressure_axis_bounds = (
    1e5,
    1e7,
)

FLO_pressure_shape_bounds = (
    -1.3,
    1.8,
)

# FNO_pressure_axis_bounds = (
#     1e5,
#     1e7,
# )

iota_bounds = iota_between_rationals(
    iota_axis = iota_axis_init,
)

optimizer = "lsq-exact"

barrier_weights = 1e5
ftol = 1e-4
xtol = 1e-6
gtol = 1e-6
maxiter = 300
max_nfev = 300
x_scale = "auto"

pressure_fxd = False
iota_fxd = True

data_grid = LinearGrid(
    L = 200,
    M = 0,
    N = 0,
)

#===================================================================================================================================================










#===========================================================
# CORE OPT TOGGLES
#===========================================================

opt_toggles_core = {
    "forcebalance_obj": {
        "use": True,
        "kwargs": {
            "weight": 1e1,
            "target": 0.0,
            "normalize": True,
        },
    },
    "qs": {
        "use": True,
        "kwargs": {
            "weight": 1e0,
            "helicity": (
                1,
                NFP,
            ),
            "normalize": True,
        },
    },
    "ballooning": {
        "use": True,
        "kwargs": {
            "weight": 1e0,
            "target": 0.0,
            "normalize": True,
        },
    },
    "mercier": {
        "use": True,
        "kwargs": {
            "bounds": (
                0.05,
                jnp.inf,
            ),
            "weight": 1e0,
            "normalize": True,
        },
    },
    "forcebalance_con": {
        "use": True,
        "kwargs": {
            "target": 0.0,
        },
    },
    "fix_pressure": {
        "use": pressure_fxd,
        "kwargs": {},
    },
    "fix_iota": {
        "use": iota_fxd,
        "kwargs": {},
    },
    "fix_psi": {
        "use": True,
        "kwargs": {},
    },
}

#===================================================================================================================================================










#===========================================================
# FLO OPT TOGGLES
#===========================================================

opt_toggles_FLO = {
    # "FLO_pressure_axis": {
    #     "use": not pressure_fxd,
    #     "kwargs": {
    #         "name": "FLO_pressure_axis",
    #         "fun": FLO_pressure_axis,
    #         "bounds": FLO_pressure_axis_bounds,
    #         "weight": barrier_weights,
    #         "normalize": True,
    #     },
    # },
    "FLO_pressure_shape": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_pressure_shape",
            "fun": FLO_pressure_shape,
            "bounds": FLO_pressure_shape_bounds,
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    "FLO_iota_axis": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "FLO_iota_axis",
            "fun": FLO_iota_axis,
            "bounds": iota_bounds,
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    "FLO_iota_edge": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "FLO_iota_edge",
            "fun": FLO_iota_edge,
            "bounds": iota_bounds,
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    "FLO_pressure_octic": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_pressure_octic",
            "fun": FLO_pressure_octic,
            "target": 0.0,
        },
    },
    "FLO_pressure_DOF": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_pressure_DOF",
            "fun": FLO_pressure_DOF,
            "target": 0.0,
        },
    },
    "FLO_pressure_edge": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_pressure_edge",
            "fun": FLO_pressure_edge,
            "target": 0.0,
        },
    },
    "FLO_grad_pressure_edge": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_grad_pressure_edge",
            "fun": FLO_grad_pressure_edge,
            "target": 0.0,
        },
    },
    # "FLO_iota_quadratic": {
    #     "use": not iota_fxd,
    #     "kwargs": {
    #         "name": "FLO_iota_quadratic",
    #         "fun": FLO_iota_quadratic,
    #         "target": 0.0,
    #     },
    # },
}

#===================================================================================================================================================










#===========================================================
# FNO OPT TOGGLES
#===========================================================

opt_toggles_FNO = {
    # "FNO_pressure_axis": {
    #     "use": not pressure_fxd,
    #     "kwargs": {
    #         "name": "FNO_pressure_axis",
    #         "fun": FNO_pressure_axis,
    #         "grid": data_grid,
    #         "bounds": FNO_pressure_axis_bounds,
    #         "weight": barrier_weights,
    #         "normalize": True,
    #     },
    # },
    "FNO_pressure_edge": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FNO_pressure_edge",
            "fun": FNO_pressure_edge,
            "grid": data_grid,
            "target": 0.0,
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    "FNO_pressure_positive": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FNO_pressure_positive",
            "fun": FNO_pressure_positive,
            "grid": data_grid,
            "bounds": (
                0.0,
                jnp.inf,
            ),
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    "FNO_pressure_monotonic": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FNO_pressure_monotonic",
            "fun": FNO_pressure_monotonic,
            "grid": data_grid,
            "bounds": (
                -jnp.inf,
                0.0,
            ),
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    "FNO_grad_pressure_edge": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FNO_grad_pressure_edge",
            "fun": FNO_grad_pressure_edge,
            "grid": data_grid,
            "target": 0.0,
            "weight": barrier_weights,
            "normalize": True,
        },
    },
    # "FNO_iota": {
    #     "use": not iota_fxd,
    #     "kwargs": {
    #         "name": "FNO_iota",
    #         "fun": FNO_iota,
    #         "grid": data_grid,
    #         "bounds": iota_bounds,
    #         "weight": barrier_weights,
    #         "normalize": True,
    #     },
    # },
}

#===================================================================================================================================================










#===========================================================
# OPT CONFIG
#===========================================================

OPT_CONFIG = {
    "optimizer":        optimizer,
    "ftol":             ftol,
    "xtol":             xtol,
    "gtol":             gtol,
    "maxiter":          maxiter,
    "max_nfev":         max_nfev,
    "x_scale":          x_scale,
    "opt_toggles_core": opt_toggles_core,
    "opt_toggles_FLO":  opt_toggles_FLO,
    "opt_toggles_FNO":  opt_toggles_FNO,
}

#===================================================================================================================================================










#===========================================================
# DRIVER CONFIG
#===========================================================

DRIVER_CONFIG = {
    "config_path": __file__,
    "selected_point": selected_point,
}

#===================================================================================================================================================
```
