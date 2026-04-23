```python
# Config.py
#===================================================================================================================================================
from pathlib import Path

import jax.numpy as jnp

from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile

from research.poly.FLO_vs_FNO.poly_constraints import (
    FLO_pressure_axis,
    FLO_pressure_shape,
    FLO_pressure_octic,
    FLO_pressure_DOF,
    FLO_pressure_edge,
    FLO_grad_pressure_edge,
    FLO_iota_axis,
    FLO_iota_edge,
    FLO_iota_quadratic,
    FNO_pressure,
    FNO_pressure_monotonic,
    FNO_grad_pressure_edge,
    FNO_iota,
)

from research.poly.exact_vs_auglag.helper import iota_between_rationals
#===================================================================================================================================================











#============== PATH / SOURCE CONFIG ================================================================================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
#===================================================================================================================================================











#============== EQUILIBRIUM INPUTS ==================================================================================================================
#==================================
#------------------------
# Equilibrium resolution:
L = 8
M = 8
N = 3
eq_resolution = [L, M, N]
#------------------------

#----------------------------------
# Number of toroidal field periods:
NFP = 4
#----------------------------------

#------------------------------------------
# Surface source from research/neural/NFP_*:
surface_source_config = {
    "dataset_path": str(
        REPO_ROOT
        / "research"
        / "neural"
        / f"NFP_{NFP}"
        / "dataset.pkl"
    ),
    "selection_method": "all_nested",
    "shuffle": True,
    "shuffle_seed": 42,
}

# Number of equilibria to test:
N_eq = 8
#------------------------------

#-----------------------
# Initializing pressure:
p_axis = 1.0e4
pressure_init = PowerSeriesProfile(
    [1.0e4, -2.0e4, 1.0e4],
    sym = True,
)
#-----------------------

#-------------------
# Initializing iota:
iota_axis_init = 0.25
iota_init = PowerSeriesProfile(
    [0.25, -0.23],
    sym = True,
)
#-------------------
#==================================


#==========================
# Grouping eq input config:
EQ_INPUT_CONFIG = {
    "NFP":                  NFP,
    "surface_source_config": surface_source_config,
    "pressure_init":        pressure_init,
    "iota_init":            iota_init,
    "eq_resolution":        eq_resolution,
    "N_eq":                 N_eq,
}
#================================
#===================================================================================================================================================











#============== OPTIMIZATION INPUTS ================================================================================================================
#===========================================
# Bounds shared by FLO / FNO objective sets:
FLO_pressure_axis_bounds = (1e4, 1e7)
FLO_pressure_shape_bounds = (-1.3, 1.8)
FNO_pressure_bounds = (0.0, 1e7)

iota_bounds = iota_between_rationals(iota_axis = iota_axis_init)
#===========================================




#======================
# Optimizer thresholds:
ftol = 1e-4
xtol = 1e-8
gtol = 1e-8
maxiter = 300
max_nfev = 300
x_scale = "auto"
#======================




#=====================
# Toggle booleans left:
pressure_fxd = False
iota_fxd = False
#=====================

#=============================
# Grid for data-based customs:
data_grid = LinearGrid(L = 200, M = 0, N = 0)

redl_grid = LinearGrid(
    L = 200,
    M = 24,
    N = 24,
    NFP = NFP,
)
#=============




#==================
# CORE OPT TOGGLES:
opt_toggles_core = {
    # Objectives:
    #====================
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
            "helicity": (1, NFP),
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
            "bounds": (0.05, jnp.inf),
            "weight": 1e0,
            "normalize": True,
        },
    },



    # Constraints:
    #=============
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
    "fix_boundary_R": {
        "use": False,
        "kwargs": {},
    },
    "fix_boundary_Z": {
        "use": False,
        "kwargs": {},
    },
}
#==================




#=================
# FLO OPT TOGGLES:
opt_toggles_FLO = {
    # Objectives:
    #============
    "FLO_pressure_axis": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_pressure_axis",
            "fun": FLO_pressure_axis,
            "bounds": FLO_pressure_axis_bounds,
            "weight": 1e10,
            "normalize": True,
        },
    },
    "FLO_pressure_shape": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FLO_pressure_shape",
            "fun": FLO_pressure_shape,
            "bounds": FLO_pressure_shape_bounds,
            "weight": 1e10,
            "normalize": True,
        },
    },
    "FLO_iota_axis": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "FLO_iota_axis",
            "fun": FLO_iota_axis,
            "bounds": iota_bounds,
            "weight": 1e10,
            "normalize": True,
        },
    },
    "FLO_iota_edge": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "FLO_iota_edge",
            "fun": FLO_iota_edge,
            "bounds": iota_bounds,
            "weight": 1e10,
            "normalize": True,
        },
    },

    # Constraints:
    #=============
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
    "FLO_iota_quadratic": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "FLO_iota_quadratic",
            "fun": FLO_iota_quadratic,
            "target": 0.0,
        },
    },
}
#=================




#=================
# FNO OPT TOGGLES:
opt_toggles_FNO = {
    # Objectives:
    #============
    "FNO_pressure": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FNO_pressure",
            "fun": FNO_pressure,
            "grid": data_grid,
            "bounds": FNO_pressure_bounds,
            "weight": 1e10,
            "normalize": True,
        },
    },
    "FNO_pressure_monotonic": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "FNO_pressure_monotonic",
            "fun": FNO_pressure_monotonic,
            "grid": data_grid,
            "bounds": (-jnp.inf, 0.0),
            "weight": 1e10,
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
            "weight": 1e10,
            "normalize": True,
        },
    },
    "FNO_iota": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "FNO_iota",
            "fun": FNO_iota,
            "grid": data_grid,
            "bounds": iota_bounds,
            "weight": 1e10,
            "normalize": True,
        },
    },

    # Constraints:
    #=============
}
#=================




#=====================
# Grouping opt inputs:
OPT_CONFIG = {
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
#=====================
#===================================================================================================================================================











#============== DRIVER INPUTS ======================================================================================================================
DRIVER_CONFIG = {
    "config_path": __file__,
    "execution_mode": "local",   # "auto", "local", or "cluster"
    "cluster_max_workers": 2,
}
#===================================================================================================================================================
```
