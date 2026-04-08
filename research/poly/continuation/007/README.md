```python

#===================================================================================================================================================
from pathlib import Path
repo_root = Path(__file__).resolve().parents[3]
import os
#================ ENVIRONMENT TOGGLE =================#
USE_SUPERCOMPUTER = False
#====================================================#
from desc import set_device
if USE_SUPERCOMPUTER:
    set_device("gpu")
from desc.backend import print_backend_info
print_backend_info()

import jax
import jax.numpy as jnp
if USE_SUPERCOMPUTER:
    cache_dir = repo_root / "jax-caches"
    cache_dir.mkdir(exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("jax devices:", jax.devices())

from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.grid import LinearGrid
from research.poly.poly_constraints import (
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    pressure_monotonicity_generator,
    pressure_monotonicity,
    iota_edge,
    iota_axis,
    grad_iota_axis,
    pressure_axis_range,
    iota_range,
    current_range,
)
from .driver import run_from_config
from research.poly.helper import(
    pressure_generator,
    iota_between_rationals,
)
#===================================================================================================================================================













#================EQUILIBRIUM INPUTS ================================================================================================================
#==================================
#----------------------------------
# Number of toroidal field periods:
NFP = 4
#-------

#------------------------
# Equilibrium resolution:
L = 6 #upgrade once GPU 
M = 6 #upgrade once GPU
N = 3 #upgrade once GPU
eq_resolution = [L, M, N]
#------------------------

#------------------------
# Initializing fixed????? surface:
surface_init = FourierRZToroidalSurface(
    R_lmn =   [ 10.0,   -1.0,   -0.3,    0.3   ],
    modes_R = [(0, 0), (1, 0), (1, 1), (-1, -1)],
    Z_lmn =   [  1.0,    -0.3,    -0.3  ],
    modes_Z = [(-1, 0), (-1, 1), (1, -1)],
    NFP = NFP,
)
#-------------

#-----------------------
# Initializing pressure:
p_init_axis = 1e5
n = int(4)
pressure_init = PowerSeriesProfile(
    pressure_generator(
        p_axis = p_init_axis,
        n = n
    ),
    sym = True,
)
#--------------------------------

#-------------------
# Initializing iota:
iota_init_axis = 0.34
iota_init = PowerSeriesProfile(
    [iota_init_axis, 0, 0.14], 
    sym=True,
)
#------------
#============


#====================
# Grouping eq inputs:
eq_config = {
    "NFP":           NFP,
    "surface_init":  surface_init,
    "pressure_init": pressure_init,
    "iota_init":     iota_init,
    "eq_resolution": eq_resolution,
}
#==================================
#=================================================#










#================= OPTIMIZATION INPUTS =========================================================================================================
#====================
#--------------------
# AspectRatio bounds:
aspect_ratio_bounds = (8, 12)
#----------------------------

#--------------------------------
pressure_axis_bounds = (1e4, 1e6)
pressure_grid_res = 50
pressure_grid = LinearGrid(L=pressure_grid_res, M=0, N=0, axis=True)
n_pressure = pressure_grid.num_nodes
#-----------------------------------

#--------------
# Iota targets:
iota_lower, iota_upper = iota_between_rationals(iota_axis = iota_init_axis)
iota_grid_res = 50
iota_grid = LinearGrid(L=iota_grid_res, M=0, N=0, axis=True)
n_iota = iota_grid.num_nodes
#---------------------------

#----------------------------------
current_lower, current_upper = 0, 1e4
current_grid_res = 50
current_grid = LinearGrid(L=current_grid_res, M=0, N=0, axis=True)
n_current = current_grid.num_nodes
#---------------------------------

#----------------------
# Optimizer thresholds:
ftol = 1e-3
xtol = 1e-6
gtol = 1e-8
maxiter = 10
max_nfev = 20
x_scale = "auto"
#---------------
#===============


#==============
opt_toggles = {
    "name": "proximal-lsq-exact", # optimizer
    #========================================
    #===============
    "toggle_BOTH": {
        #------------
        # Objectives:
        "forcebalance": {
            "use": True,
            "kwargs": {
                "weight": 1e1,
                "target": 0.0,
            },
        },
        "aspect_ratio_range": {
            "use": True,
            "kwargs": {
                "weight": 1e0,
                "bounds": aspect_ratio_bounds,
            },
        },
        "qs": {
            "use": True,
            "kwargs": {
                "weight": 1e0,
                "helicity": (1, NFP),
            },
        },
        "ballooning": {
            "use": True,
            "kwargs": {
                "weight": 1e0,
                "target": 0.0,
            },
        },
        "mercier": {
            "use": True,
            "kwargs": {
                "weight": 1e0,
                "target": 0.0,
            },
        }, #------------------
        
        #-------------
        # Constraints:
        "forcebalance": {
            "use": True,
            "kwargs": {
                "target": 0.0,
            },
        },
        "fix_psi": {
            "use": True,
            "kwargs": {},
        },
    },  #----------------
    #====================




    #===============================
    "toggle_FXD": { # fixed profiles
        #------------
        # Objectives:
        #------------

        #-------------
        # Constraints:
        "fix_iota": {
            "use": True,
            "kwargs": {},
        },
        "fix_pressure": {
            "use": True,
            "kwargs": {},
        },
    },  #----------------
    #=========================




    #===============================
    "toggle_FREE": { # free profiles
        #-----------
        # Objectives:
                # Custom:
        "pressure_axis_range": {
            "use": True,
            "kwargs": {
                "weight": 1e12,
                "grid": pressure_grid,
                "fun": pressure_axis_range,
                "bounds": pressure_axis_bounds,
                "name": "pressure_axis_range"
            },
        },
        "pressure_monotonicity": {
            "use": True,
            "kwargs": {
                "weight": 1e12,
                "fun": pressure_monotonicity,
                "target": 0.0,
                "name": "pressure_monotonicity"
            },
        },
        "iota_range": {
            "use": True,
            "kwargs": {
                "weight": 1e12,
                "grid": iota_grid,
                "fun": iota_range,
                "bounds": (iota_lower, iota_upper),
                "name": "iota_range"
            },
        },
        "current_range": {
            "use": True,
            "kwargs": {
                "weight": 1e12,
                "grid": current_grid,
                "fun": current_range,
                "bounds": (current_lower, current_upper),
                "name": "current_range"
            },
        },
        #--------------------------------------

        #-------------
        # Constraints:
            # Standard:
        "fix_iota": {
            "use": False,
            "kwargs": {},
        },
        "fix_pressure": {
            "use": False,
            "kwargs": {},
        },

            # Custom:
        "pressure_edge": {
            "use": True,
            "kwargs": {
                "name": "pressure_edge",
                "fun": pressure_edge,
                "target": 0.0,
            },
        },
        "grad_pressure_axis": {
            "use": True,
            "kwargs": {
                "name": "grad_pressure_axis",
                "fun": grad_pressure_axis,
                "target": 0.0,
            },
        },
        "grad_pressure_edge": {
            "use": True,
            "kwargs": {
                "name": "grad_pressure_edge",
                "fun": grad_pressure_edge,
                "target": 0.0,
            },
        },
        "pressure_axis": {
            "use": False,
            "kwargs": {
                "name": "pressure_axis",
                "fun": pressure_axis,
                "target": p_init_axis,
            },
        },
        "grad_iota_axis": {
            "use": True,
            "kwargs": {
                "name": "grad_iota_axis",
                "fun": grad_iota_axis,
                "target": 0.0,
            },
        },
        "iota_axis": {
            "use": False,
            "kwargs": {
                "name": "iota_axis",
                "fun": iota_axis,
                "target": iota_lower + 0.02,
            },
        },
        "iota_edge": {
            "use": False,
            "kwargs": {
                "name": "iota_edge",
                "fun": iota_edge,
                "target": iota_upper - 0.02,
            },
        },
    },  #-----------------------------------
}
#=====================================================


#=====================
# Grouping opt inputs:
opt_config = {
    "ftol":        ftol,
    "xtol":        xtol,
    "gtol":        gtol,
    "maxiter":     maxiter,
    "max_nfev":    max_nfev,
    "opt_toggles": opt_toggles,
    "x_scale":     x_scale,
}
#==============================
#===================================================================================================================================================










#================ DRIVER INPUTS ===================================================================================================================
driver_config = {
    "p_axis": p_init_axis,
    "n":    n,
    "config_path": __file__,
}
#===================================================================================================================================================










#==================== RUN IT =======================================================================================================================
def main():
    run_from_config(
        eq_config = eq_config,
        opt_config = opt_config,
        driver_config = driver_config,
    )

if __name__ == "__main__":
    main()
#===================================================================================================================================================
```
