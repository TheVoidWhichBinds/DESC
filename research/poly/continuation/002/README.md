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
from research.poly.poly_constraints import (
    pressure_axis_range,
    pressure_shape,
    pressure_octic,
    pressure_DOF,
    pressure_edge,
    grad_pressure_edge,
    iota_axis_range,
    iota_edge_range,
)
from .driver import run_from_config
from research.poly.helper import(
    pressure_generator,
    iota_between_rationals,
)
#===================================================================================================================================================










#================ EQUILIBRIUM INPUTS ================================================================================================================
#==================================
#----------------------------------
# Number of toroidal field periods:
NFP = 4
#-------

#------------------------
# Equilibrium resolution:
L = 16 
M = 16 
N = 8 
eq_resolution = [L, M, N]
#------------------------

#----------------------
# Initializing surface:
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
p_axis = 5E4
pressure_init = PowerSeriesProfile(
    pressure_generator(
        p_axis = p_axis,
        width_percentage = 50
    ),
    sym = True,
)
#--------------

#-------------------
# Initializing iota:
iota_axis_init = 0.76
iota_init = PowerSeriesProfile(
    [iota_axis_init, 0.23], 
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
#=============
#-------------
# AspectRatio:
aspect_ratio_bounds = (4, 12)
#----------------------------

#----------
# Pressure:
pressure_axis_bounds = (5E4, 5E6)
#--------------------------------

#------
# Iota:
iota_bounds = iota_between_rationals(iota_axis=iota_axis_init)
#-------------------------------------------------------

#----------------------
# Optimizer thresholds:
ftol = 1e-3
xtol = 1e-6
gtol = 1e-6
maxiter = 100
max_nfev = 200
x_scale = "auto"
#---------------
#===============


#==============
opt_toggles = {
    #------------
    # Objectives:
    #------------
        # Standard:
    "forcebalance_obj": {
        "use": True,
        "kwargs": {
            "weight": 1e2,
            "target": 0.0,
        },
    },
    "aspect_ratio": {
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
            "bounds": (0.05, jnp.inf),
            "weight": 1e0,
        },
    },

        # Custom:
    "pressure_axis_range": {
        "use": True,
        "kwargs": {
            "name": "pressure_axis_range",
            "fun": pressure_axis_range,
            "bounds": pressure_axis_bounds,
            "weight": 1e0,
        },
    },
    "pressure_shape": {
        "use": True,
        "kwargs": {
            "name": "pressure_shape",
            "fun": pressure_shape,
            "bounds": (-1.3, 1.8),
            "weight": 1e0,
        },
    },
    "iota_axis_range": {
        "use": True,
        "kwargs": {
            "name": "iota_axis_range",
            "fun": iota_axis_range,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    "iota_edge_range": {
        "use": True,
        "kwargs": {
            "name": "iota_edge_range",
            "fun": iota_edge_range,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    #---------------------


    # Constraints:
    #-------------
        # Standard:
    "forcebalance_con": { # only used for proximal
        "use": True,
        "kwargs": {
            "target": 0.0,
        },
    },
    "fix_psi": {
        "use": True,
        "kwargs": {},
    },

        # Custom:
    "pressure_octic": {
        "use": True,
        "kwargs": {
            "name": "pressure_octic",
            "fun": pressure_octic,
            "target": 0.0,
        },
    },
    "pressure_DOF": {
        "use": True,
        "kwargs": {
            "name": "pressure_DOF",
            "fun": pressure_DOF,
            "target": 0.0,
        },
    },
    "pressure_edge": {
        "use": True,
        "kwargs": {
            "name": "pressure_edge",
            "fun": pressure_edge,
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
}
#=========================


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
