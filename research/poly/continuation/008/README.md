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
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    pressure_monotonicity_generator,
    iota_edge,
    iota_axis,
    grad_iota_axis,
)
from .driver import run_from_config
from .opt import resolve_from_context
from research.poly.helper import(
    pressure_generator,
    iota_between_rationals,
)
#===================================================================================================================================================













#================EQUILIBRIUM INPUTS ================================================================================================================
#=================================
#---------------------------------
# Number of toroidalfield periods:
NFP = 4
#-------

#------------------------
# Equilibrium resolution:
L = 8
M = 8
N = 4
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
p_axis = 1e4
n = L
pressure_init = PowerSeriesProfile(
    pressure_generator(
        p_axis = p_axis,
        n = n
    ),
    sym = False
)
#--------------

#-------------------
# Initializing iota:
iota_init_axis = 0.52
iota_init = PowerSeriesProfile([iota_init_axis, 0, 0.15])
#--------------------------------------------------------
#========================


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
#==============================
#------------------------------
# AspectRatio objective target:
target_aspect_ratio = 6
#----------------------

#--------------
# Iota targets:
iota_lower, iota_upper = iota_between_rationals(iota_axis = iota_init_axis)
#--------------------------------------------------------------------------

#----------------------
# Optimizer thresholds:
ftol = 5e-4
xtol = 1e-4
gtol = 1e-3
ctol = 1e-8
maxiter = 2
#----------
#==========


#==============
opt_toggles = [
    #==========================
    {  # 1st stage optimization
        "name": "proximal-lsq-exact", # optimizer

        #===============================
        "toggle_FXD": { # fixed profiles
            #------------
            # Objectives:
                # Standard:
            "forcebalance": {
                "use": True,
                "kwargs": {
                    "weight": 1e4,
                    "target": 0.0,
                },
            },
            "aspect_ratio": {
                "use": True,
                "kwargs": {
                    "weight": 1e0,
                    "target": target_aspect_ratio,
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
            },

            #-------------
            # Constraints:
                # Standard:
            "forcebalance": {
                "use": True,
                "kwargs": {
                    "target": 0.0,
                },
            },
            "fix_iota": {
                "use": True,
                "kwargs": {},
            },
            "fix_psi": {
                "use": True,
                "kwargs": {},
            },
            "fix_pressure": {
                "use": True,
                "kwargs": {},
            },
        },
        #=========================

        #==============================
        "toggle_CON": { # free profiles
            #------------
            # Objectives:
                # Standard:
            "forcebalance": {
                "use": True,
                "kwargs": {
                    "weight": 1e4,
                    "target": 0.0,
                },
            },
            "aspect_ratio": {
                "use": True,
                "kwargs": {
                    "weight": 1e0,
                    "target": target_aspect_ratio,
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
            },

            #-------------
            # Constraints:
                # Standard:
            "forcebalance": {
                "use": True,
                "kwargs": {
                    "target": 0.0,
                },
            },
            "fix_iota": {
                "use": False,
                "kwargs": {},
            },
            "fix_psi": {
                "use": True,
                "kwargs": {},
            },
            "fix_pressure": {
                "use": False,
                "kwargs": {},
            },

                # Custom:
            "pressure_axis": {
                "use": True,
                "kwargs": {
                    "name": "pressure_axis",
                    "fun": pressure_axis,
                    "target": p_axis,
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
            "pressure_monotonicity": {
                "use": True,
                "kwargs": {
                    "name": "pressure_monotonicity",
                    "fun": pressure_monotonicity_generator(L),
                    "target": jnp.zeros(2 * L)
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
                "use": True,
                "kwargs": {
                    "name": "iota_axis",
                    "fun": iota_axis,
                    "target": iota_lower,
                },
            },
            "iota_edge": {
                "use": False,
                "kwargs": {
                    "name": "iota_edge",
                    "fun": iota_edge,
                    "target": iota_upper,
                },
            },
        },  #==================================
    },  #======================================
]
#========================================


#=====================
# Grouping opt inputs:
opt_config = {
    "ftol":        ftol,
    "xtol":        xtol,
    "gtol":        gtol,
    "ctol":        ctol,
    "maxiter":     maxiter,
    "opt_toggles": opt_toggles,
}
#==============================
#===================================================================================================================================================










#================ DRIVER INPUTS ===================================================================================================================
driver_config = {
    "p_axis": p_axis,
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
