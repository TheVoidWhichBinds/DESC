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
from desc.grid import LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from research.poly.poly_constraints import (
    # Objectives:
    pressure_axis_range,
    pressure_shape,
    pressure_monotone_obj,
    pressure_positive_obj,
    pressure_edge_obj,
    grad_pressure_edge_obj,
    iota_axis_range,
    iota_edge_range,
    # Constraints:
    pressure_axis_fxd,
    pressure_octic_con,
    pressure_DOF_con,
    pressure_edge_con,
    grad_pressure_edge_con,
)
from .driver import run_from_config
from research.poly.helper import(
    pressure_generator,
    iota_between_rationals,
)
#===================================================================================================================================================



#================ NOTES ============================================================================================================================
# DESC Part I recreation of D-shape equilibrium using proximal and Auglag
#===================================================================================================================================================










#================ EQUILIBRIUM INPUTS ================================================================================================================
#==================================
#----------------------------------
# Number of toroidal field periods:
NFP = 1
#-------

#------------------------
# Equilibrium resolution:
L = 16
M = 16
N = 16
eq_resolution = [L, M, N]
#------------------------

#----------------------
# Initializing surface:
surface_init = FourierRZToroidalSurface(
    R_lmn   = [ 3.51,   -1.0,  0.106],
    modes_R = [(0, 0), (1, 0), (2, 0)],
    Z_lmn   = [ 1.47,    0.16],
    modes_Z = [(-1, 0), (-2, 0)],
    NFP = NFP,
)
#-------------

#-----------------------
# Initializing pressure:
p_axis = 1600
pressure_init = PowerSeriesProfile(
    [1600, -3200, 1600],
    sym = True,
)
#--------------

#-------------------
# Initializing iota:
iota_axis_init = 1.0
iota_init = PowerSeriesProfile(
    [1, -0.67],
    sym = True,
)
#--------------
#==============


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
#===============================================================================================================================================










#================= OPTIMIZATION INPUTS =========================================================================================================
#=============
# AspectRatio:
aspect_ratio_bounds = (4, 12)

# Pressure:
pressure_axis_bounds = (1E4, 5E6)

# Iota:
iota_bounds = iota_between_rationals(iota_axis=iota_axis_init)
#=============================================================


#======================
# Optimizer thresholds:
ftol = 1e-3
xtol = 1e-6
gtol = 1e-6
maxiter = 100
max_nfev = 200
x_scale = "auto"
#===============




#===========================================
obj_grid = LinearGrid(L = 200, M = 0, N = 0)
#-----------------
custom_obj = False
#----------------
#------------------
pressure_fxd = True
#--------------------
#---------------
iota_fxd = True
#---------------


#=====================
# Shared toggle block:
opt_toggles_both = {
    # Objectives:
    #------------
        # Standard:
    #--------------------
    "forcebalance_obj": {
        "use": True,
        "kwargs": {
            "weight": 1e1,
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
    #---------------------

        # Custom:
    #-----------------------
    "pressure_axis_range": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis_range",
            "fun": pressure_axis_range,
            "bounds": pressure_axis_bounds,
            "weight": 1e12,
        },
    },
    "pressure_shape": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_shape",
            "fun": pressure_shape,
            "bounds": (-1.3, 1.8),
            "weight": 1e0,
        },
    },
    "iota_axis_range": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_axis_range",
            "fun": iota_axis_range,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    "iota_edge_range": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_edge_range",
            "fun": iota_edge_range,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    "pressure_monotone_obj": {
        "use": custom_obj,
        "kwargs": {
            "name": "pressure_monotone_obj",
            "fun": pressure_monotone_obj,
            "grid": obj_grid,
            "target": 0.0,
            "weight": 1e0,
        },
    },
    "pressure_positive_obj": {
        "use": custom_obj,
        "kwargs": {
            "name": "pressure_positive_obj",
            "fun": pressure_positive_obj,
            "grid": obj_grid,
            "target": 0.0,
            "weight": 1e0,
        },
    },
    "pressure_edge_obj": {
        "use": custom_obj,
        "kwargs": {
            "name": "pressure_edge_obj",
            "fun": pressure_edge_obj,
            "grid": obj_grid,
            "target": 0.0,
            "weight": 1e0,
        },
    },
    "grad_pressure_edge_obj": {
        "use": custom_obj,
        "kwargs": {
            "name": "grad_pressure_edge_obj",
            "fun": grad_pressure_edge_obj,
            "grid": obj_grid,
            "target": 0.0,
            "weight": 1e0,
        },
    },
    #---------------------


    # Constraints:
    #-------------
        # Standard:
    #--------------
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
        "use": True,
        "kwargs": {},
    },
    "fix_boundary_Z": {
        "use": True,
        "kwargs": {},
    },
    #----------------

        # Custom:
    #----------------------
    "pressure_octic_con": {
        "use": not custom_obj and not pressure_fxd,
        "kwargs": {
            "name": "pressure_octic_con",
            "fun": pressure_octic_con,
            "target": 0.0,
        },
    },
    "pressure_DOF_con": {
        "use": not custom_obj and not pressure_fxd,
        "kwargs": {
            "name": "pressure_DOF_con",
            "fun": pressure_DOF_con,
            "target": 0.0,
        },
    },
    "pressure_edge_con": {
        "use": not custom_obj and not pressure_fxd,
        "kwargs": {
            "name": "pressure_edge_con",
            "fun": pressure_edge_con,
            "target": 0.0,
        },
    },
    "grad_pressure_edge_con": {
        "use": not custom_obj and not pressure_fxd,
        "kwargs": {
            "name": "grad_pressure_edge_con",
            "fun": grad_pressure_edge_con,
            "target": 0.0,
        },
    },
}   #---------------------
#=========================




#==================
# Proximal toggles:
opt_toggles_prox = {
    # Objectives:
    #------------
    #------------


    # Constraints:
    #--------------------
    "forcebalance_con": {
        "use": True,
        "kwargs": {
            "target": 0.0,
        },
    },
}   #---------------------
#=========================




#================
# AugLag toggles:
opt_toggles_auglag = {
    # Objectives:
    #------------


    #------------
    # Constraints:
    #---------------------
    "pressure_axis_fxd": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis_fxd",
            "fun": pressure_axis_fxd,
            "target": p_axis,
        },
    },
    #------------------------
}
#============================




#=====================
# Grouping opt inputs:
opt_config = {
    "ftol":               ftol,
    "xtol":               xtol,
    "gtol":               gtol,
    "maxiter":            maxiter,
    "max_nfev":           max_nfev,
    "x_scale":            x_scale,
    "opt_toggles_both":   opt_toggles_both,
    "opt_toggles_prox":   opt_toggles_prox,
    "opt_toggles_auglag": opt_toggles_auglag,
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