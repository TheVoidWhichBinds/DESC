#===================================================================================================================================================
from pathlib import Path
repo_root = Path(__file__).resolve().parents[4]
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
from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile

from research.poly.exact_vs_auglag.poly_constraints import (
    pressure_axis,
    pressure_shape,
    iota_axis_range,
    iota_edge_range,
    pressure_octic,
    pressure_DOF,
    pressure_edge,
    grad_pressure_edge,
    iota_quadratic,
)
from research.poly.exact_vs_auglag.helper import (
    iota_between_rationals,
    pressure_generator,
)
from .driver import run_from_config
#===================================================================================================================================================



#================ NOTES ============================================================================================================================
# DESC Part I recreation of D-shape equilibrium using proximal and Auglag
#===================================================================================================================================================










#================ EQUILIBRIUM INPUTS ================================================================================================================
#==================================
#----------------------------------
# Number of toroidal field periods:
NFP = 4
#-------

#------------------------
# Equilibrium resolution:
L = 8
M = 8
N = 4
eq_resolution = [L, M, N]
#------------------------

#----------------------
# Initializing surface:
surface_init = FourierRZToroidalSurface(
    R_lmn   = [3.50, -1.00, 0.04, 0.08],
    modes_R = [(0, 0), (1, 0), (2, 0), (1, 1)],
    Z_lmn   = [1.45, 0.04, 0.01, -0.08],
    modes_Z = [(-1, 0), (-2, 0), (-3, 0), (-1, 1)],
    NFP = NFP,
)
#-------------

#-----------------------
# Initializing pressure:
p_axis = 1e4
pressure_init = PowerSeriesProfile(
    [1e4, -2e4, 1e4],
    sym=True,
)
#--------------

#-------------------
# Initializing iota:
iota_axis_init = 1.0
iota_init = PowerSeriesProfile(
    [1, -0.23],
    sym=True,
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

# Iota:
iota_bounds = iota_between_rationals(iota_axis=iota_axis_init)
#=============================================================




#======================
# Optimizer thresholds:
ftol = 1e-6
xtol = 1e-8
gtol = 1e-8
maxiter = 300
max_nfev = 300
x_scale = "auto"
#===============




#=====================
# Toggle booleans left:
pressure_fxd = False
iota_fxd = False
#=====================

#=============================
# Grid for data-based customs:
data_grid = LinearGrid(L=200, M=0, N=0)
#=============================




#==================
# CORE OPT TOGGLES:
opt_toggles_core = {
    # Objectives:
    #==============
        # Standard:
    #--------------------
    # "forcebalance_obj": {
    #     "use": True,
    #     "kwargs": {
    #         "weight": 1e1,
    #         "target": 0.0,
    #     },
    # },
    "aspect_ratio": {
        "use": False,
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


    # Constraints:
    #=============
        # Standard:
    #----------------
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
    "forcebalance_con": {
        "use": True,
        "kwargs": {
            "target": 0.0,
        },
    },
    #---------------------
}
#=========================




#====================
# CUSTOM OPT TOGGLES:
opt_toggles_custom = {
    # Objectives:
    #============
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
    #---------------------


    # Constraints:
    #============
    "pressure_axis": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis",
            "fun": pressure_axis,
            "target": p_axis,
        },
    },
    "pressure_octic": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_octic",
            "fun": pressure_octic,
            "target": 0.0,
        },
    },
    "pressure_DOF": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_DOF",
            "fun": pressure_DOF,
            "target": 0.0,
        },
    },
    "pressure_edge": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_edge",
            "fun": pressure_edge,
            "target": 0.0,
        },
    },
    "grad_pressure_edge": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "grad_pressure_edge",
            "fun": grad_pressure_edge,
            "target": 0.0,
        },
    },
    "iota_quadratic": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_quadratic",
            "fun": iota_quadratic,
            "target": 0.0,
        },
    },
    #---------------------
}
#=========================




#=====================
# Grouping opt inputs:
opt_config = {
    "ftol":               ftol,
    "xtol":               xtol,
    "gtol":               gtol,
    "maxiter":            maxiter,
    "max_nfev":           max_nfev,
    "x_scale":            x_scale,
    "opt_toggles_core":   opt_toggles_core,
    "opt_toggles_custom": opt_toggles_custom,
}
#============================================
#===================================================================================================================================================










#================ DRIVER INPUTS ===================================================================================================================
driver_config = {
    "config_path": __file__,
}
#===================================================================================================================================================










#==================== RUN IT =======================================================================================================================
def main():
    run_from_config(
        eq_config=eq_config,
        opt_config=opt_config,
        driver_config=driver_config,
    )

if __name__ == "__main__":
    main()
#===================================================================================================================================================