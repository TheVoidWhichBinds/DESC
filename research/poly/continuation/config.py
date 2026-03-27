
#===================================================================================================================================================
from pathlib import Path
import os

#================ ENVIRONMENT TOGGLE =================#
USE_SUPERCOMPUTER = False
#====================================================#

from desc import set_device

if USE_SUPERCOMPUTER:
    set_device("gpu")

import jax

repo_root = Path(__file__).resolve().parents[3]

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
from .driver import run_from_config


def obj(use: bool, weight=None):
    return {"use": use, "weight": weight}
#===================================================================================================================================================










#===================================================================================================================================================
#============== EQUILIBRIUM INPUTS ==============#
#=========================
# Number of field periods:
NFP = 10

# Initializing fixed surface:
surface_init = FourierRZToroidalSurface(
    R_lmn=[10.0, -1.2, -0.25, 0.25],
    modes_R=[(0, 0), (1, 0), (1, 1), (-1, -1)],
    Z_lmn=[1.2, -0.2, -0.2],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=NFP,
)

# Initializing fixed iota:
iota_init = PowerSeriesProfile([1, 0, 2])

# Equilibrium resolution:
L = 8
M = 8
N = 3
eq_resolution = [L, M, N]
#========================


#====================
# Grouping eq inputs:
eq_config = {
    "NFP":           NFP,
    "surface_init":  surface_init,
    "iota_init":     iota_init,
    "eq_resolution": eq_resolution,
}
#==================================
#=================================================#










#============== OPTIMIZATION INPUTS ==============#
#======================
target_aspect_ratio = 6
ftol = 5e-3
xtol = 1e-4
gtol = 1e-3
maxiter = 20
#===========


#====================
optimizer_configs = [
    #--------------------------
    {  # 1st stage optimization
        "name": "proximal-lsq-exact", # optimizer 
        "toggle_FXD": { # fixed profiles
            # constraints
            "forcebalance_con": True,
            "fix_iota_con":     True,
            "fix_psi_con":      True,
            "fix_pressure_con": True,

            # objectives
            "forcebalance_obj": obj(True, 1e4),
            "aspect_ratio_obj": obj(True, 1e0),
            "qs_obj":           obj(True, 1e0),
            "ballooning_obj":   obj(True, 1e0),
            "mercier_obj":      obj(True, 1e0),
        },

        "toggle_CON": { # free profiles
            # constraints
            "pressure_axis_con":      True,
            "pressure_edge_con":      True,
            "grad_pressure_axis_con": True,
            "grad_pressure_edge_con": True,
            "forcebalance_con":       True,
            "fix_iota_con":           True,
            "fix_psi_con":            True,

            # objectives
            "forcebalance_obj": obj(True, 1e4),
            "aspect_ratio_obj": obj(True, 1e0),
            "qs_obj":           obj(True, 1e0),
            "ballooning_obj":   obj(True, 1e0),
            "mercier_obj":      obj(True, 1e0),
            "monotonicity_obj": obj(True, 1e0),
        },
    },
    #-------------------------------------------

    #--------------------------
    {  # 2nd stage optimization
        "name": None, # optimizer
        "toggle_FXD": { # fixed profiles
            # Constraints:
            "forcebalance_con": False,
            "fix_iota_con":     False,
            "fix_psi_con":      False,
            "fix_pressure_con": False,

            # Objectives:
            "forcebalance_obj": obj(False, None),
            "aspect_ratio_obj": obj(False, None),
            "qs_obj":           obj(False, None),
            "ballooning_obj":   obj(False, None),
            "mercier_obj":      obj(False, None),
        },

        "toggle_CON": { # free profiles
            # Constraints:
            "pressure_axis_con":      False,
            "pressure_edge_con":      False,
            "grad_pressure_axis_con": False,
            "grad_pressure_edge_con": False,
            "forcebalance_con":       False,
            "fix_iota_con":           False,
            "fix_psi_con":            False,

            # Objectives:
            "forcebalance_obj": obj(False, None),
            "aspect_ratio_obj": obj(False, None),
            "qs_obj":           obj(False, None),
            "ballooning_obj":   obj(False, None),
            "mercier_obj":      obj(False, None),
            "monotonicity_obj": obj(False, None),
        },
    },
]   #--------------------------------------------
#================================================


#=====================
# Grouping opt inputs:
opt_config = {
    "target_aspect_ratio": target_aspect_ratio,
    "ftol":                ftol,
    "xtol":                xtol,
    "gtol":                gtol,
    "maxiter":             maxiter,
    "optimizer_configs":   optimizer_configs,
}
#=============================================
#=================================================#










#================ DRIVER INPUTS ==================#
#=========================
# Pressure maxima to test:
p_maxima = [1e4]

# Polynomial orders to test:
n_set = [3]
#==========


#========================
# Grouping driver inputs:
driver_config = {
    "p_maxima": p_maxima,
    "n_set":    n_set,
}
#=====================
#=================================================#
#===================================================================================================================================================










#===================================================================================================================================================
#==================== RUN IT =====================#
def main():
    run_from_config(
        eq_config=eq_config,
        opt_config=opt_config,
        driver_config=driver_config,
    )

if __name__ == "__main__":
    main()
#=================================================#
#===================================================================================================================================================