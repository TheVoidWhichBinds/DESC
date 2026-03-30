
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
from .driver import run_from_config
#===================================================================================================================================================











#===================================================================================================================================================
#============== HELPER FUNCTIONS ==============#
#==========================
def iota_between_rationals(
    iota_axis: float,
):
    """
    Generates bounds for iota optimizer constraint
    that are between low-order rational surfaces.
    """
    allowed_ranges = [
        (0.25,   0.3333),
        (0.3333, 0.5),
        (0.5,    0.6667),
        (0.6667, 0.75),
        (0.75,   1.0),
        (1.0,    1.3333),
        (1.3333, 1.5),
        (1.5,    2.0),
        (2.0,    3.0),
        (3.0,    4.0),
    ]

    matched_lower = None
    matched_upper = None

    for lower, upper in allowed_ranges:
        if lower <= iota_axis <= upper:
            matched_lower = lower
            matched_upper = upper
            break

    if matched_lower is None:
        raise ValueError("iota_axis is outside all allowed rational intervals.")

    lower_bound = matched_lower
    upper_bound = matched_upper 

    return lower_bound, upper_bound
#==================================


#===============================
def obj(use: bool, weight=None):
    """
    Compactifies True/False toggle for objectives,
    and their weights in the optimizer.
    """
    return {"use": use, "weight": weight}
#========================================
#=================================================#
#===================================================================================================================================================










#===================================================================================================================================================
#============== EQUILIBRIUM INPUTS ==============#
#=========================
#-------------------------
# Number of field periods:
NFP = 19
#-------

#----------------------------
# Initializing fixed surface:
surface_init = FourierRZToroidalSurface(
    R_lmn =   [ 10.0,   -1.0,   -0.3,    0.3   ],
    modes_R = [(0, 0), (1, 0), (1, 1), (-1, -1)],
    Z_lmn =   [  1.0,    -0.3,    -0.3  ],
    modes_Z = [(-1, 0), (-1, 1), (1, -1)],
    NFP = NFP,
)
#-------------

#-------------------------
# Initializing iota:
iota_axis = 0.52
iota_init = PowerSeriesProfile([iota_axis, 0, 0.07])
iota_lower, iota_upper = iota_between_rationals(iota_axis = iota_axis)
#---------------------------------------------------------------------

#------------------------
# Equilibrium resolution:
L = 8
M = 8
N = 3
eq_resolution = [L, M, N]
#------------------------
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
#=============================
#-----------------------------
# AspectRatio objective target
target_aspect_ratio = 6
#----------------------


#----------
ftol = 5e-4
xtol = 1e-4
gtol = 1e-3
maxiter = 2
#-----------
#===========


#==============
opt_toggles = [
    #==========================
    {  # 1st stage optimization
        #----------------------------------------
        "name": "proximal-lsq-exact", # optimizer 
        #----------------------------------------

        #-------------------------------
        "toggle_FXD": { # fixed profiles
            # Constraints:
                # Standard:
            "forcebalance_con": True,
            "fix_iota_con":     True,
            "fix_psi_con":      True,
            "fix_pressure_con": True,

            # Objectives:
                # Standard:
            "forcebalance_obj": obj(True, 1e4),
            "aspect_ratio_obj": obj(True, 1e0),
            "qs_obj":           obj(True, 1e0),
            "ballooning_obj":   obj(True, 1e0),
            "mercier_obj":      obj(True, 1e0),
        },
        #--------------------------------------

        #------------------------------
        "toggle_CON": { # free profiles
            # Constraints:
                # Standard:
            "forcebalance_con":       True,
            "fix_iota_con":           False,
            "fix_psi_con":            True,
            "fix_pressure_con":       False,
                # Custom:
            "pressure_axis_con":      True,
            "pressure_edge_con":      True,
            "grad_pressure_axis_con": True,
            "grad_pressure_edge_con": True,
            "grad_iota_axis_con":     True,
            "iota_rationals_con":     True, 
            
            # Objectives:
                # Standard:
            "forcebalance_obj":   obj(True, 1e4),
            "aspect_ratio_obj":   obj(True, 1e0),
            "qs_obj":             obj(True, 1e0),
            "ballooning_obj":     obj(True, 1e0),
            "mercier_obj":        obj(True, 1e0),
                # Custom:
            "monotonicity_obj":   obj(True, 1e0),
        },
        #--------------------------------------
    },
    #==========================================


    # #==========================
    # {  # 2nd stage optimization
    #     #----------------------------------------
    #     "name": None, # optimizer 
    #     #----------------------------------------

    #     #-------------------------------
    #     "toggle_FXD": { # fixed profiles
    #         # Constraints:
    #             # Standard:
    #         "forcebalance_con": True,
    #         "fix_iota_con":     True,
    #         "fix_psi_con":      True,
    #         "fix_pressure_con": True,

    #         # Objectives:
    #             # Standard:
    #         "forcebalance_obj": obj(True, 1e4),
    #         "aspect_ratio_obj": obj(True, 1e0),
    #         "qs_obj":           obj(True, 1e0),
    #         "ballooning_obj":   obj(True, 1e0),
    #         "mercier_obj":      obj(True, 1e0),
    #     },
    #     #--------------------------------------

    #     #------------------------------
    #     "toggle_CON": { # free profiles
    #         # Constraints:
    #             # Standard:
    #         "forcebalance_con":       True,
    #         "fix_iota_con":           False,
    #         "fix_psi_con":            True,
    #         "fix_pressure_con":       False,
    #             # Custom:
    #         "pressure_axis_con":      True,
    #         "pressure_edge_con":      True,
    #         "grad_pressure_axis_con": True,
    #         "grad_pressure_edge_con": True,
    #         "grad_iota_axis_con":     True,
    #         "iota_rationals_con":     True,
            

    #         # Objectives:
    #             # Standard:
    #         "forcebalance_obj":   obj(True, 1e4),
    #         "aspect_ratio_obj":   obj(True, 1e0),
    #         "qs_obj":             obj(True, 1e0),
    #         "ballooning_obj":     obj(True, 1e0),
    #         "mercier_obj":        obj(True, 1e0),
    #             # Custom:
    #         "monotonicity_obj":   obj(True, 1e0),
    #     },
    #     #--------------------------------------
    # },
]   #==========================================
#================================================


#=====================
# Grouping opt inputs:
opt_config = {
    "target_aspect_ratio": target_aspect_ratio,
    "iota_upper":          iota_upper,
    "iota_lower":          iota_lower,
    "ftol":                ftol,
    "xtol":                xtol,
    "gtol":                gtol,
    "maxiter":             maxiter,
    "opt_toggles":         opt_toggles,
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