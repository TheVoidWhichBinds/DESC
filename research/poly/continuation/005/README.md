```python
from research.poly.helper import (
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
    sym=True,
)
#--------------

#-------------------
# Initializing iota:
iota_axis_init = 1.0
iota_init = PowerSeriesProfile(
    [1, -0.67],
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




#=====================
# Toggle booleans left:
pressure_fxd = True
iota_fxd = True
#=====================

#=============================
# Grid for data-based customs:
data_grid = LinearGrid(L=200, M=0, N=0)
#=============================




#=====================
# Shared toggle block:
opt_toggles_all = {
    # Objectives:
    #------------
        # Standard:
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

    # Constraints:
    #-------------
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
}
#=========================




#=========================
# Proximal params toggles:
opt_toggles_prox_params = {
    # Objectives:
    #------------
    "pressure_axis_range_params": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis_range",
            "fun": pressure_axis_range_params,
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
    "iota_axis_range_params": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_axis_range",
            "fun": iota_axis_range_params,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    "iota_edge_range_params": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_edge_range",
            "fun": iota_edge_range_params,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },

    # Constraints:
    #-------------
    "forcebalance_con": {
        "use": True,
        "kwargs": {
            "target": 0.0,
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
    "pressure_edge_params": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_edge",
            "fun": pressure_edge_params,
            "target": 0.0,
        },
    },
    "grad_pressure_edge_params": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "grad_pressure_edge",
            "fun": grad_pressure_edge_params,
            "target": 0.0,
        },
    },
}
#=========================




#=======================
# Proximal data toggles:
opt_toggles_prox_data = {
    # Objectives:
    #------------
    "pressure_axis_range_data": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis_range",
            "fun": pressure_axis_range_data,
            "grid": data_grid,
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
    "iota_axis_range_data": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_axis_range",
            "fun": iota_axis_range_data,
            "grid": data_grid,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    "iota_edge_range_data": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_edge_range",
            "fun": iota_edge_range_data,
            "grid": data_grid,
            "bounds": iota_bounds,
            "weight": 1e0,
        },
    },
    "pressure_edge_data": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_edge_data",
            "fun": pressure_edge_data,
            "grid": data_grid,
            "target": 0.0,
            "weight": 1e0,
        },
    },
    "grad_pressure_edge_data": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "grad_pressure_edge_data",
            "fun": grad_pressure_edge_data,
            "grid": data_grid,
            "target": 0.0,
            "weight": 1e0,
        },
    },

    # Constraints:
    #-------------
    "forcebalance_con": {
        "use": True,
        "kwargs": {
            "target": 0.0,
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
    "pressure_edge_params": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_edge",
            "fun": pressure_edge_params,
            "target": 0.0,
        },
    },
    "grad_pressure_edge_params": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "grad_pressure_edge",
            "fun": grad_pressure_edge_params,
            "target": 0.0,
        },
    },
}
#=========================




#================
# AugLag toggles:
opt_toggles_auglag = {
    # Constraints:
    #-------------
    "pressure_axis_fxd": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis_fxd",
            "fun": pressure_axis_fxd,
            "target": p_axis,
        },
    },
}
#============================




#=====================
# Grouping opt inputs:
opt_config = {
    "ftol":                 ftol,
    "xtol":                 xtol,
    "gtol":                 gtol,
    "maxiter":              maxiter,
    "max_nfev":             max_nfev,
    "x_scale":              x_scale,
    "opt_toggles_all":      opt_toggles_all,
    "opt_toggles_prox_params": opt_toggles_prox_params,
    "opt_toggles_prox_data":   opt_toggles_prox_data,
    "opt_toggles_auglag":      opt_toggles_auglag,
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
        eq_config=eq_config,
        opt_config=opt_config,
        driver_config=driver_config,
    )

if __name__ == "__main__":
    main()
#===================================================================================================================================================
```
