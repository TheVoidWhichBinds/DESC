```python
#----------------------------------
# Number of toroidal field periods:
NFP = 4
#-------

#----------------------
# Initializing surface:
surface_init = FourierRZToroidalSurface(
    R_lmn   = [
        3.75, 
        -0.55, 
        -0.12, 
    ],
    modes_R = [
        (0, 0), 
        (1, 0), 
        (1, 1), 
    ],
    Z_lmn   = [
        0.55, 
        -0.12, 
    ],
    modes_Z = [
        (-1, 0), 
        (-1, 1), 
    ],
    NFP = NFP,
)
#-------------

#-----------------------
# Initializing pressure:
p_axis = 1.0e4
pressure_init = PowerSeriesProfile(
    [1.0e4, -2e4,  1.0e4],
    sym = True
)
#--------------

#-------------------
# Initializing iota:
iota_axis_init = 0.25
iota_init = PowerSeriesProfile(
    [0.25, -0.23],
    sym = True
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
# Pressure:
pressure_bounds = (1E4, 1E6)

# Iota:
iota_bounds = iota_between_rationals(iota_axis=iota_axis_init)
#=============================================================




#======================
# Optimizer thresholds:
ftol = 1e-4
xtol = 1e-8
gtol = 1e-8
maxiter = 300
max_nfev = 300
x_scale = "auto"
#===============




#=====================
# Toggle booleans left:
pressure_fxd = False
iota_fxd = True
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
    "forcebalance_obj": {
        "use": True,
        "kwargs": {
            "weight": 1e1,
            "target": 0.0,
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
    "pressure_axis_range": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_axis_range",
            "fun": pressure_axis_range,
            "bounds": pressure_bounds,
            "weight": 1e10
        },
    },
    "pressure_shape": {
        "use": not pressure_fxd,
        "kwargs": {
            "name": "pressure_shape",
            "fun": pressure_shape,
            "bounds": (-1.3, 1.8),
            "weight": 1e10,
        },
    },
    "iota_axis_range": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_axis_range",
            "fun": iota_axis_range,
            "bounds": iota_bounds,
            "weight": 1e10,
        },
    },
    "iota_edge_range": {
        "use": not iota_fxd,
        "kwargs": {
            "name": "iota_edge_range",
            "fun": iota_edge_range,
            "bounds": iota_bounds,
            "weight": 1e10,
        },
    },
    #---------------------


    # Constraints:
    #============
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
```
