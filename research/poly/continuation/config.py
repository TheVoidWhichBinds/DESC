import jax
print("jax devices:", jax.devices())
import os

from desc import set_device
print("after importing set_device:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))

set_device("gpu")   # or whatever your intended call is
print("after set_device:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))

print("before desc import:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))

from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from .driver import run_from_config







#===================================================================================================================================================
#============== EQUILIBRIUM INPUTS ==============#
#-------------------------
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
#------------------------


#--------------------
# Grouping eq inputs:
eq_config = {
    "NFP": NFP,
    "surface_init": surface_init,
    "iota_init": iota_init,
    "eq_resolution": eq_resolution,
}
#----------------------------------
#=================================================#










#============== OPTIMIZATION INPUTS ==============#
#----------------------
target_aspect_ratio = 6
ftol = 5e-3
xtol = 1e-4
gtol = 1e-3
maxiter = 20

# Fixed-pressure-family toggles (bools):
toggle_FXD = {
    "forcebalance_constraint": True,
    "fix_iota": True,
    "fix_psi": True,
    "fix_pressure": True,
    "forcebalance_objective": True,
    "aspect_ratio": True,
    "qs": True,
    "ballooning": True,
    "mercier": True,
}

# Optimized-pressure-family toggles (bools):
toggle_CON = {
    "pressure_axis": True,
    "pressure_edge": True,
    "grad_pressure_axis": True,
    "grad_pressure_edge": True,
    "forcebalance_constraint": True,
    "fix_iota": True,
    "fix_psi": True,
    "forcebalance_objective": True,
    "aspect_ratio": True,
    "qs": True,
    "ballooning": True,
    "mercier": True,
    "monotonicity": True,
}

# Ordered list of optimizers:
# [optimizer1, optimizer2]
optimizers = ["proximal-lsq-exact", None]
#----------------------------------------


#---------------------
# Grouping opt inputs:
opt_config = {
    "target_aspect_ratio": target_aspect_ratio,
    "ftol": ftol,
    "xtol": xtol,
    "gtol": gtol,
    "maxiter": maxiter,
    "toggle_FXD": toggle_FXD,
    "toggle_CON": toggle_CON,
    "optimizers": optimizers,
}
#----------------------------
#=================================================#










#================ DRIVER INPUTS ==================#
#-------------------
# GPU accessibility:
GPU_ON = True
if GPU_ON:
    set_device("gpu")

# Pressure maxima to test:
p_maxima = [1e4]

# Polynomial orders to test:
n_set = [3]
#----------


#------------------------
# Grouping driver inputs:
driver_config = {
    "p_maxima": p_maxima,
    "n_set": n_set,
}
#------------------
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
