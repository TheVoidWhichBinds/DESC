import sys
sys.path.append("/Users/macdaddi/DESC")
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from driver import run_from_config





#============== EQUILIBRIUM INPUTS ==============#

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
eq_resolution = [8, 8, 3]

# Grouping eq inputs:
eq_config = {
    "NFP": NFP,
    "surface_init": surface_init,
    "iota_init": iota_init,
    "eq_resolution": eq_resolution,
}
#=================================================#





#============== OPTIMIZATION INPUTS ==============#
target_aspect_ratio = 6
ftol = 5e-3
xtol = 1e-3
gtol = 1e-2
maxiter = 20

# Fixed-pressure-family toggles (bools):
toggle_FXD = {
    "forcebalance_constraint": True,
    "fix_iota": True,
    "fix_psi": True,
    "fix_pressure": True,
    "forcebalance_objective": False,
    "aspect_ratio": True,
    "qs": True,
    "ballooning": False,
    "mercier": False,
}

# Optimized-pressure-family toggles (bools):
toggle_CON = {
    "pressure_axis": True,
    "pressure_edge": True,
    "grad_pressure_axis": True,
    "grad_pressure_edge": True,
    "forcebalance_constraint": False,
    "fix_iota": True,
    "fix_psi": True,
    "forcebalance_objective": False,
    "aspect_ratio": True,
    "qs": True,
    "ballooning": False,
    "mercier": False,
    "monotonicity": True,
}

# Ordered list of optimizers:
# [optimizer1, optimizer2]
optimizers = ["proximal-lsq-exact", None]

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
#=================================================#





#============== DRIVER INPUTS ====================#
# Pressure maxima to test:
p_maxima = [1e4]

# Polynomial orders to test:
n_set = [3]

# Grouping driver inputs:
driver_config = {
    "p_maxima": p_maxima,
    "n_set": n_set,
}
#=================================================#





#============== RUN IT =======================================================================================================================================
if __name__ == "__main__":
    run_from_config(
        eq_config=eq_config,
        opt_config=opt_config,
        driver_config=driver_config,
    )
#==============================================================================================================================================================