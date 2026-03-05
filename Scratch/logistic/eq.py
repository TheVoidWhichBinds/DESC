import sys
import os
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from logistic import LogisticProfile  
sys.path.append("/Users/macdaddi/DESC")
 






#---------- FIXED INITIAL PARAMETERS ------------------------------------------------------
# Initializing fixed surface: 
surface_init = FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[(0, 0), (1, 0), (1, 1), (-1, -1)],
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=19,
)

# Initializing fixed iota:
iota_init = PowerSeriesProfile([1, 0, 2])
#------------------------------------------------------------------------------------------










#------------------------ EQUILIBRIUM SOLVER -------------------------------------------------------
def run_equilibrium(
        p_axis, 
        k_range, 
        rho_range, 
        weights, 
        rho_grid, 
        out_dir
    ):
    """
    Runs equilibirum solve given an on-axis pressure,
    and polynomial order n
    
    Returns: 
        save_path = location where eq_init saved
        n_eff = n if n>=2
    """

    # creating initial pressure profile:
    pressure = LogisticProfile(
        p_axis = p_axis,
        k_range = k_range,
        rho_range = rho_range,
        weights = weights,
        rho_grid = rho_grid,
    )

    # Prepping equilibrium:
    eq = Equilibrium(
        L=8, M=8, N=3,
        surface=surface_init,
        pressure=pressure,
        iota=iota_init,
        Psi=1.0,
    )

    # Solving initial equilibrium and returning last step of opt:
    eq_init = solve_continuation_automatic(eq.copy(), pert_order=1, verbose=3)[-1]

    save_path = os.path.join(out_dir, 'eq.h5')
    eq_init.save(save_path)

    return eq_init
#-----------------------------------------------------------------------------------------


