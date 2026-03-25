import sys
import os
from math import comb
sys.path.append("/Users/macdaddi/DESC")

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile




#============== FIXED INITIAL PARAMETERS ======================================================================================================================
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
#==============================================================================================================================================================










#============== POLYNOMIAL INITIALIZER ========================================================================================================================
def coefficients(p_scale, n, min_n=2):
    """
    Coeffs for p(rho) = p_scale * (1 - rho^2)^n
    Guarantees: p(0)=p_scale, p(1)=0, p'(0)=0, p'(1)=0 for n>=2
    Also nonnegative + monotone decreasing on [0,1].
    """
    n_eff = max(int(n), int(min_n))
    coeff = [0.0] * (2 * n_eff + 1)
    for k in range(n_eff + 1):
        coeff[2 * k] = p_scale * comb(n_eff, k) * ((-1) ** k)
    return coeff, n_eff
#==============================================================================================================================================================









#============== EQUILIBRIUM SOLVER ============================================================================================================================
def run_equilibrium(p_scale, n, out_dir):
    """
    Runs equilibirum solve given an on-axis pressure,
    and polynomial order n
    
    Returns: 
        save_path = location where eq_init saved
        n_eff = n if n>=2
    """

    # Creating polynomial coefficients in list form:
    # Checking that n>=2:
    coeff, n_eff = coefficients(p_scale, n)

    # Prepping equilibrium:
    eq = Equilibrium(
        L=8, M=8, N=3,
        surface=surface_init,
        pressure=PowerSeriesProfile(coeff),
        iota=iota_init,
        Psi=1.0,
    )

    # Solving initial equilibrium and returning last step of opt:
    eq_init = solve_continuation_automatic(eq.copy(), verbose=3)[-1]

    save_path = os.path.join(out_dir, 'eq.h5')
    eq_init.save(save_path)

    return eq_init, n_eff
#=============================================================================================================================================================


