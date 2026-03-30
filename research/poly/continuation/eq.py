import sys
import os
from math import comb
sys.path.append("/Users/macdaddi/DESC")

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile, SplineProfile



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
def run_equilibrium(p_scale, n, out_dir, eq_config):
    """
    Runs equilibrium solve given an on-axis pressure,
    and polynomial order n.

    Returns:
        eq_init = solved equilibrium
        n_eff = n if n>=2, else 2
    """

    #-----------------------------------------------
    # Unpacking equilibrium configuration variables:
    surface_init = eq_config["surface_init"]
    iota_init = eq_config["iota_init"]
    eq_resolution = eq_config["eq_resolution"]
    #-----------------------------------------------

    #-----------------------------------------------
    # Creating polynomial coefficients in list form:
    coeff, n_eff = coefficients(p_scale, n)
    pressure_init = PowerSeriesProfile(coeff)
    #-----------------------------------------------

    #----------------------
    # Prepping equilibrium:
    L, M, N = eq_resolution
    eq = Equilibrium(
        L=L, M=M, N=N,
        surface = surface_init,
        pressure = pressure_init,
        iota = iota_init,
        Psi = 1.0,
    )
    #----------------------

    #------------------------------------------------------------
    # Solving initial equilibrium and returning last step of opt:
    eq_init = solve_continuation_automatic(eq.copy(), verbose=3)[-1]
    #------------------------------------------------------------

    save_path = os.path.join(out_dir, "eq.h5")
    eq_init.save(save_path)

    return eq_init, n_eff
#==============================================================================================================================================================
