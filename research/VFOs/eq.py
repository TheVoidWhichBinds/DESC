# eq.py
#===================================================================================================================================================

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium

#===================================================================================================================================================










#===========================================================
# EQUILIBRIUM SOLVER
#===========================================================

def run_equilibrium(
        eq_config,
    ):
    """
    Solves an initial equilibrium from eq_config.
    """

    surface_init = eq_config["surface_init"]
    pressure_init = eq_config["pressure_init"]
    iota_init = eq_config["iota_init"]
    eq_resolution = eq_config["eq_resolution"]

    L, M, N = eq_resolution

    eq = Equilibrium(
        L = L,
        M = M,
        N = N,
        surface = surface_init,
        pressure = pressure_init,
        iota = iota_init,
        Psi = 1.0,
    )

    eq_init = solve_continuation_automatic(
        eq.copy(),
        verbose = 3,
    )[-1]

    return eq_init

#===================================================================================================================================================