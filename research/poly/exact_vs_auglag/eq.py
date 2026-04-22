import sys
sys.path.append("/Users/macdaddi/DESC")
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium










#============== EQUILIBRIUM SOLVER ============================================================================================================================
def run_equilibrium(
        eq_config
    ):
    """
    Solves equilibrium given initial conditions.
    """

    #-----------------------------------------------
    # Unpacking equilibrium configuration variables:
    surface_init = eq_config["surface_init"]
    pressure_init = eq_config["pressure_init"]
    iota_init = eq_config["iota_init"]
    eq_resolution = eq_config["eq_resolution"]
    #--------------------------------------------------

    #----------------------
    # Prepping equilibrium:
    L, M, N = eq_resolution
    eq = Equilibrium(
        L=L, M=M, N=N,
        surface = surface_init,
        pressure = pressure_init,
        iota = iota_init,
        Psi=1.0,
        ensure_nested = True
    )
    #----------------------

    #------------------------------------------------------------
    # Solving initial equilibrium and returning last step of opt:
    eq_init = solve_continuation_automatic(eq.copy(), verbose=3)[-1]
    #---------------------------------------------------------------

    return eq, eq_init
#==============================================================================================================================================================