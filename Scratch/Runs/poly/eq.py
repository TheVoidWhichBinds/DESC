#BASIC (FIXED BOUNDARY) EQUILIBRIUM + BASIC QS OPTIMIZATION TUTORIAL MODDED 
#TO INCLUDE NON-FIXED PRESSURE PROFILE COEFFICIENTS AND MY CONSTRAINTS FOR 
#PRESSURE (3FUNC)

#Basic Equilibrium portion:
#Import:
import sys
import os
from math import comb
sys.path.append("/Users/macdaddi/DESC")
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile




# Initializations:
#-----------------------------------------------------------
# Initializing Boundary Surface:
surface_init = FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[
        (0, 0),
        (1, 0),
        (1, 1),
        (-1, -1),
    ],  # (m,n) pairs corresponding to R_mn on previous line
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=19,
)


# Initializing Iota:
iota_init = PowerSeriesProfile([1, 0, 2]) 


# Generating polynomial coefficients that are even 
# & monotonic in [0,1]
def coefficients(p_scale, n):
    coeff = [0.0] * (2*n + 1)
    for k in range(n + 1):
        coeff[2*k] = p_scale * comb(n, k) * (-1)**k
    return coeff


# Pressure initialization:
def pressure_init(coeff):
    p_init = PowerSeriesProfile(coeff)  
    return p_init
    



#------------------------------------------------
# Looping over custom range of max pressures 
# and polynomial orders (2n maximum)
def run_equilibrium(p_scale, n):
    # Inputting parameters into coeff generator:
    coeff = coefficients(p_scale, n) 

    # Constructing Equilibrium:
    eq, eq_result = Equilibrium(
        L=8,
        M=8,
        N=3,
        surface=surface_init,
        pressure=pressure_init(coeff),
        iota=iota_init,
        Psi=1.0,
    )

    # Solving equilibrium:
    eq_init = solve_continuation_automatic(eq.copy(), verbose=3)[-1]
    
    # Saving equilibrium:
    dir_name = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir_name, f'eq_p{p_scale:.0e}_n{n}.h5')
    eq_init.save(save_path)

    # Returning solved equilibrium objective values:
    return eq_result
    



#-------------------------------------------------------
# Run single equilibrium with custom params
if __name__ == "__main__":
    base_path = "/Users/macdaddi/DESC/scratch/runs/poly"
    #
    p_scale = 1e4
    n = 1
    #
    save_path = f"{base_path}/eq_p{p_scale:.0e}_n{n}.h5"
    run_equilibrium(p_scale, n, save_path)


