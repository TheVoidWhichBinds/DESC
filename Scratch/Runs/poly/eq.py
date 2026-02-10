#BASIC (FIXED BOUNDARY) EQUILIBRIUM + BASIC QS OPTIMIZATION TUTORIAL MODDED 
#TO INCLUDE NON-FIXED PRESSURE PROFILE COEFFICIENTS AND MY CONSTRAINTS FOR 
#PRESSURE (3FUNC)

#Basic Equilibrium portion:
#Import:
import sys
import os
sys.path.append("/Users/macdaddi/DESC")
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile


#---------------------------------
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


#----------------------------------
# Initializing Pressure
def pressure_init(coeff):
    p_init = PowerSeriesProfile(coeff)  
    return p_init
    

#-----------------------------------
# Initializing Iota:
iota_init = PowerSeriesProfile([1, 0, 2]) 


#-----------------------------------
#Constructing Equilibrium 
eq = Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface = surface_init,
    pressure = pressure_init(),
    iota = iota_init,
    Psi=1.0,  # total flux, in Webers
)




#----------------------------------------------------------
# Solving & Saving Equilibrium:
eq_init = solve_continuation_automatic(eq.copy(), verbose=3)[-1] # final equilibrium
eq_init.save('/Users/macdaddi/DESC/scratch/runs/poly/eq.h5')

