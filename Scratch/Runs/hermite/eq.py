#BASIC (FIXED BOUNDARY) EQUILIBRIUM + BASIC QS OPTIMIZATION TUTORIAL MODDED 
#TO INCLUDE NON-FIXED PRESSURE PROFILE COEFFICIENTS 

# Notes: 
# Seems like bad practice to name Equilibrium() func inputs by the same name, e.g. iota = iota. 
#Fix in other eq.py files.
# 


#Basic Equilibrium portion:
#Import:
import sys
import numpy as np
import os
sys.path.append("/Users/macdaddi/DESC")
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import HermiteSplineProfile, PowerSeriesProfile

import matplotlib.pyplot as plt




#---------------------------------
# Initializing Boundary Surface:
surf= FourierRZToroidalSurface(
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


# Initializing pressure:
rho = np.linspace(0,1,10)
p_0 = 1 - ( 1 / (1 + np.exp(-20 * (rho - 0.5)))) # logistic function to initialize with arbitrary k value
gradp_0 = p_0 * (1 - p_0) # exact derivative
pressure_init = HermiteSplineProfile(
    p_0, 
    gradp_0,
    knots=rho,
)  


# Initializing iota:
iota = PowerSeriesProfile([1, 0, 2]) 




#----- Constructing Equilibrium -----# 
eq = Equilibrium( 
    # Nodes over which equilibrium solved
    L = 8,  # radial resolution
    M = 8,  # poloidal resolution
    N = 3,  # toroidal resolution
    surface = surf,
    pressure = pressure_init,
    iota = iota,
    Psi=1.0,  # total flux, in Webers
)




#----------------------------------------------------------
# Solving & Saving Final Equilibrium:
eq_0 = solve_continuation_automatic(eq.copy(), verbose=3)[-1]
eq_0.save('/Users/macdaddi/DESC/scratch/runs/hermite/eq.h5')

