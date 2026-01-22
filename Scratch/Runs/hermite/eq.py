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

# Initializing Pressure:
def hermite_initializer():
    rho = np.linspace(0,1,200) - 0.5
    p_0 = 1 - ( 1 / (1 + np.exp(-10 * rho))) # logistic function to initialize with arbitrary k value
    gradp_0 = p_0 * (1 - p_0) # exact derivative

    initial = HermiteSplineProfile(
        p_0, # num steps must match LinearGrid rho # in opt.py
        gradp_0,
        knots=None,
    )  
    return initial

# Generating initial pressure and initial slope



# Initializing iota:
iota = PowerSeriesProfile([1, 0, 2]) 

# Constructing Equilibrium: 
eq = Equilibrium(
    L = 8,  # radial resolution
    M = 8,  # poloidal resolution
    N = 3,  # toroidal resolution
    surface = surf,
    pressure = hermite_initializer(),
    iota = iota,
    Psi=1.0,  # total flux, in Webers
)




#----------------------------------------------------------
# Solving & Saving Equilibrium:
eq_init= solve_continuation_automatic(eq.copy(), verbose=3)
eq_init.save('/Users/macdaddi/DESC/scratch/runs/hermite/eq.h5')

