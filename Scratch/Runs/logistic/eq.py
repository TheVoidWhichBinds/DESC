#BASIC (FIXED BOUNDARY) EQUILIBRIUM MODDED WITH LOGISTIC PROFILE
#NOTES:


#Imports:
import numpy as np
import sys
import os
sys.path.append("/Users/macdaddi/DESC")
import desc.io
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from scratch.profiles.logistic import logistic_opt




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

# Initializing Pressure and Iota:
pressure = logistic_opt(np.random.rand(5**2)) #rand(#), # is N_k * N_shift
iota = PowerSeriesProfile([1, 0, 1.5])  # 1 + 1.5 r^2

# Constructing Equilibrium 
eq = Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface=surf,
    pressure=pressure,
    iota=iota,
    Psi=1.0,  # total flux, in Webers
)




#----------------------------------------------------------
# Solving & Saving Equilibrium:
eq_init= solve_continuation_automatic(eq.copy(), verbose=3)
eq_init.save("/Users/macdaddi/DESC/scratch/runs/logistic/eq.h5")
