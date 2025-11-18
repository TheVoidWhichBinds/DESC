#BASIC (FIXED BOUNDARY) EQUILIBRIUM + BASIC QS OPTIMIZATION TUTORIAL MODDED TO INCLUDE NON-FIXED PRESSURE PROFILE COEFFICIENTS AND MY CONSTRAINTS FOR PRESSURE (3FUNC)

#Basic Equilibrium portion:

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

import desc.io
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile


#initializing boundary surface:
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

#initializing pressure and iota:
pressure = PowerSeriesProfile(
    [1.8e4, 0, -3.6e4, 0, 1.8e4]
)  
iota = PowerSeriesProfile([1, 0, 1.5])  # 1 + 1.5 r^2

eq = Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface=surf,
    pressure=pressure,
    iota=iota,
    Psi=1.0,  # total flux, in Webers
)

eq_init= solve_continuation_automatic(eq.copy(), verbose=3)
