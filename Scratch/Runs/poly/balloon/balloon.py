import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import desc
from desc.grid import Grid, LinearGrid
from desc.optimize import Optimizer
from desc.objectives import (
    ForceBalance,
    AspectRatio,
    FixIota,
    FixPressure,
    FixPsi,
    PrincipalCurvature,
    BallooningStability,
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    GenericObjective,
)

plt.rcParams["font.size"] = 14
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))



#-------------------------- EQUILIBRIUM SOLVE & PLOTTING --------------------------------------------------------------
# Importing the HELIOTRON DESC equilibrium:
eq0 = desc.examples.get("HELIOTRON")
# Reduce resolution so it runs a bit quicker:
eq0.change_resolution(L=12, M=6, N=2, L_grid=18, M_grid=12, N_grid=4)
eq0.surface = eq0.get_surface_at(rho=1)


surfaces = np.array([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]) # flux surfaces on which ball stability evaluated
alpha = np.linspace(0, np.pi, 8, endpoint=False) # field lines on which to ball stability evaluated
nturns = 3 # num toroidal transits of the field line
N0 = nturns * 200 # number of point along a field line in ballooning space
zeta = np.linspace(-np.pi * nturns, np.pi * nturns, N0) # range of the ballooning coordinate zeta
# We need to make a special grid in field aligned coordinates, which we do here
# coordinates="raz" tells desc that this grid is in rho,alpha,zeta coordinates:
grid = Grid.create_meshgrid([surfaces, alpha, zeta], coordinates="raz")
data = eq0.compute(
    ["ideal ballooning lambda", "ideal ballooning eigenfunction"], grid=grid
)
print("Growth rate and eigenfunction calculation finished!")
eigenvals = data["ideal ballooning lambda"]
eigenfuns = data["ideal ballooning eigenfunction"]


lambda_max0 = np.zeros(surfaces.size)
eigenfunc_max0 = np.zeros((surfaces.size, N0))
for j in range(surfaces.size):
    idxmax = np.argmax(eigenvals[j])
    alpha_idx, zeta0_idx, eigval_idx = np.unravel_index(idxmax, eigenvals[j].shape)
    # max eigenvalues
    lambda_max0[j] = eigenvals[j, alpha_idx, zeta0_idx, eigval_idx]
    # eigenfunction corresponding to the max eigenvalue
    X0 = eigenfuns[j, alpha_idx, zeta0_idx, :, eigval_idx]
    sign_max = np.sign(X0[np.argmax(np.abs(X0))])
    eigenfunc_max0[j, 1:-1] = X0 / np.max(np.abs(X0)) * sign_max


# Plotting:
plt.plot(surfaces, lambda_max0, "-or", ms=4)
plt.xlabel(r"$\rho$", fontsize=18)
plt.ylabel(r"$\lambda_{\mathrm{max}}$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#
plt.figure()
plt.plot(zeta, eigenfunc_max0[3])  # plotting eigenfunction on rho=0.4
plt.xlabel(r"$\zeta$", fontsize=18)
plt.ylabel(r"$X_{\mathrm{max}}$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


# Newcomb's metric:
data = eq0.compute(["Newcomb ballooning metric"], grid=grid, data=data)
plt.plot(surfaces, data["Newcomb ballooning metric"], "-or", ms=4)
plt.xlabel(r"$\rho$", fontsize=18)
plt.ylabel("Newcomb metric", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


#-------------------------- EQUILIBRIUM SOLVE & PLOTTING --------------------------------------------------------------