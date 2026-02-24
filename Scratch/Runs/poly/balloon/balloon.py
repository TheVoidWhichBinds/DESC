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
from scratch.objectives.poly_constraints import (
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    poly_monotonicity
)


plt.rcParams["font.size"] = 14
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))



#-------------------------- EQUILIBRIUM SOLVE & PLOTTING --------------------------------------------------------------
# Importing the HELIOTRON DESC equilibrium:
eq0 = desc.examples.get("HELIOTRON")
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










#-------------------------- OPTIMIZATION -----------------------------------------------------------------------------

eq1 = eq0.copy() # save a copy of original for comparison
nzetaperturn = 200 # number of point along a field line per transit
k = 2 # determine which modes to unfix

print("\n---------------------------------------")
print(f"Optimizing boundary modes M, N <= {k}")
print("---------------------------------------")

modes_R = np.vstack(
    (
        [0, 0, 0],
        eq1.surface.R_basis.modes[np.max(np.abs(eq1.surface.R_basis.modes), 1) > k, :],
    )
)
modes_Z = eq1.surface.Z_basis.modes[np.max(np.abs(eq1.surface.Z_basis.modes), 1) > k, :]


# Compiling constraints:
constraints = (
    ForceBalance(eq=eq1),
    FixBoundaryR(eq=eq1, modes=modes_R),
    FixBoundaryZ(eq=eq1, modes=modes_Z),
    FixPressure(eq=eq1),
    FixIota(eq=eq1),
    FixPsi(eq=eq1),
)


# Grid:
Curvature_grid = LinearGrid(
    M=eq1.M_grid,
    N=eq1.N_grid,
    rho=np.array([1.0]),
    NFP=eq1.NFP,
    sym=eq1.sym,
    axis=False,
)


# Compiling objectives:
objective = ObjectiveFunction(
    (
        BallooningStability(
            eq=eq1,
            rho=np.array([0.8]),
            alpha=alpha,
            nturns=nturns,
            nzetaperturn=nzetaperturn,
            weight=2,
        ),
        AspectRatio(
            eq=eq1,
            bounds=(8, 11),
            weight=1,
        ),
        GenericObjective(
            f="curvature_k2_rho",
            thing=eq1,
            grid=Curvature_grid,
            bounds=(-np.inf, 0),
            weight=2,
        ),
    )
)


# Running optimizer:
optimizer = Optimizer("proximal-lsq-exact")
(eq1,), _ = optimizer.optimize(
    eq1,
    objective,
    constraints,
    ftol=1e-4,
    xtol=1e-6,
    gtol=1e-6,
    maxiter=5, 
    verbose=3,
    options={"initial_trust_ratio": 2e-3},
)
print("Optimization complete!")


# Plotting:
desc.plotting.plot_comparison([eq0, eq1])