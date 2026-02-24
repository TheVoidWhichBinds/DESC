
#--------------------------------------- IMPORTS ----------------------------------------------------------------------------------------------
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg") # non-GUI backend (prevents Tk crashes on macOS)
import matplotlib.pyplot as plt
sys.path.append("/Users/macdaddi/DESC")
import desc
import desc.io
from desc.grid import LinearGrid, Grid
from scratch.objectives.poly_constraints import (
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    poly_monotonicity
)
from desc.objectives import (
    ObjectiveFunction,
    FixIota,
    FixPsi,
    FixPressure,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    BallooningStability,
    MercierStability,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)
from desc.optimize import Optimizer
from desc.plotting import plot_comparison
#----------------------------------------------------------------------------------------------------------------------------------------------

# Optimizer selection:
optimizer = Optimizer("proximal-lsq-exact")










#--------------------------------------- BALLOONING METRIC (LAMBDA MAX) FUNC ---------------------------------------------------------------------
def ballooning_lambda_max(eq, surfaces, alpha, zeta):
    """
    Compute max ideal-ballooning growth rate lambda over (alpha, zeta0, eigen-index)
    for each rho in `surfaces`.
    """
    grid_b = Grid.create_meshgrid([surfaces, alpha, zeta], coordinates="raz")
    data_b = eq.compute(["ideal ballooning lambda"], grid=grid_b)
    lam = data_b["ideal ballooning lambda"]  # (rho, alpha, zeta0, eig)
    lam_rho = np.max(lam.reshape(lam.shape[0], -1), axis=1)
    lam_global = float(np.max(lam_rho))
    return lam_rho, lam_global
#------------------------------------------------------------------------------------------------------------------------------------------------









#--------------------------------------- OPTIMIZER FUNC -----------------------------------------------------------------------------------------
def run_optimization(p_scale, out_dir, fix_pressure: bool):
    """
    Build HELIOTRON example as eq_init, copy to eq_0, then optimize.
    Returns eq_opt, opt_result.
    Also saves the optimized equilibrium to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initial equilibrium: HELIOTRON example (NOT file-based)
    eq_init = desc.examples.get("HELIOTRON")
    eq_init.change_resolution(L=12, M=6, N=2, L_grid=18, M_grid=12, N_grid=4)
    eq_init.surface = eq_init.get_surface_at(rho=1)

    eq_0 = eq_init.copy()

    # Weight to assign to secondary objectives and constraints (not force balance):
    inferior_weights = 1e0


    #---------------------------------
    if fix_pressure:  # fixed pressure
        constraints = (
            ForceBalance(eq=eq_0),
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
            FixPressure(eq=eq_0),
        )

        objective = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e1),
            AspectRatio(eq=eq_0, target=6, weight=inferior_weights),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=inferior_weights),
            BallooningStability(eq=eq_0, target=0.0, weight=inferior_weights),
            MercierStability(eq=eq_0, target=0.0, weight=inferior_weights),
        ])

    else:  # optimized pressure
        pressure_axis_set = LinearObjectiveFromUser(
            fun=pressure_axis,
            thing=eq_0,
            target=p_scale,
            weight=inferior_weights,
        )
        pressure_edge_zero = LinearObjectiveFromUser(
            fun=pressure_edge,
            thing=eq_0,
            target=0.0,
            weight=inferior_weights,
        )
        grad_pressure_axis_zero = LinearObjectiveFromUser(
            fun=grad_pressure_axis,
            thing=eq_0,
            target=0.0,
            weight=inferior_weights,
        )
        grad_pressure_edge_zero = LinearObjectiveFromUser(
            fun=grad_pressure_edge,
            thing=eq_0,
            target=0.0,
            weight=inferior_weights,
        )

        constraints = (
            ForceBalance(eq=eq_0),
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
            pressure_axis_set,
            pressure_edge_zero,
            grad_pressure_axis_zero,
            grad_pressure_edge_zero,
        )

        negative_gradient = ObjectiveFromUser(
            fun=poly_monotonicity,
            grid=LinearGrid(rho=200, M=0, N=0),
            thing=eq_0,
            target=0.0,
            weight=inferior_weights,
            normalize=False,
        )

        objective = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e4),
            AspectRatio(eq=eq_0, target=6, weight=inferior_weights),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=inferior_weights),
            BallooningStability(eq=eq_0, target=0.0, weight=inferior_weights),
            MercierStability(eq=eq_0, target=0.0, weight=inferior_weights),
            negative_gradient,
        ])
    #-------------------------


    #----------------------
    # Solving optimization:
    eq_opt, opt_result = eq_0.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        ftol=5e-2,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=50,
        options={
            "perturb_options": {"order": 2, "verbose": 0},
            "solve_options": {"ftol": 5e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0},
        },
        copy=False,
        verbose=3,
    )
    #-------------


    save_name = "opt_FXP.h5" if fix_pressure else "opt.h5"
    save_path = out_dir / save_name
    eq_opt.save(save_path)

    return eq_init, eq_opt, opt_result
#-------------------------------------------------------------------------------------------------------------------------------------------------










#--------------------------------------- DRIVER --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #---------------------------------------------------------
    out_dir = "/Users/macdaddi/DESC/scratch/runs/poly/balloon"
    p_scale = 1.0

    # Build HELIOTRON once for reporting + metrics (same settings as in run_optimization):
    eq_init = desc.examples.get("HELIOTRON")
    eq_init.change_resolution(L=12, M=6, N=2, L_grid=18, M_grid=12, N_grid=4)
    eq_init.surface = eq_init.get_surface_at(rho=1)

    # Run optimizations starting from eq_init copy (inside run_optimization):
    _, eq_opt_FXD, _ = run_optimization(p_scale=p_scale, out_dir=out_dir, fix_pressure=True)
    _, eq_opt, _     = run_optimization(p_scale=p_scale, out_dir=out_dir, fix_pressure=False)

    # Ballooning lambda metric for all 3:
    surfaces = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    alpha = np.linspace(0, np.pi, 8, endpoint=False)
    nturns = 3
    N0 = nturns * 200
    zeta = np.linspace(-np.pi * nturns, np.pi * nturns, N0)
    #------------------------------------------------------


    #-----------------------------------------------------------
    # Solving for max lambda on different surfaces and globally:
    lam_init_rho, lam_init_global = ballooning_lambda_max(eq_init, surfaces, alpha, zeta)
    lam_fxd_rho,  lam_fxd_global  = ballooning_lambda_max(eq_opt_FXD, surfaces, alpha, zeta)
    lam_opt_rho,  lam_opt_global  = ballooning_lambda_max(eq_opt, surfaces, alpha, zeta)

    print("\n--- Ballooning lambda_max metrics ---")
    print("_(rh0) is the max labmda at different surfaces given by radial coordinate rho in surfaces variable")
    print("[eq_init]     lambda_max(rho) =", lam_init_rho)
    print("[eq_init]     lambda_max global =", f"{lam_init_global:.6e}")
    print("[eq_opt_FXD]  lambda_max(rho) =", lam_fxd_rho)
    print("[eq_opt_FXD]  lambda_max global =", f"{lam_fxd_global:.6e}")
    print("[eq_opt]      lambda_max(rho) =", lam_opt_rho)
    print("[eq_opt]      lambda_max global =", f"{lam_opt_global:.6e}")
    #------------------------------------------------------------------


    #----------------------------------------------------------------
    # Toroidal cuts of initial, fixed pressure opt, opt pressure opt:
    fig, ax = plot_comparison(
        eqs=[eq_init, eq_opt_FXD, eq_opt],
        labels=[
            "Initial Equilibrium",
            "Optimized (Fixed Pressure)",
            "Optimized (Optimized Pressure)",
        ],
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / "toroidal_cuts.png"
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot: {save_path}")
    #---------------------------------------------


    #--------------------------------------------------------------------------------------------------
    grid = Grid.create_meshgrid([surfaces, alpha, zeta], coordinates="raz")  # creating ballooning grid

    # Compute lambda_max(rho) for each equilibrium on the same grid:
    data_init = eq_init.compute(["ideal ballooning lambda"], grid=grid)
    lambda_max_init = data_init["ideal ballooning lambda"].max(axis=(-1, -2, -3))

    data_fxd = eq_opt_FXD.compute(["ideal ballooning lambda"], grid=grid)
    lambda_max_fxd = data_fxd["ideal ballooning lambda"].max(axis=(-1, -2, -3))

    data_opt = eq_opt.compute(["ideal ballooning lambda"], grid=grid)
    lambda_max_opt = data_opt["ideal ballooning lambda"].max(axis=(-1, -2, -3))

    # Plotting:
    plt.figure()
    plt.plot(surfaces, lambda_max_init, "-or", ms=4)
    plt.plot(surfaces, lambda_max_fxd, "-og", ms=4)
    plt.plot(surfaces, lambda_max_opt, "-ob", ms=4)

    plt.legend(
        ["eq_init", "eq_opt_FXD (fixed P)", "eq_opt (optimized P)"],
        fontsize=16,
    )
    plt.xlabel(r"$\rho$", fontsize=18)
    plt.ylabel(r"$\lambda_{\mathrm{max}}$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(out_dir / "lambda_max.png", dpi=250, bbox_inches="tight")
    plt.close()
    #----------
#---------------------------------------------------------------------------------------------------------------------------------------------------