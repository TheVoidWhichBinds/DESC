import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from .eq import run_equilibrium
from .opt import run_optimization
import os
import pandas as pd
from tabulate import tabulate
from desc.plotting import plot_comparison
from .helper import (
    sci_compact,
    _safe_extract_from_result,
    _next_run_dir,
    _write_readme,
)










#============== PROXIMAL VS. AUGLAG COMPARISON ============================================================================================================
def comparison(
        eq_config: dict,
        opt_config: dict,
        driver_config: dict,
    ):
    """
    Runs initial equilibrium solve, then optimization for
    both proximal-lsq-exact and lsq-auglag.
    Plots and objective table are also generated.
    """
    #====================================================
    # Initializations:
    #----------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    continuation_dir = os.path.join(base_dir, "continuation")
    #----------------------------------------------------

    #---------------------------------------------
    # Unpacking config variables needed in driver:
    NFP = eq_config["NFP"]
    opt_toggles = opt_config["opt_toggles"]
    config_path = driver_config["config_path"]
    #---------------------------------------------

    #----------------------------------------
    # Initializing table of objective values:
    column_map = [
        ("forcebalance_obj", f"Force error: "),
        ("qs", f"Quasi-symmetry (1,{NFP}) Boozer error: "),
        ("aspect_ratio", "Aspect ratio: "),
        ("ballooning", "Ideal ballooning lambda: "),
        ("mercier", "Mercier Stability: "),
    ]

    active_columns = []
    for key, label in column_map:
        if opt_toggles.get(key, {}).get("use", False):
            active_columns.append((key, label))
    #----------------------------------------

    #------------------------------------------
    # Generates meta-data README and directory:
    out_dir = _next_run_dir(continuation_dir)
    _write_readme(out_dir, config_path)
    #------------------------------------------
    #==========================================


    #============================================
    # Running & saving initial equilibrium solve:
    eq_init = run_equilibrium(eq_config=eq_config)
    eq_init.save(os.path.join(out_dir, "eq_init.h5"))
       
    # Plotting toroidal cross-sections (confirmation of eq health):
    plt.title("Toroidal Cross-Sections of Initial Equilibrium")
    fig, ax = plot_comparison(
        eqs=[eq_init],
        labels=["Initial Equilibrium"],
        color=["green"],
    )
    toroidal_cuts_path = os.path.join(out_dir, "initial_toroidal_cuts.png")
    plt.savefig(toroidal_cuts_path, dpi=200)
    plt.close()
    #==========
    #============================================


    #======================================
    # Running both optimizers:
    eq_opt_prox = eq_init.copy()
    eq_opt_auglag = eq_init.copy()

    optimizer_prox = "proximal-lsq-exact"
    optimizer_auglag = "lsq-auglag"

    eq_opt_prox, opt_result_prox = run_optimization(
        eq_opt_prox,
        optimizer_prox,
        opt_config=opt_config,
    )

    eq_opt_auglag, opt_result_auglag = run_optimization(
        eq_opt_auglag,
        optimizer_auglag,
        opt_config=opt_config,
    )

    #--------------------------
    # Saving optimized outputs:
    eq_opt_prox.save(os.path.join(out_dir, "opt_prox.h5"))
    eq_opt_auglag.save(os.path.join(out_dir, "opt_auglag.h5"))
    #---------------------------------------------------------
    #=========================================================


    #====================================
    # Generating comparison table labels:
    rows = []
    values = []

    rows.append(("Proximal",))
    rows.append(("AugLag",))
    rows.append(("Diff (AUG - PROX)",))

    row_prox = []
    row_auglag = []
    row_DIFF = []
    #============

    #=============================
    # Extracting objective values:
    for key, label in active_columns:
        fmin_prox, fmean_prox, fmax_prox = _safe_extract_from_result(
            opt_result_prox, label
        )
        fmin_auglag, fmean_auglag, fmax_auglag = _safe_extract_from_result(
            opt_result_auglag, label
        )

        row_prox.append(
            f"f_min={sci_compact(fmin_prox, sig=4)}, "
            f"f_mean={sci_compact(fmean_prox, sig=4)}, "
            f"f_max={sci_compact(fmax_prox, sig=4)}"
        )
        row_auglag.append(
            f"f_min={sci_compact(fmin_auglag, sig=4)}, "
            f"f_mean={sci_compact(fmean_auglag, sig=4)}, "
            f"f_max={sci_compact(fmax_auglag, sig=4)}"
        )

        dmin = fmin_auglag - fmin_prox
        dmean = fmean_auglag - fmean_prox
        dmax = fmax_auglag - fmax_prox
        row_DIFF.append(
            f"f_min diff={sci_compact(dmin, sig=4)}, "
            f"f_mean diff={sci_compact(dmean, sig=4)}, "
            f"f_max diff={sci_compact(dmax, sig=4)}"
        )
    #=====================================


    #=========================
    # Including Beta in table:
    beta_prox = float(
        eq_opt_prox.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_auglag = float(
        eq_opt_auglag.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_DIFF = beta_auglag - beta_prox

    row_prox.append(f"{beta_prox:.4g}")
    row_auglag.append(f"{beta_auglag:.4g}")
    row_DIFF.append(f"{beta_DIFF:.4g}")

    values.append(row_prox)
    values.append(row_auglag)
    values.append(row_DIFF)
    #======================


    #==============================
    # Save table inside run folder:
    index = pd.Index(
        [row[0] for row in rows],
        name="Run",
    )
    df = pd.DataFrame(
        values,
        index=index,
        columns=[label.replace(": ", "") for _, label in active_columns] + ["Beta"],
    )

    ascii_table = tabulate(df, headers="keys", tablefmt="grid")

    output_file = os.path.join(out_dir, "comparison.txt")
    with open(output_file, "w") as f:
        f.write("Comparison of Post-Optimization Objectives\n\n")
        f.write(ascii_table)
    #=======================


    #====================
    # Plotting pressures:
    rho = np.linspace(0.0, 1.0, 400)
    grid = LinearGrid(
        rho=rho,
        M=0,
        N=0,
        NFP=eq_opt_prox.NFP,
        sym=eq_opt_prox.sym,
    )

    p_prox = eq_opt_prox.compute("p", grid=grid)["p"]
    p_auglag = eq_opt_auglag.compute("p", grid=grid)["p"]

    plt.figure(figsize=(7, 5))
    plt.plot(rho, p_prox, linewidth=2, label="Proximal", color="purple")
    plt.plot(rho, p_auglag, linewidth=2, label="AugLag", color="orange")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel("Pressure", fontsize=14)
    plt.title(r"Pressure vs $\rho$", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    pressure_path = os.path.join(out_dir, "pressure_compare.png")
    plt.savefig(pressure_path, dpi=200)
    plt.close()
    #==========


    #==================================
    # Plotting toroidal cross-sections:
    plt.title("Toroidal Cross-Sections of Solved Equilibria")
    fig, ax = plot_comparison(
        eqs=[eq_init, 
             eq_opt_prox, 
             eq_opt_auglag
        ],
        labels=[
            "Initial Equilibrium",
            "Optimized (_prox)",
            "Optimized (_auglag)",
        ],
        color=[
            "green", 
            "purple", 
            "orange"
        ],
    )

    toroidal_cuts_path = os.path.join(out_dir, "toroidal_cuts.png")
    plt.savefig(toroidal_cuts_path, dpi=200)
    plt.close()
    #==========


    #=============================
    # J_parallel vs. rho plotting:
    rho_grid = np.linspace(0.0, 1.0, 100)
    grid_J = LinearGrid(
        rho=rho_grid,
        M=24,
        N=24,
        NFP=eq_opt_prox.NFP,
        sym=eq_opt_prox.sym,
    )

    def _j_parallel_profile(eq):
        data = eq.compute(["J_parallel"], grid=grid_J)
        J_parallel = np.asarray(data["J_parallel"])

        rho_nodes = grid_J.nodes[:, 0]
        rho_unique = np.unique(rho_nodes)

        J_parallel_fs = np.empty_like(rho_unique, dtype=float)
        for i, r in enumerate(rho_unique):
            mask = np.isclose(rho_nodes, r)
            J_parallel_fs[i] = np.mean(J_parallel[mask])

        return rho_unique, J_parallel_fs

    rho_u_prox, J_parallel_prox = _j_parallel_profile(eq_opt_prox)
    rho_u_auglag, J_parallel_auglag = _j_parallel_profile(eq_opt_auglag)

    plt.figure(figsize=(7, 5))
    plt.plot(rho_u_prox, J_parallel_prox, linewidth=2, label="Proximal", color="purple")
    plt.plot(rho_u_auglag, J_parallel_auglag, linewidth=2, label="AugLag", color="orange")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel(r"$\langle J_{\parallel} \rangle$", fontsize=14)
    plt.title("Parallel Current", fontsize=13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    j_parallel_path = os.path.join(out_dir, "j_parallel.png")
    plt.savefig(j_parallel_path, dpi=200)
    plt.close()
    #==========


    #=======================
    # iota vs. rho plotting:
    grid_iota = LinearGrid(
        rho=rho_grid,
        M=0,
        N=0,
        NFP=eq_opt_prox.NFP,
        sym=eq_opt_prox.sym,
    )

    iota_prox = eq_opt_prox.compute("iota", grid=grid_iota)["iota"]
    iota_auglag = eq_opt_auglag.compute("iota", grid=grid_iota)["iota"]

    plt.figure(figsize=(7, 5))
    plt.plot(rho_grid, iota_prox, linewidth=2, label="Proximal", color="purple")
    plt.plot(rho_grid, iota_auglag, linewidth=2, label="AugLag", color="orange")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel(r"$\iota$", fontsize=14)
    plt.title(r"Rotational transform vs $\rho$", fontsize=13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    iota_path = os.path.join(out_dir, "iota.png")
    plt.savefig(iota_path, dpi=200)
    plt.close()
    #==========
#==============================================================================================================================================================










#============== CONFIGURATION ENTRYPOINT ==========================================================================================================================
def run_from_config(
        eq_config: dict,
        opt_config: dict,
        driver_config: dict
    ):
    comparison(
        eq_config=eq_config,
        opt_config=opt_config,
        driver_config=driver_config,
    )
#==============================================================================================================================================================