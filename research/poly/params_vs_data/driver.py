import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from desc.grid import LinearGrid
from desc.plotting import plot_comparison
from tabulate import tabulate

from .eq import run_equilibrium
from .helper import (
    _next_run_dir,
    _safe_extract_from_result,
    _write_readme,
    sci_compact,
)
from .opt import run_optimization






#============== PROX_PARAMS VS. PROX_DATA VS. AUGLAG_PARAMS VS. AUGLAG_DATA COMPARISON ===========================================================
def comparison(
        eq_config: dict,
        opt_config: dict,
        driver_config: dict,
    ):
    """
    Runs initial equilibrium solve, then optimization for:
    proximal-lsq-exact with params-based custom objectives,
    proximal-lsq-exact with data-based custom objectives, and
    lsq-auglag. Plots and objective table are also generated.
    """
    #====================================================
    # Initializations:
    #----------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #----------------------------------------------------

    #---------------------------------------------
    # Unpacking config variables needed in driver:
    NFP = eq_config["NFP"]
    opt_toggles_all = opt_config["opt_toggles_all"]
    config_path = driver_config["config_path"]
    #-----------------------------------------

    #------------------------------------------
    # Generates meta-data README and directory:
    out_dir = _next_run_dir(base_dir)
    _write_readme(out_dir, config_path)
    #----------------------------------

    #---------------------------
    # Table of objective values:
    column_map = [
        ("forcebalance_obj", f"Force error: "),
        ("qs", f"Quasi-symmetry (1,{NFP}) Boozer error: "),
        ("aspect_ratio", "Aspect ratio: "),
        ("ballooning", "Ideal ballooning lambda: "),
        ("mercier", "Mercier Stability: "),
    ]

    active_columns = []
    for key, label in column_map:
        if opt_toggles_all.get(key, {}).get("use", False):
            active_columns.append((key, label))
    #------------------------------------------
    #==========================================




    #============================================
    # Equilibrium run:
    #--------------------------------------------
    # Running & saving initial equilibrium solve:
    eq_init = run_equilibrium(eq_config=eq_config)
    eq_init.save(os.path.join(out_dir, "eq_init.h5"))
    #------------------------------------------------

    #--------------------------------------------------------------
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
    #----------
    #==========




    #=========================
    # Optimization run:
    #-------------------------
    # Running all optimizers:
    eq_proxparams_0 = eq_init.copy()
    eq_proxdata_0 = eq_init.copy()
    eq_auglagparams_0 = eq_init.copy()
    eq_auglagdata_0 = eq_init.copy()

    optimizer_prox = "proximal-lsq-exact"
    optimizer_auglag = "lsq-auglag"

    opt_proxparams, opt_result_proxparams = run_optimization(
        eq_0 = eq_proxparams_0,
        optimizer = optimizer_prox,
        toggle_group = "opt_toggles_params",
        opt_config = opt_config,
    )

    opt_proxdata, opt_result_proxdata = run_optimization(
        eq_0 = eq_proxdata_0,
        optimizer = optimizer_prox,
        toggle_group = "opt_toggles_data",
        opt_config = opt_config,
    )

    opt_auglagparams, opt_result_auglagparams = run_optimization(
        eq_0 = eq_auglagparams_0,
        optimizer = optimizer_auglag,
        toggle_group = "opt_toggles_params",
        opt_config = opt_config,
    )

    opt_auglagdata, opt_result_auglagdata = run_optimization(
        eq_0 = eq_auglagdata_0,
        optimizer = optimizer_auglag,
        toggle_group = "opt_toggles_data",
        opt_config = opt_config,
    )
    #---------------------------

    #--------------------------
    # Saving optimized outputs:
    opt_proxparams.save(os.path.join(out_dir, "opt_proxparams.h5"))
    opt_proxdata.save(os.path.join(out_dir, "opt_proxdata.h5"))
    opt_auglagparams.save(os.path.join(out_dir, "opt_auglagparams.h5"))
    opt_auglagdata.save(os.path.join(out_dir, "opt_auglagdata.h5"))
    #----------------------------------------------------------------
    #================================================================




    #====================================
    # Generating comparison table labels:
    #------------------------------------
    rows = [
        ("Prox Params",),
        ("Prox Data",),
        ("AugLag Params",),
        ("AugLag Data",),
        ("Diff (DATA - PARAMS) Prox",),
        ("Diff (DATA - PARAMS) AugLag",),
    ]

    values = []

    row_proxparams = []
    row_proxdata = []
    row_auglagparams = []
    row_auglagdata = []
    row_diff_prox = []
    row_diff_auglag = []
    #------------------------------------

    #-----------------------------
    # Extracting objective values:
    for key, label in active_columns:
        fmin_proxparams, fmean_proxparams, fmax_proxparams = _safe_extract_from_result(
            opt_result_proxparams, label
        )
        fmin_proxdata, fmean_proxdata, fmax_proxdata = _safe_extract_from_result(
            opt_result_proxdata, label
        )
        fmin_auglagparams, fmean_auglagparams, fmax_auglagparams = _safe_extract_from_result(
            opt_result_auglagparams, label
        )
        fmin_auglagdata, fmean_auglagdata, fmax_auglagdata = _safe_extract_from_result(
            opt_result_auglagdata, label
        )

        row_proxparams.append(
            f"f_min={sci_compact(fmin_proxparams, sig=4)}, "
            f"f_mean={sci_compact(fmean_proxparams, sig=4)}, "
            f"f_max={sci_compact(fmax_proxparams, sig=4)}"
        )
        row_proxdata.append(
            f"f_min={sci_compact(fmin_proxdata, sig=4)}, "
            f"f_mean={sci_compact(fmean_proxdata, sig=4)}, "
            f"f_max={sci_compact(fmax_proxdata, sig=4)}"
        )
        row_auglagparams.append(
            f"f_min={sci_compact(fmin_auglagparams, sig=4)}, "
            f"f_mean={sci_compact(fmean_auglagparams, sig=4)}, "
            f"f_max={sci_compact(fmax_auglagparams, sig=4)}"
        )
        row_auglagdata.append(
            f"f_min={sci_compact(fmin_auglagdata, sig=4)}, "
            f"f_mean={sci_compact(fmean_auglagdata, sig=4)}, "
            f"f_max={sci_compact(fmax_auglagdata, sig=4)}"
        )


        dmin_prox = fmin_proxdata - fmin_proxparams
        dmean_prox = fmean_proxdata - fmean_proxparams
        dmax_prox = fmax_proxdata - fmax_proxparams
        row_diff_prox.append(
            f"f_min diff={sci_compact(dmin_prox, sig=4)}, "
            f"f_mean diff={sci_compact(dmean_prox, sig=4)}, "
            f"f_max diff={sci_compact(dmax_prox, sig=4)}"
        )

        dmin_auglag = fmin_auglagdata - fmin_auglagparams
        dmean_auglag = fmean_auglagdata - fmean_auglagparams
        dmax_auglag = fmax_auglagdata - fmax_auglagparams
        row_diff_auglag.append(
            f"f_min diff={sci_compact(dmin_auglag, sig=4)}, "
            f"f_mean diff={sci_compact(dmean_auglag, sig=4)}, "
            f"f_max diff={sci_compact(dmax_auglag, sig=4)}"
        )
    #-----------------------------------------------------

    #-------------------------
    # Including Beta in table:
    beta_proxparams = float(
        opt_proxparams.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_proxdata = float(
        opt_proxdata.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_auglagparams = float(
        opt_auglagparams.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_auglagdata = float(
        opt_auglagdata.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )

    beta_diff_prox = beta_proxdata - beta_proxparams
    beta_diff_auglag = beta_auglagdata - beta_auglagparams

    row_proxparams.append(f"{beta_proxparams:.4g}")
    row_proxdata.append(f"{beta_proxdata:.4g}")
    row_auglagparams.append(f"{beta_auglagparams:.4g}")
    row_auglagdata.append(f"{beta_auglagdata:.4g}")
    row_diff_prox.append(f"{beta_diff_prox:.4g}")
    row_diff_auglag.append(f"{beta_diff_auglag:.4g}")

    values.append(row_proxparams)
    values.append(row_proxdata)
    values.append(row_auglagparams)
    values.append(row_auglagdata)    
    values.append(row_diff_prox)
    values.append(row_diff_auglag)
    #-----------------------------

    #------------------------------
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
    #-----------------------
    #=======================




    #====================
    # Plotting:
    #--------------------
    # Plotting pressures:
    rho = np.linspace(0.0, 1.0, 400)
    grid = LinearGrid(
        rho=rho,
        M=0,
        N=0,
        NFP = opt_proxparams.NFP,
        sym = opt_proxparams.sym,
    )

    p_proxparams = opt_proxparams.compute("p", grid=grid)["p"]
    p_proxdata = opt_proxdata.compute("p", grid=grid)["p"]
    p_auglagparams = opt_auglagparams.compute("p", grid=grid)["p"]
    p_auglagdata = opt_auglagdata.compute("p", grid=grid)["p"]

    plt.figure(figsize=(7, 5))
    plt.plot(rho, p_proxparams, linewidth=2, label="Prox Params", color="purple")
    plt.plot(rho, p_proxdata, linewidth=2, label="Prox Data", color="blue")
    plt.plot(rho, p_auglagparams, linewidth=2, label="AugLag Params", color="orange")
    plt.plot(rho, p_auglagdata, linewidth=2, label="AugLag Data", color="red")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel("Pressure", fontsize=14)
    plt.title(r"Pressure vs $\rho$", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    pressure_path = os.path.join(out_dir, "pressure_compare.png")
    plt.savefig(pressure_path, dpi=200)
    plt.close()
    #----------

    #----------------------------------
    # Plotting toroidal cross-sections:
    plt.title("Toroidal Cross-Sections of Solved Equilibria")
    fig, ax = plot_comparison(
        eqs=[
            eq_init,
            opt_proxparams,
            opt_proxdata,
            opt_auglagparams,
            opt_auglagdata,
        ],
        labels=[
            "Initial Equilibrium",
            "Optimized (_prox_params)",
            "Optimized (_prox_data)",
            "Optimized (_auglag_params)",
            "Optimized (_auglag_data)",
        ],
        color=[
            "green",
            "purple",
            "blue",
            "orange",
            "red",
        ],
    )

    toroidal_cuts_path = os.path.join(out_dir, "toroidal_cuts.png")
    plt.savefig(toroidal_cuts_path, dpi=200)
    plt.close()
    #----------

    #-----------------------------
    # J_parallel vs. rho plotting:
    rho_grid = np.linspace(0.0, 1.0, 100)
    grid_J = LinearGrid(
        rho=rho_grid,
        M=24,
        N=24,
        NFP=opt_proxparams.NFP,
        sym=opt_proxparams.sym,
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

    rho_u_proxparams, J_parallel_proxparams = _j_parallel_profile(opt_proxparams)
    rho_u_proxdata, J_parallel_proxdata = _j_parallel_profile(opt_proxdata)
    rho_u_auglagparams, J_parallel_auglagparams = _j_parallel_profile(opt_auglagparams)
    rho_u_auglagdata, J_parallel_auglagdata = _j_parallel_profile(opt_auglagdata)

    plt.figure(figsize=(7, 5))
    plt.plot(
        rho_u_proxparams,
        J_parallel_proxparams,
        linewidth=2,
        label="Prox Params",
        color="purple",
    )
    plt.plot(
        rho_u_proxdata,
        J_parallel_proxdata,
        linewidth=2,
        label="Prox Data",
        color="blue",
    )
    plt.plot(
        rho_u_auglagparams,
        J_parallel_auglagparams,
        linewidth=2,
        label="AugLag Params",
        color="orange",
    )
    plt.plot(
        rho_u_auglagdata,
        J_parallel_auglagdata,
        linewidth=2,
        label="AugLag Data",
        color="red",
    )
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel(r"$\langle J_{\parallel} \rangle$", fontsize=14)
    plt.title("Parallel Current", fontsize=13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    j_parallel_path = os.path.join(out_dir, "j_parallel.png")
    plt.savefig(j_parallel_path, dpi=200)
    plt.close()
    #----------

    #-----------------------
    # iota vs. rho plotting:
    grid_iota = LinearGrid(
        rho=rho_grid,
        M=0,
        N=0,
        NFP=opt_proxparams.NFP,
        sym=opt_proxparams.sym,
    )

    iota_proxparams = opt_proxparams.compute("iota", grid=grid_iota)["iota"]
    iota_proxdata = opt_proxdata.compute("iota", grid=grid_iota)["iota"]
    iota_auglagparams = opt_auglagparams.compute("iota", grid=grid_iota)["iota"]
    iota_auglagdata = opt_auglagdata.compute("iota", grid=grid_iota)["iota"]

    plt.figure(figsize=(7, 5))
    plt.plot(rho_grid, iota_proxparams, linewidth=2, label="Prox Params", color="purple")
    plt.plot(rho_grid, iota_proxdata, linewidth=2, label="Prox Data", color="blue")
    plt.plot(rho_grid, iota_auglagparams, linewidth=2, label="AugLag Params", color="orange")
    plt.plot(rho_grid, iota_auglagdata, linewidth=2, label="AugLag Data", color="red")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel(r"$\iota$", fontsize=14)
    plt.title(r"Rotational transform vs $\rho$", fontsize=13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    iota_path = os.path.join(out_dir, "iota.png")
    plt.savefig(iota_path, dpi=200)
    plt.close()
    #----------
    #==========
#==================================================================================================================================================




#============== CONFIGURATION ENTRYPOINT ==========================================================================================================
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
#==================================================================================================================================================