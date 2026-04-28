import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from .eq import run_equilibrium
from .opt import run_optimization
import os
import re
import pandas as pd
from tabulate import tabulate
from desc.plotting import plot_comparison








#============== HELPER FUNCTIONS ==============================================================================================================================
#=========================
def sci_compact(x, sig=2):
    s = f"{x:.{sig-1}e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"{mant}e{exp}"
#=========================


#================
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan
#================


#============================
def _extract_f_stats(objval):
    """
    Preps final objective values to be put into comparison table.
    Returns numeric f_min, f_mean, f_max.
    """
    #---------------------------
    if isinstance(objval, list):
        chosen = None
        for item in objval:
            if isinstance(item, dict) and all(k in item for k in ("f_min", "f_mean", "f_max")):
                chosen = item
                break
        if chosen is None and len(objval) > 0:
            chosen = objval[0]
        objval = chosen
    #---------------------------

    #-----------------------------------------------------------
    if isinstance(objval, dict) and all(k in objval for k in ("f_min", "f_mean", "f_max")):
        return (
            _to_float(objval["f_min"]),
            _to_float(objval["f_mean"]),
            _to_float(objval["f_max"]),
        )
    #-----------------------------------------------------------

    #---------------------------
    if isinstance(objval, dict):
        for v in objval.values():
            val = _to_float(v)
            if not np.isnan(val):
                return val, val, val
        return np.nan, np.nan, np.nan
    #---------------------------

    val = _to_float(objval)
    return val, val, val
#============================


#=====================================
def _safe_extract_from_result(result, label):
    """
    Safely gets objective stats from result dict.
    Returns NaNs if label is absent.
    """
    objvals = result.get("Objective values", {})
    if label not in objvals:
        return np.nan, np.nan, np.nan
    return _extract_f_stats(objvals[label])
#=====================================


#===================================
def _next_run_dir(continuation_dir):
    """
    Create next zero-padded run directory: 001, 002, ...
    """
    os.makedirs(continuation_dir, exist_ok=True)

    run_nums = []
    for name in os.listdir(continuation_dir):
        path = os.path.join(continuation_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"\d{3}", name):
            run_nums.append(int(name))

    next_num = 1 if len(run_nums) == 0 else max(run_nums) + 1
    run_dir = os.path.abspath(os.path.join(continuation_dir, f"{next_num:03d}"))

    if os.path.exists(run_dir):
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    os.makedirs(run_dir)

    return run_dir
#=================


#======================================
def _write_readme(out_dir, config_path):
    """
    Write full config.py contents into README.md in a readable form.
    """
    with open(config_path, "r") as f:
        config_text = f.read()

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("```python\n")
        f.write(config_text)
        if not config_text.endswith("\n"):
            f.write("\n")
        f.write("```\n")
#======================================
#==============================================================================================================================================================









#============== FIXED VS. FREE PROFILES COMPARISON ============================================================================================================
def comparison(
        eq_config: dict,
        opt_config: dict,
        driver_config: dict,
    ):
    """
    Runs initial equilibrium solve, then optimization for
    both fixed and optimized pressure. Plots and objective
    table are also generated.
    """
    #====================================================
    # Initializations:
    #----------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    continuation_dir = os.path.join(base_dir, "continuation")
    #--------------------------------------------------------

    #---------------------------------------------
    # Unpacking config variables needed in driver:
    NFP = eq_config["NFP"]
    opt_toggles = opt_config["opt_toggles"]
    p_axis = driver_config["p_axis"]
    config_path = driver_config["config_path"]
    #-----------------------------------------

    #----------------------------------------
    # Initializing table of objective values:
    column_map = [
        ("forcebalance", f"Force error: "),
        ("qs", f"Quasi-symmetry (1,{NFP}) Boozer error: "),
        ("aspect_ratio", "Aspect ratio: "),
        ("ballooning", "Ideal ballooning lambda: "),
        ("mercier", "Mercier Stability: "),
    ]

    active_columns = []
    for key, label in column_map:
        if any(
            config["toggle_FXD"].get(key, {}).get("use", False)
            or config["toggle_CON"].get(key, {}).get("use", False)
            for config in opt_toggles
            if config["name"] is not None
        ):
            active_columns.append((key, label))
    #------------------------------------------

    #------------------------------------------
    # Generates meta-data README and directory:
    out_dir = _next_run_dir(continuation_dir)
    _write_readme(out_dir, config_path)
    #----------------------------------
    #==================================


    #============================================
    # Running & saving initial equilibrium solve:
    #-------------------------
    eq_init = run_equilibrium(
        eq_config = eq_config,
    )
    print("pressure type:", type(eq_init.pressure))
    print("pressure dict keys:", eq_init.pressure.__dict__.keys())
    #-----------------------

    #-------------------------------------------
    eq_init.save(os.path.join(out_dir, "eq.h5"))
    #-------------------------------------------
    #===========================================


    #======================================================================
    # Running & saving optimizer stages in order. 2nd stage overwrites 1st:
    #--------------------------
    eq_opt_FXD = eq_init.copy()
    eq_opt_CON = eq_init.copy()
    opt_result_FXD = None
    opt_result_CON = None
    #--------------------

    #-------------------------
    for config in opt_toggles:
        optimizer = config["name"]
        if optimizer is None:
            continue

        stage_opt_config = {
            **opt_config,
            "toggle_FXD": config["toggle_FXD"],
            "toggle_CON": config["toggle_CON"],
        }

        eq_opt_CON, opt_result_CON = run_optimization(
            eq_opt_CON,
            optimizer,
            p_axis,
            opt_config=stage_opt_config,
            FXD=False,
        )

        eq_opt_FXD, opt_result_FXD = run_optimization(
            eq_opt_FXD,
            optimizer,
            p_axis,
            opt_config=stage_opt_config,
            FXD=True,
        )
    #----------------
    
    #---------------------------------------------------
    eq_opt_CON.save(os.path.join(out_dir, "opt_CON.h5"))
    eq_opt_FXD.save(os.path.join(out_dir, "opt_FXD.h5"))
    #---------------------------------------------------
    #===================================================


    #====================================
    # Generating comparison table labels:
    rows = []
    values = []

    rows.append(("Fixed Profiles",))
    rows.append(("Free Profiles",))
    rows.append(("Difference",))

    row_FXD = []
    row_CON = []
    row_DIFF = []
    #====================================


    #=============================
    # Extracting objective values:
    for key, label in active_columns:
        fmin_FXD, fmean_FXD, fmax_FXD = _safe_extract_from_result(
            opt_result_FXD, label
        )
        fmin_CON, fmean_CON, fmax_CON = _safe_extract_from_result(
            opt_result_CON, label
        )

        row_FXD.append(
            f"f_min={sci_compact(fmin_FXD, sig=4)}, "
            f"f_mean={sci_compact(fmean_FXD, sig=4)}, "
            f"f_max={sci_compact(fmax_FXD, sig=4)}"
        )
        row_CON.append(
            f"f_min={sci_compact(fmin_CON, sig=4)}, "
            f"f_mean={sci_compact(fmean_CON, sig=4)}, "
            f"f_max={sci_compact(fmax_CON, sig=4)}"
        )

        dmin = fmin_CON - fmin_FXD
        dmean = fmean_CON - fmean_FXD
        dmax = fmax_CON - fmax_FXD
        row_DIFF.append(
            f"f_min diff={sci_compact(dmin, sig=4)}, "
            f"f_mean diff={sci_compact(dmean, sig=4)}, "
            f"f_max diff={sci_compact(dmax, sig=4)}"
        )
    #===============================================


    #=========================
    # Including Beta in table:
    beta_FXD = float(
        eq_opt_FXD.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_CON = float(
        eq_opt_CON.compute("<beta>_vol", override_grid=True)["<beta>_vol"]
    )
    beta_DIFF = beta_CON - beta_FXD

    row_FXD.append(f"{beta_FXD:.4g}")
    row_CON.append(f"{beta_CON:.4g}")
    row_DIFF.append(f"{beta_DIFF:.4g}")

    values.append(row_FXD)
    values.append(row_CON)
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
    #--------------------
    # Plotting pressures:
    rho = np.linspace(0.0, 1.0, 400)
    grid = LinearGrid(rho=rho, M=0, N=0, NFP=eq_opt_CON.NFP, sym=eq_opt_CON.sym)

    p_FXD = eq_opt_FXD.compute("p", grid=grid)["p"]
    p_CON = eq_opt_CON.compute("p", grid=grid)["p"]

    plt.figure(figsize=(7, 5))
    plt.plot(rho, p_FXD, linewidth=2, label="Fixed Pressure", color="blue")
    plt.plot(rho, p_CON, linewidth=2, label="Optimized Pressure", color="red")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel("Pressure", fontsize=14)
    plt.title(
        f"Pressure vs $\\rho$ (p_max={p_axis:.2g})",
        fontsize=14,
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    pressure_path = os.path.join(out_dir, "pressure_compare.png")
    plt.savefig(pressure_path, dpi=200)
    plt.close()
    #----------

    #------------------------------------------------------
    # Plotting gridded toroidal cross-sections of B-fields:
    plt.title("Toroidal Cross-Sections of Solved Equilibria")
    fig, ax = plot_comparison(
        eqs=[eq_init, eq_opt_FXD, eq_opt_CON],
        labels=[
            "Initial Equilibrium",
            "Optimized (Fixed Pressure)",
            "Optimized (Optimized Pressure)",
        ],
        color=["green", "blue", "red"],
    )

    toroidal_cuts_path = os.path.join(out_dir, "toroidal_cuts.png")
    plt.savefig(toroidal_cuts_path, dpi=200)
    plt.close()
    #----------

    #----------------------
    # |J| vs. rho plotting:
    rho_grid = np.linspace(0.0, 1.0, 100)
    grid_J = LinearGrid(rho=rho_grid, M=24, N=24, NFP=eq_opt_CON.NFP, sym=eq_opt_CON.sym)

    def _surface_mean_J_mag(eq):
        data = eq.compute(["|J|"], grid=grid_J)
        J_mag = np.asarray(data["|J|"])

        rho_nodes = grid_J.nodes[:, 0]
        rho_unique = np.unique(rho_nodes)

        J_mag_fs = np.empty_like(rho_unique, dtype=float)
        for i, r in enumerate(rho_unique):
            mask = np.isclose(rho_nodes, r)
            J_mag_fs[i] = np.mean(J_mag[mask])

        return rho_unique, J_mag_fs

    rho_u_FXD, J_mag_FXD = _surface_mean_J_mag(eq_opt_FXD)
    rho_u_CON, J_mag_CON = _surface_mean_J_mag(eq_opt_CON)

    plt.figure(figsize=(7, 5))
    plt.plot(rho_u_FXD, J_mag_FXD, linewidth=2, label="Fixed Pressure", color="blue")
    plt.plot(rho_u_CON, J_mag_CON, linewidth=2, label="Optimized Pressure", color="red")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel(r"$\langle |J| \rangle$", fontsize=14)
    plt.title(
        f"Flux-surface avg current magnitude (p_max={p_axis:.2g})",
        fontsize=13,
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    J_mag_path = os.path.join(out_dir, "J_mag.png")
    plt.savefig(J_mag_path, dpi=200)
    plt.close()
    #----------

    #-----------------------
    # iota vs. rho plotting:
    grid_iota = LinearGrid(rho=rho_grid, M=0, N=0, NFP=eq_opt_CON.NFP, sym=eq_opt_CON.sym)

    iota_FXD = eq_opt_FXD.compute("iota", grid=grid_iota)["iota"]
    iota_CON = eq_opt_CON.compute("iota", grid=grid_iota)["iota"]

    plt.figure(figsize=(7, 5))
    plt.plot(rho_grid, iota_FXD, linewidth=2, label="Fixed Pressure", color="blue")
    plt.plot(rho_grid, iota_CON, linewidth=2, label="Optimized Pressure", color="red")
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel(r"$\iota$", fontsize=14)
    plt.title(
        f"Rotational transform vs $\\rho$ (p_max={p_axis:.2g})",
        fontsize=13,
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    iota_path = os.path.join(out_dir, "iota.png")
    plt.savefig(iota_path, dpi=200)
    plt.close()
    #----------
    #==========
#==============================================================================================================================================================










#============== CONFIGURATION ENTRYPOINT ==========================================================================================================================
def run_from_config(
        eq_config: dict, 
        opt_config: dict, 
        driver_config: dict
    ):
    comparison(
        eq_config = eq_config,
        opt_config = opt_config,
        driver_config = driver_config,
    )
#==============================================================================================================================================================