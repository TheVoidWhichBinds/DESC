
import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from eq import run_equilibrium
from opt import run_optimization
import os
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
#==============================================================================================================================================================









#============== FIXED VS. OPTIMIZED PRESSURE COMPARISON =======================================================================================================
def comparison(
        p_maxima: list,
        n_set: list,
        eq_config: dict,
        opt_config: dict,
    ):
    """
    Runs initial equilibrium solve, then optimization for
    both fixed and optimized pressure. Plots and objective
    table are also generated.
    """

    #---------------------------------------------
    # Unpacking config variables needed in driver:
    NFP = eq_config["NFP"]

    optimizers = opt_config["optimizers"]
    optimizer1 = optimizers[0]
    optimizer2 = optimizers[1] if len(optimizers) > 1 else None

    toggle_FXD = opt_config["toggle_FXD"]
    toggle_CON = opt_config["toggle_CON"]
    #---------------------------------------------

    #========================================
    # Initializing table of objective values:
    column_map = [
        ("forcebalance_objective", f"Force error: "),
        ("qs", f"Quasi-symmetry (1,{NFP}) Boozer error: "),
        ("aspect_ratio", "Aspect ratio: "),
        ("fix_iota", "Fixed iota profile error: "),
        ("fix_psi", "Fixed Psi error: "),
        ("ballooning", "Ideal ballooning lambda: "),
        ("mercier", "Mercier Stability: "),
    ]

    active_columns = []
    for key, label in column_map:
        if toggle_FXD.get(key, False) or toggle_CON.get(key, False):
            active_columns.append((key, label))
    #========================================

    base_dir = os.path.dirname(os.path.abspath(__file__))




    #==================================================
    # Loop over on-axis pressure and polynomial orders:
    for p_axis in p_maxima:
        for n in n_set:
            #==========================================
            # Generates meta-data README and directory:
            out_dir = os.path.join(base_dir, f"p{sci_compact(p_axis)}_n{n}")
            os.makedirs(out_dir, exist_ok=True)
            #==================================


            #===================================
            # Running initial equilibrium solve:
            eq_init, n_eff = run_equilibrium(
                p_axis,
                n,
                out_dir=out_dir,
                eq_config=eq_config,
            )
            #===================================


            #=========================================================
            # 1st round of fixed and optimized pressure optimizations:
            eq_opt_FXD, opt_result_FXD = run_optimization(
                eq_init.copy(),
                optimizer1,
                p_axis,
                out_dir=out_dir,
                opt_config=opt_config,
                FXD=True,
            )
            eq_opt_CON, opt_result_CON = run_optimization(
                eq_init.copy(),
                optimizer1,
                p_axis,
                out_dir=out_dir,
                opt_config=opt_config,
                FXD=False,
            )
            #-------------

            #---------------------------------------------------------
            # 2nd round of fixed and optimized pressure optimizations:
            if optimizer2 is not None:
                eq_opt_FXD, opt_result_FXD = run_optimization(
                    eq_opt_FXD,
                    optimizer2,
                    p_axis,
                    out_dir=out_dir,
                    opt_config=opt_config,
                    FXD=True,
                )
                eq_opt_CON, opt_result_CON = run_optimization(
                    eq_opt_CON,
                    optimizer2,
                    p_axis,
                    out_dir=out_dir,
                    opt_config=opt_config,
                    FXD=False,
                )
            #=================


            #====================================
            # Generating comparison table labels:
            rows = []
            values = []

            label_n = n if n_eff == n else f"{n}→{n_eff}"
            group_label = f"Max Pressure = {p_axis}; n = {label_n} (order={2*n_eff})"

            rows.append((group_label, "Fixed Pressure"))
            rows.append((group_label, "Optimized Pressure"))
            rows.append((group_label, "Difference"))

            row_FXD = []
            row_CON = []
            row_DIFF = []
            #====================================


            #=============================
            #-----------------------------
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
            #-----------------------------------------------

            #-------------------------
            # Including Beta in table:
            beta_FXD = float(eq_opt_FXD.compute("<beta>_vol", override_grid=True)["<beta>_vol"])
            beta_CON = float(eq_opt_CON.compute("<beta>_vol", override_grid=True)["<beta>_vol"])
            beta_DIFF = beta_CON - beta_FXD

            row_FXD.append(f"{beta_FXD:.4g}")
            row_CON.append(f"{beta_CON:.4g}")
            row_DIFF.append(f"{beta_DIFF:.4g}")

            values.append(row_FXD)
            values.append(row_CON)
            values.append(row_DIFF)
            #-------------------------

            #------------------------------
            # Save table inside run folder:
            index = pd.MultiIndex.from_tuples(rows, names=["Run", "Pressure Type"])
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
            grid = LinearGrid(rho=rho, M=0, N=0, NFP=eq_opt_CON.NFP, sym=eq_opt_CON.sym)

            p_FXD = eq_opt_FXD.compute("p", grid=grid)["p"]
            p_CON = eq_opt_CON.compute("p", grid=grid)["p"]

            plt.figure(figsize=(7, 5))
            plt.plot(rho, p_FXD, linewidth=2, label="Fixed Pressure", color="blue")
            plt.plot(rho, p_CON, linewidth=2, label="Optimized Pressure", color="red")
            plt.xlabel(r"$\rho$", fontsize=14)
            plt.ylabel("Pressure", fontsize=14)
            plt.title(
                f"Pressure vs $\\rho$ (p_max={p_axis:.2g}, n={label_n})",
                fontsize=14,
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            pressure_path = os.path.join(out_dir, "pressure_compare.png")
            plt.savefig(pressure_path, dpi=200)
            plt.close()
            #----------

            #-------------------------------------------------------
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
            #-------------------------------------------------------

            #-----------------------
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
                f"Flux-surface avg current magnitude (p_max={p_axis:.2g}, n={label_n})",
                fontsize=13,
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            J_mag_path = os.path.join(out_dir, "J_mag.png")
            plt.savefig(J_mag_path, dpi=200)
            plt.close()
            #-----------------------

            #----------------------
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
                f"Rotational transform vs $\\rho$ (p_max={p_axis:.2g}, n={label_n})",
                fontsize=13,
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            iota_path = os.path.join(out_dir, "iota.png")
            plt.savefig(iota_path, dpi=200)
            plt.close()
            #----------------------
        #==========================
    #==============================
#==============================================================================================================================================================









#============== CONFIGURATION ENTRYPOINT ==========================================================================================================================
def run_from_config(eq_config: dict, opt_config: dict, driver_config: dict):
    comparison(
        p_maxima = driver_config["p_maxima"],
        n_set = driver_config["n_set"],
        eq_config = eq_config,
        opt_config = opt_config,
    )
#==============================================================================================================================================================