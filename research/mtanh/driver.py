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
#====================
    

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
    #------------------

    #--------------------------------------------------------------------------------------
    if isinstance(objval, dict) and all(k in objval for k in ("f_min", "f_mean", "f_max")):
        return (
            _to_float(objval["f_min"]),
            _to_float(objval["f_mean"]),
            _to_float(objval["f_max"]),
        )
    #----------------------------------

    #---------------------------
    if isinstance(objval, dict):
        for v in objval.values():
            val = _to_float(v)
            if not np.isnan(val):
                return val, val, val
        return np.nan, np.nan, np.nan

    val = _to_float(objval)
    return val, val, val
    #-------------------
#=======================
#==============================================================================================================================================================









#============== FIXED VS. OPTIMIZED PRESSURE COMPARISON =======================================================================================================
def comparison(p_maxima: list, n_set: list):
    """
    Runs initial equilibrium solve, then optimization for 
    both fixed and optimized pressure. Plots, and objective
    table are also generated.
    p_maxima: 
        list of maximum pressures to loop over.
    n_set: 
        list of polynomial orders (really 2*n) to loop over
        n in n_set must be >=2.
    """

    #========================================
    # Initializing table of objective values:
    columns = [
        'Force error: ',
        'Quasi-symmetry (1,19) Boozer error: ',
        'Aspect ratio: ',
        'Fixed iota profile error: ',
        'Fixed Psi error: ',
        'Ideal ballooning lambda: ',
        'Mercier Stability: ',
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    #====================================================


    #==================================================
    # Loop over on-axis pressure and polynomial orders:
    for p_axis in p_maxima:
        for n in n_set:

            #-------------------------------------------------------
            # Generates directory to store equilibria, plots, table: 
            out_dir = os.path.join(base_dir, f"p{sci_compact(p_axis)}_n{n}")
            os.makedirs(out_dir, exist_ok=True)

            # Running initial equilibrium solve:
            eq_init, n_eff = run_equilibrium(p_axis, n, out_dir=out_dir)

            # Running fixed and optimized pressure optimizations:
            eq_opt_FXP, opt_result_FXP = run_optimization(
                p_axis, out_dir=out_dir, fix_pressure=True
            )
            eq_opt, opt_result = run_optimization(
                p_axis, out_dir=out_dir, fix_pressure=False
            )
            #----------------------------------------------


            #------------------------------------
            # Generating comparison table labels:
            rows = [] # table row names
            values = [] # table elements (obj vals)
            label_n = n if n_eff == n else f"{n}→{n_eff}"
            group_label = f"Max Pressure = {p_axis}; n = {label_n} (order={2*n_eff})"
            rows.append((group_label, "Fixed Pressure"))
            rows.append((group_label, "Optimized Pressure"))
            rows.append((group_label, "Difference"))

            row_FXP = [] # fixed pressure obj vals
            row_OPT = [] # optimized pressure obj vals
            row_DIFF = [] # difference between fixed and opt vals

            # Extracting objective values:
            for key in columns:
                fmin_FXP, fmean_FXP, fmax_FXP = _extract_f_stats(
                    opt_result_FXP['Objective values'][key]
                )
                fmin_OPT, fmean_OPT, fmax_OPT = _extract_f_stats(
                    opt_result['Objective values'][key]
                )

                row_FXP.append(
                    f"f_min={sci_compact(fmin_FXP, sig=4)}, "
                    f"f_mean={sci_compact(fmean_FXP, sig=4)}, "
                    f"f_max={sci_compact(fmax_FXP, sig=4)}"
                )
                row_OPT.append(
                    f"f_min={sci_compact(fmin_OPT, sig=4)}, "
                    f"f_mean={sci_compact(fmean_OPT, sig=4)}, "
                    f"f_max={sci_compact(fmax_OPT, sig=4)}"
                )

                dmin = fmin_OPT - fmin_FXP
                dmean = fmean_OPT - fmean_FXP
                dmax = fmax_OPT - fmax_FXP
                row_DIFF.append(
                    f"f_min diff={sci_compact(dmin, sig=4)}, "
                    f"f_mean diff={sci_compact(dmean, sig=4)}, "
                    f"f_max diff={sci_compact(dmax, sig=4)}"
                )

            # Including Beta in table:
            beta_FXP = float(eq_opt_FXP.compute("<beta>_vol", override_grid=True)["<beta>_vol"])
            beta_OPT = float(eq_opt.compute("<beta>_vol", override_grid=True)["<beta>_vol"])
            beta_DIFF = beta_OPT - beta_FXP

            row_FXP.append(f"{beta_FXP:.4g}")
            row_OPT.append(f"{beta_OPT:.4g}")
            row_DIFF.append(f"{beta_DIFF:.4g}")
            
            # 
            values.append(row_FXP)
            values.append(row_OPT)
            values.append(row_DIFF)

            # Save table inside run folder:
            index = pd.MultiIndex.from_tuples(rows, names=["Run", "Pressure Type"])
            df = pd.DataFrame(
                values,
                index=index,
                columns=[col.replace(': ', '') for col in columns] + ["Beta"],
            )
        
            ascii_table = tabulate(df, headers='keys', tablefmt='grid')
            
            output_file = os.path.join(out_dir, "comparison.txt")
            with open(output_file, "w") as f:
                f.write("Comparison of Post-Optimization Objectives\n\n")
                f.write(ascii_table)
            #-----------------------


            #--------------------
            # Plotting pressures:
            rho = np.linspace(0.0, 1.0, 400)
            grid = LinearGrid(rho=rho, M=0, N=0, NFP=eq_opt.NFP, sym=eq_opt.sym)

            p_FXP = eq_opt_FXP.compute("p", grid=grid)["p"]
            p_OPT = eq_opt.compute("p", grid=grid)["p"]

            plt.figure(figsize=(7, 5))
            plt.plot(rho, p_FXP, linewidth=2, label="Fixed Pressure", color='blue')
            plt.plot(rho, p_OPT, linewidth=2, label="Optimized Pressure", color='red')
            plt.xlabel(r"$\rho$", fontsize=14)
            plt.ylabel("Pressure", fontsize=14)
            plt.title(
                f"Pressure vs $\\rho$ (p_max={p_axis:.2g}, n={label_n})",
                fontsize=14,
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            pressure_path = os.path.join(out_dir, 'pressure_compare.png')
            plt.savefig(pressure_path, dpi=200)
            plt.close()
            #----------


            #-------------------------------------------------------
            # Plotting  gridded toroidal cross-sections of B-fields:
            plt.title('Toroidal Cross-Sections of Solved Equilibria')
            fig, ax = plot_comparison(
                eqs=[eq_init, eq_opt_FXP, eq_opt],
                labels=['Initial Equilibrium', 'Optimized (Fixed Pressure)','Optimized (Optimized Pressure)'],
                color=['green', 'blue', 'red']
            )

            toroidal_cuts_path = os.path.join(out_dir, 'toroidal_cuts.png')
            plt.savefig(toroidal_cuts_path, dpi=200)
            plt.close()
            #----------


            #------------------------------------
            rho_grid = np.linspace(0.0, 1.0, 100)
            #----------------------
            # |J| vs. rho plotting:
            grid_J = LinearGrid(rho=rho_grid, M=24, N=24, NFP=eq_opt.NFP, sym=eq_opt.sym)
            # Helper func for flux-surface averaging of |J|
            def _surface_mean_J_mag(eq):
                data = eq.compute(["|J|"], grid=grid_J)
                J_mag = np.asarray(data["|J|"])  # magnitude of J

                rho_nodes = grid_J.nodes[:, 0]
                rho_unique = np.unique(rho_nodes)

                J_mag_fs = np.empty_like(rho_unique, dtype=float)
                for i, r in enumerate(rho_unique):
                    mask = np.isclose(rho_nodes, r)
                    J_mag_fs[i] = np.mean(J_mag[mask])

                return rho_unique, J_mag_fs

            # Compute |J| profiles for fixed-pressure and optimized-pressure equilibria
            rho_u_FXP, J_mag_FXP = _surface_mean_J_mag(eq_opt_FXP)
            rho_u_OPT, J_mag_OPT = _surface_mean_J_mag(eq_opt)

            plt.figure(figsize=(7, 5))
            plt.plot(rho_u_FXP, J_mag_FXP, linewidth=2, label="Fixed Pressure", color='blue')
            plt.plot(rho_u_OPT, J_mag_OPT, linewidth=2, label="Optimized Pressure", color='red')
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
            #----------


            #----------------------
            # iota vs. rho plotting:
            grid_iota = LinearGrid(rho=rho_grid, M=0, N=0, NFP=eq_opt.NFP, sym=eq_opt.sym)
            # Compute iota profiles for fixed-pressure and optimized-pressure equilibria
            iota_FXP = eq_opt_FXP.compute("iota", grid=grid_iota)["iota"]
            iota_OPT = eq_opt.compute("iota", grid=grid_iota)["iota"]

            plt.figure(figsize=(7, 5))
            plt.plot(rho_grid, iota_FXP, linewidth=2, label="Fixed Pressure", color='blue')
            plt.plot(rho_grid, iota_OPT, linewidth=2, label="Optimized Pressure", color='red')
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
            #----------
    #==================
#=============================================================================================================================================================










#============== RUN IT ==============#
comparison([1e4], [2])
