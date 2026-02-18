import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from eq import run_equilibrium
from opt import run_optimization
import os
import pandas as pd
from tabulate import tabulate
from desc.plotting import plot_comparison



def sci_compact(x, sig=2):
    """
    
    """
    s = f"{x:.{sig-1}e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"{mant}e{exp}"




#-------- FIXED VS. OPTIMIZED PRESSURE COMPARISON ---------
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



    #------------------------------------------
    # Initializing table of objective values:
    columns = [
        'Force error: ',
        'Quasi-symmetry (1,19) Boozer error: ',
        #'Aspect ratio error: '
        #'BallooningStability error: ',
        #'MercierStability error: ',
    ]

    # Converts ..........
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan
        
    
    # Preps final objective values to be put into comparison table:
    def _extract_f_stats(objval):
        if isinstance(objval, list):
            chosen = None
            for item in objval:
                if isinstance(item, dict) and all(k in item for k in ("f_min", "f_mean", "f_max")):
                    chosen = item
                    break
            if chosen is None and len(objval) > 0:
                chosen = objval[0]
            objval = chosen

        if isinstance(objval, dict) and all(k in objval for k in ("f_min", "f_mean", "f_max")):
            return _to_float(objval["f_min"]), _to_float(objval["f_mean"]), _to_float(objval["f_max"])

        if isinstance(objval, dict):
            for v in objval.values():
                val = _to_float(v)
                if not np.isnan(val):
                    return val, val, val
            return np.nan, np.nan, np.nan

        val = _to_float(objval)
        return val, val, val

    base_dir = os.path.dirname(os.path.abspath(__file__))



    #--------------------------------------------------
    # Loop over on-axis pressure and polynomial orders:
    for p_scale in p_maxima:
        for n in n_set:

            #---------------------------------------------------------------
            rows = [] # table row names
            values = [] # table elements (obj vals)

            # Generates directory to store equilibria, plots, table: 
            out_dir = os.path.join(base_dir, f"p{sci_compact(p_scale)}_n{n}")
            os.makedirs(out_dir, exist_ok=True)

            # Running initial equilibrium solve:
            eq_init, n_eff = run_equilibrium(p_scale, n, out_dir=out_dir)

            # Running fixed and optimized pressure optimizations:
            eq_opt_FXD, opt_result_FXD = run_optimization(
                p_scale, out_dir=out_dir, fix_pressure=True
            )
            eq_opt, opt_result = run_optimization(
                p_scale, out_dir=out_dir, fix_pressure=False
            )


            # Generating comparison table labels:
            label_n = n if n_eff == n else f"{n}→{n_eff}"
            group_label = f"Max Pressure = {p_scale}; n = {label_n} (order={2*n_eff})"
            rows.append((group_label, "Fixed Pressure"))
            rows.append((group_label, "Optimized Pressure"))
            rows.append((group_label, "Difference"))

            row_FXD = [] # fixed pressure obj vals
            row_OPT = [] # optimized pressure obj vals
            row_DIFF = [] # difference between fixed and opt vals

            # Extracting objective values:
            for key in columns:
                fmin_FXD, fmean_FXD, fmax_FXD = _extract_f_stats(
                    opt_result_FXD['Objective values'][key]
                )
                fmin_OPT, fmean_OPT, fmax_OPT = _extract_f_stats(
                    opt_result['Objective values'][key]
                )

                row_FXD.append(f"f_min={fmin_FXD:.4g}, f_mean={fmean_FXD:.4g}, f_max={fmax_FXD:.4g}")
                row_OPT.append(f"f_min={fmin_OPT:.4g}, f_mean={fmean_OPT:.4g}, f_max={fmax_OPT:.4g}")

                dmin = fmin_OPT - fmin_FXD
                dmean = fmean_OPT - fmean_FXD
                dmax = fmax_OPT - fmax_FXD
                row_DIFF.append(
                    f"f_min diff={dmin:.4g}, f_mean diff={dmean:.4g}, f_max diff={dmax:.4g}"
                )
            #
            values.append(row_FXD)
            values.append(row_OPT)
            values.append(row_DIFF)

            # Save table inside run folder:
            index = pd.MultiIndex.from_tuples(rows, names=["Run", "Pressure Type"])
            df = pd.DataFrame(
                values,
                index=index,
                columns=[col.replace(': ', '') for col in columns],
            )
        
            ascii_table = tabulate(df, headers='keys', tablefmt='grid')
            
            output_file = os.path.join(out_dir, "comparison.txt")
            with open(output_file, "w") as f:
                f.write("Comparison of Post-Optimization Objectives\n\n")
                f.write(ascii_table)



            #------------------------------------------------------------
            # Plotting fixed and optimized pressures:
            rho = np.linspace(0.0, 1.0, 400)
            grid = LinearGrid(rho=rho, M=0, N=0)

            p_FXD = eq_opt_FXD.compute("p", grid=grid)["p"]
            p_OPT = eq_opt.compute("p", grid=grid)["p"]

            plt.figure(figsize=(7, 5))
            plt.plot(rho, p_FXD, linewidth=2, label="Fixed Pressure")
            plt.plot(rho, p_OPT, linewidth=2, label="Optimized Pressure")
            plt.xlabel(r"$\rho$", fontsize=14)
            plt.ylabel("Pressure", fontsize=14)
            plt.title(
                f"Pressure vs $\\rho$ (p_max={p_scale:.2g}, n={label_n})",
                fontsize=14,
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            pressure_path = os.path.join(out_dir, 'pressure_compare.png')
            plt.savefig(pressure_path, dpi=200)
            plt.close()



            #--------------------------------------------------------
            # Plotting  gridded toroidal cross-sections of B-fields:
            plt.title('Toroidal Cross-Sections of Solved Equilibria')
            fig, ax = plot_comparison(
                eqs=[eq_init, eq_opt_FXD, eq_opt],
                labels=['Initial Equilibrium', 'Optimized (Fixed Pressure)','Optimized (Optimized Pressure)'],
            )

            toroidal_cuts_path = os.path.join(out_dir, 'toroidal_cuts.png')
            plt.savefig(toroidal_cuts_path, dpi=200)
            plt.close()




#-------- RUNNING IT --------
comparison([4e4], [2,3,4,5,6,7,8])
