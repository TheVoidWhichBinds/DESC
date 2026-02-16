import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from eq import run_equilibrium
from opt import run_optimization
import os
import pandas as pd
from tabulate import tabulate


def sci_compact(x, sig=2):
    s = f"{x:.{sig-1}e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"{mant}e{exp}"


def comparison(p_maxima: list, n_set: list):
    columns = [
        'Force error: ',
        'Quasi-symmetry (1,19) Boozer error: ',
    ]

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

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

    for p_scale in p_maxima:
        for n in n_set:

            rows = []
            values = []

            run_dir = os.path.join(base_dir, f"p{sci_compact(p_scale)}_n{n}")
            os.makedirs(run_dir, exist_ok=True)

            eq_path, n_eff = run_equilibrium(p_scale, n, out_dir=run_dir)

            eq_opt_FXD, opt_result_FXD, _ = run_optimization(
                p_scale, eq_path, out_dir=run_dir, fix_pressure=True
            )
            eq_opt, opt_result, _ = run_optimization(
                p_scale, eq_path, out_dir=run_dir, fix_pressure=False
            )

            label_n = n if n_eff == n else f"{n}→{n_eff}"
            group_label = f"Max Pressure = {p_scale}; n = {label_n} (order={2*n_eff})"

            rows.append((group_label, "Fixed Pressure"))
            rows.append((group_label, "Optimized Pressure"))
            rows.append((group_label, "Difference"))

            row_FXD = []
            row_OPT = []
            row_DIFF = []

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

            values.append(row_FXD)
            values.append(row_OPT)
            values.append(row_DIFF)

            # ---------- Save table inside run folder ----------
            index = pd.MultiIndex.from_tuples(rows, names=["Run", "Pressure Type"])
            df = pd.DataFrame(
                values,
                index=index,
                columns=[col.replace(': ', '') for col in columns],
            )

            ascii_table = tabulate(df, headers='keys', tablefmt='grid')

            output_file = os.path.join(run_dir, "optimization_comparison.txt")
            with open(output_file, "w") as f:
                f.write("Comparison of Post-Optimization Objectives\n\n")
                f.write(ascii_table)

            # ---------- Pressure Plot ----------
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

            pressure_path = os.path.join(run_dir, "pressure_compare.png")
            plt.savefig(pressure_path, dpi=200)
            plt.close()





comparison([1.8e4], [3])
