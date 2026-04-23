#===================================================================================================================================================
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from .config import (
    DRIVER_CONFIG,
    EQ_INPUT_CONFIG,
    OPT_CONFIG,
)
from .eq import run_equilibrium
from .helper import (
    _next_run_dir,
    _safe_extract_from_result,
    _save_optimization_status_report,
    _write_readme,
    _write_surface_source_summary,
    build_eq_config,
    sci_compact,
)
from .opt import run_optimization
from .plotting import (
    save_all_solution_plots,
    save_initial_toroidal_cuts,
)
#===================================================================================================================================================











#============== COMPARISON DRIVER ==================================================================================================================
def comparison(
        eq_input_config: dict,
        opt_config: dict,
        driver_config: dict,
    ):
    """
    Runs initial equilibrium solve, then optimization for:
        1) proximal-lsq-exact using Core + FLO
        2) proximal-lsq-exact using Core + FNO

    Run from DESC root with:
        python3 -m research.poly.FLO_vs_FNO.driver
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    eq_config, selected_point = build_eq_config(
        eq_input_config = eq_input_config,
    )

    NFP = eq_config["NFP"]
    opt_toggles_core = opt_config["opt_toggles_core"]
    opt_toggles_FLO = opt_config["opt_toggles_FLO"]
    opt_toggles_FNO = opt_config["opt_toggles_FNO"]
    config_path = driver_config["config_path"]

    out_dir = _next_run_dir(base_dir)
    _write_readme(out_dir, config_path)
    _write_surface_source_summary(out_dir, selected_point)

    column_map_core = [
        ("forcebalance_obj", "Force error: ", "Force error"),
        ("qs", f"Quasi-symmetry (1,{NFP}) Boozer error: ", "QS Boozer error"),
        ("ballooning", "Ideal ballooning lambda: ", "Ideal ballooning lambda"),
        ("mercier", "Mercier Stability: ", "Mercier Stability"),
    ]

    column_map_FLO = [
        ("FLO_pressure_axis", "FLO_pressure_axis", "FLO pressure axis"),
        ("FLO_pressure_shape", "FLO_pressure_shape", "FLO pressure shape"),
        ("FLO_iota_axis", "FLO_iota_axis", "FLO iota axis"),
        ("FLO_iota_edge", "FLO_iota_edge", "FLO iota edge"),
    ]

    column_map_FNO = [
        ("FNO_pressure", "FNO_pressure", "FNO pressure"),
        ("FNO_pressure_monotonic", "FNO_pressure_monotonic", "FNO pressure monotonic"),
        ("FNO_grad_pressure_edge", "FNO_grad_pressure_edge", "FNO grad pressure edge"),
        ("FNO_iota", "FNO_iota", "FNO iota"),
    ]

    active_columns_FLO = []
    for key, result_label, column_title in column_map_core:
        if opt_toggles_core.get(key, {}).get("use", False):
            active_columns_FLO.append((key, result_label, column_title))
    for key, result_label, column_title in column_map_FLO:
        if opt_toggles_FLO.get(key, {}).get("use", False):
            active_columns_FLO.append((key, result_label, column_title))

    active_columns_FNO = []
    for key, result_label, column_title in column_map_core:
        if opt_toggles_core.get(key, {}).get("use", False):
            active_columns_FNO.append((key, result_label, column_title))
    for key, result_label, column_title in column_map_FNO:
        if opt_toggles_FNO.get(key, {}).get("use", False):
            active_columns_FNO.append((key, result_label, column_title))

    #----------------------------
    # Initial equilibrium solve:
    eq_raw, eq_init = run_equilibrium(eq_config = eq_config)
    eq_init.save(os.path.join(out_dir, "eq_init.h5"))

    save_initial_toroidal_cuts(
        out_dir = out_dir,
        eq_raw = eq_raw,
        eq_init = eq_init,
    )
    #----------------------------

    #-------------------
    # Optimization run:
    eq_FLO_0 = eq_init.copy()
    eq_FNO_0 = eq_init.copy()

    optimizer = "proximal-lsq-exact"

    opt_FLO, opt_result_FLO, status_FLO = run_optimization(
        eq_0 = eq_FLO_0,
        optimizer = optimizer,
        opt_config = opt_config,
        formulation = "FLO",
    )

    opt_FNO, opt_result_FNO, status_FNO = run_optimization(
        eq_0 = eq_FNO_0,
        optimizer = optimizer,
        opt_config = opt_config,
        formulation = "FNO",
    )
    #-------------------

    #--------------------------------------
    # Save every returned optimized eq:
    if opt_FLO is not None:
        opt_FLO.save(os.path.join(out_dir, "opt_FLO.h5"))

    if opt_FNO is not None:
        opt_FNO.save(os.path.join(out_dir, "opt_FNO.h5"))
    #--------------------------------------

    #--------------------------------------
    # Determine which runs are plottable:
    FLO_plottable = (
        (opt_FLO is not None)
        and bool(status_FLO["plottable"])
    )
    FNO_plottable = (
        (opt_FNO is not None)
        and bool(status_FNO["plottable"])
    )

    status_FLO["plotted"] = FLO_plottable
    status_FNO["plotted"] = FNO_plottable

    optimization_status = {
        "FLO": status_FLO,
        "FNO": status_FNO,
    }

    _save_optimization_status_report(
        out_dir = out_dir,
        optimization_status = optimization_status,
    )
    #--------------------------------------

    #---------------------------
    # Build comparison table:
    all_columns = []
    seen_titles = set()

    for _, result_label, column_title in active_columns_FLO + active_columns_FNO:
        if column_title not in seen_titles:
            all_columns.append((result_label, column_title))
            seen_titles.add(column_title)

    FLO_title_to_label = {
        column_title: result_label
        for _, result_label, column_title in active_columns_FLO
    }
    FNO_title_to_label = {
        column_title: result_label
        for _, result_label, column_title in active_columns_FNO
    }

    row_FLO = []
    row_FNO = []

    for _, column_title in all_columns:
        if column_title in FLO_title_to_label:
            fmin_FLO, fmean_FLO, fmax_FLO = _safe_extract_from_result(
                opt_result_FLO,
                FLO_title_to_label[column_title],
            )
            if np.isnan(fmin_FLO) and np.isnan(fmean_FLO) and np.isnan(fmax_FLO):
                row_FLO.append("---")
            else:
                row_FLO.append(
                    f"f_min={sci_compact(fmin_FLO, sig = 4)}, "
                    f"f_mean={sci_compact(fmean_FLO, sig = 4)}, "
                    f"f_max={sci_compact(fmax_FLO, sig = 4)}"
                )
        else:
            row_FLO.append("---")

        if column_title in FNO_title_to_label:
            fmin_FNO, fmean_FNO, fmax_FNO = _safe_extract_from_result(
                opt_result_FNO,
                FNO_title_to_label[column_title],
            )
            if np.isnan(fmin_FNO) and np.isnan(fmean_FNO) and np.isnan(fmax_FNO):
                row_FNO.append("---")
            else:
                row_FNO.append(
                    f"f_min={sci_compact(fmin_FNO, sig = 4)}, "
                    f"f_mean={sci_compact(fmean_FNO, sig = 4)}, "
                    f"f_max={sci_compact(fmax_FNO, sig = 4)}"
                )
        else:
            row_FNO.append("---")

    if opt_FLO is not None:
        beta_FLO = float(
            opt_FLO.compute("<beta>_vol", override_grid = True)["<beta>_vol"]
        )
        row_FLO.append(f"{beta_FLO:.4g}")
    else:
        row_FLO.append("---")

    if opt_FNO is not None:
        beta_FNO = float(
            opt_FNO.compute("<beta>_vol", override_grid = True)["<beta>_vol"]
        )
        row_FNO.append(f"{beta_FNO:.4g}")
    else:
        row_FNO.append("---")

    values = [row_FLO, row_FNO]

    index = pd.Index(
        ["Core + FLO", "Core + FNO"],
        name = "Run",
    )
    df = pd.DataFrame(
        values,
        index = index,
        columns = [column_title for _, column_title in all_columns] + ["Beta"],
    )

    ascii_table = tabulate(df, headers = "keys", tablefmt = "grid")

    output_file = os.path.join(out_dir, "comparison.txt")
    with open(output_file, "w") as f:
        f.write("Comparison of Post-Optimization Objectives\n\n")
        f.write(ascii_table)
    #---------------------------

    #---------------------------
    # Plot only plottable runs:
    plot_eqs = []
    plot_labels = []
    plot_colors = []

    if FLO_plottable:
        plot_eqs.append(opt_FLO)
        plot_labels.append("Core + FLO")
        plot_colors.append("purple")

    if FNO_plottable:
        plot_eqs.append(opt_FNO)
        plot_labels.append("Core + FNO")
        plot_colors.append("orange")

    if len(plot_eqs) > 0:
        save_all_solution_plots(
            out_dir = out_dir,
            eqs = plot_eqs,
            labels = plot_labels,
            colors = plot_colors,
        )
    else:
        print("\nNo post-optimization plots generated.")
        print("Only initial_toroidal_cuts.png was saved.")
    #---------------------------
#===================================================================================================================================================











#============== MODULE ENTRYPOINT ==================================================================================================================
def main():
    comparison(
        eq_input_config = EQ_INPUT_CONFIG,
        opt_config = OPT_CONFIG,
        driver_config = DRIVER_CONFIG,
    )


if __name__ == "__main__":
    main()
#===================================================================================================================================================