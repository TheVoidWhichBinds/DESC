# driver.py

#===================================================================================================================================================
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from desc.io import load as desc_load
from tabulate import tabulate

from .config import (
    DRIVER_CONFIG,
    EQ_INPUT_CONFIG,
    OPT_CONFIG,
)
from .eq import run_equilibrium
from .helper import (
    _append_progress_log,
    _make_skipped_run_status,
    _next_run_dir,
    _safe_extract_from_result,
    _save_continuation_status_report,
    _save_optimization_status_report,
    _save_text_file,
    _write_readme,
    _write_surface_source_summary,
    build_eq_configs,
    sci_compact,
)
from .opt import run_optimization
from .plotting import (
    save_all_solution_plots,
    save_initial_toroidal_cuts,
)
#===================================================================================================================================================











#============== WORKER HELPERS =====================================================================================================================
#========================================
def _load_eq_from_file(
        eq_path,
    ):
    """
    Load a saved DESC equilibrium from disk.
    """
    eq = desc_load(eq_path)

    if isinstance(eq, (list, tuple)):
        if len(eq) == 0:
            raise ValueError(f"No equilibrium objects found in file: {eq_path}")
        eq = eq[-1]

    return eq
#========================================





#========================================
def _run_one_formulation(
        eq_init_path,
        optimizer,
        opt_config,
        formulation,
        out_dir,
    ):
    """
    Worker entrypoint for one optimization formulation.
    Loads eq_init from disk, runs the optimization, saves any returned
    equilibrium, and returns only lightweight metadata to the parent.
    """
    try:
        eq_0 = _load_eq_from_file(eq_init_path)

        eq_opt, opt_result, run_status, optimization_log = run_optimization(
            eq_0 = eq_0,
            optimizer = optimizer,
            opt_config = opt_config,
            formulation = formulation,
        )

        eq_opt_path = None

        if eq_opt is not None:
            eq_opt_path = os.path.join(out_dir, f"opt_{formulation}.h5")
            eq_opt.save(eq_opt_path)

        return eq_opt_path, opt_result, run_status, optimization_log

    except Exception:
        worker_trace = traceback.format_exc()

        run_status = {
            "formulation": formulation,
            "optimizer": optimizer,
            "equilibrium_returned": False,
            "exception_raised": True,
            "bad_approximation_failure": False,
            "automatic_continuation_failure": False,
            "plottable": False,
            "plotted": False,
            "failure_stage": "worker_exception",
            "message": worker_trace.strip().splitlines()[-1],
            "final_iterations": None,
            "runtime_seconds": None,
        }

        return None, None, run_status, worker_trace
#========================================





#========================================
def _resolve_execution_mode(
        driver_config,
    ):
    """
    Decide whether to run FLO/FNO sequentially or in parallel.

    Modes:
        - "local"   : always sequential
        - "cluster" : always parallel
        - "auto"    : parallel if running inside Slurm, else sequential
    """
    mode = driver_config.get("execution_mode", "auto")

    if mode not in {"auto", "local", "cluster"}:
        raise ValueError(
            f"Unsupported execution_mode='{mode}'. "
            f"Use 'auto', 'local', or 'cluster'."
        )

    if mode == "auto":
        if os.environ.get("SLURM_JOB_ID") is not None:
            return "cluster"
        return "local"

    return mode
#========================================




#========================================
def _get_slurm_array_info():
    """
    Return Slurm array task info if running as a Slurm array.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    task_min = os.environ.get("SLURM_ARRAY_TASK_MIN")

    if job_id is None or task_id is None:
        return None

    if task_count is None:
        task_count = "1"

    if task_min is None:
        task_min = task_id

    return {
        "job_id": str(job_id),
        "task_id": int(task_id),
        "task_count": int(task_count),
        "task_min": int(task_min),
    }
#========================================





#========================================
def _filter_entries_for_slurm_task(
        eq_config_entries,
        slurm_array_info,
    ):
    """
    Give each Slurm array task a disjoint subset of equilibria.
    """
    if slurm_array_info is None:
        return eq_config_entries

    task_id = slurm_array_info["task_id"]
    task_count = slurm_array_info["task_count"]
    task_min = slurm_array_info["task_min"]
    zero_based_task_index = task_id - task_min

    return [
        entry for i, entry in enumerate(eq_config_entries)
        if i % task_count == zero_based_task_index
    ]
#========================================





#========================================
def _make_root_out_dir(
        base_dir,
        execution_mode,
        slurm_array_info,
    ):
    """
    Create a collision-safe root output directory.
    """
    if execution_mode == "cluster":
        job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")

        if slurm_array_info is not None:
            task_id = slurm_array_info["task_id"]
            root_out_dir = os.path.join(
                base_dir,
                f"run_{job_id}_task_{task_id:03d}",
            )
        else:
            root_out_dir = os.path.join(
                base_dir,
                f"run_{job_id}",
            )

        os.makedirs(root_out_dir, exist_ok = False)
        return root_out_dir

    return _next_run_dir(base_dir)
#========================================





#========================================
def _collect_active_columns(
        opt_config,
        NFP,
    ):
    """
    Collect only shared core result columns for FLO and FNO comparison tables.
    """
    opt_toggles_core = opt_config["opt_toggles_core"]

    column_map_core = [
        ("forcebalance_obj", "Force error: ", "Force error"),
        ("qs", f"Quasi-symmetry (1,{NFP}) Boozer error: ", "QS Boozer error"),
        ("ballooning", "Ideal ballooning lambda: ", "Ideal ballooning lambda"),
        ("mercier", "Mercier Stability: ", "Mercier Stability"),
        ("current_Redl", "current_Redl", "Current Redl"),
    ]

    active_columns = []

    for key, result_label, column_title in column_map_core:
        if opt_toggles_core.get(key, {}).get("use", False):
            active_columns.append((key, result_label, column_title))

    return active_columns, active_columns
#========================================






#========================================
def _write_comparison_table(
        out_dir,
        active_columns_FLO,
        active_columns_FNO,
        opt_result_FLO,
        opt_result_FNO,
        opt_FLO,
        opt_FNO,
        status_FLO,
        status_FNO,
    ):
    """
    Save a text comparison table for FLO vs FNO.
    """
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

        f.write("FLO status\n")
        f.write(f"  plottable = {status_FLO.get('plottable')}\n")
        f.write(f"  runtime_seconds = {status_FLO.get('runtime_seconds')}\n")
        f.write(f"  final_iterations = {status_FLO.get('final_iterations')}\n")
        f.write(f"  message = {status_FLO.get('message')}\n\n")

        f.write("FNO status\n")
        f.write(f"  plottable = {status_FNO.get('plottable')}\n")
        f.write(f"  runtime_seconds = {status_FNO.get('runtime_seconds')}\n")
        f.write(f"  final_iterations = {status_FNO.get('final_iterations')}\n")
        f.write(f"  message = {status_FNO.get('message')}\n\n")

        f.write(ascii_table)
#========================================





#========================================
def _write_batch_summary(
        root_out_dir,
        batch_rows,
    ):
    """
    Save one batch-level summary CSV across all equilibria.
    """
    if len(batch_rows) == 0:
        return

    df = pd.DataFrame(batch_rows)
    save_path = os.path.join(root_out_dir, "batch_summary.csv")
    df.to_csv(save_path, index = False)
#========================================
#===================================================================================================================================================











#============== COMPARISON DRIVER ==================================================================================================================
#========================================
def comparison(
        eq_input_config: dict,
        opt_config: dict,
        driver_config: dict,
    ):
    """
    Runs N_eq continuation solves, where the surfaces are pulled from the
    shuffled pool of nested neural initializations.

    Behavior:
        - local computer  -> FLO and FNO run sequentially
        - supercomputer   -> FLO and FNO run in parallel

    Run from DESC root with:
        python3 -m research.poly.FLO_vs_FNO.driver
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = driver_config["config_path"]
    optimizer = "proximal-lsq-exact"
    NFP = eq_input_config["NFP"]
    execution_mode = _resolve_execution_mode(driver_config)
    cluster_max_workers = int(driver_config.get("cluster_max_workers", 2))

    slurm_array_info = _get_slurm_array_info()

    eq_config_entries = build_eq_configs(
        eq_input_config = eq_input_config,
    )

    eq_config_entries = _filter_entries_for_slurm_task(
        eq_config_entries = eq_config_entries,
        slurm_array_info = slurm_array_info,
    )

    active_columns_FLO, active_columns_FNO = _collect_active_columns(
        opt_config = opt_config,
        NFP = NFP,
    )

    root_out_dir = _make_root_out_dir(
        base_dir = base_dir,
        execution_mode = execution_mode,
        slurm_array_info = slurm_array_info,
    )

    _write_readme(root_out_dir, config_path)

    batch_rows = []

    for entry in eq_config_entries:
        run_index = entry["run_index"]
        eq_config = entry["eq_config"]
        selected_point = entry["selected_point"]

        out_dir = os.path.join(root_out_dir, f"eq_{run_index:03d}")
        os.makedirs(out_dir, exist_ok = False)

        _write_surface_source_summary(
            out_dir = out_dir,
            selected_point = selected_point,
        )

        _append_progress_log(
            out_dir = out_dir,
            message = f"Started run_index = {run_index}",
        )

        #----------------------------
        # Initial equilibrium solve:
        _append_progress_log(
            out_dir = out_dir,
            message = "Starting automatic continuation.",
        )

        eq_raw, eq_init, continuation_status, continuation_log = run_equilibrium(
            eq_config = eq_config,
        )

        _save_text_file(
            out_dir = out_dir,
            filename = "continuation.log",
            text = continuation_log,
        )

        _save_continuation_status_report(
            out_dir = out_dir,
            continuation_status = continuation_status,
        )

        _append_progress_log(
            out_dir = out_dir,
            message = (
                "Finished automatic continuation. "
                f"continuation_returned = {continuation_status['continuation_returned']}, "
                f"automatic_continuation_failure = {continuation_status['automatic_continuation_failure']}"
            ),
        )

        if eq_raw is not None:
            eq_raw_path = os.path.join(out_dir, "eq_raw.h5")
            eq_raw.save(eq_raw_path)

        if eq_init is not None:
            eq_init_path = os.path.join(out_dir, "eq_init.h5")
            eq_init.save(eq_init_path)

            try:
                save_initial_toroidal_cuts(
                    out_dir = out_dir,
                    eq_raw = eq_raw,
                    eq_init = eq_init,
                )
                _append_progress_log(
                    out_dir = out_dir,
                    message = "Saved initial_toroidal_cuts.png.",
                )

            except Exception:
                _save_text_file(
                    out_dir = out_dir,
                    filename = "initial_plot_error.txt",
                    text = traceback.format_exc(),
                )
                _append_progress_log(
                    out_dir = out_dir,
                    message = "Initial toroidal-cut plotting failed. See initial_plot_error.txt.",
                )

        else:
            eq_init_path = None
        #----------------------------

        #----------------------------
        # Abort whole run if continuation failed:
        if eq_init is None:
            status_FLO = _make_skipped_run_status(
                formulation = "FLO",
                message = "Skipped because initial continuation failed.",
                failure_stage = "skipped_due_to_continuation_failure",
            )

            status_FNO = _make_skipped_run_status(
                formulation = "FNO",
                message = "Skipped because initial continuation failed.",
                failure_stage = "skipped_due_to_continuation_failure",
            )

            optimization_status = {
                "FLO": status_FLO,
                "FNO": status_FNO,
            }

            _save_optimization_status_report(
                out_dir = out_dir,
                optimization_status = optimization_status,
            )

            _write_comparison_table(
                out_dir = out_dir,
                active_columns_FLO = active_columns_FLO,
                active_columns_FNO = active_columns_FNO,
                opt_result_FLO = None,
                opt_result_FNO = None,
                opt_FLO = None,
                opt_FNO = None,
                status_FLO = status_FLO,
                status_FNO = status_FNO,
            )

            batch_rows.append(
                {
                    "run_index": run_index,
                    "execution_mode": execution_mode,
                    "surface_is_nested": bool(selected_point.get("labels", {}).get("is_nested", False)),
                    "surface_build_ok": bool(selected_point.get("labels", {}).get("build_ok", False)),
                    "continuation_returned": bool(continuation_status["continuation_returned"]),
                    "continuation_automatic_failure": bool(continuation_status["automatic_continuation_failure"]),
                    "continuation_exception_raised": bool(continuation_status["exception_raised"]),
                    "continuation_runtime_seconds": continuation_status["runtime_seconds"],
                    "continuation_message": continuation_status["message"],
                    "FLO_equilibrium_returned": False,
                    "FLO_bad_approximation_failure": False,
                    "FLO_automatic_continuation_failure": False,
                    "FLO_final_iterations": None,
                    "FLO_runtime_seconds": None,
                    "FLO_plottable": False,
                    "FLO_message": status_FLO["message"],
                    "FNO_equilibrium_returned": False,
                    "FNO_bad_approximation_failure": False,
                    "FNO_automatic_continuation_failure": False,
                    "FNO_final_iterations": None,
                    "FNO_runtime_seconds": None,
                    "FNO_plottable": False,
                    "FNO_message": status_FNO["message"],
                }
            )

            _append_progress_log(
                out_dir = out_dir,
                message = "Skipping FLO/FNO because continuation did not return a usable eq_init.",
            )

            continue
        #----------------------------

        #-----------------------------
        # FLO / FNO solves:
        _append_progress_log(
            out_dir = out_dir,
            message = "Starting FLO/FNO optimizations.",
        )

        if execution_mode == "cluster":
            results_by_formulation = {}

            with ProcessPoolExecutor(max_workers = cluster_max_workers) as executor:
                future_map = {
                    executor.submit(
                        _run_one_formulation,
                        eq_init_path,
                        optimizer,
                        opt_config,
                        "FLO",
                        out_dir,
                    ): "FLO",
                    executor.submit(
                        _run_one_formulation,
                        eq_init_path,
                        optimizer,
                        opt_config,
                        "FNO",
                        out_dir,
                    ): "FNO",
                }

                for future in as_completed(future_map):
                    formulation = future_map[future]
                    results_by_formulation[formulation] = future.result()

                    _append_progress_log(
                        out_dir = out_dir,
                        message = f"{formulation} optimization finished.",
                    )

            opt_FLO_path, opt_result_FLO, status_FLO, log_FLO = results_by_formulation["FLO"]
            opt_FNO_path, opt_result_FNO, status_FNO, log_FNO = results_by_formulation["FNO"]

        else:
            opt_FLO_path, opt_result_FLO, status_FLO, log_FLO = _run_one_formulation(
                eq_init_path = eq_init_path,
                optimizer = optimizer,
                opt_config = opt_config,
                formulation = "FLO",
                out_dir = out_dir,
            )
            _append_progress_log(
                out_dir = out_dir,
                message = "FLO optimization finished.",
            )

            opt_FNO_path, opt_result_FNO, status_FNO, log_FNO = _run_one_formulation(
                eq_init_path = eq_init_path,
                optimizer = optimizer,
                opt_config = opt_config,
                formulation = "FNO",
                out_dir = out_dir,
            )
            _append_progress_log(
                out_dir = out_dir,
                message = "FNO optimization finished.",
            )
        #-----------------------------

        _save_text_file(
            out_dir = out_dir,
            filename = "FLO_optimization.log",
            text = log_FLO,
        )

        _save_text_file(
            out_dir = out_dir,
            filename = "FNO_optimization.log",
            text = log_FNO,
        )

        #--------------------------------------
        # Load any returned optimized eqs:
        opt_FLO = None
        opt_FNO = None

        if opt_FLO_path is not None:
            opt_FLO = _load_eq_from_file(opt_FLO_path)

        if opt_FNO_path is not None:
            opt_FNO = _load_eq_from_file(opt_FNO_path)
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

        status_FLO["plotted"] = False
        status_FNO["plotted"] = False

        optimization_status = {
            "FLO": status_FLO,
            "FNO": status_FNO,
        }
        #--------------------------------------

        #---------------------------
        # Save comparison table:
        _write_comparison_table(
            out_dir = out_dir,
            active_columns_FLO = active_columns_FLO,
            active_columns_FNO = active_columns_FNO,
            opt_result_FLO = opt_result_FLO,
            opt_result_FNO = opt_result_FNO,
            opt_FLO = opt_FLO,
            opt_FNO = opt_FNO,
            status_FLO = status_FLO,
            status_FNO = status_FNO,
        )

        _append_progress_log(
            out_dir = out_dir,
            message = "Saved comparison.txt.",
        )
        #---------------------------

        #---------------------------
        # Plot FLO, FNO, then initial continuation equilibrium:
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

        plot_eqs.append(eq_init)
        plot_labels.append("Initial (post continuation)")
        plot_colors.append("blue")

        try:
            save_all_solution_plots(
                out_dir = out_dir,
                eqs = plot_eqs,
                labels = plot_labels,
                colors = plot_colors,
            )

            status_FLO["plotted"] = FLO_plottable
            status_FNO["plotted"] = FNO_plottable

            _append_progress_log(
                out_dir = out_dir,
                message = "Saved final solution plots.",
            )

        except Exception:
            _save_text_file(
                out_dir = out_dir,
                filename = "solution_plot_error.txt",
                text = traceback.format_exc(),
            )

            _append_progress_log(
                out_dir = out_dir,
                message = "Final plotting failed. See solution_plot_error.txt.",
            )
        #---------------------------

        _save_optimization_status_report(
            out_dir = out_dir,
            optimization_status = optimization_status,
        )

        #---------------------------
        # Accumulate batch summary:
        batch_rows.append(
            {
                "run_index": run_index,
                "execution_mode": execution_mode,
                "surface_is_nested": bool(selected_point.get("labels", {}).get("is_nested", False)),
                "surface_build_ok": bool(selected_point.get("labels", {}).get("build_ok", False)),
                "continuation_returned": bool(continuation_status["continuation_returned"]),
                "continuation_automatic_failure": bool(continuation_status["automatic_continuation_failure"]),
                "continuation_exception_raised": bool(continuation_status["exception_raised"]),
                "continuation_runtime_seconds": continuation_status["runtime_seconds"],
                "continuation_message": continuation_status["message"],
                "FLO_equilibrium_returned": bool(status_FLO["equilibrium_returned"]),
                "FLO_bad_approximation_failure": bool(status_FLO["bad_approximation_failure"]),
                "FLO_automatic_continuation_failure": bool(status_FLO["automatic_continuation_failure"]),
                "FLO_final_iterations": status_FLO["final_iterations"],
                "FLO_runtime_seconds": status_FLO["runtime_seconds"],
                "FLO_plottable": bool(FLO_plottable),
                "FLO_message": status_FLO["message"],
                "FNO_equilibrium_returned": bool(status_FNO["equilibrium_returned"]),
                "FNO_bad_approximation_failure": bool(status_FNO["bad_approximation_failure"]),
                "FNO_automatic_continuation_failure": bool(status_FNO["automatic_continuation_failure"]),
                "FNO_final_iterations": status_FNO["final_iterations"],
                "FNO_runtime_seconds": status_FNO["runtime_seconds"],
                "FNO_plottable": bool(FNO_plottable),
                "FNO_message": status_FNO["message"],
            }
        )
        #---------------------------

        _append_progress_log(
            out_dir = out_dir,
            message = "Finished equilibrium run.",
        )

    _write_batch_summary(
        root_out_dir = root_out_dir,
        batch_rows = batch_rows,
    )
#========================================
#===================================================================================================================================================











#============== MODULE ENTRYPOINT ==================================================================================================================
#================
def main():
    comparison(
        eq_input_config = EQ_INPUT_CONFIG,
        opt_config = OPT_CONFIG,
        driver_config = DRIVER_CONFIG,
    )
#================


if __name__ == "__main__":
    main()
#===================================================================================================================================================