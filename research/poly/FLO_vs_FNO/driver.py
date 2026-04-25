# driver.py

#===================================================================================================================================================
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


from .config import (
    DRIVER_CONFIG,
    EQ_INPUT_CONFIG,
    OPT_CONFIG,
)
from .eq import run_equilibrium
from .helper import (
    _append_progress_log,
    _collect_active_columns,
    _filter_entries_for_slurm_task,
    _get_slurm_array_info,
    _load_eq_from_file,
    _make_root_out_dir,
    _make_skipped_run_status,
    _resolve_execution_mode,
    _run_one_formulation,
    _save_continuation_status_report,
    _save_optimization_status_report,
    _save_text_file,
    _write_batch_summary,
    _write_comparison_table,
    _write_readme,
    _write_surface_source_summary,
    build_eq_configs,
)
from .plotting import (
    save_all_solution_plots,
    save_initial_toroidal_cuts,
)
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
            plot_labels.append("FLO")
            plot_colors.append("purple")

        if FNO_plottable:
            plot_eqs.append(opt_FNO)
            plot_labels.append("FNO")
            plot_colors.append("orange")

        if eq_raw is not None:
            plot_eqs.append(eq_raw)
            plot_labels.append("Raw initial")
            plot_colors.append("green")

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