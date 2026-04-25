# helper.py

#===================================================================================================================================================
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import inspect
import os
import pickle
import re
import traceback

import pandas as pd

import jax.numpy as jnp
import numpy as np
from desc.geometry import FourierRZToroidalSurface
from desc.io import load as desc_load
from tabulate import tabulate
#===================================================================================================================================================









#========================================
class Tee:
    """
    Write stream output to multiple file-like objects.

    This lets DESC verbose output print to terminal while also
    being captured into a log buffer.
    """
    def __init__(
            self,
            *streams,
        ):
        self.streams = streams


    def write(
            self,
            data,
        ):
        for stream in self.streams:
            stream.write(data)
            stream.flush()


    def flush(
            self,
        ):
        for stream in self.streams:
            stream.flush()
#========================================




#============== CONFIG HELPERS =====================================================================================================================

#==========================
def iota_between_rationals(
        iota_axis: float,
    ):
    """
    Generates bounds for iota optimizer constraint
    that are between low-order rational surfaces.
    """
    allowed_ranges = [
        (0.0,    0.25),
        (0.25,   0.3333),
        (0.3333, 0.5),
        (0.5,    0.6667),
        (0.6667, 0.75),
        (0.75,   1.0),
        (1.0,    1.25),
        (1.25,   1.3333),
        (1.3333, 1.5),
        (1.5,    1.6667),
        (1.6667, 2.0),
        (2.0,    3.0),
        (3.0,    4.0),
    ]

    matched_lower = None
    matched_upper = None

    for lower, upper in allowed_ranges:
        if lower <= iota_axis <= upper:
            matched_lower = lower
            matched_upper = upper
            break

    if matched_lower is None:
        raise ValueError("iota_axis is outside all allowed rational intervals.")

    return matched_lower, matched_upper
#==========================
#===================================================================================================================================================











#============== SURFACE SOURCE HELPERS =============================================================================================================
#==========================
def _load_pickle_file(
        filepath,
    ):
    path = Path(filepath).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
#==========================




#==========================
def _select_surface_points(
        dataset,
        selection_method,
    ):
    """
    Select a list of saved surface descriptions from the neural dataset.
    """
    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Dataset must be a non-empty list of data points.")

    if selection_method == "all_nested":
        candidates = [
            point for point in dataset
            if bool(point.get("labels", {}).get("is_nested", False))
        ]
        if len(candidates) == 0:
            raise ValueError("No nested surfaces were found in the source dataset.")
        return candidates

    if selection_method == "all_build_ok":
        candidates = [
            point for point in dataset
            if bool(point.get("labels", {}).get("build_ok", False))
        ]
        if len(candidates) == 0:
            raise ValueError("No build_ok surfaces were found in the source dataset.")
        return candidates

    raise ValueError(
        f"Unsupported selection_method='{selection_method}'. "
        f"Use 'all_nested' or 'all_build_ok'."
    )
#==========================




#==========================
def _surface_from_features(
        features,
        NFP,
    ):
    """
    Reconstruct FourierRZToroidalSurface from one saved dataset feature block.
    """
    required_keys = [
        "R_lmn",
        "Z_lmn",
        "modes_R",
        "modes_Z",
    ]

    missing = [key for key in required_keys if key not in features]
    if missing:
        raise KeyError(
            f"Saved feature block is missing required keys: {missing}"
        )

    return FourierRZToroidalSurface(
        R_lmn = features["R_lmn"],
        modes_R = features["modes_R"],
        Z_lmn = features["Z_lmn"],
        modes_Z = features["modes_Z"],
        NFP = NFP,
    )
#==========================




#==========================
def load_surface_pool(
        surface_source_config,
        NFP,
        N_eq,
    ):
    """
    Load a shuffled pool of saved surfaces and reconstruct them.
    """
    dataset = _load_pickle_file(surface_source_config["dataset_path"])

    selection_method = surface_source_config.get("selection_method", "all_nested")
    shuffle = surface_source_config.get("shuffle", True)
    shuffle_seed = surface_source_config.get("shuffle_seed", None)

    selected_points = _select_surface_points(
        dataset = dataset,
        selection_method = selection_method,
    )

    if shuffle:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(selected_points)

    if N_eq is not None:
        selected_points = selected_points[:N_eq]

    surface_pool = []
    for selected_point in selected_points:
        features = selected_point["features"]
        surface_init = _surface_from_features(
            features = features,
            NFP = NFP,
        )
        surface_pool.append((surface_init, selected_point))

    return surface_pool
#==========================




#==========================
def build_eq_configs(
        eq_input_config,
    ):
    """
    Convert declarative config values into a list of runtime eq_config dicts.
    """
    NFP = eq_input_config["NFP"]
    N_eq = eq_input_config["N_eq"]

    surface_pool = load_surface_pool(
        surface_source_config = eq_input_config["surface_source_config"],
        NFP = NFP,
        N_eq = N_eq,
    )

    eq_configs = []

    for idx, (surface_init, selected_point) in enumerate(surface_pool):
        eq_config = {
            "NFP":           NFP,
            "surface_init":  surface_init,
            "pressure_init": eq_input_config["pressure_init"],
            "iota_init":     eq_input_config["iota_init"],
            "eq_resolution": eq_input_config["eq_resolution"],
        }

        eq_configs.append(
            {
                "run_index": idx,
                "eq_config": eq_config,
                "selected_point": selected_point,
            }
        )

    return eq_configs
#=====================
#===================================================================================================================================================











#============== DRIVER / WORKER HELPERS ============================================================================================================
#========================================
def _extract_last_equilibrium(
        eq_like,
    ):
    """
    Return the final Equilibrium from a DESC continuation/load result.

    This handles ordinary Equilibrium objects, list/tuple returns, and
    EquilibriaFamily-like objects returned by automatic continuation.

    Returns:
        eq_final,
        num_steps
    """
    if eq_like is None:
        return None, 0

    if eq_like.__class__.__name__ == "Equilibrium":
        return eq_like, 1

    try:
        num_steps = len(eq_like)
    except Exception:
        return eq_like, 1

    if num_steps == 0:
        return None, 0

    try:
        return eq_like[-1], num_steps
    except Exception:
        pass

    try:
        return list(eq_like)[-1], num_steps
    except Exception:
        return eq_like, 1
#========================================





#========================================
def _load_eq_from_file(
        eq_path,
    ):
    """
    Load a saved DESC equilibrium from disk and collapse any family/list
    return to the final equilibrium only.
    """
    eq_loaded = desc_load(eq_path)
    eq, num_steps = _extract_last_equilibrium(eq_loaded)

    if eq is None:
        raise ValueError(f"No equilibrium objects found in file: {eq_path}")

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
    from .opt import run_optimization

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
            eq_opt, _ = _extract_last_equilibrium(eq_opt)
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











#============== DRIVER / STATUS HELPERS ============================================================================================================
#=========================
def sci_compact(x, sig = 2):
    s = f"{x:.{sig - 1}e}"
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
    if isinstance(objval, list):
        chosen = None
        for item in objval:
            if isinstance(item, dict) and all(
                key in item for key in ("f_min", "f_mean", "f_max")
            ):
                chosen = item
                break
        if chosen is None and len(objval) > 0:
            chosen = objval[0]
        objval = chosen

    if isinstance(objval, dict) and all(
        key in objval for key in ("f_min", "f_mean", "f_max")
    ):
        return (
            _to_float(objval["f_min"]),
            _to_float(objval["f_mean"]),
            _to_float(objval["f_max"]),
        )

    if isinstance(objval, dict):
        for value in objval.values():
            val = _to_float(value)
            if not np.isnan(val):
                return val, val, val
        return np.nan, np.nan, np.nan

    val = _to_float(objval)
    return val, val, val
#============================




#============================================
def _safe_extract_from_result(result, label):
    """
    Safely gets objective stats from result dict.
    Returns NaNs if label is absent.
    """
    if result is None:
        return np.nan, np.nan, np.nan

    objvals = result.get("Objective values", {})
    if label not in objvals:
        return np.nan, np.nan, np.nan
    return _extract_f_stats(objvals[label])
#============================================




#===================================
def _next_run_dir(continuation_dir):
    """
    Create next zero-padded run directory: 001, 002, ...
    """
    os.makedirs(continuation_dir, exist_ok = True)

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
#===================================




#=======================================
def _write_readme(out_dir, config_path):
    """
    Write full config.py contents into README.md.
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
#=======================================




#========================================
def _write_surface_source_summary(
        out_dir,
        selected_point,
    ):
    """
    Save the selected source point for reproducibility.
    """
    summary_path = os.path.join(out_dir, "surface_source_summary.txt")

    features = selected_point.get("features", {})
    labels = selected_point.get("labels", {})

    with open(summary_path, "w") as f:
        f.write("Selected surface source point\n\n")
        f.write(f"NFP = {features.get('NFP')}\n")
        f.write(f"resolution = {features.get('resolution')}\n")
        f.write(f"modes_R = {features.get('modes_R')}\n")
        f.write(f"modes_Z = {features.get('modes_Z')}\n")
        f.write(f"R_lmn = {features.get('R_lmn')}\n")
        f.write(f"Z_lmn = {features.get('Z_lmn')}\n")
        f.write(f"build_ok = {labels.get('build_ok')}\n")
        f.write(f"is_nested = {labels.get('is_nested')}\n")
#========================================




#========================================
def _extract_result_message(
        result,
    ):
    """
    Pull the most useful optimizer message from a result dict.
    """
    if result is None:
        return None

    if isinstance(result, dict):
        for key in [
            "message",
            "Message",
            "status",
            "Status",
        ]:
            if key in result and result[key] is not None:
                return str(result[key])

    return None
#========================================




#========================================
def _extract_result_iterations(
        result,
    ):
    """
    Pull final optimizer iteration count from a DESC result dict.
    """
    if result is None:
        return None

    if not isinstance(result, dict):
        return None

    candidate_keys = [
        "nit",
        "niter",
        "n_iter",
        "iterations",
        "Iterations",
        "Number of iterations",
        "Total iterations",
    ]

    for key in candidate_keys:
        if key in result:
            try:
                return int(result[key])
            except Exception:
                pass

    for key, value in result.items():
        if "iter" in str(key).lower():
            try:
                return int(value)
            except Exception:
                pass

    return None
#========================================




#========================================
def _has_bad_approximation_failure(
        message,
    ):
    """
    Detect the specific trust-region failure string.
    """
    if message is None:
        return False

    text = str(message).strip().lower()
    target = "a bad approximation caused failure to predict improvement"

    return target in text
#========================================




#========================================
def _has_automatic_continuation_failure(
        message,
    ):
    """
    Detect DESC automatic-continuation failure text in logs or messages.
    """
    if message is None:
        return False

    text = str(message).strip().lower()
    target = "warning: automatic continuation failed"

    return target in text
#========================================




#========================================
def _save_text_file(
        out_dir,
        filename,
        text,
    ):
    """
    Save arbitrary text into the run directory.
    """
    save_path = os.path.join(out_dir, filename)

    with open(save_path, "w") as f:
        if text is None:
            text = ""
        f.write(str(text))
        if len(str(text)) > 0 and not str(text).endswith("\n"):
            f.write("\n")
#========================================




#========================================
def _append_progress_log(
        out_dir,
        message,
    ):
    """
    Append a timestamped progress line to progress.log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(out_dir, "progress.log")

    with open(save_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
#========================================




#========================================
def _save_continuation_status_report(
        out_dir,
        continuation_status,
    ):
    """
    Save continuation status summary for later analysis.
    """
    save_path = os.path.join(out_dir, "continuation_status.txt")

    with open(save_path, "w") as f:
        f.write("Continuation status summary\n\n")
        f.write(f"equilibrium_built = {continuation_status.get('equilibrium_built')}\n")
        f.write(f"continuation_returned = {continuation_status.get('continuation_returned')}\n")
        f.write(f"exception_raised = {continuation_status.get('exception_raised')}\n")
        f.write(f"automatic_continuation_failure = {continuation_status.get('automatic_continuation_failure')}\n")
        f.write(f"num_steps_returned = {continuation_status.get('num_steps_returned')}\n")
        f.write(f"runtime_seconds = {continuation_status.get('runtime_seconds')}\n")
        f.write(f"failure_stage = {continuation_status.get('failure_stage')}\n")
        f.write(f"message = {continuation_status.get('message')}\n")
#========================================




#========================================
def _save_optimization_status_report(
        out_dir,
        optimization_status,
    ):
    """
    Save FLO / FNO optimizer status summary for later analysis.
    """
    save_path = os.path.join(out_dir, "optimization_status.txt")

    with open(save_path, "w") as f:
        f.write("Optimization status summary\n\n")

        for name, status in optimization_status.items():
            f.write(f"{name}\n")
            f.write(f"  equilibrium_returned = {status.get('equilibrium_returned')}\n")
            f.write(f"  exception_raised = {status.get('exception_raised')}\n")
            f.write(f"  bad_approximation_failure = {status.get('bad_approximation_failure')}\n")
            f.write(f"  automatic_continuation_failure = {status.get('automatic_continuation_failure')}\n")
            f.write(f"  final_iterations = {status.get('final_iterations')}\n")
            f.write(f"  runtime_seconds = {status.get('runtime_seconds')}\n")
            f.write(f"  plottable = {status.get('plottable')}\n")
            f.write(f"  plotted = {status.get('plotted')}\n")
            f.write(f"  failure_stage = {status.get('failure_stage')}\n")
            f.write(f"  message = {status.get('message')}\n\n")
#========================================




#========================================
def _make_skipped_run_status(
        formulation,
        message,
        failure_stage,
    ):
    """
    Construct a consistent skipped-run status dict.
    """
    return {
        "formulation": formulation,
        "optimizer": None,
        "equilibrium_returned": False,
        "exception_raised": False,
        "bad_approximation_failure": False,
        "automatic_continuation_failure": False,
        "plottable": False,
        "plotted": False,
        "failure_stage": failure_stage,
        "message": message,
        "final_iterations": None,
        "runtime_seconds": None,
    }
#========================================
#===================================================================================================================================================











#============== OPT HELPERS ========================================================================================================================
#==============================
def resolve_from_context(func):
    """
    Mark a callable as something that should be resolved
    from the runtime context dict.
    """
    func._resolve_from_context = True
    return func
#==============================




#====================
@resolve_from_context
def _eq(ctx):
    return ctx["eq_0"]
#====================




#==================
def _resolve_value(
        value,
        context,
    ):
    """
    Resolve a config value.

    Only call callables that were explicitly marked
    as context resolvers.
    """
    if callable(value) and getattr(value, "_resolve_from_context", False):
        return value(context)
    return value
#==================




#===================
def _resolve_kwargs(
        kwargs,
        context,
    ):
    """
    Resolve all values in a kwargs dict.
    """
    return {
        key: _resolve_value(value, context)
        for key, value in kwargs.items()
    }
#===================




#====================
def _validate_kwargs(
        entry,
        key,
        wrapper,
        kind,
    ):
    """
    Validate user-provided kwargs against wrapper signature.
    """
    sig = inspect.signature(wrapper)
    valid_params = set(sig.parameters.keys())

    invalid_keys = set(entry.keys()) - valid_params
    if invalid_keys:
        raise TypeError(
            f"Invalid kwarg(s) for {kind} '{key}' using wrapper "
            f"'{wrapper.__name__}': {sorted(invalid_keys)}. "
            f"Valid kwargs are: {sorted(valid_params)}"
        )
#====================




#================
def _parse_entry(
        toggle,
        key,
        kind,
    ):
    """
    Parse a config entry of the form:
        {
            "use": bool,
            "kwargs": dict,
        }

    Missing entries default to off.
    """
    entry = toggle.get(key, {"use": False, "kwargs": {}})

    if not isinstance(entry, dict):
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' must be a dict with keys "
            f"'use' and 'kwargs'. Got {type(entry).__name__}: {entry!r}"
        )

    allowed_keys = {"use", "kwargs"}
    invalid_keys = set(entry.keys()) - allowed_keys
    if invalid_keys:
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' has invalid top-level key(s): "
            f"{sorted(invalid_keys)}. Allowed keys are: {sorted(allowed_keys)}"
        )

    use = entry.get("use", False)
    kwargs = entry.get("kwargs", {})

    if not isinstance(use, bool):
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' has non-bool 'use': {use!r}"
        )

    if not isinstance(kwargs, dict):
        raise TypeError(
            f"{kind.capitalize()} toggle '{key}' has non-dict 'kwargs': {kwargs!r}"
        )

    return use, dict(kwargs)
#================




#=========================
def _append_term(
        term_list,
        wrapper,
        key,
        toggle,
        defaults,
        context,
        kind,
    ):
    """
    Append objective or constraint from a single config entry.

    User kwargs override defaults.
    """
    use, user_kwargs = _parse_entry(toggle, key, kind)
    if not use:
        return

    _validate_kwargs(user_kwargs, key, wrapper, kind)

    kwargs = {**defaults, **user_kwargs}
    kwargs = _resolve_kwargs(kwargs, context)
    _validate_kwargs(kwargs, key, wrapper, kind)

    term_list.append(wrapper(**kwargs))
#=========================




#=========================
def _append_terms(
        term_list,
        toggle,
        registry,
        context,
        kind,
    ):
    """
    Loop over a registry and append all enabled terms.
    """
    for key, spec in registry.items():
        _append_term(
            term_list = term_list,
            wrapper = spec["wrapper"],
            key = key,
            toggle = toggle,
            defaults = spec["defaults"],
            context = context,
            kind = kind,
        )
#=========================




#=========================
def _build_terms(
        eq_0,
        optimizer,
        opt_config,
        formulation,
        core_objective_registry,
        core_constraint_registry,
        FLO_objective_registry,
        FLO_constraint_registry,
        FNO_objective_registry,
        FNO_constraint_registry,
    ):
    """
    Build objective and constraint term lists from shared core toggles
    plus either FLO or FNO toggles.
    """
    context = {
        "eq_0": eq_0,
        "optimizer": optimizer,
    }

    opt_toggles_core = opt_config["opt_toggles_core"]

    if formulation == "FLO":
        opt_toggles_formulation = opt_config["opt_toggles_FLO"]
        formulation_objective_registry = FLO_objective_registry
        formulation_constraint_registry = FLO_constraint_registry

    elif formulation == "FNO":
        opt_toggles_formulation = opt_config["opt_toggles_FNO"]
        formulation_objective_registry = FNO_objective_registry
        formulation_constraint_registry = FNO_constraint_registry

    else:
        raise ValueError(
            f"Unknown formulation '{formulation}'. Expected 'FLO' or 'FNO'."
        )

    objectives_list = []
    constraints_list = []

    _append_terms(
        term_list = objectives_list,
        toggle = opt_toggles_core,
        registry = core_objective_registry,
        context = context,
        kind = "objective",
    )

    _append_terms(
        term_list = constraints_list,
        toggle = opt_toggles_core,
        registry = core_constraint_registry,
        context = context,
        kind = "constraint",
    )

    _append_terms(
        term_list = objectives_list,
        toggle = opt_toggles_formulation,
        registry = formulation_objective_registry,
        context = context,
        kind = "objective",
    )

    _append_terms(
        term_list = constraints_list,
        toggle = opt_toggles_formulation,
        registry = formulation_constraint_registry,
        context = context,
        kind = "constraint",
    )

    return objectives_list, constraints_list
#=========================




#=========================
def _merge_toggle_dicts(*toggle_dicts):
    """
    Merge toggle dicts left-to-right.
    Later dicts override earlier dicts for shared keys.
    """
    merged = {}
    for toggle_dict in toggle_dicts:
        if toggle_dict is None:
            continue
        for key, value in toggle_dict.items():
            merged[key] = deepcopy(value)
    return merged
#=========================
#===================================================================================================================================================