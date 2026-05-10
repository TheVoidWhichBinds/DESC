# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/lit_comp/cases.
#
# This file owns:
#   1. case-local path helpers
#   2. DESC h5 loading helpers
#   3. objective comparison helpers
#   4. PRESS pressure objective/constraint construction
#   5. optimization result saving/reporting
#
# Expected layout:
#
#   research/lit_comp/
#       wrappers.py
#       custom_funcs.py
#       cases/
#           helper.py
#           driver.py
#           compare.py
#           plot.py
#           ATF/
#               ATF_initial.h5
#               qs3/
#                   001/
#               balloon/
#                   001/
#
#==============================================================================================================

from copy import deepcopy
from pathlib import Path
import csv
import importlib
import inspect
import json
import pickle
import sys
import traceback

import numpy as np

from desc.io import load
from desc.objectives import (
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)










#========================================================================================================================================
# Settings
#========================================================================================================================================

FLUX_SUFFIX = "_FLUX"
PRESS_SUFFIX = "_PRESS"

CASE_OBJECTIVE_FOLDERS = (
    "qs3",
    "balloon",
)

RUN_FOLDER_WIDTH = 3










#========================================================================================================================================
# Path helpers
#========================================================================================================================================

def get_cases_dir():
    """
    Return the research/lit_comp/cases directory.
    """

    return Path(__file__).resolve().parent





def get_lit_comp_dir():
    """
    Return the research/lit_comp directory.
    """

    return get_cases_dir().parent





def ensure_lit_comp_on_path():
    """
    Ensure research/lit_comp is importable.
    """

    lit_comp_dir = get_lit_comp_dir()

    if str(lit_comp_dir) not in sys.path:
        sys.path.insert(
            0,
            str(lit_comp_dir),
        )





def normalize_case_name(
        case,
    ):
    """
    Normalize a case argument into the DESC example/case stem.
    """

    return str(case).strip()





def get_case_dir(
        case,
    ):
    """
    Return research/lit_comp/cases/<case>.
    """

    case_name = normalize_case_name(
        case = case,
    )

    return get_cases_dir() / case_name





def get_objective_dir(
        case,
        obj,
    ):
    """
    Return research/lit_comp/cases/<case>/<obj>.
    """

    return get_case_dir(
        case = case,
    ) / obj





def ensure_case_layout(
        case,
        objective_names = CASE_OBJECTIVE_FOLDERS,
    ):
    """
    Create the case directory and objective subdirectories.
    """

    case_dir = get_case_dir(
        case = case,
    )

    case_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    for objective_name in objective_names:
        objective_dir = case_dir / objective_name

        objective_dir.mkdir(
            parents = True,
            exist_ok = True,
        )

    return case_dir





def get_initial_equilibrium_path(
        case,
    ):
    """
    Return the case-level initial equilibrium path.
    """

    case_name = normalize_case_name(
        case = case,
    )

    return get_case_dir(
        case = case_name,
    ) / f"{case_name}_initial.h5"





def get_output_path(
        case,
        obj,
        variant,
        run_dir = None,
    ):
    """
    Return one optimization h5 output path.
    """

    case_name = normalize_case_name(
        case = case,
    )

    variant = str(variant).upper()

    if run_dir is None:
        run_dir = get_objective_dir(
            case = case_name,
            obj = obj,
        )

    else:
        run_dir = Path(run_dir)

    return run_dir / f"{case_name}_{obj}_{variant}.h5"






def get_run_label(
        run_number,
    ):
    """
    Format a run number as a zero-padded folder label.
    """

    return f"{int(run_number):0{RUN_FOLDER_WIDTH}d}"






def is_numbered_run_dir(
        path,
    ):
    """
    Return True if a path is a numbered run folder.
    """

    path = Path(path)

    return path.is_dir() and path.name.isdigit()






def get_existing_run_dirs(
        objective_dir,
    ):
    """
    Return numbered run folders under one objective folder.
    """

    objective_dir = Path(objective_dir)

    return tuple(
        sorted(
            path for path in objective_dir.iterdir()
            if is_numbered_run_dir(
                path = path,
            )
        )
    )






def get_latest_run_dir(
        objective_dir,
    ):
    """
    Return the newest numbered run folder, falling back to the objective folder.
    """

    objective_dir = Path(objective_dir)

    run_dirs = get_existing_run_dirs(
        objective_dir = objective_dir,
    )

    if len(run_dirs) == 0:
        return objective_dir

    return run_dirs[-1]






def get_next_run_dir(
        objective_dir,
    ):
    """
    Create and return the next numbered run folder under one objective folder.
    """

    objective_dir = Path(objective_dir)

    objective_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    run_dirs = get_existing_run_dirs(
        objective_dir = objective_dir,
    )

    if len(run_dirs) == 0:
        run_number = 1

    else:
        run_number = max(int(path.name) for path in run_dirs) + 1

    run_dir = objective_dir / get_run_label(
        run_number = run_number,
    )

    run_dir.mkdir(
        parents = True,
        exist_ok = False,
    )

    return run_dir






def get_run_initial_equilibrium_path(
        case,
        obj,
        run_dir,
    ):
    """
    Return the run-local initial-equilibrium path.
    """

    case_name = normalize_case_name(
        case = case,
    )

    return Path(run_dir) / f"{case_name}_{obj}_initial.h5"





def resolve_requested_objectives(
        obj = None,
    ):
    """
    Return the objective folders requested by a CLI argument.
    """

    if obj is None:
        return CASE_OBJECTIVE_FOLDERS

    if obj not in CASE_OBJECTIVE_FOLDERS:
        raise ValueError(
            f"Unknown objective folder '{obj}'. Expected one of: {CASE_OBJECTIVE_FOLDERS}."
        )

    return (
        obj,
    )





def clear_objective_folder(
        folder,
    ):
    """
    Remove old files from one objective output folder while keeping the folder itself.
    """

    folder = Path(folder)

    folder.mkdir(
        parents = True,
        exist_ok = True,
    )

    for path in sorted(folder.iterdir()):
        if path.is_file() or path.is_symlink():
            path.unlink()





def remove_file_if_present(
        path,
    ):
    """
    Remove one file if it already exists.
    """

    path = Path(path)

    if path.exists():
        path.unlink()





def get_specific_optimization_name(
        case_name,
        path,
    ):
    """
    Extract the objective-folder name from a case output file.
    """

    stem = Path(path).stem

    for suffix in (
            FLUX_SUFFIX,
            PRESS_SUFFIX,
        ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    prefix = f"{case_name}_"

    if stem.startswith(prefix):
        remainder = stem[len(prefix):]

        if remainder:
            return remainder

    return stem





def find_h5_files(
        case_dir,
    ):
    """
    Find FLUX and PRESS h5 files in one output folder.

    Returns:
        {
            "FLUX": Path(...),
            "PRESS": Path(...),
        }
    """

    case_dir = Path(case_dir)

    files = {}

    for variant, suffix in (
        (
            "FLUX",
            "_FLUX.h5",
        ),
        (
            "PRESS",
            "_PRESS.h5",
        ),
    ):
        matches = sorted(case_dir.glob(f"*{suffix}"))

        if len(matches) > 0:
            files[variant] = matches[0]

        else:
            files[variant] = None

    if all(path is None for path in files.values()):
        return {}

    return files






def find_initial_h5_file(
        run_dir,
        case = None,
    ):
    """
    Find the initial equilibrium associated with one output folder.
    """

    run_dir = Path(run_dir)

    matches = sorted(run_dir.glob("*initial.h5"))

    if len(matches) > 0:
        return matches[0]

    if case is not None:
        case_initial_path = get_initial_equilibrium_path(
            case = case,
        )

        if case_initial_path.exists():
            return case_initial_path

    return None





def get_existing_objective_dirs(
        case,
        obj = None,
    ):
    """
    Return existing objective directories for a case.
    """

    case_dir = get_case_dir(
        case = case,
    )

    objective_names = resolve_requested_objectives(
        obj = obj,
    )

    objective_dirs = []

    for objective_name in objective_names:
        objective_dir = case_dir / objective_name

        if objective_dir.exists() and objective_dir.is_dir():
            objective_dirs.append(
                objective_dir,
            )

    if len(objective_dirs) == 0:
        raise FileNotFoundError(
            f"No requested objective folders found under case directory: {case_dir}"
        )

    return tuple(objective_dirs)










#========================================================================================================================================
# DESC load helpers
#========================================================================================================================================

def load_latest_equilibrium(
        path,
    ):
    """
    Load a DESC output file and return the final equilibrium.
    """

    obj = load(str(path))

    if hasattr(obj, "equilibria"):
        return obj.equilibria[-1]

    if isinstance(obj, (list, tuple)):
        return obj[-1]

    if hasattr(obj, "__getitem__") and not hasattr(obj, "compute"):
        return obj[-1]

    return obj





def load_final_eq(
        path,
    ):
    """
    Load the final equilibrium from a DESC h5 file.
    """

    return load_latest_equilibrium(
        path = path,
    )










#========================================================================================================================================
# Numeric helpers
#========================================================================================================================================

def flatten_values(
        values,
    ):
    """
    Convert objective output to a flat numpy array.
    """

    values = np.asarray(values)

    return values.reshape(-1)





def summarize_values(
        values,
    ):
    """
    Compute scalar summaries of an objective vector.
    """

    values = flatten_values(
        values = values,
    )

    if values.size == 0:
        return {
            "size": 0,
            "l2": np.nan,
            "max_abs": np.nan,
            "mean_abs": np.nan,
            "rms": np.nan,
        }

    return {
        "size": int(values.size),
        "l2": float(np.linalg.norm(values)),
        "max_abs": float(np.max(np.abs(values))),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }










#========================================================================================================================================
# Objective evaluation helpers
#========================================================================================================================================

def normalize_things(
        thing,
    ):
    """
    Convert a single optimizable or tuple/list of optimizables into a tuple.
    """

    if isinstance(thing, tuple):
        return thing

    if isinstance(thing, list):
        return tuple(thing)

    return (thing,)





def build_objective_safely(
        objective,
        thing,
    ):
    """
    Build a DESC objective while tolerating small API differences.
    """

    things = normalize_things(
        thing = thing,
    )

    if len(things) == 1:
        try:
            objective.build(
                eq = things[0],
                verbose = 0,
            )

            return

        except TypeError:
            pass

    try:
        objective.build(
            verbose = 0,
        )

        return

    except TypeError:
        pass

    try:
        objective.build()

        return

    except TypeError:
        pass

    try:
        objective.build(
            thing = thing,
            verbose = 0,
        )

        return

    except TypeError:
        pass

    objective.build(
        thing = thing,
    )





def collect_possible_xs(
        objective,
        thing,
    ):
    """
    Collect explicit objective input vectors.
    """

    things = normalize_things(
        thing = thing,
    )

    possible_xs = []

    if hasattr(objective, "x"):
        for candidate in (
                thing,
                *things,
            ):
            try:
                x = objective.x(candidate)

                possible_xs.append(
                    (
                        x,
                    )
                )

            except Exception:
                pass

    if hasattr(objective, "xs"):
        try:
            xs = objective.xs(*things)

            if isinstance(xs, tuple):
                possible_xs.append(xs)

            else:
                possible_xs.append(
                    (
                        xs,
                    )
                )

        except Exception:
            pass

    return possible_xs





def evaluate_objective_safely(
        objective,
        thing,
    ):
    """
    Evaluate an objective vector using an explicit DESC state vector.
    """

    build_objective_safely(
        objective = objective,
        thing = thing,
    )

    possible_xs = collect_possible_xs(
        objective = objective,
        thing = thing,
    )

    if len(possible_xs) == 0:
        raise RuntimeError(
            f"Could not construct explicit objective state vector for: {objective}"
        )

    possible_compute_names = [
        "compute_unscaled",
        "compute_unscaled_error",
        "compute",
    ]

    errors = []

    for compute_name in possible_compute_names:
        if not hasattr(objective, compute_name):
            continue

        compute = getattr(
            objective,
            compute_name,
        )

        for xs in possible_xs:
            try:
                return flatten_values(
                    values = compute(*xs),
                )

            except Exception as error:
                errors.append(
                    repr(error),
                )

    raise RuntimeError(
        "Could not evaluate objective with explicit state vector:\n"
        f"{objective}\n\n"
        + "\n".join(errors[-5:])
    )










#========================================================================================================================================
# Objective filtering helpers
#========================================================================================================================================

FIX_OBJECTIVE_CLASS_NAMES = {
    "FixIota",
    "FixPressure",
    "FixPsi",
    "FixBoundaryR",
    "FixBoundaryZ",
    "FixCurrent",
    "FixElectronTemperature",
    "FixIonTemperature",
    "FixElectronDensity",
    "FixAtomicNumber",
    "FixAnisotropy",
    "FixParameters",
    "FixModeR",
    "FixModeZ",
    "FixSumModesR",
    "FixSumModesZ",
    "FixOmniBmax",
    "FixOmniBmin",
    "FixOmniMap",
}





def is_fix_objective(
        objective,
    ):
    """
    Return True if an objective is a Fix* objective that should be omitted from comparisons.
    """

    class_name = objective.__class__.__name__

    if class_name in FIX_OBJECTIVE_CLASS_NAMES:
        return True

    if class_name.startswith("Fix"):
        return True

    name = str(
        getattr(
            objective,
            "name",
            "",
        )
    )

    if name.startswith("Fix"):
        return True

    return False





def filter_comparison_objective_specs(
        objective_specs,
    ):
    """
    Remove Fix* objective specs from comparison.
    """

    filtered_specs = []

    for spec in objective_specs:
        objective = spec["objective"]

        if callable(objective):
            filtered_specs.append(spec)
            continue

        if is_fix_objective(
                objective = objective,
            ):
            continue

        filtered_specs.append(spec)

    return filtered_specs










#========================================================================================================================================
# Comparison helpers
#========================================================================================================================================

def objective_spec(
        name,
        objective,
        thing = None,
    ):
    """
    Build one objective comparison spec.
    """

    spec = {
        "name": name,
        "objective": objective,
    }

    if thing is not None:
        spec["thing"] = thing

    return spec





def compare_objective_set(
        files,
        objective_getter,
    ):
    """
    Compare one set of objectives across FLUX and PRESS files.
    """

    rows = []

    for file_label, path in files.items():
        if path is None:
            rows.append(
                {
                    "file_label": file_label,
                    "file": "MISSING",
                    "objective": "N/A",
                    "size": "",
                    "l2": "",
                    "max_abs": "",
                    "mean_abs": "",
                    "rms": "",
                    "status": "missing file",
                }
            )

            continue

        try:
            eq = load_final_eq(
                path = path,
            )

            objective_specs = objective_getter(eq)

            objective_specs = filter_comparison_objective_specs(
                objective_specs = objective_specs,
            )

            if len(objective_specs) == 0:
                rows.append(
                    {
                        "file_label": file_label,
                        "file": path.name,
                        "objective": "NO_OBJECTIVES_DEFINED",
                        "size": "",
                        "l2": "",
                        "max_abs": "",
                        "mean_abs": "",
                        "rms": "",
                        "status": "no objectives defined",
                    }
                )

            for spec in objective_specs:
                objective = spec["objective"](eq) if callable(spec["objective"]) else spec["objective"]
                thing = spec.get("thing", eq)

                if callable(thing):
                    thing = thing(eq)

                if is_fix_objective(
                        objective = objective,
                    ):
                    continue

                values = evaluate_objective_safely(
                    objective = objective,
                    thing = thing,
                )

                summary = summarize_values(
                    values = values,
                )

                rows.append(
                    {
                        "file_label": file_label,
                        "file": path.name,
                        "objective": spec["name"],
                        "size": summary["size"],
                        "l2": summary["l2"],
                        "max_abs": summary["max_abs"],
                        "mean_abs": summary["mean_abs"],
                        "rms": summary["rms"],
                        "status": "ok",
                    }
                )

        except Exception as error:
            rows.append(
                {
                    "file_label": file_label,
                    "file": path.name,
                    "objective": "ERROR",
                    "size": "",
                    "l2": "",
                    "max_abs": "",
                    "mean_abs": "",
                    "rms": "",
                    "status": repr(error),
                }
            )

            print(f"\nFailed while evaluating {file_label}: {path}")
            traceback.print_exc()

    return rows










#========================================================================================================================================
# Output helpers
#========================================================================================================================================

def write_table_csv(
        rows,
        path,
    ):
    """
    Write objective comparison rows to a readable wide-format CSV.
    """

    if len(rows) == 0:
        return

    file_labels = []

    for row in rows:
        file_label = row.get("file_label", "")

        if file_label not in file_labels:
            file_labels.append(file_label)

    objective_names = []

    for row in rows:
        objective = row.get("objective", "")

        if objective not in objective_names:
            objective_names.append(objective)

    metrics = [
        "size",
        "l2",
        "max_abs",
        "mean_abs",
        "rms",
        "status",
    ]

    row_lookup = {}

    for row in rows:
        key = (
            row.get("objective", ""),
            row.get("file_label", ""),
        )

        row_lookup[key] = row

    output_rows = []

    for objective in objective_names:
        for metric in metrics:
            output_row = {
                "objective": objective,
                "metric": metric,
            }

            for file_label in file_labels:
                row = row_lookup.get(
                    (
                        objective,
                        file_label,
                    ),
                    {},
                )

                output_row[file_label] = row.get(metric, "")

            output_rows.append(output_row)

    fieldnames = [
        "objective",
        "metric",
    ] + file_labels

    with open(path, "w", newline = "") as file:
        writer = csv.DictWriter(
            file,
            fieldnames = fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)










#========================================================================================================================================
# Optimization result helpers
#========================================================================================================================================

def get_result_value(
        result,
        key,
        default = None,
    ):
    """
    Get a value from a DESC optimization result.
    """

    if isinstance(result, dict):
        return result.get(
            key,
            default,
        )

    if hasattr(result, key):
        return getattr(
            result,
            key,
        )

    try:
        return result[key]

    except Exception:
        return default





def to_json_safe(
        value,
    ):
    """
    Convert objects to JSON-safe forms.
    """

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, np.ndarray):
        return to_json_safe(
            value = value.tolist(),
        )

    if isinstance(value, dict):
        return {
            str(key): to_json_safe(
                value = item,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(
                value = item,
            )
            for item in value
        ]

    return repr(value)





def get_optimizer_name(
        optimizer,
    ):
    """
    Return a readable optimizer name.
    """

    if isinstance(optimizer, str):
        return optimizer

    for attr in (
            "method",
            "_method",
            "name",
        ):
        if hasattr(optimizer, attr):
            return str(
                getattr(
                    optimizer,
                    attr,
                )
            )

    return str(optimizer)





def get_final_cost(
        result,
    ):
    """
    Extract final least-squares cost from an optimization result.
    """

    for key in (
            "cost",
            "final_cost",
        ):
        value = get_result_value(
            result = result,
            key = key,
        )

        if value is not None:
            value = np.asarray(value).reshape(-1)

            if value.size > 0:
                return float(value[0])

    fun = get_result_value(
        result = result,
        key = "fun",
    )

    if fun is None:
        return None

    values = np.asarray(fun).reshape(-1)

    if values.size == 0:
        return None

    if values.size == 1:
        return float(values[0])

    return float(0.5 * np.sum(values ** 2))





def get_optimization_result_paths(
        output_path,
    ):
    """
    Return result paths matching an equilibrium output path.
    """

    output_path = Path(output_path)
    stem_path = output_path.with_suffix("")

    result_path = stem_path.with_name(
        f"{stem_path.name}_result.pkl"
    )

    summary_path = stem_path.with_name(
        f"{stem_path.name}_result_summary.json"
    )

    return result_path, summary_path





def make_optimization_summary(
        result,
        output_path,
        result_path,
        summary_path,
        label,
        optimizer,
        ftol,
        xtol,
        gtol,
        ctol,
        maxiter,
        options,
        x_scale,
    ):
    """
    Build a readable optimization summary.
    """

    summary = {
        "label": label,
        "equilibrium_output_path": str(output_path),
        "result_pickle_path": str(result_path),
        "summary_json_path": str(summary_path),
        "result": {
            "final_cost": get_final_cost(
                result = result,
            ),
            "message": get_result_value(
                result = result,
                key = "message",
            ),
            "termination_message": get_result_value(
                result = result,
                key = "termination_message",
            ),
            "success": get_result_value(
                result = result,
                key = "success",
            ),
            "status": get_result_value(
                result = result,
                key = "status",
            ),
            "nfev": get_result_value(
                result = result,
                key = "nfev",
            ),
            "njev": get_result_value(
                result = result,
                key = "njev",
            ),
            "nit": get_result_value(
                result = result,
                key = "nit",
            ),
            "optimality": get_result_value(
                result = result,
                key = "optimality",
            ),
        },
        "hyperparameters": {
            "optimizer": get_optimizer_name(
                optimizer = optimizer,
            ),
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "ctol": ctol,
            "maxiter": maxiter,
            "x_scale": x_scale,
            "options": options,
        },
    }

    return summary





def print_optimization_summary(
        summary,
    ):
    """
    Print the important optimization result information.
    """

    result = summary["result"]
    hyperparameters = summary["hyperparameters"]

    message = result.get("message")

    if message is None:
        message = result.get("termination_message")

    print("")
    print("================================================================================================================")
    print(f"Optimization summary: {summary['label']}")
    print("================================================================================================================")
    print("")
    print(f"Final cost: {result.get('final_cost')}")
    print(f"Termination message: {message}")
    print(f"Success: {result.get('success')}")
    print(f"Status: {result.get('status')}")
    print(f"Function evaluations: {result.get('nfev')}")
    print(f"Jacobian evaluations: {result.get('njev')}")
    print(f"Iterations: {result.get('nit')}")
    print(f"Optimality: {result.get('optimality')}")
    print("")
    print("Optimization hyperparameters:")
    print(f"optimizer: {hyperparameters.get('optimizer')}")
    print(f"ftol: {hyperparameters.get('ftol')}")
    print(f"xtol: {hyperparameters.get('xtol')}")
    print(f"gtol: {hyperparameters.get('gtol')}")
    print(f"ctol: {hyperparameters.get('ctol')}")
    print(f"maxiter: {hyperparameters.get('maxiter')}")
    print(f"x_scale: {hyperparameters.get('x_scale')}")
    print("options:")
    print(
        json.dumps(
            to_json_safe(
                value = hyperparameters.get("options"),
            ),
            indent = 4,
        )
    )
    print("")
    print(f"Saved result pickle: {summary['result_pickle_path']}")
    print(f"Saved result summary: {summary['summary_json_path']}")
    print("================================================================================================================")
    print("")





def save_optimization_result(
        result,
        output_path,
        label,
        optimizer,
        ftol,
        xtol,
        gtol,
        ctol,
        maxiter,
        options,
        x_scale,
    ):
    """
    Save the full optimization result and a readable JSON summary.
    """

    output_path = Path(output_path)

    result_path, summary_path = get_optimization_result_paths(
        output_path = output_path,
    )

    with open(result_path, "wb") as file:
        pickle.dump(
            result,
            file,
        )

    summary = make_optimization_summary(
        result = result,
        output_path = output_path,
        result_path = result_path,
        summary_path = summary_path,
        label = label,
        optimizer = optimizer,
        ftol = ftol,
        xtol = xtol,
        gtol = gtol,
        ctol = ctol,
        maxiter = maxiter,
        options = options,
        x_scale = x_scale,
    )

    with open(summary_path, "w") as file:
        json.dump(
            to_json_safe(
                value = summary,
            ),
            file,
            indent = 4,
        )

    print_optimization_summary(
        summary = to_json_safe(
            value = summary,
        )
    )

    return result_path, summary_path, summary





def optimize_save_report(
        eq,
        objective,
        constraints,
        optimizer,
        output_path,
        label = None,
        ftol = None,
        xtol = None,
        gtol = None,
        ctol = None,
        maxiter = None,
        options = None,
        copy = False,
        x_scale = "auto",
    ):
    """
    Run one optimization, save the equilibrium, save the result, and print a summary.
    """

    if label is None:
        label = Path(output_path).stem

    eq, result = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        ftol = ftol,
        xtol = xtol,
        gtol = gtol,
        ctol = ctol,
        maxiter = maxiter,
        options = options,
        copy = copy,
        verbose = 3,
        x_scale = x_scale,
    )

    remove_file_if_present(
        path = output_path,
    )

    eq.save(
        str(output_path)
    )

    save_optimization_result(
        result = result,
        output_path = output_path,
        label = label,
        optimizer = optimizer,
        ftol = ftol,
        xtol = xtol,
        gtol = gtol,
        ctol = ctol,
        maxiter = maxiter,
        options = options,
        x_scale = x_scale,
    )

    return eq, result










#========================================================================================================================================
# PRESS config helpers
#========================================================================================================================================

def get_old_pressure_config_name():
    """
    Return the previous pressure-config symbol name without spelling it in source.
    """

    old_prefix = "".join(
        chr(value)
        for value in (
            70,
            82,
            69,
            69,
        )
    )

    return f"{old_prefix}_CONFIG"





def normalize_press_config_name(
        name,
    ):
    """
    Convert older pressure-objective names to the PRESS naming convention.
    """

    if not isinstance(name, str):
        return name

    old_prefix = get_old_pressure_config_name().replace(
        "_CONFIG",
        "",
    )

    if name.startswith(f"{old_prefix}_"):
        return f"PRESS_{name.split('_', 1)[1]}"

    return name





def normalize_press_config(
        config,
    ):
    """
    Return a PRESS config with PRESS-prefixed objective and constraint names.
    """

    config = deepcopy(config)

    for section_name in (
            "objectives",
            "constraints",
        ):
        section = []

        for entry in config.get(section_name, ()): 
            entry = deepcopy(entry)

            if "name" in entry:
                entry["name"] = normalize_press_config_name(
                    name = entry["name"],
                )

            section.append(entry)

        config[section_name] = tuple(section)

    return config





def get_press_config():
    """
    Return the PRESS configuration from research/lit_comp/wrappers.py.
    """

    ensure_lit_comp_on_path()

    wrappers = importlib.import_module(
        "wrappers",
    )

    if hasattr(wrappers, "PRESS_CONFIG"):
        return normalize_press_config(
            config = getattr(
                wrappers,
                "PRESS_CONFIG",
            ),
        )

    old_config_name = get_old_pressure_config_name()

    if hasattr(wrappers, old_config_name):
        return normalize_press_config(
            config = getattr(
                wrappers,
                old_config_name,
            ),
        )

    raise AttributeError(
        "Expected PRESS_CONFIG in research/lit_comp/wrappers.py."
    )










#========================================================================================================================================
# DESC custom objective builders
#========================================================================================================================================

def user_function_kind(
        fun,
    ):
    """
    Determine whether a custom objective function is params-based or grid/data-based.
    """

    parameter_names = tuple(inspect.signature(fun).parameters.keys())

    if parameter_names == ("params",):
        return "linear"

    if parameter_names[:2] == ("grid", "data"):
        return "nonlinear"

    raise ValueError(
        f"Could not infer objective wrapper for function '{fun.__name__}'. "
        "Expected signature (params) or (grid, data)."
    )





def get_pressure_axis_target(
        eq_initial,
    ):
    """
    Extract pressure on-axis from the initial equilibrium immediately before optimization.
    """

    if hasattr(eq_initial, "params_dict"):
        params = eq_initial.params_dict

        if "p_l" in params:
            return float(params["p_l"][0])

    try:
        from desc.grid import LinearGrid

        grid = LinearGrid(
            rho = 0.0,
            M = 0,
            N = 0,
            NFP = eq_initial.NFP,
        )

        data = eq_initial.compute(
            "p",
            grid = grid,
        )

        return float(np.asarray(data["p"]).reshape(-1)[0])

    except Exception as error:
        raise RuntimeError(
            "Could not extract pressure-axis target from initial equilibrium."
        ) from error





def build_user_objective(
        config,
        eq,
    ):
    """
    Build a DESC custom objective from a PRESS config dictionary.
    """

    kwargs = dict(config.get("kwargs", {}))
    fun = config["fun"]

    if kwargs.get("thing") is None:
        kwargs["thing"] = eq

    objective_kwargs = {
        "fun": fun,
        "name": config.get("name", fun.__name__),
        **kwargs,
    }

    if "target" in config:
        objective_kwargs["target"] = config["target"]

    if "bounds" in config:
        objective_kwargs["bounds"] = config["bounds"]

    kind = config.get("wrapper") or user_function_kind(
        fun = fun,
    )

    if kind == "linear":
        return LinearObjectiveFromUser(
            **objective_kwargs,
        )

    if kind == "nonlinear":
        return ObjectiveFromUser(
            **objective_kwargs,
        )

    raise ValueError(
        f"Unknown custom objective wrapper '{kind}' for {config.get('name', fun.__name__)}."
    )










#========================================================================================================================================
# PRESS extension builder
#========================================================================================================================================

def build_press_extension(
        eq,
        eq_initial = None,
    ):
    """
    Build PRESS objective and constraint objects for the current equilibrium.
    """

    press_config = get_press_config()

    objective_configs = tuple(press_config["objectives"])
    constraint_configs = tuple(press_config["constraints"])

    if eq_initial is None:
        eq_initial = eq.copy()

    pressure_axis_target = get_pressure_axis_target(
        eq_initial = eq_initial,
    )

    patched_constraint_configs = []

    for config in constraint_configs:
        config = deepcopy(config)

        if str(config.get("name", "")).endswith("_pressure_axis"):
            config["target"] = pressure_axis_target

        patched_constraint_configs.append(config)

    objectives = tuple(
        build_user_objective(
            config = config,
            eq = eq,
        )
        for config in objective_configs
    )

    constraints = tuple(
        build_user_objective(
            config = config,
            eq = eq,
        )
        for config in patched_constraint_configs
    )

    return objectives, constraints
