# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/lit_comp/tutorials.
#
# This file owns:
#   1. tutorial-local path helpers
#   2. DESC h5 loading helpers
#   3. tutorial-objective comparison helpers
#   4. FREE config/objective construction
#   5. direct tutorial runners
#
# Expected layout:
#
#   research/lit_comp/
#       wrappers.py
#       custom_funcs.py
#       papers/
#           helper.py
#           driver.py
#           compare.py
#       tutorials/
#           helper.py
#           driver.py
#           compare.py
#           basic_qs/
#               basic_qs.py
#           adv_qs/
#               adv_qs.py
#           balloon/
#               balloon.py
#           neoclassical/
#               neoclassical.py
#
#==============================================================================================================

from copy import deepcopy
from pathlib import Path
import csv
import inspect
import os
import runpy
import sys
import traceback

import numpy as np

from desc.io import load
from desc.objectives import (
    ObjectiveFunction,
    ObjectiveFromUser,
    LinearObjectiveFromUser,
)










#========================================================================================================================================
# Settings
#========================================================================================================================================

FXD_SUFFIX = "_FXD"
FREE_SUFFIX = "_FREE"










#========================================================================================================================================
# Path helpers
#========================================================================================================================================

def get_tutorials_dir():
    """
    Return the research/lit_comp/tutorials directory.
    """

    return Path(__file__).resolve().parent





def get_lit_comp_dir():
    """
    Return the research/lit_comp directory.
    """

    return get_tutorials_dir().parent





def ensure_lit_comp_on_path():
    """
    Ensure research/lit_comp is importable.

    This allows tutorials/helper.py to import shared files from:

        research/lit_comp/wrappers.py
        research/lit_comp/custom_funcs.py

    even when running directly from:

        research/lit_comp/tutorials
    """

    lit_comp_dir = get_lit_comp_dir()

    if str(lit_comp_dir) not in sys.path:
        sys.path.insert(
            0,
            str(lit_comp_dir),
        )





def normalize_tutorial_name(
        name,
    ):
    """
    Normalize a passed tutorial filename/stem by removing .py, .h5, and known output suffixes.
    """

    path = Path(name)
    stem = path.stem

    for suffix in (
        FXD_SUFFIX,
        FREE_SUFFIX,
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem





def normalize_case_name(
        name,
    ):
    """
    Backward-compatible alias for normalize_tutorial_name.
    """

    return normalize_tutorial_name(
        name = name,
    )





def get_tutorial_dir(
        tutorial,
    ):
    """
    Return the tutorial directory.

    Expected layout:
        research/lit_comp/tutorials/<tutorial>/<tutorial>.py
    """

    tutorial_name = normalize_tutorial_name(
        name = tutorial,
    )

    return get_tutorials_dir() / tutorial_name





def get_case_dir(
        tutorial,
    ):
    """
    Backward-compatible alias for get_tutorial_dir.
    """

    return get_tutorial_dir(
        tutorial = tutorial,
    )





def get_source_file_from_tutorial(
        tutorial,
    ):
    """
    Resolve the tutorial source file.

    Expected layout:
        research/lit_comp/tutorials/<tutorial>/<tutorial>.py
    """

    tutorial_name = normalize_tutorial_name(
        name = tutorial,
    )

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial_name,
    )

    source_file = tutorial_dir / f"{tutorial_name}.py"

    if not tutorial_dir.exists():
        raise FileNotFoundError(
            f"Tutorial directory does not exist: {tutorial_dir}"
        )

    if not source_file.exists():
        raise FileNotFoundError(
            "Could not find tutorial source file:\n"
            f"{source_file}\n\n"
            "Expected layout:\n"
            f"research/lit_comp/tutorials/{tutorial_name}/{tutorial_name}.py"
        )

    return source_file, tutorial_name





def get_specific_optimization_name(
        tutorial_name,
        path,
    ):
    """
    Extract the specific optimization name from:

        {tutorial}_{specific}_{FXD/FREE}.h5

    Example:
        basic_qs_C_FXD.h5 -> C
        basic_qs_T_FREE.h5 -> T
    """

    stem = Path(path).stem

    for suffix in (
        FXD_SUFFIX,
        FREE_SUFFIX,
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    prefix = f"{tutorial_name}_"

    if stem.startswith(prefix):
        return stem[len(prefix):]

    if stem == tutorial_name:
        return "main"

    return stem





def find_h5_files(
        case_dir : Path,
    ):
    """
    Find all FXD and FREE h5 files in the tutorial folder.

    Returns:
        {
            "C": {
                "FXD": basic_qs_C_FXD.h5,
                "FREE": basic_qs_C_FREE.h5,
            },
            "T": {
                "FXD": basic_qs_T_FXD.h5,
                "FREE": basic_qs_T_FREE.h5,
            },
        }
    """

    case_dir = Path(case_dir)
    tutorial_name = case_dir.name

    files = {}

    for variant, suffix in (
        (
            "FXD",
            "_FXD.h5",
        ),
        (
            "FREE",
            "_FREE.h5",
        ),
    ):
        matches = sorted(case_dir.glob(f"*{suffix}"))

        if len(matches) == 0:
            matches = sorted(case_dir.rglob(f"*{suffix}"))

        for path in matches:
            optimization_name = get_specific_optimization_name(
                tutorial_name = tutorial_name,
                path = path,
            )

            if optimization_name not in files:
                files[optimization_name] = {
                    "FXD": None,
                    "FREE": None,
                }

            files[optimization_name][variant] = path

    return files





def find_tutorial_h5_files(
        tutorial,
    ):
    """
    Find FXD, and FREE h5 files for a tutorial.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    return find_h5_files(
        case_dir = tutorial_dir,
    )





def resolve_source_file(
        source_file,
    ):
    """
    Resolve a tutorial source file path.

    Supports:
        absolute/path/to/basic_qs.py

    or, from research/lit_comp/tutorials:
        basic_qs/basic_qs.py

    or:
        basic_qs
    """

    source_file = Path(source_file)

    candidates = []

    if source_file.is_absolute():
        candidates.append(source_file)

    else:
        if source_file.suffix == "":
            tutorial_name = normalize_tutorial_name(
                name = source_file,
            )

            candidates.append(get_tutorials_dir() / tutorial_name / f"{tutorial_name}.py")

        candidates.append(Path.cwd() / source_file)
        candidates.append(get_tutorials_dir() / source_file)

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find tutorial source file. Tried:\n"
        + "\n".join(str(candidate.resolve()) for candidate in candidates)
    )










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
        path : Path,
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

    This intentionally does not fall back to compute() with no arguments,
    because that can evaluate cached/default objective state instead of the
    loaded equilibrium.
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
    Return True if an objective is a Fix* objective that should be omitted from tutorial comparisons.
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

def compare_objective_set(
        files : dict,
        objective_getter,
    ):
    """
    Compare one set of objectives across FXD, and FREE files.
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
        rows : list,
        path,
    ):
    """
    Write objective comparison rows to a readable wide-format CSV.

    Output format:

        objective, metric, FXD, FREE
        ForceBalance, l2, ...
        ForceBalance, max_abs, ...
        ...
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
# FREE config helpers
#========================================================================================================================================

def get_free_config():
    """
    Return the FREE configuration from research/lit_comp/wrappers.py.
    """

    ensure_lit_comp_on_path()

    from wrappers import FREE_CONFIG

    return deepcopy(FREE_CONFIG)










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
    Build a DESC custom objective from a FREE config dictionary.

    Any config with wrapper = "linear" is built with LinearObjectiveFromUser.
    Any config with wrapper = "nonlinear" is built with ObjectiveFromUser.
    If wrapper is omitted, the function signature decides:
        (params)     -> LinearObjectiveFromUser
        (grid, data) -> ObjectiveFromUser
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
# FREE extension builder
#========================================================================================================================================

def build_free_extension(
        eq,
        eq_initial = None,
    ):
    """
    Build FREE objective and constraint objects for the current equilibrium.
    """

    free_config = get_free_config()

    objective_configs = tuple(free_config["objectives"])
    constraint_configs = tuple(free_config["constraints"])

    if eq_initial is None:
        eq_initial = eq.copy()

    pressure_axis_target = get_pressure_axis_target(
        eq_initial = eq_initial,
    )

    patched_constraint_configs = []

    for config in constraint_configs:
        config = deepcopy(config)

        if config.get("name") == "FREE_pressure_axis":
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





def append_free_objectives(
        objective,
        free_objectives,
    ):
    """
    Append FREE objectives to an existing ObjectiveFunction.
    """

    if len(free_objectives) == 0:
        return objective

    existing_objectives = tuple(objective.objectives)

    return ObjectiveFunction(
        objectives = existing_objectives + tuple(free_objectives),
    )










#========================================================================================================================================
# Tutorial runners
#========================================================================================================================================

def run_file(
        source_file,
    ):
    """
    Run one tutorial source file.

    The tutorial file itself is responsible for saving both FXD and FREE outputs.
    """

    source_file = resolve_source_file(
        source_file = source_file,
    )

    print("")
    print("################################################################################################################")
    print(f"Starting tutorial run for {source_file}")
    print("################################################################################################################")
    print("")

    runpy.run_path(
        path_name = str(source_file),
        run_name = "__main__",
    )

    tutorial_dir = source_file.parent

    files = find_h5_files(
        case_dir = tutorial_dir,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No FXD or FREE output files were saved by tutorial source: {source_file}"
        )

    print("")
    print("Saved tutorial files:")
    print("")

    for optimization_name, group_files in files.items():
        for variant, path in group_files.items():
            if path is not None:
                print(f"{optimization_name} {variant}: {path}")

    print("")

    return files