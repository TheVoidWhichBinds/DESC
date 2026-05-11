# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/lit_comp/cases.
#
# This file owns:
#   1. case-local path helpers
#   2. DESC h5 loading helpers
#   3. objective comparison helpers
#   4. FREE pressure objective/constraint construction
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
#               force/
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

FXD_SUFFIX = "_FXD"
FREE_SUFFIX = "_FREE"

CASE_OBJECTIVE_FOLDERS = (
    "qs3",
    "balloon",
    "force",
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
            FXD_SUFFIX,
            FREE_SUFFIX,
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
    Find FXD and FREE h5 files in one output folder.

    Returns:
        {
            "FXD": Path(...),
            "FREE": Path(...),
        }
    """

    case_dir = Path(case_dir)

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




