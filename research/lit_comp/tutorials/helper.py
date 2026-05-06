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
#   6. optimization result saving/reporting
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
import json
import os
import pickle
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

TOLERANCE_CASE_REPORT_NAME = "tolerance_case_report.json"

DESC_SWEEP_INDEX_ENV_NAME = "DESC_SWEEP_INDEX"
DESC_CASE_START_INDEX_ENV_NAME = "DESC_CASE_START_INDEX"
DESC_CREATED_CASE_LABELS_ENV_NAME = "DESC_CREATED_CASE_LABELS"










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
    Find FXD and FREE h5 files in one output folder.

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
    tutorial_name = case_dir.parent.name if case_dir.name.isdigit() else case_dir.name

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







def find_tutorial_case_h5_files(
        tutorial,
    ):
    """
    Find FXD and FREE h5 files for each output case of a tutorial.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    case_dirs = get_output_case_dirs(
        tutorial_dir = tutorial_dir,
    )

    return {
        case_dir.name: find_h5_files(
            case_dir = case_dir,
        )
        for case_dir in case_dirs
    }






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
# Tolerance sweep helpers
#========================================================================================================================================

def is_numbered_case_dir(
        path,
    ):
    """
    Return True for output folders named 001, 002, 003, ...
    """

    path = Path(path)

    return path.is_dir() and path.name.isdigit()





def find_numbered_case_dirs(
        tutorial_dir,
    ):
    """
    Return numbered tolerance-case folders in a tutorial directory.
    """

    tutorial_dir = Path(tutorial_dir)

    if not tutorial_dir.exists():
        return ()

    return tuple(
        sorted(
            path
            for path in tutorial_dir.iterdir()
            if is_numbered_case_dir(
                path = path,
            )
        )
    )





def get_largest_numbered_case_index(
        tutorial_dir,
    ):
    """
    Return the largest existing numbered output-folder index.
    """

    numbered_case_dirs = find_numbered_case_dirs(
        tutorial_dir = tutorial_dir,
    )

    if len(numbered_case_dirs) == 0:
        return 0

    return max(
        int(path.name)
        for path in numbered_case_dirs
    )





def get_next_numbered_case_index(
        tutorial_dir,
    ):
    """
    Return the next numbered output-folder index.

    This makes repeated driver.py calls continue from the largest existing
    folder number instead of starting over at 001.
    """

    return get_largest_numbered_case_index(
        tutorial_dir = tutorial_dir,
    ) + 1





def get_output_case_dirs(
        tutorial_dir,
    ):
    """
    Return output case directories.

    If numbered folders exist, return:
        001/
        002/
        ...

    Otherwise return the tutorial directory itself for backward compatibility.
    """

    tutorial_dir = Path(tutorial_dir)

    numbered_case_dirs = find_numbered_case_dirs(
        tutorial_dir = tutorial_dir,
    )

    if len(numbered_case_dirs) > 0:
        return numbered_case_dirs

    return (
        tutorial_dir,
    )





def make_tolerance_case_label(
        index,
    ):
    """
    Convert a one-based tolerance-case index to 001, 002, ...
    """

    return f"{int(index):03d}"





def make_tolerance_value(
        order,
    ):
    """
    Convert an order of magnitude to a tolerance value.

    Example:
        order = 6 -> 1e-6
    """

    return 10.0 ** (-int(order))





def update_options_with_inner_tolerances(
        options,
        inner_tolerances,
    ):
    """
    Add inner proximal-solve tolerances to a DESC optimizer options dictionary.
    """

    options = deepcopy(options)

    solve_options = options.get(
        "solve_options",
        {},
    )

    if solve_options is None:
        solve_options = {}

    else:
        solve_options = deepcopy(solve_options)

    solve_options["ftol"] = inner_tolerances["ftol"]
    solve_options["xtol"] = inner_tolerances["xtol"]
    solve_options["gtol"] = inner_tolerances["gtol"]

    options["solve_options"] = solve_options

    return options





def make_tolerance_case(
        index,
        ftol_order,
        xtol_order,
        gtol_order,
        initial_trust_ratio_order,
        inner_ftol_order = None,
        inner_xtol_order = None,
        inner_gtol_order = None,
        base_options = None,
    ):
    """
    Build one tolerance-case dictionary.

    The outer tolerances are passed directly to eq.optimize(...). The inner
    tolerances are passed through options["solve_options"] for the inner
    proximal solve.
    """

    if inner_ftol_order is None:
        inner_ftol_order = ftol_order

    if inner_xtol_order is None:
        inner_xtol_order = xtol_order

    if inner_gtol_order is None:
        inner_gtol_order = gtol_order

    if base_options is None:
        options = {}

    else:
        options = deepcopy(base_options)

    outer_ftol = make_tolerance_value(
        order = ftol_order,
    )

    outer_xtol = make_tolerance_value(
        order = xtol_order,
    )

    outer_gtol = make_tolerance_value(
        order = gtol_order,
    )

    inner_ftol = make_tolerance_value(
        order = inner_ftol_order,
    )

    inner_xtol = make_tolerance_value(
        order = inner_xtol_order,
    )

    inner_gtol = make_tolerance_value(
        order = inner_gtol_order,
    )

    initial_trust_ratio = make_tolerance_value(
        order = initial_trust_ratio_order,
    )

    outer_tolerances = {
        "ftol": outer_ftol,
        "xtol": outer_xtol,
        "gtol": outer_gtol,
        "initial_trust_ratio": initial_trust_ratio,
    }

    inner_tolerances = {
        "ftol": inner_ftol,
        "xtol": inner_xtol,
        "gtol": inner_gtol,
    }

    options = update_options_with_inner_tolerances(
        options = options,
        inner_tolerances = inner_tolerances,
    )

    options["initial_trust_ratio"] = initial_trust_ratio

    return {
        "sweep_index": int(index),
        "case_index": int(index),
        "case_label": make_tolerance_case_label(
            index = index,
        ),
        "orders": {
            "ftol_order": int(ftol_order),
            "xtol_order": int(xtol_order),
            "gtol_order": int(gtol_order),
            "outer_ftol_order": int(ftol_order),
            "outer_xtol_order": int(xtol_order),
            "outer_gtol_order": int(gtol_order),
            "inner_ftol_order": int(inner_ftol_order),
            "inner_xtol_order": int(inner_xtol_order),
            "inner_gtol_order": int(inner_gtol_order),
            "initial_trust_ratio_order": int(initial_trust_ratio_order),
        },
        "tolerances": outer_tolerances,
        "outer_tolerances": outer_tolerances,
        "inner_tolerances": inner_tolerances,
        "options": options,
    }





def iter_tolerance_cases(
        ftol_orders,
        xtol_orders,
        gtol_orders,
        initial_trust_ratio_orders,
        inner_ftol_orders = None,
        inner_xtol_orders = None,
        inner_gtol_orders = None,
        base_options = None,
    ):
    """
    Yield tolerance cases over all requested outer and inner combinations.
    """

    if inner_ftol_orders is None:
        inner_ftol_orders = ftol_orders

    if inner_xtol_orders is None:
        inner_xtol_orders = xtol_orders

    if inner_gtol_orders is None:
        inner_gtol_orders = gtol_orders

    index = 1

    for ftol_order in ftol_orders:
        for xtol_order in xtol_orders:
            for gtol_order in gtol_orders:
                for inner_ftol_order in inner_ftol_orders:
                    for inner_xtol_order in inner_xtol_orders:
                        for inner_gtol_order in inner_gtol_orders:
                            for initial_trust_ratio_order in initial_trust_ratio_orders:

                                yield make_tolerance_case(
                                    index = index,
                                    ftol_order = ftol_order,
                                    xtol_order = xtol_order,
                                    gtol_order = gtol_order,
                                    initial_trust_ratio_order = initial_trust_ratio_order,
                                    inner_ftol_order = inner_ftol_order,
                                    inner_xtol_order = inner_xtol_order,
                                    inner_gtol_order = inner_gtol_order,
                                    base_options = base_options,
                                )

                                index += 1





def get_optional_int_env(
        name,
    ):
    """
    Return an optional integer environment variable.
    """

    value = os.environ.get(
        name,
        None,
    )

    if value is None or value == "":
        return None

    return int(value)





def get_requested_sweep_index():
    """
    Return the requested one-based tolerance-sweep index, if any.
    """

    return get_optional_int_env(
        name = DESC_SWEEP_INDEX_ENV_NAME,
    )





def get_case_start_index_from_env():
    """
    Return the requested first output-folder index, if any.
    """

    return get_optional_int_env(
        name = DESC_CASE_START_INDEX_ENV_NAME,
    )





def should_run_tolerance_case(
        tolerance_case,
    ):
    """
    Return True if this tolerance case should run in the current process.

    If DESC_SWEEP_INDEX is unset, every case is run serially.
    If DESC_SWEEP_INDEX is set, only the matching sweep_index is run.
    """

    requested_sweep_index = get_requested_sweep_index()

    if requested_sweep_index is None:
        return True

    return int(tolerance_case["sweep_index"]) == int(requested_sweep_index)





def get_created_case_labels_from_env():
    """
    Return output case labels created by this Python process.
    """

    value = os.environ.get(
        DESC_CREATED_CASE_LABELS_ENV_NAME,
        "[]",
    )

    try:
        labels = json.loads(value)

    except json.JSONDecodeError:
        labels = []

    return tuple(
        str(label)
        for label in labels
    )





def register_created_case_label(
        case_label,
    ):
    """
    Record one output case label created by this Python process.
    """

    labels = list(get_created_case_labels_from_env())

    case_label = str(case_label)

    if case_label not in labels:
        labels.append(case_label)

    os.environ[DESC_CREATED_CASE_LABELS_ENV_NAME] = json.dumps(labels)





def reserve_next_numbered_case_dir(
        output_dir,
    ):
    """
    Atomically reserve the next numbered output folder.

    This is safe for SLURM array jobs because mkdir(..., exist_ok = False) is
    used as the reservation operation. If another task gets the same folder
    first, this process tries the next number.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    starting_index = get_next_numbered_case_index(
        tutorial_dir = output_dir,
    )

    for case_index in range(starting_index, starting_index + 100000):

        case_label = make_tolerance_case_label(
            index = case_index,
        )

        case_dir = output_dir / case_label

        try:
            case_dir.mkdir(
                parents = False,
                exist_ok = False,
            )

            return case_index, case_label, case_dir

        except FileExistsError:
            continue

    raise RuntimeError(
        f"Could not reserve a numbered output folder in {output_dir}."
    )





def reserve_requested_case_dir(
        output_dir,
        case_index,
    ):
    """
    Reserve a specific numbered output folder.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    case_label = make_tolerance_case_label(
        index = case_index,
    )

    case_dir = output_dir / case_label

    try:
        case_dir.mkdir(
            parents = False,
            exist_ok = False,
        )

    except FileExistsError as error:
        raise FileExistsError(
            f"Requested output folder already exists: {case_dir}\n"
            "Choose a new DESC_CASE_START_INDEX or remove the existing folder."
        ) from error

    return case_label, case_dir





def write_tolerance_case_report(
        case_dir,
        tolerance_case,
    ):
    """
    Save the tolerance combination report for one numbered folder.
    """

    case_dir = Path(case_dir)

    case_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    report_path = case_dir / TOLERANCE_CASE_REPORT_NAME

    with open(report_path, "w") as file:
        json.dump(
            to_json_safe(
                value = tolerance_case,
            ),
            file,
            indent = 4,
        )

    return report_path





def prepare_tolerance_case_dir(
        output_dir,
        tolerance_case,
        case_start_index = None,
    ):
    """
    Create one numbered output folder and write its tolerance report.

    If case_start_index or DESC_CASE_START_INDEX is provided, the output label is
    deterministic:

        case_index = case_start_index + sweep_index - 1

    Otherwise, the next available numbered folder is reserved atomically.
    """

    output_dir = Path(output_dir)

    if case_start_index is None:
        case_start_index = get_case_start_index_from_env()

    if case_start_index is None:
        case_index, case_label, case_dir = reserve_next_numbered_case_dir(
            output_dir = output_dir,
        )

    else:
        sweep_index = int(
            tolerance_case.get(
                "sweep_index",
                tolerance_case.get(
                    "case_index",
                    1,
                ),
            )
        )

        case_index = int(case_start_index) + sweep_index - 1

        case_label, case_dir = reserve_requested_case_dir(
            output_dir = output_dir,
            case_index = case_index,
        )

    tolerance_case["case_index"] = int(case_index)
    tolerance_case["case_label"] = case_label

    register_created_case_label(
        case_label = case_label,
    )

    report_path = write_tolerance_case_report(
        case_dir = case_dir,
        tolerance_case = tolerance_case,
    )

    return case_dir, report_path





def print_tolerance_case_header(
        tolerance_case,
        case_dir,
    ):
    """
    Print the current tolerance case.
    """

    outer_tolerances = tolerance_case.get(
        "outer_tolerances",
        tolerance_case["tolerances"],
    )

    inner_tolerances = tolerance_case.get(
        "inner_tolerances",
        {},
    )

    print("")
    print("================================================================================================================")
    print(f"Running tolerance case {tolerance_case['case_label']}")
    print("================================================================================================================")
    print("")
    print(f"Sweep index: {tolerance_case.get('sweep_index', tolerance_case['case_index'])}")
    print(f"Output folder: {case_dir}")
    print("")
    print("Outer optimizer tolerances:")
    print(f"ftol: {outer_tolerances['ftol']}")
    print(f"xtol: {outer_tolerances['xtol']}")
    print(f"gtol: {outer_tolerances['gtol']}")
    print(f"initial_trust_ratio: {outer_tolerances['initial_trust_ratio']}")
    print("")
    print("Inner proximal-solve tolerances:")
    print(f"solve_options.ftol: {inner_tolerances.get('ftol')}")
    print(f"solve_options.xtol: {inner_tolerances.get('xtol')}")
    print(f"solve_options.gtol: {inner_tolerances.get('gtol')}")
    print("")




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

    Uses result.cost when available. If only result.fun exists, falls back to
    0.5 * ||fun||^2 for vector-valued least-squares residuals.
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

    Example:
        basic_qs_T_FXD.h5
        basic_qs_T_FXD_result.pkl
        basic_qs_T_FXD_result_summary.json
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
        maxiter,
        options,
        x_scale,
        tolerance_case = None,
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
            "maxiter": maxiter,
            "x_scale": x_scale,
            "options": options,
        },
    }

    if tolerance_case is not None:
        summary["hyperparameters"]["outer_tolerances"] = tolerance_case.get(
            "outer_tolerances",
            tolerance_case.get(
                "tolerances",
                None,
            ),
        )

        summary["hyperparameters"]["inner_tolerances"] = tolerance_case.get(
            "inner_tolerances",
            None,
        )

        summary["hyperparameters"]["solve_options"] = options.get(
            "solve_options",
            None,
        ) if isinstance(options, dict) else None

        summary["tolerance_case"] = tolerance_case

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
        maxiter,
        options,
        x_scale,
        tolerance_case = None,
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
        maxiter = maxiter,
        options = options,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
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
        maxiter = None,
        options = None,
        copy = False,
        verbose = 3,
        x_scale = "auto",
        tolerance_case = None,
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
        maxiter = maxiter,
        options = options,
        copy = copy,
        verbose = verbose,
        x_scale = x_scale,
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
        maxiter = maxiter,
        options = options,
        x_scale = x_scale,
        tolerance_case = tolerance_case,
    )

    return eq, result










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

    The tutorial file itself is responsible for saving FXD and FREE outputs.
    If the tutorial performs a tolerance sweep, outputs are expected inside
    numbered folders.
    """

    source_file = resolve_source_file(
        source_file = source_file,
    )

    if DESC_CREATED_CASE_LABELS_ENV_NAME in os.environ:
        del os.environ[DESC_CREATED_CASE_LABELS_ENV_NAME]

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

    created_case_labels = get_created_case_labels_from_env()

    if len(created_case_labels) > 0:
        case_dirs = tuple(
            tutorial_dir / case_label
            for case_label in created_case_labels
        )

    else:
        case_dirs = get_output_case_dirs(
            tutorial_dir = tutorial_dir,
        )

    files_by_case = {
        case_dir.name: find_h5_files(
            case_dir = case_dir,
        )
        for case_dir in case_dirs
    }

    has_files = any(
        len(case_files) > 0
        for case_files in files_by_case.values()
    )

    if not has_files:
        raise FileNotFoundError(
            f"No FXD or FREE output files were saved by tutorial source: {source_file}"
        )

    print("")
    print("Saved tutorial files:")
    print("")

    for case_label, case_files in files_by_case.items():

        print(f"Case: {case_label}")

        for optimization_name, group_files in case_files.items():
            for variant, path in group_files.items():
                if path is not None:
                    print(f"    {optimization_name} {variant}: {path}")

        print("")

    return files_by_case
