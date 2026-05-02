# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/lit_comp.
#
# This file owns:
#   1. FREE config access
#   2. FixPressure removal for FREE runs
#   3. Initial-equilibrium injection through kwargs["thing"]
#   4. DESC custom objective construction
#   5. Optimizer patching for recreation-file FREE runs
#   6. Recreation-file output path patching
#   7. Case-objective comparison helpers
#
#==============================================================================================================

from copy import deepcopy
from pathlib import Path
import csv
import inspect
import os
import runpy
import traceback

import numpy as np

from desc.io import load
from desc.optimize import Optimizer
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    ObjectiveFromUser,
    LinearObjectiveFromUser,
)










#========================================================================================================================================
# Check / comparison settings
#========================================================================================================================================

ORIGINAL_SUFFIX = "_OG"
FXD_SUFFIX = "_FXD"
FREE_SUFFIX = "_FREE"

SUMMARY_KEYS = [
    "value",
    "max_abs",
    "mean_abs",
    "rms",
    "min",
    "max",
    "size",
]

QUANTITY_CANDIDATES = [
    #-----------------------------------
    # Main equilibrium / objective-like quantities
    #-----------------------------------
    "F_rho",
    "F_theta",
    "F_zeta",
    "|F|",
    "force",
    "p",
    "iota",
    "current",
    "lambda",

    #-----------------------------------
    # Common natural DESC scalar outputs
    #-----------------------------------
    "R0",
    "a",
    "A",
    "V",
    "S",
    "R0/a",
    "<R>",
    "<Z>",
    "<B>",
    "<B^2>",
    "B0",
    "W",
    "W_p",
    "W_B",
    "beta",
    "<beta>",
    "beta_a",
    "beta_p",
    "beta_t",
    "shear",
    "well",
    "magnetic well",

    #-----------------------------------
    # Stability / optimization metrics used across papers
    #-----------------------------------
    "D_Mercier",
    "Mercier",
    "ideal ballooning lambda",
    "ideal ballooning",
    "QS error",
    "quasisymmetry",
    "mirror ratio",
    "elongation",
    "curvature",
    "curvature_k1_rho",
    "curvature_k2_rho",
]










#========================================================================================================================================
# Path helpers
#========================================================================================================================================

def get_lit_comp_dir():
    """
    Returns the research/lit_comp directory.
    """

    return Path(__file__).resolve().parent





def get_papers_dir():
    """
    Return the papers directory containing paper recreation cases.
    """

    return get_lit_comp_dir() / "papers"





def get_case_dir(
        paper,
        case,
    ):
    """
    Return the paper case directory.
    """

    return get_papers_dir() / paper / case





def normalize_case_name(
        name,
    ):
    """
    Normalize a passed filename/case stem by removing .h5, _OG, _FXD, and _FREE.
    """

    path = Path(name)
    stem = path.stem

    for suffix in (
        ORIGINAL_SUFFIX,
        FXD_SUFFIX,
        FREE_SUFFIX,
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem





def find_matching_fxd_files(
        case_dir,
        file_name,
    ):
    """
    Find the OG and FXD files matching the requested case.
    """

    case_name = normalize_case_name(
        name = file_name,
    )

    h5_files = sorted(case_dir.glob("*.h5"))

    original_matches = []
    fxd_matches = []

    for path in h5_files:
        normalized = normalize_case_name(
            name = path.name,
        )

        if normalized != case_name:
            continue

        if path.stem.endswith(ORIGINAL_SUFFIX):
            original_matches.append(path)

        elif path.stem.endswith(FXD_SUFFIX):
            fxd_matches.append(path)

    if len(original_matches) == 0:
        raise FileNotFoundError(
            "Could not find OG file for case '{}' in {}".format(
                case_name,
                case_dir,
            )
        )

    if len(fxd_matches) == 0:
        raise FileNotFoundError(
            "Could not find FXD file for case '{}' in {}".format(
                case_name,
                case_dir,
            )
        )

    if len(original_matches) > 1:
        raise RuntimeError(
            "Found multiple OG files for case '{}':\n{}".format(
                case_name,
                "\n".join(str(path) for path in original_matches),
            )
        )

    if len(fxd_matches) > 1:
        raise RuntimeError(
            "Found multiple FXD files for case '{}':\n{}".format(
                case_name,
                "\n".join(str(path) for path in fxd_matches),
            )
        )

    return original_matches[0], fxd_matches[0], case_name





def find_h5_files(
        case_dir : Path,
    ):
    """
    Finds OG, FXD, and FREE h5 files in the case folder.
    """

    suffixes = {
        "OG": "_OG.h5",
        "FXD": "_FXD.h5",
        "FREE": "_FREE.h5",
    }

    files = {}

    for label, suffix in suffixes.items():
        matches = sorted(case_dir.glob(f"*{suffix}"))

        if len(matches) == 0:
            matches = sorted(case_dir.rglob(f"*{suffix}"))

        if len(matches) == 0:
            files[label] = None

        elif len(matches) == 1:
            files[label] = matches[0]

        else:
            raise RuntimeError(
                f"Multiple files ending in {suffix} found in {case_dir}:\n"
                + "\n".join(str(match) for match in matches)
            )

    return files





def find_case_h5_files(
        paper : str,
        case : str,
    ):
    """
    Finds OG, FXD, and FREE h5 files for a paper/case.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    return find_h5_files(
        case_dir = case_dir,
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
    Loads the final equilibrium from a DESC h5 file.
    """

    return load_latest_equilibrium(
        path = path,
    )





def safe_compute(
        eq,
        quantity,
    ):
    """
    Compute a DESC quantity if available.
    """

    try:
        data = eq.compute(quantity)

    except Exception:
        return None

    if isinstance(data, dict):
        if quantity in data:
            return data[quantity]

        if len(data) == 1:
            return next(iter(data.values()))

        return data

    return data










#========================================================================================================================================
# Numeric comparison helpers
#========================================================================================================================================

def to_numeric_array(
        value,
    ):
    """
    Convert a DESC compute output to a finite numeric numpy array.
    """

    if isinstance(value, dict):
        return None

    try:
        array = np.asarray(value)

    except Exception:
        return None

    if array.dtype == object:
        return None

    try:
        array = array.astype(float)

    except Exception:
        return None

    array = array[np.isfinite(array)]

    if array.size == 0:
        return None

    return array





def summarize_value(
        value,
    ):
    """
    Summarize scalar or array-valued DESC output.
    """

    array = to_numeric_array(
        value = value,
    )

    if array is None:
        return None

    flat = array.reshape(-1)

    if flat.size == 1:
        scalar = float(flat[0])

        return {
            "value": scalar,
            "max_abs": abs(scalar),
            "mean_abs": abs(scalar),
            "rms": abs(scalar),
            "min": scalar,
            "max": scalar,
            "size": 1,
        }

    return {
        "value": np.nan,
        "max_abs": float(np.max(np.abs(flat))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "rms": float(np.sqrt(np.mean(flat ** 2))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "size": int(flat.size),
    }





def relative_difference(
        original,
        check,
    ):
    """
    Return relative difference using the OG value as reference.
    """

    if not np.isfinite(original) or not np.isfinite(check):
        return np.nan

    scale = max(abs(original), 1.0e-30)

    return abs(check - original) / scale





def format_float(
        value,
    ):
    """
    Format floats for terminal output.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        if not np.isfinite(value):
            return "nan"

        return "{:.6e}".format(value)

    except Exception:
        return str(value)





def flatten_values(
        values,
    ):
    """
    Converts objective output to a flat numpy array.
    """

    values = np.asarray(values)

    return values.reshape(-1)





def summarize_values(
        values,
    ):
    """
    Computes scalar summaries of an objective vector.
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
    Collect possible objective input vectors for different DESC objective APIs.
    """

    things = normalize_things(
        thing = thing,
    )

    possible_xs = []

    if hasattr(objective, "x"):
        try:
            possible_xs.append((objective.x(thing),))

        except Exception:
            pass

        try:
            possible_xs.append((objective.x(*things),))

        except Exception:
            pass

        if len(things) == 1:
            try:
                possible_xs.append((objective.x(things[0]),))

            except Exception:
                pass

    if hasattr(objective, "xs"):
        try:
            xs = objective.xs(thing)

            if isinstance(xs, tuple):
                possible_xs.append(xs)

            else:
                possible_xs.append((xs,))

        except Exception:
            pass

        try:
            xs = objective.xs(*things)

            if isinstance(xs, tuple):
                possible_xs.append(xs)

            else:
                possible_xs.append((xs,))

        except Exception:
            pass

        if len(things) == 1:
            try:
                xs = objective.xs(things[0])

                if isinstance(xs, tuple):
                    possible_xs.append(xs)

                else:
                    possible_xs.append((xs,))

            except Exception:
                pass

    return possible_xs





def evaluate_objective_safely(
        objective,
        thing,
    ):
    """
    Evaluate an objective vector, preferring unscaled output when available.

    The input thing may be:
        eq
    or:
        (eq, field)
    """

    things = normalize_things(
        thing = thing,
    )

    build_objective_safely(
        objective = objective,
        thing = thing,
    )

    possible_xs = collect_possible_xs(
        objective = objective,
        thing = thing,
    )

    possible_compute_names = [
        "compute_unscaled",
        "compute_unscaled_error",
        "compute",
    ]

    for compute_name in possible_compute_names:
        if not hasattr(objective, compute_name):
            continue

        compute = getattr(objective, compute_name)

        for xs in possible_xs:
            try:
                return flatten_values(
                    values = compute(*xs),
                )

            except Exception:
                pass

        try:
            return flatten_values(
                values = compute(thing),
            )

        except Exception:
            pass

        try:
            return flatten_values(
                values = compute(*things),
            )

        except Exception:
            pass

        if len(things) == 1:
            try:
                return flatten_values(
                    values = compute(things[0]),
                )

            except Exception:
                pass

        try:
            return flatten_values(
                values = compute(),
            )

        except Exception:
            pass

    raise RuntimeError(f"Could not evaluate objective: {objective}")










#========================================================================================================================================
# Case-specific objective constructors
#========================================================================================================================================

def make_dudt2024_helical_qs_field(
        eq,
    ):
    """
    Recreate the OmnigenousField used by dudt2024/helical_qs.py.
    """

    from desc.magnetic_fields import OmnigenousField

    field = OmnigenousField(
        L_B = 4,
        M_B = 8,
        L_x = 0,
        M_x = 0,
        N_x = 0,
        NFP = eq.NFP,
        helicity = (1, eq.NFP),
    )

    return field





def make_dudt2024_helical_qs_omnigenity_objective(
        eq,
        field,
        rho,
        eta_weight,
    ):
    """
    Recreate one Omnigenity objective from dudt2024/helical_qs.py.
    """

    from desc.grid import LinearGrid
    from desc.objectives import Omnigenity

    M_booz = min(int(np.ceil(1.5 * eq.M)), 16)
    N_booz = min(int(np.ceil(1.5 * eq.N)), 16)

    eq_grid = LinearGrid(
        rho = rho,
        M = int(np.ceil(1.5 * M_booz)),
        N = int(np.ceil(1.5 * N_booz)),
        NFP = eq.NFP,
        sym = False,
    )

    field_grid = LinearGrid(
        rho = rho,
        theta = 2 * M_booz,
        zeta = 2 * N_booz,
        NFP = field.NFP,
        sym = False,
    )

    objective = Omnigenity(
        eq = eq,
        field = field,
        eq_grid = eq_grid,
        field_grid = field_grid,
        eta_weight = eta_weight,
    )

    return objective





def dudt2024_helical_qs_objectives(
        eq,
    ):
    """
    Return the actual case objectives used by dudt2024/helical_qs.py.
    """

    from desc.objectives import CurrentDensity

    field = make_dudt2024_helical_qs_field(
        eq = eq,
    )

    surfaces = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]

    objective_specs = [
        {
            "name": "helical_qs CurrentDensity",
            "objective": CurrentDensity(
                eq = eq,
                weight = 4e0,
            ),
            "thing": eq,
        },
    ]

    for rho in surfaces:
        objective_specs.append(
            {
                "name": f"helical_qs Omnigenity rho={rho}",
                "objective": make_dudt2024_helical_qs_omnigenity_objective(
                    eq = eq,
                    field = field,
                    rho = rho,
                    eta_weight = 2,
                ),
                "thing": (eq, field),
            }
        )

    return objective_specs





def get_case_objectives(
        paper,
        case_name,
        eq,
    ):
    """
    Return case-specific objective specs if this paper/case has them.
    """

    if paper == "dudt2024" and case_name == "helical_qs":
        return dudt2024_helical_qs_objectives(
            eq = eq,
        )

    return []










#========================================================================================================================================
# Comparison helpers
#========================================================================================================================================

def compare_summaries(
        category,
        quantity,
        original_summary,
        check_summary,
    ):
    """
    Create comparison rows from two summary dictionaries.
    """

    rows = []

    for key in SUMMARY_KEYS:
        original_stat = original_summary[key]
        check_stat = check_summary[key]

        if isinstance(original_stat, int) or isinstance(check_stat, int):
            absolute = check_stat - original_stat
            relative = 0.0 if absolute == 0 else np.nan

        else:
            absolute = abs(check_stat - original_stat)
            relative = relative_difference(
                original = original_stat,
                check = check_stat,
            )

        rows.append(
            {
                "category": category,
                "quantity": quantity,
                "statistic": key,
                "original": original_stat,
                "check": check_stat,
                "absolute_difference": absolute,
                "relative_difference": relative,
            }
        )

    return rows





def compare_equilibria(
        eq_original,
        eq_check,
    ):
    """
    Compare supported DESC quantities between OG and FXD equilibria.
    """

    rows = []

    for quantity in QUANTITY_CANDIDATES:
        original_value = safe_compute(
            eq = eq_original,
            quantity = quantity,
        )

        check_value = safe_compute(
            eq = eq_check,
            quantity = quantity,
        )

        if original_value is None and check_value is None:
            continue

        original_summary = summarize_value(
            value = original_value,
        )

        check_summary = summarize_value(
            value = check_value,
        )

        if original_summary is None or check_summary is None:
            rows.append(
                {
                    "category": "DESC compute",
                    "quantity": quantity,
                    "statistic": "status",
                    "original": "computed" if original_value is not None else "missing",
                    "check": "computed" if check_value is not None else "missing",
                    "absolute_difference": "",
                    "relative_difference": "",
                }
            )

            continue

        rows.extend(
            compare_summaries(
                category = "DESC compute",
                quantity = quantity,
                original_summary = original_summary,
                check_summary = check_summary,
            )
        )

    return rows





def compare_case_objectives(
        paper,
        case_name,
        eq_original,
        eq_check,
    ):
    """
    Compare case-specific optimization objectives between OG and FXD equilibria.
    """

    rows = []

    original_specs = get_case_objectives(
        paper = paper,
        case_name = case_name,
        eq = eq_original,
    )

    check_specs = get_case_objectives(
        paper = paper,
        case_name = case_name,
        eq = eq_check,
    )

    if len(original_specs) == 0 and len(check_specs) == 0:
        return rows

    if len(original_specs) != len(check_specs):
        rows.append(
            {
                "category": "case objective",
                "quantity": "objective registry",
                "statistic": "status",
                "original": f"{len(original_specs)} objectives",
                "check": f"{len(check_specs)} objectives",
                "absolute_difference": "",
                "relative_difference": "",
            }
        )

        return rows

    for original_spec, check_spec in zip(original_specs, check_specs):
        objective_name = original_spec["name"]

        try:
            original_value = evaluate_objective_safely(
                objective = original_spec["objective"],
                thing = original_spec.get("thing", eq_original),
            )

            check_value = evaluate_objective_safely(
                objective = check_spec["objective"],
                thing = check_spec.get("thing", eq_check),
            )

            original_summary = summarize_value(
                value = original_value,
            )

            check_summary = summarize_value(
                value = check_value,
            )

            if original_summary is None or check_summary is None:
                rows.append(
                    {
                        "category": "case objective",
                        "quantity": objective_name,
                        "statistic": "status",
                        "original": "computed" if original_summary is not None else "not numeric",
                        "check": "computed" if check_summary is not None else "not numeric",
                        "absolute_difference": "",
                        "relative_difference": "",
                    }
                )

                continue

            rows.extend(
                compare_summaries(
                    category = "case objective",
                    quantity = objective_name,
                    original_summary = original_summary,
                    check_summary = check_summary,
                )
            )

        except Exception as error:
            rows.append(
                {
                    "category": "case objective",
                    "quantity": objective_name,
                    "statistic": "status",
                    "original": "error",
                    "check": "error",
                    "absolute_difference": "",
                    "relative_difference": repr(error),
                }
            )

    return rows





def compare_all(
        paper,
        case_name,
        eq_original,
        eq_check,
    ):
    """
    Compare DESC quantities and case-specific objectives.

    This is used by recreate.py for OG vs FXD recreation checks.
    compare.py only writes the case-objective CSV.
    """

    rows = []

    rows.extend(
        compare_equilibria(
            eq_original = eq_original,
            eq_check = eq_check,
        )
    )

    rows.extend(
        compare_case_objectives(
            paper = paper,
            case_name = case_name,
            eq_original = eq_original,
            eq_check = eq_check,
        )
    )

    return rows





def compare_objective_set(
        files : dict,
        objective_getter,
    ):
    """
    Compares one set of objectives across OG, FXD, and FREE files.
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

def print_comparison_rows(
        rows,
    ):
    """
    Print comparison rows to terminal.
    """

    if len(rows) == 0:
        print("")
        print("No comparable DESC quantities or case objectives were found.")
        return

    print("")
    print("================================================================")
    print("Comparison")
    print("================================================================")
    print(
        "{:<18} {:<36} {:<14} {:>16} {:>16} {:>16} {:>16}".format(
            "category",
            "quantity",
            "statistic",
            "OG",
            "FXD",
            "abs diff",
            "rel diff",
        )
    )
    print("-" * 138)

    for row in rows:
        print(
            "{:<18} {:<36} {:<14} {:>16} {:>16} {:>16} {:>16}".format(
                row["category"][:18],
                row["quantity"][:36],
                row["statistic"][:14],
                format_float(row["original"]),
                format_float(row["check"]),
                format_float(row["absolute_difference"]),
                format_float(row["relative_difference"]),
            )
        )





def save_comparison_rows(
        rows,
        path,
    ):
    """
    Save comparison rows to CSV.
    """

    with open(path, "w", newline = "") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames = [
                "category",
                "quantity",
                "statistic",
                "original",
                "check",
                "absolute_difference",
                "relative_difference",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)





def write_table_csv(
        rows : list,
        path,
    ):
    """
    Writes objective comparison rows to a readable wide-format CSV.

    Output format:

        objective, metric, OG, FXD, FREE
        CurrentDensity, l2, ...
        CurrentDensity, max_abs, ...
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





def write_table_markdown(
        rows : list,
        path,
    ):
    """
    Writes rows to a markdown table.
    """

    if len(rows) == 0:
        path.write_text("No rows produced.\n")
        return

    headers = list(rows[0].keys())

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")

    path.write_text("\n".join(lines) + "\n")










#========================================================================================================================================
# FREE Helpers
#========================================================================================================================================

def get_free_config():
    """
    Return the FREE configuration.
    """

    try:
        from .wrappers import FREE_CONFIG
    except ImportError:
        from research.lit_comp.wrappers import FREE_CONFIG

    return deepcopy(FREE_CONFIG)










#========================================================================================================================================
# DESC Custom Objective Builders
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
# Constraint Removal Helpers
#========================================================================================================================================

FIXED_PRESSURE_CLASS_NAMES = {
    "FixPressure",
    "PressureFixed",
}










def is_fixed_pressure_constraint(
        constraint,
    ):
    """
    Return True when a constraint fixes the pressure profile.
    """

    class_name = constraint.__class__.__name__

    if class_name in FIXED_PRESSURE_CLASS_NAMES:
        return True

    name = getattr(
        constraint,
        "name",
        "",
    )

    name = str(name)

    return any(
        fixed_name in name
        for fixed_name in FIXED_PRESSURE_CLASS_NAMES
    )





def remove_fixed_pressure_constraints_from_objects(
        constraints,
    ):
    """
    Remove pressure-profile-fixing constraints from instantiated constraint objects.
    """

    if constraints is None:
        return ()

    kept_constraints = []

    for constraint in tuple(constraints):
        if is_fixed_pressure_constraint(
                constraint = constraint,
            ):
            print(f"Removing fixed pressure constraint: {constraint.__class__.__name__}")
            continue

        kept_constraints.append(constraint)

    return tuple(kept_constraints)










#========================================================================================================================================
# Equilibrium Helpers
#========================================================================================================================================

def find_equilibrium_in_object(
        obj,
    ):
    """
    Find the first Equilibrium object inside an object, tuple, or list.
    """

    if isinstance(obj, Equilibrium):
        return obj

    if isinstance(obj, (tuple, list)):
        for item in obj:
            eq = find_equilibrium_in_object(
                obj = item,
            )

            if eq is not None:
                return eq

    return None










#========================================================================================================================================
# FREE Extension Builder
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
# Optimizer Patch
#========================================================================================================================================

class FREEOptimizePatch:
    """
    Patch Optimizer.optimize for one FREE paper recreation-file run.
    """

    def __init__(
            self,
        ):
        self.original_optimize = None
        self.eq_initial = None

    def __enter__(self):
        self.original_optimize = Optimizer.optimize

        def patched_optimize(
                optimizer_self,
                things,
                objective,
                constraints = (),
                *args,
                **kwargs,
            ):
            eq = find_equilibrium_in_object(
                obj = things,
            )

            if eq is None:
                raise ValueError(
                    "Could not find an Equilibrium object in optimizer.optimize things."
                )

            if self.eq_initial is None:
                self.eq_initial = eq.copy()

            free_objectives, free_constraints = build_free_extension(
                eq = eq,
                eq_initial = self.eq_initial,
            )

            objective_patched = append_free_objectives(
                objective = objective,
                free_objectives = free_objectives,
            )

            constraints_patched = (
                remove_fixed_pressure_constraints_from_objects(
                    constraints = constraints,
                )
                + free_constraints
            )

            print("")
            print("================================================================================================================")
            print("Running with variant: FREE")
            print(f"Added objectives: {len(free_objectives)}")
            print(f"Added constraints: {len(free_constraints)}")
            print(f"Total constraints after fixed pressure removal: {len(constraints_patched)}")
            print("================================================================================================================")
            print("")

            return self.original_optimize(
                optimizer_self,
                things,
                objective_patched,
                constraints_patched,
                *args,
                **kwargs,
            )

        Optimizer.optimize = patched_optimize

        return self

    def __exit__(
            self,
            exc_type,
            exc_value,
            exc_traceback,
        ):
        Optimizer.optimize = self.original_optimize










#========================================================================================================================================
# Recreation File Path Helpers
#========================================================================================================================================

def resolve_source_file(
        source_file,
    ):
    """
    Resolve a source file path.

    Supports either:
        research/lit_comp/papers/dudt2024/helical_qs/helical_qs.py
    or, from DESC:
        dudt2024/helical_qs/helical_qs.py
    """

    source_file = Path(source_file)

    candidates = []

    if source_file.is_absolute():
        candidates.append(source_file)

    else:
        candidates.append(Path.cwd() / source_file)
        candidates.append(get_papers_dir() / source_file)

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find source file. Tried:\n"
        + "\n".join(str(candidate.resolve()) for candidate in candidates)
    )





def get_case_name_from_source(
        source_file,
    ):
    """
    Return the case name for a source file.

    Supports:
        papers/dudt2024/input/helical_qs/helical_qs.py
        papers/dudt2024/helical_qs/helical_qs.py
        papers/dudt2024/input/helical_qs.py
    """

    source_file = Path(source_file).resolve()

    if source_file.parent.name == "input":
        return source_file.stem

    return source_file.parent.name





def get_paper_dir_for_source(
        source_file,
    ):
    """
    Return the paper directory for a recreation source file.

    Supports:
        papers/dudt2024/input/helical_qs/helical_qs.py  -> papers/dudt2024
        papers/dudt2024/helical_qs/helical_qs.py        -> papers/dudt2024
        papers/dudt2024/input/helical_qs.py             -> papers/dudt2024
    """

    source_file = Path(source_file).resolve()

    if source_file.parent.name == "input":
        return source_file.parent.parent

    if source_file.parent.parent.name == "input":
        return source_file.parent.parent.parent

    return source_file.parent.parent





def get_output_dir_for_source(
        source_file,
    ):
    """
    Return the directory containing the source file.

    FREE output files are written beside the recreation file passed to --file.
    """

    source_file = Path(source_file).resolve()

    return source_file.parent





def get_final_output_path_for_source(
        source_file,
    ):
    """
    Return <source-file-dir>/<source-file-stem>_FREE.h5.
    """

    source_file = Path(source_file).resolve()

    return source_file.parent / f"{source_file.stem}_FREE.h5"





def get_failure_output_path_for_source(
        source_file,
    ):
    """
    Return <source-file-dir>/<source-file-stem>_FREE_FAILURE.h5.
    """

    source_file = Path(source_file).resolve()

    return source_file.parent / f"{source_file.stem}_FREE_FAILURE.h5"










#========================================================================================================================================
# Recreation Source Patching
#========================================================================================================================================

def patch_recreation_source_text(
        source_text,
    ):
    """
    Patch recreation source text so final/failure output paths can be overridden by this driver.

    Important:
        Do not override OUTPUT_DIR, because recreation files often use OUTPUT_DIR
        to locate local input files such as *_initial.h5.
    """

    patched = source_text

    if "import os" not in patched:
        patched = patched.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\nimport os\n",
            1,
        )

    if "VFO_FINAL_PATH" in patched:
        return patched

    lines = patched.splitlines()
    new_lines = []

    for line in lines:
        new_lines.append(line)

        if line.lstrip().startswith("FINAL_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FINAL_PATH"):
    FINAL_PATH = Path(os.environ["VFO_FINAL_PATH"])
""".strip()
            )

        if line.lstrip().startswith("FXD_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FINAL_PATH"):
    FXD_PATH = Path(os.environ["VFO_FINAL_PATH"])
""".strip()
            )

        if line.lstrip().startswith("CHECK_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FINAL_PATH"):
    CHECK_PATH = Path(os.environ["VFO_FINAL_PATH"])
""".strip()
            )

        if line.lstrip().startswith("FAILURE_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FAILURE_PATH"):
    FAILURE_PATH = Path(os.environ["VFO_FAILURE_PATH"])
""".strip()
            )

    return "\n".join(new_lines) + "\n"





def make_temporary_recreation_runner(
        source_file,
    ):
    """
    Create a patched temporary copy of the target recreation source file.

    The temporary runner is written beside the original file so __file__.parent
    still points to the original input folder.
    """

    source_file = Path(source_file).resolve()

    source_text = source_file.read_text()

    patched_text = patch_recreation_source_text(
        source_text = source_text,
    )

    runner_path = source_file.parent / f"_{source_file.stem}_FREE_runner.py"

    runner_path.write_text(
        patched_text,
    )

    return runner_path










#========================================================================================================================================
# Recreation Runners
#========================================================================================================================================

def run_source_as_is(
        source_file,
    ):
    """
    Run one recreation source file exactly as written.
    """

    source_file = resolve_source_file(
        source_file = source_file,
    )

    print("")
    print("################################################################################################################")
    print(f"Starting unmodified run for {source_file}")
    print("################################################################################################################")
    print("")

    runpy.run_path(
        path_name = str(source_file),
        run_name = "__main__",
    )

    return None





def run_source_with_free(
        source_file,
    ):
    """
    Run one paper recreation source file with FREE enabled.
    """

    source_file = resolve_source_file(
        source_file = source_file,
    )

    output_dir = get_output_dir_for_source(
        source_file = source_file,
    )

    output_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    final_path = get_final_output_path_for_source(
        source_file = source_file,
    )

    failure_path = get_failure_output_path_for_source(
        source_file = source_file,
    )

    runner_path = make_temporary_recreation_runner(
        source_file = source_file,
    )

    old_env = dict(os.environ)

    os.environ["VFO_VARIANT"] = "FREE"
    os.environ["VFO_OUTPUT_DIR"] = str(output_dir)
    os.environ["VFO_FINAL_PATH"] = str(final_path)
    os.environ["VFO_FAILURE_PATH"] = str(failure_path)

    try:
        with FREEOptimizePatch():
            runpy.run_path(
                path_name = str(runner_path),
                run_name = "__main__",
            )

    finally:
        os.environ.clear()
        os.environ.update(old_env)

        if runner_path.exists():
            runner_path.unlink()

    if not final_path.exists():
        raise FileNotFoundError(
            f"Expected final FREE output was not created: {final_path}"
        )

    return final_path





def run_file(
        source_file,
        free = False,
    ):
    """
    Run one recreation source file.

    If free is False:
        Run the source file exactly as-is.

    If free is True:
        Run the source file with FREE objectives added and fixed pressure removed.
    """

    source_file = resolve_source_file(
        source_file = source_file,
    )

    if free is False:
        return run_source_as_is(
            source_file = source_file,
        )

    if free is True:
        print("")
        print("################################################################################################################")
        print(f"Starting FREE run for {source_file}")
        print("################################################################################################################")
        print("")

        try:
            output = run_source_with_free(
                source_file = source_file,
            )

            print("")
            print(f"FREE saved to: {output}")
            print("")

            return output

        except Exception:
            print("")
            print(f"FREE failed for {source_file}")
            traceback.print_exc()
            raise

    raise ValueError(
        "free must be True or False."
    )