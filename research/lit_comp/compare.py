# compare.py
#==============================================================================================================
#
# Compare DESC objective outputs and paper/case-specific objective outputs between OG, CHECK, and FNO files.
#
# Usage:
#   python3 research/lit_comp/compare.py --paper dudt2024 --case helical_qs
#
#==============================================================================================================

import argparse
import csv
import traceback

import numpy as np

from helper import (
    find_h5_files,
    get_case_dir,
    load_final_eq,
)










#==============================================================================================================
# Case-specific objective registry:
#==============================================================================================================

CASE_OBJECTIVES = {
    "dudt2024": {
        "helical_qs": [
            # Fill this in later.
            #
            # Example:
            #
            # {
            #     "name": "Quasisymmetry",
            #     "objective": lambda eq: QuasisymmetryBoozer(
            #         eq = eq,
            #         helicity = (1, eq.NFP),
            #         M_booz = 2 * eq.M,
            #         N_booz = 2 * eq.N,
            #     ),
            # },
        ],
    },
}










#==============================================================================================================
# Objective evaluation helpers:
#==============================================================================================================

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

    values = flatten_values(values)

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





def build_objective_safely(
        objective,
        eq,
    ):
    """
    Builds a DESC objective while tolerating small API differences.
    """

    try:
        objective.build(
            eq = eq,
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

    objective.build()





def evaluate_objective_safely(
        objective,
        eq,
    ):
    """
    Evaluates an objective vector, preferring unnormalized/unscaled output when available.
    """

    build_objective_safely(
        objective = objective,
        eq = eq,
    )

    possible_xs = []

    if hasattr(objective, "x"):
        try:
            possible_xs.append((objective.x(eq),))

        except Exception:
            pass

    if hasattr(objective, "xs"):
        try:
            xs = objective.xs(eq)

            if isinstance(xs, tuple):
                possible_xs.append(xs)

            else:
                possible_xs.append((xs,))

        except Exception:
            pass

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
                return flatten_values(compute(*xs))

            except Exception:
                pass

        try:
            return flatten_values(compute(eq))

        except Exception:
            pass

        try:
            return flatten_values(compute())

        except Exception:
            pass

    raise RuntimeError(f"Could not evaluate objective: {objective}")





def default_desc_objectives(
        eq,
    ):
    """
    Returns the general DESC objectives to compare for every paper/case.

    This currently uses DESC's default force-balance equilibrium objective.
    """

    from desc.objectives import get_equilibrium_objective

    return [
        {
            "name": "DESC ForceBalance objective",
            "objective": get_equilibrium_objective(
                eq = eq,
                mode = "force",
            ),
        },
    ]










#==============================================================================================================
# Table creation:
#==============================================================================================================

def write_table_csv(
        rows : list,
        path,
    ):
    """
    Writes rows to CSV.
    """

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline = "") as file:
        writer = csv.DictWriter(
            file,
            fieldnames = fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)





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





def compare_objective_set(
        files : dict,
        objective_getter,
    ):
    """
    Compares one set of objectives across OG, CHECK, and FNO files.
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
            eq = load_final_eq(path)
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

                values = evaluate_objective_safely(
                    objective = objective,
                    eq = eq,
                )

                summary = summarize_values(values)

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








#==============================================================================================================
# Public comparison functions:
#==============================================================================================================

def DESC_obj(
        paper : str,
        case : str,
    ):
    """
    Compares the general DESC objective values between OG, CHECK, and FNO files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(case_dir)

    rows = compare_objective_set(
        files = files,
        objective_getter = default_desc_objectives,
    )

    csv_path = case_dir / f"{case}_DESC_obj.csv"
    md_path = case_dir / f"{case}_DESC_obj.md"

    write_table_csv(
        rows = rows,
        path = csv_path,
    )

    write_table_markdown(
        rows = rows,
        path = md_path,
    )

    return rows





def case_obj(
        paper : str,
        case : str,
    ):
    """
    Compares paper/case-specific objective values between OG, CHECK, and FNO files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(case_dir)

    def get_case_objectives(eq):
        return CASE_OBJECTIVES.get(paper, {}).get(case, [])

    rows = compare_objective_set(
        files = files,
        objective_getter = get_case_objectives,
    )

    csv_path = case_dir / f"{case}_case_obj.csv"
    md_path = case_dir / f"{case}_case_obj.md"

    write_table_csv(
        rows = rows,
        path = csv_path,
    )

    write_table_markdown(
        rows = rows,
        path = md_path,
    )

    return rows








#==============================================================================================================
# Command-line interface:
#==============================================================================================================

def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper",
        required = True,
        help = "Paper name, e.g. dudt2024.",
    )

    parser.add_argument(
        "--case",
        required = True,
        help = "Case name, e.g. helical_qs.",
    )

    return parser.parse_args()





def main():
    """
    Runs both DESC and case-specific comparisons.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Comparing objectives for paper = {args.paper}, case = {args.case}")
    print("=" * 120 + "\n")

    DESC_obj(
        paper = args.paper,
        case = args.case,
    )

    case_obj(
        paper = args.paper,
        case = args.case,
    )

    case_dir = get_case_dir(
        paper = args.paper,
        case = args.case,
    )

    print("\nFinished.")
    print(f"Tables written to: {case_dir}\n")





if __name__ == "__main__":
    main()