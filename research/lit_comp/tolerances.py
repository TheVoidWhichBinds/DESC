"""Create tolerance tables for tutorial and benchmark-case runs.

Place this file in:
    research/lit_comp/tolerances.py

Run from the lit_comp folder:
    python tolerances.py

Or run from anywhere:
    python tolerances.py --root /path/to/research/lit_comp

This creates two table types:

1. Tutorial tolerance tables
   Reads:
       tutorials/**/tolerance_case_report.json

   Uses:
       outer_tolerances: ftol, xtol, gtol
       inner_tolerances: ftol, xtol, gtol

   Writes one CSV table per optimization folder, for example:
       tutorials/basic_qs/tripleQS/basic_qs_tripleQS_tolerances.csv
       tutorials/basic_qs/twotermQH/basic_qs_twotermQH_tolerances.csv
       tutorials/balloon/balloon/balloon_balloon_tolerances.csv

2. Case tolerance tables
   Reads:
       cases/**/*.json files matching *_hyperparameters.json

   Uses:
       ftol, xtol, gtol, ctol

   Writes one CSV table per benchmark case, for example:
       cases/ARIES-CS/ARIES-CS_tolerances.csv
       cases/W7-X/W7-X_tolerances.csv
       cases/HELIOTRON/HELIOTRON_tolerances.csv

The case tables include an Objective column because each case can contain multiple
objective folders, such as qs3, balloon, and force.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TUTORIAL_COLUMNS = [
    "Run",
    "Outer ftol",
    "Outer xtol",
    "Outer gtol",
    "Inner ftol",
    "Inner xtol",
    "Inner gtol",
]

CASE_COLUMNS = [
    "Objective",
    "Run",
    "ftol",
    "xtol",
    "gtol",
    "ctol",
]


#==========
# PATH HELPERS
#==========

def get_lit_comp_root(input_root: str | None = None) -> Path:
    """Return the lit_comp folder path."""

    if input_root is not None:
        return Path(input_root).expanduser().resolve()

    return Path(__file__).resolve().parent





def is_run_folder(path: Path) -> bool:
    """Return True if a folder name matches the run-number format."""

    return path.is_dir() and re.fullmatch(r"\d{3}", path.name) is not None





def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""

    with path.open("r", encoding = "utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        return {}

    return data





def format_value(value: Any) -> str:
    """Format tolerance values compactly for CSV tables."""

    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.4e}"

    return str(value)





#==========
# TUTORIAL TABLES
#==========

def find_tutorial_optimization_folders(tutorials_root: Path) -> list[Path]:
    """Find folders that directly contain numbered tutorial run folders."""

    optimization_folders: list[Path] = []

    if not tutorials_root.exists():
        return optimization_folders

    for folder in sorted(tutorials_root.rglob("*")):
        if not folder.is_dir():
            continue

        run_folders = [
            child for child in folder.iterdir()
            if is_run_folder(child)
        ]

        if run_folders:
            optimization_folders.append(folder)

    return optimization_folders





def build_tutorial_table(optimization_folder: Path) -> list[dict[str, str]]:
    """Build one tutorial tolerance table from numbered run folders."""

    rows: list[dict[str, str]] = []

    run_folders = sorted(
        folder for folder in optimization_folder.iterdir()
        if is_run_folder(folder)
    )

    for run_folder in run_folders:
        report_path = run_folder / "tolerance_case_report.json"

        if not report_path.exists():
            continue

        data = read_json(report_path)

        outer_tolerances = data.get("outer_tolerances", {})
        inner_tolerances = data.get("inner_tolerances", {})

        if not isinstance(outer_tolerances, dict):
            outer_tolerances = {}

        if not isinstance(inner_tolerances, dict):
            inner_tolerances = {}

        rows.append(
            {
                "Run": run_folder.name,
                "Outer ftol": format_value(outer_tolerances.get("ftol")),
                "Outer xtol": format_value(outer_tolerances.get("xtol")),
                "Outer gtol": format_value(outer_tolerances.get("gtol")),
                "Inner ftol": format_value(inner_tolerances.get("ftol")),
                "Inner xtol": format_value(inner_tolerances.get("xtol")),
                "Inner gtol": format_value(inner_tolerances.get("gtol")),
            }
        )

    return rows





def tutorial_output_path(lit_comp_root: Path, optimization_folder: Path) -> Path:
    """Return output CSV path for a tutorial optimization table."""

    relative_parts = optimization_folder.relative_to(lit_comp_root / "tutorials").parts
    table_name = "_".join(relative_parts) + "_tolerances.csv"

    return optimization_folder / table_name





def write_tutorial_tables(lit_comp_root: Path, print_tables: bool = True) -> list[Path]:
    """Write all tutorial tolerance tables."""

    tutorials_root = lit_comp_root / "tutorials"
    output_paths: list[Path] = []

    for optimization_folder in find_tutorial_optimization_folders(tutorials_root):
        rows = build_tutorial_table(optimization_folder)

        if not rows:
            continue

        output_path = tutorial_output_path(lit_comp_root, optimization_folder)
        write_csv(rows = rows, fieldnames = TUTORIAL_COLUMNS, output_path = output_path)
        output_paths.append(output_path)

        if print_tables:
            title = str(optimization_folder.relative_to(tutorials_root))
            print_markdown_table(title = f"Tutorial: {title}", fieldnames = TUTORIAL_COLUMNS, rows = rows)

    return output_paths





#==========
# CASE TABLES
#==========

def find_case_folders(cases_root: Path) -> list[Path]:
    """Find benchmark case folders."""

    if not cases_root.exists():
        return []

    return sorted(
        folder for folder in cases_root.iterdir()
        if folder.is_dir()
    )





def build_case_table(case_folder: Path) -> list[dict[str, str]]:
    """Build one case tolerance table from all objective/run folders."""

    rows: list[dict[str, str]] = []

    hyperparameter_paths = sorted(case_folder.glob("*/*/*_hyperparameters.json"))

    for hyperparameter_path in hyperparameter_paths:
        run_folder = hyperparameter_path.parent

        if not is_run_folder(run_folder):
            continue

        objective_folder = run_folder.parent
        data = read_json(hyperparameter_path)

        rows.append(
            {
                "Objective": objective_folder.name,
                "Run": run_folder.name,
                "ftol": format_value(data.get("ftol")),
                "xtol": format_value(data.get("xtol")),
                "gtol": format_value(data.get("gtol")),
                "ctol": format_value(data.get("ctol")),
            }
        )

    rows.sort(key = lambda row: (row["Objective"], row["Run"]))

    return rows





def write_case_tables(lit_comp_root: Path, print_tables: bool = True) -> list[Path]:
    """Write all benchmark-case tolerance tables."""

    cases_root = lit_comp_root / "cases"
    output_paths: list[Path] = []

    for case_folder in find_case_folders(cases_root):
        rows = build_case_table(case_folder)

        if not rows:
            continue

        output_path = case_folder / f"{case_folder.name}_tolerances.csv"
        write_csv(rows = rows, fieldnames = CASE_COLUMNS, output_path = output_path)
        output_paths.append(output_path)

        if print_tables:
            print_markdown_table(title = f"Case: {case_folder.name}", fieldnames = CASE_COLUMNS, rows = rows)

    return output_paths





#==========
# OUTPUT HELPERS
#==========

def write_csv(rows: list[dict[str, str]], fieldnames: list[str], output_path: Path) -> None:
    """Write rows to a CSV file."""

    with output_path.open("w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)





def print_markdown_table(title: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Print a markdown table."""

    print()
    print(f"## {title}")
    print()

    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |"

    print(header)
    print(divider)

    for row in rows:
        print("| " + " | ".join(row.get(fieldname, "") for fieldname in fieldnames) + " |")





#==========
# DRIVER
#==========

def main() -> None:
    """Create tutorial and case tolerance tables."""

    parser = argparse.ArgumentParser(
        description = "Create tolerance tables for tutorials and benchmark cases."
    )
    parser.add_argument(
        "--root",
        default = None,
        help = "Path to the lit_comp folder. Defaults to the folder containing tolerances.py.",
    )
    parser.add_argument(
        "--type",
        choices = ["all", "tutorials", "cases"],
        default = "all",
        help = "Which table type to create.",
    )
    parser.add_argument(
        "--no-print",
        action = "store_true",
        help = "Write CSV files without printing markdown tables.",
    )

    args = parser.parse_args()

    lit_comp_root = get_lit_comp_root(args.root)

    if not lit_comp_root.exists():
        raise FileNotFoundError(f"lit_comp root does not exist: {lit_comp_root}")

    output_paths: list[Path] = []

    if args.type in {"all", "tutorials"}:
        output_paths.extend(
            write_tutorial_tables(
                lit_comp_root = lit_comp_root,
                print_tables = not args.no_print,
            )
        )

    if args.type in {"all", "cases"}:
        output_paths.extend(
            write_case_tables(
                lit_comp_root = lit_comp_root,
                print_tables = not args.no_print,
            )
        )

    print()
    print("Wrote tolerance tables:")

    for output_path in output_paths:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
