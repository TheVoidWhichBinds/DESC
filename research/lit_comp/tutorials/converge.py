"""Build convergence tables for tutorial optimization runs.

Run this file from either:
    1. the ``research/lit_comp/tutorials`` folder, or
    2. anywhere else, using ``python converge.py --root /path/to/tutorials``.

The script searches:
    tutorials/basic_qs/<optimization>/<run>/*FXD_result_summary.json
    tutorials/basic_qs/<optimization>/<run>/*FREE_result_summary.json
    tutorials/balloon/<optimization>/<run>/*FXD_result_summary.json
    tutorials/balloon/<optimization>/<run>/*FREE_result_summary.json

For each optimization folder, it writes:
    <optimization>_convergence_table.csv

For example:
    basic_qs/tripleQS/basic_qs_tripleQS_convergence_table.csv
    basic_qs/twotermQH/basic_qs_twotermQH_convergence_table.csv
    balloon/balloon/balloon_balloon_convergence_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METHODS = (
    "FXD",
    "FREE",
)

TUTORIAL_FOLDERS = (
    "basic_qs",
    "balloon",
)

CONDITION_NAMES = (
    "ftol",
    "xtol",
    "gtol",
)


#==========
# PATH HELPERS
#==========

def get_tutorial_root(input_root: str | None = None) -> Path:
    """Return the tutorials folder path."""

    if input_root is not None:
        return Path(input_root).expanduser().resolve()

    return Path(__file__).resolve().parent





def is_run_folder(path: Path) -> bool:
    """Return True if a folder name matches the run-number format."""

    return path.is_dir() and re.fullmatch(r"\d{3}", path.name) is not None





def find_summary_json(run_folder: Path, method: str) -> Path | None:
    """Find the result summary JSON for a given run folder and method."""

    exact_name = f"{method}_result_summary.json"
    exact_path = run_folder / exact_name

    if exact_path.exists():
        return exact_path

    matches = sorted(run_folder.glob(f"*{method}_result_summary.json"))

    if not matches:
        return None

    return matches[0]





#==========
# MESSAGE PARSING
#==========

def read_optimizer_message(summary_path: Path | None) -> str:
    """Read the optimizer message from a result-summary JSON file."""

    if summary_path is None:
        return ""

    with summary_path.open("r", encoding = "utf-8") as file:
        data: dict[str, Any] = json.load(file)

    result = data.get("result", {})

    if not isinstance(result, dict):
        return ""

    message = result.get("message", "")

    if message is None:
        return ""

    return str(message)





def parse_condition(message: str) -> str:
    """Extract ftol, xtol, or gtol from the optimizer termination message."""

    if not message:
        return "missing"

    lower_message = message.lower()

    for condition_name in CONDITION_NAMES:
        if f"`{condition_name}`" in lower_message:
            return condition_name

        if f'"{condition_name}"' in lower_message:
            return condition_name

        if condition_name in lower_message:
            return condition_name

    return "unknown"





def parse_converged(message: str) -> bool:
    """Return True when ftol, xtol, or gtol appears in the optimizer message."""

    return parse_condition(message) in CONDITION_NAMES





def parse_message(message: str) -> tuple[bool, str]:
    """Parse convergence status and stopping condition from a message."""

    condition = parse_condition(message)
    converged = condition in CONDITION_NAMES

    if not converged and condition == "unknown":
        condition = "not converged"

    return converged, condition





#==========
# TABLE BUILDING
#==========

def build_table_for_optimization(optimization_folder: Path) -> list[dict[str, str]]:
    """Build the convergence table rows for one optimization folder."""

    rows: list[dict[str, str]] = []

    run_folders = sorted(
        folder for folder in optimization_folder.iterdir()
        if is_run_folder(folder)
    )

    for run_folder in run_folders:
        row = {
            "Run": run_folder.name,
        }

        for method in METHODS:
            summary_path = find_summary_json(run_folder, method)
            message = read_optimizer_message(summary_path)
            converged, condition = parse_message(message)

            row[f"{method} Converged"] = str(converged)
            row[f"{method} Condition"] = condition

        rows.append(row)

    return rows





def write_csv_table(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write convergence table rows to a CSV file."""

    fieldnames = [
        "Run",
        "FXD Converged",
        "FXD Condition",
        "FREE Converged",
        "FREE Condition",
    ]

    with output_path.open("w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)





def print_markdown_table(title: str, rows: list[dict[str, str]]) -> None:
    """Print a small markdown version of a convergence table."""

    print()
    print(f"## {title}")
    print()
    print("| Run | FXD Converged | FXD Condition | FREE Converged | FREE Condition |")
    print("|---:|:---:|:---:|:---:|:---:|")

    for row in rows:
        print(
            f"| {row['Run']} "
            f"| {row['FXD Converged']} "
            f"| {row['FXD Condition']} "
            f"| {row['FREE Converged']} "
            f"| {row['FREE Condition']} |"
        )





#==========
# DRIVER
#==========

def main() -> None:
    """Create convergence tables for all tutorial optimization folders."""

    parser = argparse.ArgumentParser(
        description = "Create convergence tables from tutorial result-summary JSON files."
    )
    parser.add_argument(
        "--root",
        default = None,
        help = "Path to the tutorials folder. Defaults to the folder containing converge.py.",
    )
    parser.add_argument(
        "--no-print",
        action = "store_true",
        help = "Write CSV files without printing markdown tables.",
    )

    args = parser.parse_args()

    tutorial_root = get_tutorial_root(args.root)

    if not tutorial_root.exists():
        raise FileNotFoundError(f"Tutorial root does not exist: {tutorial_root}")

    for tutorial_name in TUTORIAL_FOLDERS:
        tutorial_folder = tutorial_root / tutorial_name

        if not tutorial_folder.exists():
            continue

        optimization_folders = sorted(
            folder for folder in tutorial_folder.iterdir()
            if folder.is_dir()
        )

        for optimization_folder in optimization_folders:
            rows = build_table_for_optimization(optimization_folder)

            if not rows:
                continue

            output_path = (
                optimization_folder
                / f"{tutorial_name}_{optimization_folder.name}_convergence_table.csv"
            )

            write_csv_table(rows, output_path)

            if not args.no_print:
                title = f"{tutorial_name}/{optimization_folder.name}"
                print_markdown_table(title, rows)

            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
