# plot.py
#==============================================================================================================
#
# Plot comparisons between OG, CHECK, and FNO files.
#
# Usage:
#   python3 research/lit_comp/plot.py --paper dudt2024 --case helical_qs
#
#==============================================================================================================

import argparse
import traceback

import numpy as np
import matplotlib.pyplot as plt

from desc.grid import LinearGrid

from helper import (
    find_h5_files,
    get_case_dir,
    load_final_eq,
)










#==============================================================================================================
# Plot helpers:
#==============================================================================================================

def compute_pressure_profile(
        eq,
        num_rho : int = 200,
    ):
    """
    Computes pressure on a radial grid.
    """

    rho = np.linspace(0.0, 1.0, num_rho)

    grid = LinearGrid(
        rho = rho,
        M = 0,
        N = 0,
        NFP = eq.NFP,
    )

    data = eq.compute(
        "p",
        grid = grid,
    )

    pressure = np.asarray(data["p"]).reshape(-1)

    return rho, pressure








#==============================================================================================================
# Public plotting functions:
#==============================================================================================================

def plot_pressure(
        paper : str,
        case : str,
    ):
    """
    Plots pressure profiles for OG, CHECK, and FNO files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(case_dir)

    plt.figure()

    plotted_anything = False

    for label, path in files.items():
        if path is None:
            print(f"Skipping missing {label} file.")
            continue

        try:
            eq = load_final_eq(path)

            rho, pressure = compute_pressure_profile(eq)

            plt.plot(
                rho,
                pressure,
                label = label,
            )

            plotted_anything = True

        except Exception:
            print(f"\nFailed while plotting pressure for {label}: {path}")
            traceback.print_exc()

    if not plotted_anything:
        print("No pressure profiles plotted.")
        return None

    plt.xlabel(r"$\rho$")
    plt.ylabel("Pressure")
    plt.title(f"{paper}: {case} pressure profile")
    plt.legend()
    plt.tight_layout()

    output_path = case_dir / f"{case}_pressure.png"

    plt.savefig(
        output_path,
        dpi = 300,
    )

    plt.close()

    return output_path





def plot_case(
        paper : str,
        case : str,
    ):
    """
    Placeholder for paper/case-specific plots.

    Fill this in later with the plots from the original paper.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(case_dir)

    if paper == "dudt2024" and case == "helical_qs":
        # Fill this in later.
        return []

    return []










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
    Runs all plots for the given paper/case.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Plotting for paper = {args.paper}, case = {args.case}")
    print("=" * 120 + "\n")

    pressure_path = plot_pressure(
        paper = args.paper,
        case = args.case,
    )

    case_paths = plot_case(
        paper = args.paper,
        case = args.case,
    )

    case_dir = get_case_dir(
        paper = args.paper,
        case = args.case,
    )

    print("\nFinished.")
    print(f"Plots written to: {case_dir}")

    if pressure_path is not None:
        print(f"Pressure plot: {pressure_path}")

    for path in case_paths:
        print(f"Case plot: {path}")

    print()





if __name__ == "__main__":
    main()