# compare.py
#==============================================================================================================
#
# Compare paper/case-specific objective outputs between OG, FXD, and FREE files.
# Also owns plotting now that plot.py has been removed.
#
# Usage:
#   python3 research/lit_comp/compare.py --paper dudt2024 --case helical_qs
#   python3 research/lit_comp/compare.py --paper dudt2024 --case helical_qs --plot
#
#==============================================================================================================

import argparse
import traceback

import matplotlib.pyplot as plt
import numpy as np

from desc.grid import LinearGrid

from helper import (
    compare_objective_set,
    dudt2024_helical_qs_objectives,
    find_h5_files,
    get_case_dir,
    load_final_eq,
    write_table_csv,
)










#==============================================================================================================
# Case-Specific Objective Registry
#==============================================================================================================

CASE_OBJECTIVES = {
    "dudt2024": {
        "helical_qs": dudt2024_helical_qs_objectives,
    },
}










#==============================================================================================================
# Case Objective Comparison
#==============================================================================================================

def case_obj(
        paper : str,
        case : str,
    ):
    """
    Compares paper/case-specific objective values between OG, FXD, and FREE files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(
        case_dir = case_dir,
    )

    def get_case_objectives(eq):
        objective_getter = CASE_OBJECTIVES.get(paper, {}).get(case, None)

        if objective_getter is None:
            return []

        return objective_getter(eq)

    rows = compare_objective_set(
        files = files,
        objective_getter = get_case_objectives,
    )

    csv_path = case_dir / f"{case}_case_obj.csv"

    write_table_csv(
        rows = rows,
        path = csv_path,
    )

    return rows, csv_path










#==============================================================================================================
# Plot Helpers
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





def plot_pressure(
        paper : str,
        case : str,
    ):
    """
    Plots pressure profiles for OG, FXD, and FREE files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(
        case_dir = case_dir,
    )

    plt.figure()

    plotted_anything = False

    for label, path in files.items():
        if path is None:
            print(f"Skipping missing {label} file.")
            continue

        try:
            eq = load_final_eq(
                path = path,
            )

            rho, pressure = compute_pressure_profile(
                eq = eq,
            )

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
        plot_folder : str,
    ):
    """
    Paper/case/folder-specific plot router.

    Add folder-specific branches here as needed.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(
        case_dir = case_dir,
    )

    if paper == "dudt2024" and case == "helical_qs" and plot_folder == "pressure":
        pressure_path = plot_pressure(
            paper = paper,
            case = case,
        )

        if pressure_path is None:
            return []

        return [pressure_path]

    if plot_folder == "pressure":
        pressure_path = plot_pressure(
            paper = paper,
            case = case,
        )

        if pressure_path is None:
            return []

        return [pressure_path]

    print("")
    print(f"No plot branch defined for paper = {paper}, case = {case}, plot_folder = {plot_folder}.")
    print("")

    return []










#==============================================================================================================
# Command-Line Interface
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

    parser.add_argument(
        "--plot",
        action = "store_true",
        help = "Also generate plots.",
    )

    parser.add_argument(
        "--plot-folder",
        default = "pressure",
        help = "Plot branch/folder key to route through plot_case, e.g. pressure.",
    )

    return parser.parse_args()





def main():
    """
    Runs case-specific objective comparison and optional plots.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Comparing case objectives for paper = {args.paper}, case = {args.case}")
    print("=" * 120 + "\n")

    rows, csv_path = case_obj(
        paper = args.paper,
        case = args.case,
    )

    print("")
    print("Finished case-objective comparison.")
    print(f"CSV written to: {csv_path}")

    if args.plot:
        print("")
        print("=" * 120)
        print(f"Plotting for paper = {args.paper}, case = {args.case}, plot_folder = {args.plot_folder}")
        print("=" * 120)
        print("")

        paths = plot_case(
            paper = args.paper,
            case = args.case,
            plot_folder = args.plot_folder,
        )

        if len(paths) == 0:
            print("No plots written.")

        for path in paths:
            print(f"Plot written to: {path}")

    print()





if __name__ == "__main__":
    main()