# plot.py
#==============================================================================================================
#
# Plot case output comparisons from research/lit_comp/cases/<case>/<objective>/.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 plot.py --case ATF
#   python3 plot.py --case ATF --obj qs3
#
# This script only plots:
#   1. FLUX/PRESS pressure profiles
#   2. FLUX/PRESS toroidal cross-sections
#
#==============================================================================================================

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from desc.grid import LinearGrid
from desc.plotting import plot_surfaces

try:
    from .helper import (
        find_h5_files,
        get_existing_objective_dirs,
        load_final_eq,
        normalize_case_name,
    )

except ImportError:
    from helper import (
        find_h5_files,
        get_existing_objective_dirs,
        load_final_eq,
        normalize_case_name,
    )










#========================================================================================================================================
# PROFILE HELPERS
#========================================================================================================================================

def compute_radial_profile(
        eq,
        quantity,
        num_points = 200,
    ):
    """
    Compute a radial profile for a scalar quantity.
    """

    rho = np.linspace(
        0.0,
        1.0,
        num_points,
    )

    grid = LinearGrid(
        rho = rho,
        M = 0,
        N = 0,
        NFP = eq.NFP,
    )

    data = eq.compute(
        quantity,
        grid = grid,
    )

    values = np.asarray(data[quantity]).reshape(-1)

    return rho, values





def load_pressure_profiles(
        files,
    ):
    """
    Load FLUX and PRESS pressure profiles for one objective folder.
    """

    profiles = {}

    for variant in (
        "FLUX",
        "PRESS",
    ):
        path = files.get(
            variant,
            None,
        )

        if path is None:
            continue

        eq = load_final_eq(
            path = path,
        )

        rho, pressure = compute_radial_profile(
            eq = eq,
            quantity = "p",
        )

        profiles[variant] = (
            rho,
            pressure,
        )

    return profiles










#========================================================================================================================================
# PLOTTING HELPERS
#========================================================================================================================================

def make_pressure_plot(
        profiles,
        title,
        save_path,
    ):
    """
    Make and save one pressure-profile comparison plot.
    """

    if len(profiles) == 0:
        return None

    fig, ax = plt.subplots(
        figsize = (9, 6),
    )

    dash_styles = {
        "FLUX": (
            0,
            (
                6,
                3,
            ),
        ),
        "PRESS": (
            2,
            (
                10,
                3,
            ),
        ),
    }

    for label, (rho, values) in profiles.items():
        linestyle = dash_styles.get(
            label,
            "--",
        )

        ax.plot(
            rho,
            values,
            label = label,
            linestyle = linestyle,
            linewidth = 2.25,
        )

    ax.set_xlabel("rho")
    ax.set_ylabel("p")
    ax.set_title(title)

    ax.grid(
        True,
        alpha = 0.30,
    )

    ax.legend(
        fontsize = 9,
        loc = "best",
        frameon = True,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight",
    )

    plt.close(fig)

    return save_path





def make_toroidal_cross_section_plot(
        equilibrium_data,
        title,
        save_path,
        rho = 8,
        theta = 8,
        num_phi = 6,
    ):
    """
    Make and save one toroidal cross-section figure for one objective folder.
    """

    if len(equilibrium_data) == 0:
        return None

    fig, axes = plt.subplots(
        len(equilibrium_data),
        num_phi,
        figsize = (
            3.0 * num_phi,
            3.4 * len(equilibrium_data),
        ),
        squeeze = False,
    )

    for row, (variant, eq) in enumerate(equilibrium_data):
        phi_values = np.linspace(
            0.0,
            2.0 * np.pi / eq.NFP,
            num_phi,
            endpoint = False,
        )

        for col, phi_value in enumerate(phi_values):
            ax = axes[row, col]

            plot_surfaces(
                eq,
                rho = rho,
                theta = theta,
                phi = float(phi_value),
                ax = np.atleast_1d(ax),
            )

            if row == 0:
                ax.set_title(
                    rf"$\phi = {phi_value:.3f}$",
                    fontsize = 10,
                )

            if col == 0:
                ax.text(
                    -0.20,
                    0.50,
                    variant,
                    transform = ax.transAxes,
                    rotation = 90,
                    va = "center",
                    ha = "center",
                    fontsize = 11,
                )

    fig.suptitle(
        title,
        fontsize = 12,
    )

    fig.tight_layout(
        rect = (
            0.02,
            0.02,
            1.00,
            0.94,
        ),
    )

    fig.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight",
    )

    plt.close(fig)

    return save_path










#========================================================================================================================================
# Case Plotters
#========================================================================================================================================

def plot_pressure_profiles(
        case,
        objective_label,
        objective_dir,
        files,
    ):
    """
    Plot FLUX and PRESS pressure profiles for one objective folder.
    """

    profiles = load_pressure_profiles(
        files = files,
    )

    case_name = normalize_case_name(
        case = case,
    )

    save_path = objective_dir / f"{case_name}_{objective_label}_pressure.png"

    saved = make_pressure_plot(
        profiles = profiles,
        title = f"{case_name} {objective_label}: pressure profiles",
        save_path = save_path,
    )

    if saved is None:
        return []

    return [saved]





def plot_toroidal_cross_sections(
        case,
        objective_label,
        objective_dir,
        files,
    ):
    """
    Plot FLUX and PRESS toroidal cross-sections for one objective folder.
    """

    equilibrium_data = []

    for variant in (
        "FLUX",
        "PRESS",
    ):
        path = files.get(
            variant,
            None,
        )

        if path is None:
            continue

        try:
            eq = load_final_eq(
                path = path,
            )

        except Exception as error:
            print("")
            print(f"Skipping toroidal cross-sections for {objective_label} {variant}: {error}")
            print("")
            continue

        equilibrium_data.append(
            (
                variant,
                eq,
            )
        )

    if len(equilibrium_data) == 0:
        return []

    case_name = normalize_case_name(
        case = case,
    )

    save_path = objective_dir / f"{case_name}_{objective_label}_toroidal_cross_sections.png"

    saved = make_toroidal_cross_section_plot(
        equilibrium_data = equilibrium_data,
        title = f"{case_name} {objective_label}: toroidal cross-sections",
        save_path = save_path,
        num_phi = 6,
    )

    if saved is None:
        return []

    return [saved]





def plot_objective_folder(
        case,
        objective_dir,
    ):
    """
    Plot pressure profiles and toroidal cross-sections for one objective folder.
    """

    objective_label = objective_dir.name

    files = find_h5_files(
        case_dir = objective_dir,
    )

    if len(files) == 0:
        print("")
        print(f"No *_FLUX.h5 or *_PRESS.h5 files were found in: {objective_dir}")
        print("")
        return []

    saved_paths = []

    saved_paths += plot_pressure_profiles(
        case = case,
        objective_label = objective_label,
        objective_dir = objective_dir,
        files = files,
    )

    saved_paths += plot_toroidal_cross_sections(
        case = case,
        objective_label = objective_label,
        objective_dir = objective_dir,
        files = files,
    )

    return saved_paths





def plot_case(
        case,
        obj = None,
    ):
    """
    Plot relevant comparisons for one case.
    """

    objective_dirs = get_existing_objective_dirs(
        case = case,
        obj = obj,
    )

    saved_paths = []

    for objective_dir in objective_dirs:
        saved_paths += plot_objective_folder(
            case = case,
            objective_dir = objective_dir,
        )

    if len(saved_paths) == 0:
        raise FileNotFoundError(
            "No plots were generated because no *_FLUX.h5 or *_PRESS.h5 files were found."
        )

    return saved_paths










#========================================================================================================================================
# CLI
#========================================================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = ArgumentParser(
        description = "Plot DESC case FLUX/PRESS comparisons.",
    )

    parser.add_argument(
        "--case",
        required = True,
        help = "Case name, e.g. ATF, HELIOTRON, NCSX, W7-X.",
    )

    parser.add_argument(
        "--obj",
        default = None,
        choices = [
            "qs3",
            "iso",
            "balloon",
            "all",
        ],
        help = "Optional objective folder to plot. If omitted, all four folders are plotted.",
    )

    return parser.parse_args()





def main():
    """
    CLI entry point.
    """

    args = parse_args()

    print("")
    print("================================================================================================================")
    print(f"Generating plots for case = {args.case}")

    if args.obj is not None:
        print(f"Objective folder = {args.obj}")

    print("================================================================================================================")
    print("")

    saved_paths = plot_case(
        case = args.case,
        obj = args.obj,
    )

    print("")
    print("Saved plots:")
    print("")

    for path in saved_paths:
        print(path)

    print("")





if __name__ == "__main__":
    main()
