# plot.py
#==============================================================================================================
#
# Plot case output comparisons from research/lit_comp/cases/<case>/<objective>/<run>/.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 plot.py --case ATF
#   python3 plot.py --case ATF --obj qs3
#
# This script plots:
#   1. INITIAL/FXD/FREE pressure profiles
#   2. INITIAL/FXD/FREE toroidal cross-section overlays
#
#==============================================================================================================

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from desc.grid import LinearGrid
from desc.plotting import plot_surfaces

try:
    from .helper import (
        find_h5_files,
        find_initial_h5_file,
        get_existing_objective_dirs,
        load_final_eq,
        normalize_case_name,
    )

except ImportError:
    from helper import (
        find_h5_files,
        find_initial_h5_file,
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
        initial_path = None,
    ):
    """
    Load INITIAL, FXD, and FREE pressure profiles for one output folder.
    """

    profiles = {}

    if initial_path is not None:
        eq = load_final_eq(
            path = initial_path,
        )

        rho, pressure = compute_radial_profile(
            eq = eq,
            quantity = "p",
        )

        profiles["INITIAL"] = (
            rho,
            pressure,
        )

    for variant in (
        "FXD",
        "FREE",
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

def get_line_styles():
    """
    Return line styles used consistently across pressure and surface plots.
    """

    return {
        "INITIAL": {
            "color": "black",
            "linestyle": "-",
            "linewidth": 2,
            "alpha": 0.38,
            "zorder": 1,
        },
        "FXD": {
            "color": "tab:green",
            "linestyle": "--",
            "linewidth": 1,
            "alpha": 0.82,
            "zorder": 2,
        },
        "FREE": {
            "color": "tab:purple",
            "linestyle": ":",
            "linewidth": 1,
            "alpha": 0.98,
            "zorder": 3,
        },
    }





def get_surface_plot_kwargs(
        label,
    ):
    """
    Return keyword arguments passed directly into DESC plot_surfaces.
    """

    style = get_line_styles()[label]

    return {
        "label": label,
        "rho_color": style["color"],
        "theta_color": style["color"],
        "lcfs_color": style["color"],
        "axis_color": style["color"],
        "rho_ls": style["linestyle"],
        "theta_ls": style["linestyle"],
        "lcfs_ls": style["linestyle"],
        "rho_lw": style["linewidth"],
        "theta_lw": style["linewidth"],
        "lcfs_lw": style["linewidth"],
        "axis_alpha": style["alpha"],
        "axis_size": 2.5,
    }





def apply_line_style(
        line,
        style,
    ):
    """
    Apply style settings to one matplotlib line.
    """

    line.set_color(
        style.get(
            "color",
            None,
        )
    )

    line.set_linestyle(
        style.get(
            "linestyle",
            "-",
        )
    )

    line.set_linewidth(
        style.get(
            "linewidth",
            1.8,
        )
    )

    line.set_alpha(
        style.get(
            "alpha",
            0.90,
        )
    )

    line.set_zorder(
        style.get(
            "zorder",
            2,
        )
    )

    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")





def apply_collection_style(
        collection,
        style,
    ):
    """
    Apply style settings to one matplotlib collection.
    """

    color = style.get(
        "color",
        None,
    )

    if color is not None:
        try:
            collection.set_color(color)
        except Exception:
            pass

        try:
            collection.set_edgecolor(color)
        except Exception:
            pass

        try:
            collection.set_edgecolors(color)
        except Exception:
            pass

    try:
        collection.set_facecolor("none")
    except Exception:
        pass

    try:
        collection.set_linestyle(
            style.get(
                "linestyle",
                "-",
            )
        )
    except Exception:
        pass

    try:
        collection.set_linestyles(
            style.get(
                "linestyle",
                "-",
            )
        )
    except Exception:
        pass

    try:
        collection.set_linewidth(
            style.get(
                "linewidth",
                1.8,
            )
        )
    except Exception:
        pass

    try:
        collection.set_alpha(
            style.get(
                "alpha",
                0.90,
            )
        )
    except Exception:
        pass

    try:
        collection.set_zorder(
            style.get(
                "zorder",
                2,
            )
        )
    except Exception:
        pass





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

    line_styles = get_line_styles()

    for label, (rho, values) in profiles.items():
        style = line_styles.get(
            label,
            {},
        )

        ax.plot(
            rho,
            values,
            label = label,
            color = style.get(
                "color",
                None,
            ),
            linestyle = style.get(
                "linestyle",
                "--",
            ),
            linewidth = style.get(
                "linewidth",
                2.25,
            ),
            alpha = style.get(
                "alpha",
                1.0,
            ),
            zorder = style.get(
                "zorder",
                2,
            ),
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





def style_new_surface_artists(
        ax,
        start_line_index,
        start_collection_index,
        start_patch_index,
        label,
    ):
    """
    Apply the plot style for one equilibrium to artists newly added by plot_surfaces.
    """

    line_styles = get_line_styles()
    style = line_styles.get(
        label,
        {},
    )

    new_lines = ax.lines[start_line_index:]
    new_collections = ax.collections[start_collection_index:]
    new_patches = ax.patches[start_patch_index:]

    for line in new_lines:
        apply_line_style(
            line = line,
            style = style,
        )

    for collection in new_collections:
        apply_collection_style(
            collection = collection,
            style = style,
        )

    for patch in new_patches:
        try:
            patch.set_edgecolor(
                style.get(
                    "color",
                    None,
                )
            )
            patch.set_facecolor("none")
            patch.set_linestyle(
                style.get(
                    "linestyle",
                    "-",
                )
            )
            patch.set_linewidth(
                style.get(
                    "linewidth",
                    1.8,
                )
            )
            patch.set_alpha(
                style.get(
                    "alpha",
                    0.90,
                )
            )
            patch.set_zorder(
                style.get(
                    "zorder",
                    2,
                )
            )
        except Exception:
            pass





def make_toroidal_legend_handles():
    """
    Build legend handles for the toroidal overlay plot.
    """

    handles = []
    line_styles = get_line_styles()

    for label in (
        "INITIAL",
        "FXD",
        "FREE",
    ):
        style = line_styles[label]

        handles.append(
            Line2D(
                [0],
                [0],
                label = label,
                color = style["color"],
                linestyle = style["linestyle"],
                linewidth = style["linewidth"],
                alpha = style["alpha"],
            )
        )

    return handles





def get_toroidal_phi_values(
        reference_eq,
        num_phi = 4,
        N_Xsec_index = 0,
        N_Xsecs = 1,
    ):
    """
    Return one interleaved toroidal angle set over one field period.
    """

    field_period = 2.0 * np.pi / reference_eq.NFP

    base_phi_values = np.linspace(
        0.0,
        field_period,
        num_phi,
        endpoint = False,
    )

    phi_offset = (
        float(N_Xsec_index)
        * field_period
        / float(num_phi * N_Xsecs)
    )

    phi_values = np.mod(
        base_phi_values + phi_offset,
        field_period,
    )

    return phi_values





def print_toroidal_phi_values(
        phi_values,
        reference_eq,
        N_Xsec_index = 0,
        N_Xsecs = 1,
    ):
    """
    Print the toroidal angles used in one cross-section figure.
    """

    field_period = 2.0 * np.pi / reference_eq.NFP

    phi_string = ", ".join(
        [
            f"{float(phi_value):.6f}"
            for phi_value in phi_values
        ]
    )

    normalized_string = ", ".join(
        [
            f"{float(phi_value / field_period):.6f}"
            for phi_value in phi_values
        ]
    )

    print("")
    print("----------------------------------------------------------------------------------------------------------------")
    print(f"Toroidal N-Xsec set {N_Xsec_index + 1:03d} / {N_Xsecs:03d}")
    print(f"phi [rad]            = {phi_string}")
    print(f"phi / (2*pi / NFP)   = {normalized_string}")
    print("----------------------------------------------------------------------------------------------------------------")
    print("")





def make_toroidal_cross_section_plot(
        equilibrium_data,
        title,
        save_path,
        rho = 8,
        theta = 8,
        num_phi = 4,
        N_Xsec_index = 0,
        N_Xsecs = 1,
    ):
    """
    Make and save one toroidal cross-section overlay figure.
    """

    if len(equilibrium_data) == 0:
        return None

    reference_eq = equilibrium_data[0][1]

    nrows = 2
    ncols = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize = (
            8.0,
            8.0,
        ),
        squeeze = False,
    )

    phi_values = get_toroidal_phi_values(
        reference_eq = reference_eq,
        num_phi = num_phi,
        N_Xsec_index = N_Xsec_index,
        N_Xsecs = N_Xsecs,
    )

    print_toroidal_phi_values(
        phi_values = phi_values,
        reference_eq = reference_eq,
        N_Xsec_index = N_Xsec_index,
        N_Xsecs = N_Xsecs,
    )

    for index, phi_value in enumerate(phi_values):
        row = index // ncols
        col = index % ncols
        ax = axes[row, col]

        for label, eq in equilibrium_data:
            start_line_index = len(ax.lines)
            start_collection_index = len(ax.collections)
            start_patch_index = len(ax.patches)

            plot_surfaces(
                eq,
                rho = rho,
                theta = theta,
                phi = float(phi_value),
                ax = np.atleast_1d(ax),
                **get_surface_plot_kwargs(
                    label = label,
                ),
            )

            style_new_surface_artists(
                ax = ax,
                start_line_index = start_line_index,
                start_collection_index = start_collection_index,
                start_patch_index = start_patch_index,
                label = label,
            )

        ax.set_title(
            rf"$\phi = {phi_value:.3f}$",
            fontsize = 10,
        )

        ax.set_aspect(
            "equal",
            adjustable = "box",
        )

    if N_Xsecs > 1:
        title = f"{title}: N-Xsec set {N_Xsec_index + 1:03d} of {N_Xsecs:03d}"

    fig.suptitle(
        title,
        fontsize = 12,
    )

    fig.legend(
        handles = make_toroidal_legend_handles(),
        loc = "upper center",
        ncol = 3,
        frameon = True,
        fontsize = 9,
        bbox_to_anchor = (
            0.5,
            0.965,
        ),
    )

    fig.tight_layout(
        rect = (
            0.02,
            0.02,
            1.00,
            0.93,
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

def get_all_run_dirs(
        objective_dir,
    ):
    """
    Return all numbered run directories within one objective folder.
    """

    run_dirs = sorted(
        [
            path
            for path in Path(objective_dir).iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key = lambda path: int(path.name),
    )

    if len(run_dirs) == 0:
        raise FileNotFoundError(
            f"No numbered run folders were found in: {objective_dir}"
        )

    return run_dirs





def get_plot_stem(
        case,
        objective_label,
        output_dir,
    ):
    """
    Return a file stem for plot outputs.
    """

    case_name = normalize_case_name(
        case = case,
    )

    if output_dir.name.isdigit():
        return f"{case_name}_{objective_label}_{output_dir.name}"

    return f"{case_name}_{objective_label}"





def plot_pressure_profiles(
        case,
        objective_label,
        output_dir,
        files,
        initial_path,
    ):
    """
    Plot INITIAL, FXD, and FREE pressure profiles for one output folder.
    """

    profiles = load_pressure_profiles(
        files = files,
        initial_path = initial_path,
    )

    plot_stem = get_plot_stem(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
    )

    save_path = output_dir / f"{plot_stem}_pressure.png"

    saved = make_pressure_plot(
        profiles = profiles,
        title = f"{plot_stem}: pressure profiles",
        save_path = save_path,
    )

    if saved is None:
        return []

    return [saved]





def plot_toroidal_cross_sections(
        case,
        objective_label,
        output_dir,
        files,
        initial_path,
        N_Xsecs = 1,
    ):
    """
    Plot INITIAL, FXD, and FREE toroidal cross-section overlays.
    """

    equilibrium_data = []

    if initial_path is not None:
        try:
            equilibrium_data.append(
                (
                    "INITIAL",
                    load_final_eq(
                        path = initial_path,
                    ),
                )
            )

        except Exception as error:
            print("")
            print(f"Skipping toroidal cross-sections for {objective_label} INITIAL: {error}")
            print("")

    for variant in (
        "FXD",
        "FREE",
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

    plot_stem = get_plot_stem(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
    )

    saved_paths = []

    for N_Xsec_index in range(N_Xsecs):
        if N_Xsecs == 1:
            save_path = output_dir / f"{plot_stem}_toroidal_cross_sections.png"

        else:
            save_path = output_dir / f"{plot_stem}_toroidal_cross_sections_N_Xsec_{N_Xsec_index + 1:03d}.png"

        saved = make_toroidal_cross_section_plot(
            equilibrium_data = equilibrium_data,
            title = f"{plot_stem}: toroidal cross-sections",
            save_path = save_path,
            num_phi = 4,
            N_Xsec_index = N_Xsec_index,
            N_Xsecs = N_Xsecs,
        )

        if saved is not None:
            saved_paths.append(saved)

    return saved_paths





def plot_objective_run_folder(
        case,
        objective_dir,
        output_dir,
        N_Xsecs = 1,
    ):
    """
    Plot pressure profiles and toroidal cross-sections for one numbered run folder.
    """

    objective_label = objective_dir.name

    files = find_h5_files(
        case_dir = output_dir,
    )

    if len(files) == 0:
        print("")
        print(f"No *_FXD.h5 or *_FREE.h5 files were found in: {output_dir}")
        print("")
        return []

    initial_path = find_initial_h5_file(
        run_dir = output_dir,
        case = case,
    )

    saved_paths = []

    print("")
    print("----------------------------------------------------------------------------------------------------------------")
    print(f"Plotting objective = {objective_label}, run = {output_dir.name}")
    print("----------------------------------------------------------------------------------------------------------------")
    print("")

    saved_paths += plot_pressure_profiles(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
        files = files,
        initial_path = initial_path,
    )

    saved_paths += plot_toroidal_cross_sections(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
        files = files,
        initial_path = initial_path,
        N_Xsecs = N_Xsecs,
    )

    return saved_paths





def plot_case(
        case,
        obj = None,
        N_Xsecs = 1,
    ):
    """
    Plot relevant comparisons for one case across all numbered run folders.
    """

    objective_dirs = get_existing_objective_dirs(
        case = case,
        obj = obj,
    )

    saved_paths = []

    for objective_dir in objective_dirs:
        run_dirs = get_all_run_dirs(
            objective_dir = objective_dir,
        )

        print("")
        print("================================================================================================================")
        print(f"Objective folder = {objective_dir.name}")
        print("Run folders to plot:")

        for run_dir in run_dirs:
            print(f"  {run_dir.name}")

        print("================================================================================================================")
        print("")

        for output_dir in run_dirs:
            saved_paths += plot_objective_run_folder(
                case = case,
                objective_dir = objective_dir,
                output_dir = output_dir,
                N_Xsecs = N_Xsecs,
            )

    if len(saved_paths) == 0:
        raise FileNotFoundError(
            "No plots were generated because no *_FXD.h5 or *_FREE.h5 files were found."
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
        description = "Plot DESC case FXD/FREE comparisons.",
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
            "balloon",
        ],
        help = "Optional objective folder to plot. If omitted, qs3 and balloon are plotted.",
    )

    parser.add_argument(
        "--N-Xsecs",
        type = int,
        default = 1,
        help = "Number of interleaved four-cut toroidal cross-section figures to save.",
    )

    args = parser.parse_args()

    if args.N_Xsecs < 1:
        parser.error("--N-Xsecs must be at least 1.")

    return args





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

    print("Run folders = all numbered runs")
    print(f"Toroidal N-Xsec sets = {args.N_Xsecs}")

    print("================================================================================================================")
    print("")

    saved_paths = plot_case(
        case = args.case,
        obj = args.obj,
        N_Xsecs = args.N_Xsecs,
    )

    print("")
    print("Saved plots:")
    print("")

    for path in saved_paths:
        print(path)

    print("")





if __name__ == "__main__":
    main()