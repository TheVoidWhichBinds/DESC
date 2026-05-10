# plot.py
#==============================================================================================================
#
# Plot case output comparisons from research/lit_comp/cases/<case>/<objective>/<run>/.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 plot.py --case ATF
#   python3 plot.py --case ATF --obj qs3
#   python3 plot.py --case ATF --obj balloon --run 001
#
# This script plots:
#   1. INITIAL/FLUX/PRESS pressure profiles
#   2. INITIAL/FLUX/PRESS toroidal cross-section overlays
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
        get_latest_run_dir,
        load_final_eq,
        normalize_case_name,
    )

except ImportError:
    from helper import (
        find_h5_files,
        find_initial_h5_file,
        get_existing_objective_dirs,
        get_latest_run_dir,
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
    Load INITIAL, FLUX, and PRESS pressure profiles for one output folder.
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
        "FLUX": {
            "color": "tab:red",
            "linestyle": "--",
            "linewidth": 1,
            "alpha": 0.82,
            "zorder": 2,
        },
        "PRESS": {
            "color": "tab:green",
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
        "FLUX",
        "PRESS",
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





def make_toroidal_cross_section_plot(
        equilibrium_data,
        title,
        save_path,
        rho = 8,
        theta = 8,
        num_phi = 6,
    ):
    """
    Make and save one toroidal cross-section overlay figure.
    """

    if len(equilibrium_data) == 0:
        return None

    reference_eq = equilibrium_data[0][1]

    nrows = 3
    ncols = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize = (
            8.0,
            10.5,
        ),
        squeeze = False,
    )

    phi_values = np.linspace(
        0.0,
        2.0 * np.pi / reference_eq.NFP,
        num_phi,
        endpoint = False,
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

def get_plot_output_dir(
        objective_dir,
        run = None,
    ):
    """
    Return the output directory to plot for one objective folder.
    """

    if run is None:
        return get_latest_run_dir(
            objective_dir = objective_dir,
        )

    run_label = f"{int(run):03d}"
    run_dir = Path(objective_dir) / run_label

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Requested run folder does not exist: {run_dir}"
        )

    return run_dir





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
    Plot INITIAL, FLUX, and PRESS pressure profiles for one output folder.
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
    ):
    """
    Plot INITIAL, FLUX, and PRESS toroidal cross-section overlays.
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

    plot_stem = get_plot_stem(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
    )

    save_path = output_dir / f"{plot_stem}_toroidal_cross_sections.png"

    saved = make_toroidal_cross_section_plot(
        equilibrium_data = equilibrium_data,
        title = f"{plot_stem}: toroidal cross-sections",
        save_path = save_path,
        num_phi = 6,
    )

    if saved is None:
        return []

    return [saved]





def plot_objective_folder(
        case,
        objective_dir,
        run = None,
    ):
    """
    Plot pressure profiles and toroidal cross-sections for one objective output folder.
    """

    objective_label = objective_dir.name

    output_dir = get_plot_output_dir(
        objective_dir = objective_dir,
        run = run,
    )

    files = find_h5_files(
        case_dir = output_dir,
    )

    if len(files) == 0:
        print("")
        print(f"No *_FLUX.h5 or *_PRESS.h5 files were found in: {output_dir}")
        print("")
        return []

    initial_path = find_initial_h5_file(
        run_dir = output_dir,
        case = case,
    )

    saved_paths = []

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
    )

    return saved_paths





def plot_case(
        case,
        obj = None,
        run = None,
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
            run = run,
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
            "balloon",
        ],
        help = "Optional objective folder to plot. If omitted, qs3 and balloon are plotted.",
    )

    parser.add_argument(
        "--run",
        default = None,
        help = "Optional numbered run folder to plot, e.g. 001. If omitted, the latest run is plotted.",
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

    if args.run is not None:
        print(f"Run folder = {int(args.run):03d}")

    print("================================================================================================================")
    print("")

    saved_paths = plot_case(
        case = args.case,
        obj = args.obj,
        run = args.run,
    )

    print("")
    print("Saved plots:")
    print("")

    for path in saved_paths:
        print(path)

    print("")





if __name__ == "__main__":
    main()
