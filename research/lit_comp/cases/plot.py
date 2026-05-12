# plot.py
#==============================================================================================================
#
# Plot case output comparisons from research/lit_comp/cases/<case>/<objective>/<run>/.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 plot.py --case ARIES-CS
#   python3 plot.py --case ARIES-CS --obj qs3
#   python3 plot.py --case ARIES-CS --obj qs3 --N-Xsecs 3
#
# This script plots:
#   1. fixed-pressure/free-pressure pressure profiles
#   2. fixed-pressure/free-pressure toroidal cross-section overlays
#
#==============================================================================================================


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from desc.grid import LinearGrid
from desc.plotting import plot_comparison

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
# PLOT STYLE
#========================================================================================================================================
VARIANT_LABELS = {
    "FXD": "fixed-pressure",
    "FLUX": "fixed-pressure",
    "FREE": "free-pressure",
    "PRESS": "free-pressure",
}

CANONICAL_VARIANT_LABELS = {
    # Files may be named FXD/FREE or FLUX/PRESS, but all plot legends use
    # the thesis-facing construction names below.
    "FXD": "fixed-pressure",
    "FREE": "free-pressure",
}

VARIANT_FILE_SUFFIXES = {
    "FXD": (
        "_FXD.h5",
        "_FLUX.h5",
    ),
    "FREE": (
        "_FREE.h5",
        "_PRESS.h5",
    ),
}

VARIANT_DISPLAY_ORDER = (
    "FXD",
    "FREE",
)

DEFAULT_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

VARIANT_COLORS = {
    "FXD": DEFAULT_COLOR_CYCLE[0],
    "FLUX": DEFAULT_COLOR_CYCLE[0],
    "FREE": DEFAULT_COLOR_CYCLE[1],
    "PRESS": DEFAULT_COLOR_CYCLE[1],
}

OBJECTIVE_TITLES = {
    "qs3": "Quasi-Symmetry Triple Product",
    "balloon": "Infinite-n Ideal Ballooning Mode",
    "force": "Force Balance",
}








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
    Load fixed-pressure and free-pressure profiles for one output folder.
    """

    profiles = {}

    for variant in VARIANT_DISPLAY_ORDER:
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










def remove_existing_plot_file(
        save_path,
    ):
    """
    Remove an existing plot file before saving a newly generated plot.
    """

    save_path = Path(save_path)

    if save_path.exists():
        save_path.unlink()





#========================================================================================================================================
# PLOTTING HELPERS
#========================================================================================================================================
def get_case_plot_title(
        case,
        objective_label,
    ):
    """
    Return a publication-style title for one benchmark case/objective pair.
    """

    case_name = normalize_case_name(
        case = case,
    )

    objective_title = OBJECTIVE_TITLES.get(
        objective_label,
        objective_label,
    )

    return f"{case_name} {objective_title}"





def make_profile_plot(
        profiles,
        title,
        ylabel,
        save_path,
    ):
    """
    Make and save one radial profile comparison plot.

    Initial equilibria are intentionally not plotted. All curves are solid and
    colored consistently with the toroidal cut plots.
    """

    if len(profiles) == 0:
        return None

    fig, ax = plt.subplots(
        figsize = (9, 6),
    )

    for variant, (rho, values) in profiles.items():

        label = CANONICAL_VARIANT_LABELS.get(
            variant,
            variant,
        )

        color = VARIANT_COLORS.get(
            variant,
            None,
        )

        ax.plot(
            rho,
            values,
            label = label,
            color = color,
            linestyle = "-",
            linewidth = 2.35,
            alpha = 0.95,
        )

    ax.set_xlabel(
        r"Normalized radial toroidal flux coordinate $\rho$",
        fontsize = 14,
    )

    ax.set_ylabel(
        ylabel,
        fontsize = 14,
    )

    ax.set_title(
        title,
        fontsize = 16,
    )

    ax.tick_params(
        axis = "both",
        labelsize = 11,
    )

    ax.grid(
        True,
        alpha = 0.30,
    )

    ax.legend(
        fontsize = 10,
        loc = "best",
        frameon = True,
    )

    fig.tight_layout()

    remove_existing_plot_file(
        save_path = save_path,
    )

    fig.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight",
    )

    plt.close(fig)

    return save_path





def get_toroidal_phi_values(
        reference_eq,
        num_phi = 6,
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
        num_phi = 6,
        N_Xsec_index = 0,
        N_Xsecs = 1,
    ):
    """
    Make and save one toroidal cross-section figure.

    The figure has 6 toroidal cuts in a 2-by-3 layout. Fixed-pressure and
    free-pressure equilibria are overlaid using DESC plot_comparison.
    """

    if len(equilibrium_data) == 0:
        return None

    fig, axes = plt.subplots(
        2,
        3,
        figsize = (
            17,
            15,
        ),
        squeeze = False,
    )

    axes_flat = axes.reshape(-1)

    variants = [
        variant
        for variant, _ in equilibrium_data
    ]

    eqs = [
        eq
        for _, eq in equilibrium_data
    ]

    labels = [
        CANONICAL_VARIANT_LABELS.get(
            variant,
            variant,
        )
        for variant in variants
    ]

    colors = [
        VARIANT_COLORS.get(
            variant,
            None,
        )
        for variant in variants
    ]

    linewidths = [
        1.55
        for _ in eqs
    ]

    linestyles = [
        "-"
        for _ in eqs
    ]

    reference_eq = eqs[0]

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

    plot_comparison(
        eqs,
        rho = rho,
        theta = theta,
        phi = phi_values,
        ax = axes_flat,
        color = colors,
        labels = labels,
        lw = linewidths,
        ls = linestyles,
        legend = False,
    )

    axis_padding = 0.08

    for ax, phi_value in zip(axes_flat, phi_values):

        ax.set_title(
            rf"$\phi = {phi_value:.2f}$",
            fontsize = 22,
            y = 1.035,
        )

        ax.set_xlabel(
            "R [m]",
            fontsize = 18,
        )

        ax.set_ylabel(
            "Z [m]",
            fontsize = 18,
        )

        ax.tick_params(
            axis = "both",
            labelsize = 16,
        )

        ax.grid(
            True,
            alpha = 0.25,
        )

        ax.set_box_aspect(
            1.0,
        )

    for ax in axes_flat:

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        x_center = 0.5 * (
            xmin + xmax
        )

        y_center = 0.5 * (
            ymin + ymax
        )

        x_width = xmax - xmin
        y_width = ymax - ymin

        half_width = 0.5 * max(
            x_width,
            y_width,
        ) * (
            1.0 + axis_padding
        )

        ax.set_xlim(
            x_center - half_width,
            x_center + half_width,
        )

        ax.set_ylim(
            y_center - half_width,
            y_center + half_width,
        )

        ax.set_box_aspect(
            1.0,
        )

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()

    suptitle_y = 0.985
    legend_y = suptitle_y - 0.060

    if len(handles) > 0:
        fig.legend(
            handles = handles,
            labels = legend_labels,
            loc = "upper center",
            ncol = len(handles),
            fontsize = 20,
            frameon = True,
            bbox_to_anchor = (
                0.5,
                legend_y,
            ),
        )

    if N_Xsecs > 1:
        title = f"{title}: N-Xsec set {N_Xsec_index + 1:03d} of {N_Xsecs:03d}"

    fig.suptitle(
        title,
        fontsize = 28,
        y = suptitle_y,
    )

    fig.subplots_adjust(
        left = 0.065,
        right = 0.985,
        bottom = 0.075,
        top = 0.825,
        wspace = 0.28,
        hspace = -0.4,
    )

    remove_existing_plot_file(
        save_path = save_path,
    )

    fig.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight",
    )

    plt.close(fig)

    return save_path








#========================================================================================================================================
# CASE PLOTTERS
#========================================================================================================================================
def find_case_h5_files(
        output_dir,
    ):
    """
    Find fixed-pressure and free-pressure h5 files in one numbered run folder.

    This accepts both naming schemes used by the case outputs:
        fixed-pressure: *_FXD.h5 or *_FLUX.h5
        free-pressure:  *_FREE.h5 or *_PRESS.h5
    """

    output_dir = Path(output_dir)
    files = {}

    for canonical_variant in VARIANT_DISPLAY_ORDER:
        matches = []

        for suffix in VARIANT_FILE_SUFFIXES[canonical_variant]:
            matches.extend(
                sorted(
                    output_dir.glob(f"*{suffix}"),
                )
            )

        if len(matches) > 0:
            files[canonical_variant] = matches[0]

        else:
            files[canonical_variant] = None

    if all(path is None for path in files.values()):
        return {}

    return files





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
    ):
    """
    Plot fixed-pressure and free-pressure profiles for one output folder.
    """

    profiles = load_pressure_profiles(
        files = files,
    )

    plot_stem = get_plot_stem(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
    )

    save_path = output_dir / f"{plot_stem}_pressure.png"

    saved = make_profile_plot(
        profiles = profiles,
        title = get_case_plot_title(
            case = case,
            objective_label = objective_label,
        ),
        ylabel = "pressure [Pa]",
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
        N_Xsecs = 1,
    ):
    """
    Plot fixed-pressure and free-pressure toroidal cross-section overlays.
    """

    equilibrium_data = []

    for variant in VARIANT_DISPLAY_ORDER:
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

    base_title = get_case_plot_title(
        case = case,
        objective_label = objective_label,
    )

    saved_paths = []

    for N_Xsec_index in range(N_Xsecs):
        if N_Xsecs == 1:
            save_path = output_dir / f"{plot_stem}_toroidal_cross_sections.png"

        else:
            save_path = output_dir / f"{plot_stem}_toroidal_cross_sections_N_Xsec_{N_Xsec_index + 1:03d}.png"

        saved = make_toroidal_cross_section_plot(
            equilibrium_data = equilibrium_data,
            title = base_title,
            save_path = save_path,
            num_phi = 6,
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

    files = find_case_h5_files(
        output_dir = output_dir,
    )

    if len(files) == 0:
        print("")
        print(f"No fixed/free h5 files were found in: {output_dir}")
        print("Expected one of *_FXD.h5 or *_FLUX.h5, and one of *_FREE.h5 or *_PRESS.h5.")
        print("")
        return []

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
    )

    saved_paths += plot_toroidal_cross_sections(
        case = case,
        objective_label = objective_label,
        output_dir = output_dir,
        files = files,
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

    Existing plot PNG files are overwritten, not skipped.
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
            "No plots were generated because no fixed/free h5 files were found."
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
        description = "Plot DESC case fixed-pressure/free-pressure comparisons.",
    )

    parser.add_argument(
        "--case",
        required = True,
        help = "Case name, e.g. ARIES-CS, HELIOTRON, or W7-X.",
    )

    parser.add_argument(
        "--obj",
        default = None,
        choices = [
            "qs3",
            "balloon",
            "force",
        ],
        help = "Optional objective folder to plot. If omitted, all existing objective folders are plotted.",
    )

    parser.add_argument(
        "--N-Xsecs",
        type = int,
        default = 1,
        help = "Number of interleaved six-cut toroidal cross-section figures to save.",
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
    print("Existing plots = overwritten")
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
