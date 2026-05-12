# plot.py
#==============================================================================================================
#
# Plot tutorial output comparisons from research/lit_comp/tutorials/<tutorial>/.
#
# Usage:
#   cd DESC
#   python3 research/lit_comp/tutorials/plot.py --tutorial basic_qs
#   python3 research/lit_comp/tutorials/plot.py --tutorial balloon
#   python3 research/lit_comp/tutorials/plot.py --tutorial basic_qs --case 001
#
# New output layout:
#   tutorials/basic_qs/tripleQS/001/
#   tutorials/basic_qs/twotermQH/001/
#   tutorials/balloon/balloon/001/
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
from desc.io import load
from desc.plotting import plot_comparison
from matplotlib.colors import to_rgba








#========================================================================================================================================
# PATHS
#========================================================================================================================================
TUTORIALS_DIR = Path(__file__).resolve().parent
LIT_COMP_DIR = TUTORIALS_DIR.parent










#========================================================================================================================================
# PLOT STYLE
#========================================================================================================================================
VARIANT_LABELS = {
    "FXD": "fixed-pressure",
    "FREE": "free-pressure",
}

DEFAULT_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

VARIANT_COLORS = {
    "FXD": DEFAULT_COLOR_CYCLE[0],
    "FREE": DEFAULT_COLOR_CYCLE[1],
}










#========================================================================================================================================
# PATH / FILE HELPERS
#========================================================================================================================================
def normalize_tutorial_name(
        tutorial,
    ):
    """
    Normalize a tutorial argument into a tutorial stem.
    """

    return Path(tutorial).stem





def get_tutorial_dir(
        tutorial,
    ):
    """
    Return research/lit_comp/tutorials/<tutorial>.
    """

    tutorial_name = normalize_tutorial_name(
        tutorial = tutorial,
    )

    tutorial_dir = TUTORIALS_DIR / tutorial_name

    if not tutorial_dir.exists():
        raise FileNotFoundError(
            f"Tutorial directory does not exist: {tutorial_dir}"
        )

    return tutorial_dir





def get_tutorial_plot_title(
        tutorial_name,
    ):
    """
    Return a publication-style tutorial title.
    """

    title_map = {
        "basic_qs": "Basic Quasi-Symmetry Triple Product Tutorial",
        "balloon": "Infinite-n Ideal Ballooning Mode Tutorial",
    }

    return title_map.get(
        tutorial_name,
        f"{tutorial_name} Tutorial",
    )





def is_numbered_case_dir(
        path,
    ):
    """
    Return True for output folders named 001, 002, 003, ...
    """

    path = Path(path)

    return path.is_dir() and path.name.isdigit()





def find_numbered_case_dirs(
        tutorial_dir,
    ):
    """
    Return numbered tolerance-case folders directly under a folder.
    """

    tutorial_dir = Path(tutorial_dir)

    return tuple(
        sorted(
            path
            for path in tutorial_dir.iterdir()
            if is_numbered_case_dir(
                path = path,
            )
        )
    )





def get_optimization_dirs(
        tutorial_dir,
    ):
    """
    Return tutorial-local optimization folders in the new nested layout.
    """

    optimization_dirs = []

    for path in sorted(tutorial_dir.iterdir()):
        if not path.is_dir():
            continue

        if path.name.startswith("__"):
            continue

        if len(find_numbered_case_dirs(path)) > 0:
            optimization_dirs.append(path)

    return tuple(optimization_dirs)





def get_output_case_dirs(
        tutorial_dir,
    ):
    """
    Return output case directories for the new layout, with old-layout fallback.
    """

    tutorial_dir = Path(tutorial_dir)

    nested_case_dirs = []

    for optimization_dir in get_optimization_dirs(
            tutorial_dir = tutorial_dir,
        ):

        nested_case_dirs.extend(
            find_numbered_case_dirs(
                tutorial_dir = optimization_dir,
            )
        )

    if len(nested_case_dirs) > 0:
        return tuple(nested_case_dirs)

    direct_case_dirs = find_numbered_case_dirs(
        tutorial_dir = tutorial_dir,
    )

    if len(direct_case_dirs) > 0:
        return direct_case_dirs

    return (
        tutorial_dir,
    )





def normalize_case_label(
        case,
    ):
    """
    Normalize a requested output folder label.

    Examples:
        1   -> 001
        001 -> 001
    """

    if case is None:
        return None

    case = str(case)

    if case.isdigit():
        return f"{int(case):03d}"

    return case





def get_requested_case_dirs(
        tutorial_dir,
        case = None,
    ):
    """
    Return either all output folders or every optimization folder for one case label.
    """

    tutorial_dir = Path(tutorial_dir)

    if case is None:
        return get_output_case_dirs(
            tutorial_dir = tutorial_dir,
        )

    case_label = normalize_case_label(
        case = case,
    )

    requested_case_dirs = []

    direct_case_dir = tutorial_dir / case_label

    if direct_case_dir.exists() and direct_case_dir.is_dir():
        requested_case_dirs.append(direct_case_dir)

    for optimization_dir in get_optimization_dirs(
            tutorial_dir = tutorial_dir,
        ):

        case_dir = optimization_dir / case_label

        if case_dir.exists() and case_dir.is_dir():
            requested_case_dirs.append(case_dir)

    if len(requested_case_dirs) == 0:
        raise FileNotFoundError(
            f"Requested output folder does not exist for case {case_label} under {tutorial_dir}."
        )

    return tuple(requested_case_dirs)





def load_final_equilibrium(
        path,
    ):
    """
    Load the final equilibrium from a DESC .h5 file.
    """

    obj = load(str(path))

    if hasattr(obj, "equilibria"):
        return obj.equilibria[-1]

    if isinstance(obj, (list, tuple)):
        return obj[-1]

    if hasattr(obj, "__getitem__") and not hasattr(obj, "compute"):
        return obj[-1]

    return obj





def get_specific_optimization_name(
        tutorial_name,
        path,
    ):
    """
    Extract the specific optimization name from a file name.
    """

    stem = Path(path).stem

    for suffix in (
        "_FXD",
        "_FREE",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    if stem == tutorial_name:
        return "main"

    prefix = f"{tutorial_name}_"

    if stem.startswith(prefix):
        remainder = stem[len(prefix):]

        if remainder == "":
            return "main"

        return remainder

    return stem





def get_optimization_label_from_case_dir(
        tutorial_name,
        case_dir,
    ):
    """
    Return the optimization label for an output case folder.
    """

    case_dir = Path(case_dir)

    tutorial_dir = TUTORIALS_DIR / tutorial_name

    if case_dir.parent == tutorial_dir:
        return "main"

    if case_dir.parent.parent == tutorial_dir:
        return case_dir.parent.name

    return "main"





def find_h5_files(
        tutorial_dir,
    ):
    """
    Find FXD and FREE files in one output folder.
    """

    case_dir = Path(tutorial_dir)

    if case_dir.name.isdigit() and case_dir.parent.parent == TUTORIALS_DIR:
        tutorial_name = case_dir.parent.name
        optimization_label = None

    elif case_dir.name.isdigit():
        tutorial_name = case_dir.parent.parent.name
        optimization_label = case_dir.parent.name

    else:
        tutorial_name = case_dir.name
        optimization_label = None

    files = {}

    for variant, suffix in (
        ("FXD", "_FXD.h5"),
        ("FREE", "_FREE.h5"),
    ):
        matches = sorted(case_dir.glob(f"*{suffix}"))

        for path in matches:
            if optimization_label is None:
                group_name = get_specific_optimization_name(
                    tutorial_name = tutorial_name,
                    path = path,
                )

            else:
                group_name = optimization_label

            if group_name not in files:
                files[group_name] = {
                    "FXD": None,
                    "FREE": None,
                }

            files[group_name][variant] = path

    return files





def get_output_stem(
        tutorial_name,
        optimization_name,
    ):
    """
    Return the common output stem for plot filenames.
    """

    if optimization_name == "main":
        return tutorial_name

    return f"{tutorial_name}_{optimization_name}"










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





def load_profiles_for_group(
        group_files,
        quantity,
    ):
    """
    Load FXD and FREE profiles for one optimization group.
    """

    profiles = {}

    for variant in (
        "FXD",
        "FREE",
    ):
        path = group_files.get(variant)

        if path is None:
            continue

        eq = load_final_equilibrium(
            path = path,
        )

        rho, values = compute_radial_profile(
            eq = eq,
            quantity = quantity,
        )

        profiles[variant] = (
            rho,
            values,
        )

    return profiles





def load_pressure_gradient_profiles_for_group(
        group_files,
    ):
    """
    Load numerical dp/drho profiles for one optimization group.
    """

    profiles = {}

    for variant in (
        "FXD",
        "FREE",
    ):
        path = group_files.get(variant)

        if path is None:
            continue

        eq = load_final_equilibrium(
            path = path,
        )

        rho, pressure = compute_radial_profile(
            eq = eq,
            quantity = "p",
        )

        dp_drho = np.gradient(
            pressure,
            rho,
        )

        profiles[variant] = (
            rho,
            dp_drho,
        )

    return profiles










#========================================================================================================================================
# PLOTTING HELPERS
#========================================================================================================================================
def make_profile_plot(
        profiles,
        title,
        ylabel,
        save_path,
    ):
    """
    Make and save one radial profile comparison plot.

    All curves are solid and colored consistently with toroidal cut plots.
    """

    if len(profiles) == 0:
        return None

    fig, ax = plt.subplots(
        figsize = (9, 6),
    )

    for variant, (rho, values) in profiles.items():

        label = VARIANT_LABELS.get(
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
    Make and save one toroidal cross-section figure for one optimization.

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
        VARIANT_LABELS.get(
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

    phi_values = np.linspace(
        0.0,
        2.0 * np.pi / reference_eq.NFP,
        num_phi,
        endpoint = False,
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

    fig.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight",
    )

    plt.close(fig)

    return save_path





def plot_toroidal_cross_sections(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plot toroidal cross-sections one figure per optimization.

    Each figure has 6 overlaid toroidal cuts in a 2-by-3 layout.
    """

    saved_paths = []

    tutorial_plot_title = get_tutorial_plot_title(
        tutorial_name = tutorial_name,
    )

    for optimization_name, group_files in files.items():

        equilibrium_data = []

        for variant in (
            "FXD",
            "FREE",
        ):
            path = group_files.get(
                variant,
                None,
            )

            if path is None:
                continue

            try:
                eq = load_final_equilibrium(
                    path = path,
                )

            except Exception as error:
                print("")
                print(f"Skipping toroidal cross-sections for {optimization_name} {variant}: {error}")
                print("")
                continue

            equilibrium_data.append(
                (
                    variant,
                    eq,
                )
            )

        if len(equilibrium_data) == 0:
            continue

        output_stem = get_output_stem(
            tutorial_name = tutorial_name,
            optimization_name = optimization_name,
        )

        save_path = tutorial_dir / f"{output_stem}_toroidal_cross_sections.png"

        if tutorial_name == "basic_qs":
            title = tutorial_plot_title

        elif optimization_name == "main":
            title = f"{tutorial_plot_title}: toroidal cross-sections"

        else:
            title = f"{tutorial_plot_title} {optimization_name}: toroidal cross-sections"

        saved = make_toroidal_cross_section_plot(
            equilibrium_data = equilibrium_data,
            title = title,
            save_path = save_path,
            num_phi = 6,
        )

        if saved is not None:
            saved_paths.append(saved)

    return saved_paths





def plot_pressure_profiles(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plot all FXD and FREE pressure profiles in one tutorial-level figure.
    """

    profiles = {}

    for optimization_name, group_files in files.items():
        for variant in (
            "FXD",
            "FREE",
        ):
            path = group_files.get(
                variant,
                None,
            )

            if path is None:
                continue

            eq = load_final_equilibrium(
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

    save_path = tutorial_dir / f"{tutorial_name}_pressure.png"

    saved = make_profile_plot(
        profiles = profiles,
        title = get_tutorial_plot_title(
            tutorial_name = tutorial_name,
        ),
        ylabel = "pressure [Pa]",
        save_path = save_path,
    )

    if saved is None:
        return []

    return [saved]










#========================================================================================================================================
# TUTORIAL-SPECIFIC PLOTTERS
#========================================================================================================================================
def plot_standard_tutorial_outputs(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plot the standard tutorial-level comparisons.
    """

    saved_paths = []

    saved_paths += plot_pressure_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    saved_paths += plot_toroidal_cross_sections(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    return saved_paths





def plot_basic_qs(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plots for the basic_qs tutorial.
    """

    return plot_standard_tutorial_outputs(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )





def plot_balloon(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plots for the balloon tutorial.
    """

    return plot_standard_tutorial_outputs(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )










#========================================================================================================================================
# PLOT REGISTRY
#========================================================================================================================================
PLOTTERS = {
    "basic_qs": plot_basic_qs,
    "balloon": plot_balloon,
}










#========================================================================================================================================
# PUBLIC ENTRY POINT
#========================================================================================================================================
def plot_tutorial(
        tutorial,
        case = None,
    ):
    """
    Plot relevant comparisons for one tutorial.

    If case is None, plots are generated for all numbered output folders.
    If case is given, plots are generated for every optimization folder with that case label.
    """

    tutorial_name = normalize_tutorial_name(
        tutorial = tutorial,
    )

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial_name,
    )

    case_dirs = get_requested_case_dirs(
        tutorial_dir = tutorial_dir,
        case = case,
    )

    plotter = PLOTTERS.get(
        tutorial_name,
        None,
    )

    if plotter is None:
        raise ValueError(
            f"No plot registry entry found for tutorial = {tutorial_name}"
        )

    saved_paths = []

    for case_dir in case_dirs:

        files = find_h5_files(
            tutorial_dir = case_dir,
        )

        if len(files) == 0:
            print("")
            print(f"No *_FXD.h5 or *_FREE.h5 files were found in: {case_dir}")
            print("")
            continue

        saved_paths += plotter(
            tutorial_name = tutorial_name,
            tutorial_dir = case_dir,
            files = files,
        )

    if len(saved_paths) == 0:
        raise FileNotFoundError(
            "No plots were generated because no *_FXD.h5 or *_FREE.h5 files were found in:\n"
            f"{tutorial_dir}"
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
        description = "Plot DESC tutorial FXD/FREE comparisons.",
    )

    parser.add_argument(
        "--tutorial",
        required = True,
        help = "Tutorial name, e.g. basic_qs or balloon.",
    )

    parser.add_argument(
        "--case",
        default = None,
        help = "Optional numbered output folder to plot, e.g. 001 or 1. If omitted, all numbered folders are plotted.",
    )

    return parser.parse_args()





def main():
    """
    CLI entry point.
    """

    args = parse_args()

    print("")
    print("================================================================================================================")
    print(f"Generating plots for tutorial = {args.tutorial}")
    print("================================================================================================================")
    print("")

    saved_paths = plot_tutorial(
        tutorial = args.tutorial,
        case = args.case,
    )

    print("")
    print("Saved plots:")
    print("")

    for path in saved_paths:
        print(path)

    print("")










if __name__ == "__main__":
    main()