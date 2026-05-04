# plot.py
#==============================================================================================================
#
# Plot tutorial output comparisons from research/lit_comp/tutorials/<tutorial>/.
#
# Usage:
#   cd DESC
#   python3 research/lit_comp/tutorials/plot.py --tutorial basic_qs
#   python3 research/lit_comp/tutorials/plot.py --tutorial adv_qs
#   python3 research/lit_comp/tutorials/plot.py --tutorial balloon
#   python3 research/lit_comp/tutorials/plot.py --tutorial neoclassical
#
# This script:
#   1. finds *_FXD.h5 and *_FREE.h5 files inside the requested tutorial folder
#   2. always plots all pressure profiles together in one tutorial-level figure
#   3. makes tutorial-specific additional plots
#   4. saves all plots into that same tutorial folder
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










#========================================================================================================================================
# PATHS
#========================================================================================================================================
TUTORIALS_DIR = Path(__file__).resolve().parent
LIT_COMP_DIR = TUTORIALS_DIR.parent










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

    Examples:
        basic_qs_C_FXD.h5   -> C
        basic_qs_T_FREE.h5  -> T
        balloon_FXD.h5      -> main
        balloon_FREE.h5     -> main
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





def find_h5_files(
        tutorial_dir,
    ):
    """
    Find FXD and FREE files in the tutorial folder.

    Returns:
        {
            "main": {
                "FXD": Path(...),
                "FREE": Path(...),
            },
            "C": {
                "FXD": Path(...),
                "FREE": Path(...),
            },
            "T": {
                "FXD": Path(...),
                "FREE": Path(...),
            },
        }
    """

    tutorial_dir = Path(tutorial_dir)
    tutorial_name = tutorial_dir.name

    files = {}

    for variant, suffix in (
        ("FXD", "_FXD.h5"),
        ("FREE", "_FREE.h5"),
    ):
        matches = sorted(tutorial_dir.glob(f"*{suffix}"))

        for path in matches:
            optimization_name = get_specific_optimization_name(
                tutorial_name = tutorial_name,
                path = path,
            )

            if optimization_name not in files:
                files[optimization_name] = {
                    "FXD": None,
                    "FREE": None,
                }

            files[optimization_name][variant] = path

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

    All curves are dashed.
    FXD curves use dash phase offsets relative to one another.
    FREE curves use dash phase offsets relative to one another.
    No markers or data offsets are used.
    """

    if len(profiles) == 0:
        return None

    fig, ax = plt.subplots(
        figsize = (9, 6),
    )

    phase_steps = (
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
    )

    group_counts = {
        "FXD": 0,
        "FREE": 0,
        "OTHER": 0,
    }

    for label, (rho, values) in profiles.items():

        if label.endswith("FXD"):
            group = "FXD"
            dash_sequence = (6, 3)
            linewidth = 2.25
            alpha = 0.90

        elif label.endswith("FREE"):
            group = "FREE"
            dash_sequence = (10, 3)
            linewidth = 2.25
            alpha = 0.90

        else:
            group = "OTHER"
            dash_sequence = (6, 3)
            linewidth = 2.00
            alpha = 0.85

        phase_index = group_counts[group]
        phase_offset = phase_steps[phase_index % len(phase_steps)]
        group_counts[group] += 1

        ax.plot(
            rho,
            values,
            label = label,
            linestyle = (phase_offset, dash_sequence),
            linewidth = linewidth,
            alpha = alpha,
        )

    ax.set_xlabel("rho")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        True,
        alpha = 0.30,
    )

    ax.legend(
        fontsize = 8,
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







def plot_pressure_profiles(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plot all FXD and FREE pressure profiles in one tutorial-level figure.

    If a tutorial has multiple optimizations, all of them are included in the
    same pressure plot and labeled by optimization name and variant.
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

            if optimization_name == "main":
                label = variant

            else:
                label = f"{optimization_name} {variant}"

            profiles[label] = (
                rho,
                pressure,
            )

    save_path = tutorial_dir / f"{tutorial_name}_pressure.png"

    saved = make_profile_plot(
        profiles = profiles,
        title = f"{tutorial_name}: pressure profiles",
        ylabel = "p",
        save_path = save_path,
    )

    if saved is None:
        return []

    return [saved]





def plot_iota_profiles(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plot all FXD and FREE iota profiles in one tutorial-level figure.

    If a tutorial has multiple optimizations, all of them are included in the
    same iota plot and labeled by optimization name and variant.
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

            try:
                eq = load_final_equilibrium(
                    path = path,
                )

                rho, iota = compute_radial_profile(
                    eq = eq,
                    quantity = "iota",
                )

            except Exception as error:
                print("")
                print(f"Skipping iota profile for {optimization_name} {variant}: {error}")
                print("")
                continue

            if optimization_name == "main":
                label = variant

            else:
                label = f"{optimization_name} {variant}"

            profiles[label] = (
                rho,
                iota,
            )

    save_path = tutorial_dir / f"{tutorial_name}_iota.png"

    saved = make_profile_plot(
        profiles = profiles,
        title = f"{tutorial_name}: iota profiles",
        ylabel = "iota",
        save_path = save_path,
    )

    if saved is None:
        return []

    return [saved]












#========================================================================================================================================
# TUTORIAL-SPECIFIC PLOTTERS
#========================================================================================================================================
def plot_basic_qs(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plots for the basic_qs tutorial.
    """

    saved_paths = []

    saved_paths += plot_pressure_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    saved_paths += plot_iota_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    return saved_paths





def plot_adv_qs(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plots for the adv_qs tutorial.
    """

    saved_paths = []

    saved_paths += plot_pressure_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    saved_paths += plot_iota_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    return saved_paths





def plot_balloon(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plots for the balloon tutorial.
    """

    saved_paths = []

    saved_paths += plot_pressure_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    return saved_paths





def plot_neoclassical(
        tutorial_name,
        tutorial_dir,
        files,
    ):
    """
    Plots for the neoclassical tutorial.
    """

    saved_paths = []

    saved_paths += plot_pressure_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    saved_paths += plot_iota_profiles(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )

    return saved_paths










#========================================================================================================================================
# PLOT REGISTRY
#========================================================================================================================================
PLOTTERS = {
    "basic_qs": plot_basic_qs,
    "adv_qs": plot_adv_qs,
    "balloon": plot_balloon,
    "neoclassical": plot_neoclassical,
}










#========================================================================================================================================
# PUBLIC ENTRY POINT
#========================================================================================================================================
def plot_tutorial(
        tutorial,
    ):
    """
    Plot all relevant comparisons for one tutorial.
    """

    tutorial_name = normalize_tutorial_name(
        tutorial = tutorial,
    )

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial_name,
    )

    files = find_h5_files(
        tutorial_dir = tutorial_dir,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            "No *_FXD.h5 or *_FREE.h5 files were found in:\n"
            f"{tutorial_dir}"
        )

    plotter = PLOTTERS.get(
        tutorial_name,
        None,
    )

    if plotter is None:
        raise ValueError(
            f"No plot registry entry found for tutorial = {tutorial_name}"
        )

    return plotter(
        tutorial_name = tutorial_name,
        tutorial_dir = tutorial_dir,
        files = files,
    )










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
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
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
    )

    print("")
    print("Saved plots:")
    print("")

    for path in saved_paths:
        print(path)

    print("")










if __name__ == "__main__":
    main()