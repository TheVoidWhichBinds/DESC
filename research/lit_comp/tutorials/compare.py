# compare.py
#==============================================================================================================
#
# Compare tutorial-specific objective outputs between FXD, and FREE files.
# Also owns plotting now that plot.py has been removed.
#
# Usage:
#   python3 research/lit_comp/tutorials/compare.py --tutorial balloon
#   python3 research/lit_comp/tutorials/compare.py --tutorial balloon --plot
#
#==============================================================================================================

import argparse
import traceback

import matplotlib.pyplot as plt
import numpy as np

from desc.grid import LinearGrid

from helper import (
    compare_objective_set,
    find_h5_files,
    get_tutorial_dir,
    load_final_eq,
    write_table_csv,
)










#==============================================================================================================
# Tutorial Objective Helpers
#==============================================================================================================

def objective_spec(
        name,
        objective,
        thing = None,
    ):
    """
    Build one objective comparison spec.
    """

    spec = {
        "name": name,
        "objective": objective,
    }

    if thing is not None:
        spec["thing"] = thing

    return spec





def basic_qs_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the basic_qs tutorial.
    """

    from desc.objectives import (
        ForceBalance,
        QuasisymmetryBoozer,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "QuasisymmetryBoozer",
            objective = QuasisymmetryBoozer(
                eq = eq,
                helicity = (1, 0),
                M_booz = 2 * eq.M,
                N_booz = 2 * eq.N,
                grid = grid,
            ),
        ),
    ]





def adv_qs_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the adv_qs tutorial.
    """

    from desc.objectives import (
        ForceBalance,
        QuasisymmetryBoozer,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "QuasisymmetryBoozer",
            objective = QuasisymmetryBoozer(
                eq = eq,
                helicity = (1, 0),
                M_booz = 2 * eq.M,
                N_booz = 2 * eq.N,
                grid = grid,
            ),
        ),
    ]





def balloon_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the ballooning tutorial.

    FixIota, FixPressure, FixPsi, FixBoundaryR, and FixBoundaryZ are intentionally
    omitted because they are linear constraints and should not be compared.
    """

    from desc.objectives import (
        ForceBalance,
        AspectRatio,
        PrincipalCurvature,
        BallooningStability,
        GenericObjective,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    boundary_grid = LinearGrid(
        rho = 1.0,
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    ballooning_grid = LinearGrid(
        rho = np.linspace(0.1, 1.0, 10),
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "AspectRatio",
            objective = AspectRatio(
                eq = eq,
            ),
        ),
        objective_spec(
            name = "PrincipalCurvature",
            objective = PrincipalCurvature(
                eq = eq,
                grid = boundary_grid,
            ),
        ),
        objective_spec(
            name = "BallooningStability",
            objective = BallooningStability(
                eq = eq,
                grid = ballooning_grid,
            ),
        ),
        objective_spec(
            name = "GenericObjective maximum principal curvature",
            objective = GenericObjective(
                f = "curvature_k1_rho",
                eq = eq,
                grid = boundary_grid,
            ),
        ),
    ]





def neoclassical_objectives(
        eq,
    ):
    """
    Return non-Fix objectives specific to the neoclassical tutorial.
    """

    from desc.objectives import (
        ForceBalance,
        EffectiveRipple,
    )

    grid = LinearGrid(
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    neoclassical_grid = LinearGrid(
        rho = np.linspace(0.1, 1.0, 10),
        M = eq.M_grid,
        N = eq.N_grid,
        NFP = eq.NFP,
        sym = eq.sym,
    )

    return [
        objective_spec(
            name = "ForceBalance",
            objective = ForceBalance(
                eq = eq,
                grid = grid,
            ),
        ),
        objective_spec(
            name = "EffectiveRipple",
            objective = EffectiveRipple(
                eq = eq,
                grid = neoclassical_grid,
            ),
        ),
    ]










#==============================================================================================================
# Tutorial Objective Registry
#==============================================================================================================

TUTORIAL_OBJECTIVES = {
    "basic_qs": basic_qs_objectives,
    "adv_qs": adv_qs_objectives,
    "balloon": balloon_objectives,
    "neoclassical": neoclassical_objectives,
}










#==============================================================================================================
# Tutorial Objective Comparison
#==============================================================================================================

def tutorial_obj(
        tutorial : str,
    ):
    """
    Compare tutorial-specific objective values between FXD, and FREE files.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    files = find_h5_files(
        case_dir = tutorial_dir,
    )

    objective_getter = TUTORIAL_OBJECTIVES.get(
        tutorial,
        None,
    )

    if objective_getter is None:
        raise ValueError(
            f"No tutorial objective registry entry found for tutorial = {tutorial}."
        )

    rows = compare_objective_set(
        files = files,
        objective_getter = objective_getter,
    )

    csv_path = tutorial_dir / f"{tutorial}_case_obj.csv"

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





def compute_iota_profile(
        eq,
        num_rho : int = 200,
    ):
    """
    Computes iota on a radial grid.
    """

    rho = np.linspace(0.0, 1.0, num_rho)

    grid = LinearGrid(
        rho = rho,
        M = 0,
        N = 0,
        NFP = eq.NFP,
    )

    data = eq.compute(
        "iota",
        grid = grid,
    )

    iota = np.asarray(data["iota"]).reshape(-1)

    return rho, iota





def plot_pressure(
        tutorial : str,
    ):
    """
    Plot pressure profiles for FXD, and FREE files.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    files = find_h5_files(
        case_dir = tutorial_dir,
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
    plt.title(f"{tutorial} pressure profile")
    plt.legend()
    plt.tight_layout()

    output_path = tutorial_dir / f"{tutorial}_pressure.png"

    plt.savefig(
        output_path,
        dpi = 300,
    )

    plt.close()

    return output_path





def plot_iota(
        tutorial : str,
    ):
    """
    Plot iota profiles for FXD, and FREE files.
    """

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial,
    )

    files = find_h5_files(
        case_dir = tutorial_dir,
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

            rho, iota = compute_iota_profile(
                eq = eq,
            )

            plt.plot(
                rho,
                iota,
                label = label,
            )

            plotted_anything = True

        except Exception:
            print(f"\nFailed while plotting iota for {label}: {path}")
            traceback.print_exc()

    if not plotted_anything:
        print("No iota profiles plotted.")
        return None

    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\iota$")
    plt.title(f"{tutorial} iota profile")
    plt.legend()
    plt.tight_layout()

    output_path = tutorial_dir / f"{tutorial}_iota.png"

    plt.savefig(
        output_path,
        dpi = 300,
    )

    plt.close()

    return output_path





def plot_tutorial(
        tutorial : str,
        plot_folder : str,
    ):
    """
    Tutorial-specific plot router.
    """

    if plot_folder == "pressure":
        pressure_path = plot_pressure(
            tutorial = tutorial,
        )

        if pressure_path is None:
            return []

        return [pressure_path]

    if plot_folder == "iota":
        iota_path = plot_iota(
            tutorial = tutorial,
        )

        if iota_path is None:
            return []

        return [iota_path]

    if plot_folder == "profiles":
        paths = []

        pressure_path = plot_pressure(
            tutorial = tutorial,
        )

        iota_path = plot_iota(
            tutorial = tutorial,
        )

        if pressure_path is not None:
            paths.append(pressure_path)

        if iota_path is not None:
            paths.append(iota_path)

        return paths

    print("")
    print(f"No plot branch defined for tutorial = {tutorial}, plot_folder = {plot_folder}.")
    print("")

    return []





def plot_case(
        tutorial : str,
        plot_folder : str,
    ):
    """
    Backward-compatible alias for plot_tutorial.
    """

    return plot_tutorial(
        tutorial = tutorial,
        plot_folder = plot_folder,
    )










#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tutorial",
        required = True,
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
    )

    parser.add_argument(
        "--plot",
        action = "store_true",
        help = "Also generate plots.",
    )

    parser.add_argument(
        "--plot-folder",
        default = "pressure",
        help = "Plot branch key: pressure, iota, or profiles.",
    )

    return parser.parse_args()





def main():
    """
    Runs tutorial-specific objective comparison and optional plots.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Comparing tutorial objectives for tutorial = {args.tutorial}")
    print("=" * 120 + "\n")

    rows, csv_path = tutorial_obj(
        tutorial = args.tutorial,
    )

    print("")
    print("Finished tutorial-objective comparison.")
    print(f"CSV written to: {csv_path}")

    if args.plot:
        print("")
        print("=" * 120)
        print(f"Plotting for tutorial = {args.tutorial}, plot_folder = {args.plot_folder}")
        print("=" * 120)
        print("")

        paths = plot_tutorial(
            tutorial = args.tutorial,
            plot_folder = args.plot_folder,
        )

        if len(paths) == 0:
            print("No plots written.")

        for path in paths:
            print(f"Plot written to: {path}")

    print()





if __name__ == "__main__":
    main()