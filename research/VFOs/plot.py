# plotting.py

#===================================================================================================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
from desc.grid import LinearGrid
from desc.plotting import plot_comparison
#===================================================================================================================================================















#============== SHARED SOLUTION PLOT HELPERS ======================================================================================================
#======================
def _get_reference_eq(
        eqs,
    ):
    if len(eqs) == 0:
        raise ValueError("Need at least one equilibrium to plot.")

    return eqs[0]
#======================
#===================================================================================================================================================











#============== SOLUTION COMPARISON PLOTS ==========================================================================================================
#========================================
def save_pressure_plot(
        out_dir,
        eqs,
        labels,
        colors,
    ):
    ref_eq = _get_reference_eq(eqs)

    rho = np.linspace(0.0, 1.0, 400)
    grid = LinearGrid(
        rho = rho,
        M = 0,
        N = 0,
        NFP = ref_eq.NFP,
        sym = ref_eq.sym,
    )

    plt.figure(figsize = (7, 5))

    for eq, label, color in zip(eqs, labels, colors):
        p = eq.compute("p", grid = grid)["p"]
        plt.plot(rho, p, linewidth = 2, label = label, color = color)

    plt.xlabel(r"$\rho$", fontsize = 14)
    plt.ylabel("Pressure", fontsize = 14)
    plt.title(r"Pressure vs $\rho$", fontsize = 14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "pressure_compare.png")
    plt.savefig(save_path, dpi = 200)
    plt.close()
#========================================





#========================================
def save_solution_toroidal_cuts(
        out_dir,
        eqs,
        labels,
        colors,
    ):
    plt.title("Toroidal Cross-Sections of Compared Equilibria")
    fig, ax = plot_comparison(
        eqs = eqs,
        labels = labels,
        color = colors,
    )

    save_path = os.path.join(out_dir, "toroidal_cuts.png")
    plt.savefig(save_path, dpi = 200)
    plt.close()
#========================================





#========================================
def save_j_parallel_plot(
        out_dir,
        eqs,
        labels,
        colors,
    ):
    ref_eq = _get_reference_eq(eqs)

    rho_grid = np.linspace(0.0, 1.0, 100)
    grid_J = LinearGrid(
        rho = rho_grid,
        M = 24,
        N = 24,
        NFP = ref_eq.NFP,
        sym = ref_eq.sym,
    )

    def _j_parallel_profile(eq):
        data = eq.compute(["J_parallel"], grid = grid_J)
        J_parallel = np.asarray(data["J_parallel"])

        rho_nodes = grid_J.nodes[:, 0]
        rho_unique = np.unique(rho_nodes)

        J_parallel_fs = np.empty_like(rho_unique, dtype = float)
        for i, r in enumerate(rho_unique):
            mask = np.isclose(rho_nodes, r)
            J_parallel_fs[i] = np.mean(J_parallel[mask])

        return rho_unique, J_parallel_fs

    plt.figure(figsize = (7, 5))

    for eq, label, color in zip(eqs, labels, colors):
        rho_u, J_parallel = _j_parallel_profile(eq)
        plt.plot(
            rho_u,
            J_parallel,
            linewidth = 2,
            label = label,
            color = color,
        )

    plt.xlabel(r"$\rho$", fontsize = 14)
    plt.ylabel(r"$\langle J_{\parallel} \rangle$", fontsize = 14)
    plt.title("Parallel Current", fontsize = 13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "j_parallel.png")
    plt.savefig(save_path, dpi = 200)
    plt.close()
#========================================





#========================================
def save_iota_plot(
        out_dir,
        eqs,
        labels,
        colors,
    ):
    ref_eq = _get_reference_eq(eqs)

    rho_grid = np.linspace(0.0, 1.0, 100)
    grid_iota = LinearGrid(
        rho = rho_grid,
        M = 0,
        N = 0,
        NFP = ref_eq.NFP,
        sym = ref_eq.sym,
    )

    plt.figure(figsize = (7, 5))

    for eq, label, color in zip(eqs, labels, colors):
        iota = eq.compute("iota", grid = grid_iota)["iota"]
        plt.plot(
            rho_grid,
            iota,
            linewidth = 2,
            label = label,
            color = color,
        )

    plt.xlabel(r"$\rho$", fontsize = 14)
    plt.ylabel(r"$\iota$", fontsize = 14)
    plt.title(r"Rotational transform vs $\rho$", fontsize = 13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "iota.png")
    plt.savefig(save_path, dpi = 200)
    plt.close()
#========================================




#========================================
def save_initial_toroidal_cut(
        out_dir,
        eqs,
        labels,
        colors,
    ):
    if len(eqs) == 0:
        return

    fig, ax = plot_comparison(
        eqs = [
            eqs[0],
        ],
        labels = [
            labels[0],
        ],
        color = [
            colors[0],
        ],
    )

    plt.title("Initial Equilibrium Toroidal Cross-Section")

    save_path = os.path.join(out_dir, "initial_toroidal_cut.png")
    plt.savefig(save_path, dpi = 200)
    plt.close()
#========================================




#========================================
def save_all_solution_plots(
        out_dir,
        eqs,
        labels,
        colors,
    ):
    if len(eqs) == 0:
        return

    save_initial_toroidal_cut(
        out_dir = out_dir,
        eqs = eqs,
        labels = labels,
        colors = colors,
    )

    save_pressure_plot(
        out_dir = out_dir,
        eqs = eqs,
        labels = labels,
        colors = colors,
    )

    save_solution_toroidal_cuts(
        out_dir = out_dir,
        eqs = eqs,
        labels = labels,
        colors = colors,
    )

    save_j_parallel_plot(
        out_dir = out_dir,
        eqs = eqs,
        labels = labels,
        colors = colors,
    )

    save_iota_plot(
        out_dir = out_dir,
        eqs = eqs,
        labels = labels,
        colors = colors,
    )
#===================================================================================================================================================