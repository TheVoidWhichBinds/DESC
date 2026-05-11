# driver.py
#==============================================================================================================
#
# Top-level CLI for running DESC case optimizations with FXD/FREE pressure pairs.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 driver.py --case ATF
#   python3 driver.py --case ATF --obj qs3
#   python3 driver.py --case ATF --obj balloon
#   python3 driver.py --case ATF --obj force
#
#==============================================================================================================

import os
os.environ["JAX_PLATFORMS"] = "cpu"

if "JAX_PLATFORM_NAME" in os.environ:
    del os.environ["JAX_PLATFORM_NAME"]

from desc import set_device
set_device("cpu")

import argparse
import json

import numpy as np

import desc.examples
from desc.grid import ConcentricGrid, LinearGrid
from desc.objectives import (
    BallooningStability,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    Isodynamicity,
    ObjectiveFunction,
    QuasisymmetryTripleProduct,
)

try:
    from .helper import (
        ensure_case_layout,
        get_initial_equilibrium_path,
        get_next_run_dir,
        get_objective_dir,
        get_output_path,
        get_run_initial_equilibrium_path,
        optimize_save_report,
        resolve_requested_objectives,
        build_press_extension,
    )

except ImportError:
    from helper import (
        ensure_case_layout,
        get_initial_equilibrium_path,
        get_next_run_dir,
        get_objective_dir,
        get_output_path,
        get_run_initial_equilibrium_path,
        optimize_save_report,
        resolve_requested_objectives,
        build_press_extension,
    )










#==============================================================================================================
# HYPERPARAMETERS
#==============================================================================================================

OPTIMIZER = "lsq-auglag"

FTOL = 1e-5
XTOL = 1e-5
GTOL = 1e-5
CTOL = 1e-3

MAXITER = 500
MAX_NFEV = None

X_SCALE = "auto"

APPLY_INITIAL_EQUILIBRIUM_RESOLUTION = False

INITIAL_EQUILIBRIUM_L = 8
INITIAL_EQUILIBRIUM_M = 8
INITIAL_EQUILIBRIUM_N = 8


ISO_RHO = np.array(
    [
        0.25,
        0.50,
        0.75,
        1.00,
    ]
)

BALLOON_RHO = np.array(
    [
        0.5,
        0.6,
        0.7,
    ]
)

BALLOON_ALPHA = np.linspace(
    0.0,
    np.pi,
    8,
    endpoint = False,
)

BALLOON_NTURNS = 3
BALLOON_NZETA_PER_TURN = 150

BOUNDARY_MODE_CUTOFF = 2
FIX_MAJOR_RADIUS_MODE = True

FIX_IOTA_CASES = {
    "HELIOTRON",
    "W7-X",
}

FIX_CURRENT_CASES = {
    "ARIES-CS",
}










#==============================================================================================================
# Resolution / Grid Helpers
#==============================================================================================================

def get_grid_resolution(
        eq,
    ):
    """
    Return grid resolution equal to the working spectral resolution.
    """

    return {
        "L": int(eq.L),
        "M": int(eq.M),
        "N": int(eq.N),
    }




def get_equilibrium_resolution(
        eq,
    ):
    """
    Return the spectral and grid resolution for one equilibrium.
    """

    return {
        "L": int(eq.L),
        "M": int(eq.M),
        "N": int(eq.N),
        "L_grid": int(eq.L_grid),
        "M_grid": int(eq.M_grid),
        "N_grid": int(eq.N_grid),
    }





def get_boundary_basis_modes(
        eq,
        basis_name,
    ):
    """
    Return the boundary basis modes for R or Z boundary coefficients.
    """

    surface = getattr(
        eq,
        "surface",
        None,
    )

    if surface is None:
        raise AttributeError(
            "Equilibrium does not expose eq.surface, so boundary modes cannot be selected."
        )

    basis = getattr(
        surface,
        basis_name,
        None,
    )

    if basis is None or not hasattr(basis, "modes"):
        raise AttributeError(
            f"Equilibrium surface does not expose surface.{basis_name}.modes."
        )

    modes = np.asarray(
        basis.modes,
        dtype = int,
    )

    if modes.ndim != 2 or modes.shape[1] < 2:
        raise ValueError(
            f"Expected surface.{basis_name}.modes to have at least two columns, got shape {modes.shape}."
        )

    return modes





def get_boundary_mode_mn_columns(
        modes,
    ):
    """
    Return m and n columns from a boundary mode table.
    """

    if modes.shape[1] == 2:
        return modes[:, 0], modes[:, 1]

    return modes[:, -2], modes[:, -1]





def get_low_order_boundary_mode_indices(
        eq,
        basis_name,
        mode_cutoff = BOUNDARY_MODE_CUTOFF,
    ):
    """
    Return indices of boundary modes left free during optimization.
    """

    modes = get_boundary_basis_modes(
        eq = eq,
        basis_name = basis_name,
    )

    m_modes, n_modes = get_boundary_mode_mn_columns(
        modes = modes,
    )

    free_mode_mask = (np.abs(m_modes) <= mode_cutoff) & (np.abs(n_modes) <= mode_cutoff)

    return np.where(free_mode_mask)[0]





def get_free_boundary_modes(
        eq,
        basis_name,
        mode_cutoff = BOUNDARY_MODE_CUTOFF,
    ):
    """
    Return boundary modes excluded from FixBoundaryR/FixBoundaryZ.
    """

    modes = get_boundary_basis_modes(
        eq = eq,
        basis_name = basis_name,
    )

    free_mode_indices = get_low_order_boundary_mode_indices(
        eq = eq,
        basis_name = basis_name,
        mode_cutoff = mode_cutoff,
    )

    free_modes = modes[free_mode_indices]

    if basis_name == "R_basis" and FIX_MAJOR_RADIUS_MODE:
        r00_mask = np.all(
            free_modes == np.array([0, 0, 0]),
            axis = 1,
        )

        free_modes = free_modes[~r00_mask]

    return free_modes





def get_fixed_boundary_modes(
        eq,
        basis_name,
        mode_cutoff = BOUNDARY_MODE_CUTOFF,
    ):
    """
    Return boundary modes passed to FixBoundaryR/FixBoundaryZ.
    """

    modes = get_boundary_basis_modes(
        eq = eq,
        basis_name = basis_name,
    )

    free_mode_indices = get_low_order_boundary_mode_indices(
        eq = eq,
        basis_name = basis_name,
        mode_cutoff = mode_cutoff,
    )

    fixed_modes = np.delete(
        modes,
        free_mode_indices,
        axis = 0,
    )

    if basis_name == "R_basis" and FIX_MAJOR_RADIUS_MODE:
        r00_mask = np.all(
            modes == np.array([0, 0, 0]),
            axis = 1,
        )

        if np.any(r00_mask):
            fixed_modes = np.vstack(
                (
                    modes[r00_mask],
                    fixed_modes,
                )
            )

    return np.unique(
        fixed_modes,
        axis = 0,
    )





def get_boundary_mode_summary(
        eq,
    ):
    """
    Return the exact boundary modes fixed and excluded by FixBoundaryR/FixBoundaryZ.
    """

    r_fixed_modes = get_fixed_boundary_modes(
        eq = eq,
        basis_name = "R_basis",
    )

    z_fixed_modes = get_fixed_boundary_modes(
        eq = eq,
        basis_name = "Z_basis",
    )

    r_free_modes = get_free_boundary_modes(
        eq = eq,
        basis_name = "R_basis",
    )

    z_free_modes = get_free_boundary_modes(
        eq = eq,
        basis_name = "Z_basis",
    )

    return {
        "mode_cutoff": BOUNDARY_MODE_CUTOFF,
        "free_rule": "Exclude modes from FixBoundaryR/FixBoundaryZ where abs(m) <= 2 and abs(n) <= 2, except the R major-radius mode [0, 0, 0].",
        "fixed_rule": "Pass all remaining modes to FixBoundaryR/FixBoundaryZ, so modes with abs(m) > 2 or abs(n) > 2 are fixed. Also pass R mode [0, 0, 0] to FixBoundaryR as DESC tutorial regularization.",
        "fix_major_radius_mode": bool(FIX_MAJOR_RADIUS_MODE),
        "R_free_mode_count": int(r_free_modes.shape[0]),
        "Z_free_mode_count": int(z_free_modes.shape[0]),
        "R_fixed_mode_count": int(r_fixed_modes.shape[0]),
        "Z_fixed_mode_count": int(z_fixed_modes.shape[0]),
        "R_free_modes_excluded_from_FixBoundaryR": r_free_modes.tolist(),
        "Z_free_modes_excluded_from_FixBoundaryZ": z_free_modes.tolist(),
        "R_fixed_modes_passed_to_FixBoundaryR": r_fixed_modes.tolist(),
        "Z_fixed_modes_passed_to_FixBoundaryZ": z_fixed_modes.tolist(),
    }



