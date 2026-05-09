# driver.py
#==============================================================================================================
#
# Top-level CLI for running DESC case optimizations with FLUX/PRESS pressure pairs.
#
# Usage:
#   cd research/lit_comp/cases
#   python3 driver.py --case ATF
#   python3 driver.py --case ATF --obj qs3
#   python3 driver.py --case ATF --obj iso
#   python3 driver.py --case ATF --obj balloon
#   python3 driver.py --case ATF --obj all
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
        clear_objective_folder,
        ensure_case_layout,
        get_initial_equilibrium_path,
        get_objective_dir,
        get_output_path,
        optimize_save_report,
        remove_file_if_present,
        resolve_requested_objectives,
        build_press_extension,
    )

except ImportError:
    from helper import (
        clear_objective_folder,
        ensure_case_layout,
        get_initial_equilibrium_path,
        get_objective_dir,
        get_output_path,
        optimize_save_report,
        remove_file_if_present,
        resolve_requested_objectives,
        build_press_extension,
    )










#==============================================================================================================
# HYPERPARAMETERS
#==============================================================================================================

OPTIMIZER = "lsq-auglag"

FTOL = 1e-3
XTOL = 1e-3
GTOL = 1e-3
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
    "NCSX",
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





def make_hyperparameter_payload(
        case,
        obj,
        eq,
        eq_loaded,
    ):
    """
    Build a JSON-safe hyperparameter payload for one objective folder.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    loaded_resolution = get_equilibrium_resolution(
        eq = eq_loaded,
    )

    working_resolution = get_equilibrium_resolution(
        eq = eq,
    )

    return {
        "case": str(case),
        "objective_folder": str(obj),
        "variants": [
            "FLUX",
            "PRESS",
        ],
        "optimizer": OPTIMIZER,
        "ftol": FTOL,
        "xtol": XTOL,
        "gtol": GTOL,
        "ctol": CTOL,
        "maxiter": MAXITER,
        "max_nfev": MAX_NFEV,
        "x_scale": X_SCALE,
        "apply_initial_equilibrium_resolution": bool(APPLY_INITIAL_EQUILIBRIUM_RESOLUTION),
        "initial_equilibrium_resolution_target": {
            "L": int(INITIAL_EQUILIBRIUM_L),
            "M": int(INITIAL_EQUILIBRIUM_M),
            "N": int(INITIAL_EQUILIBRIUM_N),
        },
        "initial_equilibrium_grid_resolution_policy": "always_match_working_equilibrium_spectral_resolution",
        "incoming_loaded_equilibrium_resolution": loaded_resolution,
        "working_initial_equilibrium_resolution_after_change": working_resolution,
        "objective_grid_resolution": {
            "L": int(grid_resolution["L"]),
            "M": int(grid_resolution["M"]),
            "N": int(grid_resolution["N"]),
        },
        "profile_constraint": {
            "active_constraint": get_profile_constraint_type(
                case = case,
            ),
            "fix_iota_cases": sorted(FIX_IOTA_CASES),
            "fix_current_cases": sorted(FIX_CURRENT_CASES),
        },
        "boundary_constraints": get_boundary_mode_summary(
            eq = eq,
        ),
        "isodynamicity": {
            "rho": ISO_RHO.tolist(),
        },
        "ballooning": {
            "rho": BALLOON_RHO.tolist(),
            "alpha": BALLOON_ALPHA.tolist(),
            "nturns": BALLOON_NTURNS,
            "nzeta_per_turn": BALLOON_NZETA_PER_TURN,
        },
        "optimizer_options": make_optimizer_options(),
    }





def save_hyperparameter_file(
        case,
        obj,
        objective_dir,
        eq,
        eq_loaded,
    ):
    """
    Save the hyperparameters used for one objective folder.
    """

    case_name = str(case).strip()

    hyperparameter_path = objective_dir / f"{case_name}_{obj}_hyperparameters.json"

    payload = make_hyperparameter_payload(
        case = case,
        obj = obj,
        eq = eq,
        eq_loaded = eq_loaded,
    )

    with open(hyperparameter_path, "w") as file:
        json.dump(
            payload,
            file,
            indent = 4,
        )

    print("")
    print(f"Saved objective-folder hyperparameters: {hyperparameter_path}")
    print("")

    return hyperparameter_path





def print_hyperparameter_report(
        case,
        eq_loaded,
        eq,
    ):
    """
    Print optimizer hyperparameters and equilibrium resolutions.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    loaded_resolution = get_equilibrium_resolution(
        eq = eq_loaded,
    )

    working_resolution = get_equilibrium_resolution(
        eq = eq,
    )

    print("")
    print("================================================================================================================")
    print("HYPERPARAMETERS")
    print("================================================================================================================")
    print("")
    print(f"case: {case}")
    print(f"optimizer: {OPTIMIZER}")
    print(f"ftol: {FTOL}")
    print(f"xtol: {XTOL}")
    print(f"gtol: {GTOL}")
    print(f"ctol: {CTOL}")
    print(f"maxiter: {MAXITER}")
    print(f"max_nfev: {MAX_NFEV}")
    print(f"x_scale: {X_SCALE}")
    print("")
    print("Profile constraint:")
    print(f"Active profile constraint = {get_profile_constraint_type(case = case)}")
    print(f"FIX_IOTA_CASES = {sorted(FIX_IOTA_CASES)}")
    print(f"FIX_CURRENT_CASES = {sorted(FIX_CURRENT_CASES)}")
    print("")
    print("Initial equilibrium spectral-resolution change:")
    print(f"APPLY_INITIAL_EQUILIBRIUM_RESOLUTION = {APPLY_INITIAL_EQUILIBRIUM_RESOLUTION}")
    print(f"INITIAL_EQUILIBRIUM_L = {INITIAL_EQUILIBRIUM_L}")
    print(f"INITIAL_EQUILIBRIUM_M = {INITIAL_EQUILIBRIUM_M}")
    print(f"INITIAL_EQUILIBRIUM_N = {INITIAL_EQUILIBRIUM_N}")
    print("")
    print("Initial equilibrium grid-resolution policy:")
    print("L_grid, M_grid, and N_grid are always set equal to the working equilibrium L, M, and N.")
    print("")
    print("Incoming loaded equilibrium resolution before change_resolution:")
    print(f"L      = {loaded_resolution['L']}")
    print(f"M      = {loaded_resolution['M']}")
    print(f"N      = {loaded_resolution['N']}")
    print(f"L_grid = {loaded_resolution['L_grid']}")
    print(f"M_grid = {loaded_resolution['M_grid']}")
    print(f"N_grid = {loaded_resolution['N_grid']}")
    print("")
    print("Working initial equilibrium resolution after copy/change_resolution:")
    print(f"L      = {working_resolution['L']}")
    print(f"M      = {working_resolution['M']}")
    print(f"N      = {working_resolution['N']}")
    print(f"L_grid = {working_resolution['L_grid']}")
    print(f"M_grid = {working_resolution['M_grid']}")
    print(f"N_grid = {working_resolution['N_grid']}")
    print("")
    print("Optimization/comparison grid resolution:")
    print(f"L_grid_objectives = {grid_resolution['L']}")
    print(f"M_grid_objectives = {grid_resolution['M']}")
    print(f"N_grid_objectives = {grid_resolution['N']}")
    print("")
    print("Isodynamicity surfaces:")
    print(f"ISO_RHO = {ISO_RHO}")
    print("")
    print("Ballooning settings:")
    print(f"BALLOON_RHO = {BALLOON_RHO}")
    print(f"BALLOON_ALPHA = {BALLOON_ALPHA}")
    print(f"BALLOON_NTURNS = {BALLOON_NTURNS}")
    print(f"BALLOON_NZETA_PER_TURN = {BALLOON_NZETA_PER_TURN}")
    print("")
    boundary_summary = get_boundary_mode_summary(
        eq = eq,
    )

    print("Boundary mode constraints:")
    print(f"BOUNDARY_MODE_CUTOFF = {BOUNDARY_MODE_CUTOFF}")
    print(f"FIX_MAJOR_RADIUS_MODE = {FIX_MAJOR_RADIUS_MODE}")
    print("FixBoundaryR/FixBoundaryZ are present in both FLUX and PRESS.")
    print("Excluded from FixBoundaryR/Z and therefore optimized: modes where abs(m) <= 2 and abs(n) <= 2, except R [0, 0, 0].")
    print("Passed to FixBoundaryR/Z and therefore fixed: all remaining higher modes, plus R [0, 0, 0].")
    print(f"R free mode count = {boundary_summary['R_free_mode_count']}")
    print(f"Z free mode count = {boundary_summary['Z_free_mode_count']}")
    print(f"R fixed mode count = {boundary_summary['R_fixed_mode_count']}")
    print(f"Z fixed mode count = {boundary_summary['Z_fixed_mode_count']}")
    print("================================================================================================================")
    print("")





def build_force_balance_grid(
        eq,
    ):
    """
    Build the force-balance grid.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    return LinearGrid(
        M = grid_resolution["M"],
        N = grid_resolution["N"],
        NFP = eq.NFP,
        sym = eq.sym,
    )





def build_qs3_grid(
        eq,
    ):
    """
    Build the quasi-symmetry triple-product grid.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    return ConcentricGrid(
        L = grid_resolution["L"],
        M = grid_resolution["M"],
        N = grid_resolution["N"],
        NFP = eq.NFP,
        sym = eq.sym,
    )





def build_iso_grid(
        eq,
    ):
    """
    Build the isodynamicity grid.
    """

    grid_resolution = get_grid_resolution(
        eq = eq,
    )

    return LinearGrid(
        rho = ISO_RHO,
        M = grid_resolution["M"],
        N = grid_resolution["N"],
        NFP = eq.NFP,
        sym = eq.sym,
    )










#==============================================================================================================
# Objective Builders
#==============================================================================================================

def build_qs3_objective(
        eq,
    ):
    """
    Build the quasi-symmetry triple-product objective.
    """

    return QuasisymmetryTripleProduct(
        eq = eq,
        grid = build_qs3_grid(
            eq = eq,
        ),
        normalize = True,
    )





def build_iso_objective(
        eq,
    ):
    """
    Build the isodynamicity objective.
    """

    return Isodynamicity(
        eq = eq,
        grid = build_iso_grid(
            eq = eq,
        ),
        normalize = True,
    )





def build_balloon_objective(
        eq,
    ):
    """
    Build the ideal ballooning stability objective.
    """

    return BallooningStability(
        eq = eq,
        rho = BALLOON_RHO,
        alpha = BALLOON_ALPHA,
        nturns = BALLOON_NTURNS,
        nzetaperturn = BALLOON_NZETA_PER_TURN,
        normalize = True,
    )





def build_primary_objectives(
        eq,
        obj,
    ):
    """
    Build the requested objective tuple for one optimization folder.
    """

    if obj == "qs3":
        return (
            build_qs3_objective(
                eq = eq,
            ),
        )

    if obj == "iso":
        return (
            build_iso_objective(
                eq = eq,
            ),
        )

    if obj == "balloon":
        return (
            build_balloon_objective(
                eq = eq,
            ),
        )

    if obj == "all":
        return (
            build_qs3_objective(
                eq = eq,
            ),
            build_iso_objective(
                eq = eq,
            ),
            build_balloon_objective(
                eq = eq,
            ),
        )

    raise ValueError(
        f"Unknown objective folder: {obj}"
    )





def normalize_case_name(
        case,
    ):
    """
    Return the case name in the normalized form used by the profile-constraint sets.
    """

    return str(case).strip().upper()





def get_profile_constraint_type(
        case,
    ):
    """
    Return whether this case should use FixIota or FixCurrent.
    """

    case_name = normalize_case_name(
        case = case,
    )

    use_iota = case_name in FIX_IOTA_CASES
    use_current = case_name in FIX_CURRENT_CASES

    if use_iota and use_current:
        raise ValueError(
            f"Case {case} appears in both FIX_IOTA_CASES and FIX_CURRENT_CASES. "
            "Each case must use exactly one profile constraint."
        )

    if use_iota:
        return "FixIota"

    if use_current:
        return "FixCurrent"

    raise ValueError(
        f"Case {case} is not listed in FIX_IOTA_CASES or FIX_CURRENT_CASES. "
        "Add this case to exactly one of those sets in the HYPERPARAMETERS section."
    )





def build_profile_constraint(
        eq,
        case,
    ):
    """
    Build the case-dependent profile constraint.
    """

    profile_constraint_type = get_profile_constraint_type(
        case = case,
    )

    if profile_constraint_type == "FixIota":
        return FixIota(
            eq = eq,
            name = "FixIota",
        )

    if profile_constraint_type == "FixCurrent":
        return FixCurrent(
            eq = eq,
            name = "FixCurrent",
        )

    raise ValueError(
        f"Unknown profile constraint type: {profile_constraint_type}"
    )





def build_core_constraints(
        eq,
        case,
    ):
    """
    Build constraints shared by FLUX and PRESS pressure optimizations.
    """

    constraints = (
        FixBoundaryZ(
            eq = eq,
            name = "FixBoundaryZ",
            modes = get_fixed_boundary_modes(
                eq = eq,
                basis_name = "Z_basis",
            ),
        ),
        FixBoundaryR(
            eq = eq,
            name = "FixBoundaryR",
            modes = get_fixed_boundary_modes(
                eq = eq,
                basis_name = "R_basis",
            ),
        ),
        ForceBalance(
            eq = eq,
            grid = build_force_balance_grid(
                eq = eq,
            ),
            normalize = True,
        ),
        FixPsi(
            eq = eq,
            name = "FixPsi",
        ),
    )

    constraints = constraints + (
        build_profile_constraint(
            eq = eq,
            case = case,
        ),
    )

    return constraints





def build_optimization_problem(
        eq,
        eq_initial,
        obj,
        variant,
        case,
    ):
    """
    Build the ObjectiveFunction objects for one FLUX or PRESS optimization.
    """

    primary_objectives = build_primary_objectives(
        eq = eq,
        obj = obj,
    )

    constraints = build_core_constraints(
        eq = eq,
        case = case,
    )

    variant = str(variant).upper()

    if variant == "FLUX":
        constraints = constraints + (
            FixPressure(
                eq = eq,
                name = "FixPressure",
            ),
        )

    elif variant == "PRESS":
        press_objectives, press_constraints = build_press_extension(
            eq = eq,
            eq_initial = eq_initial,
        )

        primary_objectives = primary_objectives + tuple(press_objectives)
        constraints = constraints + tuple(press_constraints)

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    objective = ObjectiveFunction(
        objectives = primary_objectives,
    )

    return objective, constraints





def print_optimization_stack(
        objective,
        constraints,
    ):
    """
    Print objective and constraint class names before DESC build messages.
    """

    objective_components = getattr(
        objective,
        "objectives",
        None,
    )

    if objective_components is None:
        objective_components = getattr(
            objective,
            "_objectives",
            (),
        )

    print("DESC objective stack:")

    for component in objective_components:
        print(f"  {component.__class__.__name__}: name = {component.name}")

    print("DESC constraint stack:")

    for component in constraints:
        print(f"  {component.__class__.__name__}: name = {component.name}")

    print("")










#==============================================================================================================
# Optimization Helpers
#==============================================================================================================

def make_optimizer_options():
    """
    Build the optimizer options dictionary.
    """

    options = {
        "initial_trust_ratio": 0.01,
    }

    if MAX_NFEV is not None:
        options["max_nfev"] = MAX_NFEV

    return options





def apply_initial_equilibrium_resolution(
        eq,
    ):
    """
    Set the working equilibrium resolution and force grid resolution to match it.
    """

    target_l = int(eq.L)
    target_m = int(eq.M)
    target_n = int(eq.N)

    if APPLY_INITIAL_EQUILIBRIUM_RESOLUTION:
        target_l = int(INITIAL_EQUILIBRIUM_L)
        target_m = int(INITIAL_EQUILIBRIUM_M)
        target_n = int(INITIAL_EQUILIBRIUM_N)

    eq.change_resolution(
        L = target_l,
        M = target_m,
        N = target_n,
        L_grid = target_l,
        M_grid = target_m,
        N_grid = target_n,
    )

    eq.surface = eq.get_surface_at(
        rho = 1.0,
    )

    return eq




def print_loaded_equilibrium_resolution(
        eq_loaded,
    ):
    """
    Print the raw resolution of the DESC example immediately after loading.
    """

    loaded_resolution = get_equilibrium_resolution(
        eq = eq_loaded,
    )

    print("Incoming loaded equilibrium resolution before change_resolution:")
    print(f"L      = {loaded_resolution['L']}")
    print(f"M      = {loaded_resolution['M']}")
    print(f"N      = {loaded_resolution['N']}")
    print(f"L_grid = {loaded_resolution['L_grid']}")
    print(f"M_grid = {loaded_resolution['M_grid']}")
    print(f"N_grid = {loaded_resolution['N_grid']}")
    print("")





def load_initial_equilibrium(
        case,
    ):
    """
    Load the initial equilibrium from DESC examples and apply the requested resolution to a copy.
    """

    print("")
    print("================================================================================================================")
    print(f"Loading DESC example equilibrium: {case}")
    print("================================================================================================================")
    print("")

    eq_loaded = desc.examples.get(
        case,
    )

    eq_initial = eq_loaded.copy()

    eq_initial = apply_initial_equilibrium_resolution(
        eq = eq_initial,
    )

    return eq_loaded, eq_initial





def save_initial_equilibrium(
        case,
        eq,
    ):
    """
    Save the incoming DESC example equilibrium inside the case folder.
    """

    initial_path = get_initial_equilibrium_path(
        case = case,
    )

    remove_file_if_present(
        path = initial_path,
    )

    eq.save(
        str(initial_path)
    )

    print("")
    print(f"Saved initial equilibrium: {initial_path}")
    print("")

    return initial_path





def run_one_variant(
        case,
        obj,
        variant,
        eq_initial,
    ):
    """
    Run one FLUX or PRESS pressure optimization.
    """

    variant = str(variant).upper()

    eq = eq_initial.copy()

    objective, constraints = build_optimization_problem(
        eq = eq,
        eq_initial = eq_initial,
        obj = obj,
        variant = variant,
        case = case,
    )

    output_path = get_output_path(
        case = case,
        obj = obj,
        variant = variant,
    )

    label = f"{case}_{obj}_{variant}"

    print("")
    print("################################################################################################################")
    print(f"Starting optimization: {label}")
    print(f"Output path: {output_path}")
    print("################################################################################################################")
    print("")

    print_optimization_stack(
        objective = objective,
        constraints = constraints,
    )

    eq, result = optimize_save_report(
        eq = eq,
        objective = objective,
        constraints = constraints,
        optimizer = OPTIMIZER,
        output_path = output_path,
        label = label,
        ftol = FTOL,
        xtol = XTOL,
        gtol = GTOL,
        ctol = CTOL,
        maxiter = MAXITER,
        options = make_optimizer_options(),
        copy = False,
        x_scale = X_SCALE,
    )

    return eq, result





def run_objective_pair(
        case,
        obj,
        eq_loaded,
        eq_initial,
    ):
    """
    Run the FLUX/PRESS pair for one objective folder.
    """

    objective_dir = get_objective_dir(
        case = case,
        obj = obj,
    )

    clear_objective_folder(
        folder = objective_dir,
    )

    save_hyperparameter_file(
        case = case,
        obj = obj,
        objective_dir = objective_dir,
        eq = eq_initial,
        eq_loaded = eq_loaded,
    )

    run_one_variant(
        case = case,
        obj = obj,
        variant = "FLUX",
        eq_initial = eq_initial,
    )

    run_one_variant(
        case = case,
        obj = obj,
        variant = "PRESS",
        eq_initial = eq_initial,
    )





def run_case(
        case,
        obj = None,
    ):
    """
    Run requested optimization pairs for one DESC example case.
    """

    objective_names = resolve_requested_objectives(
        obj = obj,
    )

    ensure_case_layout(
        case = case,
    )

    eq_loaded, eq_initial = load_initial_equilibrium(
        case = case,
    )

    print_hyperparameter_report(
        case = case,
        eq_loaded = eq_loaded,
        eq = eq_initial,
    )

    save_initial_equilibrium(
        case = case,
        eq = eq_initial,
    )

    for objective_name in objective_names:
        run_objective_pair(
            case = case,
            obj = objective_name,
            eq_loaded = eq_loaded,
            eq_initial = eq_initial,
        )

    return objective_names





def run_comparison_outputs(
        case,
        obj = None,
    ):
    """
    Run compare.py logic after optimization outputs are written.
    """

    try:
        from .compare import case_obj

    except ImportError:
        from compare import case_obj

    print("")
    print("================================================================================================================")
    print("Running case comparison CSV generation")
    print("================================================================================================================")
    print("")

    rows, csv_paths = case_obj(
        case = case,
        obj = obj,
    )

    print("")
    print("Finished case-objective comparison.")
    print("CSV files written to:")
    print("")

    for case_label, csv_path in csv_paths.items():
        print(f"{case_label}: {csv_path}")

    print("")

    return rows, csv_paths










#==============================================================================================================
# CLI
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        required = True,
        help = "DESC example/case name, e.g. ATF, HELIOTRON, NCSX, W7-X.",
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
        help = "Optional objective pair to run. If omitted, all four pairs run.",
    )

    parser.add_argument(
        "--skip-compare",
        action = "store_true",
        help = "Skip comparison CSV generation after the case run.",
    )

    return parser.parse_args()





def main():
    """
    Run one DESC example case.
    """

    args = parse_args()

    print("")
    print("================================================================================================================")
    print(f"Running case = {args.case}")

    if args.obj is None:
        print("Objective mode: all objective pairs")

    else:
        print(f"Objective mode: {args.obj}")

    print("Backend: CPU")
    print("Overwrite mode: enabled")
    print("================================================================================================================")
    print("")

    run_case(
        case = args.case,
        obj = args.obj,
    )

    if not args.skip_compare:
        run_comparison_outputs(
            case = args.case,
            obj = args.obj,
        )





if __name__ == "__main__":
    main()
