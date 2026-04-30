# helical_qs.py
"""Quasi-symmetry with helical contours.

Modernized for current DESC APIs.
"""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from pathlib import Path
import os
import signal
import sys
import traceback

from desc import set_device
set_device("gpu")

import numpy as np

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.io import load
from desc.magnetic_fields import OmnigenousField
from desc.objectives import (
    CurrentDensity,
    FixOmniBmax,
    FixOmniMap,
    ObjectiveFunction,
    Omnigenity,
    get_NAE_constraints,
)
from desc.optimize import Optimizer

from qsc import Qsc










#========================================================================================================================================
# PATHS
#========================================================================================================================================
INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

INITIAL_PATH = INPUT_DIR / "helical_qs_initial.h5"










#========================================================================================================================================
# SETUP
#========================================================================================================================================
fname = "helical_qs"

CHECK_PATH = OUTPUT_DIR / f"{fname}_CHECK.h5"
FAILURE_PATH = OUTPUT_DIR / f"{fname}_FAILURE.h5"

sym = True
NFP = 5
helicity = (1, NFP)

LM = [8, 10, 12]
N = 12

L_well = 4
M_well = 8
L_omni = 0
M_omni = 0
N_omni = 0

well_weight = 2
eq_weights = [1e0, 2e0, 4e0]

surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

assert len(LM) == len(eq_weights)










#========================================================================================================================================
# HELPERS
#========================================================================================================================================
def eq_error(eq):
    """Compute normalized force-balance error."""

    grid = QuadratureGrid(
        L = 32,
        M = 32,
        N = 32,
        NFP = NFP,
    )

    data = eq.compute(
        [
            "<|F|>_vol",
            "<|grad(|B|^2)|/2mu0>_vol",
        ],
        grid = grid,
    )

    return data["<|F|>_vol"] / data["<|grad(|B|^2)|/2mu0>_vol"]










def load_initial_equilibrium(path):
    """Load the saved initial equilibrium."""

    obj = load(str(path))

    if isinstance(obj, EquilibriaFamily):
        eq = obj[-1]

    elif isinstance(obj, Equilibrium):
        eq = obj

    else:
        raise TypeError(
            "Expected Equilibrium or EquilibriaFamily from "
            f"{path}, got {type(obj)}."
        )

    fam = EquilibriaFamily()
    fam.append(eq)

    return fam, eq










def save_family_to_path(fam, path):
    """Save the equilibrium family."""

    fam.save(str(path))
    print(f"saved: {path}")










def save_equilibrium_to_path(eq, path):
    """Save a single optimized equilibrium."""

    eq.save(str(path))
    print(f"saved: {path}")










def save_failure_checkpoint(fam, stage_message):
    """Save the latest completed stage as the failure-mode checkpoint."""

    print(stage_message)
    save_family_to_path(
        fam = fam,
        path = FAILURE_PATH,
    )










def handle_signal(signum, frame):
    """Exit cleanly on catchable termination signals."""

    print("")
    print(f"Received signal {signum}.")
    print(f"Latest completed stage should be saved at: {FAILURE_PATH}")
    sys.exit(128 + signum)










def make_omnigenous_field(eq):
    """Create the modern DESC omnigenous target field."""

    field = OmnigenousField(
        L_B = L_well,
        M_B = M_well,
        L_x = L_omni,
        M_x = M_omni,
        N_x = N_omni,
        NFP = eq.NFP,
        helicity = helicity,
    )

    return field










def make_omnigenity_objective(eq, field, rho, eta_weight):
    """Create a modern DESC omnigenity objective on a single flux surface."""

    M_booz = min(int(np.ceil(1.5 * eq.M)), 16)
    N_booz = min(int(np.ceil(1.5 * eq.N)), 16)

    eq_grid = LinearGrid(
        rho = rho,
        M = int(np.ceil(1.5 * M_booz)),
        N = int(np.ceil(1.5 * N_booz)),
        NFP = eq.NFP,
        sym = False,
    )

    field_grid = LinearGrid(
        rho = rho,
        theta = 2 * M_booz,
        zeta = 2 * N_booz,
        NFP = field.NFP,
        sym = False,
    )

    objective = Omnigenity(
        eq = eq,
        field = field,
        eq_grid = eq_grid,
        field_grid = field_grid,
        eta_weight = eta_weight,
    )

    return objective










def make_omnigenity_constraints(eq, field, qsc):
    """Create constraints for modern omnigenity optimization."""

    omni_map_indices = np.where(field.x_basis.modes[:, 1] == 0)[0]

    constraints = get_NAE_constraints(
        eq,
        qsc,
        order = 1,
    ) + (
        FixOmniBmax(
            field = field,
        ),
        FixOmniMap(
            field = field,
            indices = omni_map_indices,
        ),
    )

    return constraints










#========================================================================================================================================
# SIGNAL HANDLING
#========================================================================================================================================
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)










#========================================================================================================================================
# INITIAL EQUILIBRIUM
#========================================================================================================================================
fam, eq = load_initial_equilibrium(INITIAL_PATH)

print("loaded initial equilibrium:", INITIAL_PATH)
print("equilibrium error: {:.2e}".format(eq_error(eq)))










#========================================================================================================================================
# NEAR-AXIS TARGET
#========================================================================================================================================
qsc = Qsc(
    nfp = NFP,
    rc = [
        1.00000000e00,
        1.36094189e-01,
        1.15569807e-02,
        5.77324841e-04,
        -2.13812436e-05,
        -6.86819127e-06,
        -3.25014052e-07,
        7.33393963e-08,
        1.42375011e-08,
        7.98521016e-10,
    ],
    zs = [
        0.00000000e00,
        -1.25243961e-01,
        -1.10551096e-02,
        -5.87380185e-04,
        1.59769355e-05,
        6.41471931e-06,
        3.47327323e-07,
        -6.49967945e-08,
        -1.39333315e-08,
        -8.47322874e-10,
    ],
    B0 = 1.0,
    B2c = -0.38028563,
    etabar = 2.179209340685954,
    I2 = 0.0,
    p2 = 0.0,
    order = "r1",
)

field = make_omnigenous_field(eq)










#========================================================================================================================================
# OPT
#========================================================================================================================================
try:
    for i in range(len(LM)):
        print("")
        print("================================================================================================================")
        print(f"Starting resolution stage {i + 1}/{len(LM)}: L = {LM[i]}, M = {LM[i]}, N = {N}")
        print("================================================================================================================")
        print("")

        eq.change_resolution(
            L = LM[i],
            M = LM[i],
            N = N,
            L_grid = 2 * LM[i],
            M_grid = 2 * LM[i],
            N_grid = 2 * N,
            sym = sym,
        )

        objectives = [
            CurrentDensity(
                eq = eq,
                weight = eq_weights[i],
            )
        ]

        for rho in surfaces:
            objectives.append(
                make_omnigenity_objective(
                    eq = eq,
                    field = field,
                    rho = rho,
                    eta_weight = well_weight,
                )
            )

        objective = ObjectiveFunction(tuple(objectives))

        constraints = make_omnigenity_constraints(
            eq = eq,
            field = field,
            qsc = qsc,
        )

        optimizer = Optimizer("lsq-exact")

        (eq, field), result = optimizer.optimize(
            (eq, field),
            objective,
            constraints,
            ftol = 1e-3,
            xtol = 1e-6,
            gtol = 1e-6,
            maxiter = 200,
            verbose = 3,
        )

        stage_path = OUTPUT_DIR / f"{fname}_{LM[i]}.h5"

        save_equilibrium_to_path(
            eq = eq,
            path = stage_path,
        )

        fam.append(eq)

        print("equilibrium error: {:.2e}".format(eq_error(eq)))

        save_failure_checkpoint(
            fam = fam,
            stage_message = (
                f"Completed stage {i + 1}/{len(LM)}. "
                f"Saving failure checkpoint in case a later stage fails."
            ),
        )

except Exception:
    print("")
    print("Optimization failed with an exception.")
    print(f"Latest completed stage is saved at: {FAILURE_PATH}")
    traceback.print_exc()
    raise










#========================================================================================================================================
# FINAL SAVE
#========================================================================================================================================
save_family_to_path(
    fam = fam,
    path = CHECK_PATH,
)

if FAILURE_PATH.exists():
    FAILURE_PATH.unlink()
    print(f"removed failure checkpoint: {FAILURE_PATH}")

print("final saved equilibrium resolution:")
print("L =", eq.L)
print("M =", eq.M)
print("N =", eq.N)
print("NFP =", eq.NFP)
print("equilibrium error: {:.2e}".format(eq_error(eq)))