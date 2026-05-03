# toroidal.py
"""Omnigenity with toroidal contours."""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from pathlib import Path
import sys

from desc import set_device
set_device("gpu")

import numpy as np

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import (
    FixOmniBmax,
    FixOmniMap,
    ForceBalance,
    ObjectiveFunction,
    Omnigenity,
)
from desc.objectives import get_fixed_boundary_constraints, get_NAE_constraints










#========================================================================================================================================
# PATHS
#========================================================================================================================================
INPUT_DIR = Path(__file__).resolve().parent

REPO_DIR = INPUT_DIR.parents[4]
DUDT2024_PUBLICATION_DIR = REPO_DIR / "publications" / "dudt2024"

if str(DUDT2024_PUBLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(DUDT2024_PUBLICATION_DIR))

from qsc import Qsc

fname = "toroidal"

OG_PATH = INPUT_DIR / (fname + "_OG.h5")
FXD_PATH = INPUT_DIR / (fname + "_FXD.h5")
FAILURE_PATH = INPUT_DIR / (fname + "_FXD_FAILURE.h5")










#========================================================================================================================================
# SETUP
#========================================================================================================================================
sym = True
NFP = 1
helicity = (1, 0)
LM = [8, 10, 12]
N = 12
L_well = 4
M_well = 8
L_omni = 0
M_omni = 1
N_omni = 1
well_weight = 2
eq_weights = [5e2, 1e3, 2e3]
aspect_ratio = 20
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]
target_mode = [0, 1, -1]
target_amplitude = np.pi / 6

assert len(LM) == len(eq_weights)










#========================================================================================================================================
# HELPERS
#========================================================================================================================================
def eq_error(eq):
    grid = QuadratureGrid(
        L = 32,
        M = 32,
        N = 32,
        NFP = NFP,
    )

    data = eq.compute(
        ["<|F|>_vol", "<|grad(p)|>_vol"],
        grid = grid,
    )

    return data["<|F|>_vol"] / data["<|grad(p)|>_vol"]










#========================================================================================================================================
# INITIAL NAE SOLUTION
#========================================================================================================================================
try:
    fam = EquilibriaFamily()

    qsc = Qsc(
        nfp = NFP,
        rc = [1, 0.3],
        zs = [0, -0.3],
        B0 = 1.0,
        etabar = 1.0,
        I2 = 1.0,
        p2 = -4e6,
        order = "r1",
    )

    eq = Equilibrium.from_near_axis(
        qsc,
        r = 1 / aspect_ratio,
        L = LM[0],
        M = LM[0],
        N = N,
    )

    # Old DESC omnigenity-map API.
    # Modern DESC no longer accepts L_well/M_well/L_omni/M_omni/N_omni in
    # Equilibrium.from_near_axis(...), and the old direct eq.omni_basis /
    # eq._omni_lmn mutation is no longer valid in the modern API.
    #
    # idx = np.nonzero((eq.omni_basis.modes == target_mode).all(axis = 1))[0]
    # omni_lmn = np.zeros(eq.omni_basis.num_modes)
    # omni_lmn[idx] = target_amplitude
    # eq._omni_lmn = omni_lmn

    fam.append(eq)

    if not OG_PATH.exists():
        raise ValueError("OG file doesn't exist")
    else:
        print(f"kept existing OG file: {OG_PATH}")

    print("equlibrium error: {:.2e}".format(eq_error(eq)))










#========================================================================================================================================
# RE-SOLVE WITH NAE CONSTRAINTS
#========================================================================================================================================
    constraints = get_NAE_constraints(
        eq,
        qsc,
        order = 1,
    )

    eq, result = eq.solve(
        objective = "force",
        constraints = constraints,
        ftol = 1e-2,
        xtol = 1e-6,
        gtol = 1e-6,
        maxiter = 200,
        verbose = 3,
        copy = True,
    )

    fam.append(eq)
    fam.save(str(FXD_PATH))
    print("equlibrium error: {:.2e}".format(eq_error(eq)))










#========================================================================================================================================
# OPTIMIZE WITH INCREASING RESOLUTION
#========================================================================================================================================
    for i in range(len(LM)):
        eq.change_resolution(
            L = LM[i],
            M = LM[i],
            L_grid = 2 * LM[i],
            M_grid = 2 * LM[i],
            sym = sym,
        )

        M_booz = min(int(np.ceil(1.5 * LM[i])), 16)
        N_booz = min(int(np.ceil(1.5 * N)), 16)
        M_grid = int(np.ceil(1.5 * M_booz))
        N_grid = int(np.ceil(1.5 * N_booz))

        grids = {}
        objs = {}

        for rho in surfaces:
            grids[rho] = LinearGrid(
                M = M_grid,
                N = N_grid,
                NFP = eq.NFP,
                sym = False,
                rho = rho,
            )

            objs[rho] = Omnigenity(
                grid = grids[rho],
                helicity = helicity,
                M_booz = M_booz,
                N_booz = N_booz,
                well_weight = well_weight,
            )

        objective = ObjectiveFunction(
            (ForceBalance(weight = eq_weights[i]),) + tuple(objs.values())
        )

        constraints = get_NAE_constraints(
            eq,
            qsc,
            order = 1,
        ) + (
            FixOmniMap(eq),
            FixOmniBmax(eq),
        )

        eq, result = eq.solve(
            objective = objective,
            constraints = constraints,
            optimizer = "lsq-exact",
            ftol = 1e-3,
            xtol = 1e-6,
            gtol = 1e-6,
            maxiter = 200,
            verbose = 3,
            copy = True,
        )

        fam.append(eq)
        fam.save(str(FXD_PATH))
        print("equlibrium error: {:.2e}".format(eq_error(eq)))










#========================================================================================================================================
# RE-SOLVE WITH FIXED BOUNDARY CONSTRAINTS
#========================================================================================================================================
    constraints = get_fixed_boundary_constraints(
        iota = False,
    )

    eq, result = eq.solve(
        objective = "force",
        constraints = constraints,
        ftol = 1e-2,
        xtol = 1e-6,
        gtol = 1e-6,
        maxiter = 200,
        verbose = 3,
        copy = True,
    )

    fam.append(eq)
    fam.save(str(FXD_PATH))
    print("equlibrium error: {:.2e}".format(eq_error(eq)))

except Exception:
    try:
        fam.save(str(FAILURE_PATH))
        print(f"saved failure file: {FAILURE_PATH}")

    except Exception:
        pass

    raise