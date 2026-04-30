# base.py
#==============================================================================================================
#
# Paper-specific configuration registry for DESC/research/final.
#
# This file should contain only configuration dictionaries. It should not construct DESC objects directly.
#
# Each paper entry contains:
#   1. Paper metadata
#   2. Path to a saved initial equilibrium object
#   3. Equilibrium information
#   4. Optimization settings
#   5. Base objectives
#   6. Base constraints
#
# FLO and FNO objectives are intentionally NOT included here. They are defined in FLO.py and FNO.py,
# using functions from VFOs.py, and are appended only when variant = "FLO" or variant = "FNO".
#
# Plotting is intentionally NOT included here. Plotting lives in plot.py.
#
#==============================================================================================================










#==============================================================================================================
# Paper Configuration Registry
#==============================================================================================================

PAPERS = {
    "dudt2022": {
        "paper": {
            "main_author": "Dudt",
            "year": 2022,
            "label": "Dudt 2022",
            "notes": "DESC quasi-symmetry optimization paper.",
        },

        "eq_path": "publications/dudt2022/data/initial.h5",

        "equilibrium": {
            "NFP": 2,
            "helicity": (1, 2),
            "L": None,
            "M": None,
            "N": None,
            "sym": True,
            "notes": (
                "Initial equilibrium is loaded from the saved publication h5 file. "
                "Resolution, pressure, current, and iota information should be read from the saved equilibrium."
            ),
        },

        "optimization": {
            "optimizer": "proximal-lsq-exact",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "maxiter": 100,
            "verbose": 3,
        },

        "objectives": (
            {
                "name": "ForceBalance",
                "kwargs": {
                    "weight": 1,
                },
            },
            {
                "name": "QuasisymmetryBoozer",
                "kwargs": {
                    "helicity": (1, 2),
                    "weight": 1,
                },
            },
        ),

        "constraints": (
            {
                "name": "FixBoundaryR",
                "kwargs": {},
            },
            {
                "name": "FixBoundaryZ",
                "kwargs": {},
            },
            {
                "name": "FixPressure",
                "kwargs": {},
            },
            {
                "name": "FixCurrent",
                "kwargs": {},
            },
        ),
    },










    "conlin2022": {
        "paper": {
            "main_author": "Conlin",
            "year": 2022,
            "label": "Conlin 2022",
            "notes": "DESC continuation / equilibrium construction cases.",
        },

        "eq_path": "research/final/equilibria/conlin2022.h5",

        "equilibrium": {
            "NFP": None,
            "helicity": None,
            "L": None,
            "M": None,
            "N": None,
            "sym": None,
            "notes": (
                "Placeholder until the recreated Conlin 2022 initial equilibrium is saved. "
                "Use the saved h5 file as source of truth for resolution and profiles."
            ),
        },

        "optimization": {
            "optimizer": "proximal-lsq-exact",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "maxiter": 100,
            "verbose": 3,
        },

        "objectives": (
            {
                "name": "ForceBalance",
                "kwargs": {
                    "weight": 1,
                },
            },
        ),

        "constraints": (
            {
                "name": "FixBoundaryR",
                "kwargs": {},
            },
            {
                "name": "FixBoundaryZ",
                "kwargs": {},
            },
            {
                "name": "FixPressure",
                "kwargs": {},
            },
            {
                "name": "FixCurrent",
                "kwargs": {},
            },
        ),
    },










    "panici2023": {
        "paper": {
            "main_author": "Panici",
            "year": 2023,
            "label": "Panici 2023",
            "notes": "DESC paper involving continuation and nonlinear equilibrium solve methodology.",
        },

        "input_eq_path": "research/final/equilibria/panici2023/input/W7X_M16_N16_ansi_cpu1_f2_compute_branch",
        "output_eq_path": "research/final/equilibria/panici2023/output/W7X_M16_N16_ansi_cpu1_f2_compute_branch_output.h5",

        "equilibrium": {
            "NFP": None,
            "helicity": None,
            "L": 16,
            "M": 16,
            "N": 16,
            "sym": None,
            "notes": (
                "Placeholder until the recreated Panici 2023 initial equilibrium is saved. "
                "Use the saved h5 file as source of truth for resolution and profiles."
            ),
        },

        "optimization": {
            "optimizer": "proximal-lsq-exact",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "maxiter": 100,
            "verbose": 3,
        },

        "objectives": (
            {
                "name": "ForceBalance",
                "kwargs": {
                    "weight": 1,
                },
            },
        ),

        "constraints": (
            {
                "name": "FixBoundaryR",
                "kwargs": {},
            },
            {
                "name": "FixBoundaryZ",
                "kwargs": {},
            },
            {
                "name": "FixPressure",
                "kwargs": {},
            },
            {
                "name": "FixCurrent",
                "kwargs": {},
            },
        ),
    },










    "dudt2024": {
        "paper": {
            "main_author": "Dudt",
            "year": 2024,
            "label": "Dudt 2024",
            "notes": "DESC optimization paper with additional physics objectives.",
        },

        "eq_path": "research/final/equilibria/dudt2024.h5",

        "equilibrium": {
            "NFP": None,
            "helicity": None,
            "L": None,
            "M": None,
            "N": None,
            "sym": None,
            "notes": (
                "Placeholder until the recreated Dudt 2024 initial equilibrium is saved. "
                "Use the saved h5 file as source of truth for resolution, profiles, and added physics targets."
            ),
        },

        "optimization": {
            "optimizer": "proximal-lsq-exact",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "maxiter": 100,
            "verbose": 3,
        },

        "objectives": (
            {
                "name": "ForceBalance",
                "kwargs": {
                    "weight": 1,
                },
            },
        ),

        "constraints": (
            {
                "name": "FixBoundaryR",
                "kwargs": {},
            },
            {
                "name": "FixBoundaryZ",
                "kwargs": {},
            },
            {
                "name": "FixPressure",
                "kwargs": {},
            },
            {
                "name": "FixCurrent",
                "kwargs": {},
            },
        ),
    },










    "conlin2024constraints": {
        "paper": {
            "main_author": "Conlin",
            "year": 2024,
            "label": "Conlin 2024 Constraints",
            "notes": "DESC constraints paper.",
        },

        "eq_path": "research/final/equilibria/conlin2024constraints.h5",

        "equilibrium": {
            "NFP": None,
            "helicity": None,
            "L": None,
            "M": None,
            "N": None,
            "sym": None,
            "notes": (
                "Placeholder until the recreated Conlin 2024 constraints equilibrium is saved. "
                "Use the saved h5 file as source of truth for resolution, profiles, and constraints."
            ),
        },

        "optimization": {
            "optimizer": "proximal-lsq-exact",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "maxiter": 100,
            "verbose": 3,
        },

        "objectives": (
            {
                "name": "ForceBalance",
                "kwargs": {
                    "weight": 1,
                },
            },
        ),

        "constraints": (
            {
                "name": "FixBoundaryR",
                "kwargs": {},
            },
            {
                "name": "FixBoundaryZ",
                "kwargs": {},
            },
            {
                "name": "FixPressure",
                "kwargs": {},
            },
            {
                "name": "FixCurrent",
                "kwargs": {},
            },
        ),
    },
}








