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
#   3. Optimization settings
#   4. Base objectives
#   5. Base constraints
#   6. Plot settings
#
# FLO and FNO objectives are intentionally NOT included here. They are defined in FLO.py and FNO.py,
# using functions from VFOs.py, and are appended only when variant = "FLO" or variant = "FNO".
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

        "eq_path": "research/final/equilibria/dudt2022.h5",

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

        "plots": {
            "figures": (
                "objective_history",
                "force_error",
                "boundary",
                "comparison_table",
            ),

            "kwargs": {
                "figsize": (7, 5),
                "dpi": 300,
                "save_format": "png",
            },
        },
    },










    "conlin2022": {
        "paper": {
            "main_author": "Conlin",
            "year": 2022,
            "label": "Conlin 2022",
            "notes": "DESC continuation / equilibrium construction cases.",
        },

        "eq_path": "research/final/equilibria/conlin2022.h5",

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

        "plots": {
            "figures": (
                "objective_history",
                "force_error",
                "boundary",
                "comparison_table",
            ),

            "kwargs": {
                "figsize": (7, 5),
                "dpi": 300,
                "save_format": "png",
            },
        },
    },










    "panici2023": {
        "paper": {
            "main_author": "Panici",
            "year": 2023,
            "label": "Panici 2023",
            "notes": "DESC paper involving continuation and nonlinear equilibrium solve methodology.",
        },

        "eq_path": "research/final/equilibria/panici2023.h5",

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

        "plots": {
            "figures": (
                "objective_history",
                "force_error",
                "boundary",
                "comparison_table",
            ),

            "kwargs": {
                "figsize": (7, 5),
                "dpi": 300,
                "save_format": "png",
            },
        },
    },










    "dudt2024": {
        "paper": {
            "main_author": "Dudt",
            "year": 2024,
            "label": "Dudt 2024",
            "notes": "DESC optimization paper with additional physics objectives.",
        },

        "eq_path": "research/final/equilibria/dudt2024.h5",

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

        "plots": {
            "figures": (
                "objective_history",
                "force_error",
                "boundary",
                "comparison_table",
            ),

            "kwargs": {
                "figsize": (7, 5),
                "dpi": 300,
                "save_format": "png",
            },
        },
    },










    "conlin2024constraints": {
        "paper": {
            "main_author": "Conlin",
            "year": 2024,
            "label": "Conlin 2024 Constraints",
            "notes": "DESC constraints paper.",
        },

        "eq_path": "research/final/equilibria/conlin2024constraints.h5",

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

        "plots": {
            "figures": (
                "objective_history",
                "force_error",
                "boundary",
                "comparison_table",
            ),

            "kwargs": {
                "figsize": (7, 5),
                "dpi": 300,
                "save_format": "png",
            },
        },
    },
}
