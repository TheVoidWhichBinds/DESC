from configs.global_config import PUBLICATIONS_DIR










#============== DUDT 2022 / DESC PART III QS CONFIG =============================================================
PAPER_CONFIG = {
    "paper_id": "dudt2022_qs",

    "paper_name": "Dudt et al. 2023 / DESC Part III: Quasi-symmetry optimization",

    "links": {
        "pdf": "https://arxiv.org/pdf/2204.00078",
        "code_data": "https://github.com/PlasmaControl/DESC/tree/master/publications/dudt2022",
    },

    "initial_equilibrium": {
        "source_type": "h5",
        "path": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "initial.h5"),
        "input_file": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "initial_input"),
        "load_final_member_if_family": True,
    },

    "physics": {
        "NFP": 4,
        "Psi": 4.444E-1,
    },

    "resolution": {
        "L": None,
        "M": 16,
        "N": 8,
        "extract_L_from_h5": True,
    },

    "continuation_resolution_sequence": {
        "M_pol": [8, 10, 12, 14, 16],
        "N_tor": [0, 2, 4, 6, 8],
    },

    "grid_resolution_sequence": {
        "M": [12, 15, 18, 21, 24],
        "N": [0, 3, 6, 9, 12],
    },

    "profile_config": {
        "profile_mode": "from_h5",

        "reference_pressure": {
            "type": "PowerSeriesProfile",
            "modes": [0, 2, 4, 6],
            "params": [1.0E3, -1.0E3, 0.0, 0.0],
            "description": "p(rho) = 1000 - 1000 rho^2",
            "fixed": True,
        },

        "reference_iota": {
            "type": "PowerSeriesProfile",
            "modes": [0, 2, 4, 6],
            "params": [
                1.03734956306034,
                4.59845331123661E-2,
                9.12296626003860E-2,
                4.37359156248255E-2,
            ],
            "fixed": True,
        },

        "pressure_fixed": True,
        "iota_fixed": True,
        "current_fixed": False,
    },

    "boundary_free_subspace": {
        "mode_description": "Special 2D degraded m = 1, n = 2 boundary subspace",
        "source_file": str(PUBLICATIONS_DIR / "dudt2022" / "driver.py"),
        "function_name": "getOptSubspace",
    },

    "paper_objectives": {
        "objective_family": "qs",

        "available_qs_objectives": {
            "B": {
                "class": "QuasisymmetryBoozer",
                "kwargs": {
                    "norm": True,
                    "helicity": ("one", "NFP"),
                    "M_booz": 24,
                    "N_booz": 12,
                },
            },

            "C": {
                "class": "QuasisymmetryFluxFunction",
                "kwargs": {
                    "norm": True,
                    "helicity": ("one", "NFP"),
                },
            },

            "T": {
                "class": "QuasisymmetryTripleProduct",
                "kwargs": {
                    "norm": True,
                },
            },
        },

        "qs": "B",
        "order": 1,
        "allowed_qs": ["B", "C", "T"],
        "allowed_order": [1, 2],
    },

    "paper_constraints": [
        {
            "class": "FixedBoundaryR",
            "kwargs": {},
        },
        {
            "class": "FixedBoundaryZ",
            "kwargs": {},
        },
        {
            "class": "FixedPressure",
            "kwargs": {},
        },
        {
            "class": "FixedIota",
            "kwargs": {},
        },
        {
            "class": "FixedPsi",
            "kwargs": {},
        },
        {
            "class": "LCFSBoundary",
            "kwargs": {},
        },
    ],

    "optimizer": {
        "method": "proximal-lsq-exact",
    },

    "perturb_options": {
        "dRb": True,
        "dZb": True,
        "opt_subspace_from_paper_driver": True,
        "order": 1,
        "verbose": 2,
    },

    "solve_options": {
        "verbose": 3,
    },

    "optimize_options": {
        "verbose": 3,
        "copy": True,
    },

    "frozen_keys": [
        "initial_equilibrium",
        "physics",
        "resolution",
        "continuation_resolution_sequence",
        "grid_resolution_sequence",
        "profile_config",
        "optimizer",
        "perturb_options",
        "solve_options",
        "optimize_options",
    ],
}
#==============================================================================================================