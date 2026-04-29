# research/final/papers.py











#============== PAPER CONFIG REGISTRY ==========================================================================
"""
Paper-level configuration registry for research/final.

This file intentionally stores paper information as plain dictionaries, not
DESC class objects. The runner resolves strings such as "ForceBalance" or
"QuasisymmetryBoozer" into the current DESC API in src/builders.py.

Rules:
    1. Do not import old publication driver.py files.
    2. Do not store DESC objective/constraint instances here.
    3. Put paper-specific physics here.
    4. Put modern DESC object construction in src/builders.py.
"""

from copy import deepcopy
from pathlib import Path











#============== PATHS ==========================================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLICATIONS_DIR = REPO_ROOT / "publications"
OUTPUT_ROOT = REPO_ROOT / "research" / "final" / "runs"











#============== SHARED DEFAULTS ================================================================================
DEFAULT_SOLVE_CONFIG = {
    "run_continuation": True,
    "continuation_kwargs": {
        "objective": "force",
        "verbose": 3,
    },
    "solve_kwargs": {
        "verbose": 3,
    },
}


DEFAULT_OPTIMIZATION_CONFIG = {
    "optimizer": "proximal-lsq-exact",
    "optimize_kwargs": {
        "verbose": 3,
        "maxiter": 100,
        "ftol": 1.0e-8,
        "xtol": 1.0e-8,
        "gtol": 1.0e-8,
    },
}


DEFAULT_OUTPUT_CONFIG = {
    "save_equilibrium": True,
    "save_status": True,
    "save_plots": True,
    "save_objective_table": True,
}











#============== OBJECTIVE / METRIC TEMPLATES ===================================================================
OBJECTIVE_TEMPLATES = {
    "force_balance": {
        "name": "ForceBalance",
        "kwargs": {},
        "weight": 1.0,
    },

    "qs_boozer": {
        "name": "QuasisymmetryBoozer",
        "kwargs": {
            "helicity": [1, "NFP"],
        },
        "weight": 1.0,
    },

    "qs_two_term": {
        "name": "QuasisymmetryTwoTerm",
        "kwargs": {
            "helicity": [1, "NFP"],
        },
        "weight": 1.0,
        "optional": True,
    },

    "qs_triple_product": {
        "name": "QuasisymmetryTripleProduct",
        "kwargs": {
            "helicity": [1, "NFP"],
        },
        "weight": 1.0,
        "optional": True,
    },

    "magnetic_well": {
        "name": "MagneticWell",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "mercier": {
        "name": "MercierStability",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "good_coordinates": {
        "name": "GoodCoordinates",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "mean_curvature": {
        "name": "MeanCurvature",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "principal_curvature": {
        "name": "PrincipalCurvature",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "aspect_ratio": {
        "name": "AspectRatio",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "volume": {
        "name": "Volume",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "elongation": {
        "name": "Elongation",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
    },

    "quasiisodynamicity": {
        "name": "QuasiIsodynamicResidual",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
        "notes": "Not a standard DESC objective name in all versions. Keep optional.",
    },

    "greene_residue": {
        "name": "GreeneResidue",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
        "notes": "SPEC/SIMSOPT-style good-surface metric, not a generic DESC equilibrium objective.",
    },

    "coil_nested_surface_residual": {
        "name": "NestedSurfaceCoilResidual",
        "kwargs": {},
        "weight": 1.0,
        "optional": True,
        "notes": "Coil-space metric from direct coil optimization papers; not a standard DESC equilibrium objective.",
    },
}











#============== CONSTRAINT TEMPLATES ===========================================================================
CONSTRAINT_TEMPLATES = {
    "fixed_boundary": [
        {
            "name": "FixBoundaryR",
            "kwargs": {},
        },
        {
            "name": "FixBoundaryZ",
            "kwargs": {},
        },
    ],

    "fixed_pressure": [
        {
            "name": "FixPressure",
            "kwargs": {},
        },
    ],

    "fixed_iota": [
        {
            "name": "FixIota",
            "kwargs": {},
        },
    ],

    "fixed_current": [
        {
            "name": "FixCurrent",
            "kwargs": {},
        },
    ],

    "fixed_axis": [
        {
            "name": "FixAxisR",
            "kwargs": {},
            "optional": True,
        },
        {
            "name": "FixAxisZ",
            "kwargs": {},
            "optional": True,
        },
    ],
}











#============== HELPERS ========================================================================================
def _objectives(*template_names):
    objectives = []

    for template_name in template_names:
        objectives.append(deepcopy(OBJECTIVE_TEMPLATES[template_name]))

    return objectives





def _constraints(*template_names):
    constraints = []

    for template_name in template_names:
        constraints.extend(deepcopy(CONSTRAINT_TEMPLATES[template_name]))

    return constraints





def _desc_h5_equilibrium(
        source_path,
        L = None,
        M = None,
        N = None,
        L_grid = None,
        M_grid = None,
        N_grid = None,
        NFP = None,
        sym = True,
    ):
    return {
        "source_type": "desc_h5",
        "source_path": str(source_path),
        "NFP": NFP,
        "sym": sym,
        "resolution": {
            "L": L,
            "M": M,
            "N": N,
            "L_grid": L_grid,
            "M_grid": M_grid,
            "N_grid": N_grid,
        },
        "profiles": {
            "pressure": None,
            "iota": None,
            "current": None,
        },
        "surface": None,
    }





def _unknown_equilibrium(
        source_notes,
        NFP = None,
        sym = True,
    ):
    return {
        "source_type": "unknown",
        "source_path": None,
        "source_notes": source_notes,
        "NFP": NFP,
        "sym": sym,
        "resolution": {
            "L": None,
            "M": None,
            "N": None,
            "L_grid": None,
            "M_grid": None,
            "N_grid": None,
        },
        "profiles": {
            "pressure": None,
            "iota": None,
            "current": None,
        },
        "surface": None,
    }





def _paper(
        paper_id,
        title,
        authors,
        year,
        publication_folder,
        equilibrium,
        objectives,
        constraints,
        primary_metrics,
        cases = None,
        solve = None,
        optimization = None,
        outputs = None,
        notes = None,
    ):
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "publication_folder": publication_folder,
        "equilibrium": equilibrium,
        "primary_metrics": primary_metrics,
        "solve": deepcopy(solve or DEFAULT_SOLVE_CONFIG),
        "optimization": {
            **deepcopy(DEFAULT_OPTIMIZATION_CONFIG),
            **deepcopy(optimization or {}),
            "objectives": objectives,
            "constraints": constraints,
        },
        "cases": deepcopy(cases or {"base": {}}),
        "variants": {
            "base": {
                "description": "Original paper-style objective construction.",
                "add_objectives": [],
                "remove_objectives": [],
                "override_optimization": {},
            },

            "flo": {
                "description": "Original paper-style objectives plus linear feature objective constraints.",
                "add_objectives": [
                    {
                        "name": "FLOPressureAxis",
                        "kwargs": {},
                        "weight": 1.0,
                        "optional": True,
                    },
                    {
                        "name": "FLOPressureShape",
                        "kwargs": {},
                        "weight": 1.0,
                        "optional": True,
                    },
                    {
                        "name": "FLOIotaAxis",
                        "kwargs": {},
                        "weight": 1.0,
                        "optional": True,
                    },
                    {
                        "name": "FLOIotaEdge",
                        "kwargs": {},
                        "weight": 1.0,
                        "optional": True,
                    },
                ],
                "remove_objectives": [],
                "override_optimization": {},
            },

            "fno": {
                "description": "Original paper-style objectives plus nonlinear feature objective constraints.",
                "add_objectives": [
                    {
                        "name": "FNOPressureProfile",
                        "kwargs": {},
                        "weight": 1.0,
                        "optional": True,
                    },
                    {
                        "name": "FNOIotaProfile",
                        "kwargs": {},
                        "weight": 1.0,
                        "optional": True,
                    },
                ],
                "remove_objectives": [],
                "override_optimization": {},
            },
        },
        "outputs": deepcopy(outputs or DEFAULT_OUTPUT_CONFIG),
        "notes": notes or [],
    }











#============== PAPER CONFIGS ==================================================================================
PAPER_CONFIGS = {
    "dudt2022_desc_qs": _paper(
        paper_id = "dudt2022_desc_qs",
        title = "The DESC Stellarator Code Suite Part III: Quasi-symmetry Optimization",
        authors = "Dudt et al.",
        year = 2022,
        publication_folder = "publications/dudt2022",
        equilibrium = _desc_h5_equilibrium(
            source_path = PUBLICATIONS_DIR / "dudt2022" / "data" / "initial.h5",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "qs_boozer",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "quasisymmetry",
            "force_balance",
            "optimization_path_in_boundary_mode_subspace",
        ],
        cases = {
            "fB_or1": {
                "qs_metric": "B",
                "perturbation_order": 1,
                "reference_equilibrium": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "eq_fB_or1.h5"),
            },
            "fB_or2": {
                "qs_metric": "B",
                "perturbation_order": 2,
                "reference_equilibrium": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "eq_fB_or2.h5"),
            },
            "fC_or1": {
                "qs_metric": "C",
                "perturbation_order": 1,
                "reference_equilibrium": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "eq_fC_or1.h5"),
            },
            "fC_or2": {
                "qs_metric": "C",
                "perturbation_order": 2,
                "reference_equilibrium": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "eq_fC_or2.h5"),
            },
            "fT_or1": {
                "qs_metric": "T",
                "perturbation_order": 1,
                "reference_equilibrium": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "eq_fT_or1.h5"),
            },
            "fT_or2": {
                "qs_metric": "T",
                "perturbation_order": 2,
                "reference_equilibrium": str(PUBLICATIONS_DIR / "dudt2022" / "data" / "eq_fT_or2.h5"),
            },
        },
        notes = [
            "This is the only listed paper here for which the visible DESC/publications folder exposes concrete reproduction files.",
            "Original publication driver used old APIs. This config does not import that driver.",
            "Exact old qs='B'/'C'/'T' implementation should be mapped to modern objective names in src/builders.py if needed.",
        ],
    ),

    "landreman_paul2022_precise_qs": _paper(
        paper_id = "landreman_paul2022_precise_qs",
        title = "Magnetic Fields with Precise Quasisymmetry for Plasma Confinement",
        authors = "Landreman and Paul",
        year = 2022,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "Near-axis / precise-quasisymmetry construction. Not visible as a DESC/publications folder in the current repository listing.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "qs_boozer",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "quasisymmetry",
            "effective_ripple_or_particle_confinement",
            "force_balance",
        ],
        notes = [
            "Treat as QS reproduction once exact boundary/equilibrium data are provided.",
            "Do not guess Fourier coefficients. Fill from paper supplement or local source files.",
        ],
    ),

    "landreman_jorge2020_magnetic_well_mercier": _paper(
        paper_id = "landreman_jorge2020_magnetic_well_mercier",
        title = "Magnetic Well and Mercier-Stable Stellarators Near the Magnetic Axis",
        authors = "Landreman and Jorge",
        year = 2020,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "Near-axis expansion configuration. Not visible as a DESC/publications folder in the current repository listing.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "magnetic_well",
            "mercier",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "magnetic_well",
            "Mercier_stability",
            "near_axis_stability",
        ],
        notes = [
            "Primary reproduction target is stability near the magnetic axis, not only final QS error.",
            "Exact near-axis parameters must be filled from source/supplement.",
        ],
    ),

    "jorge_sengupta_landreman2020_direct_coordinate_qs": _paper(
        paper_id = "jorge_sengupta_landreman2020_direct_coordinate_qs",
        title = "Construction of Quasisymmetric Stellarators Using a Direct Coordinate Approach",
        authors = "Jorge, Sengupta, and Landreman",
        year = 2020,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "Direct-coordinate construction, not a standard fixed-boundary DESC optimization entry in the visible publications folder.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "qs_boozer",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "quasisymmetry",
            "direct_coordinate_residual",
            "force_balance",
        ],
        notes = [
            "Use as a constructed-equilibrium config once the direct-coordinate coefficients are converted into DESC boundary/profile inputs.",
        ],
    ),

    "goodman2023_precise_qi": _paper(
        paper_id = "goodman2023_precise_qi",
        title = "Constructing Precisely Quasi-Isodynamic Magnetic Fields",
        authors = "Goodman et al.",
        year = 2023,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "Quasi-isodynamic configurations. Not visible as a DESC/publications folder in the current repository listing.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "quasiisodynamicity",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "quasiisodynamicity",
            "fast_particle_confinement",
            "neoclassical_transport",
        ],
        notes = [
            "The title in your list says quasisymmetric, but the identifiable Goodman et al. paper is quasi-isodynamic.",
            "The quasi-isodynamic residual is marked optional because current DESC may not expose this as a built-in objective.",
        ],
    ),

    "bader2020_new_qhs": _paper(
        paper_id = "bader2020_new_qhs",
        title = "A New Optimized Quasihelically Symmetric Stellarator",
        authors = "Bader et al.",
        year = 2020,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "QHS configuration. Not visible as a DESC/publications folder in the current repository listing.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "qs_boozer",
            "aspect_ratio",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "quasihelical_symmetry",
            "aspect_ratio",
            "effective_ripple_or_particle_confinement",
        ],
        notes = [
            "Your supplied title says good magnetic surfaces at the plasma edge; the identifiable Bader paper is a QHS optimization paper.",
            "Fill exact boundary/profile data from supplement or local files before running as a reproduction.",
        ],
    ),

    "landreman_medasani_zhu2021_good_surfaces": _paper(
        paper_id = "landreman_medasani_zhu2021_good_surfaces",
        title = "Stellarator Optimization for Good Magnetic Surfaces at the Same Time as Quasisymmetry",
        authors = "Landreman, Medasani, and Zhu",
        year = 2021,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "SPEC/SIMSOPT-style good-surface optimization; not visible as a DESC/publications folder in the current repository listing.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "qs_boozer",
            "greene_residue",
        ),
        constraints = [],
        primary_metrics = [
            "Greene_residue",
            "island_suppression",
            "quasisymmetry",
            "good_magnetic_surfaces",
        ],
        notes = [
            "This is included because it matches the good-surface title from your message more closely than Bader et al.",
            "Greene residue is not a standard DESC fixed-boundary equilibrium objective; use this as a reference/diagnostic config unless you implement the metric.",
        ],
    ),

    "henneberg2019_new_qa": _paper(
        paper_id = "henneberg2019_new_qa",
        title = "Properties of a New Quasi-Axisymmetric Configuration",
        authors = "Henneberg et al.",
        year = 2019,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "QA configuration. Not visible as a DESC/publications folder in the current repository listing.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "force_balance",
            "qs_boozer",
            "mercier",
        ),
        constraints = _constraints(
            "fixed_boundary",
            "fixed_pressure",
            "fixed_iota",
        ),
        primary_metrics = [
            "quasi_axisymmetry",
            "MHD_stability",
            "fast_particle_confinement",
            "effective_ripple",
        ],
        notes = [
            "This is primarily a properties/configuration paper. Treat as reproduce equilibrium then evaluate diagnostics.",
        ],
    ),

    "giuliani2023_direct_coil_nested_qs": _paper(
        paper_id = "giuliani2023_direct_coil_nested_qs",
        title = "Direct Stellarator Coil Optimization for Nested Magnetic Surfaces with Precise Quasi-Symmetry",
        authors = "Giuliani et al.",
        year = 2023,
        publication_folder = None,
        equilibrium = _unknown_equilibrium(
            source_notes = "Direct coil optimization. This is not a standard DESC fixed-boundary equilibrium reproduction without coil-field machinery.",
            NFP = None,
            sym = True,
        ),
        objectives = _objectives(
            "coil_nested_surface_residual",
            "qs_boozer",
        ),
        constraints = [],
        primary_metrics = [
            "nested_flux_surfaces",
            "coil_optimization",
            "quasisymmetry",
            "island_healing",
        ],
        notes = [
            "This should not be run through the same path as fixed-boundary DESC equilibrium optimization unless you provide a DESC equilibrium extracted from the coil result.",
            "Use as a diagnostic/reference config unless coil-field objectives are implemented.",
        ],
    ),
}











#============== PUBLIC API =====================================================================================
def list_papers():
    return sorted(PAPER_CONFIGS.keys())





def get_paper_config(paper_id):
    if paper_id not in PAPER_CONFIGS:
        valid = ", ".join(list_papers())
        raise KeyError(f"Unknown paper_id={paper_id!r}. Valid paper ids: {valid}")

    return deepcopy(PAPER_CONFIGS[paper_id])





def get_case_config(paper_id, case_id = "base"):
    paper_config = get_paper_config(paper_id)
    cases = paper_config.get("cases", {})

    if case_id not in cases:
        valid = ", ".join(sorted(cases.keys()))
        raise KeyError(f"Unknown case_id={case_id!r} for paper_id={paper_id!r}. Valid case ids: {valid}")

    return deepcopy(cases[case_id])





def get_variant_config(paper_id, variant_id = "base"):
    paper_config = get_paper_config(paper_id)
    variants = paper_config.get("variants", {})

    if variant_id not in variants:
        valid = ", ".join(sorted(variants.keys()))
        raise KeyError(f"Unknown variant_id={variant_id!r} for paper_id={paper_id!r}. Valid variant ids: {valid}")

    return deepcopy(variants[variant_id])





def compose_run_config(
        paper_id,
        case_id = "base",
        variant_id = "base",
    ):
    paper_config = get_paper_config(paper_id)
    case_config = get_case_config(paper_id, case_id)
    variant_config = get_variant_config(paper_id, variant_id)

    run_config = deepcopy(paper_config)
    run_config["case_id"] = case_id
    run_config["variant_id"] = variant_id
    run_config["case"] = case_config
    run_config["variant"] = variant_config

    optimization = run_config["optimization"]

    remove_objectives = set(variant_config.get("remove_objectives", []))
    base_objectives = [
        objective
        for objective in optimization.get("objectives", [])
        if objective.get("name") not in remove_objectives
    ]

    optimization["objectives"] = base_objectives + deepcopy(variant_config.get("add_objectives", []))
    optimization.update(deepcopy(variant_config.get("override_optimization", {})))

    run_dir = OUTPUT_ROOT / paper_id / case_id / variant_id
    run_config["run_dir"] = str(run_dir)

    return run_config