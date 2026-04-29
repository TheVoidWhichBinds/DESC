#==============================================================================================================
# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/final.
#
# This file owns:
#   1. Paper registry access
#   2. Variant registry access
#   3. FixPressure removal for FLO/FNO runs
#   4. Initial-equilibrium injection through kwargs["thing"]
#   5. Symbolic rational-bound resolution
#   6. Saved equilibrium loading
#   7. DESC objective/constraint construction
#   8. Output-directory helpers
#   9. Result summaries and saving helpers
#
#==============================================================================================================

from copy import deepcopy
from pathlib import Path
import json

from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    QuasisymmetryBoozer,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixCurrent,
    GenericObjective,
)

from base import PAPERS










#==============================================================================================================
# DESC Class Registries
#==============================================================================================================

OBJECTIVE_REGISTRY = {
    "ForceBalance": ForceBalance,
    "QuasisymmetryBoozer": QuasisymmetryBoozer,
}


CONSTRAINT_REGISTRY = {
    "FixBoundaryR": FixBoundaryR,
    "FixBoundaryZ": FixBoundaryZ,
    "FixPressure": FixPressure,
    "FixCurrent": FixCurrent,
}










#==============================================================================================================
# Paper Helpers
#==============================================================================================================

def get_paper_config(
        paper_id,
    ):
    """
    Return the configuration dictionary for one paper.
    """

    if paper_id not in PAPERS:
        valid_papers = ", ".join(sorted(PAPERS.keys()))
        raise KeyError(
            f"Unknown paper_id '{paper_id}'. Valid paper IDs are: {valid_papers}"
        )

    return deepcopy(PAPERS[paper_id])










def list_papers():
    """
    Return available paper IDs.
    """

    return tuple(sorted(PAPERS.keys()))










#==============================================================================================================
# Saved Equilibrium Helpers
#==============================================================================================================

def load_saved_equilibrium(
        eq_path,
    ):
    """
    Load a saved DESC equilibrium object.
    """

    eq_path = Path(eq_path)

    if not eq_path.exists():
        raise FileNotFoundError(
            f"Could not find saved equilibrium file: {eq_path}"
        )

    eq = Equilibrium.load(
        load_from = str(eq_path),
    )

    return eq










#==============================================================================================================
# Variant Helpers
#==============================================================================================================

def get_variant_config(
        variant,
    ):
    """
    Return the variant configuration.

    Parameters
    ----------
    variant : str or None
        None, "FLO", or "FNO".

    Returns
    -------
    dict
        Variant configuration with objectives and constraints.
    """

    if variant is None:
        return {
            "objectives": (),
            "constraints": (),
        }

    if variant == "FLO":
        from FLO import FLO_CONFIG

        return deepcopy(FLO_CONFIG)

    if variant == "FNO":
        from FNO import FNO_CONFIG

        return deepcopy(FNO_CONFIG)

    raise ValueError(
        f"Unknown variant '{variant}'. Valid variants are: None, 'FLO', 'FNO'."
    )










def variant_label(
        variant,
    ):
    """
    Return a filesystem-safe label for one variant.
    """

    if variant is None:
        return "base"

    return str(variant)










#==============================================================================================================
# Constraint / Objective Config Helpers
#==============================================================================================================

def remove_fix_pressure(
        constraint_configs,
    ):
    """
    Remove FixPressure from a tuple/list of constraint config dictionaries.
    """

    return tuple(
        config for config in constraint_configs
        if config.get("name") != "FixPressure"
    )










def inject_initial_equilibrium(
        configs,
        eq_initial,
    ):
    """
    Fill kwargs["thing"] with the initial equilibrium object when present.
    """

    updated_configs = []

    for config in configs:
        config = deepcopy(config)
        kwargs = dict(config.get("kwargs", {}))

        if "thing" in kwargs:
            kwargs["thing"] = eq_initial

        config["kwargs"] = kwargs
        updated_configs.append(config)

    return tuple(updated_configs)










def iota_between_rationals(
        iota_axis,
    ):
    """
    Return neighboring low-order rational bounds around the on-axis iota.
    """

    rationals = (
        0.25,
        1.0 / 3.0,
        0.5,
        2.0 / 3.0,
        0.75,
        1.0,
        4.0 / 3.0,
        1.5,
        2.0,
        3.0,
        4.0,
    )

    iota_axis = float(iota_axis)

    if iota_axis <= rationals[0]:
        return rationals[0], rationals[1]

    if iota_axis >= rationals[-1]:
        return rationals[-2], rationals[-1]

    for lower_rational, upper_rational in zip(rationals[:-1], rationals[1:]):
        if lower_rational <= iota_axis <= upper_rational:
            return lower_rational, upper_rational

    raise RuntimeError(
        f"Could not find rational bounds for iota_axis = {iota_axis}."
    )










def resolve_symbolic_bounds(
        configs,
        lower_rational,
        upper_rational,
    ):
    """
    Replace symbolic rational-bound entries with numeric bounds.
    """

    updated_configs = []

    for config in configs:
        config = deepcopy(config)

        if config.get("bounds") == ("lower_rational", "upper_rational"):
            config["bounds"] = (lower_rational, upper_rational)

        updated_configs.append(config)

    return tuple(updated_configs)










def get_iota_axis_from_equilibrium(
        eq,
    ):
    """
    Extract the initial on-axis iota value from an equilibrium object.
    """

    if hasattr(eq, "i_l"):
        return float(eq.i_l[0])

    if hasattr(eq, "iota") and hasattr(eq.iota, "params"):
        return float(eq.iota.params[0])

    raise AttributeError(
        "Could not extract on-axis iota. Expected eq.i_l or eq.iota.params."
    )










def prepare_variant_configs(
        paper_config,
        variant = None,
        eq_initial = None,
    ):
    """
    Merge base paper configs with optional FLO/FNO variant configs.
    """

    base_objectives = tuple(paper_config["objectives"])
    base_constraints = tuple(paper_config["constraints"])

    variant_config = get_variant_config(
        variant = variant,
    )

    if variant in ("FLO", "FNO"):
        base_constraints = remove_fix_pressure(
            constraint_configs = base_constraints,
        )

        iota_axis = get_iota_axis_from_equilibrium(
            eq = eq_initial,
        )

        lower_rational, upper_rational = iota_between_rationals(
            iota_axis = iota_axis,
        )

        variant_objectives = inject_initial_equilibrium(
            configs = tuple(variant_config["objectives"]),
            eq_initial = eq_initial,
        )

        variant_constraints = inject_initial_equilibrium(
            configs = tuple(variant_config["constraints"]),
            eq_initial = eq_initial,
        )

        variant_objectives = resolve_symbolic_bounds(
            configs = variant_objectives,
            lower_rational = lower_rational,
            upper_rational = upper_rational,
        )

        variant_constraints = resolve_symbolic_bounds(
            configs = variant_constraints,
            lower_rational = lower_rational,
            upper_rational = upper_rational,
        )

    else:
        variant_objectives = ()
        variant_constraints = ()

    objective_configs = base_objectives + variant_objectives
    constraint_configs = base_constraints + variant_constraints

    return objective_configs, constraint_configs










#==============================================================================================================
# DESC Objective / Constraint Builders
#==============================================================================================================

def build_named_objective(
        config,
        eq,
    ):
    """
    Build a standard DESC objective from a config dictionary.
    """

    name = config["name"]
    kwargs = dict(config.get("kwargs", {}))

    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(
            f"Unknown base objective '{name}'. Add it to OBJECTIVE_REGISTRY in helper.py."
        )

    objective_cls = OBJECTIVE_REGISTRY[name]

    return objective_cls(
        eq = eq,
        **kwargs,
    )










def build_named_constraint(
        config,
        eq,
    ):
    """
    Build a standard DESC constraint from a config dictionary.
    """

    name = config["name"]
    kwargs = dict(config.get("kwargs", {}))

    if name not in CONSTRAINT_REGISTRY:
        raise KeyError(
            f"Unknown base constraint '{name}'. Add it to CONSTRAINT_REGISTRY in helper.py."
        )

    constraint_cls = CONSTRAINT_REGISTRY[name]

    return constraint_cls(
        eq = eq,
        **kwargs,
    )










def build_generic_objective(
        config,
        eq,
    ):
    """
    Build a DESC GenericObjective from a FLO/FNO config dictionary.
    """

    kwargs = dict(config.get("kwargs", {}))

    generic_kwargs = {
        "fun": config["fun"],
        "eq": eq,
        **kwargs,
    }

    if "target" in config:
        generic_kwargs["target"] = config["target"]

    if "bounds" in config:
        generic_kwargs["bounds"] = config["bounds"]

    return GenericObjective(
        **generic_kwargs,
    )










def build_objective(
        config,
        eq,
    ):
    """
    Build either a standard DESC objective or a GenericObjective.
    """

    if "fun" in config:
        return build_generic_objective(
            config = config,
            eq = eq,
        )

    return build_named_objective(
        config = config,
        eq = eq,
    )










def build_constraint(
        config,
        eq,
    ):
    """
    Build either a standard DESC constraint or a GenericObjective constraint.
    """

    if "fun" in config:
        return build_generic_objective(
            config = config,
            eq = eq,
        )

    return build_named_constraint(
        config = config,
        eq = eq,
    )










def build_objective_function(
        objective_configs,
        eq,
    ):
    """
    Build DESC ObjectiveFunction from objective config dictionaries.
    """

    objectives = [
        build_objective(
            config = config,
            eq = eq,
        )
        for config in objective_configs
    ]

    return ObjectiveFunction(
        objectives = objectives,
    )










def build_constraints(
        constraint_configs,
        eq,
    ):
    """
    Build a tuple of DESC constraints from constraint config dictionaries.
    """

    constraints = tuple(
        build_constraint(
            config = config,
            eq = eq,
        )
        for config in constraint_configs
    )

    return constraints










#==============================================================================================================
# Output Helpers
#==============================================================================================================

def make_output_dir(
        output_dir,
    ):
    """
    Create and return an output directory path.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    return output_dir










def save_json(
        data,
        path,
    ):
    """
    Save JSON data.
    """

    path = Path(path)
    path.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    with open(path, "w") as file:
        json.dump(
            data,
            file,
            indent = 4,
            default = str,
        )










def save_text(
        text,
        path,
    ):
    """
    Save text to a file.
    """

    path = Path(path)
    path.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    with open(path, "w") as file:
        file.write(text)










#==============================================================================================================
# Result Summary / Saving Helpers
#==============================================================================================================

def summarize_result(
        result,
    ):
    """
    Convert DESC optimizer result into a lightweight serializable summary.
    """

    if result is None:
        return {
            "success": False,
            "message": "No result object returned.",
        }

    summary = {
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
    }

    for key in (
        "nit",
        "nfev",
        "njev",
        "optimality",
        "cost",
        "fun",
    ):
        if hasattr(result, key):
            summary[key] = getattr(result, key)

    return summary










def save_optimization_outputs(
        eq,
        result,
        output_dir,
        label,
    ):
    """
    Save optimized equilibrium and optimizer summary.
    """

    output_dir = make_output_dir(
        output_dir = output_dir,
    )

    eq_path = output_dir / f"{label}_optimized.h5"
    summary_path = output_dir / f"{label}_result_summary.json"

    eq.save(
        file_name = str(eq_path),
        overwrite = True,
    )

    save_json(
        data = summarize_result(
            result = result,
        ),
        path = summary_path,
    )

    return {
        "eq_path": str(eq_path),
        "summary_path": str(summary_path),
    }