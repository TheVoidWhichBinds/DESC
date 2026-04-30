# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/final.
#
# This file owns:
#   1. Variant registry access
#   2. FixPressure/FixIota removal for FLO/FNO runs
#   3. Initial-equilibrium injection through kwargs["thing"]
#   4. Symbolic rational-bound resolution
#   5. Saved equilibrium loading
#   6. DESC objective/constraint construction
#   7. Recreation-file variant running
#   8. Output-directory helpers
#   9. Result summaries and saving helpers
#
#==============================================================================================================

from copy import deepcopy
from pathlib import Path
import inspect
import json
import os
import runpy
import traceback


from desc.optimize import Optimizer
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    QuasisymmetryBoozer,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixCurrent,
    ObjectiveFromUser,
    LinearObjectiveFromUser,
)

PAPERS = {}










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

    if hasattr(Equilibrium, "load"):
        return Equilibrium.load(
            load_from = str(eq_path),
        )

    from desc.io import load

    return load(
        load_from = str(eq_path),
    )









#==============================================================================================================
# Variant Helpers
#==============================================================================================================

def get_variant_config(
        variant,
    ):
    """
    Return the variant configuration.
    """

    if variant is None:
        return {
            "objectives": (),
            "constraints": (),
        }

    if variant == "FLO":
        try:
            from .FLO import FLO_CONFIG
        except ImportError:
            from FLO import FLO_CONFIG

        return deepcopy(FLO_CONFIG)

    if variant == "FNO":
        try:
            from .FNO import FNO_CONFIG
        except ImportError:
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

def remove_fix_pressure_iota(
        constraint_configs,
    ):
    """
    Remove fixed pressure/iota constraints from config dictionaries.
    """

    fixed_names = {
        "FixPressure",
        "PressureFixed",
        "IotaFixed",
        "FixIota",
        "FixIotaProfile",
    }

    return tuple(
        config for config in constraint_configs
        if config.get("name") not in fixed_names
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
        if eq_initial is None:
            raise ValueError(
                "eq_initial must be supplied when preparing FLO/FNO variant configs."
            )

        base_constraints = remove_fix_pressure_iota(
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









def user_function_kind(
        fun,
    ):
    """
    Determine whether a custom objective function is params-based or grid/data-based.
    """

    parameter_names = tuple(inspect.signature(fun).parameters.keys())

    if parameter_names == ("params",):
        return "linear"

    if parameter_names[:2] == ("grid", "data"):
        return "nonlinear"

    raise ValueError(
        f"Could not infer objective wrapper for function '{fun.__name__}'. "
        "Expected signature (params) or (grid, data)."
    )









def build_user_objective(
        config,
        eq,
    ):
    """
    Build a DESC custom objective from a FLO/FNO config dictionary.
    """

    kwargs = dict(config.get("kwargs", {}))
    fun = config["fun"]

    if kwargs.get("thing") is None:
        kwargs["thing"] = eq

    objective_kwargs = {
        "fun": fun,
        "name": config.get("name", fun.__name__),
        **kwargs,
    }

    if "target" in config:
        objective_kwargs["target"] = config["target"]

    if "bounds" in config:
        objective_kwargs["bounds"] = config["bounds"]

    kind = config.get("wrapper") or user_function_kind(
        fun = fun,
    )

    if kind == "linear":
        return LinearObjectiveFromUser(
            **objective_kwargs,
        )

    if kind == "nonlinear":
        return ObjectiveFromUser(
            **objective_kwargs,
        )

    raise ValueError(
        f"Unknown custom objective wrapper '{kind}' for {config.get('name', fun.__name__)}."
    )









def build_objective(
        config,
        eq,
    ):
    """
    Build either a standard DESC objective or a custom objective.
    """

    if "fun" in config:
        return build_user_objective(
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
    Build either a standard DESC constraint or a custom objective used as a constraint.
    """

    if "fun" in config:
        return build_user_objective(
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






#==============================================================================================================
# Plotting Helpers
#==============================================================================================================

import pandas as pd










#==============================================================================================================
# Output Path Helpers
#==============================================================================================================

def get_paper_outputs_dir(outputs_dir, paper_id):
    return Path(outputs_dir) / paper_id





def get_variant_outputs_dir(outputs_dir, paper_id, variant_id):
    return get_paper_outputs_dir(
        outputs_dir = outputs_dir,
        paper_id = paper_id,
    ) / variant_id





def get_paper_plot_dir(outputs_dir, paper_id):
    plot_dir = get_paper_outputs_dir(
        outputs_dir = outputs_dir,
        paper_id = paper_id,
    ) / "plots"

    plot_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    return plot_dir





def get_existing_variant_dirs(outputs_dir, paper_id, variant_ids):
    variant_dirs = {}

    for variant_id in variant_ids:
        variant_dir = get_variant_outputs_dir(
            outputs_dir = outputs_dir,
            paper_id = paper_id,
            variant_id = variant_id,
        )

        if variant_dir.exists():
            variant_dirs[variant_id] = variant_dir

    return variant_dirs










#==============================================================================================================
# Plot Data Loading Helpers
#==============================================================================================================

def load_objective_history(variant_dir):
    variant_dir = Path(variant_dir)

    possible_paths = (
        variant_dir / "objective_history.csv",
        variant_dir / "optimization_history.csv",
        variant_dir / "history.csv",
    )

    for possible_path in possible_paths:
        if possible_path.exists():
            history = pd.read_csv(possible_path)

            if "iteration" not in history.columns:
                history = history.reset_index().rename(
                    columns = {
                        "index": "iteration",
                    },
                )

            return history

    return None





def choose_objective_column(history):
    preferred_columns = (
        "objective",
        "objective_value",
        "total_objective",
        "loss",
        "cost",
    )

    for column in preferred_columns:
        if column in history.columns:
            return column

    numeric_columns = list(history.select_dtypes(include = "number").columns)

    if "iteration" in numeric_columns:
        numeric_columns.remove("iteration")

    if len(numeric_columns) == 0:
        return None

    return numeric_columns[-1]





def load_comparison_text(variant_dir):
    variant_dir = Path(variant_dir)

    possible_paths = (
        variant_dir / "comparison.txt",
        variant_dir / "optimization_status.txt",
        variant_dir / "status.txt",
    )

    for possible_path in possible_paths:
        if possible_path.exists():
            return possible_path.read_text()

    return None










#==============================================================================================================
# Recreation File Variant Helpers
#==============================================================================================================

FIXED_PRESSURE_IOTA_CLASS_NAMES = {
    "FixPressure",
    "PressureFixed",
    "IotaFixed",
    "FixIota",
    "FixIotaProfile",
}










def is_fixed_pressure_iota_constraint(
        constraint,
    ):
    """
    Return True when a constraint fixes pressure or iota profiles.
    """

    class_name = constraint.__class__.__name__

    if class_name in FIXED_PRESSURE_IOTA_CLASS_NAMES:
        return True

    name = getattr(
        constraint,
        "name",
        "",
    )

    name = str(name)

    return any(
        fixed_name in name
        for fixed_name in FIXED_PRESSURE_IOTA_CLASS_NAMES
    )









def remove_fixed_pressure_iota_constraints_from_objects(
        constraints,
    ):
    """
    Remove pressure/iota profile-fixing constraints from instantiated constraint objects.
    """

    if constraints is None:
        return ()

    kept_constraints = []

    for constraint in tuple(constraints):
        if is_fixed_pressure_iota_constraint(
                constraint = constraint,
            ):
            print(f"Removing fixed pressure/iota constraint: {constraint.__class__.__name__}")
            continue

        kept_constraints.append(constraint)

    return tuple(kept_constraints)









def find_equilibrium_in_object(
        obj,
    ):
    """
    Find the first Equilibrium object inside an object, tuple, or list.
    """

    if isinstance(obj, Equilibrium):
        return obj

    if isinstance(obj, (tuple, list)):
        for item in obj:
            eq = find_equilibrium_in_object(
                obj = item,
            )

            if eq is not None:
                return eq

    return None









def build_variant_extension(
        variant,
        eq,
        eq_initial,
    ):
    """
    Build FLO/FNO objective and constraint objects for the current equilibrium.
    """

    variant_config = get_variant_config(
        variant = variant,
    )

    iota_axis = get_iota_axis_from_equilibrium(
        eq = eq_initial,
    )

    lower_rational, upper_rational = iota_between_rationals(
        iota_axis = iota_axis,
    )

    objective_configs = inject_initial_equilibrium(
        configs = tuple(variant_config["objectives"]),
        eq_initial = eq_initial,
    )

    constraint_configs = inject_initial_equilibrium(
        configs = tuple(variant_config["constraints"]),
        eq_initial = eq_initial,
    )

    objective_configs = resolve_symbolic_bounds(
        configs = objective_configs,
        lower_rational = lower_rational,
        upper_rational = upper_rational,
    )

    constraint_configs = resolve_symbolic_bounds(
        configs = constraint_configs,
        lower_rational = lower_rational,
        upper_rational = upper_rational,
    )

    objectives = tuple(
        build_objective(
            config = config,
            eq = eq,
        )
        for config in objective_configs
    )

    constraints = tuple(
        build_constraint(
            config = config,
            eq = eq,
        )
        for config in constraint_configs
    )

    return objectives, constraints









def append_variant_objectives(
        objective,
        variant_objectives,
    ):
    """
    Append FLO/FNO objectives to an existing ObjectiveFunction.
    """

    if len(variant_objectives) == 0:
        return objective

    existing_objectives = tuple(objective.objectives)

    return ObjectiveFunction(
        objectives = existing_objectives + tuple(variant_objectives),
    )










class VariantOptimizePatch:
    """
    Patch Optimizer.optimize for one FLO/FNO recreation-file run.
    """

    def __init__(
            self,
            variant,
        ):
        self.variant = variant
        self.original_optimize = None
        self.eq_initial = None

    def __enter__(self):
        self.original_optimize = Optimizer.optimize

        def patched_optimize(
                optimizer_self,
                things,
                objective,
                constraints = (),
                *args,
                **kwargs,
            ):
            eq = find_equilibrium_in_object(
                obj = things,
            )

            if eq is None:
                raise ValueError(
                    "Could not find an Equilibrium object in optimizer.optimize things."
                )

            if self.eq_initial is None:
                self.eq_initial = eq.copy()

            variant_objectives, variant_constraints = build_variant_extension(
                variant = self.variant,
                eq = eq,
                eq_initial = self.eq_initial,
            )

            objective_patched = append_variant_objectives(
                objective = objective,
                variant_objectives = variant_objectives,
            )

            constraints_patched = (
                remove_fixed_pressure_iota_constraints_from_objects(
                    constraints = constraints,
                )
                + variant_constraints
            )

            print("")
            print("================================================================================================================")
            print(f"Running with variant: {self.variant}")
            print(f"Added objectives: {len(variant_objectives)}")
            print(f"Added constraints: {len(variant_constraints)}")
            print(f"Total constraints after fixed pressure/iota removal: {len(constraints_patched)}")
            print("================================================================================================================")
            print("")

            return self.original_optimize(
                optimizer_self,
                things,
                objective_patched,
                constraints_patched,
                *args,
                **kwargs,
            )

        Optimizer.optimize = patched_optimize

        return self

    def __exit__(
            self,
            exc_type,
            exc_value,
            exc_traceback,
        ):
        Optimizer.optimize = self.original_optimize










#==============================================================================================================
# Recreation File Path Helpers
#==============================================================================================================

def find_input_dir(
        source_file,
    ):
    """
    Find the nearest parent directory named input.
    """

    source_file = Path(source_file).resolve()

    for parent in source_file.parents:
        if parent.name == "input":
            return parent

    raise ValueError(
        f"Could not find an input directory above {source_file}."
    )









def get_case_name_from_source(
        source_file,
    ):
    """
    Return the case name for a source file.

    Supports both:
        input/helical_qs.py
        input/helical_qs/helical_qs.py
    """

    source_file = Path(source_file)

    if source_file.parent.name != "input":
        return source_file.parent.name

    return source_file.stem









def get_output_dir_for_source(
        source_file,
    ):
    """
    Return output/<case> for a source file inside input/<case>.
    """

    source_file = Path(source_file).resolve()

    input_dir = find_input_dir(
        source_file = source_file,
    )

    paper_dir = input_dir.parent

    case_name = get_case_name_from_source(
        source_file = source_file,
    )

    return paper_dir / "output" / case_name









def get_final_output_path_for_source(
        source_file,
        variant,
    ):
    """
    Return output/<case>/<case>_<variant>.h5.
    """

    case_name = get_case_name_from_source(
        source_file = source_file,
    )

    output_dir = get_output_dir_for_source(
        source_file = source_file,
    )

    return output_dir / f"{case_name}_{variant}.h5"









def get_failure_output_path_for_source(
        source_file,
        variant,
    ):
    """
    Return output/<case>/<case>_<variant>_FAILURE.h5.
    """

    case_name = get_case_name_from_source(
        source_file = source_file,
    )

    output_dir = get_output_dir_for_source(
        source_file = source_file,
    )

    return output_dir / f"{case_name}_{variant}_FAILURE.h5"










#==============================================================================================================
# Recreation Source Patching
#==============================================================================================================

def patch_recreation_source_text(
        source_text,
    ):
    """
    Patch recreation source text so output paths can be overridden by this driver.
    """

    patched = source_text

    if "import os" not in patched:
        patched = patched.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\nimport os\n",
            1,
        )

    if "VFO_OUTPUT_DIR" in patched:
        return patched

    lines = patched.splitlines()
    new_lines = []

    for line in lines:
        new_lines.append(line)

        if line.lstrip().startswith("OUTPUT_DIR ="):
            new_lines.append(
                """
if os.environ.get("VFO_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["VFO_OUTPUT_DIR"])
""".strip()
            )

        if line.lstrip().startswith("OUTPUT_DIR.mkdir"):
            new_lines.append(
                """
if os.environ.get("VFO_OUTPUT_DIR"):
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
""".strip()
            )

        if line.lstrip().startswith("CHECK_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FINAL_PATH"):
    CHECK_PATH = Path(os.environ["VFO_FINAL_PATH"])
""".strip()
            )

        if line.lstrip().startswith("FAILURE_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FAILURE_PATH"):
    FAILURE_PATH = Path(os.environ["VFO_FAILURE_PATH"])
""".strip()
            )

    return "\n".join(new_lines) + "\n"









def make_temporary_recreation_runner(
        source_file,
        variant,
    ):
    """
    Create a patched temporary copy of the target recreation source file.

    The temporary runner is written beside the original file so __file__.parent
    still points to the original input folder.
    """

    source_file = Path(source_file).resolve()

    source_text = source_file.read_text()

    patched_text = patch_recreation_source_text(
        source_text = source_text,
    )

    runner_path = source_file.parent / f"_{source_file.stem}_{variant}_runner.py"

    runner_path.write_text(
        patched_text,
    )

    return runner_path










#==============================================================================================================
# Recreation Variant Runners
#==============================================================================================================

def run_source_with_variant(
        source_file,
        variant,
    ):
    """
    Run one recreation source file with one FLO/FNO variant.
    """

    source_file = Path(source_file).resolve()

    output_dir = get_output_dir_for_source(
        source_file = source_file,
    )

    output_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    final_path = get_final_output_path_for_source(
        source_file = source_file,
        variant = variant,
    )

    failure_path = get_failure_output_path_for_source(
        source_file = source_file,
        variant = variant,
    )

    runner_path = make_temporary_recreation_runner(
        source_file = source_file,
        variant = variant,
    )

    old_env = dict(os.environ)

    os.environ["VFO_VARIANT"] = variant
    os.environ["VFO_OUTPUT_DIR"] = str(output_dir)
    os.environ["VFO_FINAL_PATH"] = str(final_path)
    os.environ["VFO_FAILURE_PATH"] = str(failure_path)

    try:
        with VariantOptimizePatch(
                variant = variant,
            ):
            runpy.run_path(
                path_name = str(runner_path),
                run_name = "__main__",
            )

    finally:
        os.environ.clear()
        os.environ.update(old_env)

        if runner_path.exists():
            runner_path.unlink()

    if not final_path.exists():
        raise FileNotFoundError(
            f"Expected final variant output was not created: {final_path}"
        )

    return final_path









def run_variants_for_file(
        source_file,
        variants = ("FLO", "FNO"),
    ):
    """
    Run all requested variants for one recreation source file.
    """

    source_file = Path(source_file).resolve()
    outputs = {}

    for variant in variants:
        print("")
        print("################################################################################################################")
        print(f"Starting {variant} run for {source_file}")
        print("################################################################################################################")
        print("")

        try:
            outputs[variant] = run_source_with_variant(
                source_file = source_file,
                variant = variant,
            )

            print("")
            print(f"{variant} saved to: {outputs[variant]}")
            print("")

        except Exception:
            print("")
            print(f"{variant} failed for {source_file}")
            traceback.print_exc()
            raise

    return outputs

