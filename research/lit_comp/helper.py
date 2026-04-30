# helper.py
#==============================================================================================================
#
# Shared helper functions for DESC/research/final.
#
# This file owns:
#   1. Variant registry access
#   2. FixPressure removal for FLO/FNO runs
#   3. Initial-equilibrium injection through kwargs["thing"]
#   4. DESC custom objective/constraint construction
#   5. Optimizer patching for recreation-file variant runs
#   6. Recreation-file output path patching
#
#==============================================================================================================

from copy import deepcopy
from pathlib import Path
import inspect
import os
import runpy
import traceback


from desc.optimize import Optimizer
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    ObjectiveFromUser,
    LinearObjectiveFromUser,
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
        f"Unknown variant '{variant}'. Valid variants are: 'FLO', 'FNO'."
    )













#==============================================================================================================
# DESC Custom Objective / Constraint Builders
#==============================================================================================================

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










#==============================================================================================================
# Constraint Removal Helpers
#==============================================================================================================

FIXED_PRESSURE_CLASS_NAMES = {
    "FixPressure",
    "PressureFixed",
    # "IotaFixed",
    # "FixIota",
    # "FixIotaProfile",
}









def is_fixed_pressure_constraint(
        constraint,
    ):
    """
    Return True when a constraint fixes the pressure profile.
    """

    class_name = constraint.__class__.__name__

    if class_name in FIXED_PRESSURE_CLASS_NAMES:
        return True

    name = getattr(
        constraint,
        "name",
        "",
    )

    name = str(name)

    return any(
        fixed_name in name
        for fixed_name in FIXED_PRESSURE_CLASS_NAMES
    )









def remove_fixed_pressure_constraints_from_objects(
        constraints,
    ):
    """
    Remove pressure-profile-fixing constraints from instantiated constraint objects.
    """

    if constraints is None:
        return ()

    kept_constraints = []

    for constraint in tuple(constraints):
        if is_fixed_pressure_constraint(
                constraint = constraint,
            ):
            print(f"Removing fixed pressure constraint: {constraint.__class__.__name__}")
            continue

        kept_constraints.append(constraint)

    return tuple(kept_constraints)










#==============================================================================================================
# Equilibrium Helpers
#==============================================================================================================

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










#==============================================================================================================
# Variant Extension Builder
#==============================================================================================================

def build_variant_extension(
        variant,
        eq,
        eq_initial = None,
    ):
    """
    Build FLO/FNO objective and constraint objects for the current equilibrium.
    """

    variant_config = get_variant_config(
        variant = variant,
    )

    objective_configs = tuple(variant_config["objectives"])
    constraint_configs = tuple(variant_config["constraints"])

    objectives = tuple(
        build_user_objective(
            config = config,
            eq = eq,
        )
        for config in objective_configs
    )

    constraints = tuple(
        build_user_objective(
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










#==============================================================================================================
# Optimizer Patch
#==============================================================================================================

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
                remove_fixed_pressure_constraints_from_objects(
                    constraints = constraints,
                )
                + variant_constraints
            )

            print("")
            print("================================================================================================================")
            print(f"Running with variant: {self.variant}")
            print(f"Added objectives: {len(variant_objectives)}")
            print(f"Added constraints: {len(variant_constraints)}")
            print(f"Total constraints after fixed pressure removal: {len(constraints_patched)}")
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

def get_case_name_from_source(
        source_file,
    ):
    """
    Return the case name for a source file.

    Supports:
        recreations/dudt2024/input/helical_qs/helical_qs.py
        recreations/dudt2024/helical_qs/helical_qs.py
        recreations/dudt2024/input/helical_qs.py
    """

    source_file = Path(source_file).resolve()

    if source_file.parent.name == "input":
        return source_file.stem

    return source_file.parent.name









def get_paper_dir_for_source(
        source_file,
    ):
    """
    Return the paper directory for a recreation source file.

    Supports:
        recreations/dudt2024/input/helical_qs/helical_qs.py  -> recreations/dudt2024
        recreations/dudt2024/helical_qs/helical_qs.py        -> recreations/dudt2024
        recreations/dudt2024/input/helical_qs.py             -> recreations/dudt2024
    """

    source_file = Path(source_file).resolve()

    if source_file.parent.name == "input":
        return source_file.parent.parent

    if source_file.parent.parent.name == "input":
        return source_file.parent.parent.parent

    return source_file.parent.parent









def get_output_dir_for_source(
        source_file,
    ):
    """
    Return output/<case> for a recreation source file.
    """

    paper_dir = get_paper_dir_for_source(
        source_file = source_file,
    )

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
    Patch recreation source text so final/failure output paths can be overridden by this driver.

    Important:
        Do not override OUTPUT_DIR, because recreation files often use OUTPUT_DIR
        to locate local input files such as *_initial.h5.
    """

    patched = source_text

    if "import os" not in patched:
        patched = patched.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\nimport os\n",
            1,
        )

    if "VFO_FINAL_PATH" in patched:
        return patched

    lines = patched.splitlines()
    new_lines = []

    for line in lines:
        new_lines.append(line)

        if line.lstrip().startswith("FINAL_PATH ="):
            new_lines.append(
                """
if os.environ.get("VFO_FINAL_PATH"):
    FINAL_PATH = Path(os.environ["VFO_FINAL_PATH"])
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
