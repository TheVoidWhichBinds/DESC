import importlib.util
from pathlib import Path

from desc.objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
    LCFSBoundary,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)










#============== CONSTRAINT REGISTRY ==============================================================================
CONSTRAINT_CLASS_REGISTRY = {
    "FixBoundaryR": FixBoundaryR,
    "FixBoundaryZ": FixBoundaryZ,
    "FixPressure": FixPressure,
    "FixIota": FixIota,
    "FixPsi": FixPsi,

    "FixedBoundaryR": FixBoundaryR,
    "FixedBoundaryZ": FixBoundaryZ,
    "FixedPressure": FixPressure,
    "FixedIota": FixIota,
    "FixedPsi": FixPsi,

    "LCFSBoundary": LCFSBoundary,

    "LinearObjectiveFromUser": LinearObjectiveFromUser,
    "ObjectiveFromUser": ObjectiveFromUser,
}

USER_CONSTRAINT_CLASSES = {
    "LinearObjectiveFromUser",
    "ObjectiveFromUser",
}
#==============================================================================================================










#========== build_constraints =====================================================================================
def build_constraints(
        eq,
        run_config,
    ):
    constraints = []

    constraint_removals = set(run_config.get("variant_constraint_removals", []))

    for constraint_entry in run_config["paper_constraints"]:
        if should_remove_constraint(
            constraint_entry = constraint_entry,
            constraint_removals = constraint_removals,
        ):
            continue

        constraints.append(
            build_single_constraint(
                eq = eq,
                run_config = run_config,
                constraint_entry = constraint_entry,
            )
        )

    for constraint_entry in run_config.get("variant_constraints", []):
        constraints.append(
            build_single_constraint(
                eq = eq,
                run_config = run_config,
                constraint_entry = constraint_entry,
            )
        )

    return tuple(constraints)
#==============================================================================================================










#========== should_remove_constraint =============================================================================
def should_remove_constraint(
        constraint_entry,
        constraint_removals,
    ):
    class_name = constraint_entry.get("class", None)
    name = constraint_entry.get("name", None)

    return class_name in constraint_removals or name in constraint_removals
#==============================================================================================================










#========== build_single_constraint ==============================================================================
def build_single_constraint(
        eq,
        run_config,
        constraint_entry,
    ):
    class_name = constraint_entry["class"]

    kwargs = resolve_constraint_kwargs(
        kwargs = constraint_entry.get("kwargs", {}),
        run_config = run_config,
    )

    if class_name not in CONSTRAINT_CLASS_REGISTRY:
        raise ValueError(
            f"Constraint class '{class_name}' is not registered. "
            f"Add it to CONSTRAINT_CLASS_REGISTRY in src/build_constraints.py."
        )

    constraint_class = CONSTRAINT_CLASS_REGISTRY[class_name]

    if class_name in USER_CONSTRAINT_CLASSES:
        return constraint_class(
            thing = eq,
            **kwargs,
        )

    return constraint_class(
        eq = eq,
        **kwargs,
    )
#==============================================================================================================










#========== resolve_constraint_kwargs ============================================================================
def resolve_constraint_kwargs(
        kwargs,
        run_config,
    ):
    resolved = {}

    for key, value in kwargs.items():
        resolved[key] = value

    return resolved
#==============================================================================================================










#========== get_paper_opt_subspace ===============================================================================
def get_paper_opt_subspace(
        eq,
        run_config,
    ):
    subspace_config = run_config["boundary_free_subspace"]

    source_file = Path(subspace_config["source_file"])
    function_name = subspace_config["function_name"]

    if not source_file.exists():
        raise FileNotFoundError(f"Could not find paper driver file: {source_file}")

    spec = importlib.util.spec_from_file_location(
        "paper_driver",
        source_file,
    )

    paper_driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper_driver)

    if not hasattr(paper_driver, function_name):
        raise AttributeError(
            f"Could not find function '{function_name}' in {source_file}"
        )

    get_subspace = getattr(paper_driver, function_name)

    return get_subspace(eq)
#==============================================================================================================