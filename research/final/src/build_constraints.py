#==============================================================================================================
# IMPORTS
#==============================================================================================================

import importlib.util
from pathlib import Path

import desc.objectives as desc_objectives





#==============================================================================================================
# CONSTRAINT CLASS ALIASES
#==============================================================================================================

CONSTRAINT_CLASS_ALIASES = {
    "FixBoundaryR": [
        "FixBoundaryR",
    ],

    "FixBoundaryZ": [
        "FixBoundaryZ",
    ],

    "FixPressure": [
        "FixPressure",
    ],

    "FixIota": [
        "FixIota",
    ],

    "FixPsi": [
        "FixPsi",
    ],

    "FixedBoundaryR": [
        "FixBoundaryR",
    ],

    "FixedBoundaryZ": [
        "FixBoundaryZ",
    ],

    "FixedPressure": [
        "FixPressure",
    ],

    "FixedIota": [
        "FixIota",
    ],

    "FixedPsi": [
        "FixPsi",
    ],

    "FixCurrent": [
        "FixCurrent",
    ],

    "FixedCurrent": [
        "FixCurrent",
    ],

    "LinearObjectiveFromUser": [
        "LinearObjectiveFromUser",
    ],

    "ObjectiveFromUser": [
        "ObjectiveFromUser",
    ],
}


USER_CONSTRAINT_CLASSES = {
    "LinearObjectiveFromUser",
    "ObjectiveFromUser",
}


DEPRECATED_OR_UNAVAILABLE_CONSTRAINTS = {
    "LCFSBoundary",
}





#==============================================================================================================
# CONSTRAINT CLASS LOOKUP
#==============================================================================================================

def get_desc_constraint_class(
        class_name,
    ):
    possible_names = CONSTRAINT_CLASS_ALIASES.get(
        class_name,
        [class_name],
    )

    for possible_name in possible_names:
        if hasattr(desc_objectives, possible_name):
            return getattr(
                desc_objectives,
                possible_name,
            )

    raise ImportError(
        f"Could not find DESC constraint/objective class '{class_name}'. "
        f"Tried aliases: {possible_names}."
    )





#==============================================================================================================
# BUILD CONSTRAINTS
#==============================================================================================================

def build_constraints(
        eq,
        run_config,
    ):
    constraints = []

    constraint_removals = set(
        run_config.get("variant_constraint_removals", [])
    )

    for constraint_entry in run_config["paper_constraints"]:
        if should_remove_constraint(
            constraint_entry = constraint_entry,
            constraint_removals = constraint_removals,
        ):
            continue

        if should_skip_unavailable_constraint(
            constraint_entry = constraint_entry,
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
        if should_skip_unavailable_constraint(
            constraint_entry = constraint_entry,
        ):
            continue

        constraints.append(
            build_single_constraint(
                eq = eq,
                run_config = run_config,
                constraint_entry = constraint_entry,
            )
        )

    return tuple(constraints)





#==============================================================================================================
# SHOULD REMOVE / SKIP CONSTRAINT
#==============================================================================================================

def should_remove_constraint(
        constraint_entry,
        constraint_removals,
    ):
    class_name = constraint_entry.get("class", None)
    name = constraint_entry.get("name", None)

    return class_name in constraint_removals or name in constraint_removals





def should_skip_unavailable_constraint(
        constraint_entry,
    ):
    class_name = constraint_entry.get("class", None)

    if class_name in DEPRECATED_OR_UNAVAILABLE_CONSTRAINTS:
        print(
            f"Skipping unavailable/deprecated constraint: {class_name}",
            flush = True,
        )
        return True

    return False





#==============================================================================================================
# BUILD SINGLE CONSTRAINT
#==============================================================================================================

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

    constraint_class = get_desc_constraint_class(
        class_name = class_name,
    )

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
# RESOLVE CONSTRAINT KWARGS
#==============================================================================================================

def resolve_constraint_kwargs(
        kwargs,
        run_config,
    ):
    resolved = {}

    for key, value in kwargs.items():
        resolved[key] = value

    return resolved





#==============================================================================================================
# PAPER OPTIMIZATION SUBSPACE
#==============================================================================================================

def get_paper_opt_subspace(
        eq,
        run_config,
    ):
    subspace_config = run_config["boundary_free_subspace"]
    source_file = Path(subspace_config["source_file"])
    function_name = subspace_config["function_name"]

    if not source_file.exists():
        raise FileNotFoundError(
            f"Could not find paper driver file: {source_file}"
        )

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

    get_subspace = getattr(
        paper_driver,
        function_name,
    )

    return get_subspace(eq)

#==============================================================================================================