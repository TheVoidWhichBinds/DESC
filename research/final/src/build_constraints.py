#==============================================================================================================
# IMPORTS
#==============================================================================================================

import ast
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

    "FixAxisR": [
        "FixAxisR",
    ],

    "FixAxisZ": [
        "FixAxisZ",
    ],

    "FixModeR": [
        "FixModeR",
    ],

    "FixModeZ": [
        "FixModeZ",
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

    "FixCurrent": [
        "FixCurrent",
    ],

    "FixedBoundaryR": [
        "FixBoundaryR",
    ],

    "FixedBoundaryZ": [
        "FixBoundaryZ",
    ],

    "FixedAxisR": [
        "FixAxisR",
    ],

    "FixedAxisZ": [
        "FixAxisZ",
    ],

    "FixedModeR": [
        "FixModeR",
    ],

    "FixedModeZ": [
        "FixModeZ",
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
    "QuasisymmetryFluxFunction",
}


DESC_PUBLICATION_COMPATIBILITY_ALIASES = {
    "FixedBoundaryR": "FixBoundaryR",
    "FixedBoundaryZ": "FixBoundaryZ",
    "FixedAxisR": "FixAxisR",
    "FixedAxisZ": "FixAxisZ",
    "FixedModeR": "FixModeR",
    "FixedModeZ": "FixModeZ",
    "FixedPressure": "FixPressure",
    "FixedIota": "FixIota",
    "FixedPsi": "FixPsi",
    "FixedCurrent": "FixCurrent",
}


UNAVAILABLE_PUBLICATION_IMPORTS = {
    "LCFSBoundary",
    "QuasisymmetryFluxFunction",
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
# DESC PUBLICATION DRIVER COMPATIBILITY
#==============================================================================================================

class UnavailablePublicationConstraint:
    """
    Placeholder for old publication-driver classes that no longer exist in the
    current DESC API.

    This lets old publication files be parsed for helper functions without
    silently using an incorrect replacement objective or constraint.
    """

    def __init__(
            self,
            *args,
            **kwargs,
        ):
        class_name = self.__class__.__name__

        raise ImportError(
            f"This old publication driver tried to instantiate {class_name}, "
            f"but {class_name} is not available in the current DESC API. "
            "Define the modern equivalent directly in research/final configs instead."
        )





def install_desc_publication_compatibility_aliases():
    """
    Adds compatibility names to desc.objectives before parsing old publication
    drivers.
    """

    for old_name, new_name in DESC_PUBLICATION_COMPATIBILITY_ALIASES.items():
        if hasattr(desc_objectives, old_name):
            continue

        if not hasattr(desc_objectives, new_name):
            continue

        setattr(
            desc_objectives,
            old_name,
            getattr(
                desc_objectives,
                new_name,
            ),
        )

    for unavailable_name in UNAVAILABLE_PUBLICATION_IMPORTS:
        if hasattr(desc_objectives, unavailable_name):
            continue

        unavailable_class = type(
            unavailable_name,
            (UnavailablePublicationConstraint,),
            {},
        )

        setattr(
            desc_objectives,
            unavailable_name,
            unavailable_class,
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

def load_publication_function_without_running_script(
        source_file,
        function_name,
    ):
    """
    Loads one function from an old publication driver without executing the
    driver's top-level script body.

    This avoids running old lines like:
        Equilibrium.load("data/initial.h5")
        eq.surface.R_basis._create_idx()

    Only imports, functions, classes, and safe constant assignments are kept.
    """

    source_text = source_file.read_text()
    source_tree = ast.parse(
        source_text,
        filename = str(source_file),
    )

    kept_nodes = []

    for node in source_tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
            ),
        ):
            kept_nodes.append(node)
            continue

        if isinstance(node, ast.Assign) and is_safe_constant_assignment(node):
            kept_nodes.append(node)
            continue

    filtered_tree = ast.Module(
        body = kept_nodes,
        type_ignores = [],
    )

    ast.fix_missing_locations(filtered_tree)

    namespace = {
        "__file__": str(source_file),
        "__name__": f"{source_file.stem}_function_loader",
    }

    exec(
        compile(
            filtered_tree,
            filename = str(source_file),
            mode = "exec",
        ),
        namespace,
    )

    if function_name not in namespace:
        raise AttributeError(
            f"Could not find function '{function_name}' in {source_file}"
        )

    return namespace[function_name]





def is_safe_constant_assignment(
        node,
    ):
    """
    Keeps harmless top-level constants from old publication drivers while
    excluding script execution assignments such as:
        eq = Equilibrium.load(...)
        result = eq.optimize(...)
    """

    return isinstance(
        node.value,
        (
            ast.Constant,
            ast.Tuple,
            ast.List,
            ast.Dict,
            ast.Set,
        ),
    )





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

    install_desc_publication_compatibility_aliases()

    get_subspace = load_publication_function_without_running_script(
        source_file = source_file,
        function_name = function_name,
    )

    return get_subspace(eq)

#==============================================================================================================