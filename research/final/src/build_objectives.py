#==============================================================================================================
# IMPORTS
#==============================================================================================================

import desc.objectives as desc_objectives
import inspect




#==============================================================================================================
# OBJECTIVE CLASS ALIASES
#==============================================================================================================

OBJECTIVE_CLASS_ALIASES = {
    "ObjectiveFunction": [
        "ObjectiveFunction",
    ],

    "QuasisymmetryBoozer": [
        "QuasisymmetryBoozer",
    ],

    "QuasisymmetryFluxFunction": [
        "QuasisymmetryFluxFunction",
        "QuasisymmetryTwoTerm",
    ],

    "QuasisymmetryTwoTerm": [
        "QuasisymmetryTwoTerm",
        "QuasisymmetryFluxFunction",
    ],

    "QuasisymmetryTripleProduct": [
        "QuasisymmetryTripleProduct",
    ],

    "LinearObjectiveFromUser": [
        "LinearObjectiveFromUser",
    ],

    "ObjectiveFromUser": [
        "ObjectiveFromUser",
    ],
}


USER_OBJECTIVE_CLASSES = {
    "LinearObjectiveFromUser",
    "ObjectiveFromUser",
}





#==============================================================================================================
# OBJECTIVE CLASS LOOKUP
#==============================================================================================================

def get_desc_objective_class(
        class_name,
    ):
    possible_names = OBJECTIVE_CLASS_ALIASES.get(
        class_name,
        [class_name],
    )

    for possible_name in possible_names:
        if hasattr(desc_objectives, possible_name):
            return getattr(
                desc_objectives,
                possible_name,
            )

    available = [
        name
        for name in possible_names
        if hasattr(desc_objectives, name)
    ]

    raise ImportError(
        f"Could not find DESC objective class '{class_name}'. "
        f"Tried aliases: {possible_names}. "
        f"Available matches: {available}."
    )






#==============================================================================================================
# SANITIZE OBJECTIVE KWARGS
#==============================================================================================================

def sanitize_objective_kwargs(
        objective_class,
        kwargs,
    ):
    """
    Remove or rename kwargs that are not accepted by the installed DESC class.

    This makes older paper configs more robust to DESC API changes.
    """

    signature = inspect.signature(objective_class.__init__)
    accepted_kwargs = set(signature.parameters.keys())

    aliases = {
        "norm": "normalize",
    }

    sanitized = {}

    for key, value in kwargs.items():
        mapped_key = aliases.get(key, key)

        if mapped_key in accepted_kwargs:
            sanitized[mapped_key] = value

        else:
            print(
                f"Skipping unsupported kwarg for {objective_class.__name__}: {key}",
                flush = True,
            )

    return sanitized






#==============================================================================================================
# BUILD OBJECTIVE
#==============================================================================================================

def build_objective(
        eq,
        run_config,
    ):
    objective_terms = []

    paper_objective = build_paper_objective(
        eq = eq,
        run_config = run_config,
    )

    if isinstance(paper_objective, tuple):
        objective_terms.extend(paper_objective)

    else:
        objective_terms.append(paper_objective)

    variant_objectives = build_variant_objectives(
        eq = eq,
        run_config = run_config,
    )

    objective_terms.extend(variant_objectives)

    objective_function_class = get_desc_objective_class(
        class_name = "ObjectiveFunction",
    )

    return objective_function_class(objective_terms)





#==============================================================================================================
# BUILD PAPER OBJECTIVE
#==============================================================================================================

def build_paper_objective(
        eq,
        run_config,
    ):
    objective_config = run_config["paper_objectives"]
    objective_family = objective_config["objective_family"]

    if objective_family == "qs":
        return build_qs_objective(
            eq = eq,
            run_config = run_config,
        )

    raise ValueError(
        f"Unknown paper objective_family '{objective_family}'"
    )





#==============================================================================================================
# BUILD QS OBJECTIVE
#==============================================================================================================

def build_qs_objective(
        eq,
        run_config,
    ):
    objective_config = run_config["paper_objectives"]

    qs = objective_config["qs"]
    qs_entry = objective_config["available_qs_objectives"][qs]

    class_name = qs_entry["class"]

    kwargs = resolve_kwargs(
        kwargs = qs_entry.get("kwargs", {}),
        eq = eq,
        run_config = run_config,
    )

    objective_class = get_desc_objective_class(
        class_name = class_name,
    )

    kwargs = sanitize_objective_kwargs(
        objective_class = objective_class,
        kwargs = kwargs,
    )

    return objective_class(
        eq = eq,
        **kwargs,
    )





#==============================================================================================================
# BUILD VARIANT OBJECTIVES
#==============================================================================================================

def build_variant_objectives(
        eq,
        run_config,
    ):
    objectives = []

    for objective_entry in run_config.get("variant_objectives", []):
        objectives.append(
            build_single_objective(
                eq = eq,
                run_config = run_config,
                objective_entry = objective_entry,
            )
        )

    return objectives





#==============================================================================================================
# BUILD SINGLE OBJECTIVE
#==============================================================================================================

def build_single_objective(
        eq,
        run_config,
        objective_entry,
    ):
    class_name = objective_entry["class"]

    kwargs = resolve_kwargs(
        kwargs = objective_entry.get("kwargs", {}),
        eq = eq,
        run_config = run_config,
    )

    objective_class = get_desc_objective_class(
        class_name = class_name,
    )

    kwargs = sanitize_objective_kwargs(
        objective_class = objective_class,
        kwargs = kwargs,
    )

    if class_name in USER_OBJECTIVE_CLASSES:
        return objective_class(
            thing = eq,
            **kwargs,
        )

    return objective_class(
        eq = eq,
        **kwargs,
    )





#==============================================================================================================
# RESOLVE KWARGS
#==============================================================================================================

def resolve_kwargs(
        kwargs,
        eq,
        run_config,
    ):
    resolved = {}

    for key, value in kwargs.items():
        if value == ("one", "NFP"):
            resolved[key] = (1, eq.NFP)

        elif value == ("iota_between_rationals", "axis"):
            resolved[key] = get_iota_bounds_from_run_config(
                run_config = run_config,
            )

        else:
            resolved[key] = value

    return resolved





#==============================================================================================================
# GET IOTA BOUNDS FROM RUN CONFIG
#==============================================================================================================

def get_iota_bounds_from_run_config(
        run_config,
    ):
    iota_params = (
        run_config
        .get("profile_config", {})
        .get("reference_iota", {})
        .get("params", None)
    )

    if iota_params is None or len(iota_params) == 0:
        raise ValueError(
            "Could not compute iota bounds because "
            "run_config['profile_config']['reference_iota']['params'] is missing."
        )

    iota_axis = float(iota_params[0])

    rational_values = [
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
    ]

    lower = None
    upper = None

    for rational_value in rational_values:
        if rational_value < iota_axis:
            lower = rational_value

        elif rational_value > iota_axis and upper is None:
            upper = rational_value
            break

    if lower is None or upper is None:
        raise ValueError(
            f"Could not bracket iota_axis = {iota_axis} between known rational values."
        )

    return (lower, upper)

#==============================================================================================================