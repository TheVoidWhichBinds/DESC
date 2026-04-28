from desc.objectives import (
    ObjectiveFunction,
    QuasisymmetryBoozer,
    QuasisymmetryFluxFunction,
    QuasisymmetryTripleProduct,
)










#============== OBJECTIVE REGISTRY ===============================================================================
OBJECTIVE_CLASS_REGISTRY = {
    "QuasisymmetryBoozer": QuasisymmetryBoozer,
    "QuasisymmetryFluxFunction": QuasisymmetryFluxFunction,
    "QuasisymmetryTripleProduct": QuasisymmetryTripleProduct,
}
#==============================================================================================================










#========== build_objective =======================================================================================
def build_objective(
        eq,
        run_config,
    ):
    objective_config = run_config["paper_objectives"]

    objective_family = objective_config["objective_family"]

    if objective_family == "qs":
        objective = build_qs_objective(
            eq = eq,
            run_config = run_config,
        )

    elif objective_family in ["FLO", "FNO"]:
        objective = build_variant_objective(
            eq = eq,
            objective_config = objective_config,
        )

    else:
        raise ValueError(f"Unknown objective_family '{objective_family}'")

    return ObjectiveFunction(objective)
#==============================================================================================================










#========== build_qs_objective ====================================================================================
def build_qs_objective(
        eq,
        run_config,
    ):
    objective_config = run_config["paper_objectives"]

    qs = objective_config["qs"]
    qs_entry = objective_config["available_qs_objectives"][qs]

    class_name = qs_entry["class"]
    kwargs = resolve_objective_kwargs(
        kwargs = qs_entry.get("kwargs", {}),
        eq = eq,
    )

    objective_class = OBJECTIVE_CLASS_REGISTRY[class_name]

    return objective_class(
        eq = eq,
        **kwargs,
    )
#==============================================================================================================










#========== build_variant_objective ==============================================================================
def build_variant_objective(
        eq,
        objective_config,
    ):
    objectives = []

    for objective_entry in objective_config.get("objectives", []):
        class_name = objective_entry["class"]
        kwargs = objective_entry.get("kwargs", {})

        if class_name not in OBJECTIVE_CLASS_REGISTRY:
            raise ValueError(
                f"Objective class '{class_name}' is not registered. "
                f"Add it to OBJECTIVE_CLASS_REGISTRY in src/build_objectives.py."
            )

        objective_class = OBJECTIVE_CLASS_REGISTRY[class_name]

        objectives.append(
            objective_class(
                eq = eq,
                **kwargs,
            )
        )

    return tuple(objectives)
#==============================================================================================================










#========== resolve_objective_kwargs =============================================================================
def resolve_objective_kwargs(
        kwargs,
        eq,
    ):
    resolved = {}

    for key, value in kwargs.items():
        if value == ("one", "NFP"):
            resolved[key] = (1, eq.NFP)

        else:
            resolved[key] = value

    return resolved
#==============================================================================================================