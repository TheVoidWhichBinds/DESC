# research/final/src/builders.py











#============== IMPORTS ========================================================================================
from copy import deepcopy











#============== DESC IMPORTS ===================================================================================
def _import_desc_objects():
    from desc.objectives import ObjectiveFunction

    objective_registry = {}

    constraint_registry = {}

    try:
        from desc.objectives import ForceBalance
        objective_registry["ForceBalance"] = ForceBalance
    except Exception:
        pass

    try:
        from desc.objectives import QuasisymmetryBoozer
        objective_registry["QuasisymmetryBoozer"] = QuasisymmetryBoozer
    except Exception:
        pass

    try:
        from desc.objectives import QuasisymmetryTwoTerm
        objective_registry["QuasisymmetryTwoTerm"] = QuasisymmetryTwoTerm
    except Exception:
        pass

    try:
        from desc.objectives import QuasisymmetryTripleProduct
        objective_registry["QuasisymmetryTripleProduct"] = QuasisymmetryTripleProduct
    except Exception:
        pass

    try:
        from desc.objectives import MercierStability
        objective_registry["MercierStability"] = MercierStability
    except Exception:
        pass

    try:
        from desc.objectives import MagneticWell
        objective_registry["MagneticWell"] = MagneticWell
    except Exception:
        pass

    try:
        from desc.objectives import GoodCoordinates
        objective_registry["GoodCoordinates"] = GoodCoordinates
    except Exception:
        pass

    try:
        from desc.objectives import MeanCurvature
        objective_registry["MeanCurvature"] = MeanCurvature
    except Exception:
        pass

    try:
        from desc.objectives import PrincipalCurvature
        objective_registry["PrincipalCurvature"] = PrincipalCurvature
    except Exception:
        pass

    try:
        from desc.objectives import AspectRatio
        objective_registry["AspectRatio"] = AspectRatio
    except Exception:
        pass

    try:
        from desc.objectives import Volume
        objective_registry["Volume"] = Volume
    except Exception:
        pass

    try:
        from desc.objectives import Elongation
        objective_registry["Elongation"] = Elongation
    except Exception:
        pass

    try:
        from desc.objectives import FixBoundaryR
        constraint_registry["FixBoundaryR"] = FixBoundaryR
    except Exception:
        pass

    try:
        from desc.objectives import FixBoundaryZ
        constraint_registry["FixBoundaryZ"] = FixBoundaryZ
    except Exception:
        pass

    try:
        from desc.objectives import FixPressure
        constraint_registry["FixPressure"] = FixPressure
    except Exception:
        pass

    try:
        from desc.objectives import FixIota
        constraint_registry["FixIota"] = FixIota
    except Exception:
        pass

    try:
        from desc.objectives import FixCurrent
        constraint_registry["FixCurrent"] = FixCurrent
    except Exception:
        pass

    try:
        from desc.objectives import FixAxisR
        constraint_registry["FixAxisR"] = FixAxisR
    except Exception:
        pass

    try:
        from desc.objectives import FixAxisZ
        constraint_registry["FixAxisZ"] = FixAxisZ
    except Exception:
        pass

    return ObjectiveFunction, objective_registry, constraint_registry











#============== HELPERS ========================================================================================
def _resolve_special_values(value, eq):
    if value == "NFP":
        return eq.NFP

    if isinstance(value, list):
        return tuple(_resolve_special_values(entry, eq) for entry in value)

    if isinstance(value, tuple):
        return tuple(_resolve_special_values(entry, eq) for entry in value)

    if isinstance(value, dict):
        return {
            key: _resolve_special_values(item, eq)
            for key, item in value.items()
        }

    return value





def _clean_kwargs(config, eq):
    kwargs = deepcopy(config.get("kwargs", {}))
    kwargs = _resolve_special_values(kwargs, eq)

    if "weight" in config:
        kwargs["weight"] = config["weight"]

    return kwargs











#============== OBJECTIVE BUILDERS =============================================================================
def build_objective_function(eq, objective_configs):
    ObjectiveFunction, objective_registry, _ = _import_desc_objects()

    objectives = []
    skipped = []

    for objective_config in objective_configs:
        name = objective_config["name"]
        optional = objective_config.get("optional", False)

        if name not in objective_registry:
            message = f"Objective {name!r} is not available in this DESC version."

            if optional:
                skipped.append(message)
                continue

            raise ImportError(message)

        objective_cls = objective_registry[name]
        kwargs = _clean_kwargs(objective_config, eq)

        objectives.append(objective_cls(eq = eq, **kwargs))

    if not objectives:
        raise ValueError("No objectives were built. Check objective configs and optional flags.")

    return ObjectiveFunction(objectives), skipped





def build_constraint_function(eq, constraint_configs):
    ObjectiveFunction, _, constraint_registry = _import_desc_objects()

    constraints = []
    skipped = []

    for constraint_config in constraint_configs:
        name = constraint_config["name"]
        optional = constraint_config.get("optional", False)

        if name not in constraint_registry:
            message = f"Constraint {name!r} is not available in this DESC version."

            if optional:
                skipped.append(message)
                continue

            raise ImportError(message)

        constraint_cls = constraint_registry[name]
        kwargs = _clean_kwargs(constraint_config, eq)

        constraints.append(constraint_cls(eq = eq, **kwargs))

    if not constraints:
        return None, skipped

    return ObjectiveFunction(constraints), skipped