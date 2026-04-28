import importlib.util
from pathlib import Path

from desc.objectives import (
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    FixedIota,
    FixedPsi,
    LCFSBoundary,
)










#============== CONSTRAINT REGISTRY ==============================================================================
CONSTRAINT_CLASS_REGISTRY = {
    "FixedBoundaryR": FixedBoundaryR,
    "FixedBoundaryZ": FixedBoundaryZ,
    "FixedPressure": FixedPressure,
    "FixedIota": FixedIota,
    "FixedPsi": FixedPsi,
    "LCFSBoundary": LCFSBoundary,
}
#==============================================================================================================










#========== build_constraints =====================================================================================
def build_constraints(
        eq,
        run_config,
    ):
    constraints = []

    for constraint_entry in run_config["paper_constraints"]:
        class_name = constraint_entry["class"]
        kwargs = constraint_entry.get("kwargs", {})

        if class_name not in CONSTRAINT_CLASS_REGISTRY:
            raise ValueError(
                f"Constraint class '{class_name}' is not registered. "
                f"Add it to CONSTRAINT_CLASS_REGISTRY in src/build_constraints.py."
            )

        constraint_class = CONSTRAINT_CLASS_REGISTRY[class_name]

        constraints.append(
            constraint_class(
                eq = eq,
                **kwargs,
            )
        )

    return tuple(constraints)
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