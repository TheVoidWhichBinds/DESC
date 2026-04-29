# research/final/src/run_case.py











#============== IMPORTS ========================================================================================
import json
import traceback
from pathlib import Path

from src.builders import build_constraint_function, build_objective_function
from src.equilibria import load_equilibrium











#============== RUNNER =========================================================================================
def run_case(run_config):
    run_dir = Path(run_config["run_dir"])
    run_dir.mkdir(parents = True, exist_ok = True)

    status = {
        "paper_id": run_config["paper_id"],
        "case_id": run_config["case_id"],
        "variant_id": run_config["variant_id"],
        "success": False,
        "equilibrium_loaded": False,
        "continuation_returned": False,
        "optimization_returned": False,
        "skipped_objectives": [],
        "skipped_constraints": [],
        "message": None,
    }

    try:
        eq = load_equilibrium(run_config["equilibrium"])
        status["equilibrium_loaded"] = True

        solve_config = run_config.get("solve", {})
        if solve_config.get("run_continuation", False):
            continuation_kwargs = solve_config.get("continuation_kwargs", {})
            eq = eq.solve_continuation_automatic(**continuation_kwargs)
            status["continuation_returned"] = True

        optimization_config = run_config["optimization"]

        objective, skipped_objectives = build_objective_function(
            eq = eq,
            objective_configs = optimization_config.get("objectives", []),
        )

        constraints, skipped_constraints = build_constraint_function(
            eq = eq,
            constraint_configs = optimization_config.get("constraints", []),
        )

        status["skipped_objectives"] = skipped_objectives
        status["skipped_constraints"] = skipped_constraints

        optimizer = optimization_config.get("optimizer", "proximal-lsq-exact")
        optimize_kwargs = optimization_config.get("optimize_kwargs", {})

        if constraints is None:
            eq, result = eq.optimize(
                objective = objective,
                optimizer = optimizer,
                **optimize_kwargs,
            )
        else:
            eq, result = eq.optimize(
                objective = objective,
                constraints = constraints,
                optimizer = optimizer,
                **optimize_kwargs,
            )

        status["optimization_returned"] = True
        status["success"] = True
        status["message"] = "success"

        if run_config.get("outputs", {}).get("save_equilibrium", True):
            eq.save(str(run_dir / "optimized.h5"))

        result_path = run_dir / "optimizer_result.txt"
        with open(result_path, "w") as file:
            file.write(str(result))

    except Exception as error:
        status["success"] = False
        status["message"] = f"{type(error).__name__}: {error}"

        traceback_path = run_dir / "traceback.txt"
        with open(traceback_path, "w") as file:
            file.write(traceback.format_exc())

    status_path = run_dir / "status.json"
    with open(status_path, "w") as file:
        json.dump(status, file, indent = 4)

    return status