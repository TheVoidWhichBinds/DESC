import copy

from configs.global_config import RUNS_DIR
from src.build_eq import build_initial_equilibrium
from src.build_objectives import build_objective
from src.build_constraints import build_constraints
from src.build_optimizer import build_optimizer
from src.save import save_run_config, save_equilibrium, save_pickle










#============== CONFIG MERGING ===================================================================================
def merge_paper_with_variant(
        paper_config,
        variant_config,
    ):
    run_config = copy.deepcopy(paper_config)

    run_config["variant"] = {
        "variant_id": variant_config["variant_id"],
        "description": variant_config.get("description", ""),
    }

    run_config["variant_objectives"] = copy.deepcopy(
        variant_config.get("objective_additions", [])
    )

    run_config["variant_constraints"] = copy.deepcopy(
        variant_config.get("constraint_additions", [])
    )

    run_config["variant_constraint_removals"] = copy.deepcopy(
        variant_config.get("constraint_removals", [])
    )

    run_config["run_id"] = f"{run_config['paper_id']}/{variant_config['variant_id']}"

    return run_config
#==============================================================================================================










#========== prepare_run_directory ================================================================================
def prepare_run_directory(
        run_config,
    ):
    paper_id = run_config["paper_id"]
    variant_id = run_config["variant"]["variant_id"]

    run_dir = RUNS_DIR / paper_id / variant_id
    plot_dir = run_dir / "plots"

    run_dir.mkdir(parents = True, exist_ok = True)
    plot_dir.mkdir(parents = True, exist_ok = True)

    run_config["paths"] = {
        "run_dir": str(run_dir),
        "plot_dir": str(plot_dir),
        "run_config": str(run_dir / "run_config.json"),
        "eq_initial": str(run_dir / "eq_initial.h5"),
        "eq_final": str(run_dir / "eq_final.h5"),
        "optimization_result": str(run_dir / "optimization_result.pkl"),
        "metrics": str(run_dir / "metrics.json"),
        "objective_history": str(run_dir / "objective_history.csv"),
        "optimization_log": str(run_dir / "optimization_log.txt"),
    }

    return run_config
#==============================================================================================================










#========== run_case =============================================================================================
def run_case(
        paper_config,
        variant_config,
        qs = None,
        order = None,
    ):
    run_config = merge_paper_with_variant(
        paper_config = paper_config,
        variant_config = variant_config,
    )

    if qs is not None:
        if qs not in run_config["paper_objectives"].get("allowed_qs", []):
            raise ValueError(f"Invalid qs = {qs}")

        run_config["paper_objectives"]["qs"] = qs

    if order is not None:
        if order not in run_config["paper_objectives"].get("allowed_order", []):
            raise ValueError(f"Invalid order = {order}")

        run_config["paper_objectives"]["order"] = order
        run_config["perturb_options"]["order"] = order

    run_config = prepare_run_directory(
        run_config = run_config,
    )

    save_run_config(
        run_config = run_config,
        path = run_config["paths"]["run_config"],
    )

    eq_initial = build_initial_equilibrium(
        run_config = run_config,
    )

    objective = build_objective(
        eq = eq_initial,
        run_config = run_config,
    )

    constraints = build_constraints(
        eq = eq_initial,
        run_config = run_config,
    )

    optimizer = build_optimizer(
        run_config = run_config,
    )

    save_equilibrium(
        eq = eq_initial,
        path = run_config["paths"]["eq_initial"],
    )

    eq_final, result = run_optimization(
        eq = eq_initial,
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        run_config = run_config,
    )

    save_equilibrium(
        eq = eq_final,
        path = run_config["paths"]["eq_final"],
    )

    save_pickle(
        obj = result,
        path = run_config["paths"]["optimization_result"],
    )

    return eq_final, result, run_config
#==============================================================================================================










#========== run_optimization ======================================================================================
def run_optimization(
        eq,
        objective,
        constraints,
        optimizer,
        run_config,
    ):
    optimize_options = copy.deepcopy(run_config.get("optimize_options", {}))
    perturb_options = copy.deepcopy(run_config.get("perturb_options", {}))
    solve_options = copy.deepcopy(run_config.get("solve_options", {}))

    if perturb_options.get("opt_subspace_from_paper_driver", False):
        from src.build_constraints import get_paper_opt_subspace

        perturb_options["opt_subspace"] = get_paper_opt_subspace(
            eq = eq,
            run_config = run_config,
        )

        perturb_options.pop("opt_subspace_from_paper_driver")

    eq_final, result = eq.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        perturb_options = perturb_options,
        solve_options = solve_options,
        **optimize_options,
    )

    return eq_final, result
#==============================================================================================================