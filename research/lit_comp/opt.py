# opt.py
#==============================================================================================================
#
# Runs one DESC optimization for one saved initial equilibrium, one paper, and one variant.
#
# Variant behavior:
#
#   variant = None:
#       Uses only base.py objectives and constraints.
#
#   variant = "FLO":
#       Uses base.py objectives.
#       Uses base.py constraints except FixPressure.
#       Appends FLO.py objectives and constraints.
#
#   variant = "FNO":
#       Uses base.py objectives.
#       Uses base.py constraints except FixPressure.
#       Appends FNO.py objectives and constraints.
#
#==============================================================================================================

from pathlib import Path
import traceback

try:
    from .helper import (
        get_paper_config,
        prepare_variant_configs,
        variant_label,
        make_output_dir,
        build_objective_function,
        build_constraints,
        save_optimization_outputs,
        summarize_result,
        save_json,
        save_text,
    )
except ImportError:
    from helper import (
        get_paper_config,
        prepare_variant_configs,
        variant_label,
        make_output_dir,
        build_objective_function,
        build_constraints,
        save_optimization_outputs,
        summarize_result,
        save_json,
        save_text,
    )










#==============================================================================================================
# Main Optimization Function
#==============================================================================================================

def run_optimization(
        eq,
        paper_id,
        variant = None,
        output_dir = None,
    ):
    """
    Run one optimization.
    """

    paper_config = get_paper_config(
        paper_id = paper_id,
    )

    label = variant_label(
        variant = variant,
    )

    if output_dir is None:
        output_dir = Path("research/final/outputs") / paper_id / label

    output_dir = make_output_dir(
        output_dir = output_dir,
    )

    eq_initial = eq.copy()

    optimization_config = dict(paper_config["optimization"])

    objective_configs, constraint_configs = prepare_variant_configs(
        paper_config = paper_config,
        variant = variant,
        eq_initial = eq_initial,
    )

    try:
        objective = build_objective_function(
            objective_configs = objective_configs,
            eq = eq,
        )

        constraints = build_constraints(
            constraint_configs = constraint_configs,
            eq = eq,
        )

        result = eq.optimize(
            objective = objective,
            constraints = constraints,
            optimizer = optimization_config.get("optimizer", "proximal-lsq-exact"),
            ftol = optimization_config.get("ftol", 1e-8),
            xtol = optimization_config.get("xtol", 1e-8),
            gtol = optimization_config.get("gtol", 1e-8),
            maxiter = optimization_config.get("maxiter", 100),
            verbose = optimization_config.get("verbose", 3),
        )

        saved_outputs = save_optimization_outputs(
            eq = eq,
            result = result,
            output_dir = output_dir,
            label = label,
        )

        run_summary = {
            "paper_id": paper_id,
            "variant": label,
            "success": True,
            "result": summarize_result(
                result = result,
            ),
            "outputs": saved_outputs,
        }

        save_json(
            data = run_summary,
            path = output_dir / f"{label}_run_summary.json",
        )

        return run_summary

    except Exception as error:
        error_text = traceback.format_exc()

        save_text(
            text = error_text,
            path = output_dir / f"{label}_error.txt",
        )

        run_summary = {
            "paper_id": paper_id,
            "variant": label,
            "success": False,
            "error": str(error),
            "traceback_path": str(output_dir / f"{label}_error.txt"),
        }

        save_json(
            data = run_summary,
            path = output_dir / f"{label}_run_summary.json",
        )

        return run_summary
