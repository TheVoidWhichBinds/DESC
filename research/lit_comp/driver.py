# driver.py
#==============================================================================================================
#
# Top-level controller for DESC/research/final.
#
# For one paper_id, this file:
#   1. Loads the saved initial equilibrium from base.py
#   2. Runs the base optimization
#   3. Runs the FLO optimization
#   4. Runs the FNO optimization
#   5. Compares the final run summaries in a dedicated comparison function
#   6. Saves comparison outputs
#
#==============================================================================================================

from pathlib import Path
import argparse

try:
    from .opt import run_optimization

    from .helper import (
        get_paper_config,
        load_saved_equilibrium,
        make_output_dir,
        save_json,
        save_text,
    )
except ImportError:
    from opt import run_optimization

    from helper import (
        get_paper_config,
        load_saved_equilibrium,
        make_output_dir,
        save_json,
        save_text,
    )










#==============================================================================================================
# Comparison Helpers
#==============================================================================================================

def extract_result_value(
        run_summary,
        key,
        default = None,
    ):
    """
    Safely extract a value from a run summary result dictionary.
    """

    if not run_summary:
        return default

    result = run_summary.get("result", {})

    if not isinstance(result, dict):
        return default

    return result.get(key, default)









def compare_variant_results(
        results,
        output_dir,
    ):
    """
    Compare base, FLO, and FNO optimization outputs.
    """

    output_dir = make_output_dir(
        output_dir = output_dir,
    )

    comparison = {}

    for label in (
            "base",
            "FLO",
            "FNO",
        ):
        run_summary = results.get(label, {})

        comparison[label] = {
            "success": bool(run_summary.get("success", False)),
            "cost": extract_result_value(
                run_summary = run_summary,
                key = "cost",
            ),
            "optimality": extract_result_value(
                run_summary = run_summary,
                key = "optimality",
            ),
            "nit": extract_result_value(
                run_summary = run_summary,
                key = "nit",
            ),
            "nfev": extract_result_value(
                run_summary = run_summary,
                key = "nfev",
            ),
            "message": extract_result_value(
                run_summary = run_summary,
                key = "message",
                default = run_summary.get("error", ""),
            ),
            "eq_path": run_summary.get("outputs", {}).get("eq_path"),
            "summary_path": run_summary.get("outputs", {}).get("summary_path"),
        }

    save_json(
        data = comparison,
        path = output_dir / "comparison.json",
    )

    comparison_text = format_comparison_table(
        comparison = comparison,
    )

    save_text(
        text = comparison_text,
        path = output_dir / "comparison.txt",
    )

    return comparison









def format_comparison_table(
        comparison,
    ):
    """
    Format a compact comparison table for base, FLO, and FNO.
    """

    lines = []

    lines.append("Optimization comparison")
    lines.append("=" * 100)
    lines.append(
        f"{'variant':<12} {'success':<10} {'cost':<20} {'optimality':<20} {'nit':<10} {'nfev':<10}"
    )
    lines.append("-" * 100)

    for label in (
            "base",
            "FLO",
            "FNO",
        ):
        row = comparison.get(label, {})

        lines.append(
            f"{label:<12} "
            f"{str(row.get('success')):<10} "
            f"{str(row.get('cost')):<20} "
            f"{str(row.get('optimality')):<20} "
            f"{str(row.get('nit')):<10} "
            f"{str(row.get('nfev')):<10}"
        )

    lines.append("=" * 100)

    return "\n".join(lines)









#==============================================================================================================
# Main Driver
#==============================================================================================================

def run_all_variants(
        paper_id,
        case_id = "case_000",
        output_root = "research/final/outputs",
    ):
    """
    Run base, FLO, and FNO optimizations for one saved paper equilibrium.
    """

    paper_config = get_paper_config(
        paper_id = paper_id,
    )

    case_output_dir = make_output_dir(
        output_dir = Path(output_root) / paper_id / case_id,
    )

    eq_initial = load_saved_equilibrium(
        eq_path = paper_config["eq_path"],
    )

    results = {}

    for variant in (
            None,
            "FLO",
            "FNO",
        ):
        label = "base" if variant is None else variant

        variant_output_dir = make_output_dir(
            output_dir = case_output_dir / label,
        )

        eq_start = eq_initial.copy()

        results[label] = run_optimization(
            eq = eq_start,
            paper_id = paper_id,
            variant = variant,
            output_dir = variant_output_dir,
        )

    save_json(
        data = results,
        path = case_output_dir / "all_run_summaries.json",
    )

    comparison = compare_variant_results(
        results = results,
        output_dir = case_output_dir,
    )

    return results, comparison









#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description = "Run base, FLO, and FNO optimizations for one DESC paper case.",
    )

    parser.add_argument(
        "--paper_id",
        type = str,
        required = True,
        help = "Paper ID from base.PAPERS.",
    )

    parser.add_argument(
        "--case_id",
        type = str,
        default = "case_000",
        help = "Case ID for output folder naming.",
    )

    parser.add_argument(
        "--output_root",
        type = str,
        default = "research/final/outputs",
        help = "Root directory for outputs.",
    )

    return parser.parse_args()









def main():
    """
    Command-line entry point.
    """

    args = parse_args()

    run_all_variants(
        paper_id = args.paper_id,
        case_id = args.case_id,
        output_root = args.output_root,
    )









if __name__ == "__main__":
    main()
