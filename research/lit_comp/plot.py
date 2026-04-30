# plot.py
#==============================================================================================================
#
# Paper-specific plotting for DESC/research/final.
#
# This file takes a paper_id and overlays base/FLO/FNO outputs for that paper.
#
# Usage from repo root:
#
#   python research/final/plot.py --paper_id dudt2022
#
# Optional:
#
#   python research/final/plot.py --paper_id dudt2022 --outputs_dir research/final/outputs
#
#==============================================================================================================










import argparse

import matplotlib.pyplot as plt

from .helper import (
    choose_objective_column,
    get_existing_variant_dirs,
    get_paper_plot_dir,
    load_comparison_text,
    load_objective_history,
)










#==============================================================================================================
# Plot Configuration Registry
#==============================================================================================================

PLOT_CONFIGS = {
    "dudt2022": {
        "figures": (
            "objective_history",
            "comparison_table",
        ),

        "kwargs": {
            "figsize": (7, 5),
            "dpi": 300,
            "save_format": "png",
        },
    },










    "conlin2022": {
        "figures": (
            "objective_history",
            "comparison_table",
        ),

        "kwargs": {
            "figsize": (7, 5),
            "dpi": 300,
            "save_format": "png",
        },
    },










    "panici2023": {
        "figures": (
            "objective_history",
            "comparison_table",
        ),

        "kwargs": {
            "figsize": (7, 5),
            "dpi": 300,
            "save_format": "png",
        },
    },










    "dudt2024": {
        "figures": (
            "objective_history",
            "comparison_table",
        ),

        "kwargs": {
            "figsize": (7, 5),
            "dpi": 300,
            "save_format": "png",
        },
    },










    "conlin2024constraints": {
        "figures": (
            "objective_history",
            "comparison_table",
        ),

        "kwargs": {
            "figsize": (7, 5),
            "dpi": 300,
            "save_format": "png",
        },
    },
}




















#==============================================================================================================
# Shared Plot Settings
#==============================================================================================================

VARIANT_IDS = (
    "base",
    "FLO",
    "FNO",
)










#==============================================================================================================
# Objective History Overlay
#==============================================================================================================

def plot_objective_history_overlay(paper_id, outputs_dir, plot_kwargs):
    variant_dirs = get_existing_variant_dirs(
        outputs_dir = outputs_dir,
        paper_id = paper_id,
        variant_ids = VARIANT_IDS,
    )

    if len(variant_dirs) == 0:
        return None

    fig, ax = plt.subplots(
        figsize = plot_kwargs["figsize"],
    )

    plotted_anything = False

    for variant_id, variant_dir in variant_dirs.items():
        history = load_objective_history(
            variant_dir = variant_dir,
        )

        if history is None:
            continue

        objective_column = choose_objective_column(
            history = history,
        )

        if objective_column is None:
            continue

        ax.plot(
            history["iteration"],
            history[objective_column],
            label = variant_id,
        )

        plotted_anything = True

    if plotted_anything is False:
        plt.close(fig)
        return None

    ax.set_title(f"{paper_id}: Objective History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(
        True,
        alpha = 0.3,
    )

    plot_dir = get_paper_plot_dir(
        outputs_dir = outputs_dir,
        paper_id = paper_id,
    )

    output_path = plot_dir / f"objective_history_overlay.{plot_kwargs['save_format']}"

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi = plot_kwargs["dpi"],
    )
    plt.close(fig)

    return output_path










#==============================================================================================================
# Comparison Text Overlay
#==============================================================================================================

def write_comparison_overlay(paper_id, outputs_dir):
    variant_dirs = get_existing_variant_dirs(
        outputs_dir = outputs_dir,
        paper_id = paper_id,
        variant_ids = VARIANT_IDS,
    )

    if len(variant_dirs) == 0:
        return None

    lines = []

    title = f"{paper_id} Variant Comparison"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    found_anything = False

    for variant_id, variant_dir in variant_dirs.items():
        comparison_text = load_comparison_text(
            variant_dir = variant_dir,
        )

        if comparison_text is None:
            continue

        found_anything = True

        lines.append("")
        lines.append(variant_id)
        lines.append("-" * len(variant_id))
        lines.append(comparison_text.strip())
        lines.append("")

    if found_anything is False:
        return None

    plot_dir = get_paper_plot_dir(
        outputs_dir = outputs_dir,
        paper_id = paper_id,
    )

    output_path = plot_dir / "comparison_overlay.txt"

    output_path.write_text(
        "\n".join(lines),
    )

    return output_path










#==============================================================================================================
# Plot Dispatcher
#==============================================================================================================

def plot_paper(paper_id, outputs_dir):
    if paper_id not in PLOT_CONFIGS:
        raise ValueError(
            f"No plot configuration found for paper_id = {paper_id!r}."
        )

    plot_config = PLOT_CONFIGS[paper_id]
    plot_kwargs = plot_config["kwargs"]

    output_paths = []

    if "objective_history" in plot_config["figures"]:
        output_path = plot_objective_history_overlay(
            paper_id = paper_id,
            outputs_dir = outputs_dir,
            plot_kwargs = plot_kwargs,
        )

        if output_path is not None:
            output_paths.append(output_path)

    if "comparison_table" in plot_config["figures"]:
        output_path = write_comparison_overlay(
            paper_id = paper_id,
            outputs_dir = outputs_dir,
        )

        if output_path is not None:
            output_paths.append(output_path)

    return output_paths










#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper_id",
        required = True,
        type = str,
        help = "Paper ID to plot.",
    )

    parser.add_argument(
        "--outputs_dir",
        default = "research/final/outputs",
        type = str,
        help = "Directory containing paper output folders.",
    )

    return parser.parse_args()





def main():
    args = parse_args()

    output_paths = plot_paper(
        paper_id = args.paper_id,
        outputs_dir = args.outputs_dir,
    )

    if len(output_paths) == 0:
        print(f"No plot outputs were generated for paper_id = {args.paper_id}")

    else:
        print(f"Generated {len(output_paths)} plot outputs:")

        for output_path in output_paths:
            print(f"  {output_path}")










if __name__ == "__main__":
    main()