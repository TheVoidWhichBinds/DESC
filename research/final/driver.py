import argparse
import sys
from pathlib import Path
from configs.papers.registry import get_paper_config
from configs.variants.registry import get_variant_config
from src.run_case import run_case










#============== PATH SETUP =======================================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
#==============================================================================================================










#========== parse_args ===========================================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description = "Run recreated literature DESC optimization with optional FLO/FNO overlay.",
    )

    parser.add_argument(
        "--paper_id",
        required = True,
        help = "Paper configuration ID.",
    )

    parser.add_argument(
        "--variant_id",
        required = True,
        choices = ["base", "FLO", "FNO"],
        help = "Variant ID.",
    )

    parser.add_argument(
        "--qs",
        default = None,
        choices = ["B", "C", "T"],
        help = "QS objective choice for Dudt 2022 base-paper runs.",
    )

    parser.add_argument(
        "--order",
        default = None,
        type = int,
        choices = [1, 2],
        help = "Perturbation order for Dudt 2022 runs.",
    )

    return parser.parse_args()
#==============================================================================================================










#========== main =================================================================================================
def main():
    args = parse_args()

    paper_config = get_paper_config(
        paper_id = args.paper_id,
    )

    variant_config = get_variant_config(
        variant_id = args.variant_id,
    )

    eq_final, result, run_config = run_case(
        paper_config = paper_config,
        variant_config = variant_config,
        qs = args.qs,
        order = args.order,
    )

    print(f"Finished run: {run_config['run_id']}")
    print(f"Saved outputs to: {run_config['paths']['run_dir']}")
#==============================================================================================================


if __name__ == "__main__":
    main()