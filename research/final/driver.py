# research/final/driver.py











#============== IMPORTS ========================================================================================
import argparse
import json
from pathlib import Path

from papers import compose_run_config, list_papers
from src.run_case import run_case











#============== CLI ============================================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description = "Run a paper-configured DESC optimization from research/final/papers.py."
    )

    parser.add_argument(
        "--paper_id",
        type = str,
        required = True,
        choices = list_papers(),
        help = "Paper config key from papers.py.",
    )

    parser.add_argument(
        "--case_id",
        type = str,
        default = "base",
        help = "Case key inside the selected paper config.",
    )

    parser.add_argument(
        "--variant_id",
        type = str,
        default = "base",
        choices = ["base", "flo", "fno"],
        help = "Objective construction variant.",
    )

    parser.add_argument(
        "--dry_run",
        action = "store_true",
        help = "Write resolved config but do not run DESC.",
    )

    return parser.parse_args()











#============== MAIN ===========================================================================================
def main():
    args = parse_args()

    run_config = compose_run_config(
        paper_id = args.paper_id,
        case_id = args.case_id,
        variant_id = args.variant_id,
    )

    run_dir = Path(run_config["run_dir"])
    run_dir.mkdir(parents = True, exist_ok = True)

    resolved_config_path = run_dir / "resolved_config.json"
    with open(resolved_config_path, "w") as file:
        json.dump(run_config, file, indent = 4)

    if args.dry_run:
        print(f"Wrote resolved config to {resolved_config_path}")
        return

    status = run_case(run_config = run_config)

    status_path = run_dir / "status.json"
    with open(status_path, "w") as file:
        json.dump(status, file, indent = 4)

    print(f"Finished run. Status saved to {status_path}")





if __name__ == "__main__":
    main()