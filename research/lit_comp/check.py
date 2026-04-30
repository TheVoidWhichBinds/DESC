# check.py
"""Compare OG and recreated CHECK DESC output files for any paper recreation."""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from argparse import ArgumentParser

from helper import (
    compare_all,
    find_matching_check_files,
    get_case_dir,
    load_latest_equilibrium,
    print_comparison_rows,
    save_comparison_rows,
)










#========================================================================================================================================
# CLI
#========================================================================================================================================
def parse_args():
    """Parse command-line arguments."""

    parser = ArgumentParser(
        description = "Compare OG and CHECK DESC output files for a paper recreation.",
    )

    parser.add_argument(
        "--paper",
        required = True,
        help = "Paper recreation directory name, e.g. dudt2024.",
    )

    parser.add_argument(
        "--file",
        required = True,
        help = "Case name or HDF5 file name to compare.",
    )

    parser.add_argument(
        "--save",
        action = "store_true",
        help = "Save comparison CSV in the paper output directory.",
    )

    return parser.parse_args()










#========================================================================================================================================
# MAIN
#========================================================================================================================================
def main():
    """Run comparison."""

    args = parse_args()

    case_dir = get_case_dir(
        paper = args.paper,
        case = args.file,
    )

    if not case_dir.exists():
        raise FileNotFoundError(
            "Case directory does not exist: {}".format(case_dir)
        )

    original_file, check_file, case_name = find_matching_check_files(
        case_dir = case_dir,
        file_name = args.file,
    )

    print("")
    print("================================================================")
    print("DESC recreation check")
    print("================================================================")
    print("Paper:")
    print(args.paper)
    print("")
    print("Case:")
    print(case_name)
    print("")
    print("OG:")
    print(original_file)
    print("")
    print("CHECK:")
    print(check_file)

    eq_original = load_latest_equilibrium(
        path = original_file,
    )

    eq_check = load_latest_equilibrium(
        path = check_file,
    )

    rows = compare_all(
        paper = args.paper,
        case_name = case_name,
        eq_original = eq_original,
        eq_check = eq_check,
    )

    print_comparison_rows(
        rows = rows,
    )

    if args.save:
        csv_path = case_dir / "{}_OG_vs_CHECK_comparison.csv".format(case_name)

        save_comparison_rows(
            rows = rows,
            path = csv_path,
        )

        print("")
        print("Saved comparison CSV:")
        print(csv_path)










#========================================================================================================================================
# RUN
#========================================================================================================================================
if __name__ == "__main__":
    main()