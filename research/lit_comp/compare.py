# compare.py
#==============================================================================================================
#
# Compare DESC objective outputs and paper/case-specific objective outputs between OG, CHECK, and FNO files.
#
# Usage:
#   python3 research/lit_comp/compare.py --paper dudt2024 --case helical_qs
#
#==============================================================================================================

import argparse

from helper import (
    compare_objective_set,
    default_desc_objectives,
    dudt2024_helical_qs_objectives,
    find_h5_files,
    get_case_dir,
    write_table_csv,
)










#==============================================================================================================
# Case-specific objective registry:
#==============================================================================================================

CASE_OBJECTIVES = {
    "dudt2024": {
        "helical_qs": dudt2024_helical_qs_objectives,
    },
}










#==============================================================================================================
# Public comparison functions:
#==============================================================================================================

def DESC_obj(
        paper : str,
        case : str,
    ):
    """
    Compares the general DESC objective values between OG, CHECK, and FNO files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(case_dir)

    rows = compare_objective_set(
        files = files,
        objective_getter = default_desc_objectives,
    )

    csv_path = case_dir / f"{case}_DESC_obj.csv"

    write_table_csv(
        rows = rows,
        path = csv_path,
    )

    return rows





def case_obj(
        paper : str,
        case : str,
    ):
    """
    Compares paper/case-specific objective values between OG, CHECK, and FNO files.
    """

    case_dir = get_case_dir(
        paper = paper,
        case = case,
    )

    files = find_h5_files(case_dir)

    def get_case_objectives(eq):
        objective_getter = CASE_OBJECTIVES.get(paper, {}).get(case, None)

        if objective_getter is None:
            return []

        return objective_getter(eq)

    rows = compare_objective_set(
        files = files,
        objective_getter = get_case_objectives,
    )

    csv_path = case_dir / f"{case}_case_obj.csv"

    write_table_csv(
        rows = rows,
        path = csv_path,
    )

    return rows










#==============================================================================================================
# Command-line interface:
#==============================================================================================================

def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper",
        required = True,
        help = "Paper name, e.g. dudt2024.",
    )

    parser.add_argument(
        "--case",
        required = True,
        help = "Case name, e.g. helical_qs.",
    )

    return parser.parse_args()





def main():
    """
    Runs both DESC and case-specific comparisons.
    """

    args = parse_args()

    print("\n" + "=" * 120)
    print(f"Comparing objectives for paper = {args.paper}, case = {args.case}")
    print("=" * 120 + "\n")

    DESC_obj(
        paper = args.paper,
        case = args.case,
    )

    case_obj(
        paper = args.paper,
        case = args.case,
    )

    case_dir = get_case_dir(
        paper = args.paper,
        case = args.case,
    )

    print("\nFinished.")
    print(f"Tables written to: {case_dir}\n")





if __name__ == "__main__":
    main()