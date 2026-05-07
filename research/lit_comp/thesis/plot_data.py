# plot_data.py
#========================================================================================================================================
# IMPORTS
#========================================================================================================================================

import argparse
from pathlib import Path

import pandas as pd










#========================================================================================================================================
# PATHS
#========================================================================================================================================

THIS_DIR = Path(__file__).resolve().parent
TUTORIALS_DIR = THIS_DIR / "tutorials"










#========================================================================================================================================
# GENERAL CSV HELPERS
#========================================================================================================================================

def get_numbered_run_dirs(tutorial_dir):
    return sorted(
        [
            path
            for path in tutorial_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key = lambda path: int(path.name),
    )





def get_only_csv_file(run_dir):
    csv_files = sorted(run_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found in {run_dir}")

    if len(csv_files) > 1:
        raise RuntimeError(
            f"Expected exactly one CSV file in {run_dir}, but found {len(csv_files)}:\n"
            + "\n".join(str(path) for path in csv_files)
        )

    return csv_files[0]





def read_case_csv(csv_path):
    return pd.read_csv(csv_path)





def extract_cell(df, row, column):
    if isinstance(row, int):
        row_data = df.iloc[row]
    else:
        row_matches = df[df.iloc[:, 0].astype(str) == str(row)]

        if len(row_matches) == 0:
            raise KeyError(f"Could not find row label {row}")

        if len(row_matches) > 1:
            raise KeyError(f"Found multiple rows matching label {row}")

        row_data = row_matches.iloc[0]

    if isinstance(column, int):
        return row_data.iloc[column]

    if column not in df.columns:
        raise KeyError(f"Could not find column {column}")

    return row_data[column]










#========================================================================================================================================
# TUTORIAL-SPECIFIC EXTRACTION FUNCTIONS
#========================================================================================================================================

def extract_basic_qs_data(df):
    return {
        "FXD main objective": extract_cell(df, row = 0, column = 1),
        "FREE main objective": extract_cell(df, row = 0, column = 2),
    }





def extract_adv_qs_data(df):
    return {
        "FXD main objective": extract_cell(df, row = 0, column = 1),
        "FREE main objective": extract_cell(df, row = 0, column = 2),
    }





def extract_balloon_data(df):
    return {
        "FXD main objective": extract_cell(df, row = 0, column = 1),
        "FREE main objective": extract_cell(df, row = 0, column = 2),
    }





def extract_neoclassical_data(df):
    return {
        "FXD main objective": extract_cell(df, row = 0, column = 1),
        "FREE main objective": extract_cell(df, row = 0, column = 2),
    }










#========================================================================================================================================
# DISPATCH
#========================================================================================================================================

TUTORIAL_EXTRACTORS = {
    "basic_qs": extract_basic_qs_data,
    "adv_qs": extract_adv_qs_data,
    "balloon": extract_balloon_data,
    "neoclassical": extract_neoclassical_data,
}










#========================================================================================================================================
# TABLE CONSTRUCTION
#========================================================================================================================================

def build_tutorial_table(tutorial):
    if tutorial not in TUTORIAL_EXTRACTORS:
        raise ValueError(
            f"Unknown tutorial {tutorial}. Options are: "
            + ", ".join(sorted(TUTORIAL_EXTRACTORS))
        )

    tutorial_dir = TUTORIALS_DIR / tutorial

    if not tutorial_dir.exists():
        raise FileNotFoundError(f"Could not find tutorial directory: {tutorial_dir}")

    extractor = TUTORIAL_EXTRACTORS[tutorial]
    run_dirs = get_numbered_run_dirs(tutorial_dir)

    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No numbered run folders found in {tutorial_dir}")

    rows = []

    for run_dir in run_dirs:
        csv_path = get_only_csv_file(run_dir)
        df = read_case_csv(csv_path)

        extracted = extractor(df)

        row = {
            "Run": run_dir.name,
        }

        row.update(extracted)

        rows.append(row)

    table = pd.DataFrame(rows)
    return table





def save_tutorial_table(tutorial, table):
    output_path = TUTORIALS_DIR / tutorial / f"{tutorial}_objective_table.csv"
    table.to_csv(output_path, index = False)

    return output_path










#========================================================================================================================================
# CLI
#========================================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tutorial",
        required = True,
        choices = sorted(TUTORIAL_EXTRACTORS),
        help = "Tutorial folder to process.",
    )

    return parser.parse_args()





def main():
    args = parse_args()

    table = build_tutorial_table(args.tutorial)
    output_path = save_tutorial_table(args.tutorial, table)

    print()
    print(f"{args.tutorial} objective table")
    print("=" * len(f"{args.tutorial} objective table"))
    print(table.to_string(index = False))
    print()
    print(f"Saved table to: {output_path}")










#========================================================================================================================================
# ENTRY POINT
#========================================================================================================================================

if __name__ == "__main__":
    main()