# data.py
#==============================================================================================================
#
# Build thesis CSV tables from tutorial comparison CSV files.
#
# Usage:
#   cd /Users/macdaddi/DESC
#   python3 research/lit_comp/thesis/data.py
#
# Outputs:
#   research/lit_comp/thesis/tables/balloon_balloon_table.csv
#   research/lit_comp/thesis/tables/basic_qs_tripleQS_table.csv
#   research/lit_comp/thesis/tables/basic_qs_twotermQH_table.csv
#
#==============================================================================================================

from pathlib import Path
import csv
import math









#==============================================================================================================
# CONFIGURATION
#==============================================================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LIT_COMP_DIR = SCRIPT_DIR.parent
TUTORIALS_DIR = LIT_COMP_DIR / "tutorials"
OUTPUT_DIR = SCRIPT_DIR / "tables"

SIG_FIGS = 4

EXCLUDED_METRICS = {
    "size",
    "status",
}

METRIC_DISPLAY_NAMES = {
    "l2": "l2",
    "max_abs": "max",
    "mean_abs": "mean",
    "rms": "rms",
}

OBJECTIVE_DISPLAY_NAMES = {
    "ForceBalance": "FB",
    "BallooningStability": "BS",
    "Aspect Ratio": "AR",
    "AspectRatio": "AR",
    "GenericObjective curvature_k2_rho": "K2",
    "QuasisymmetryTripleProduct": "QS3",
    "QuasisymmetryTwoTerm": "QS2",
}

TABLE_SPECS = (
    {
        "name": "balloon_balloon",
        "source_dir": TUTORIALS_DIR / "balloon" / "balloon",
    },
    {
        "name": "basic_qs_tripleQS",
        "source_dir": TUTORIALS_DIR / "basic_qs" / "tripleQS",
    },
    {
        "name": "basic_qs_twotermQH",
        "source_dir": TUTORIALS_DIR / "basic_qs" / "twotermQH",
    },
)









#==============================================================================================================
# PATH HELPERS
#==============================================================================================================

def is_run_dir(
        path,
    ):
    """
    Return True for numbered run folders.
    """

    path = Path(path)

    return path.is_dir() and path.name.isdigit()





def find_run_dirs(
        source_dir,
    ):
    """
    Return numbered run folders sorted numerically.
    """

    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir}"
        )

    run_dirs = sorted(
        (
            path
            for path in source_dir.iterdir()
            if is_run_dir(
                path = path,
            )
        ),
        key = lambda path: int(path.name),
    )

    if len(run_dirs) == 0:
        raise FileNotFoundError(
            f"No numbered run folders found in: {source_dir}"
        )

    return tuple(run_dirs)





def find_case_csv(
        run_dir,
    ):
    """
    Return the one case-objective CSV file inside a run folder.
    """

    run_dir = Path(run_dir)

    csv_paths = sorted(
        run_dir.glob("*_case_obj.csv")
    )

    if len(csv_paths) == 0:
        raise FileNotFoundError(
            f"No *_case_obj.csv file found in: {run_dir}"
        )

    if len(csv_paths) > 1:
        raise RuntimeError(
            f"Expected one *_case_obj.csv file in {run_dir}, found {len(csv_paths)}."
        )

    return csv_paths[0]





def run_label_from_dir(
        run_dir,
    ):
    """
    Convert 001, 002, ... into 01, 02, ... for table rows.
    """

    return f"{int(Path(run_dir).name):02d}"









#==============================================================================================================
# DISPLAY HELPERS
#==============================================================================================================

def display_metric(
        metric,
    ):
    """
    Return a compact metric name for output headers.
    """

    return METRIC_DISPLAY_NAMES.get(
        metric,
        metric,
    )





def display_objective(
        objective,
    ):
    """
    Return a compact objective name for output headers with no spaces.
    """

    display_name = OBJECTIVE_DISPLAY_NAMES.get(
        objective,
        objective,
    )

    return str(display_name).replace(
        " ",
        "",
    )





def display_column(
        objective,
        metric,
        construction,
    ):
    """
    Return one compact output column name with no spaces.
    """

    return "".join(
        (
            display_objective(
                objective = objective,
            ),
            display_metric(
                metric = metric,
            ),
            construction,
        )
    )









#==============================================================================================================
# VALUE HELPERS
#==============================================================================================================

def parse_float(
        value,
    ):
    """
    Convert a CSV cell to float when possible.
    """

    try:
        parsed = float(value)

    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None

    return parsed





def format_scientific(
        value,
        sig_figs = SIG_FIGS,
    ):
    """
    Format numeric values in scientific notation with the requested significant figures.
    """

    parsed = parse_float(
        value = value,
    )

    if parsed is None:
        return str(value)

    decimal_places = max(
        sig_figs - 1,
        0,
    )

    return f"{parsed:.{decimal_places}e}"









#==============================================================================================================
# CSV READING
#==============================================================================================================

def read_case_rows(
        csv_path,
    ):
    """
    Read one case-objective CSV into ordered objective-metric rows.
    """

    rows = []

    with open(
            csv_path,
            "r",
            newline = "",
        ) as file_obj:

        reader = csv.DictReader(
            file_obj,
        )

        required_fields = {
            "objective",
            "metric",
            "FXD",
            "FREE",
        }

        missing_fields = required_fields.difference(
            reader.fieldnames or []
        )

        if len(missing_fields) > 0:
            raise ValueError(
                f"CSV file {csv_path} is missing columns: {sorted(missing_fields)}"
            )

        for row in reader:
            objective = row["objective"]
            metric = row["metric"]

            if metric in EXCLUDED_METRICS:
                continue

            rows.append(
                {
                    "objective": objective,
                    "metric": metric,
                    "FXD": row["FXD"],
                    "FREE": row["FREE"],
                }
            )

    return rows





def build_table_data(
        source_dir,
    ):
    """
    Build a wide table from all run-local case-objective CSV files.
    """

    columns = []
    column_set = set()
    table_rows = []

    for run_dir in find_run_dirs(
            source_dir = source_dir,
        ):

        csv_path = find_case_csv(
            run_dir = run_dir,
        )

        case_rows = read_case_rows(
            csv_path = csv_path,
        )

        values = {}

        for case_row in case_rows:
            key = (
                case_row["objective"],
                case_row["metric"],
            )

            if key not in column_set:
                column_set.add(
                    key,
                )
                columns.append(
                    key,
                )

            values[key] = {
                "FXD": format_scientific(
                    value = case_row["FXD"],
                ),
                "FREE": format_scientific(
                    value = case_row["FREE"],
                ),
            }

        table_rows.append(
            {
                "run": run_label_from_dir(
                    run_dir = run_dir,
                ),
                "values": values,
            }
        )

    return columns, table_rows









#==============================================================================================================
# TABLE WRITERS
#==============================================================================================================

def make_header(
        columns,
    ):
    """
    Make the wide CSV header.
    """

    header = [
        "Run",
    ]

    for objective, metric in columns:
        header.extend(
            [
                display_column(
                    objective = objective,
                    metric = metric,
                    construction = "FXD",
                ),
                display_column(
                    objective = objective,
                    metric = metric,
                    construction = "FREE",
                ),
            ]
        )

    return header





def make_output_rows(
        columns,
        table_rows,
    ):
    """
    Make the wide CSV body rows.
    """

    output_rows = []

    for table_row in table_rows:
        output_row = [
            table_row["run"],
        ]

        for key in columns:
            pair = table_row["values"].get(
                key,
                {},
            )
            output_row.extend(
                [
                    pair.get(
                        "FXD",
                        "",
                    ),
                    pair.get(
                        "FREE",
                        "",
                    ),
                ]
            )

        output_rows.append(
            output_row,
        )

    return output_rows





def write_wide_csv(
        output_path,
        columns,
        table_rows,
    ):
    """
    Write one wide CSV table with compact headers and scientific-notation values.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    header = make_header(
        columns = columns,
    )
    output_rows = make_output_rows(
        columns = columns,
        table_rows = table_rows,
    )

    with open(
            output_path,
            "w",
            newline = "",
        ) as file_obj:

        writer = csv.writer(
            file_obj,
        )
        writer.writerow(
            header,
        )
        writer.writerows(
            output_rows,
        )









#==============================================================================================================
# MAIN
#==============================================================================================================

def make_table(
        spec,
    ):
    """
    Make one CSV table from one source optimization folder.
    """

    columns, table_rows = build_table_data(
        source_dir = spec["source_dir"],
    )

    output_path = OUTPUT_DIR / f"{spec['name']}_table.csv"

    write_wide_csv(
        output_path = output_path,
        columns = columns,
        table_rows = table_rows,
    )

    return output_path





def main():
    """
    Build all requested tutorial data tables.
    """

    OUTPUT_DIR.mkdir(
        parents = True,
        exist_ok = True,
    )

    for spec in TABLE_SPECS:
        output_path = make_table(
            spec = spec,
        )

        print(
            f"Wrote {output_path}"
        )





if __name__ == "__main__":
    main()
