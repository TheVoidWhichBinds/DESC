# data.py
#==============================================================================================================
#
# Build thesis-ready objective tables from tutorial comparison CSV files.
#
# Usage:
#   cd research/lit_comp/thesis
#   python3 data.py
#
# Outputs:
#   thesis/tripleQS_table.txt
#   thesis/twotermQH_table.txt
#   thesis/balloon_table.txt
#
#==============================================================================================================

from pathlib import Path
import csv
import math








#==============================================================================================================
# TABLE SETTINGS
#==============================================================================================================

TABLE_CONFIGS = (
    {
        "name": "tripleQS",
        "folder": Path("tutorials") / "basic_qs" / "tripleQS",
        "output": "tripleQS_table.txt",
    },
    {
        "name": "twotermQH",
        "folder": Path("tutorials") / "basic_qs" / "twotermQH",
        "output": "twotermQH_table.txt",
    },
    {
        "name": "balloon",
        "folder": Path("tutorials") / "balloon" / "balloon",
        "output": "balloon_table.txt",
    },
)

OBJECTIVE_ACRONYMS = {
    "ForceBalance": "FB",
    "BallooningStability": "BS",
    "AspectRatio": "AR",
    "GenericObjective curvature_k2_rho": "K2",
    "QuasisymmetryTripleProduct": "QS3",
    "QuasisymmetryTwoTerm": "QS2",
    "QuasisymmetryTwoTermQH": "QS2",
    "QuasisymmetryBoozer": "QSB",
    "Isodynamicity": "ISO",
    "MercierStability": "MERC",
}

METRIC_NAMES = {
    "l2": "l2",
    "max_abs": "max",
    "mean_abs": "mean",
    "rms": "rms",
}

METRIC_ORDER = (
    "l2",
    "max_abs",
    "mean_abs",
    "rms",
)

VARIANTS = (
    "FXD",
    "FREE",
)








#==============================================================================================================
# PATH HELPERS
#==============================================================================================================

def get_thesis_dir():
    """
    Return the thesis folder containing this script.
    """

    return Path(__file__).resolve().parent





def get_lit_comp_dir():
    """
    Return the research/lit_comp folder.
    """

    return get_thesis_dir().parent





def get_run_dirs(
        folder,
    ):
    """
    Return sorted numbered run folders inside one optimization folder.
    """

    folder = Path(folder)

    return tuple(
        sorted(
            path for path in folder.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    )





def find_case_csv(
        run_dir,
    ):
    """
    Return the run-local case-objective CSV file.
    """

    matches = sorted(
        Path(run_dir).glob("*_case_obj.csv")
    )

    if len(matches) == 0:
        return None

    return matches[0]








#==============================================================================================================
# VALUE FORMATTING
#==============================================================================================================

def get_objective_acronym(
        objective,
    ):
    """
    Return a compact objective acronym.
    """

    if objective in OBJECTIVE_ACRONYMS:
        return OBJECTIVE_ACRONYMS[objective]

    compact = "".join(
        character for character in objective
        if character.isalnum()
    )

    return compact





def make_column_name(
        objective,
        metric,
        variant,
    ):
    """
    Build one compact table column name.
    """

    return f"{METRIC_NAMES[metric]}{get_objective_acronym(objective)}{variant}"





def format_value(
        value,
    ):
    """
    Format one table value in four-significant-figure scientific notation.
    """

    if value is None or value == "":
        return ""

    number = float(value)

    if not math.isfinite(number):
        return "nan"

    return f"{number:.3e}"








#==============================================================================================================
# CSV PARSING
#==============================================================================================================

def read_case_csv(
        csv_path,
    ):
    """
    Read one comparison CSV into an ordered row dictionary and ordered column list.
    """

    row_values = {}
    columns = []

    with open(csv_path, newline = "") as file:
        reader = csv.DictReader(file)

        for row in reader:
            objective = row["objective"]
            metric = row["metric"]

            if metric not in METRIC_ORDER:
                continue

            for variant in VARIANTS:
                column_name = make_column_name(
                    objective = objective,
                    metric = metric,
                    variant = variant,
                )

                if column_name not in columns:
                    columns.append(
                        column_name,
                    )

                row_values[column_name] = format_value(
                    value = row.get(
                        variant,
                        "",
                    )
                )

    return row_values, columns





def build_table_rows(
        folder,
    ):
    """
    Build table rows and columns from one tutorial optimization folder.
    """

    run_dirs = get_run_dirs(
        folder = folder,
    )

    rows = []
    columns = []

    for run_dir in run_dirs:
        csv_path = find_case_csv(
            run_dir = run_dir,
        )

        if csv_path is None:
            continue

        row_values, row_columns = read_case_csv(
            csv_path = csv_path,
        )

        for column in row_columns:
            if column not in columns:
                columns.append(
                    column,
                )

        rows.append(
            {
                "Run": run_dir.name,
                **row_values,
            }
        )

    return rows, columns








#==============================================================================================================
# TEXT TABLE WRITING
#==============================================================================================================

def build_separator(
        widths,
    ):
    """
    Build a horizontal separator for a fixed-width ASCII table.
    """

    pieces = []

    for width in widths:
        pieces.append(
            "-" * (width + 2)
        )

    return "+" + "+".join(pieces) + "+"





def build_table_line(
        values,
        widths,
    ):
    """
    Build one aligned row for a fixed-width ASCII table.
    """

    pieces = []

    for value, width in zip(values, widths):
        pieces.append(
            f" {str(value):>{width}} "
        )

    return "|" + "|".join(pieces) + "|"





def render_ascii_table(
        title,
        rows,
        columns,
    ):
    """
    Render rows as an aligned ASCII table.
    """

    headers = [
        "Run",
    ] + list(columns)

    widths = []

    for header in headers:
        max_width = len(header)

        for row in rows:
            max_width = max(
                max_width,
                len(str(row.get(header, ""))),
            )

        widths.append(
            max_width,
        )

    separator = build_separator(
        widths = widths,
    )

    lines = [
        title,
        separator,
        build_table_line(
            values = headers,
            widths = widths,
        ),
        separator,
    ]

    for row in rows:
        values = [
            row.get(
                header,
                "",
            )
            for header in headers
        ]

        lines.append(
            build_table_line(
                values = values,
                widths = widths,
            )
        )

    lines.append(
        separator,
    )

    return "\n".join(lines) + "\n"





def write_table(
        title,
        rows,
        columns,
        output_path,
    ):
    """
    Write one aligned ASCII table.
    """

    table = render_ascii_table(
        title = title,
        rows = rows,
        columns = columns,
    )

    with open(output_path, "w") as file:
        file.write(table)

    return output_path








#==============================================================================================================
# MAIN
#==============================================================================================================

def build_all_tables():
    """
    Build all thesis tables from tutorial CSV outputs.
    """

    lit_comp_dir = get_lit_comp_dir()
    thesis_dir = get_thesis_dir()

    output_paths = []

    for config in TABLE_CONFIGS:
        folder = lit_comp_dir / config["folder"]

        rows, columns = build_table_rows(
            folder = folder,
        )

        output_path = thesis_dir / config["output"]

        write_table(
            title = config["name"],
            rows = rows,
            columns = columns,
            output_path = output_path,
        )

        output_paths.append(
            output_path,
        )

    return output_paths





def main():
    """
    Build and save the thesis tables.
    """

    output_paths = build_all_tables()

    print("")
    print("Saved thesis tables:")
    print("")

    for output_path in output_paths:
        print(output_path)

    print("")





if __name__ == "__main__":
    main()
