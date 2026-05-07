# plot_data.py
#==============================================================================================================
#
# Build compact results tables from tutorial comparison CSV files.
#
# Expected layout after running the tutorial driver and compare.py:
#
#   research/lit_comp/
#       thesis/
#           plot_data.py
#       tutorials/
#           basic_qs/
#           tripleQS/
#               001/
#                   *.csv
#               002/
#                   *.csv
#           twotermQH/
#               001/
#                   *.csv
#       adv_qs/
#           multigrid/
#               001/
#                   *.csv
#           auglag/
#               001/
#                   *.csv
#       balloon/
#           balloon/
#               001/
#                   *.csv
#
# Usage:
#   cd research/lit_comp/thesis
#   python3 plot_data.py --tutorial basic_qs
#   python3 plot_data.py --tutorial adv_qs
#   python3 plot_data.py --tutorial balloon
#   python3 plot_data.py --tutorial neoclassical
#
# Optional:
#   python3 plot_data.py --tutorial adv_qs --table auglag
#   python3 plot_data.py --tutorial basic_qs --metric rms
#   python3 plot_data.py --tutorial basic_qs --strict
#
# Outputs, one pair per optimization folder:
#   ../tutorials/<tutorial>/<optimization>/<tutorial>_<optimization>_data_table.csv
#   ../tutorials/<tutorial>/<optimization>/<tutorial>_<optimization>_data_table.png
#
#==============================================================================================================


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt










#========================================================================================================================================
# PATHS / SETTINGS
#========================================================================================================================================
LIT_COMP_DIR = Path(__file__).resolve().parent.parent
TUTORIALS_DIR = LIT_COMP_DIR / "tutorials"

DEFAULT_METRIC = "l2"
SIG_FIGS = 4

FXD = "FXD"
FREE = "FREE"
VARIANTS = (
    FXD,
    FREE,
)










#========================================================================================================================================
# TABLE SPECIFICATIONS
#========================================================================================================================================
@dataclass(frozen = True)
class ObjectiveRequest:
    """
    One CSV row/column extraction request.

    objective:
        Entry in the source CSV objective column.

    metric:
        Entry in the source CSV metric column, usually l2, rms, max_abs, or mean_abs.

    label:
        Prefix used in the output table headers.

    fxd_column/free_column:
        Optional exact source CSV column names. Leave as None unless a CSV has
        multiple FXD/FREE columns and automatic detection is ambiguous.
    """

    label : str
    objective : str
    metric : str = DEFAULT_METRIC
    fxd_column : str | None = None
    free_column : str | None = None




@dataclass(frozen = True)
class TableSpec:
    """
    One optimization-folder table specification.
    """

    folder : str
    objectives : tuple[ObjectiveRequest, ...]
    title : str | None = None





def obj(
        label,
        objective,
        metric = DEFAULT_METRIC,
        fxd_column = None,
        free_column = None,
    ):
    """
    Build one ObjectiveRequest with compact syntax.
    """

    return ObjectiveRequest(
        label = label,
        objective = objective,
        metric = metric,
        fxd_column = fxd_column,
        free_column = free_column,
    )





def table(
        folder,
        objectives,
        title = None,
    ):
    """
    Build one TableSpec with compact syntax.
    """

    return TableSpec(
        folder = folder,
        objectives = tuple(objectives),
        title = title,
    )










#========================================================================================================================================
# TUTORIAL-SPECIFIC TABLE SPECS
#========================================================================================================================================
def get_basic_qs_table_specs(
        metric = DEFAULT_METRIC,
    ):
    """
    Tables for basic_qs.

    Edit objective/metric strings here if compare.py writes different CSV rows.
    """

    objectives = (
        obj(
            label = "ForceBalance",
            objective = "ForceBalance",
            metric = metric,
        ),
        obj(
            label = "QuasisymmetryBoozer",
            objective = "QuasisymmetryBoozer",
            metric = metric,
        ),
    )

    return (
        table(
            folder = "tripleQS",
            objectives = objectives,
            title = "basic_qs / tripleQS",
        ),
        table(
            folder = "twotermQH",
            objectives = objectives,
            title = "basic_qs / twotermQH",
        ),
    )





def get_adv_qs_table_specs(
        metric = DEFAULT_METRIC,
    ):
    """
    Tables for adv_qs.

    Edit objective/metric strings here if compare.py writes different CSV rows.
    """

    objectives = (
        obj(
            label = "ForceBalance",
            objective = "ForceBalance",
            metric = metric,
        ),
        obj(
            label = "QuasisymmetryBoozer",
            objective = "QuasisymmetryBoozer",
            metric = metric,
        ),
    )

    return (
        table(
            folder = "auglag",
            objectives = objectives,
            title = "adv_qs / auglag",
        ),
        table(
            folder = "multigrid",
            objectives = objectives,
            title = "adv_qs / multigrid",
        ),
    )





def get_balloon_table_specs(
        metric = DEFAULT_METRIC,
    ):
    """
    Tables for balloon.

    Edit objective/metric strings here if compare.py writes different CSV rows.
    """

    return (
        table(
            folder = "balloon",
            title = "balloon / balloon",
            objectives = (
                obj(
                    label = "ForceBalance",
                    objective = "ForceBalance",
                    metric = metric,
                ),
                obj(
                    label = "BallooningStability",
                    objective = "BallooningStability",
                    metric = metric,
                ),
                obj(
                    label = "AspectRatio",
                    objective = "AspectRatio",
                    metric = metric,
                ),
                obj(
                    label = "curvature_k2_rho",
                    objective = "GenericObjective curvature_k2_rho",
                    metric = metric,
                ),
            ),
        ),
    )





def get_neoclassical_table_specs(
        metric = DEFAULT_METRIC,
    ):
    """
    Tables for neoclassical.

    Edit objective/metric strings here if compare.py writes different CSV rows.
    """

    return (
        table(
            folder = "neoclassical",
            title = "neoclassical / neoclassical",
            objectives = (
                obj(
                    label = "ForceBalance",
                    objective = "ForceBalance",
                    metric = metric,
                ),
                obj(
                    label = "EffectiveRipple",
                    objective = "EffectiveRipple",
                    metric = metric,
                ),
            ),
        ),
    )





TUTORIAL_TABLE_SPEC_GETTERS = {
    "basic_qs": get_basic_qs_table_specs,
    "adv_qs": get_adv_qs_table_specs,
    "balloon": get_balloon_table_specs,
    "neoclassical": get_neoclassical_table_specs,
}










#========================================================================================================================================
# PATH HELPERS
#========================================================================================================================================
def normalize_tutorial_name(
        tutorial,
    ):
    """
    Normalize a tutorial argument into a tutorial-folder name.
    """

    return Path(tutorial).stem





def get_tutorial_dir(
        tutorial,
    ):
    """
    Return research/lit_comp/tutorials/<tutorial>.
    """

    tutorial_name = normalize_tutorial_name(
        tutorial = tutorial,
    )

    tutorial_dir = TUTORIALS_DIR / tutorial_name

    if not tutorial_dir.exists():
        raise FileNotFoundError(
            f"Tutorial directory does not exist: {tutorial_dir}"
        )

    return tutorial_dir





def is_numbered_case_dir(
        path,
    ):
    """
    Return True for folders named 001, 002, 003, ...
    """

    path = Path(path)

    return path.is_dir() and path.name.isdigit()





def find_numbered_case_dirs(
        table_dir,
    ):
    """
    Return numbered tolerance-run folders inside one optimization folder.
    """

    table_dir = Path(table_dir)

    return tuple(
        sorted(
            path
            for path in table_dir.iterdir()
            if is_numbered_case_dir(
                path = path,
            )
        )
    )





def get_case_sort_key(
        path,
    ):
    """
    Sort numbered folders by integer value.
    """

    return int(Path(path).name)





def find_comparison_csv(
        case_dir,
    ):
    """
    Find the comparison CSV in one numbered tolerance-run folder.

    The intended state is exactly one CSV per run folder. To avoid accidentally
    re-reading a generated summary table, files ending in _data_table.csv are
    ignored.
    """

    case_dir = Path(case_dir)

    csv_paths = tuple(
        sorted(
            path
            for path in case_dir.glob("*.csv")
            if not path.name.endswith("_data_table.csv")
        )
    )

    if len(csv_paths) == 0:
        raise FileNotFoundError(
            f"No CSV file found in run folder: {case_dir}"
        )

    if len(csv_paths) == 1:
        return csv_paths[0]

    preferred_paths = tuple(
        path
        for path in csv_paths
        if "case_obj" in path.name
    )

    if len(preferred_paths) == 1:
        return preferred_paths[0]

    raise RuntimeError(
        "Expected one comparison CSV in run folder, but found multiple:\n"
        + "\n".join(str(path) for path in csv_paths)
    )










#========================================================================================================================================
# CSV EXTRACTION HELPERS
#========================================================================================================================================
def read_csv_rows(
        csv_path,
    ):
    """
    Read a comparison CSV into a list of dict rows.
    """

    with open(csv_path, "r", newline = "") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())

    return fieldnames, rows





def normalize_text(
        text,
    ):
    """
    Normalize text for robust column and row matching.
    """

    return str(text).strip().lower()





def find_row(
        rows,
        objective,
        metric,
    ):
    """
    Find one objective/metric row in a compare.py output CSV.
    """

    objective_norm = normalize_text(
        text = objective,
    )

    metric_norm = normalize_text(
        text = metric,
    )

    for row in rows:
        row_objective = normalize_text(
            text = row.get("objective", ""),
        )

        row_metric = normalize_text(
            text = row.get("metric", ""),
        )

        if row_objective == objective_norm and row_metric == metric_norm:
            return row

    return None





def get_available_objective_metrics(
        rows,
    ):
    """
    Return available objective/metric pairs for error messages.
    """

    pairs = []

    for row in rows:
        objective = row.get(
            "objective",
            "",
        )

        metric = row.get(
            "metric",
            "",
        )

        pair = f"{objective} / {metric}"

        if pair not in pairs:
            pairs.append(pair)

    return pairs





def find_variant_column(
        fieldnames,
        variant,
        requested_column = None,
    ):
    """
    Find the FXD or FREE source column in a comparison CSV.
    """

    if requested_column is not None:
        if requested_column not in fieldnames:
            raise KeyError(
                f"Requested column '{requested_column}' was not found. Available columns: {fieldnames}"
            )

        return requested_column

    variant_norm = normalize_text(
        text = variant,
    )

    exact_matches = [
        fieldname
        for fieldname in fieldnames
        if normalize_text(
            text = fieldname,
        ) == variant_norm
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    suffix_matches = [
        fieldname
        for fieldname in fieldnames
        if normalize_text(
            text = fieldname,
        ).endswith(f"_{variant_norm}")
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    candidates = exact_matches + suffix_matches

    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous {variant} columns: {candidates}. "
            "Set fxd_column/free_column explicitly in the relevant table spec."
        )

    raise KeyError(
        f"Could not find a {variant} column. Available columns: {fieldnames}"
    )





def format_sig_figs(
        value,
        sig_figs = SIG_FIGS,
    ):
    """
    Format numeric values to at most sig_figs significant figures.
    """

    if value is None:
        return ""

    text = str(value).strip()

    if text == "":
        return ""

    try:
        number = float(text)

    except ValueError:
        return text

    if not math.isfinite(number):
        return text

    if number == 0:
        return "0"

    return f"{number:.{sig_figs}g}"





def extract_value(
        fieldnames,
        rows,
        request,
        variant,
        strict = False,
    ):
    """
    Extract one objective/metric/variant value from one source CSV.
    """

    source_row = find_row(
        rows = rows,
        objective = request.objective,
        metric = request.metric,
    )

    if source_row is None:
        message = (
            f"Missing row objective = {request.objective}, metric = {request.metric}. "
            "Available objective/metric rows:\n"
            + "\n".join(get_available_objective_metrics(rows = rows))
        )

        if strict:
            raise KeyError(message)

        print("WARNING:")
        print(message)
        print("")

        return ""

    if variant == FXD:
        requested_column = request.fxd_column

    else:
        requested_column = request.free_column

    source_column = find_variant_column(
        fieldnames = fieldnames,
        variant = variant,
        requested_column = requested_column,
    )

    return format_sig_figs(
        value = source_row.get(
            source_column,
            "",
        ),
    )










#========================================================================================================================================
# TABLE BUILDING HELPERS
#========================================================================================================================================
def make_output_fieldnames(
        spec,
    ):
    """
    Make output fieldnames with FXD and FREE of each objective adjacent.
    """

    fieldnames = [
        "run",
    ]

    for request in spec.objectives:
        for variant in VARIANTS:
            fieldnames.append(
                f"{request.label} {variant}"
            )

    return fieldnames





def build_table_rows(
        table_dir,
        spec,
        strict = False,
    ):
    """
    Build one summary table from all numbered run folders in an optimization folder.
    """

    case_dirs = find_numbered_case_dirs(
        table_dir = table_dir,
    )

    if len(case_dirs) == 0:
        raise FileNotFoundError(
            f"No numbered run folders found in optimization folder: {table_dir}"
        )

    rows_out = []

    for case_dir in sorted(case_dirs, key = get_case_sort_key):
        csv_path = find_comparison_csv(
            case_dir = case_dir,
        )

        fieldnames, rows = read_csv_rows(
            csv_path = csv_path,
        )

        output_row = {
            "run": case_dir.name,
        }

        for request in spec.objectives:
            for variant in VARIANTS:
                output_column = f"{request.label} {variant}"

                output_row[output_column] = extract_value(
                    fieldnames = fieldnames,
                    rows = rows,
                    request = request,
                    variant = variant,
                    strict = strict,
                )

        rows_out.append(output_row)

    return rows_out





def write_output_csv(
        rows,
        fieldnames,
        save_path,
    ):
    """
    Write one generated data table CSV.
    """

    with open(save_path, "w", newline = "") as file:
        writer = csv.DictWriter(
            file,
            fieldnames = fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return save_path





def write_output_png(
        rows,
        fieldnames,
        title,
        save_path,
    ):
    """
    Write one generated table image.
    """

    num_rows = max(
        len(rows),
        1,
    )

    num_cols = max(
        len(fieldnames),
        1,
    )

    fig_width = max(
        12.0,
        1.65 * num_cols,
    )

    fig_height = max(
        2.8,
        0.42 * (num_rows + 3),
    )

    fig, ax = plt.subplots(
        figsize = (
            fig_width,
            fig_height,
        ),
    )

    ax.axis("off")

    cell_text = [
        [
            row.get(
                fieldname,
                "",
            )
            for fieldname in fieldnames
        ]
        for row in rows
    ]

    table_artist = ax.table(
        cellText = cell_text,
        colLabels = fieldnames,
        loc = "center",
        cellLoc = "center",
    )

    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(
        1.0,
        1.35,
    )

    ax.set_title(
        title,
        pad = 16,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight",
    )

    plt.close(fig)

    return save_path





def build_one_spec_outputs(
        tutorial_name,
        tutorial_dir,
        spec,
        strict = False,
    ):
    """
    Build CSV and PNG outputs for one optimization folder.
    """

    table_dir = tutorial_dir / spec.folder

    if not table_dir.exists():
        raise FileNotFoundError(
            f"Optimization folder does not exist: {table_dir}"
        )

    if not table_dir.is_dir():
        raise NotADirectoryError(
            f"Optimization path is not a folder: {table_dir}"
        )

    fieldnames = make_output_fieldnames(
        spec = spec,
    )

    rows = build_table_rows(
        table_dir = table_dir,
        spec = spec,
        strict = strict,
    )

    output_stem = f"{tutorial_name}_{spec.folder}_data_table"

    csv_path = table_dir / f"{output_stem}.csv"
    png_path = table_dir / f"{output_stem}.png"

    title = spec.title

    if title is None:
        title = f"{tutorial_name} / {spec.folder}"

    write_output_csv(
        rows = rows,
        fieldnames = fieldnames,
        save_path = csv_path,
    )

    write_output_png(
        rows = rows,
        fieldnames = fieldnames,
        title = title,
        save_path = png_path,
    )

    return csv_path, png_path





def get_table_specs(
        tutorial_name,
        metric = DEFAULT_METRIC,
        selected_table = None,
    ):
    """
    Return table specs for one tutorial.
    """

    getter = TUTORIAL_TABLE_SPEC_GETTERS.get(
        tutorial_name,
        None,
    )

    if getter is None:
        raise ValueError(
            f"No table-spec function found for tutorial = {tutorial_name}. "
            "Add one to TUTORIAL_TABLE_SPEC_GETTERS."
        )

    specs = tuple(
        getter(
            metric = metric,
        )
    )

    if selected_table is None:
        return specs

    selected_specs = tuple(
        spec
        for spec in specs
        if spec.folder == selected_table
    )

    if len(selected_specs) == 0:
        known_tables = ", ".join(
            spec.folder
            for spec in specs
        )

        raise ValueError(
            f"Table '{selected_table}' is not registered for tutorial '{tutorial_name}'. "
            f"Known tables: {known_tables}"
        )

    return selected_specs










#========================================================================================================================================
# PUBLIC ENTRY POINT
#========================================================================================================================================
def plot_data_tutorial(
        tutorial,
        metric = DEFAULT_METRIC,
        selected_table = None,
        strict = False,
    ):
    """
    Build all configured data tables for one tutorial.
    """

    tutorial_name = normalize_tutorial_name(
        tutorial = tutorial,
    )

    tutorial_dir = get_tutorial_dir(
        tutorial = tutorial_name,
    )

    specs = get_table_specs(
        tutorial_name = tutorial_name,
        metric = metric,
        selected_table = selected_table,
    )

    saved_paths = []

    for spec in specs:
        csv_path, png_path = build_one_spec_outputs(
            tutorial_name = tutorial_name,
            tutorial_dir = tutorial_dir,
            spec = spec,
            strict = strict,
        )

        saved_paths.append(csv_path)
        saved_paths.append(png_path)

    return saved_paths










#========================================================================================================================================
# CLI
#========================================================================================================================================
def parse_args():
    """
    Parse command-line arguments.
    """

    parser = ArgumentParser(
        description = "Build compact DESC tutorial data tables from compare.py CSV files.",
    )

    parser.add_argument(
        "--tutorial",
        required = True,
        help = "Tutorial name, e.g. basic_qs, adv_qs, balloon, or neoclassical.",
    )

    parser.add_argument(
        "--table",
        default = None,
        help = "Optional optimization-folder table name, e.g. auglag or tripleQS.",
    )

    parser.add_argument(
        "--metric",
        default = DEFAULT_METRIC,
        help = "CSV metric row to extract for each objective. Default: l2.",
    )

    parser.add_argument(
        "--strict",
        action = "store_true",
        help = "Raise an error when a requested objective/metric row is missing.",
    )

    return parser.parse_args()





def main():
    """
    CLI entry point.
    """

    args = parse_args()

    print("")
    print("================================================================================================================")
    print(f"Building data tables for tutorial = {args.tutorial}")

    if args.table is not None:
        print(f"Requested table: {args.table}")

    print(f"Metric: {args.metric}")
    print("================================================================================================================")
    print("")

    saved_paths = plot_data_tutorial(
        tutorial = args.tutorial,
        metric = args.metric,
        selected_table = args.table,
        strict = args.strict,
    )

    print("")
    print("Saved data tables:")
    print("")

    for path in saved_paths:
        print(path)

    print("")





if __name__ == "__main__":
    main()
