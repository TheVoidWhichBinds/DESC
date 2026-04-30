# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning DESC recreation files with FLO/FNO variants.
#
#==============================================================================================================

import os

os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if "JAX_PLATFORM_NAME" in os.environ:
    del os.environ["JAX_PLATFORM_NAME"]

from desc import set_device
set_device("gpu")

import argparse

try:
    from .helper import run_variants_for_file
except ImportError:
    from helper import run_variants_for_file








#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description = "Run one DESC recreation file with FLO and FNO variants.",
    )

    parser.add_argument(
        "--file",
        type = str,
        required = True,
        help = "Path to the recreation .py file to rerun.",
    )

    parser.add_argument(
        "--variants",
        nargs = "+",
        default = ["FLO", "FNO"],
        choices = ["FLO", "FNO"],
        help = "Variants to run.",
    )

    return parser.parse_args()









#==============================================================================================================
# Main
#==============================================================================================================

def main():
    """
    Command-line entry point.
    """

    args = parse_args()

    outputs = run_variants_for_file(
        source_file = args.file,
        variants = tuple(args.variants),
    )

    print("")
    print("Completed variant runs:")
    print("================================================================================================================")

    for variant, path in outputs.items():
        print(f"{variant}: {path}")










if __name__ == "__main__":
    main()
