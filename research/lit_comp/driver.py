# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC paper recreation file.
#
# If --free = False, the file runs exactly as-is.
# If --free = True, the file runs with the FREE pressure-profile patch.
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
    from .helper import run_file
except ImportError:
    from helper import run_file










#==============================================================================================================
# Command-Line Helpers
#==============================================================================================================

def str_to_bool(
        value,
    ):
    """
    Convert command-line True/False strings to booleans.
    """

    if isinstance(value, bool):
        return value

    value = value.strip().lower()

    if value in ("true", "t", "yes", "y", "1"):
        return True

    if value in ("false", "f", "no", "n", "0"):
        return False

    raise argparse.ArgumentTypeError(
        "Expected True or False."
    )










#==============================================================================================================
# Command-Line Interface
#==============================================================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description = "Run one DESC paper recreation file as-is or with the FREE variant.",
    )

    parser.add_argument(
        "--file",
        type = str,
        required = True,
        help = "Path to the paper recreation .py file to rerun.",
    )

    parser.add_argument(
        "--free",
        type = str_to_bool,
        required = True,
        help = "If True, run with FREE. If False, run the source file as-is.",
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

    output = run_file(
        source_file = args.file,
        free = args.free,
    )

    print("")
    print("Completed run:")
    print("================================================================================================================")

    if output is None:
        print("free = False: source file ran as-is.")
    else:
        print(f"free = True: {output}")










if __name__ == "__main__":
    main()