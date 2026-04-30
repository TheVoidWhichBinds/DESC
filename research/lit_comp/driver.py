# driver.py
#==============================================================================================================
#
# Top-level CLI for rerunning one DESC recreation file.
#
# If --variant = False, the file runs exactly as-is.
# If --variant = True, the file runs with the FNO objective patch.
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
        description = "Run one DESC recreation file as-is or with the FNO variant.",
    )

    parser.add_argument(
        "--file",
        type = str,
        required = True,
        help = "Path to the recreation .py file to rerun.",
    )

    parser.add_argument(
        "--variant",
        type = str_to_bool,
        required = True,
        help = "If True, run with FNO. If False, run the source file as-is.",
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
        variant = args.variant,
    )

    print("")
    print("Completed run:")
    print("================================================================================================================")

    if output is None:
        print("variant = False: source file ran as-is.")
    else:
        print(f"variant = True: {output}")










if __name__ == "__main__":
    main()