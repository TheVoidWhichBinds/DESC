# panici2023.py
"""Recreate Panici 2023 W7X outputs using the DESC input parser."""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from pathlib import Path
import re
import subprocess
import sys
import tempfile










#========================================================================================================================================
# PATHS
#========================================================================================================================================
INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)










#========================================================================================================================================
# CONTINUATION PATCH SETTINGS
#========================================================================================================================================
BDRY_RATIO = "0x4, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1"
PRES_RATIO = "0x2, 0.5, 1x15"
PERT_ORDER = "2"

L10_PROFILE_LINE = "l:  10  p =   0.00000000E+00    i =   0.00000000E+00"










#========================================================================================================================================
# CASES
#========================================================================================================================================
CASES = [
    {
        "label": "Panici 2023 M8/N8",
        "input_file": "W7X_M8_N8_ansi_cpu1_compute_branch",
        "output_file": "W7X_M8_N8_ansi_cpu1_compute_branch_CHECK.h5",
    },
    {
        "label": "Panici 2023 M10/N10",
        "input_file": "W7X_M10_N10_ansi_cpu1_compute_branch",
        "output_file": "W7X_M10_N10_ansi_cpu1_compute_branch_CHECK.h5",
    },
    {
        "label": "Panici 2023 M12/N12",
        "input_file": "W7X_M12_N12_ansi_cpu1_compute_branch",
        "output_file": "W7X_M12_N12_ansi_cpu1_compute_branch_CHECK.h5",
    },
    {
        "label": "Panici 2023 M14/N14",
        "input_file": "W7X_M14_N14_ansi_cpu1_compute_branch",
        "output_file": "W7X_M14_N14_ansi_cpu1_compute_branch_CHECK.h5",
    },
    {
        "label": "Panici 2023 M16/N16",
        "input_file": "W7X_M16_N16_ansi_cpu1_compute_branch",
        "output_file": "W7X_M16_N16_ansi_cpu1_compute_branch_CHECK.h5",
    },
    {
        "label": "Panici 2023 M16/N16 f2",
        "input_file": "W7X_M16_N16_ansi_cpu1_f2_compute_branch",
        "output_file": "W7X_M16_N16_ansi_cpu1_f2_compute_branch_CHECK.h5",
    },
]










#========================================================================================================================================
# PATCH HELPERS
#========================================================================================================================================
def patch_continuation_parameters(text):
    """Patch continuation settings to use smaller boundary steps."""

    text = re.sub(
        r"^bdry_ratio\s*=.*$",
        "bdry_ratio = {}".format(BDRY_RATIO),
        text,
        flags = re.MULTILINE,
    )

    text = re.sub(
        r"^pres_ratio\s*=.*$",
        "pres_ratio = {}".format(PRES_RATIO),
        text,
        flags = re.MULTILINE,
    )

    text = re.sub(
        r"^pert_order\s*=.*$",
        "pert_order = {}".format(PERT_ORDER),
        text,
        flags = re.MULTILINE,
    )

    return text










def patch_l10_profile_coefficient(text):
    """Add an explicit zero l=10 pressure/iota profile coefficient if missing."""

    profile_block = re.search(
        r"(# pressure and rotational transform profiles\n.*?)(\n# magnetic axis initial guess)",
        text,
        flags = re.DOTALL,
    )

    if profile_block is None:
        raise RuntimeError("Could not find pressure/iota profile block.")

    block = profile_block.group(1)

    if re.search(r"^l:\s*10\s+p\s*=", block, flags = re.MULTILINE):
        return text

    old = profile_block.group(1) + profile_block.group(2)

    new = (
        profile_block.group(1).rstrip()
        + "\n"
        + L10_PROFILE_LINE
        + profile_block.group(2)
    )

    return text.replace(old, new)










def get_patched_input_text(input_file):
    """Return patched input text without modifying the original input file."""

    text = input_file.read_text()

    text = patch_l10_profile_coefficient(text)
    text = patch_continuation_parameters(text)

    return text










#========================================================================================================================================
# RUN
#========================================================================================================================================
for case in CASES:
    input_file = INPUT_DIR / case["input_file"]
    output_file = OUTPUT_DIR / case["output_file"]

    patched_text = get_patched_input_text(input_file)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_input_file = Path(temporary_directory) / case["input_file"]
        temporary_input_file.write_text(patched_text)

        command = [
            sys.executable,
            "-m",
            "desc",
            str(temporary_input_file),
            "-o",
            str(output_file),
        ]

        print("")
        print("================================================================")
        print("Running {} recreation".format(case["label"]))
        print("================================================================")
        print("Original input:")
        print(input_file)
        print("")
        print("Temporary patched input:")
        print(temporary_input_file)
        print("")
        print("Output:")
        print(output_file)
        print("")
        print("Continuation settings:")
        print("bdry_ratio = {}".format(BDRY_RATIO))
        print("pres_ratio = {}".format(PRES_RATIO))
        print("pert_order = {}".format(PERT_ORDER))
        print("")
        print("Command:")
        print(" ".join(command))

        subprocess.run(command, check = True)

    print("")
    print("Saved recreated output:")
    print(output_file)










#========================================================================================================================================
# SAVES
#========================================================================================================================================
print("")
print("================================================================")
print("Finished Panici 2023 recreations")
print("================================================================")

for case in CASES:
    print(OUTPUT_DIR / case["output_file"])