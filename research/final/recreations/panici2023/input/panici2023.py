# panici2023.py
"""Recreate Panici 2023 W7X outputs using the DESC input parser."""


#========================================================================================================================================
# IMPORTS
#========================================================================================================================================
from pathlib import Path
import subprocess
import sys










#========================================================================================================================================
# PATHS
#========================================================================================================================================
INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)










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
# RUN
#========================================================================================================================================
for case in CASES:
    input_file = INPUT_DIR / case["input_file"]
    output_file = OUTPUT_DIR / case["output_file"]

    command = [
        sys.executable,
        "-m",
        "desc",
        str(input_file),
        "-o",
        str(output_file),
    ]

    print("")
    print("================================================================")
    print("Running {} recreation".format(case["label"]))
    print("================================================================")
    print("Input:")
    print(input_file)
    print("")
    print("Output:")
    print(output_file)
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