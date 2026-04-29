# helper.py

from itertools import product
import os
import numpy as np
import pickle
import torch
import io
import contextlib
import warnings
from desc.continuation import solve_continuation_automatic

base_dir = os.path.dirname(os.path.abspath(__file__))





#=========================================================================================
def run_continuation_check(eq, failure_fragment="WARNING: Automatic continuation failed"):
    """
    Runs DESC automatic continuation and returns only serializable status data.
    """
    result = {
        "enabled": True,
        "success": False,
        "broke": False,
        "warning_detected": False,
        "message": "",
        "error": None,
    }

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            solved_eq = solve_continuation_automatic(
                eq,
                verbose = 2,
            )

        output_text = "\n".join(
            [
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue(),
                "\n".join(str(w.message) for w in caught_warnings),
            ]
        )

        result["message"] = output_text.strip()
        result["warning_detected"] = failure_fragment in output_text
        result["success"] = solved_eq is not None and not result["warning_detected"]
        result["broke"] = not result["success"]

    except Exception as e:
        result["success"] = False
        result["broke"] = True
        result["error"] = repr(e)
        result["message"] = "\n".join(
            [stdout_buffer.getvalue(), stderr_buffer.getvalue()]
        ).strip()

    return result
#=========================================================================================






#============== MAIN HELPERS ===================================================================================
#==================
def cond_generator(
    resolution: tuple,
    NFP: int,
    modes_R: list,
    modes_Z: list,
    R_lmn_options: list,
    Z_lmn_options: list,
):
    """
    Generates initial-condition dictionaries for all continuous
    feature-vector combinations, while keeping the discrete
    structure fixed across the whole dataset.
    """
    for R_lmn, Z_lmn in product(
        R_lmn_options,
        Z_lmn_options,
    ):
        yield {
            "resolution": resolution,
            "NFP": NFP,
            "modes_R": modes_R,
            "modes_Z": modes_Z,
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
        }
#==================






#==================
def feature_generator(
    config: dict,
):
    """
    Generates continuous feature options from config-controlled candidate pools.

    The discrete structure is fixed for the entire dataset:
        - resolution
        - NFP
        - modes_R
        - modes_Z

    Only the continuous amplitudes are varied:
        - R_lmn
        - Z_lmn
    """
    #------------------------
    # Fixed quantities:
    resolution = config["resolution"]
    NFP = config["NFP"]
    modes_R = config["modes_R"]
    modes_Z = config["modes_Z"]

    counts = config["counts"]
    ranges = config["ranges"]
    #------------------------

    #-------------------
    # R-FK coefficients:
    modeR00_vals = np.linspace(
        ranges["modeR00"][0],
        ranges["modeR00"][1],
        counts["N_R0"],
    )
    modeR10_vals = np.linspace(
        ranges["modeR10"][0],
        ranges["modeR10"][1],
        counts["N_R10"],
    )
    modeR20_vals = np.linspace(
        ranges["modeR20"][0],
        ranges["modeR20"][1],
        counts["N_R20"],
    )
    modeR11_vals = np.linspace(
        ranges["modeR11"][0],
        ranges["modeR11"][1],
        counts["N_R11"],
    )

    R_lmn_pool = [
        [
            float(modeR00),
            float(modeR10),
            float(modeR11),
            float(modeR20),
        ]
        for modeR00 in modeR00_vals
        for modeR10 in modeR10_vals
        for modeR11 in modeR11_vals
        for modeR20 in modeR20_vals
    ]
    #-------------------

    #-------------------
    # Z-FK coefficients:
    modeZ10_vals = np.linspace(
        ranges["modeZ10"][0],
        ranges["modeZ10"][1],
        counts["N_Z10"],
    )
    modeZ20_vals = np.linspace(
        ranges["modeZ20"][0],
        ranges["modeZ20"][1],
        counts["N_Z20"],
    )
    modeZ11_vals = np.linspace(
        ranges["modeZ11"][0],
        ranges["modeZ11"][1],
        counts["N_Z11"],
    )

    Z_lmn_pool = [
        [
            float(modeZ10),
            float(modeZ11),
            float(modeZ20),
        ]
        for modeZ10 in modeZ10_vals
        for modeZ11 in modeZ11_vals
        for modeZ20 in modeZ20_vals
    ]
    #------------------------------

    return {
        "resolution": resolution,
        "NFP": NFP,
        "modes_R": modes_R,
        "modes_Z": modes_Z,
        "R_lmn_options": R_lmn_pool,
        "Z_lmn_options": Z_lmn_pool,
    }
#==================
#==============================================================================================================











#============== SAVE HELPERS ===================================================================================
#==================
def get_nfp_save_dir(
    NFP: int,
):
    """
    Returns the save directory for the given NFP, creating it if needed.
    """
    save_dir = os.path.join(base_dir, f"NFP_{NFP}")
    os.makedirs(save_dir, exist_ok = True)
    return save_dir
#==================






#==================
def data_saver(
    data,
    NFP,
    filename = "dataset.pkl",
):
    """
    Save raw Python dataset to disk with pickle.
    """
    save_dir = get_nfp_save_dir(NFP = NFP)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return save_path
#==================






#==================
def build_torch_dataset(
    data,
    target_key: str,
):
    """
    Build PyTorch-ready tensors for binary classification.

    X contains continuous numeric features only.
    y contains one binary target column.
    """
    X_rows = []
    y_rows = []

    for point in data:
        feat = point["features"]
        lab = point["labels"]

        x = []
        x.extend([float(v) for v in feat["R_lmn"]])
        x.extend([float(v) for v in feat["Z_lmn"]])

        y = [float(lab[target_key])]

        X_rows.append(x)
        y_rows.append(y)

    X = torch.tensor(X_rows, dtype = torch.float32)
    y = torch.tensor(y_rows, dtype = torch.float32)

    return {
        "X": X,
        "y": y,
        "target_key": target_key,
    }
#==================






#==================
def torch_data_saver(
    torch_data,
    NFP,
    filename = "torch_dataset.pt",
):
    """
    Save PyTorch-ready dataset to disk.
    """
    save_dir = get_nfp_save_dir(NFP = NFP)
    save_path = os.path.join(save_dir, filename)

    torch.save(torch_data, save_path)

    return save_path
#==================
#==============================================================================================================