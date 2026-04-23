# helper.py

from itertools import product
import os
import numpy as np
import pickle
import torch


base_dir = os.path.dirname(os.path.abspath(__file__))











#============== FILE / SAVE DEFAULTS ===========================================================================
VALID_EQ_DIRNAME = "valid_eq"
VALID_EQ_PREFIX = "eq_"
VALID_EQ_DIGITS = 3

DATASET_FILENAME = "dataset.pkl"
TORCH_DATASET_FILENAME = "torch_dataset.pt"
#==============================================================================================================











#============== MAIN HELPERS ===================================================================================
#==================
def cond_generator(
    resolution: tuple,
    NFP: int,
    modes_R: list,
    modes_Z: list,
    R_lmn_options: list,
    Z_lmn_options: list,
    p_l_options: list,
    i_l_options: list,
    Psi_options: list,
    weights: list,
):
    """
    Generates initial-condition dictionaries for all continuous
    feature-vector combinations, while keeping the discrete
    structure fixed across the whole dataset.
    """
    for R_lmn, Z_lmn, p_l, i_l, Psi in product(
        R_lmn_options,
        Z_lmn_options,
        p_l_options,
        i_l_options,
        Psi_options,
    ):
        yield {
            "resolution": resolution,
            "NFP": NFP,
            "modes_R": modes_R,
            "modes_Z": modes_Z,
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "p_l": p_l,
            "i_l": i_l,
            "Psi": float(Psi),
            "weights": weights,
        }
#================================





#==================================
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
        - p_l
        - i_l
        - Psi
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
    modeR21_vals = np.linspace(
        ranges["modeR21"][0],
        ranges["modeR21"][1],
        counts["N_R21"],
    )

    R_lmn_pool = [
        [
            float(modeR00),
            float(modeR10),
            float(modeR20),
            float(modeR11),
            float(modeR21),
        ]
        for modeR00 in modeR00_vals
        for modeR10 in modeR10_vals
        for modeR20 in modeR20_vals
        for modeR11 in modeR11_vals
        for modeR21 in modeR21_vals
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
    modeZ21_vals = np.linspace(
        ranges["modeZ21"][0],
        ranges["modeZ21"][1],
        counts["N_Z21"],
    )

    Z_lmn_pool = [
        [
            float(modeZ10),
            float(modeZ20),
            float(modeZ11),
            float(modeZ21),
        ]
        for modeZ10 in modeZ10_vals
        for modeZ20 in modeZ20_vals
        for modeZ11 in modeZ11_vals
        for modeZ21 in modeZ21_vals
    ]
    #-------------------

    #-------------------------
    # Pressure coefficients:
    p0_vals = np.linspace(
        ranges["p0"][0],
        ranges["p0"][1],
        counts["N_p0"],
    )

    p_l_pool = [
        [
            float(p0),
            float(p0 * -2.0),
            float(p0),
        ]
        for p0 in p0_vals
    ]
    #-------------------------

    #-------------------
    # Iota coefficients:
    i0_vals = np.linspace(
        ranges["i0"][0],
        ranges["i0"][1],
        counts["N_i0"],
    )
    i2_vals = np.linspace(
        ranges["i2"][0],
        ranges["i2"][1],
        counts["N_i2"],
    )

    i_l_pool = [
        [float(i0), float(i2)]
        for i0 in i0_vals
        for i2 in i2_vals
    ]
    #-------------------

    #-----------
    # Psi range:
    Psi_pool = [
        float(psi)
        for psi in np.linspace(
            ranges["Psi"][0],
            ranges["Psi"][1],
            counts["N_psi"],
        )
    ]
    #-----------

    return {
        "resolution": resolution,
        "NFP": NFP,
        "modes_R": modes_R,
        "modes_Z": modes_Z,
        "R_lmn_options": R_lmn_pool,
        "Z_lmn_options": Z_lmn_pool,
        "p_l_options": p_l_pool,
        "i_l_options": i_l_pool,
        "Psi_options": Psi_pool,
    }
#==================================
#==============================================================================================================











#============== SAVE HELPERS ===================================================================================
#=================================
def get_nfp_save_dir(
    NFP: int,
):
    """
    Returns the save directory for the given NFP, creating it if needed.
    """
    save_dir = os.path.join(base_dir, f"NFP_{NFP}")
    os.makedirs(save_dir, exist_ok = True)
    return save_dir
#=================================





#=====================================
def get_valid_eq_dir():
    """
    Returns the directory used to store valid nested equilibria.
    """
    save_dir = os.path.join(base_dir, VALID_EQ_DIRNAME)
    os.makedirs(save_dir, exist_ok = True)
    return save_dir
#=====================================





#==========================
def save_valid_equilibrium(
    eq,
):
    """
    Save a valid nested DESC equilibrium into ./valid_eq.
    """
    save_dir = get_valid_eq_dir()

    existing = [
        name for name in os.listdir(save_dir)
        if name.startswith(VALID_EQ_PREFIX) and name.endswith(".h5")
    ]
    next_idx = len(existing) + 1

    filename = f"{VALID_EQ_PREFIX}{next_idx:0{VALID_EQ_DIGITS}d}.h5"
    save_path = os.path.join(save_dir, filename)
    eq.save(save_path)

    return save_path
#===================





#=========================
def data_saver(
    data,
    NFP,
    filename = DATASET_FILENAME,
):
    """
    Save raw nested Python dataset to disk with pickle.
    """
    save_dir = get_nfp_save_dir(NFP = NFP)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return save_path
#=========================





#=======================
def build_torch_dataset(
    data,
    label_keys,
):
    """
    Build PyTorch-ready tensors from nested dataset.

    X contains continuous numeric features only.
    y contains requested labels in the order of label_keys.
    """
    X_rows = []
    y_rows = []

    for point in data:
        feat = point["features"]
        lab = point["labels"]

        x = []
        x.extend([float(v) for v in feat["R_lmn"]])
        x.extend([float(v) for v in feat["Z_lmn"]])
        x.extend([float(v) for v in feat["p_l"]])
        x.extend([float(v) for v in feat["i_l"]])
        x.append(float(feat["Psi"]))

        y = []
        for key in label_keys:
            val = lab[key]
            y.append(float(val))

        X_rows.append(x)
        y_rows.append(y)

    X = torch.tensor(X_rows, dtype = torch.float32)
    y = torch.tensor(y_rows, dtype = torch.float32)

    return {
        "X": X,
        "y": y,
        "label_keys": label_keys,
    }
#=======================





#==============================
def torch_data_saver(
    torch_data,
    NFP,
    filename = TORCH_DATASET_FILENAME,
):
    """
    Save PyTorch-ready dataset to disk.
    """
    save_dir = get_nfp_save_dir(NFP = NFP)
    save_path = os.path.join(save_dir, filename)

    torch.save(torch_data, save_path)

    return save_path
#==============================
#==============================================================================================================