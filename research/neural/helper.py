from itertools import product
import os
import numpy as np
import pickle
import torch
base_dir = os.path.dirname(os.path.abspath(__file__))









#==================== INITIALIZER.PY =====================================================================================
#===================
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
#==============================





#========================
def feature_generator(N):
    """
    Generates continuous feature options from hard-coded candidate pools.

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
    # Fixed quantities
    resolution = (16, 16, 16)
    NFP = 4

    modes_R = [
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
    ]

    modes_Z = [
        (1, 0),
        (2, 0),
        (3, 0),
        (1, 1),
    ]
    #----------

    #-------------------
    # R-FK coefficients:
    modeR00_vals = np.linspace(4.5, 4.7, N)
    modeR10_vals = np.linspace(-0.6, 0.8, N)
    modeR20_vals = np.linspace(0.10, 0.20, N)
    modeR11_vals = np.linspace(-0.08, -0.01, N)
    R_lmn_pool = [
        [float(modeR00), float(modeR10), float(modeR20), float(modeR11)]
        for modeR00 in modeR00_vals
        for modeR10 in modeR10_vals
        for modeR20 in modeR20_vals
        for modeR11 in modeR11_vals
    ]
    #-------------------------------

    #-------------------
    # Z-FK coefficients:
    modeZ10_vals = np.linspace(0.9, 1.10, N)
    modeZ20_vals = np.linspace(0.05, 0.10, N)
    modeZ30_vals = np.linspace(-0.03, -0.01, N)
    modeZ11_vals = np.linspace(-0.03, -0.01, N)
    Z_lmn_pool = [
        [float(modeZ10), float(modeZ20), float(modeZ30), float(modeZ11)]
        for modeZ10 in modeZ10_vals
        for modeZ20 in modeZ20_vals
        for modeZ30 in modeZ30_vals
        for modeZ11 in modeZ11_vals
    ]
    #------------------------------

    #-------------------------------
    # Pressure coefficients (octic):
    p0_vals = np.linspace(1e4, 1e7, N)
    p8_vals = np.linspace(-1.3, 1.8, N)
    p_l_pool = [
        [
            float(p0),
            float(p0 * (-2 + (-3.45 * p8) + 2 * p8)),
            float(p0 * (1 - 2 * (-3.45 * p8) - 3 * p8)),
            float(p0 * (-3.45 * p8)),
            float(p0 * p8),
        ]
        for p0 in p0_vals
        for p8 in p8_vals
    ]
    #--------------------

    #-------------------
    # Iota coefficients:
    i0_vals = np.linspace(0.25, 1.33, N)
    i2_vals = np.linspace(-0.23, 0.23, N)
    i_l_pool = [
        [float(i0), float(i2)]
        for i0 in i0_vals
        for i2 in i2_vals
    ]
    #--------------------

    #-----------
    # Psi range:
    Psi_pool = [float(psi) for psi in np.linspace(0.9, 1.0, N)]
    #----------------------------------------------------------


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
#===============================




#==============
def data_saver(
    data,
    filename = "dataset.pkl",
):
    """
    Save raw nested Python dataset to disk with pickle.
    """
    save_path = os.path.join(base_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return save_path
#===================




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
        x.extend([float(v) for v in feat["resolution"]])
        x.append(float(feat["NFP"]))
        x.extend([float(v) for v in feat["R_lmn"]])
        x.extend([float(v) for v in feat["Z_lmn"]])
        x.extend([float(v) for v in feat["p_l"]])
        x.extend([float(v) for v in feat["i_l"]])
        x.append(float(feat["Psi"]))

        y = []
        for key in label_keys:
            val = lab[key]
            if isinstance(val, (bool, np.bool_)):
                y.append(float(val))
            else:
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
#================================




#====================
def torch_data_saver(
    torch_data,
    filename = "torch_dataset.pt",
):
    """
    Save PyTorch-ready dataset to disk.
    """
    save_path = os.path.join(base_dir, filename)
    torch.save(torch_data, save_path)
    return save_path
#===================
#=============================================================================================================================