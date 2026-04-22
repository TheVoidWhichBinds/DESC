# helper.py

from itertools import product
import os
import numpy as np
import pickle
import torch


base_dir = os.path.dirname(os.path.abspath(__file__))











#============== MAIN HELPERS  ===========================================================================
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
    NFP,
    N_R0,
    N_Rs,
    N_Z,
    N_p0,
    N_i0,
    N_i2,
    N_psi,
):
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

    For NFP = 4, the R/Z pools below are chosen to stay in a mild-shaping,
    low-curvature regime, ranging from nearly circular cross-sections to
    slightly banana-like cross-sections.
    """
    #------------------------
    # Fixed quantities
    resolution = (16, 16, 16)

    modes_R = [
        (0, 0),
        (1, 0),
        (1, 1),
    ]

    modes_Z = [
        (-1, 0),
        (-1, 1)
    ]

    modeR00_vals = np.linspace(3.75, 10, N_R0)
    modeR10_vals = np.linspace(-0.55, -0.30, N_Rs)
    modeR11_vals = np.linspace(-0.12, -0.04, N_Rs)
    R_lmn_pool = [
        [float(modeR00), float(modeR10), float(modeR11)]
        for modeR00 in modeR00_vals
        for modeR10 in modeR10_vals
        for modeR11 in modeR11_vals
    ]

    modeZ10_vals = np.linspace(0.55, 0.30, N_Z)
    modeZ11_vals = np.linspace(-0.12, -0.04, N_Z)
    Z_lmn_pool = [
        [float(modeZ10), float(modeZ11)]
        for modeZ10 in modeZ10_vals
        for modeZ11 in modeZ11_vals
    ]
    #-----------------------------

    #-------------------------------
    # Pressure coefficients (octic):
    p0_vals = np.linspace(1E4, 1e7, N_p0)
    p_l_pool = [
        [
            float(p0),
            float(p0 * -2),
            float(p0),
        ]
        for p0 in p0_vals
    ]
    #-------------------------------

    #-------------------
    # Iota coefficients:
    i0_vals = np.linspace(0.25, 1.33, N_i0)
    i2_vals = np.linspace(-0.23, 0.23, N_i2)
    i_l_pool = [
        [float(i0), float(i2)]
        for i0 in i0_vals
        for i2 in i2_vals
    ]
    #-------------------

    #-----------
    # Psi range:
    Psi_pool = [float(psi) for psi in np.linspace(1.0, 1.0, N_psi)]
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
#==========================================================================================================================














#=================== HENCHMEN HELPERS ============================================================================
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
    save_dir = os.path.join(base_dir, "valid_eq")
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
        if name.startswith("eq_") and name.endswith(".h5")
    ]
    next_idx = len(existing) + 1

    save_path = os.path.join(save_dir, f"eq_{next_idx:06d}.h5")
    eq.save(save_path)

    return save_path
#===================





#=========================
def data_saver(
    data,
    NFP,
    filename = "dataset.pkl",
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
    filename = "torch_dataset.pt",
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