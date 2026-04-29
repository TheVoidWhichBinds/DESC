# config.py











#============== FEATURE / SWEEP CONFIG =========================================================================
FEATURE_CONFIG = {
    "resolution": (14, 14, 14),
    "NFP": 4,
    "modes_R": [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
    ],
    "modes_Z": [
        (-1, 0),
        (-1, 1),
        (-2, 0),
    ],
    "counts": {
        "N_R0": 1,
        "N_R10": 1,
        "N_Z10": 1,
        "N_R11": 1,
        "N_Z11": 1,
        "N_R20": 2,
        "N_Z20": 2,
    },
    "ranges": {
        "modeR00": (4.8, 6.2),  # major radius

        "modeR10": (-0.50, -0.42),  # radial minor size
        "modeZ10": (0.42, 0.50),    # vertical minor size, less elongated on average

        "modeR11": (-0.025, 0.025), # mild helical radial variation
        "modeZ11": (-0.025, 0.025), # actual helical vertical variation

        "modeR20": (-0.085, 0.0), # very weak bean/triangular shaping
        "modeZ20": (-0.028, 0.0), # weak vertical second harmonic
    },
}
#==============================================================================================================





#============== DATA CONFIG ====================================================================================
DATA_CONFIG = {
    "target_key": "is_nested",
    "dataset_filename": "dataset.pkl",
    "torch_dataset_filename": "torch_dataset.pt",
}
#==============================================================================================================





#============== RUN CONFIG =====================================================================================
RUN_CONFIG = {
    "use_parallel": False,
    "nprocs": None,
    "chunksize": 1,
    "continuation_check": False,
    "continuation_failure_fragment": "WARNING: Automatic continuation failed",
}
#==============================================================================================================