# config.py











#============== FEATURE / SWEEP CONFIG =========================================================================
FEATURE_CONFIG = {
    "resolution": (16, 16, 16),
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
        "N_R0": 2,
        "N_R10": 2,
        "N_Z10": 2,
        "N_R11": 2,
        "N_Z11": 2,
        "N_R20": 2,
        "N_Z20": 2,
    },
    "ranges": {
        "modeR00": (4.8, 6.2), # major radius

        "modeR10": (-0.52, -0.38), # radial minor size
        "modeZ10": (0.38, 0.52), # vertical minor size

        "modeR11": (-0.045, 0.045), # mild helical variation
        "modeZ11": (-0.045, 0.045), # mild helical variation

        "modeR20": (-0.035, 0.035), # weak bean/triangular shaping
        "modeZ20": (-0.005, 0.005), # very weak vertical second harmonic
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