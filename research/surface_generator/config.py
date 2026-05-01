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
        (2, 1),
    ],
    "modes_Z": [
        (-1, 0),
        (-1, 1),
        (-2, 0),
        (-2, 1),
    ],
    "counts": {
        "N_R0": 1,
        "N_R10": 1,
        "N_Z10": 1,
        "N_R11": 2,
        "N_Z11": 2,
        "N_R20": 2,
        "N_Z20": 1,
        "N_R21": 1,
        "N_Z21": 1,
    },
    "ranges": {
        "modeR00": (6.0, 6.2),

        # main axisymmetric cross-section
        # nearly fixed elliptical cross-section
        "modeR10": (-0.5, -0.51),
        "modeZ10": (0.49, 0.50),

        # primary helical rotation / weak 3D deformation
        "modeR11": (-0.025, 0.025),
        "modeZ11": (-0.025, 0.025),

        # very small axisymmetric triangularity
        "modeR20": (-0.004, 0.0),
        "modeZ20": (-0.004, 0.0),

        # weak helical second harmonic
        "modeR21": (-0.025, 0.025),
        "modeZ21": (-0.025, 0.025),
    },
}
#==============================================================================================================





#============== DATA CONFIG ====================================================================================
DATA_CONFIG = {
    "dataset_prefix": "dataset",
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