# initializer.py

from multiprocessing import Pool, cpu_count
import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile

from .config import FEATURE_CONFIG, DATA_CONFIG, RUN_CONFIG
from .helper import (
    cond_generator,
    feature_generator,
    data_saver,
    build_torch_dataset,
    torch_data_saver,
    run_continuation_check,
)

import os
import matplotlib.pyplot as plt
from desc.plotting import plot_comparison








#============== SINGLE-CONDITION EVALUATION ====================================================================
#==================
def evaluate_condition(
    cond,
):
    """
    Build one equilibrium from one condition dictionary and assign
    binary-classification labels.

    Returns one ML-style data point with:
        data_point["features"][...]
        data_point["labels"][...]
        data_point["meta"][...]
    """

    #-----------------------
    # Unpack feature vector:
    L, M, N = cond["resolution"]
    NFP = cond["NFP"]
    modes_R = cond["modes_R"]
    modes_Z = cond["modes_Z"]
    R_lmn = cond["R_lmn"]
    Z_lmn = cond["Z_lmn"]
    #--------------------

    #--------------------------
    # Default data-point build:
    data_point = {
        "features": {
            "resolution": (L, M, N),
            "NFP": NFP,
            "modes_R": modes_R,
            "modes_Z": modes_Z,
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
        },
        "labels": {
            "build_ok": False,
            "is_nested": False,
        },
        "meta": {
            "errors": [],
        },
        "continuation": {
            "enabled": bool(RUN_CONFIG.get("continuation_check", False)),
            "success": False,
            "broke": False,
            "warning_detected": False,
            "message": "",
            "error": None,
        },
    }
    #--------------------------

    try:
        #--------------------------
        # Equilibrium construction:
        surface_init = FourierRZToroidalSurface(
            R_lmn = R_lmn,
            modes_R = modes_R,
            Z_lmn = Z_lmn,
            modes_Z = modes_Z,
            NFP = NFP,
        )

        pressure_init = PowerSeriesProfile(
            [1E4, -2E4, 1E4],
            sym = True,
        )

        iota_init = PowerSeriesProfile(
            [0.5, 0.15],
            sym = True,
        )

        Psi = 1.0

        eq = Equilibrium(
            L = L,
            M = M,
            N = N,
            surface = surface_init,
            pressure = pressure_init,
            iota = iota_init,
            Psi = Psi,
            ensure_nested = False,
        )

        data_point["labels"]["build_ok"] = True

        #--------------------------------------------------------------
        # Plotting toroidal cross-sections:
        # try:
        #     base_dir = os.path.dirname(os.path.abspath(__file__))
        #     toroidal_cuts_path = os.path.join(base_dir, "initial_toroidal_cuts.png")

        #     fig, ax = plot_comparison(
        #         eqs = [eq],
        #         labels = ["Raw Eq"],
        #         color = ["green"],
        #     )
        #     fig.savefig(
        #         toroidal_cuts_path,
        #         dpi = 200,
        #         bbox_inches = "tight",
        #     )
        #     plt.close(fig)

        # except Exception as e:
        #     print("PLOTTING FAILED:", repr(e))
        #     data_point["meta"]["errors"].append(
        #         {
        #             "stage": "plotting",
        #             "error": repr(e),
        #         }
        #     )
        #--------------------------------------------------------------


        #-------------------
        # Nested flux check:
        try:
            data_point["labels"]["is_nested"] = bool(eq.is_nested())
        except Exception as e:
            data_point["labels"]["is_nested"] = False
            data_point["meta"]["errors"].append(
                {
                    "stage": "is_nested",
                    "error": repr(e),
                    "build_ok": data_point["labels"]["build_ok"],
                    "is_nested": data_point["labels"]["is_nested"],
                }
            )

        print("is_nested =", data_point["labels"]["is_nested"])
        #----------------------------------------------------------





        #----------------------------------------------
        if RUN_CONFIG.get("continuation_check", False):
            data_point["continuation"] = run_continuation_check(
                eq = eq,
                failure_fragment = RUN_CONFIG.get(
                    "continuation_failure_fragment",
                    "WARNING: Automatic continuation failed",
                ),
            )
        #---------------------------------------------------

    except Exception as e:
        print("EQUILIBRIUM BUILD FAILED:", repr(e))

        data_point["meta"]["errors"].append(
            {
                "stage": "EquilibriumBuild",
                "error": repr(e),
                "build_ok": data_point["labels"]["build_ok"],
                "is_nested": data_point["labels"]["is_nested"],
            }
        )

    return data_point
#==================
#==============================================================================================================










#============== SERIAL OR PARALLEL RUNNER ======================================================================
#==================
def run_serial(
    resolution: tuple,
    NFP: int,
    modes_R: list,
    modes_Z: list,
    R_lmn_options: list,
    Z_lmn_options: list,
):
    """
    Serial evaluation of all conditions.
    """
    conds = cond_generator(
        resolution = resolution,
        NFP = NFP,
        modes_R = modes_R,
        modes_Z = modes_Z,
        R_lmn_options = R_lmn_options,
        Z_lmn_options = Z_lmn_options,
    )

    return [evaluate_condition(cond) for cond in conds]
#==================






#==================
def run_parallel(
    resolution: tuple,
    NFP: int,
    modes_R: list,
    modes_Z: list,
    R_lmn_options: list,
    Z_lmn_options: list,
    nprocs: int | None = None,
    chunksize: int = 1,
):
    """
    Parallel evaluation of all conditions using multiprocessing.
    """
    #-----------------
    # Process count:
    if nprocs is None:
        nprocs = cpu_count()
    #-----------------

    #-----------------------
    # Materialize generator:
    conds = list(
        cond_generator(
            resolution = resolution,
            NFP = NFP,
            modes_R = modes_R,
            modes_Z = modes_Z,
            R_lmn_options = R_lmn_options,
            Z_lmn_options = Z_lmn_options,
        )
    )
    #-----------------------

    #--------------------
    # Parallel mapping:
    with Pool(processes = nprocs) as pool:
        data = pool.map(
            evaluate_condition,
            conds,
            chunksize = chunksize,
        )
    #--------------------

    return data
#==================
#==============================================================================================================










#============== MAIN ===========================================================================================
#==================
def main():
    """
    Runs equilibrium constructions from generated continuous feature options,
    saves the raw classification dataset, and builds PyTorch-ready tensors.
    """
    #---------------------------------------------
    # Generate continuous feature-option lists:
    options = feature_generator(
        config = FEATURE_CONFIG,
    )
    #---------------------------------------------

    #-----------------------------------------
    # Number of equilibrium constructions:
    N_eq = (
        len(options["R_lmn_options"])
        * len(options["Z_lmn_options"])
    )
    print("N_eq =", N_eq)
    #-----------------------------------------

    #-----------------------------------------
    # Choose whichever runner you want here:
    if RUN_CONFIG["use_parallel"]:
        data = run_parallel(
            resolution = options["resolution"],
            NFP = options["NFP"],
            modes_R = options["modes_R"],
            modes_Z = options["modes_Z"],
            R_lmn_options = options["R_lmn_options"],
            Z_lmn_options = options["Z_lmn_options"],
            nprocs = RUN_CONFIG["nprocs"],
            chunksize = RUN_CONFIG["chunksize"],
        )
    else:
        data = run_serial(
            resolution = options["resolution"],
            NFP = options["NFP"],
            modes_R = options["modes_R"],
            modes_Z = options["modes_Z"],
            R_lmn_options = options["R_lmn_options"],
            Z_lmn_options = options["Z_lmn_options"],
        )
    #-----------------------------------------

    #------------------------------
    # Dataset summary for sanity:
    n_total = len(data)
    n_build_ok = sum(bool(point["labels"]["build_ok"]) for point in data)
    n_nested = sum(bool(point["labels"]["is_nested"]) for point in data)

    frac_nested = float(n_nested) / float(n_total) if n_total > 0 else np.nan

    # print("n_total =", n_total)
    # print("n_build_ok =", n_build_ok)
    # print("n_nested =", n_nested)
    print(f"frac_nested = {frac_nested:.4f}")
    #----------------------------------------

    #--------------------------
    # Save raw dataset to disk:
    data_save_path = data_saver(
        data = data,
        NFP = options["NFP"],
        filename = DATA_CONFIG["dataset_filename"],
    )
    #---------------------------------------------

    #-----------------------------------------
    # Build PyTorch-ready feature/target data:
    torch_data = build_torch_dataset(
        data = data,
        target_key = DATA_CONFIG["target_key"],
    )
    #-----------------------------------------

    #----------------------------------
    # Save PyTorch-ready tensors to disk:
    torch_save_path = torch_data_saver(
        torch_data = torch_data,
        NFP = options["NFP"],
        filename = DATA_CONFIG["torch_dataset_filename"],
    )
    #----------------------------------------------------
#========================================================






#==================
if __name__ == "__main__":
    main()
#==================
#==============================================================================================================