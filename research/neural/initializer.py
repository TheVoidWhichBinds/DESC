# initializer.py

from multiprocessing import Pool, cpu_count
import numpy as np
import os
import matplotlib.pyplot as plt
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.objectives import PrincipalCurvature, MeanCurvature, GoodCoordinates
from desc.plotting import plot_comparison

from .config import FEATURE_CONFIG, SCORING_CONFIG, RUN_CONFIG
from .helper import (
    cond_generator,
    feature_generator,
    data_saver,
    build_torch_dataset,
    torch_data_saver,
    save_valid_equilibrium,
)











#============== LOCAL DEFAULTS =================================================================================
PRESSURE_SYM = True
IOTA_SYM = True

OBJECTIVE_LOSS_FUNCTION = "mean"
OBJECTIVE_NORMALIZE = False

ENSURE_NESTED = False

TOROIDAL_CUTS_FILENAME = "toroidal_cuts.png"
PLOT_DPI = 200

LABEL_KEYS = [
    "score_total",
    "gc_mean",
    "pc_mean",
    "mc_mean",
    "is_nested",
    "build_ok",
]
#==============================================================================================================











#============== SINGLE-CONDITION SCORING =======================================================================
#====================
def score(cond):
    """
    Build one equilibrium from one condition dictionary and score it.

    Returns one nested ML-style data point with:
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
    p_l = cond["p_l"]
    i_l = cond["i_l"]
    Psi = float(cond["Psi"])
    w_gc, w_pc, w_c = cond["weights"]
    #-----------------------

    #---------------------
    # Config shortcuts:
    fallback_penalty = SCORING_CONFIG["fallback_penalty"]
    default_score_total = SCORING_CONFIG["default_score_total"]
    #---------------------

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
            "p_l": p_l,
            "i_l": i_l,
            "Psi": Psi,
        },
        "labels": {
            "build_ok": False,
            "is_nested": False,
            "gc_mean": np.nan,
            "pc_mean": np.nan,
            "mc_mean": np.nan,
            "score_total": default_score_total,
        },
        "meta": {
            "errors": [],
            "valid_eq_path": None,
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
            p_l,
            sym = PRESSURE_SYM,
        )

        iota_init = PowerSeriesProfile(
            i_l,
            sym = IOTA_SYM,
        )

        eq = Equilibrium(
            L = L,
            M = M,
            N = N,
            surface = surface_init,
            pressure = pressure_init,
            iota = iota_init,
            Psi = Psi,
            ensure_nested = ENSURE_NESTED,
        )

        data_point["labels"]["build_ok"] = True
        #--------------------------------------

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
        #-------------------

        #-----------------------------------------
        # Save equilibrium if it is actually valid:
        try:
            if data_point["labels"]["is_nested"]:
                save_path = save_valid_equilibrium(eq = eq)
                data_point["meta"]["valid_eq_path"] = save_path
        except Exception as e:
            data_point["meta"]["errors"].append(
                {
                    "stage": "save_valid_equilibrium",
                    "error": repr(e),
                    "build_ok": data_point["labels"]["build_ok"],
                    "is_nested": data_point["labels"]["is_nested"],
                }
            )
        #-----------------------------------------

        #-------------------------
        # GoodCoordinates penalty:
        try:
            gc_obj = GoodCoordinates(
                eq = eq,
                loss_function = OBJECTIVE_LOSS_FUNCTION,
                normalize = OBJECTIVE_NORMALIZE,
            )
            gc_obj.build()
            gc_val = gc_obj.compute_unscaled(eq.params_dict)
            data_point["labels"]["gc_mean"] = float(np.asarray(gc_val).item())
        except Exception as e:
            data_point["meta"]["errors"].append(
                {
                    "stage": "GoodCoordinates",
                    "error": repr(e),
                    "build_ok": data_point["labels"]["build_ok"],
                    "is_nested": data_point["labels"]["is_nested"],
                }
            )
        #-------------------------

        #------------------------------
        # PrincipalCurvature penalty:
        try:
            pc_obj = PrincipalCurvature(
                eq = eq,
                loss_function = OBJECTIVE_LOSS_FUNCTION,
                normalize = OBJECTIVE_NORMALIZE,
            )
            pc_obj.build()
            pc_val = pc_obj.compute_unscaled(eq.params_dict)
            data_point["labels"]["pc_mean"] = float(np.asarray(pc_val).item())
        except Exception as e:
            data_point["meta"]["errors"].append(
                {
                    "stage": "PrincipalCurvature",
                    "error": repr(e),
                    "build_ok": data_point["labels"]["build_ok"],
                    "is_nested": data_point["labels"]["is_nested"],
                }
            )
        #------------------------------

        #-------------------------------
        # MeanCurvature penalty, mean:
        try:
            mc_obj = MeanCurvature(
                eq = eq,
                loss_function = OBJECTIVE_LOSS_FUNCTION,
                normalize = OBJECTIVE_NORMALIZE,
            )
            mc_obj.build()
            mc_val = mc_obj.compute_unscaled(eq.params_dict)
            data_point["labels"]["mc_mean"] = float(np.asarray(mc_val).item())
        except Exception as e:
            data_point["meta"]["errors"].append(
                {
                    "stage": "MeanCurvature",
                    "error": repr(e),
                    "build_ok": data_point["labels"]["build_ok"],
                    "is_nested": data_point["labels"]["is_nested"],
                }
            )
        #-------------------------------

        #----------------------------------------
        # Assigning criteria residuals to scores:
        gc_penalty = data_point["labels"]["gc_mean"]
        if not np.isfinite(gc_penalty):
            gc_penalty = fallback_penalty

        pc_penalty = data_point["labels"]["pc_mean"]
        if not np.isfinite(pc_penalty):
            pc_penalty = fallback_penalty

        c_penalty = data_point["labels"]["mc_mean"]
        if not np.isfinite(c_penalty):
            c_penalty = fallback_penalty

        print("is_nested =", data_point["labels"]["is_nested"])

        data_point["labels"]["score_total"] = float(
            np.linalg.norm(
                [
                    w_gc * gc_penalty,
                    w_pc * pc_penalty,
                    w_c * c_penalty,
                ]
            )
        )

        print(f'Total Score: {data_point["labels"]["score_total"]:.4e}')
        #----------------------------------------







        base_dir = os.path.dirname(os.path.abspath(__file__))
        #--------------------------------------------------------------
        # Plotting toroidal cross-sections (confirmation of eq health):
        plt.title("Toroidal Cross-Sections of Initial Equilibrium")
        fig, ax = plot_comparison(
            eqs = [eq],
            labels = ["Initial Equilibrium"],
            color = ["green"],
        )
        toroidal_cuts_path = os.path.join(base_dir, TOROIDAL_CUTS_FILENAME)
        plt.savefig(toroidal_cuts_path, dpi = PLOT_DPI)
        plt.close()
        #----------





    #----------
    except Exception as e:
        data_point["meta"]["errors"].append(
            {
                "stage": "EquilibriumBuild",
                "error": repr(e),
                "build_ok": data_point["labels"]["build_ok"],
                "is_nested": data_point["labels"]["is_nested"],
            }
        )

    return data_point
#====================











#============== SERIAL OR PARALLEL RUNNER ======================================================================
#=====================
def run_serial(
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
    Serial evaluation of all conditions.
    """
    conds = cond_generator(
        resolution = resolution,
        NFP = NFP,
        modes_R = modes_R,
        modes_Z = modes_Z,
        R_lmn_options = R_lmn_options,
        Z_lmn_options = Z_lmn_options,
        p_l_options = p_l_options,
        i_l_options = i_l_options,
        Psi_options = Psi_options,
        weights = weights,
    )

    return [score(cond) for cond in conds]
#=========================================





#================
def run_parallel(
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
            p_l_options = p_l_options,
            i_l_options = i_l_options,
            Psi_options = Psi_options,
            weights = weights,
        )
    )
    #-----------------------

    #--------------------
    # Parallel mapping:
    with Pool(processes = nprocs) as pool:
        data = pool.map(
            score,
            conds,
            chunksize = chunksize,
        )
    #--------------------

    return data
#==============
#==============================================================================================================











#============== MAIN ===========================================================================================
#================
def main():
    """
    Runs equilibrium constructions from generated continuous feature options,
    saves the nested raw dataset, and builds PyTorch-ready tensors.
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
        * len(options["p_l_options"])
        * len(options["i_l_options"])
        * len(options["Psi_options"])
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
            p_l_options = options["p_l_options"],
            i_l_options = options["i_l_options"],
            Psi_options = options["Psi_options"],
            weights = SCORING_CONFIG["weights"],
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
            p_l_options = options["p_l_options"],
            i_l_options = options["i_l_options"],
            Psi_options = options["Psi_options"],
            weights = SCORING_CONFIG["weights"],
        )
    #-----------------------------------------

    #----------------------------------
    # Save raw nested dataset to disk:
    data_save_path = data_saver(
        data = data,
        NFP = options["NFP"],
    )
    #----------------------------------

    #-----------------------------------------
    # Build PyTorch-ready feature/label data:
    torch_data = build_torch_dataset(
        data = data,
        label_keys = LABEL_KEYS,
    )
    #-----------------------------------------

    #-----------------------------------
    # Save PyTorch-ready tensors to disk:
    torch_save_path = torch_data_saver(
        torch_data = torch_data,
        NFP = options["NFP"],
    )
    #-----------------------------------
#=====================================================================





#=========================
if __name__ == "__main__":
    main()
#=========================
#==============================================================================================================