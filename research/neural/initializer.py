from multiprocessing import Pool, cpu_count
import numpy as np
import time

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.objectives import PrincipalCurvature, MeanCurvature, GoodCoordinates

from .helper import (
    cond_generator,
    feature_generator,
    data_saver,
    build_torch_dataset,
    torch_data_saver,
)




#============== HYPERPARAMETERS ================================================================================
weights = [5, 1, 2]   # w_gc, w_pc, w_c
N_options = 1
#==============================================================================================================




#============== SINGLE-CONDITION SCORING =======================================================================
#==================
def score(cond):
    """
    Build one equilibrium from one condition dictionary and score it.

    Returns one nested ML-style data point with:
        data_point["features"][...]
        data_point["labels"][...]
        data_point["meta"][...]
    """

    #=======================
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
    #=======================


    #==========================
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
            "mc_max": np.nan,
            "score_total": 100.0,
        },
        "meta": {
            "error": None,
        },
    }
    #==========================


    try:
        #==========================
        # Equilibrium construction:
        surface_init = FourierRZToroidalSurface(
            R_lmn=R_lmn,
            modes_R=modes_R,
            Z_lmn=Z_lmn,
            modes_Z=modes_Z,
            NFP=NFP,
        )

        pressure_init = PowerSeriesProfile(
            p_l,
            sym=True,
        )

        iota_init = PowerSeriesProfile(
            i_l,
            sym=True,
        )

        eq = Equilibrium(
            L=L,
            M=M,
            N=N,
            surface=surface_init,
            pressure=pressure_init,
            iota=iota_init,
            Psi=Psi,
            ensure_nested=False,
        )

        data_point["labels"]["build_ok"] = True
        #======================================


        #================================
        # Extracting raw criteria values:
        #-------------------
        # Nested flux check:
        try:
            data_point["labels"]["is_nested"] = bool(eq.is_nested())
        except Exception:
            data_point["labels"]["is_nested"] = False
        #--------------------

        #-------------------------
        # GoodCoordinates penalty:
        try:
            gc_obj = GoodCoordinates(
                eq=eq,
                loss_function="mean",
                normalize=False,
            )
            gc_obj.build()
            gc_val = gc_obj.compute_unscaled(eq.params_dict)
            data_point["labels"]["gc_mean"] = float(np.asarray(gc_val).item())
        except Exception as e:
            print("GoodCoordinates error:", repr(e))
        #-------------------------

        #------------------------------
        # PrincipalCurvature penalty:
        try:
            pc_obj = PrincipalCurvature(
                eq=eq,
                loss_function="mean",
                normalize=False,
            )
            pc_obj.build()
            pc_val = pc_obj.compute_unscaled(eq.params_dict)
            data_point["labels"]["pc_mean"] = float(np.asarray(pc_val).item())
        except Exception as e:
            print("PrincipalCurvature error:", repr(e))
        #------------------------------

        #--------------------------------------------
        # MeanCurvature penalty, max over all nodes:
        try:
            mc_obj = MeanCurvature(
                eq=eq,
                loss_function="max",
                normalize=False,
            )
            mc_obj.build()
            mc_val = mc_obj.compute_unscaled(eq.params_dict)
            data_point["labels"]["mc_max"] = float(np.asarray(mc_val).item())
        except Exception as e:
            print("MeanCurvature error:", repr(e))
        #-----------------------------------------
        #=========================================


        #========================================
        # Assigning criteria residuals to scores:
        gc_penalty = data_point["labels"]["gc_mean"]
        if not np.isfinite(gc_penalty):
            gc_penalty = 100.0

        pc_penalty = data_point["labels"]["pc_mean"]
        if not np.isfinite(pc_penalty):
            pc_penalty = 100.0

        c_penalty = data_point["labels"]["mc_max"]
        if not np.isfinite(c_penalty):
            c_penalty = 100.0
        if c_penalty < 0:
            c_penalty = 0.0

        print("is_nested =", data_point["labels"]["is_nested"])
        print("gc_mean =", data_point["labels"]["gc_mean"])
        print("pc_mean =", data_point["labels"]["pc_mean"])
        print("mc_max =", data_point["labels"]["mc_max"])

        data_point["labels"]["score_total"] = float(
            np.linalg.norm([
                w_gc * gc_penalty,
                w_pc * pc_penalty,
                w_c * c_penalty,
            ])
        )
        #=======================


    except Exception as e:
        data_point["meta"]["error"] = repr(e)

    return data_point
#==============================
#==============================================================================================================










#============== SERIAL RUNNER =================================================================================
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
        resolution=resolution,
        NFP=NFP,
        modes_R=modes_R,
        modes_Z=modes_Z,
        R_lmn_options=R_lmn_options,
        Z_lmn_options=Z_lmn_options,
        p_l_options=p_l_options,
        i_l_options=i_l_options,
        Psi_options=Psi_options,
        weights=weights,
    )
    return [score(cond) for cond in conds]
#==============================================================================================================










#============== PARALLEL RUNNER ===============================================================================
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
    if nprocs is None:
        nprocs = cpu_count()
    #-----------------------

    #------------
    conds = list(
        cond_generator(
            resolution=resolution,
            NFP=NFP,
            modes_R=modes_R,
            modes_Z=modes_Z,
            R_lmn_options=R_lmn_options,
            Z_lmn_options=Z_lmn_options,
            p_l_options=p_l_options,
            i_l_options=i_l_options,
            Psi_options=Psi_options,
            weights=weights,
        )
    )
    #------------------------

    #-----------------------------------
    with Pool(processes=nprocs) as pool:
        data = pool.map(
            score,
            conds,
            chunksize=chunksize,
        )
    #---------------------------

    return data
#==============================================================================================================


















#============== MAIN ===========================================================================================
#==========
def main():
    """
    Runs equilibrium constructions from generated continuous feature options,
    saves the nested raw dataset, and builds PyTorch-ready tensors.
    """
    t_start = time.perf_counter()

    #---------------------------------------------
    # Generate continuous feature-option lists:
    options = feature_generator(N = N_options)
    #---------------------------------------------

    #-----------------------------------------
    # Choose whichever runner you want here:
    data = run_serial(
        resolution=options["resolution"],
        NFP=options["NFP"],
        modes_R=options["modes_R"],
        modes_Z=options["modes_Z"],
        R_lmn_options=options["R_lmn_options"],
        Z_lmn_options=options["Z_lmn_options"],
        p_l_options=options["p_l_options"],
        i_l_options=options["i_l_options"],
        Psi_options=options["Psi_options"],
        weights=weights,
    )
    #-----------------------------------------

    #----------------------------------
    # Save raw nested dataset to disk:
    data_save_path = data_saver(
        data=data,
        filename="dataset.pkl",
    )
    #----------------------------------

    #-----------------------------------------
    # Build PyTorch-ready feature/label data:
    torch_data = build_torch_dataset(
        data=data,
        label_keys=[
            "score_total",
            "gc_mean",
            "pc_mean",
            "mc_max",
            "is_nested",
            "build_ok",
        ],
    )
    #-----------------------------------------

    #-----------------------------------
    # Save PyTorch-ready tensors to disk:
    torch_save_path = torch_data_saver(
        torch_data=torch_data,
        filename="torch_dataset.pt",
    )
    #-----------------------------------

    t_end = time.perf_counter()

    print("elapsed =", t_end - t_start, "s")
    print("saved raw data to =", data_save_path)
    print("saved torch data to =", torch_save_path)
    print("n_data =", len(data))

    if len(data) > 0:
        print("first score_total =", data[0]["labels"]["score_total"])

    print("X shape =", tuple(torch_data["X"].shape))
    print("y shape =", tuple(torch_data["y"].shape))
#===================================================





#-------------------------
if __name__ == "__main__":
    main()
#---------
#==============================================================================================================