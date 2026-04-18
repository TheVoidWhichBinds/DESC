

from multiprocessing import Pool, cpu_count
import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.objectives import ForceBalance, GoodCoordinates
from .helper import cond_generator










#============== SINGLE-CONDITION SCORING =============================================================================
def score(cond):
    """
    Build one equilibrium from one condition dictionary and score it.

    Returns a row dict suitable for ML dataset construction.
    """
    #=======================
    # Unpack feature vector:
    L, M, N = cond["resolution"]
    NFP = cond["NFP"]
    R_lmn, modes_R = cond["R"]
    Z_lmn, modes_Z = cond["Z"]
    p_l = cond["p_l"]
    i_l = cond["i_l"]
    Psi = cond["Psi"]
    #========================

    #===================
    # Default row build:
    row = {
        "L": L,
        "M": M,
        "N": N,
        "NFP": NFP,
        "R_lmn": R_lmn,
        "modes_R": modes_R,
        "Z_lmn": Z_lmn,
        "modes_Z": modes_Z,
        "p_l": p_l,
        "i_l": i_l,
        "Psi": Psi,
        "build_ok": False,
        "is_nested": False,
        "goodcoords_mean": np.nan,
        "forcebalance_mean": np.nan,
        "score_total": 100.0, # if eq construction unsuccessful, loss defaults 100 (worst)
        "error": None,
    }
    #=================



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
        #=======================



        row["build_ok"] = True 
        #================================
        # Extracting raw criteria values:
        #-------------------
        # Nested flux check:
        try:
            row["is_nested"] = bool(eq.is_nested())
        except Exception:
            row["is_nested"] = False
        #---------------------------

        #-----------------------
        # GoodCoordinates check:
        try:
            gc_obj = GoodCoordinates(eq=eq)
            gc_val = gc_obj.compute_unscaled(eq.params_dict)
            gc_arr = np.asarray(gc_val, dtype=float)
            row["goodcoords_mean"] = float(np.mean(np.abs(gc_arr)))
        except Exception:
            row["goodcoords_mean"] = np.nan
        #----------------------------------

        #--------------------
        # ForceBalance check:
        try:
            fb_obj = ForceBalance(eq=eq)
            fb_val = fb_obj.compute_unscaled(eq.params_dict)
            fb_arr = np.asarray(fb_val, dtype=float)
            row["forcebalance_mean"] = float(np.mean(np.abs(fb_arr)))
        except Exception:
            row["forcebalance_mean"] = np.nan
        #------------------------------------
        #====================================



        #========================================
        # Assigning criteria residuals to scores:
        #---------------------
        # Nested flux scoring:
        nested_penalty = 0.0 if row["is_nested"] else 10.0
        #-------------------------------------------------

        #-------------------------
        # GoodCoordinates scoring:
        gc_penalty = row["goodcoords_mean"]
        if not np.isfinite(gc_penalty):
            gc_penalty = 50.0
        #--------------------

        #----------------------
        # ForceBalance scoring:
        fb_penalty = row["forcebalance_mean"]
        if not np.isfinite(fb_penalty):
            fb_penalty = 50.0
        #--------------------
        #====================



        #=================================================
        # Normalizing and compiling scores into loss func:
        row["score_total"] = float(
            np.linalg.norm([
                nested_penalty,
                gc_penalty,
                fb_penalty,
            ])
        )
        #==================

    except Exception as e:
        row["error"] = repr(e)

    return row
#==============================================================================================================










#============== SERIAL RUNNER ===================================================================================
def run_serial(
    resolution_range: list,
    NFP_range: list,
    R_range: list,
    Z_range: list,
    p_l_range: list,
    i_l_range: list,
    Psi_range: list,
):
    """
    Serial evaluation of all conditions.
    """
    conds = cond_generator(
        resolution_range = resolution_range,
        NFP_range = NFP_range,
        R_range = R_range,
        Z_range = Z_range,
        p_l_range = p_l_range,
        i_l_range = i_l_range,
        Psi_range = Psi_range,
    )

    return [score(cond) for cond in conds] # looping over all initial conditions
#==============================================================================================================










#============== PARALLEL RUNNER =================================================================================
def run_parallel(
    resolution_range: list,
    NFP_range: list,
    R_range: list,
    Z_range: list,
    p_l_range: list,
    i_l_range: list,
    Psi_range: list,
    nprocs: int | None = None,
    chunksize: int = 1,
):
    """
    Parallel evaluation of all conditions using multiprocessing.
    """
    #-----------------
    if nprocs is None:
        nprocs = cpu_count() # num CPU cores available
    #-------------------------------------------------

    #---------------------------------------------------------
    # Generating initial conditions for all range permutations
    conds = list(
        cond_generator(
            resolution_range = resolution_range,
            NFP_range = NFP_range,
            R_range = R_range,
            Z_range = Z_range,
            p_l_range = p_l_range,
            i_l_range = i_l_range,
            Psi_range = Psi_range,
        )
    )
    #-----------------------------

    #-----------------------------------------------------------
    # Assigning initial conditions to be evaluated by score func
    with Pool(processes = nprocs) as pool: # CPU cores = nprocs
        rows = pool.map(
            score, # func
            conds, # initial conditions
            chunksize = chunksize # score evaluations per process
        )
    #------------------------------------------------------------

    return rows
#==============================================================================================================

