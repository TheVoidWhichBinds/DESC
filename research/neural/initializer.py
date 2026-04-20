
from multiprocessing import Pool, cpu_count
import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.objectives import PrincipalCurvature, MeanCurvature, GoodCoordinates
from .helper import cond_generator
import time









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
    w_n, w_gc, w_pc, w_c = cond["weights"]
    #=====================================

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
        "goodcoordinates_mean": np.nan,
        "principalcurvature_mean": np.nan,
        "meancurvature_max": np.nan,
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
        #eq = solve_continuation_automatic(eq = eq, verbose=0)[-1]
        #========================================================



        row["build_ok"] = True 
        #================================
        # Extracting raw criteria values:
        #-------------------
        # Nested flux check:
        try:
            row["is_nested"] = bool(eq.is_nested())
        except Exception:
            row["is_nested"] = False
        
        if not row["is_nested"]:
            row["score_total"] = float(w_n * 100.0)
            return row # terminates build early if flux surfaces not nested
        #------------------------------------------------------------------

        #---------------------------------------------------
        # Non-intersecting field lines, mean over all nodes:
        try:
            gc_obj = GoodCoordinates(
                eq = eq,
                loss_function = "mean",
                normalize = False,
            )
            gc_obj.build()
            gc_val = gc_obj.compute_unscaled(eq.params_dict)
            row["goodcoordinates_mean"] = float(np.asarray(gc_val).item())
        except Exception as e:
            row["goodcoordinates_mean"] = np.nan
            print("GoodCoordinates error:", repr(e))
        #-------------------------------------------

        #---------------------------------------------------------------
        # Max of two curvatures at each node, mean taken over all nodes:
        try:
            pc_obj = PrincipalCurvature(
                eq = eq, 
                loss_function = "mean", 
                normalize = False,
            )
            pc_obj.build()
            pc_val = pc_obj.compute_unscaled(eq.params_dict)
            row["principalcurvature_mean"] = float(np.asarray(pc_val).item())
        except Exception as e:
            row["principalcurvature_mean"] = np.nan
            print("PrincipalCurvature error:", repr(e))
        #----------------------------------------------

        #-------------------------------------------------------------------------------
        # Mean of two curvatures at each node, max over all nodes - penalizes concavity:
        try:
            mc_obj = MeanCurvature(
                eq = eq, 
                loss_function = "max", 
                normalize = False,
            )
            mc_obj.build()
            mc_val = mc_obj.compute_unscaled(eq.params_dict)
            row["meancurvature_max"] = float(np.asarray(mc_val).item())
        except Exception as e:
            row["meancurvature_max"] = np.nan
            print("MeanCurvature error:", repr(e))
        #-----------------------------------------
        #=========================================



        #========================================
        # Assigning criteria residuals to scores:
        #---------------------
        # Nested flux scoring:
        n_penalty = 0.0 if row["is_nested"] else 100.0
        #---------------------------------------------

        #-------------------------
        # GoodCoordinates scoring:
        gc_penalty = row["goodcoordinates_mean"]
        if not np.isfinite(gc_penalty):
            gc_penalty = 100.0
        #--------------------

        #----------------------------
        # PrincipalCurvature scoring:
        pc_penalty = row["principalcurvature_mean"]
        if not np.isfinite(pc_penalty):
            pc_penalty = 100.0
        #--------------------

        #-------------------
        # Concavity scoring:
        c_penalty = row["meancurvature_max"]
        if not np.isfinite(c_penalty):
            c_penalty = 100.0
        if c_penalty < 0:
            c_penalty = 0 # convexity penalized by pc_penalty, not here
        #---------------------------------------------------------------
        #===============================================================



        #=================================================
        # Normalizing and compiling scores into loss func:
        row["score_total"] = float(
            np.linalg.norm([
                w_n * n_penalty,
                w_gc * gc_penalty,
                w_pc * pc_penalty,
                w_c * c_penalty,
            ])
        )
        #=======================

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
    weights: list,
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
        weights = weights,
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
    weights: list,
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
            weights = weights,
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










#============== PARAMETER RANGES & RUNNING IT =================================================================================
#==================
resolution_range = [
    (16, 16, 8),
]

NFP_range = [
    6,
]

R_range = [
    (
        [4.60, -0.72, 0.14, -0.04],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
    ),
]

Z_range = [
    (
        [1.02, 0.08, -0.02, 0.025],
        [(1, 0), (2, 0), (3, 0), (1, 1)],
    ),
]

p_l_range = [
    [5e5, -1e6, 5e5],
]

i_l_range = [
    [1.55, 0.12],
]

Psi_range = [
    1.0,
]

weights = [10, 5, 1, 2] # w_n, w_gc, w_pc, w_c
#=============================================




#==========
def main():
    """
    Runs Equilibria constructions from chosen
    hyperparameter ranges.
    """
    t_start = time.perf_counter()
    #-----------------
    rows = run_serial(
        resolution_range = resolution_range,
        NFP_range = NFP_range,
        R_range = R_range,
        Z_range = Z_range,
        p_l_range = p_l_range,
        i_l_range = i_l_range,
        Psi_range = Psi_range,
        weights = weights,
    )
    #---------------------
    t_end = time.perf_counter()
    print("elapsed =", t_end - t_start, "s")
    print("score_total =", rows[0]["score_total"])
#===========================================




if __name__ == "__main__":
    main()
#==============================================================================================================
