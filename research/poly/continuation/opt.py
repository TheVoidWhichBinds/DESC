import numpy as np
import os
import sys
sys.path.append("/Users/macdaddi/DESC")

from desc.grid import LinearGrid
from research.poly.poly_constraints import (
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    poly_monotonicity
)
from desc.objectives import (
    ObjectiveFunction,
    FixIota,
    FixPsi,
    FixPressure,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    BallooningStability,
    MercierStability,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)






#============== OPTIMIZER FUNCTION ============================================================================================================================
def run_optimization(eq_0, optimizer, p_scale, out_dir, opt_config, FXD: bool):
    """
    Run optimization with objectives/constraints controlled by config dict.

    FXD = True  -> fixed-pressure family
    FXD = False -> constrained/optimized-pressure family
    """

    #=============================================
    # Unpacking optimization configuration values:
    target_aspect_ratio = opt_config["target_aspect_ratio"]
    ftol = opt_config["ftol"]
    xtol = opt_config["xtol"]
    gtol = opt_config["gtol"]
    maxiter = opt_config["maxiter"]

    toggle_FXD = opt_config["toggle_FXD"]
    toggle_CON = opt_config["toggle_CON"]
    toggle = toggle_FXD if FXD else toggle_CON
    #=============================================

    constraints_list = []
    objectives_list = []

    #=========================
    if FXD:  # (fixed pressure)
        #----------------------
        # Construction of constraints:
        if toggle.get("forcebalance_constraint", False):
            constraints_list.append(ForceBalance(eq=eq_0))
        if toggle.get("fix_iota", True):
            constraints_list.append(FixIota(eq=eq_0))
        if toggle.get("fix_psi", True):
            constraints_list.append(FixPsi(eq=eq_0))
        if toggle.get("fix_pressure", True):
            constraints_list.append(FixPressure(eq=eq_0))
        #------------------------------------------------

        #----------------------------
        # Construction of objectives:
        if toggle.get("forcebalance_objective", False):
            objectives_list.append(ForceBalance(eq=eq_0, target=0.0))
        if toggle.get("aspect_ratio", False):
            objectives_list.append(AspectRatio(eq=eq_0, target=target_aspect_ratio))
        if toggle.get("qs", False):
            objectives_list.append(
                QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP))
            )
        if toggle.get("ballooning", False):
            objectives_list.append(BallooningStability(eq=eq_0, target=0.0))
        if toggle.get("mercier", False):
            objectives_list.append(MercierStability(eq=eq_0, target=0.0))
        #----------------------------------------------------------------
    #====================================================================

    #===========================
    else:  # (optimized pressure)
        #------------------------------------
        # Construction of custom constraints:
        if toggle.get("pressure_axis", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun=pressure_axis,
                    thing=eq_0,
                    target=p_scale,
                )
            )
        if toggle.get("pressure_edge", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun=pressure_edge,
                    thing=eq_0,
                    target=0.0,
                )
            )
        if toggle.get("grad_pressure_axis", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun=grad_pressure_axis,
                    thing=eq_0,
                    target=0.0,
                )
            )
        if toggle.get("grad_pressure_edge", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun=grad_pressure_edge,
                    thing=eq_0,
                    target=0.0,
                )
            )
        #------------------------------------

        #--------------------------------------
        # Construction of standard constraints:
        if toggle.get("forcebalance_constraint", False):
            constraints_list.append(ForceBalance(eq=eq_0))
        if toggle.get("fix_iota", False):
            constraints_list.append(FixIota(eq=eq_0))
        if toggle.get("fix_psi", False):
            constraints_list.append(FixPsi(eq=eq_0))
        #--------------------------------------

        #----------------------------
        # Construction of objectives:
        if toggle.get("forcebalance_objective", False):
            objectives_list.append(ForceBalance(eq=eq_0, target=0.0))
        if toggle.get("aspect_ratio", False):
            objectives_list.append(AspectRatio(eq=eq_0, target=target_aspect_ratio))
        if toggle.get("qs", False):
            objectives_list.append(
                QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP))
            )
        if toggle.get("ballooning", False):
            objectives_list.append(BallooningStability(eq=eq_0, target=0.0))
        if toggle.get("mercier", False):
            objectives_list.append(MercierStability(eq=eq_0, target=0.0))
        if toggle.get("monotonicity", False):
            objectives_list.append(
                ObjectiveFromUser(
                    fun=poly_monotonicity,
                    grid=LinearGrid(rho=200, M=0, N=0),
                    thing=eq_0,
                    target=0.0,
                    normalize=False,
                )
            )
        #---------------------------
    #===============================

    #=======================================
    # Finalizing optimization objects/setup:
    constraints = tuple(constraints_list)

    if len(objectives_list) == 0:
        raise ValueError("No optimization objectives were selected.")

    objectives = ObjectiveFunction(objectives_list)
    #=======================================

    #======================
    # Running optimization:
    eq_opt, opt_result = eq_0.optimize(
        objective=objectives,
        constraints=constraints,
        optimizer=optimizer,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        maxiter=maxiter,
        copy=True,
        verbose=3,
    )
    #======================

    save_name = "opt_FXD.h5" if FXD else "opt_CON.h5"
    save_path = os.path.join(out_dir, save_name)
    eq_opt.save(save_path)

    return eq_opt, opt_result
#==============================================================================================================================================================