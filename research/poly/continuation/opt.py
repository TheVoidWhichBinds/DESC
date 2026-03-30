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
    pressure_monotonicity,
    iota_rationals,
    grad_iota_axis
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







def _obj_settings(toggle_dict, key):
    """
    Return objective settings as:
        use (bool), weight (number or None)

    Missing objective keys default to off.
    """
    entry = toggle_dict.get(key, {"use": False, "weight": None})
    return entry.get("use", False), entry.get("weight", None)






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
    iota_upper = opt_config["iota_upper"]
    iota_lower = opt_config["iota_lower"]
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
    if FXD:  # (fixed profiles)
        #--------------------------------------
        # Standard constraints:
        if toggle.get("forcebalance_con", False):
            constraints_list.append(ForceBalance(eq=eq_0))
        if toggle.get("fix_iota_con", True):
            constraints_list.append(FixIota(eq=eq_0))
        if toggle.get("fix_psi_con", True):
            constraints_list.append(FixPsi(eq=eq_0))
        if toggle.get("fix_pressure_con", True):
            constraints_list.append(FixPressure(eq=eq_0))
        #------------------------------------------------

        #---------------------
        # Standard objectives:
        use, weight = _obj_settings(toggle, "forcebalance_obj")
        if use:
            kwargs = {"eq": eq_0, "target": 0.0}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(ForceBalance(**kwargs))

        use, weight = _obj_settings(toggle, "aspect_ratio_obj")
        if use:
            kwargs = {"eq": eq_0, "target": target_aspect_ratio}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(AspectRatio(**kwargs))

        use, weight = _obj_settings(toggle, "qs_obj")
        if use:
            kwargs = {"eq": eq_0, "helicity": (1, eq_0.NFP)}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(QuasisymmetryBoozer(**kwargs))

        use, weight = _obj_settings(toggle, "ballooning_obj")
        if use:
            kwargs = {"eq": eq_0, "target": 0.0}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(BallooningStability(**kwargs))

        use, weight = _obj_settings(toggle, "mercier_obj")
        if use:
            kwargs = {"eq": eq_0, "target": 0.0}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(MercierStability(**kwargs))
        #----------------------------------------------------------------
    #====================================================================


    #===========================
    else:  # (free profiles)
        #--------------------------------------
        # Standard constraints:
        if toggle.get("forcebalance_con", False):
            constraints_list.append(ForceBalance(eq=eq_0))
        if toggle.get("fix_iota_con", False):
            constraints_list.append(FixIota(eq=eq_0))
        if toggle.get("fix_psi_con", False):
            constraints_list.append(FixPsi(eq=eq_0))
        #-------------------------------------------

        #------------------------------------
        # Custom constraints:
        if toggle.get("pressure_axis_con", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun=pressure_axis,
                    thing=eq_0,
                    target=p_scale,
                )
            )
        if toggle.get("pressure_edge_con", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun = pressure_edge,
                    thing = eq_0,
                    target = 0.0,
                )
            )
        if toggle.get("grad_pressure_axis_con", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun=grad_pressure_axis,
                    thing = eq_0,
                    target = 0.0,
                )
            )
        if toggle.get("grad_pressure_edge_con", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun = grad_pressure_edge,
                    thing = eq_0,
                    target = 0.0,
                )
            )
        if toggle.get("grad_iota_axis_con", False):
            constraints_list.append(
                LinearObjectiveFromUser(
                    fun = grad_iota_axis,
                    thing = eq_0,
                    target = 0.0,
                )
            )
        #-----------------------------


        #---------------------
        # Standard objectives:
        use, weight = _obj_settings(toggle, "forcebalance_obj")
        if use:
            kwargs = {"eq": eq_0, "target": 0.0}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(ForceBalance(**kwargs))

        use, weight = _obj_settings(toggle, "aspect_ratio_obj")
        if use:
            kwargs = {"eq": eq_0, "target": target_aspect_ratio}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(AspectRatio(**kwargs))

        use, weight = _obj_settings(toggle, "qs_obj")
        if use:
            kwargs = {"eq": eq_0, "helicity": (1, eq_0.NFP)}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(QuasisymmetryBoozer(**kwargs))

        use, weight = _obj_settings(toggle, "ballooning_obj")
        if use:
            kwargs = {"eq": eq_0, "target": 0.0}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(BallooningStability(**kwargs))

        use, weight = _obj_settings(toggle, "mercier_obj")
        if use:
            kwargs = {"eq": eq_0, "target": 0.0}
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(MercierStability(**kwargs))
        #-----------------------------------------------------

        #-------------------
        # Custom objectives:
        use, weight = _obj_settings(toggle, "monotonicity_obj")
        if use:
            kwargs = {
                "fun": pressure_monotonicity,
                "grid": LinearGrid(rho=200, M=0, N=0),
                "thing": eq_0,
                "target": 0.0,
                "normalize": False,
            }
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(ObjectiveFromUser(**kwargs))
        
        use, weight = _obj_settings(toggle, "iota_rationals_obj")
        if use:
            kwargs = {
                "fun": iota_rationals,
                "grid": LinearGrid(rho=200, M=0, N=0),
                "thing": eq_0,
                "bounds": (iota_lower, iota_upper),
                "normalize": False,
            }
            if weight is not None:
                kwargs["weight"] = weight
            objectives_list.append(ObjectiveFromUser(**kwargs))
        
        #------------------------------------------------------
  
    #---------------------------------------
    # Finalizing optimization objects/setup:
    constraints = tuple(constraints_list)
    objectives = ObjectiveFunction(objectives_list)
    #----------------------------------------------
    #==============================================


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
    #=============


    save_name = "opt_FXD.h5" if FXD else "opt_CON.h5"
    save_path = os.path.join(out_dir, save_name)
    eq_opt.save(save_path)

    return eq_opt, opt_result
#==============================================================================================================================================================