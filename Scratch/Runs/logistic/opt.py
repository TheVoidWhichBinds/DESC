import numpy as np
import os
import sys
sys.path.append("/Users/macdaddi/DESC")
import desc.io
from desc.grid import LinearGrid
from scratch.objectives.poly_constraints import (
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
from desc.optimize import Optimizer



#------------------- OPTIMIZER SELECTION ------------------------------------------------------------------------
optimizer = Optimizer("proximal-lsq-exact")
#----------------------------------------------------------------------------------------------------------------



#------------------- OPTIMIZER FUNCTION -------------------------------------------------------------------------
def run_optimization(p_axis, out_dir, fix_pressure: bool):
    
    eq_init = desc.io.load(os.path.join(out_dir, 'eq.h5')) # loading initial eq solve
    eq_0 = eq_init.copy() # copying initial eq solve so as to not alter it

    # Weight to assign to secondary objectives and constraints (not force balance):
    inferior_weights = 1e0 

    #------------------------------------------------------------------
    # Division of constraints and objectives depending on fix_pressure:
    if fix_pressure: # fixed pressure
        # Compiling constraints:
        constraints = (
            ForceBalance(eq=eq_0),
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
            FixPressure(eq=eq_0),
        )
        # Compiling objectives:
        objective = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e1),
            AspectRatio(eq=eq_0, target=6, weight=inferior_weights),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=inferior_weights),
            BallooningStability(eq=eq_0, target=0.0, weight=inferior_weights),
            MercierStability(eq=eq_0, target=0.0, weight=inferior_weights),
        ])


    else: # optimized pressure
        
        # Compiling constraints:
        constraints = (
            ForceBalance(eq=eq_0),
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
        )
        
        # Compiling objectives:
        objective = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e4),
            AspectRatio(eq=eq_0, target=6, weight=inferior_weights),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=inferior_weights),
            BallooningStability(eq=eq_0, target=0.0, weight=inferior_weights),
            MercierStability(eq=eq_0, target=0.0, weight=inferior_weights),
        ])
    #----------------------------------------------------------------------


    #----------------------
    # Solving optimization:
    eq_opt, opt_result = eq_0.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        ftol=5e-2,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=50,
        options={
            "perturb_options": {"order": 2, "verbose": 0},
            "solve_options": {"ftol": 5e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0},
        },
        copy=False,
        verbose=3,
    )
    #-------------

    
    save_name = "opt_FXP.h5" if fix_pressure else "opt.h5"
    save_path = os.path.join(out_dir, save_name)
    eq_opt.save(save_path)
    
    return eq_opt, opt_result
#--------------------------------------------------------------------------------------------------------------------