import numpy as np
import os
import sys
sys.path.append("/Users/macdaddi/DESC")
import desc.io
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
from desc.optimize import Optimizer






#----------------------- OPTIMIZER FUNCTION --------------------------
def run_optimization(p_scale, out_dir, fix_pressure: bool):
    
    eq_init = desc.io.load(os.path.join(out_dir, 'eq.h5')) # loading initial eq solve
    eq_0 = eq_init.copy() # copying initial eq solve so as to not alter it


    # Weight to assign to secondary objectives and constraints (not force balance):
    inferior_weights = 1e4


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
        objectives = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0),
            AspectRatio(eq=eq_0, target=6),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP)),
            BallooningStability(eq=eq_0, target=0.0),
            MercierStability(eq=eq_0, target=0.0),
        ])


    else: # optimized pressure
        pressure_axis_set = LinearObjectiveFromUser(
            fun=pressure_axis,
            thing=eq_0,
            target=p_scale,
        )
        pressure_edge_zero = LinearObjectiveFromUser(
            fun=pressure_edge,
            thing=eq_0,
            target=0.0,
        )
        grad_pressure_axis_zero = LinearObjectiveFromUser(
            fun=grad_pressure_axis,
            thing=eq_0,
            target=0.0,
        )
        grad_pressure_edge_zero = LinearObjectiveFromUser(
            fun=grad_pressure_edge,
            thing=eq_0,
            target=0.0,
        )
        # Compiling constraints:
        constraints = ( # constraints don't need weights - exactly fulfilled 
            ForceBalance(eq=eq_0), # nonlinear but can be put into "constraints" via "proximal-xxx" optimizers
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
            pressure_axis_set,
            pressure_edge_zero,
            grad_pressure_axis_zero,
            grad_pressure_edge_zero,
        )
        # Building monotonicity objective with wrapper:
        negative_gradient = ObjectiveFromUser(
            fun=poly_monotonicity,
            grid=LinearGrid(rho=200, M=0, N=0),
            thing=eq_0,
            target=0.0,
            normalize=False,
        )
        # Compiling objectives:
        objectives = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0),
            AspectRatio(eq=eq_0, target=6),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP)),
            BallooningStability(eq=eq_0, target=0.0),
            MercierStability(eq=eq_0, target=0.0),
            negative_gradient,
        ])


    # Solving optimization:
    eq_opt, opt_result = eq_0.optimize(
        objective = objectives,
        constraints = constraints,
        optimizer = Optimizer("lsq-auglag"), # trust region augmented lagrangian
        ftol = 5e-2,
        xtol = 1e-3,
        gtol = 1e-4,
        maxiter=100,
        copy=True,
        verbose=3,
    )

    
    save_name = "opt_FXP.h5" if fix_pressure else "opt.h5"
    save_path = os.path.join(out_dir, save_name)
    eq_opt.save(save_path)
    
    return eq_opt, opt_result
