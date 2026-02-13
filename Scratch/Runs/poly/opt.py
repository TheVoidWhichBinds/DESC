#BASIS FIXED BOUNDARY QS OPTIMIZATION MODDED TO TAKE POLYNOMIAL COEFFICIENTS AS OPTIMIZATION VARIABLES AND PRESSURE CONSTRAINTS AS ADDITIONAL OBJECTIVES
#NOTES:
#play with lowering ftol


import numpy as np
import matplotlib.patheffects as pe
import os
import sys
sys.path.append("/Users/macdaddi/DESC")
import matplotlib.pyplot as plt
import desc.io
from desc.grid import LinearGrid




#-----------------------------------------------
# Importing custom constraint funcs:
from scratch.objectives.poly_constraints import( 
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    poly_monotonicity
)

# Importing objectives:
from desc.objectives import (
    ObjectiveFunction,
    FixIota,
    FixPsi,
    FixPressure,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)

# Importing Optimizer:
from desc.optimize import Optimizer




#-----------------------------------------------------------------
# Global settings:
#Optimizer of Choice:
optimizer = Optimizer("proximal-fmintr-bfgs") #choice of optimizer




#--------------------------------------------------------------
def run_optimization(load_path, fix_pressure: bool):
    # Loading Final Iteration of Equilibrium Family from eq.py:
    eq_init = desc.io.load(load_path)
    eq_0 = eq_init.copy()



    # EITHER pressure is fixed OR custom constraints implemented: 
    #------------------------------------------------------------
    if fix_pressure == True: # Pressure EXcluded from optimization
        # Combining constraints:
        constraints = (
            ForceBalance(eq = eq_0), # enforce JxB-grad(p)=0 during optimization
            FixIota(eq = eq_0), # fix rotational transform profile
            FixPsi(eq = eq_0), # fix total toroidal magnetic flux
            FixPressure(eq = eq_0), # EITHER fix pressure OR custom objectives + constraints
        )

        # Crombining objectives:
        objective= ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e1), # J x B - Grad(P) = 0
            AspectRatio(eq=eq_0, target=6, weight=1e-1), # acceptable range: 
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=1e-2), #TARGET??? acceptable range: 
        ])
        

    else: # Pressure INcluded in optmization:
        # Custom objective wrapper for constraints:
        pressure_axis_set = LinearObjectiveFromUser(
            fun = pressure_axis, 
            thing = eq_0,
            target = 1.8e4, # pressure = initial pressure on axis
            weight = 1.0,
        )
        pressure_edge_zero = LinearObjectiveFromUser(
            fun = pressure_edge,
            thing = eq_0,
            target = 0.0, # pressure = 0 on edge
            weight = 1.0,
        )
        grad_pressure_axis_zero = LinearObjectiveFromUser(
            fun = grad_pressure_axis, # Grad(P) = 0 on axis
            thing = eq_0,
            target = 0.0,
            weight = 1.0,
        )
        grad_pressure_edge_zero = LinearObjectiveFromUser(
            fun = grad_pressure_edge, # Grad(P) = 0 on edge
            thing = eq_0,
            target = 0.0,
            weight = 1.0,
        )
        # Combining constraints:
        constraints = (
            ForceBalance(eq = eq_0), # enforce JxB-grad(p)=0 during optimization
            FixIota(eq = eq_0), # fix rotational transform profile
            FixPsi(eq = eq_0), # fix total toroidal magnetic flux
            pressure_axis_set, #pressure = pressure initial on axis
            pressure_edge_zero, # pressure = 0 on edge
            grad_pressure_axis_zero, #grad(P) = 0 on axis
            grad_pressure_edge_zero, #grad(P) = 0 on edge
        )



        # Custom objective wrapper for monotonicity func:
        negative_gradient = ObjectiveFromUser( 
            fun = poly_monotonicity,
            grid = LinearGrid(rho=200,M=0,N=0),
            thing = eq_0,
            target = 0.0,
            weight = 1e2, # weight > other objective weights
            normalize = False,
        )
        # Crombining objectives:
        objective= ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e1), # J x B - Grad(P) = 0
            AspectRatio(eq=eq_0, target=6, weight=1e-1), # acceptable range: 
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=1e-2), #TARGET??? acceptable range: 
            negative_gradient, # monotonicity
        ])



    # Running the optimizer:
    eq_opt, opt_result = eq_0.optimize(
        objective = objective,
        constraints = constraints,
        optimizer = optimizer,
        ftol=5e-2,  # stopping tolerance on the function value
        xtol=1e-6,  # stopping tolerance on the step size
        gtol=1e-6,  # stopping tolerance on the gradient
        maxiter=50, # maximum number of iterations
        options={
            "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
            "solve_options": {
                "ftol": 5e-3,
                "xtol": 1e-6,
                "gtol": 1e-6,
                "verbose": 0,
            },  # for equilibrium subproblem
        },
        copy=False,  # copy=False we will overwrite the eq_0 object with the optimized result
        verbose=3,
    )



    # Derive save_path from load_path:
    dir_name = os.path.dirname(load_path) # directory same as eq.h5
    file_name_type = os.path.basename(load_path) # name of eq.h5, e.g. eq_p1e4_n1.h5
    file_name = os.path.splitext(file_name_type)[0] # stripping off file type e.g. eq_p1e4_n1
    if fix_pressure == True:
        save_path = os.path.join(dir_name, f'opt_{file_name[3:]}_FXP.h5')
    else: 
        save_path = os.path.join(dir_name, f'opt_{file_name[3:]}.h5')
    # Saving optimized equilibrium:
    eq_opt.save(save_path)

    # Returning optimized equilibrium objective values:
    return opt_result




