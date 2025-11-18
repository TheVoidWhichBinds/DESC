#FIXED BOUNDARY QS OPTIMIZATION MODDED TO TAKE 
#LOGISTIC FUNCTION FAMILY WEIGHTS AS OPTIMIZATION VARIABLES
#NOTES:
#play with lowering ftol


#Imports:
import numpy as np
import matplotlib.pyplot as plt
import desc.io
from desc.objectives import (
    ObjectiveFunction,
   #FixPressure, #skipping this line
    FixIota,
    FixPsi,
    ForceBalance,
    LinearObjectiveFromUser,
)
from desc.optimize import Optimizer


#Loading Equilibria Family and Making a Copy of Final Iteration:
eq_init_fixbound= desc.io.load("scratch/runs/logistic/eq_init_fixbound.h5") #initial equilibrium load - might need to generalize path
eq_init= eq_init_fixbound[-1].copy() #copy of final eq in family




#-----------------------------------------------------
#Objective of Choice:
objective= ObjectiveFunction(ForceBalance(eq=eq_init))

#Importing Custom Constraint Funcs:
from scratch.objectives.constraints import ( 
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge
)

#Constructing the Objective Wrappers for Custom Func:
pressure_edge_zero = LinearObjectiveFromUser(
    fun=pressure_edge,
    thing=eq_init,
    target=0.0,
    weight=1.0,
)
grad_pressure_axis_zero = LinearObjectiveFromUser(
    fun=grad_pressure_axis,
    thing=eq_init,
    target=0.0,
    weight=1.0,
)
grad_pressure_edge_zero = LinearObjectiveFromUser(
    fun=grad_pressure_edge,
    thing=eq_init,
    target=0.0,
    weight=1.0,
)

#List of Constraints:
constraints = (
    ForceBalance(eq=eq_init),  # enforce JxB-grad(p)=0 during optimization
   #FixPressure(eq=eq_init),  # fix pressure profile - OMITTED
    FixIota(eq=eq_init),  # fix rotational transform profile
    FixPsi(eq=eq_init),  # fix total toroidal magnetic flux
)

#Optimizer of Choice:
optimizer = Optimizer("proximal-lsq-exact") #choice of optimizer




#----------------------------------
#Running the Optimizer:
eq_opt, result = eq_init.optimize(
    objective=objective,
    constraints=constraints,
    optimizer=optimizer,
    ftol=5e-2,  # stopping tolerance on the function value
    xtol=1e-6,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    maxiter=50,  # maximum number of iterations
    options={
        "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
        "solve_options": {
            "ftol": 5e-3,
            "xtol": 1e-6,
            "gtol": 1e-6,
            "verbose": 0,
        },  # for equilibrium subproblem
    },
    copy=False,  # copy=False we will overwrite the eq_init object with the optimized result
    verbose=3,
)