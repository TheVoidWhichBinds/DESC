#BASIS FIXED BOUNDARY QS OPTIMIZATION MODDED TO TAKE POLYNOMIAL COEFFICIENTS AS OPTIMIZATION VARIABLES AND PRESSURE CONSTRAINTS AS ADDITIONAL OBJECTIVES
#NOTES:
#play with lowering ftol




#---------- Imports & Loads ----------#
import numpy as np
import matplotlib.patheffects as pe
import sys
sys.path.append("/Users/macdaddi/DESC")
import matplotlib.pyplot as plt
import desc.io
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    FixIota,
    FixPsi,
    ForceBalance,
    AspectRatio,
    QuasisymmetryBoozer,
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)
from desc.optimize import Optimizer


#Loading Equilibria Family and Making a Copy of Final Iteration:
eq_init_fixbound = desc.io.load("scratch/runs/poly/eq.h5") #initial equilibrium load - might need to generalize path
eq_init = eq_init_fixbound[-1].copy() #copy of final eq in family



#-----------------------------------------------
# Importing custom constraint funcs:
from scratch.objectives.poly_constraints import( 
    pressure_axis,
    pressure_edge,
    grad_pressure_axis,
    grad_pressure_edge,
    poly_monotonicity
)

# Custom objective wrapper for constraints:
pressure_axis_normalized = LinearObjectiveFromUser(
    fun=pressure_axis, 
    thing=eq_init,
    target=1.0, # pressure = 1 on axis
    weight=1.0,
)
pressure_edge_zero = LinearObjectiveFromUser(
    fun=pressure_edge,
    thing=eq_init,
    target=0.0, # pressure = 0 on edge
    weight=1.0,
)
grad_pressure_axis_zero = LinearObjectiveFromUser(
    fun=grad_pressure_axis, # Grad(P) = 0 on axis
    thing=eq_init,
    target=0.0,
    weight=1.0,
)
grad_pressure_edge_zero = LinearObjectiveFromUser(
    fun=grad_pressure_edge, # Grad(P) = 0 on edge
    thing=eq_init,
    target=0.0,
    weight=1.0,
)

#List of Constraints: EITHER FixPressure OR 4 Pressure Constraints Active
constraints = (
    ForceBalance(eq=eq_init),  # enforce JxB-grad(p)=0 during optimization
    FixIota(eq=eq_init),  # fix rotational transform profile
    FixPsi(eq=eq_init),  # fix total toroidal magnetic flux
    pressure_axis_normalized, #pressure = 1 on axis
    pressure_edge_zero, # pressure = 0 on edge
    grad_pressure_axis_zero, #grad(P) = 0 on axis
    grad_pressure_edge_zero, #grad(P) = 0 on edge
)



#------------------------------------------------
# Custom objective wrapper for monotonicity func:
negative_gradient = ObjectiveFromUser( 
    fun=poly_monotonicity,
    grid=LinearGrid(rho=200,M=0,N=0),
    thing=eq_init,
    target=0.0,
    weight=1e2, # weight > other objective weights
    normalize=False,
)

# Creating objective:
objective= ObjectiveFunction([
    ForceBalance(eq=eq_init, target=0, weight=1e1), # J x B - Grad(P) = 0
    AspectRatio(eq=eq_init, target=6, weight=1e-1), # acceptable range: 
    QuasisymmetryBoozer(eq=eq_init, helicity=(1, eq_init.NFP), weight=1e-2), #TARGET??? acceptable range: 
    negative_gradient, # monotonicity
    ])



#---------------------------------------------------------------
#Optimizer of Choice:
optimizer = Optimizer("proximal-lsq-exact") #choice of optimizer



#---------------- Running the Optimizer -----------------#
eq_opt, result = eq_init.optimize(
    objective=objective,
    constraints=constraints,
    optimizer=optimizer,
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
    copy=False,  # copy=False we will overwrite the eq_init object with the optimized result
    verbose=3,
)

# Autosaving optimized equilibrium:
eq_opt.save('/Users/macdaddi/DESC/scratch/runs/poly/opt.h5')




#------ Pre & Post Optimization Pressure Profile Plotting ------#
rho = np.linspace(0, 1, 400)
grid = LinearGrid(rho=rho, M=0, N=0)   
pressure_init = eq_init.compute('p', grid=grid)['p']
pressure_opt = eq_opt.compute('p', grid=grid)['p']
#
plt.figure(figsize=(7,5))
plt.plot(rho, pressure_init, linewidth=2, color='g',
    path_effects=[
        pe.Stroke(linewidth=6, foreground='lightgreen'),
        pe.Normal()
    ],
    label='init'
)
plt.plot(rho, 1.8e4 -3.6e4*rho**2 + 1e2*rho**3 + 1.8e4*rho**4, color='y')
plt.plot(rho, pressure_opt, linewidth=2, color='r', label='opt')
plt.xlabel(r"$\rho$", fontsize=14)
plt.ylabel(r"$Pressure$", fontsize=14)
plt.title("Post-Optimization Pressure", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('/Users/macdaddi/DESC/scratch/runs/poly/pressure.png')
