#FIXED BOUNDARY QS OPTIMIZATION MODDED TO TAKE 
#LOGISTIC FUNCTION FAMILY WEIGHTS AS OPTIMIZATION VARIABLES
#NOTES:
#play with lowering ftol


#Imports:
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/macdaddi/DESC")
import desc.io
from desc.grid import LinearGrid
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
eq_init_fixbound= desc.io.load("scratch/runs/logistic/eq.h5") #initial equilibrium load - might need to generalize path
eq_init= eq_init_fixbound[-1].copy() #copy of final eq in family




#-----------------------------------------------------
#Objectives of Choice:
objective= ObjectiveFunction(ForceBalance(eq=eq_init))


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
    gtol=1e-7,  # stopping tolerance on the gradient
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



# Autosaving optimized equilibrium:
eq_opt.save('/Users/macdaddi/DESC/scratch/runs/logistic/opt.h5')



#------ Optimized Pressure Profile Plotting ------#
rho = np.linspace(0, 1, 400)
grid = LinearGrid(rho=rho, M=0, N=0)   
data = eq_opt.compute("p", grid=grid)
P = data["p"]
#
plt.figure(figsize=(7,5))
plt.plot(rho, P, linewidth=2)
plt.xlabel(r"$\rho$", fontsize=14)
plt.ylabel(r"$P(\rho)$", fontsize=14)
plt.title("Post-Optimization Pressure", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('/Users/macdaddi/DESC/scratch/runs/logistic/pressure.png')
print(eq_opt.pressure.params)