import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
import desc.io
from eq import run_equilibrium
from opt import run_optimizer
import os 


#-----------------------------------------------------------------------
# Generating set of initial conditions to loop over:
p_maxima = [1e4,1e5] # on-axis pressures to feed into initial eq solver
n_set = [1,2,3] # polynomials will be generated of degree 2*n
for p_scale in p_maxima:
    for n in n_set:
        eq_result = run_equilibrium(p_scale, n) # solving initial equilibrium
        
        # Getting file name of initial eq solve for optimizer:
        dir_name = os.path.dirname(os.path.abspath(__file__))
        load_path = os.path.join(dir_name, f'eq_p{p_scale:.0e}_n{n}.h5')
        
        # Running optimizer with fixed and optimized pressure:
        for fix_pressure in [True, False]:
            if fix_pressure == True: 
                opt_result_FXD = run_optimizer(load_path, fix_pressure) 
            else:
                opt_result = run_optimizer(load_path, fix_pressure) 



        # Pre & post optimization pressure profile plotting
        # loading 
        # rho = np.linspace(0, 1, 400)
        # grid = LinearGrid(rho=rho, M=0, N=0)   
        # pressure_eq = eq_.compute('p', grid=grid)['p']
        # pressure_opt = .compute('p', grid=grid)['p']
        # #
        # plt.figure(figsize=(7,5))

        # plt.plot(rho, pressure_init, linewidth=2, color='g',
        #     path_effects=[
        #         pe.Stroke(linewidth=6, foreground='lightgreen'),
        #         pe.Normal()
        #     ],
        #     label='initial solved equilibrium'
        # )
        # plt.plot(rho, pressure_opt, linewidth=2, color='r', label='optimized')

        # plt.xlabel(r"$\rho$", fontsize=14)
        # plt.ylabel(r"$Pressure$", fontsize=14)
        # plt.title("Pressure Evolution", fontsize=16)
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('/Users/macdaddi/DESC/scratch/runs/poly/pressure.png')
