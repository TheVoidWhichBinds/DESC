# NOTES: 
# Change target values from 0.0 to 1e-10

import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
from eq import run_equilibrium
from opt import run_optimization
import os 
import pandas as pd
from tabulate import tabulate


#-----------------------------------------------------------------------
# Generating set of initial conditions to loop over:
p_maxima = [1e4] # on-axis pressures to feed into initial eq solver
n_set = [2] # polynomials will be generated of degree 2*n

# Generating figure for comparison of objective values:
# NOT SURE WHAT KIND OF FIGURE TO USE TO SHOW DIFFERENCE


# Prepping for table generation that compares fixed and optimized 
# pressure for different max pressure and polynomial orders:
import pandas as pd
import os
from tabulate import tabulate

# Exact objective keys from DESC
columns = [
    'Force error: ',
    'Quasi-symmetry (1,19) Boozer error: ',
    'Fixed iota profile error: ',
    'Fixed Psi error: '
]

rows = []
values = []

for p_scale in p_maxima:
    for n in n_set:
        # Solve initial equilibrium
        run_equilibrium(p_scale, n)

        dir_name = os.path.dirname(os.path.abspath(__file__))
        load_path = os.path.join(dir_name, f'eq_p{p_scale:.0e}_n{n}.h5')

        # Run optimizer
        opt_result_FXD = run_optimization(load_path, fix_pressure=True)
        opt_result = run_optimization(load_path, fix_pressure=False)

        # Row labels
        group_label = f"Max Pressure = {p_scale}; order = {2*n}"
        rows.append((group_label, "Fixed Pressure"))
        rows.append((group_label, "Optimized Pressure"))

        # Take the Objective values exactly as stored
        row_FXD = [opt_result_FXD['Objective values'][key] for key in columns]
        row_OPT = [opt_result['Objective values'][key] for key in columns]

        values.append(row_FXD)
        values.append(row_OPT)

# Create MultiIndex
index = pd.MultiIndex.from_tuples(rows, names=["Run", "Pressure Type"])

# Build DataFrame
df = pd.DataFrame(values, index=index, columns=[col.replace(': ', '') for col in columns])

# Convert DataFrame to ASCII table with grid lines
ascii_table = tabulate(df, headers='keys', tablefmt='grid')

# Save to same folder as this script
output_file = os.path.join(dir_name, "optimization_comparison.txt")
with open(output_file, "w") as f:
    f.write("Comparison of Post-Optimization Objectives (raw Objective values)\n")
    f.write(ascii_table)

print(f"Table saved to {output_file}")









        
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
