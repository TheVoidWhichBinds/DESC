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
    LinearObjectiveFromUser,
    ObjectiveFromUser,
)

from desc.optimize import Optimizer

optimizer = Optimizer("proximal-fmintr-bfgs")


def run_optimization(p_scale, load_path, out_dir, fix_pressure: bool):
    os.makedirs(out_dir, exist_ok=True)

    eq_init = desc.io.load(load_path)
    eq_0 = eq_init.copy()

    if fix_pressure:
        constraints = (
            ForceBalance(eq=eq_0),
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
            FixPressure(eq=eq_0),
        )

        objective = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e1),
            AspectRatio(eq=eq_0, target=6, weight=1e-1),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=1e-2),
        ])

    else:
        pressure_axis_set = LinearObjectiveFromUser(
            fun=pressure_axis,
            thing=eq_0,
            target=p_scale,
            weight=1.0,
        )
        pressure_edge_zero = LinearObjectiveFromUser(
            fun=pressure_edge,
            thing=eq_0,
            target=1e-12,
            weight=1.0,
        )
        grad_pressure_axis_zero = LinearObjectiveFromUser(
            fun=grad_pressure_axis,
            thing=eq_0,
            target=1e-12,
            weight=1.0,
        )
        grad_pressure_edge_zero = LinearObjectiveFromUser(
            fun=grad_pressure_edge,
            thing=eq_0,
            target=1e-12,
            weight=1.0,
        )

        constraints = (
            ForceBalance(eq=eq_0),
            FixIota(eq=eq_0),
            FixPsi(eq=eq_0),
            pressure_axis_set,
            pressure_edge_zero,
            grad_pressure_axis_zero,
            grad_pressure_edge_zero,
        )

        negative_gradient = ObjectiveFromUser(
            fun=poly_monotonicity,
            grid=LinearGrid(rho=200, M=0, N=0),
            thing=eq_0,
            target=1e-12,
            weight=1e2,
            normalize=False,
        )

        objective = ObjectiveFunction([
            ForceBalance(eq=eq_0, target=0, weight=1e2),
            AspectRatio(eq=eq_0, target=6, weight=1e1),
            QuasisymmetryBoozer(eq=eq_0, helicity=(1, eq_0.NFP), weight=1e1),
            negative_gradient,
        ])

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
            "solve_options": {"ftol": 5e-4, "xtol": 1e-6, "gtol": 1e-7, "verbose": 0},
        },
        copy=False,
        verbose=3,
    )

    save_name = "opt_FXP.h5" if fix_pressure else "opt.h5"
    save_path = os.path.join(out_dir, save_name)
    eq_opt.save(save_path)

    return eq_opt, opt_result, save_path
