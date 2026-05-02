# custom_funcs.py
#==============================================================================================================
#
# Shared FREE objective definitions for DESC/research/lit_comp.
#
# Params-based FREE functions are wrapped by LinearObjectiveFromUser.
# Grid/data-based FREE functions are wrapped by ObjectiveFromUser.
#
#==============================================================================================================

import jax.numpy as jnp








#==============================================================================================================
# FREE Linear Constraints:
#==============================================================================================================

def FREE_pressure_axis(
        params,
    ):
    """
    Pressure on-axis.

    This is constrained to the value from the initial equilibrium immediately before
    optimization.
    """

    return jnp.atleast_1d(params["p_l"][0])





def FREE_pressure_edge(
        params,
    ):
    """
    Pressure at rho = 1.

    For a PowerSeriesProfile this is the sum of pressure coefficients.
    """

    p_l = params["p_l"]

    return jnp.atleast_1d(jnp.sum(p_l))





def FREE_grad_pressure_edge(
        params,
    ):
    """
    Radial pressure gradient at rho = 1.

    For a PowerSeriesProfile p(rho) = sum_l p_l rho^l, so
    dp/drho at rho = 1 is sum_l l p_l.
    """

    p_l = params["p_l"]
    powers = jnp.arange(p_l.size)

    return jnp.atleast_1d(jnp.sum(powers * p_l))










#==============================================================================================================
# FREE Nonlinear Objectives:
#==============================================================================================================

def FREE_pressure_positive(
        grid,
        data,
    ):
    """
    Maintain positive pressure.
    """

    return data["p"]





def FREE_pressure_monotonic(
        grid,
        data,
    ):
    """
    Enforce monotonic pressure.
    """

    return data["p_r"]