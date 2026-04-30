# custom_funcs.py
#==============================================================================================================
#
# Shared FNO objective definitions for DESC/research/final.
#
# FNO functions take grid, data and are wrapped by ObjectiveFromUser.
#
#==============================================================================================================

import jax.numpy as jnp








#==============================================================================================================
# FNO Objectives:
#==============================================================================================================

def FNO_pressure_edge(
        grid,
        data,
    ):
    """
    Pressure is zero on edge.
    """
    p = data["p"][-1]
    return jnp.atleast_1d(p)


def FNO_pressure_positive(
        grid,
        data,
    ):
    """
    Maintain positive pressure.
    """
    p = data["p"]
    return p


def FNO_pressure_monotonic(
        grid,
        data,
    ):
    """
    Enforce monotonic pressure.
    """
    dp_dr = data["p_r"]
    return dp_dr


def FNO_grad_pressure_edge(
        grid,
        data,
    ):
    """
    Pressure gradient on edge equals zero.
    """
    dp_dr = data["p_r"][-1]
    return jnp.atleast_1d(dp_dr)