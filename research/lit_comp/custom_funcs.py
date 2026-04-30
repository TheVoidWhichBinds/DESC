# VFOs.py
#==============================================================================================================
#
# Shared variant-function-objective definitions for DESC/research/final.
#
# This file contains all FLO_* and FNO_* functions.
#
# Naming convention:
#   FLO_* functions are used only by FLO.py.
#   FNO_* functions are used only by FNO.py.
#
# FLO functions take params and are wrapped by LinearObjectiveFromUser.
# FNO functions take grid, data and are wrapped by ObjectiveFromUser.
#
#==============================================================================================================

import jax.numpy as jnp








#==============================================================================================================
# FLO Objectives:
#==============================================================================================================

def FLO_pressure_axis(
        params,
    ):
    """
    Range of acceptable pressure on-axis.
    """

    return jnp.atleast_1d(params["p_l"][0])


def FLO_pressure_shape(
        params,
    ):
    """
    Bounded coefficient of 8th-order pressure term.
    """

    return jnp.atleast_1d(params["p_l"][4])









#==============================================================================================================
# FLO Constraints
#==============================================================================================================

def FLO_pressure_octic(
        params,
    ):
    """
    Limits the pressure profile to an 8th-order polynomial max.
    """
    c = params["p_l"]
    higher_orders = c[5:]
    return higher_orders


def FLO_pressure_DOF(
        params,
    ):
    """
    Constrains pressure coefficients so the pressure-shape degree of freedom
    stays on the smooth-profile relation.
    """
    c = params["p_l"]
    DOF_relation = c[3] + 3.45 * c[4]
    return jnp.atleast_1d(DOF_relation)


def FLO_pressure_edge(
        params,
    ):
    """
    Pressure on edge, rho = 1.
    """
    return jnp.atleast_1d(params["p_l"].sum())


def FLO_grad_pressure_edge(
        params,
    ):
    """
    Pressure gradient on edge, rho = 1.
    """
    c = params["p_l"][1:]
    order = jnp.arange(1, len(c) + 1)
    return jnp.atleast_1d((order * c).sum())


def FLO_iota_axis(
        params,
    ):
    """
    Allowable range of axis iota.
    """
    return jnp.atleast_1d(params["i_l"][0])


def FLO_iota_edge(
        params,
    ):
    """
    Allowable range of edge iota.
    """
    return jnp.atleast_1d(params["i_l"].sum())


def FLO_iota_quadratic(
        params,
    ):
    """
    Limits iota to a second-order polynomial.
    """
    c = params["i_l"]
    higher_orders = c[2:]
    return higher_orders










#==============================================================================================================
# FNO Objectives:
#==============================================================================================================

def FNO_pressure_axis(
        grid,
        data,
    ):
    """
    Range of values of pressure on-axis.
    """
    p = data["p"][0]
    return jnp.atleast_1d(p)


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


def FNO_iota(
        grid,
        data,
    ):
    """
    Keeps iota from crossing rational surfaces.
    """
    iota = data["iota"]
    return iota
