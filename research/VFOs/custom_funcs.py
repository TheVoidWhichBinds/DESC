# custom_funcs.py
#===================================================================================================================================================

import jax.numpy as jnp

#===================================================================================================================================================










#===========================================================
# FLO OBJECTIVES
#===========================================================

def FLO_pressure_axis(
        params,
    ):
    """
    Range of acceptable pressure on-axis.
    Bounds = (1E4, 1E7).
    """

    return params["p_l"][0]





def FLO_pressure_shape(
        params,
    ):
    """
    Bounded coefficient of 8th-order term.
    Gives narrow-wideness of pressure.
    Bounds = (-1.3, 1.8).
    """

    return params["p_l"][4]

#===================================================================================================================================================










#===========================================================
# FLO PRESSURE CONSTRAINTS
#===========================================================

def FLO_pressure_octic(
        params,
    ):
    """
    Limits the pressure profile to an 8th-order polynomial.
    Target = 0.
    """

    c = params["p_l"]
    higher_orders = c[5:]

    return higher_orders





def FLO_pressure_DOF(
        params,
    ):
    """
    Constrains pressure coefficients so the pressure-shape
    degree of freedom stays on the smooth-profile relation.
    Target = 0.
    """

    c = params["p_l"]
    DOF_relation = c[3] + 3.45 * c[4]

    return DOF_relation





def FLO_pressure_edge(
        params,
    ):
    """
    Pressure on edge.
    Target = 0.
    """

    return params["p_l"].sum()





def FLO_grad_pressure_edge(
        params,
    ):
    """
    Pressure gradient on edge.
    Target = 0.
    """

    c = params["p_l"][1:]
    order = jnp.arange(1, len(c) + 1)

    return (order * c).sum()

#===================================================================================================================================================










#===========================================================
# FLO IOTA CONSTRAINTS
#===========================================================

def FLO_iota_axis(
        params,
    ):
    """
    Allowable range of axis iota.
    Bounds = (lower rational, upper rational).
    """

    return params["i_l"][0]





def FLO_iota_edge(
        params,
    ):
    """
    Allowable range of edge iota.
    Bounds = (lower rational, upper rational).
    """

    return params["i_l"].sum()





def FLO_iota_quadratic(
        params,
    ):
    """
    Limits iota to a 2nd-order polynomial.
    Target = 0.
    """

    c = params["i_l"]
    higher_orders = c[2:]

    return higher_orders

#===================================================================================================================================================










#===========================================================
# FNO PRESSURE OBJECTIVES
#===========================================================

def FNO_pressure_axis(
        grid,
        data,
    ):
    """
    Range of values of pressure on-axis.
    Bounds = (1E4, 1E7).
    """

    p = data["p"][0]

    return p





def FNO_pressure_edge(
        grid,
        data,
    ):
    """
    Pressure is zero on-edge.
    Target = 0.
    """

    p = data["p"][-1]

    return p





def FNO_pressure_positive(
        grid,
        data,
    ):
    """
    Maintains positive pressure.
    Bounds = (0, inf).
    """

    p = data["p"]

    return p





def FNO_pressure_monotonic(
        grid,
        data,
    ):
    """
    Enforces monotonic pressure.
    Bounds = (-inf, 0).
    """

    dp_dr = data["p_r"]

    return dp_dr





def FNO_grad_pressure_edge(
        grid,
        data,
    ):
    """
    Pressure gradient on edge.
    Target = 0.
    """

    dp_dr = data["p_r"][-1]

    return dp_dr

#===================================================================================================================================================










#===========================================================
# FNO IOTA OBJECTIVES
#===========================================================

def FNO_iota(
        grid,
        data,
    ):
    """
    Keeps iota from crossing rational surfaces.
    Bounds = (lower rational, upper rational).
    """

    iota = data["iota"]

    return iota

#===================================================================================================================================================