import jax.numpy as jnp


#============== CORE ==============================================================================================================================
# No core custom functions currently live here.
#==================================================================================================================================================










#============== CUSTOM ============================================================================================================================
# Objectives:
#==========
# Pressure:
#-------------------------------
def pressure_axis_range(params):
    """
    Range of acceptable pressure on-axis.
    """
    return params["p_l"][0]


def pressure_shape(params):
    """
    Bounded coefficient of 8th-order term.
    Gives narrow-wideness of pressure.
    """
    return params["p_l"][4]
#--------------


# Iota:
#---------------------------
def iota_axis_range(params):
    """
    Allowable range of axis iota.
    """
    return params["i_l"][0]


def iota_edge_range(params):
    """
    Allowable range of edge iota.
    """
    return params["i_l"].sum()
#------------------
#==================




# Constraints:
#==========
# Pressure:
#--------------------------
def pressure_octic(params):
    """
    Limits the pressure profile to an 8th-order polynomial
    max, in order to prevent non-monotonicity and have only
    one objective on the pressure profile (pressure_shape).
    """
    c = params["p_l"]
    higher_orders = c[5:]
    return higher_orders


def pressure_DOF(params):
    """
    Constrains pressure coefficients so the pressure-shape
    degree of freedom stays on the smooth-profile relation.
    """
    c = params["p_l"]
    DOF_relation = c[3] + 3.45 * c[4]
    return DOF_relation


def pressure_edge(params):
    """
    Pressure on edge (rho=1).
    Target: P(1)=0.
    """
    return params["p_l"].sum()


def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1).
    Target: GradP=0 (no surface current J).
    """
    c = params["p_l"][1:]
    order = jnp.arange(1, len(c) + 1)
    return (order * c).sum()
#---------------------------


# Iota:
#-----------------
def iota_quadratic(params):
    """
    Limits iota to 2nd order polynomial.
    """
    c = params['i_l']
    higher_orders = c[2:]
    return higher_orders
#-----------------------
#=======================
#==================================================================================================================================================