import jax.numpy as jnp


#============== MISC ===============================================================================================================================
#============
# Shared by more than one optimization mode:
#============

def pressure_shape(params):
    """
    Bounded coefficient of 8th-order term.
    Gives narrow-wideness of pressure.
    """
    c = params["p_l"]
    return c[4]
#==================================================================================================================================================




#============== PROXIMAL_PARAMS ====================================================================================================================
#============
# Custom objectives built from fun(params):
#============

#--------------
# Pressure:
def pressure_axis_range_params(params):
    """
    Allowable range of max pressures.
    """
    c = params["p_l"]
    return c[0]


#---------------------------
# Rotational transform:
def iota_axis_range_params(params):
    """
    Allowable range of axis iota.
    """
    return params["i_l"][0]


def iota_edge_range_params(params):
    """
    Allowable range of edge iota.
    """
    return params["i_l"].sum()
#==================================================================================================================================================




#============== PROXIMAL_DATA ======================================================================================================================
#============
# Custom objectives built from fun(grid, data):
#============

#--------------
# Pressure:
def pressure_axis_range_data(grid, data):
    """
    Allowable range of max pressures,
    evaluated from profile data on the grid.
    """
    p = data["p"]
    return p[0]


def pressure_edge_data(grid, data):
    """
    Pressure on edge (rho=1).
    Target: P(1)=0.
    """
    p = data["p"]
    return p[-1]


def grad_pressure_edge_data(grid, data):
    """
    Pressure gradient on edge (rho=1).
    Target: GradP=0 (no surface current J).
    """
    dp_dr = data["p_r"]
    return dp_dr[-1]


#---------------------------
# Rotational transform:
def iota_axis_range_data(grid, data):
    """
    Allowable range of axis iota,
    evaluated from profile data on the grid.
    """
    iota = data["iota"]
    return iota[0]


def iota_edge_range_data(grid, data):
    """
    Allowable range of edge iota,
    evaluated from profile data on the grid.
    """
    iota = data["iota"]
    return iota[-1]
#==================================================================================================================================================




#============== AUGLAG =============================================================================================================================
#============
# Custom constraints built from fun(params):
#============

#--------------
# Pressure:
def pressure_axis_fxd(params):
    """
    Auglag constraint that keeps the max
    pressure on-axis constant.
    """
    c = params["p_l"]
    return c[0]


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


def pressure_edge_params(params):
    """
    Pressure on edge (rho=1).
    Target: P(1)=0.
    """
    c = params["p_l"]
    return c.sum()


def grad_pressure_edge_params(params):
    """
    Pressure gradient on edge (rho=1).
    Target: GradP=0 (no surface current J).
    """
    c = params["p_l"][1:]
    order = jnp.arange(1, len(c) + 1)
    return (order * c).sum()
#==================================================================================================================================================