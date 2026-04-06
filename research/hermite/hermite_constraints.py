import jax.numpy as jnp


#============== PRESSURE ============================================================================================================
#===========================
# Constraints:
#=========================
def pressure_axis(params):
    """
    Pressure on axis (rho=0)
    Target: P(0)=1 (normalized)
    """
    p = params["p_l"]
    p_axis = p[0]
    return p_axis # pressure on axis
#===================================


#=========================
def pressure_edge(params):
    """
    Pressure on edge (rho=1)
    Target: P(1)=0
    """
    p = params["p_l"]
    n = len(p) // 2
    p_edge = p[n - 1]
    return p_edge # pressure on edge
#===================================


#==============================
def grad_pressure_axis(params): 
    """
    Pressure gradient on axis (rho=0)
    Target: GradP=0 (no discontinuity)
    """
    p = params["p_l"]
    n = len(p) // 2
    gradp_axis = p[n]
    return gradp_axis
#====================


#==============================
def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J)
    """ 
    p = params["p_l"]
    gradp_edge = p[-1]
    return gradp_edge
#====================
#=======================




#============
# Objectives:
#===================================
def pressure_axis_range(grid, data):
    """
    Allows pressure on-axis to be within specified bounds.
    """
    p_axis = data["p"][0]
    return p_axis
#================


#==========================================
def pressure_monotonicity(grid, data):
    """
    Ensures monotonic decrease of pressure for hermite cubic spline profile
    """
    dp_dr = data["p_r"] # gradient of pressure at grid points
    return dp_dr
#===============
#==================
#================================================================================================================================









#============== IOTA  ====================================================================================================
#=============
# Constraints:
#=====================
def iota_axis(params):
    """
    Iota on axis.
    For Hermite basis, first half of i_l stores knot values.
    Axis is the first knot value.
    """
    c = params["i_l"]
    f = c[:(len(c)//2)]
    return f[0]
#==============


#=====================
def iota_edge(params):
    """
    Iota on edge.
    For Hermite basis, edge is the last knot value.
    """
    c = params["i_l"]
    f = c[:(len(c)//2)]
    return f[-1]
#=================


#==========================
def grad_iota_axis(params):
    """
    Iota gradient on axis.
    For Hermite basis, second half of i_l stores knot derivatives.
    Axis gradient is the first derivative entry.
    """
    c = params["i_l"]
    df = c[(len(c)//2):]
    return df[0]
#==============
#=================
#===================================================================================================================================
