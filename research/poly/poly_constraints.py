
import numpy as np
import jax.numpy as jnp






#============== HELPER FUNCTIONS ==============================================================================================================================
#--------------------------
def iota_between_rationals(
    iota_axis: float,
    iota_edge: float,
    ):
    """
    Generates bounds for iota optimizer constraint
    that are between low-order rational surfaces.
    """
    allowed_ranges = [
        (0.25,   0.3333),
        (0.3333, 0.5),
        (0.5,    0.6667),
        (0.6667, 0.75),
        (0.75,   1.0),
        (1.0,    1.3333),
        (1.3333, 1.5),
        (1.5,    2.0),
        (2.0,    3.0),
        (3.0,    4.0),
    ]

    matched_lower = None
    matched_upper = None

    for lower, upper in allowed_ranges:
        if lower <= iota_axis <= upper:
            matched_lower = lower
            matched_upper = upper
            break

    if matched_lower is None:
        raise ValueError("iota_axis is outside all allowed rational intervals.")

    if not (matched_lower <= iota_edge <= matched_upper):
        raise ValueError("iota_axis and iota_edge are not in the same allowed interval.")

    lower_bound = matched_lower
    upper_bound = matched_upper 

    return lower_bound, upper_bound
#-------------------------



















#============== PRESSURE CONSTRAINTS/OBJECTIVES =====================================================================================
#=============
# Constraints:
def pressure_axis(params):
    """
    Pressure on axis (rho=0)
    Target: P(0)=1 (normalized)
    Target: P(0)=p_axis (defined max)
    """
    c_0 = params['p_l'][0]
    return c_0 # only first coeff survives


def pressure_edge(params):
    """
    Pressure on edge (rho=1)
    Target: P(1)=0
    """
    p_coeff = params['p_l']
    return p_coeff.sum()


def grad_pressure_axis(params): 
    """
    Pressure gradient on axis (rho=0)
    Target: GradP=0 (no discontinuity)
    """
    c_1 = params['p_l'][1]
    return c_1


def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J)
    """ 
    grad_coeff = params['p_l'][1:]
    order = jnp.arange(1, len(grad_coeff)+1)
    return (order * grad_coeff).sum()
#====================================


#============
# Objectives:
def pressure_monotonicity(grid, data):
    """
    Ensures monotonic decrease of pressure for polynomial profile
    Parameters
    -----------
    grid: desc.grid.Grid
        grid object - only array of rho is used
    data: dict[str, ndarray]
        dictionary of optimizer outputs - only pressure array is used
    Returns
    -------
    jnp.max(violations): scalar
        largest dp over all grid points
    """
    p = data["p"] # pressure at grid points
    dp = p[1:] - p[:-1] # pressure differences: p[i+1] - p[i]
    violations = jnp.maximum(0.0, dp) # array where nonzero values = positive slope 
    return jnp.sum(violations**2) # largest dp chosen, penalized by optimizer
#============================================================================
#===================================================================================================================================










#============== IOTA CONSTRAINTS/OBJECTIVES ========================================================================================
#=============
# Constraints:
#--------------------------
def grad_iota_axis(params):
    """
    Iota gradient on axis (rho=0).
    Target: GradP=0 (no discontinuity).
    """
    c_1 = params['i_l'][1]
    return c_1
#-------------

#--------------------------
def iota_rational_edge(params):
    """
    Iota on edge (b = 0 using grad_iota_axis).
    P(rho=1) = a + c = upper rational iota.
    """
    c_0 = params["i_l"][0]
    c_2 = params["i_l"][2]
    return c_2 + c_0
#-------------------

#-------------------------------
def iota_rational_range(params):
    """
    Iota range.
    P(rho=1) - P(rho=0) = (a+c) - c = delta(iota)
    between rational iota.
    """
    c_2 = params["i_l"][2]
    return c_2
#-------------
#=====================



#============
# Objectives:
#-----------------------------
def iota_positive(grid, data):
    """
    Iota on axis greater than at least
    the first low-order rational surface
    at iota = 0.25.
    """
    iota_axis = data["iota"][0]
    violation = jnp.maximum(0.25 - iota_axis, 0.0)
    return violation
#====================
#===================================================================================================================================










#============== PSI CONSTRAINTS/OBJECTIVES =========================================================================================
#=============
# Constraints:
#===================


#============
# Objectives:
#===================
#===================================================================================================================================