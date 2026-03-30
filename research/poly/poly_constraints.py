
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
#------------------------------
def iota_rationals(grid, data):
    """
    """
    iota_values = data["iota"]
    return iota_values
#---------------------

#--------------------------
def grad_iota_axis(params):
    """
    """
    c_1 = params['i_l'][1]
    return c_1
#-------------
#=====================



#============
# Objectives:

#=====================
#===================================================================================================================================










#============== PSI CONSTRAINTS/OBJECTIVES =========================================================================================
#=============
# Constraints:
#===================


#============
# Objectives:
#===================
#===================================================================================================================================