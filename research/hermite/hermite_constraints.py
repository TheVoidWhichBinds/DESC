# Revamp
import numpy as np
import jax.numpy as jnp


#------------- Hermite Constraints -------------#
def pressure_axis(params):
    """
    Pressure on axis (rho=0)
    Target: P(0)=1 (normalized)
    """
    p_axis = params['p_l'][0]
    return p_axis # pressure on axis


def pressure_edge(params):
    """
    Pressure on edge (rho=1)
    Target: P(1)=0
    """
    p_edge = params['p_l'][len(params)//2]
    return p_edge # pressure on edge


def grad_pressure_axis(params): 
    """
    Pressure gradient on axis (rho=0)
    Target: GradP=0 (no discontinuity)
    """
    gradp_axis = params['p_l'][len(params)//2 + 1]
    return gradp_axis


def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J)
    """ 
    gradp_edge = params['p_l'][-1]
    return gradp_edge




#--------- Hermite Objectives for Monotonicity ---------#
def hermite_monotonicity_pressure(grid, data):
    """
    Ensures monotonic decrease of pressure
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
    p = data['p'] # gradient of pressure at grid points
    rho = grid.nodes[:, 0] # rho gridpoints
    dp = p[1:] - p[0:-1] # upper limit on change between two points
    violation = jnp.maximum(0, dp) # array where nonzero values = positive slope 
    return jnp.max(violation) # largest dp_dr chosen, penalized by optimizer



def hermite_monotonicity_slope(grid, data):
    """
    Ensures monotonic decrease of pressure for hermite cubic spline profile
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
    dp_dr = data['p_r'] # gradient of pressure at grid points
    rho = grid.nodes[:, 0] # rho gridpoints
    violation = jnp.maximum(0, dp_dr) # array where nonzero values = positive slope 
    return jnp.max(violation) # largest dp_dr chosen, penalized by optimizer


