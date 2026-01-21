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
    e = len(params) // 2
    p_edge = params['p_l'][e]
    return p_edge # pressure on edge


def grad_pressure_axis(params): 
    """
    Pressure gradient on axis (rho=0)
    Target: GradP=0 (no discontinuity)
    """
    e = len(params) // 2
    gradp_axis = params['p_l'][e+1]
    return gradp_axis


def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J)
    """ 
    gradp_edge = params['p_l'][-1]
    return gradp_edge


def hermite_monotonicity(grid, data):
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
    rho = grid.nodes[:, 0] # rho gridpoints
    gradp = p[1::2] # gradient of pressure: 0,2,4... indices are pressure at each node, 1,3,5... indices are grad p
    violations = jnp.maximum(0.0, gradp) # array where nonzero values = positive slope 
    return jnp.max(violations) # largest dp chosen, penalized by optimizer


