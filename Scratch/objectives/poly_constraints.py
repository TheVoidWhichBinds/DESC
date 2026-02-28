import numpy as np
import jax.numpy as jnp


#------------------------------------------
# Polynomial constraints:
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




#----------------------------------------------------------------
# Polynomial objective(s):
def poly_monotonicity(grid, data):
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
    dp = p[1:] - p[:-1] # pressure differences: p[i+1] - p[i]
    violations = jnp.maximum(0.0, dp) # array where nonzero values = positive slope 
    return jnp.sum(violations**2) # largest dp chosen, penalized by optimizer


