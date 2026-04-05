
import numpy as np
import jax.numpy as jnp
import math






#============== HELPER FUNCTION & VARIABLE ======================================================================================================================
#==========================
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
#==================================
#===================================================================================================================================










#============== PRESSURE CONSTRAINTS ================================================================================================
#=========================
def pressure_axis(params):
    """
    Pressure on axis (rho=0)
    Target: P(0)=1 (normalized)
    Target: P(0)=p_axis (defined max)
    """
    c = params['p_l']
    return c[0]
#=========================================


#=========================
def pressure_edge(params):
    """
    Pressure on edge (rho=1)
    Target: P(1)=0
    """
    c = params['p_l']
    return c.sum()
#=======================


#==============================
def grad_pressure_axis(params): 
    """
    Pressure gradient on axis (rho=0)
    Target: GradP=0 (no discontinuity)
    """
    c = params['p_l']
    return c[1]
#=============


#==============================
def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J)
    """ 
    grad_coeff = params['p_l'][1:]
    order = jnp.arange(1, len(grad_coeff)+1)
    return (order * grad_coeff).sum()
#====================================


#======================================
def pressure_monotonicity_generator(L):
    #---------------------------------
    def pressure_monotonicity(params):
        c = params["p_l"]
        if len(c) < 2*L + 1:
            raise ValueError(f"Need at least {2*L+1} pressure coefficients for L={L}")

        vals = []
        # Coeff of odd powers = 0:
        for i in range(1, 2*L + 1, 2):
            vals.append(c[i])
        # Coeff of even powers follow binomial pattern:
        for k in range(1, L + 1):
            vals.append(c[2*k] - ((-1)**k) * math.comb(L, k) * c[0])

        return jnp.array(vals)
    #-------------------------
    return pressure_monotonicity
#===============================
#===================================================================================================================================










#============== IOTA CONSTRAINTS ====================================================================================================
#=====================
def iota_axis(params):
    """
    Iota on axis (only constant term survives)
    P(rho=0) = d
    """
    c = params["i_l"] 
    return c[0]
#==============


#=====================
def iota_edge(params):
    """
    Iota on edge (b = 0 using grad_iota_axis).
    P(rho=1) = a + c = upper rational iota.
    """
    c = params["i_l"] 
    return c.sum() 
#=================


#==========================
def grad_iota_axis(params):
    """
    Iota gradient on axis (rho=0).
    Target: GradP=0 (no discontinuity).
    """
    c = params["i_l"] 
    return c[1]
#==============
#===================================================================================================================================










#============== PSI CONSTRAINTS ===================================================================================================
#=============

#===================
#===================================================================================================================================