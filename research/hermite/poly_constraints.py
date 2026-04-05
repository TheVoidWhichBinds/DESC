
import numpy as np
import jax.numpy as jnp
import math










#============== PRESSURE =============================================================================================================
# Constraints:
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
        print(f"# of stored coeff: {len(c)}")

        if len(c) < 2*L + 1:
            raise ValueError(f"Need at least {2*L+1} pressure coefficients for L={L}")

        vals = []

        # Coefficients of odd powers must vanish
        for i in range(1, 2*L + 1, 2):
            vals.append(c[i])

        # Even coefficients must follow binomial pattern
        for k in range(1, L + 1):
            vals.append(c[2*k] - ((-1)**k) * math.comb(L, k) * c[0])

        return jnp.array(vals)
    #-------------------------
    return pressure_monotonicity
#===============================
#==================================




#============
# Objectives:
#===================================
def pressure_axis_range(grid, data):
    """
    Allows pressure on-axis to be within specified bounds.
    """
    p_axis = data['p'][0]
    return p_axis
#================
#===================
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