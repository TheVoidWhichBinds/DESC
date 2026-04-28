
import jax.numpy as jnp











#============== PRESSURE =============================================================================================================
#============
# Objectives:
#===============================
def pressure_axis_range(params):
    """
    Fixes the axis pressure.
    """
    c = params['p_l']
    p_axis = c[0] 
    return p_axis
#================

#==========================
def pressure_shape(params):
    """
    Places bounds on coefficient of the 8th-order term.
    In conjunction with the constraints, creates a good
    profile.
    """
    c = params['p_l']
    return c[4] 
#============================




#=============
# Constraints:
#==========================
def pressure_octic(params):
    """
    Limits the pressure profile to an 8th-order polynomial 
    max, in order toprevent non-monotonicity and have only 
    one objective on the pressure profile (pressure_shape)
    """
    c = params['p_l']
    higher_orders = c[5:]
    return higher_orders
#=======================

#========================
def pressure_DOF(params):
    """
    P = 1 + ... + c[4]x^6 + c[5]x^8. Constrains c[4] and 
    c[5] to be related in such a way that at the bounds of 
    c[5] (enforced by pressure_ ) produce smooth profiles 
    (relation is at the threshold of generating a saddle-point
    inside rho = (0,1) at the bounds of c[5]).
    """
    c = params['p_l']
    DOF_relation = c[3] + 3.45*c[4]
    return DOF_relation
#======================

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
def grad_pressure_edge(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J)
    """ 
    c = params['p_l'][1:]
    order = jnp.arange(1, len(c)+1)
    return (order * c).sum()
#====================================
#==================================
#===================================================================================================================================










#============== ROTATIONAL TRANSFORM ===================================================================================================
#============
# Objectives:
#===========================
def iota_axis_range(params):
    iota_axis = params['i_l'][0]
    return iota_axis
#===================

#===========================
def iota_edge_range(params):
    iota_edge = params['i_l'].sum()
    return iota_edge
#===================
#===================




#=============
# Constraints:
#============================
#===================================================================================================================================


