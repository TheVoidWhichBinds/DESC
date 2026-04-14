
import jax.numpy as jnp











#============== PRESSURE =============================================================================================================
#============
# Objectives:
#===============================
def pressure_axis_range(params):
    """
    Allowable range of max pressures
    """
    c = params['p_l']
    p_axis = c[0] 
    return p_axis
#================

#=========================================
def pressure_monotone_obj(grid, data):
    """

    """
    dp_dr = data['p_r']
    return jnp.maximum(0.0, dp_dr)
#=================================

#=====================================
def pressure_positive_obj(grid, data):
    """
   
    """
    p = data['p']
    return jnp.minimum(0.0, p)
#=============================

#=================================
def pressure_edge_obj(grid, data):
    """
    Pressure on edge (rho=1)
    Target: P(1)=0
    """
    p = data['p']
    return p[-1]
#=================

#======================================
def grad_pressure_edge_obj(grid, data):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J).
    """ 
    dp_dr = data['p_r']
    return dp_dr[-1]
#===========================



#=============
# Constraints:
#==========================
def pressure_octic_con(params):
    """
    Limits the pressure profile to an 8th-order polynomial 
    max, in order toprevent non-monotonicity and have only 
    one objective on the pressure profile (pressure_shape).
    """
    c = params['p_l']
    higher_orders = c[5:]
    return higher_orders
#=======================

#========================
def pressure_DOF_con(params):
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

#=============================
def pressure_edge_con(params):
    """
    Pressure on edge (rho=1)
    Target: P(1)=0.
    """
    c = params['p_l']
    return c.sum()
#=================

#==================================
def grad_pressure_edge_con(params):
    """
    Pressure gradient on edge (rho=1)
    Target: GradP=0 (no surface current J).
    """ 
    c = params['p_l'][1:]
    order = jnp.arange(1, len(c)+1)
    return (order * c).sum()
#===========================
#===========================
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


