
import jax.numpy as jnp











#============== PRESSURE =============================================================================================================
#============
# Objectives:
#=====================================
def pressure_monotonicity(params):
    """
    Monotonic pressure.
    """
    c = params['p_l']
    return 
#=================================

#===================================
def pressure_axis_range(params):
    """
    Allows pressure on-axis to be within specified bounds.
    """
    p_axis = params['p_l'][0]
    return p_axis
#================
#=================================




#=============
# Constraints:
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


