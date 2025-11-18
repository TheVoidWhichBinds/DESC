import numpy as np


def pressure_edge(params): #psi=1 pressure --> 0 
    p_coeff= params["p_l"]
    return p_coeff.sum()


def grad_pressure_axis(params): #psi=0 grad(P) --> 0
    c_1= params["p_l"][1]
    return c_1


def grad_pressure_edge(params): #psi=1 grad(P) --> 0 
    gradp_coeff= params["p_l"][1:]
    order= np.arange(1, len(gradp_coeff)+1)
    return (order * gradp_coeff).sum()


