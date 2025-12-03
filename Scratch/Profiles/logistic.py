#NOTES:
# should L and k in logistic(L,k) be given names to reflect they're scalars, not arrays?
# have some way to call iota and/or curvature
# consider upper limit to k_max, as decided by theory
# logistic_opt(weights) needs to be in form acceptable for _Profile

import os
import numpy as np
import matplotlib.pyplot as plt


#------------- Global Variables -------------#
NPTS = 400 #number of points
X = np.linspace(0, 1, NPTS) #x axis generation




#---------------------- Logistic Function ----------------------#
def logistic(k, rho_shift):
    """
    Calculates the logistic function and its derivative function
    --------- Parameters ---------
    L: scalar
        Max value of the curve
    k: scalar
        Steepness of the curve
    rho: array-like
        Toroidal flux values

    --------- Returns ---------
    f: array-like
        Logistic function
    """
    rho = X - 0.5 #shift s.t. the func is no longer centered at rho=0
    f = 1 - ( 1 / (1 + np.exp(-k* (rho - rho_shift/10)))) #the logistic func
    return f



#----------------------- Node Distribution Functions ------------------------#

def generate_nodes(min_val, max_val, N, *, quadratic=False, equidistant=False):
    """
    Generates nodes in [min_val, max_val] using either a quadratic
    or equidistant spacing.
    Exactly one of `quadratic` or `equidistant` must be True.

    Parameters
    ----------
    min_val: float
        Lower bound of interval.
    max_val: float
        Upper bound of interval.
    N: int
        Number of nodes.
    quadratic : bool, optional
        If True, use quadratically graded nodes (cluster near max_val).
    equidistant : bool, optional
        If True, use equally spaced nodes.

    Returns
    -------
    nodes: ndarray
        1D array of nodes in [min_val, max_val].
    """
    #
    if quadratic == equidistant: # if both True or both False, error
        raise ValueError(
            "Exactly one of 'quadratic' or 'equidistant' must be True."
        )
    #
    t = np.linspace(0.0, 1.0, N)
    # Quadratically graded nodes, higher density near max_val:
    if quadratic:
        s = 1.0 - (1.0 - t)**2
        nodes = min_val + (max_val - min_val) * s
    # Equidistantly spaced nodes:
    else:  
        nodes = min_val + (max_val - min_val) * t
    #
    return nodes



#------------------- Superposition of Logistic Functions --------------------#
def logistic_super(N_k, k_min, k_max, N_shift, shift_min, shift_max, weights):
    """
    Generates a 3D array family of logistic functions, where
    axis 0 = value of the function, of length resol
    axis 1 = values of parameter k
    axis 2 = values of parameter rho_shift,
    then brings in optimization parameter array, weights,
    and generates a superposition of all N^2 vectors.
    --------- Parameters ---------
    N_k: scalar
        number of k values to generate
    k_min: scalar
        parameter k, minimum steepness
    N_shift: scalar
        number of rho_shifts to generate
    shift_min: scalar
        value of minimum rho_shift, horizontal displacement
    --------- Returns ------------
    superpos: array-like
        (NPTS,1) array that is a superposition of all logisitic
        function permutations of the chosen range of k and rho_shift,
        weighted by the optimization parameter "weights".
    """
    fam = np.empty((len(X), N_k, N_shift), dtype=float)
    k_nodes = generate_nodes( #distribution of k values 
        k_min, k_max, N_k, quadratic=False, equidistant=True)
    shift_nodes = generate_nodes( #distribution of rho_shifts
        shift_min, shift_max, N_shift, quadratic=False, equidistant=True) 


    for i, k_val in enumerate(k_nodes):
        for j, rho_shift in enumerate(shift_nodes):
            fam[:, i, j] = logistic(k_val, rho_shift)
    
    fam_flat = fam.reshape(NPTS, -1) #reshaping 3D krho_fam (resol,N,N) into 2D krho_flat (resol, N^2)
    superpos = np.empty(NPTS, dtype=float) #initializing (N,1) superposition array

    for w in range(N_k*N_shift): #scaling each func in family by optimization weights
        superpos += weights[w] * fam_flat[:,w] 
        
    superpos= superpos/(superpos[0] - superpos[-1]+ 0.05) #normalizing superposition
    return superpos
    



#------- Profile to be Passed to the Optimizer ------#
def logistic_opt(weights):
    """
    Function to be passed into the Optimizer.
    N, k_min and shift_min are defined here and
    passed into logistic_super. Weights are the 
    only optimization variables.
    -------- Parameters -------
    weights: array-like 
        Nx1 array of weights that scales each logistic
        func in the generated family
    --------- Returns ---------
    pressure profile, superposition of logistic funcs
    weighted by "weights"
    """
    #see logistic_super for definitions:
    N_k = 5 
    k_min = 20 
    k_max = 40 #steepness
    N_shift = 5
    shift_min = -2 #hard clamp
    shift_max = 2 #hard clamp

    opt_pressure= logistic_super(
        N_k, k_min, k_max, N_shift, shift_min, shift_max, weights)
    return opt_pressure 



#---------- Plotting -----------#
plt.figure(figsize=(8, 5))
plt.title('Logistic Function')
plt.xlabel(r'$\rho$')
plt.ylabel('f(x)')
#plot currently uses random weights:
plt.plot(X, logistic_super(5, 20, 40, 8, -3, 3, np.random.rand(5*8)))
plt.grid()
plt.tight_layout()
plt.savefig('logistic.png')

