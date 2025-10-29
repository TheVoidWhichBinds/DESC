import os
import numpy as np
import matplotlib.pyplot as plt
downloads_dir = os.path.expanduser('~/Downloads')

#Notes:
#consider changing df_dpsi name
#consider changing f_call name
#should L and k in logistic(L,k) be given names to reflect they're scalars, not arrays?
#have some way to call iota and/or curvature
#add multiple functions in the family/superposition all of them?
#generation of N^2 vectors should be independent of optimization
resol = 400
x = np.linspace(0, 1, resol) #normalized psi from 0 to 1



#----------- Logistic Function -----------#
def logistic(k, psi_shift):
    """
    Calculates the logistic function and its derivative function
    --------- Parameters ---------
    L: scalar
        Max value of the curve
    k: scalar
        Steepness of the curve
    psi: array-like
        Toroidal flux values

    --------- Returns ---------
    f: array-like
        Logistic function
    df_dpsi: array-like 
        Derivative of the logistic function with respect to psi
    """
    psi = x - 0.5
    f = 1 - ( 1 / (1 + np.exp(-k*(psi - psi_shift/10)))) #the logistic func

    return f



#----------- Superposition of Logistic Functions -----------#
def logistic_super(N, weights):
    """
    Generates a 3D array family of logistic functions, where
    axis 0 = value of the function, of length resol
    axis 1 = values of parameter k
    axis 2 = values of parameter psi_shift,
    then brings in optimization parameter array, weights,
    and generates a superposition of all N^2 vectors 
    --------- Parameters ---------
    N: scalar
        number of k and psi_shift values to generate
        N^2 total permutations of k and psi_shift
    k_val: scalar
        values of parameter k, steepness factor
    psi_shift: scalar
        values of psi_shift, horizontal displacement
    --------- Returns ------------
    superpos: array-like
        (resol,1) array that is a superposition of all logisitic
        function permutations of the chosen range of k and psi_shift,
        weighted by the optimization parameter "weights"
    """
    kpsi_fam = np.empty((len(x), N, N), dtype=float)
    #k_val and psi_shift need to be more flexible
    #psi_shift crashes if you change range values
    for i, k_val in enumerate(range(10, 10 + 2*N, 2)):
        for j, psi_shift in enumerate(range(-2, -2 + N, 1)):
            kpsi_fam[:, i, j] = logistic(k_val, psi_shift)
    
    #reshaping 3D kpsi_fam (resol,N,N) into 2D kpsi_flat (resol, N^2)
    kpsi_flat = kpsi_fam.reshape(resol, -1) 
    superpos = np.empty(resol, dtype=float)

    for w in range(N**2):
        superpos += weights[w] * kpsi_flat[:,w] 
        superpos = superpos/(superpos[0] - superpos[-1])

    return superpos
    


#---------- Plotting -----------#
plt.figure(figsize=(8, 5))
plt.title('Logistic Function')
plt.xlabel(r'$\psi$')
plt.ylabel('f(x)')
#plot currently uses random weights
plt.plot(x, logistic_super(5, np.random.rand(5**2)))
plt.grid()
plt.tight_layout()
plt.savefig('logistic.png')

