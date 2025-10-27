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

x = np.linspace(0, 1, 400) #normalized psi from 0 to 1
psi = x - 0.5 #offset function along x axis since its center is usually at 0



def logistic(L, k):
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
    f = 1 - (L / (1 + np.exp(-k*(psi)))) #the logistic func
    df_dpsi = k * f(1 - f) #derivative of the logistic func with respect to psi

    return f, df_dpsi



def logistic_fam(L, k):
    """
    Generates a family of logistic functions and 
    the corresponding family of derivatives
    --------- Parameters ---------
    L: array-like
        Maximum values for the curves
    k: array-like
        Steepness values for the curves
    --------- Returns ---------
    f_fam: 3D Array
        Family of logistic functions
    df_dpsi_fam: 3D Array
        Family of derivatives
    """
    #initializing arrays:
    f_fam = np.zeros((len(psi), len(L), len(k)), dtype = float) #each column is f with 1 L, 1 k 
    df_dpsi_fam = np.zeros((len(psi), len(L))) #each column is df_dpsi with 1 L, 1 k

    for l in range(len(L)): #looping over all values of L
        for i in range(len(k)):

            f_call = logistic(L[l], k[i])
            f_fam[:,l,i] = f_call[0]
            df_dpsi_fam[:,l,i] = f_call[1]

    return f_fam, df_dpsi_fam
    

#kill:?
#def gen_Lk(L_min, k_0, n):
    """
    Generates array of Ls and ks
    ------- Parameters ---------
    L_min: scalar
        minimum height, min L in array
    k_0: scalar
        ideal steepness, other k centered about this one
    n: scalar
        number of variations of L and k
    ------- Returns ------------------------------------
    L: array-like
        array of Ls
    k: array-like
        array of ks
    """
    L = np.linspace(L_min, 1, 0.1)




def scrapbook(L, k):
    #explain better in preamble:
    """
    Generates a family of monotonic functions by finding ("one and"?) two regions of psi
    where one function is spliced onto another by matching their derivative values and 
    moving that part of the second logistic func up and over to the index of the df_dpsi
    array where the first function matches 
    -------- Parameters ---------------------
    L: array-like
        Maximum values for the curves
    k: array-like
        Steepness values for the curves
    --------- Returns ---------------------------------------------------------
    spliced_fam: 2D-array
        Family of monotonic, Frankenstein function profiles
    """

    L = np.linspace(0.1,1,0.1)
    k = np.linspace(5,10,1)

    fam = logistic_fam(L, k) #generating 3D arrays of family of logistic func & family of derivatives
    f_fam = fam[0]
    df_dpsi_fam = fam[1]




















#---------- Plotting -----------#
plt.figure(figsize=(8, 5))
plt.title('Logistic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()

plt.plot(x, logistic(1, 10), label='L=1, k=10', color='green')
plt.plot(x, logistic(0.8, 10), label='L=0.8, k=10', color='blue')
plt.plot(x, logistic(0.6, 10), label='L=0.6, k=10', color='red')
plt.plot(x, logistic(0.4, 10), label='L=0.4, k=10', color='violet')

plt.legend()
plt.tight_layout()
plt.savefig('logistic.png')

