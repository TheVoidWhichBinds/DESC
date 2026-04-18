
from itertools import product










#============== INITIALIZER =====================================================================================
#==================
def cond_generator(
    resolution_range: list,
    NFP_range: list,
    R_range: list,
    Z_range: list,
    p_l_range: list,
    i_l_range: list,
    Psi_range: list,
    weights: list,
    ):
    """
    Generates initial condition dictionaries for all feature vectors.
    """
    for resolution, NFP, R, Z, p_l, i_l, Psi in product(
        resolution_range,
        NFP_range,
        R_range,
        Z_range,
        p_l_range,
        i_l_range,
        Psi_range,
    ):
        yield {
            "resolution": resolution,
            "NFP": NFP,
            "R": R,         # (R_lmn, modes_R)
            "Z": Z,         # (Z_lmn, modes_Z)
            "p_l": p_l,
            "i_l": i_l,
            "Psi": Psi,
            "weights": weights,
        }
#======================
#===================================================================================================================


