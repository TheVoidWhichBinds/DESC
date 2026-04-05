import jax.numpy as jnp







#============== HELPER FUNCTION ====================================================================================================
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


