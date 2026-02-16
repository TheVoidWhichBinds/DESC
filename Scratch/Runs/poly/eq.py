import sys
import os
from math import comb
sys.path.append("/Users/macdaddi/DESC")

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile


surface_init = FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[(0, 0), (1, 0), (1, 1), (-1, -1)],
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=19,
)

iota_init = PowerSeriesProfile([1, 0, 2])


def coefficients(p_scale, n, min_n=2):
    """
    Coeffs for p(rho) = p_scale * (1 - rho^2)^n
    Guarantees:
      p(0)=p_scale, p(1)=0, p'(0)=0, p'(1)=0  for n>=2
    Also nonnegative + monotone decreasing on [0,1].
    """
    n_eff = max(int(n), int(min_n))
    coeff = [0.0] * (2 * n_eff + 1)
    for k in range(n_eff + 1):
        coeff[2 * k] = p_scale * comb(n_eff, k) * ((-1) ** k)
    return coeff, n_eff


def run_equilibrium(p_scale, n, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    coeff, n_eff = coefficients(p_scale, n)

    eq = Equilibrium(
        L=8, M=8, N=3,
        surface=surface_init,
        pressure=PowerSeriesProfile(coeff),
        iota=iota_init,
        Psi=1.0,
    )

    eq_init = solve_continuation_automatic(eq.copy(), verbose=3)[-1]

    save_path = os.path.join(out_dir, "eq.h5")
    eq_init.save(save_path)
    return save_path, n_eff


if __name__ == "__main__":
    # Standalone test run: saves into ./p1.0e4_n2/eq.h5 next to this file
    dir_name = os.path.dirname(os.path.abspath(__file__))
    p_scale = 1e4
    n = 2
    out_dir = os.path.join(dir_name, f"p{p_scale:.1e}_n{n}")
    run_equilibrium(p_scale, n, out_dir)
