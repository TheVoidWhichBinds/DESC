import numpy as np
import matplotlib.pyplot as plt

def mtanh_profile(x, ped, offset, sym, width, core_poly):
    z = (x - sym) / width
    base = 0.5 * (ped - offset) * (1 - np.tanh(z)) + offset

    poly_coeffs = np.pad(core_poly, (1, 0))   # [0, a1, a2, ...]
    zz = z / (1 + np.exp(2 * z))
    poly_part = np.polyval(poly_coeffs[::-1], zz)

    y = base + poly_part * (offset - ped) / 2
    return y


def plot_mtanh(x, ped, offset, sym, width, core_poly):
    y = mtanh_profile(x, ped, offset, sym, width, core_poly)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Modified tanh profile")
    plt.grid(True)
    plt.show()


x = np.linspace(0, 1, 200)

plot_mtanh(
    x=x,
    ped=1,
    offset=0,
    sym=0.5,
    width=0.06,
    core_poly=[0, 0, 0]
)