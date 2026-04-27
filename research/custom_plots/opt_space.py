import os
import numpy as np
import matplotlib.pyplot as plt











#============== SETTINGS =======================================================================================

axis_color = "black"
grid_color = "gray"

surface_alpha = 0.42
surface_cmap = "viridis"

plane_color = "deepskyblue"
plane_alpha = 0.22
intersection_color = "navy"
intersection_linewidth = 3.0

point_size = 110

n_radial_axes = 9
axis_length = 4.0
z_axis_length = 8.0

x_lim = (-4.0, 4.0)
y_lim = (-4.0, 4.0)

n_grid = 300
n_base_grid = 9

y_slice = 0.0

view_elev = 16
view_azim = 150

show_figures = True

output_dir = os.path.dirname(os.path.abspath(__file__))
output_file_1 = "Opt_Space.png"
output_file_2 = "Opt_Space_FXD.png"

# Same red point in both figures, and it lies on the fixed plane y = y_slice
x_initial = -3.0
y_initial = y_slice

# Orange point in file 1: off-plane local minimum
x_guess_off = 2.2
y_guess_off = 1.6

# Orange point in file 2: on-plane local minimum
x_guess_on = 1.2











#============== COST LANDSCAPE =================================================================================

def f_xy(
        x,
        y,
    ):
    """
    Cost surface with multiple local minima.
    One local minimum is placed near (2.2, 1.6), off the plane.
    Another is placed near (1.2, 0.0), on the plane y = y_slice.
    """
    bowl = 0.10 * (x**2 + y**2)

    ripples = (
        0.45 * np.sin(1.25 * x)
        - 0.25 * np.cos(1.55 * y)
        + 0.35 * np.sin(1.10 * x - 0.85 * y)
    )

    well_on_slice = -2.9 * np.exp(
        -((x - 1.2)**2 / 0.55 + (y - 0.0)**2 / 0.40)
    )

    well_off_slice = -3.2 * np.exp(
        -((x - 2.2)**2 / 0.55 + (y - 1.6)**2 / 0.50)
    )

    extra_structure = -1.5 * np.exp(
        -((x + 0.9)**2 / 0.80 + (y + 1.8)**2 / 0.70)
    )

    return bowl + ripples + well_on_slice + well_off_slice + extra_structure











#============== HELPERS ========================================================================================

def find_surface_min_near_guess(
        X,
        Y,
        Z,
        x_guess,
        y_guess,
        window_x = 0.9,
        window_y = 0.9,
    ):
    """
    Find the lowest sampled point in a window around a guess.
    Used to identify a local minimum on the full 2D surface.
    """
    mask = (
        (X >= x_guess - window_x)
        & (X <= x_guess + window_x)
        & (Y >= y_guess - window_y)
        & (Y <= y_guess + window_y)
    )

    if not np.any(mask):
        raise ValueError("No grid points found in the requested search window.")

    Z_masked = np.where(mask, Z, np.inf)
    idx = np.unravel_index(np.argmin(Z_masked), Z.shape)

    return X[idx], Y[idx], Z[idx]





def find_slice_min_near_guess(
        x_values,
        y_value,
        x_guess,
        window_x = 1.0,
    ):
    """
    Find a local minimum along the fixed-y slice near x_guess.
    """
    z_values = f_xy(x_values, y_value)

    mask = (x_values >= x_guess - window_x) & (x_values <= x_guess + window_x)

    if not np.any(mask):
        raise ValueError("No slice points found in the requested search window.")

    x_local = x_values[mask]
    z_local = z_values[mask]

    i_min = np.argmin(z_local)

    return x_local[i_min], y_value, z_local[i_min]





def draw_base_grid(
        ax,
        xlim,
        ylim,
        z0,
        n_lines,
        color = "gray",
        linewidth = 0.9,
        alpha = 0.50,
    ):
    grid_x = np.linspace(xlim[0], xlim[1], n_lines)
    grid_y = np.linspace(ylim[0], ylim[1], n_lines)

    for gx in grid_x:
        ax.plot(
            [gx, gx],
            [ylim[0], ylim[1]],
            [z0, z0],
            color = color,
            linewidth = linewidth,
            alpha = alpha,
        )

    for gy in grid_y:
        ax.plot(
            [xlim[0], xlim[1]],
            [gy, gy],
            [z0, z0],
            color = color,
            linewidth = linewidth,
            alpha = alpha,
        )





def draw_custom_axes(
        ax,
        z0,
        axis_color = "black",
        axis_length = 4.0,
        z_axis_length = 8.0,
        n_radial_axes = 9,
    ):
    #------ z-axis ---------------------------------------------------------------------------------------------

    ax.quiver(
        0, 0, z0,
        0, 0, z_axis_length,
        color = axis_color,
        linewidth = 2.2,
        arrow_length_ratio = 0.08,
    )

    #------ 9 planar axes --------------------------------------------------------------------------------------

    angles = np.linspace(0, 2 * np.pi, n_radial_axes, endpoint = False)

    for angle in angles:
        dx = axis_length * np.cos(angle)
        dy = axis_length * np.sin(angle)

        ax.quiver(
            0, 0, z0,
            dx, dy, 0,
            color = axis_color,
            linewidth = 2.0,
            arrow_length_ratio = 0.08,
        )





def draw_bounding_box(
        ax,
        xlim,
        ylim,
        zlim,
        color = "black",
        linewidth = 1.8,
        alpha = 0.95,
    ):
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim

    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        ax.plot(
            [corners[i][0], corners[j][0]],
            [corners[i][1], corners[j][1]],
            [corners[i][2], corners[j][2]],
            color = color,
            linewidth = linewidth,
            alpha = alpha,
        )





def style_axes(
        ax,
        xlim,
        ylim,
        zlim,
    ):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("Parameter Plane")
    ax.set_zlabel("Cost", labelpad = 12)

    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))

    # draw_bounding_box(
    #     ax,
    #     xlim = xlim,
    #     ylim = ylim,
    #     zlim = zlim,
    #     color = "black",
    #     linewidth = 1.8,
    #     alpha = 0.95,
    # )





def add_point_and_drop_line(
        ax,
        x,
        y,
        z,
        z0,
        color,
    ):
    ax.scatter(
        x,
        y,
        z,
        s = point_size,
        color = color,
        edgecolor = "black",
        linewidth = 0.8,
    )

    ax.plot(
        [x, x],
        [y, y],
        [z0, z],
        linestyle = "--",
        linewidth = 1.0,
        color = color,
        alpha = 0.75,
    )









#============== PLOTTING =======================================================================================

def make_plot(
        show_plane,
        orange_point,
        output_path,
        X,
        Y,
        Z,
        z0,
        z1,
        red_point,
    ):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection = "3d")

    #------ base x-y grid --------------------------------------------------------------------------------------

    draw_base_grid(
        ax,
        xlim = x_lim,
        ylim = y_lim,
        z0 = z0,
        n_lines = n_base_grid,
        color = grid_color,
        linewidth = 0.9,
        alpha = 0.50,
    )

    #------ main surface ---------------------------------------------------------------------------------------

    ax.plot_surface(
        X,
        Y,
        Z,
        cmap = surface_cmap,
        alpha = surface_alpha,
        linewidth = 0,
        antialiased = True,
    )

    #------ optional fixed slice plane -------------------------------------------------------------------------

    if show_plane:
        x_plane = np.linspace(x_lim[0], x_lim[1], 2)
        z_plane_mesh = np.linspace(z0, z1, 2)

        X_plane, Z_plane_mesh = np.meshgrid(x_plane, z_plane_mesh)
        Y_plane = np.full_like(X_plane, y_slice)

        ax.plot_surface(
            X_plane,
            Y_plane,
            Z_plane_mesh,
            color = plane_color,
            alpha = plane_alpha,
            shade = False,
        )

        x_curve = np.linspace(x_lim[0], x_lim[1], 700)
        y_curve = np.full_like(x_curve, y_slice)
        z_curve = f_xy(x_curve, y_curve)

        ax.plot(
            x_curve,
            y_curve,
            z_curve,
            color = intersection_color,
            linewidth = intersection_linewidth,
        )

    #------ custom axes ----------------------------------------------------------------------------------------

    draw_custom_axes(
        ax,
        z0 = z0,
        axis_color = axis_color,
        axis_length = axis_length,
        z_axis_length = z_axis_length,
        n_radial_axes = n_radial_axes,
    )

    #------ red and orange points ------------------------------------------------------------------------------

    add_point_and_drop_line(
        ax,
        x = red_point[0],
        y = red_point[1],
        z = red_point[2],
        z0 = z0,
        color = "red",
    )

    add_point_and_drop_line(
        ax,
        x = orange_point[0],
        y = orange_point[1],
        z = orange_point[2],
        z0 = z0,
        color = "orange",
    )

    #------ axes styling ---------------------------------------------------------------------------------------

    style_axes(
        ax,
        xlim = x_lim,
        ylim = y_lim,
        zlim = (z0, z1),
    )

    #------ view ------------------------------------------------------------------------------------------------

    ax.view_init(elev = view_elev, azim = view_azim)

    plt.tight_layout()
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight")

    if show_figures:
        plt.show()

    plt.close(fig)









#============== MAIN ===========================================================================================

def main():
    x = np.linspace(x_lim[0], x_lim[1], n_grid)
    y = np.linspace(y_lim[0], y_lim[1], n_grid)

    X, Y = np.meshgrid(x, y)
    Z = f_xy(X, Y)

    z_plane = np.min(Z) - 1.0
    z_top = np.max(Z) + 1.0

    #------ same red point for both plots ---------------------------------------------------------------------

    z_initial = f_xy(x_initial, y_initial)
    red_point = (x_initial, y_initial, z_initial)

    #------ orange point for file 1: off-plane local equilibrium ----------------------------------------------

    orange_off = find_surface_min_near_guess(
        X,
        Y,
        Z,
        x_guess = x_guess_off,
        y_guess = y_guess_off,
        window_x = 0.9,
        window_y = 0.9,
    )

    #------ orange point for file 2: on-plane local equilibrium -----------------------------------------------

    x_slice_dense = np.linspace(x_lim[0], x_lim[1], 3000)

    orange_on = find_slice_min_near_guess(
        x_values = x_slice_dense,
        y_value = y_slice,
        x_guess = x_guess_on,
        window_x = 1.0,
    )

    #------ output paths ---------------------------------------------------------------------------------------

    output_path_1 = os.path.join(output_dir, output_file_1)
    output_path_2 = os.path.join(output_dir, output_file_2)

    #------ make file 1 ----------------------------------------------------------------------------------------

    make_plot(
        show_plane = False,
        orange_point = orange_off,
        output_path = output_path_1,
        X = X,
        Y = Y,
        Z = Z,
        z0 = z_plane,
        z1 = z_top,
        red_point = red_point,
    )

    #------ make file 2 ----------------------------------------------------------------------------------------

    make_plot(
        show_plane = True,
        orange_point = orange_on,
        output_path = output_path_2,
        X = X,
        Y = Y,
        Z = Z,
        z0 = z_plane,
        z1 = z_top,
        red_point = red_point,
    )

    print(f"Saved: {output_path_1}")
    print(f"Saved: {output_path_2}")





if __name__ == "__main__":
    main()