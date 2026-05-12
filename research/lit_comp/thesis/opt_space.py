# opt_space.py
#==============================================================================================================
#
# Conceptual visualization of a high-dimensional, non-convex stellarator optimization space.
#
# This figure is pedagogical, not DESC data. It represents:
#   1. optimization parameters as many radial axes,
#   2. objective residuals as smooth colored fog volumes,
#   3. broad continuous residual structure throughout the full cube,
#   4. interspersed objective residual structure across the optimization space.
#
# The key idea in this version is that each objective fog is not a single cloud centered at one location.
# Instead, each fog is a smooth function of x, y, and z across the entire cube.
#
# This version removes:
#   1. the sweep-plane slicing,
#   2. the slider,
#   3. the good and poor local-minimum markers.
#
# The output is saved as a PNG.
#
#==============================================================================================================

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"








#==============================================================================================================
# SETTINGS
#==============================================================================================================

GRID_N = 60

X_LIM = (
    -2.5,
    2.5,
)

Y_LIM = (
    -2.5,
    2.5,
)

Z_LIM = (
    -2.5,
    2.5,
)

N_PARAMETER_AXES = 14
AXIS_RADIUS = 2.35

FOG_MIN_FIELD = 0.020
FOG_OPACITY = 0.3
FOG_SURFACE_COUNT = 18

SAVE_HTML = False
SAVE_PNG = True
SHOW_FIGURE = False

HTML_NAME = "stellarator_optimization_space.html"
PNG_NAME = "Opt_Space.png"








#==============================================================================================================
# OBJECTIVE DEFINITIONS
#==============================================================================================================

OBJECTIVES = [
    {
        "name": "Force balance residual",
        "color": "rgb(220, 70, 70)",
        "enabled": True,
        "seed": 1,
        "phases": np.array(
            [
                0.20,
                1.10,
                -0.55,
                0.85,
            ]
        ),
        "mix": 0.62,
    },
    {
        "name": "Quasisymmetry residual",
        "color": "rgb(70, 110, 230)",
        "enabled": True,
        "seed": 202,
        "phases": np.array(
            [
                -0.10,
                1.55,
                -0.95,
                0.30,
            ]
        ),
        "mix": 0.54,
    },
]








#==============================================================================================================
# GRID
#==============================================================================================================

def make_grid():
    """
    Build the 3D grid on which the objective scalar fields are defined.
    """

    x = np.linspace(
        X_LIM[0],
        X_LIM[1],
        GRID_N,
    )

    y = np.linspace(
        Y_LIM[0],
        Y_LIM[1],
        GRID_N,
    )

    z = np.linspace(
        Z_LIM[0],
        Z_LIM[1],
        GRID_N,
    )

    X, Y, Z = np.meshgrid(
        x,
        y,
        z,
        indexing = "ij",
    )

    return X, Y, Z








#==============================================================================================================
# FIELD HELPERS
#==============================================================================================================

def smooth_field_once(
        field,
    ):
    """
    Apply one light smoothing pass to remove grid texture while keeping broad structure.
    """

    smoothed = 0.52 * field

    for axis in range(
        3,
    ):
        smoothed += (0.24 / 3.0) * np.roll(
            field,
            shift = 1,
            axis = axis,
        )

        smoothed += (0.24 / 3.0) * np.roll(
            field,
            shift = -1,
            axis = axis,
        )

    return smoothed




def smooth_field(
        field,
        passes = 2,
    ):
    """
    Repeatedly smooth the objective field.

    Fewer passes keeps the fog density more sporadic.
    """

    for _ in range(
        passes,
    ):
        field = smooth_field_once(
            field = field,
        )

    return field




def normalize_field(
        field,
    ):
    """
    Normalize a scalar field to [0, 1].
    """

    field = field - np.min(
        field,
    )

    maximum = np.max(
        field,
    )

    if maximum <= 0.0:
        return field

    return field / maximum




def make_coordinate_field(
        X,
        Y,
        Z,
        phases,
    ):
    """
    Build a sporadic full-cube field directly from x, y, and z.

    Larger values represent denser objective fog.
    Smaller values represent lower residual regions.
    """

    x = X / X_LIM[1]
    y = Y / Y_LIM[1]
    z = Z / Z_LIM[1]

    p0, p1, p2, p3 = phases

    field = 0.34

    #----------------------------------------------------------------------------------------------------------
    # Broad background variation
    #----------------------------------------------------------------------------------------------------------

    field += 0.10 * np.sin(1.25 * np.pi * x + 0.70 * y + p0)
    field += 0.09 * np.cos(1.10 * np.pi * y - 0.60 * z + p1)
    field += 0.08 * np.sin(0.95 * np.pi * z + 0.55 * x + p2)
    field += 0.27 * np.cos(0.80 * np.pi * (x - y + 0.50 * z) + p3)

    #----------------------------------------------------------------------------------------------------------
    # Sporadic mixed-frequency structure
    #----------------------------------------------------------------------------------------------------------

    field += 0.060 * np.sin(2.40 * np.pi * x + 1.15 * y - 0.50 * z + p1)
    field += 0.055 * np.cos(2.15 * np.pi * y - 0.90 * x + 0.75 * z + p2)
    field += 0.050 * np.sin(2.70 * np.pi * z + 0.85 * x + 0.40 * y + p3)

    field += 0.045 * np.sin(3.20 * np.pi * (x + 0.35 * y) + p0)
    field += 0.040 * np.cos(3.60 * np.pi * (y - 0.25 * z) + p1)
    field += 0.038 * np.sin(3.00 * np.pi * (z + 0.20 * x) + p2)

    #----------------------------------------------------------------------------------------------------------
    # Nonlinear interference terms
    #----------------------------------------------------------------------------------------------------------

    field += 0.050 * np.sin(
        2.20 * np.pi * (
            x * y
            - 0.65 * y * z
            + 0.45 * x * z
        )
        + p3
    )

    field += 0.045 * np.cos(
        2.80 * np.pi * (
            x**2
            - 0.70 * y**2
            + 0.55 * z**2
        )
        + p0
    )

    field += 0.035 * np.sin(
        4.00 * np.pi * (
            x * y * z
            + 0.25 * x
            - 0.20 * y
        )
        + p2
    )

    #----------------------------------------------------------------------------------------------------------
    # Mild global bowl so the domain still reads as an optimization space
    #----------------------------------------------------------------------------------------------------------

    field += 0.045 * (0.55 * x**2 + 0.35 * y**2 + 0.25 * z**2)
    field += 0.030 * (x * y - 0.60 * y * z + 0.40 * x * z)

    return field




def make_objective_field(
        X,
        Y,
        Z,
        objective,
    ):
    """
    Create one sporadic objective fog field without Gaussian blobs or Gaussian cavities.

    Larger values = higher residual / denser fog.
    Smaller values = lower residual regions.
    """

    base_field = make_coordinate_field(
        X = X,
        Y = Y,
        Z = Z,
        phases = objective["phases"],
    )

    shared_field = make_coordinate_field(
        X = X,
        Y = Y,
        Z = Z,
        phases = np.array(
            [
                0.35,
                -0.80,
                1.20,
                -0.45,
            ]
        ),
    )

    field = (
        objective.get(
            "mix",
            0.60,
        )
        * base_field
        + (
            1.0
            - objective.get(
                "mix",
                0.60,
            )
        )
        * shared_field
    )

    field = smooth_field(
        field = field,
        passes = 2,
    )

    field = normalize_field(
        field = field,
    )

    field = np.clip(
        field,
        0.0,
        1.0,
    )

    return field








#==============================================================================================================
# PARAMETER AXES
#==============================================================================================================

def make_radial_axes(
        n_axes,
        radius,
    ):
    """
    Approximate many high-dimensional parameter axes by directions on a sphere.
    """

    directions = []

    golden_angle = np.pi * (
        3.0 - np.sqrt(
            5.0,
        )
    )

    for i in range(
        n_axes,
    ):
        z = 1.0 - 2.0 * i / max(
            n_axes - 1,
            1,
        )

        r = np.sqrt(
            max(
                0.0,
                1.0 - z**2,
            )
        )

        theta = golden_angle * i

        x = r * np.cos(
            theta,
        )

        y = r * np.sin(
            theta,
        )

        directions.append(
            radius * np.array(
                [
                    x,
                    y,
                    z,
                ]
            )
        )

    return directions








#==============================================================================================================
# PLOT HELPERS
#==============================================================================================================

def add_parameter_axes(
        fig,
    ):
    """
    Add radial parameter axes and labels.
    """

    directions = make_radial_axes(
        n_axes = N_PARAMETER_AXES,
        radius = AXIS_RADIUS,
    )

    for index, direction in enumerate(
        directions,
    ):
        fig.add_trace(
            go.Scatter3d(
                x = [
                    0.0,
                    direction[0],
                ],
                y = [
                    0.0,
                    direction[1],
                ],
                z = [
                    0.0,
                    direction[2],
                ],
                mode = "lines",
                line = dict(
                    color = "rgba(35, 35, 35, 0.38)",
                    width = 3,
                ),
                showlegend = False,
                hoverinfo = "skip",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x = [
                    1.08 * direction[0],
                ],
                y = [
                    1.08 * direction[1],
                ],
                z = [
                    1.08 * direction[2],
                ],
                mode = "text",
                text = [
                    f"x{index + 1}",
                ],
                textfont = dict(
                    size = 10,
                    color = "rgba(0, 0, 0, 0.72)",
                ),
                showlegend = False,
                hoverinfo = "skip",
            )
        )




def add_legend_stub(
        fig,
        name,
        color,
    ):
    """
    Add a no-data trace so each objective appears once in the legend.
    """

    fig.add_trace(
        go.Scatter3d(
            x = [None],
            y = [None],
            z = [None],
            mode = "markers",
            marker = dict(
                size = 8,
                color = color,
                opacity = 0.8,
            ),
            name = name,
            legendgroup = name,
            showlegend = True,
            hoverinfo = "skip",
        )
    )




def rgb_string_to_tuple(
        color,
    ):
    """
    Convert a string like "rgb(220, 70, 70)" to an integer tuple.
    """

    values = color.replace(
        "rgb(",
        "",
    ).replace(
        ")",
        "",
    )

    parts = [
        int(part.strip())
        for part in values.split(
            ",",
        )
    ]

    return tuple(
        parts,
    )




def make_volume_colorscale(
        color,
    ):
    """
    Build a soft, transparent-to-colored volume colorscale.
    """

    r, g, b = rgb_string_to_tuple(
        color = color,
    )

    return [
        [
            0.00,
            f"rgba({r}, {g}, {b}, 0.00)",
        ],
        [
            0.35,
            f"rgba({r}, {g}, {b}, 0.00)",
        ],
        [
            0.55,
            f"rgba({r}, {g}, {b}, 0.06)",
        ],
        [
            0.75,
            f"rgba({r}, {g}, {b}, 0.16)",
        ],
        [
            1.00,
            f"rgba({r}, {g}, {b}, 0.30)",
        ],
    ]




def add_fog_volume(
        fig,
        X,
        Y,
        Z,
        field,
        name,
        color,
    ):
    """
    Add one objective as a continuous translucent volume.

    This replaces the previous x-slice construction.
    """

    fig.add_trace(
        go.Volume(
            x = X.ravel(),
            y = Y.ravel(),
            z = Z.ravel(),
            value = field.ravel(),
            isomin = FOG_MIN_FIELD,
            isomax = 1.0,
            opacity = FOG_OPACITY,
            surface_count = FOG_SURFACE_COUNT,
            colorscale = make_volume_colorscale(
                color = color,
            ),
            showscale = False,
            showlegend = False,
            name = name,
            legendgroup = name,
            hoverinfo = "skip",
        )
    )








#==============================================================================================================
# FIGURE BUILD
#==============================================================================================================

def build_figure():
    """
    Construct the full optimization-space figure.
    """

    X, Y, Z = make_grid()

    fields = []

    for objective in OBJECTIVES:
        if not objective.get(
            "enabled",
            True,
        ):
            continue

        field = make_objective_field(
            X = X,
            Y = Y,
            Z = Z,
            objective = objective,
        )

        fields.append(
            (
                objective,
                field,
            )
        )

    fig = go.Figure()

    add_parameter_axes(
        fig = fig,
    )

    for objective, _ in fields:
        add_legend_stub(
            fig = fig,
            name = objective["name"],
            color = objective["color"],
        )

    for objective, field in fields:
        add_fog_volume(
            fig = fig,
            X = X,
            Y = Y,
            Z = Z,
            field = field,
            name = objective["name"],
            color = objective["color"],
        )

    fig.update_layout(
        title = "Conceptual Non-Convex Stellarator Optimization Space",
        scene = dict(
            xaxis = dict(
                title = "",
                visible = False,
                showbackground = False,
                range = [
                    X_LIM[0],
                    X_LIM[1],
                ],
            ),
            yaxis = dict(
                title = "",
                visible = False,
                showbackground = False,
                range = [
                    Y_LIM[0],
                    Y_LIM[1],
                ],
            ),
            zaxis = dict(
                title = "",
                visible = False,
                showbackground = False,
                range = [
                    Z_LIM[0],
                    Z_LIM[1],
                ],
            ),
            aspectmode = "cube",
            camera = dict(
                eye = dict(
                    x = 1.55,
                    y = 1.45,
                    z = 1.15,
                )
            ),
        ),
        legend = dict(
            x = 0.01,
            y = 0.98,
            bgcolor = "rgba(255, 255, 255, 0.60)",
        ),
        margin = dict(
            l = 0,
            r = 0,
            b = 0,
            t = 45,
        ),
        width = 1400,
        height = 1100,
    )

    return fig








#==============================================================================================================
# MAIN
#==============================================================================================================

def main():
    """
    Build, save, and optionally show the figure.
    """

    fig = build_figure()

    if SAVE_HTML:
        fig.write_html(
            HTML_NAME,
            auto_open = True,
        )

    if SAVE_PNG:
        try:
            fig.write_image(
                PNG_NAME,
                scale = 2,
            )

            print(
                f"Saved PNG to {PNG_NAME}",
            )

        except Exception as error:
            print(
                "Could not save PNG. Install kaleido if you want static image output:",
            )
            print(
                "    pip install -U kaleido",
            )
            print(
                f"Error: {error}",
            )

    if SHOW_FIGURE:
        fig.show(
            renderer = "browser",
        )








#==============================================================================================================
# RUN
#==============================================================================================================

if __name__ == "__main__":
    main()