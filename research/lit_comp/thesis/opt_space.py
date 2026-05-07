# opt_space.py
#==============================================================================================================
#
# Conceptual visualization of a high-dimensional, non-convex stellarator optimization space.
#
# This figure is pedagogical, not DESC data. It represents:
#   1. optimization parameters as many radial axes,
#   2. objective residuals as smooth colored fog volumes,
#   3. broad continuous residual structure throughout the full cube,
#   4. irregular low-residual pockets corresponding to local minima,
#   5. a sweep plane that reveals fog on one side and no fog on the other side,
#   6. a good local minimum and a poor local minimum.
#
# The key idea in this version is that each objective fog is not a single cloud centered at one location.
# Instead, each fog is a smooth function of x, y, and z across the entire cube.
#
# The good and poor local-minimum marker locations are computed from the superposition of the enabled scalar
# fields:
#   - good local minimum: 2nd-smallest value of the superposition field
#   - poor local minimum: 4th-smallest value of the superposition field
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
FOG_N_SLICES_X = 100
FOG_COLORSCALE_ALPHA_MAX = 0.30

CUT_PLANE_OPACITY = 0.16
CUT_PLANE_COLOR = "rgba(70, 70, 70, 0.16)"
INITIAL_CLIP_STEP = 10

SAVE_HTML = False
SAVE_PNG = False

HTML_NAME = "stellarator_optimization_space.html"
PNG_NAME = "stellarator_optimization_space.png"








#==============================================================================================================
# KEY POINTS
#==============================================================================================================

ORIGIN = np.array(
    [
        0.0,
        0.0,
        0.0,
    ]
)








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
    },
    {
        "name": "Quasisymmetry residual",
        "color": "rgb(70, 110, 230)",
        "enabled": True,
        "seed": 202,
        "phases": np.array(
            [
                -0.65,
                0.45,
                1.35,
                -0.25,
            ]
        ),
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
    Smaller values = lower residual / local-minimum candidates.
    """

    field = make_coordinate_field(
        X = X,
        Y = Y,
        Z = Z,
        phases = objective["phases"],
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




def find_ranked_superposition_points(
        X,
        Y,
        Z,
        fields,
        ranks,
        search_cube_size = 1.0,
    ):
    """
    Locate points by ranking the superposition of all enabled scalar fields.

    The minima search is restricted to a cube centered on the origin.

    search_cube_size = 1.0 means:
        x in [-0.5, 0.5]
        y in [-0.5, 0.5]
        z in [-0.5, 0.5]

    ranks are one-indexed.
    rank = 2 means the point with the second-smallest superposition value inside the cube.
    rank = 4 means the point with the fourth-smallest superposition value inside the cube.
    """

    if len(fields) == 0:
        raise ValueError(
            "At least one objective must be enabled.",
        )

    superposition = np.zeros_like(
        fields[0][1],
    )

    for _, field in fields:
        superposition += field

    half_width = 0.5 * search_cube_size

    search_mask = (
        (X >= -half_width)
        & (X <= half_width)
        & (Y >= -half_width)
        & (Y <= half_width)
        & (Z >= -half_width)
        & (Z <= half_width)
    )

    allowed_flat_indices = np.flatnonzero(
        search_mask.ravel(),
    )

    allowed_values = superposition.ravel()[
        allowed_flat_indices
    ]

    allowed_order = np.argsort(
        allowed_values,
    )

    ranked_points = []

    for rank in ranks:
        allowed_rank_index = allowed_order[
            rank - 1
        ]

        flat_index = allowed_flat_indices[
            allowed_rank_index
        ]

        grid_index = np.unravel_index(
            flat_index,
            superposition.shape,
        )

        point = np.array(
            [
                X[grid_index],
                Y[grid_index],
                Z[grid_index],
            ]
        )

        ranked_points.append(
            {
                "rank": rank,
                "point": point,
                "value": float(
                    superposition[grid_index],
                ),
            }
        )

    return ranked_points








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




def make_fog_colorscale(
        color,
        alpha_max = FOG_COLORSCALE_ALPHA_MAX,
    ):
    """
    Colorscale that fades from fully transparent to softly opaque.
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
            0.38,
            f"rgba({r}, {g}, {b}, 0.00)",
        ],
        [
            0.60,
            f"rgba({r}, {g}, {b}, {0.08 * alpha_max:.5f})",
        ],
        [
            0.78,
            f"rgba({r}, {g}, {b}, {0.26 * alpha_max:.5f})",
        ],
        [
            0.90,
            f"rgba({r}, {g}, {b}, {0.62 * alpha_max:.5f})",
        ],
        [
            1.00,
            f"rgba({r}, {g}, {b}, {alpha_max:.5f})",
        ],
    ]




def evenly_spaced_indices(
        n_total,
        n_keep,
    ):
    """
    Evenly spaced integer indices avoiding extreme domain boundaries.
    """

    if n_keep <= 0:
        return []

    if n_keep >= n_total:
        return list(
            range(
                n_total,
            )
        )

    return list(
        np.unique(
            np.round(
                np.linspace(
                    2,
                    n_total - 3,
                    n_keep,
                )
            ).astype(
                int,
            )
        )
    )




def add_surface_slice(
        fig,
        X_slice,
        Y_slice,
        Z_slice,
        field_slice,
        name,
        color,
    ):
    """
    Add one translucent x-slice of the fog field.
    """

    masked_field = np.where(
        field_slice >= FOG_MIN_FIELD,
        field_slice,
        np.nan,
    )

    if np.all(
        np.isnan(
            masked_field,
        )
    ):
        return False

    fig.add_trace(
        go.Surface(
            x = X_slice,
            y = Y_slice,
            z = Z_slice,
            surfacecolor = masked_field,
            cmin = 0.0,
            cmax = 1.0,
            colorscale = make_fog_colorscale(
                color = color,
            ),
            showscale = False,
            showlegend = False,
            name = name,
            legendgroup = name,
            hoverinfo = "skip",
            contours = dict(
                x = dict(
                    show = False,
                ),
                y = dict(
                    show = False,
                ),
                z = dict(
                    show = False,
                ),
            ),
        )
    )

    return True




def add_cut_plane(
        fig,
        x_value,
    ):
    """
    Add the sweep plane x = constant.
    """

    y = np.linspace(
        Y_LIM[0],
        Y_LIM[1],
        2,
    )

    z = np.linspace(
        Z_LIM[0],
        Z_LIM[1],
        2,
    )

    Y_plane, Z_plane = np.meshgrid(
        y,
        z,
        indexing = "ij",
    )

    X_plane = np.full_like(
        Y_plane,
        fill_value = x_value,
        dtype = float,
    )

    surfacecolor = np.zeros_like(
        X_plane,
        dtype = float,
    )

    fig.add_trace(
        go.Surface(
            x = X_plane,
            y = Y_plane,
            z = Z_plane,
            surfacecolor = surfacecolor,
            cmin = 0.0,
            cmax = 1.0,
            colorscale = [
                [0.0, CUT_PLANE_COLOR],
                [1.0, CUT_PLANE_COLOR],
            ],
            opacity = CUT_PLANE_OPACITY,
            showscale = False,
            showlegend = False,
            name = "Sweep plane",
            hoverinfo = "skip",
            contours = dict(
                x = dict(
                    show = False,
                ),
                y = dict(
                    show = False,
                ),
                z = dict(
                    show = False,
                ),
            ),
        )
    )




def add_fog_trace(
        fig,
        X,
        Y,
        Z,
        field,
        name,
        color,
        slice_indices,
        fog_trace_info,
    ):
    """
    Add one objective as a smooth fog assembled from many translucent x-slices.
    """

    for i in slice_indices:
        added = add_surface_slice(
            fig = fig,
            X_slice = X[i, :, :],
            Y_slice = Y[i, :, :],
            Z_slice = Z[i, :, :],
            field_slice = field[i, :, :],
            name = name,
            color = color,
        )

        if added:
            fog_trace_info.append(
                {
                    "trace_index": len(fig.data) - 1,
                    "x_value": float(X[i, 0, 0]),
                }
            )




def add_key_points(
        fig,
        good_local_minimum,
        poor_local_minimum,
        good_value,
        poor_value,
    ):
    """
    Add the origin plus the ranked good and poor local-minimum points.
    """

    fig.add_trace(
        go.Scatter3d(
            x = [
                ORIGIN[0],
            ],
            y = [
                ORIGIN[1],
            ],
            z = [
                ORIGIN[2],
            ],
            mode = "markers",
            marker = dict(
                size = 5,
                color = "black",
            ),
            name = "Origin",
            showlegend = False,
            hoverinfo = "skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [
                good_local_minimum[0],
            ],
            y = [
                good_local_minimum[1],
            ],
            z = [
                good_local_minimum[2],
            ],
            mode = "markers+text",
            marker = dict(
                size = 7,
                color = "rgb(80, 180, 80)",
            ),
            text = [
                "good local minimum",
            ],
            textposition = "top center",
            name = f"Good local minimum, 2nd lowest superposition = {good_value:.4f}",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [
                poor_local_minimum[0],
            ],
            y = [
                poor_local_minimum[1],
            ],
            z = [
                poor_local_minimum[2],
            ],
            mode = "markers+text",
            marker = dict(
                size = 7,
                color = "rgb(145, 90, 40)",
            ),
            text = [
                "poor local minimum",
            ],
            textposition = "top center",
            name = f"Poor local minimum, 4th lowest superposition = {poor_value:.4f}",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [
                ORIGIN[0],
                good_local_minimum[0],
            ],
            y = [
                ORIGIN[1],
                good_local_minimum[1],
            ],
            z = [
                ORIGIN[2],
                good_local_minimum[2],
            ],
            mode = "lines",
            line = dict(
                color = "rgb(80, 180, 80)",
                width = 6,
                dash = "dash",
            ),
            name = "Good local minimum guide",
            showlegend = False,
            hoverinfo = "skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [
                ORIGIN[0],
                poor_local_minimum[0],
            ],
            y = [
                ORIGIN[1],
                poor_local_minimum[1],
            ],
            z = [
                ORIGIN[2],
                poor_local_minimum[2],
            ],
            mode = "lines",
            line = dict(
                color = "rgb(145, 90, 40)",
                width = 6,
                dash = "dash",
            ),
            name = "Poor local minimum guide",
            showlegend = False,
            hoverinfo = "skip",
        )
    )




def make_slider_steps(
        fig,
        fog_trace_info,
        cut_plane_trace_indices,
        always_visible_trace_indices,
        clip_positions,
    ):
    """
    Build slider steps that sweep the cut plane and reveal fog only for x >= x_clip.
    """

    n_traces = len(
        fig.data,
    )

    steps = []

    for step_index, x_clip in enumerate(
        clip_positions,
    ):
        visible = [
            False
            for _ in range(
                n_traces,
            )
        ]

        for trace_index in always_visible_trace_indices:
            visible[trace_index] = True

        for info in fog_trace_info:
            if info["x_value"] >= x_clip:
                visible[info["trace_index"]] = True

        visible[cut_plane_trace_indices[step_index]] = True

        steps.append(
            dict(
                method = "update",
                args = [
                    {
                        "visible": visible,
                    },
                    {
                        "title": f"Conceptual Non-Convex Stellarator Optimization Space (sweep plane x = {x_clip:.2f})",
                    },
                ],
                label = f"{x_clip:.2f}",
            )
        )

    return steps








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

    ranked_points = find_ranked_superposition_points(
        X = X,
        Y = Y,
        Z = Z,
        fields = fields,
        ranks = (
            2,
            4,
        ),
        search_cube_size = 1.0,
    )

    good_local_minimum = ranked_points[0]["point"]
    poor_local_minimum = ranked_points[1]["point"]

    good_value = ranked_points[0]["value"]
    poor_value = ranked_points[1]["value"]

    fig = go.Figure()

    add_parameter_axes(
        fig = fig,
    )

    always_visible_trace_indices = list(
        range(
            len(fig.data),
        )
    )

    for objective, _ in fields:
        add_legend_stub(
            fig = fig,
            name = objective["name"],
            color = objective["color"],
        )

    always_visible_trace_indices.extend(
        range(
            always_visible_trace_indices[-1] + 1,
            len(fig.data),
        )
    )

    slice_indices = evenly_spaced_indices(
        n_total = X.shape[0],
        n_keep = FOG_N_SLICES_X,
    )

    clip_positions = [
        float(X[i, 0, 0])
        for i in slice_indices
    ]

    fog_trace_info = []

    for objective, field in fields:
        add_fog_trace(
            fig = fig,
            X = X,
            Y = Y,
            Z = Z,
            field = field,
            name = objective["name"],
            color = objective["color"],
            slice_indices = slice_indices,
            fog_trace_info = fog_trace_info,
        )

    add_key_points(
        fig = fig,
        good_local_minimum = good_local_minimum,
        poor_local_minimum = poor_local_minimum,
        good_value = good_value,
        poor_value = poor_value,
    )

    always_visible_trace_indices.extend(
        range(
            len(fig.data) - 5,
            len(fig.data),
        )
    )

    cut_plane_trace_indices = []

    for x_clip in clip_positions:
        add_cut_plane(
            fig = fig,
            x_value = x_clip,
        )

        cut_plane_trace_indices.append(
            len(fig.data) - 1,
        )

    initial_step = min(
        max(
            INITIAL_CLIP_STEP,
            0,
        ),
        len(clip_positions) - 1,
    )

    initial_visible = [
        False
        for _ in range(
            len(fig.data),
        )
    ]

    for trace_index in always_visible_trace_indices:
        initial_visible[trace_index] = True

    for info in fog_trace_info:
        if info["x_value"] >= clip_positions[initial_step]:
            initial_visible[info["trace_index"]] = True

    initial_visible[cut_plane_trace_indices[initial_step]] = True

    for trace_index, is_visible in enumerate(
        initial_visible,
    ):
        fig.data[trace_index].visible = is_visible

    slider_steps = make_slider_steps(
        fig = fig,
        fog_trace_info = fog_trace_info,
        cut_plane_trace_indices = cut_plane_trace_indices,
        always_visible_trace_indices = always_visible_trace_indices,
        clip_positions = clip_positions,
    )

    fig.update_layout(
        title = f"Conceptual Non-Convex Stellarator Optimization Space (sweep plane x = {clip_positions[initial_step]:.2f})",
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
        sliders = [
            dict(
                active = initial_step,
                currentvalue = dict(
                    prefix = "Sweep plane x = ",
                ),
                pad = dict(
                    t = 18,
                ),
                steps = slider_steps,
                x = 0.12,
                len = 0.76,
            )
        ],
        margin = dict(
            l = 0,
            r = 0,
            b = 0,
            t = 45,
        ),
    )

    return fig








#==============================================================================================================
# MAIN
#==============================================================================================================

def main():
    """
    Build, save, and show the figure.
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

        except Exception as error:
            print(
                "Could not save PNG. Install kaleido if you want static image output.",
            )
            print(
                f"Error: {error}",
            )

    if not SAVE_HTML:
        fig.show(
            renderer = "browser",
        )








#==============================================================================================================
# RUN
#==============================================================================================================

if __name__ == "__main__":
    main()