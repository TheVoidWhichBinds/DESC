#==============================================================================================================
# IMPORTS
#==============================================================================================================

from pathlib import Path

import numpy as np

from desc.equilibrium import Equilibrium
from desc.io import load
from desc.profiles import PowerSeriesProfile, SplineProfile





#==============================================================================================================
# EQUILIBRIUM LOADING / CONSTRUCTION
#==============================================================================================================

def build_initial_equilibrium(
        run_config,
    ):
    """
    Build the initial DESC equilibrium from a flexible paper-specific config.

    Supported initial_equilibrium formats:

        1. H5 / saved DESC object:
            {
                "source_type": "h5",
                "path": ".../initial.h5",
                "load_final_member_if_family": True,
            }

        2. DESC input file:
            {
                "source_type": "input_file",
                "path": ".../input_file",
                "kwargs": {...},
            }

        3. Direct construction:
            {
                "source_type": "construct",
                "surface": {...} or [[l, m, n, R, Z], ...],
                "axis": {...} or [[n, R, Z], ...],
                "pressure": {...},
                "iota": {...},
                "current": {...},
                "physics": {...},
                "resolution": {...},
                "kwargs": {...},
            }

    The goal is that every paper config may store its initial conditions in the
    paper's natural format, while the optimization pipeline always receives a
    DESC Equilibrium object.
    """

    initial_config = run_config["initial_equilibrium"]
    source_type = normalize_source_type(
        source_type = initial_config.get("source_type", None),
    )

    if source_type == "h5":
        eq = build_from_h5(
            initial_config = initial_config,
        )

    elif source_type == "input_file":
        eq = build_from_input_file(
            initial_config = initial_config,
        )

    elif source_type == "construct":
        eq = build_from_constructed_config(
            initial_config = initial_config,
            run_config = run_config,
        )

    else:
        raise NotImplementedError(
            f"Initial equilibrium source_type '{source_type}' is not implemented. "
            "Use one of: 'h5', 'input_file', or 'construct'."
        )

    eq = apply_optional_equilibrium_updates(
        eq = eq,
        initial_config = initial_config,
        run_config = run_config,
    )

    return eq





#==============================================================================================================
# SOURCE TYPE NORMALIZATION
#==============================================================================================================

def normalize_source_type(
        source_type,
    ):
    if source_type is None:
        raise ValueError(
            "initial_equilibrium['source_type'] is required."
        )

    aliases = {
        "h5": "h5",
        "H5": "h5",
        "saved": "h5",
        "saved_h5": "h5",
        "equilibrium_h5": "h5",
        "equilibria_family": "h5",

        "input": "input_file",
        "input_file": "input_file",
        "desc_input": "input_file",
        "DESC_input": "input_file",

        "construct": "construct",
        "constructed": "construct",
        "direct": "construct",
        "dict": "construct",
        "from_coefficients": "construct",
        "coefficients": "construct",
    }

    if source_type not in aliases:
        raise ValueError(
            f"Unknown initial equilibrium source_type '{source_type}'. "
            f"Allowed aliases are: {sorted(aliases.keys())}"
        )

    return aliases[source_type]





#==============================================================================================================
# H5 / SAVED DESC OBJECT SUPPORT
#==============================================================================================================

def build_from_h5(
        initial_config,
    ):
    path = Path(initial_config["path"]).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find initial equilibrium H5 file: {path}"
        )

    eq_obj = load(str(path))

    if initial_config.get("load_final_member_if_family", True):
        eq_obj = get_final_equilibrium_from_loaded_object(
            eq_obj = eq_obj,
        )

    return eq_obj





def get_final_equilibrium_from_loaded_object(
        eq_obj,
    ):
    if hasattr(eq_obj, "equilibria"):
        return eq_obj.equilibria[-1]

    try:
        return eq_obj[-1]

    except TypeError:
        return eq_obj





#==============================================================================================================
# DESC INPUT FILE SUPPORT
#==============================================================================================================

def build_from_input_file(
        initial_config,
    ):
    path = Path(initial_config["path"]).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find DESC input file: {path}"
        )

    kwargs = dict(initial_config.get("kwargs", {}))

    return Equilibrium.from_input_file(
        path = str(path),
        **kwargs,
    )





#==============================================================================================================
# DIRECT EQUILIBRIUM CONSTRUCTION SUPPORT
#==============================================================================================================

def build_from_constructed_config(
        initial_config,
        run_config,
    ):
    physics_config = merge_nested_dicts(
        base = run_config.get("physics", {}),
        override = initial_config.get("physics", {}),
    )

    resolution_config = merge_nested_dicts(
        base = run_config.get("resolution", {}),
        override = initial_config.get("resolution", {}),
    )

    profile_config = merge_nested_dicts(
        base = run_config.get("profile_config", {}),
        override = initial_config.get("profile_config", {}),
    )

    kwargs = dict(initial_config.get("kwargs", {}))

    surface = make_surface_argument(
        surface_config = first_existing_key(
            data = initial_config,
            keys = ["surface", "boundary", "lcfs", "boundary_surface"],
            default = None,
        ),
    )

    axis = make_axis_argument(
        axis_config = first_existing_key(
            data = initial_config,
            keys = ["axis", "magnetic_axis"],
            default = None,
        ),
    )

    pressure = make_profile_argument(
        profile_config = first_existing_key(
            data = initial_config,
            keys = ["pressure", "p", "pressure_profile"],
            default = profile_config.get("reference_pressure", None),
        ),
        profile_name = "pressure",
    )

    iota = make_profile_argument(
        profile_config = first_existing_key(
            data = initial_config,
            keys = ["iota", "rotational_transform", "iota_profile"],
            default = profile_config.get("reference_iota", None),
        ),
        profile_name = "iota",
    )

    current = make_profile_argument(
        profile_config = first_existing_key(
            data = initial_config,
            keys = ["current", "current_profile"],
            default = profile_config.get("reference_current", None),
        ),
        profile_name = "current",
    )

    eq_kwargs = {
        "Psi": physics_config.get("Psi", physics_config.get("psi", 1.0)),
        "NFP": physics_config.get("NFP", physics_config.get("nfp", None)),
        "L": resolution_config.get("L", None),
        "M": resolution_config.get("M", None),
        "N": resolution_config.get("N", None),
        "L_grid": resolution_config.get("L_grid", None),
        "M_grid": resolution_config.get("M_grid", None),
        "N_grid": resolution_config.get("N_grid", None),
        "pressure": pressure,
        "iota": iota,
        "current": current,
        "surface": surface,
        "axis": axis,
        "sym": initial_config.get("sym", kwargs.pop("sym", None)),
        "spectral_indexing": initial_config.get(
            "spectral_indexing",
            kwargs.pop("spectral_indexing", "ansi"),
        ),
        "ensure_nested": initial_config.get(
            "ensure_nested",
            kwargs.pop("ensure_nested", True),
        ),
        "check_orientation": initial_config.get(
            "check_orientation",
            kwargs.pop("check_orientation", True),
        ),
    }

    eq_kwargs.update(kwargs)

    eq_kwargs = drop_none_values(
        data = eq_kwargs,
    )

    return Equilibrium(
        **eq_kwargs,
    )





#==============================================================================================================
# PROFILE NORMALIZATION
#==============================================================================================================

def make_profile_argument(
        profile_config,
        profile_name,
    ):
    if profile_config is None:
        return None

    if hasattr(profile_config, "compute"):
        return profile_config

    if isinstance(profile_config, np.ndarray):
        return profile_config

    if isinstance(profile_config, (list, tuple)):
        return normalize_profile_array(
            profile_config = profile_config,
            profile_name = profile_name,
        )

    if not isinstance(profile_config, dict):
        raise TypeError(
            f"{profile_name} profile must be a dict, list, tuple, ndarray, "
            f"or DESC Profile object. Got {type(profile_config)}."
        )

    profile_type = normalize_profile_type(
        profile_type = profile_config.get("type", "PowerSeriesProfile"),
    )

    if profile_type == "power_series":
        return make_power_series_profile(
            profile_config = profile_config,
            profile_name = profile_name,
        )

    if profile_type == "spline":
        return make_spline_profile(
            profile_config = profile_config,
            profile_name = profile_name,
        )

    if profile_type == "array":
        return normalize_profile_array(
            profile_config = profile_config,
            profile_name = profile_name,
        )

    raise NotImplementedError(
        f"Profile type '{profile_type}' is not implemented for {profile_name}."
    )





def normalize_profile_type(
        profile_type,
    ):
    aliases = {
        "PowerSeriesProfile": "power_series",
        "power_series": "power_series",
        "powerseries": "power_series",
        "polynomial": "power_series",
        "poly": "power_series",

        "SplineProfile": "spline",
        "spline": "spline",

        "array": "array",
        "ndarray": "array",
        "mode_array": "array",
    }

    if profile_type not in aliases:
        raise ValueError(
            f"Unknown profile type '{profile_type}'. "
            f"Allowed aliases are: {sorted(aliases.keys())}"
        )

    return aliases[profile_type]





def make_power_series_profile(
        profile_config,
        profile_name,
    ):
    modes, params = parse_modes_and_params(
        config = profile_config,
        allowed_mode_keys = ["modes", "l", "L", "orders", "powers"],
        allowed_param_keys = ["params", "coeffs", "coefficients", "values"],
        object_name = profile_name,
    )

    sym = profile_config.get("sym", profile_config.get("stellarator_symmetry", "auto"))

    return PowerSeriesProfile(
        params = params,
        modes = modes,
        sym = sym,
        name = profile_config.get("name", profile_name),
    )





def make_spline_profile(
        profile_config,
        profile_name,
    ):
    x = first_existing_key(
        data = profile_config,
        keys = ["rho", "x", "knots", "nodes"],
        default = None,
    )

    y = first_existing_key(
        data = profile_config,
        keys = ["values", "params", "p", "iota"],
        default = None,
    )

    if x is None or y is None:
        raise ValueError(
            f"Spline profile '{profile_name}' requires radial nodes and values."
        )

    return SplineProfile(
        values = np.asarray(y, dtype = float),
        knots = np.asarray(x, dtype = float),
        name = profile_config.get("name", profile_name),
    )





def normalize_profile_array(
        profile_config,
        profile_name,
    ):
    if isinstance(profile_config, dict):
        modes, params = parse_modes_and_params(
            config = profile_config,
            allowed_mode_keys = ["modes", "l", "L", "orders", "powers"],
            allowed_param_keys = ["params", "coeffs", "coefficients", "values"],
            object_name = profile_name,
        )

        return np.column_stack(
            [
                np.asarray(modes, dtype = float),
                np.asarray(params, dtype = float),
            ]
        )

    arr = np.asarray(profile_config, dtype = float)

    if arr.ndim == 1:
        modes = np.arange(arr.size)
        return np.column_stack(
            [
                modes,
                arr,
            ]
        )

    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr

    raise ValueError(
        f"Could not parse {profile_name} profile array. "
        "Use either [coeff0, coeff1, ...] or [[mode, coeff], ...]."
    )





#==============================================================================================================
# SURFACE / BOUNDARY NORMALIZATION
#==============================================================================================================

def make_surface_argument(
        surface_config,
    ):
    if surface_config is None:
        return None

    if hasattr(surface_config, "R_lmn") or hasattr(surface_config, "params"):
        return surface_config

    if isinstance(surface_config, np.ndarray):
        return surface_config

    if isinstance(surface_config, (list, tuple)):
        return normalize_surface_array(
            surface_config = surface_config,
        )

    if not isinstance(surface_config, dict):
        raise TypeError(
            f"Surface config must be dict, list, tuple, ndarray, or DESC Surface object. "
            f"Got {type(surface_config)}."
        )

    if "array" in surface_config:
        return normalize_surface_array(
            surface_config = surface_config["array"],
        )

    if "modes" in surface_config:
        return surface_from_modes_dict(
            surface_config = surface_config,
        )

    if "RBC" in surface_config or "ZBS" in surface_config:
        return surface_from_vmec_style_dict(
            surface_config = surface_config,
        )

    if "Rb_lmn" in surface_config or "Zb_lmn" in surface_config:
        return surface_from_desc_boundary_dict(
            surface_config = surface_config,
        )

    raise ValueError(
        "Could not parse surface config. Supported formats are: "
        "'array', 'modes', 'RBC/ZBS', or 'Rb_lmn/Zb_lmn'."
    )





def normalize_surface_array(
        surface_config,
    ):
    arr = np.asarray(surface_config, dtype = float)

    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(
            "Surface array must have shape (k, 5) with columns [l, m, n, R, Z]."
        )

    return arr





def surface_from_modes_dict(
        surface_config,
    ):
    rows = []

    for entry in surface_config["modes"]:
        if isinstance(entry, dict):
            l = entry.get("l", 0)
            m = entry["m"]
            n = entry["n"]
            R = entry.get("R", entry.get("R_coeff", 0.0))
            Z = entry.get("Z", entry.get("Z_coeff", 0.0))

        else:
            if len(entry) == 4:
                m, n, R, Z = entry
                l = 0

            elif len(entry) == 5:
                l, m, n, R, Z = entry

            else:
                raise ValueError(
                    "Each surface mode entry must be [m, n, R, Z] or [l, m, n, R, Z]."
                )

        rows.append([l, m, n, R, Z])

    return normalize_surface_array(
        surface_config = rows,
    )





def surface_from_vmec_style_dict(
        surface_config,
    ):
    rows_by_mode = {}

    for key, value in surface_config.get("RBC", {}).items():
        m, n = parse_mode_key(
            mode_key = key,
        )
        rows_by_mode.setdefault((m, n), [0, m, n, 0.0, 0.0])
        rows_by_mode[(m, n)][3] = value

    for key, value in surface_config.get("ZBS", {}).items():
        m, n = parse_mode_key(
            mode_key = key,
        )
        rows_by_mode.setdefault((m, n), [0, m, n, 0.0, 0.0])
        rows_by_mode[(m, n)][4] = value

    return normalize_surface_array(
        surface_config = list(rows_by_mode.values()),
    )





def surface_from_desc_boundary_dict(
        surface_config,
    ):
    rows_by_mode = {}

    for entry in surface_config.get("Rb_lmn", []):
        l, m, n, coeff = parse_boundary_entry(
            entry = entry,
            coeff_name = "R",
        )
        rows_by_mode.setdefault((l, m, n), [l, m, n, 0.0, 0.0])
        rows_by_mode[(l, m, n)][3] = coeff

    for entry in surface_config.get("Zb_lmn", []):
        l, m, n, coeff = parse_boundary_entry(
            entry = entry,
            coeff_name = "Z",
        )
        rows_by_mode.setdefault((l, m, n), [l, m, n, 0.0, 0.0])
        rows_by_mode[(l, m, n)][4] = coeff

    return normalize_surface_array(
        surface_config = list(rows_by_mode.values()),
    )





#==============================================================================================================
# AXIS NORMALIZATION
#==============================================================================================================

def make_axis_argument(
        axis_config,
    ):
    if axis_config is None:
        return None

    if hasattr(axis_config, "R_n") or hasattr(axis_config, "params"):
        return axis_config

    if isinstance(axis_config, np.ndarray):
        return normalize_axis_array(
            axis_config = axis_config,
        )

    if isinstance(axis_config, (list, tuple)):
        return normalize_axis_array(
            axis_config = axis_config,
        )

    if not isinstance(axis_config, dict):
        raise TypeError(
            f"Axis config must be dict, list, tuple, ndarray, or DESC Curve object. "
            f"Got {type(axis_config)}."
        )

    if "array" in axis_config:
        return normalize_axis_array(
            axis_config = axis_config["array"],
        )

    rows_by_mode = {}

    for key, value in axis_config.get("Ra_n", {}).items():
        n = int(key)
        rows_by_mode.setdefault(n, [n, 0.0, 0.0])
        rows_by_mode[n][1] = value

    for key, value in axis_config.get("Za_n", {}).items():
        n = int(key)
        rows_by_mode.setdefault(n, [n, 0.0, 0.0])
        rows_by_mode[n][2] = value

    if len(rows_by_mode) == 0:
        raise ValueError(
            "Could not parse axis config. Use 'array' or dictionaries 'Ra_n'/'Za_n'."
        )

    return normalize_axis_array(
        axis_config = list(rows_by_mode.values()),
    )





def normalize_axis_array(
        axis_config,
    ):
    arr = np.asarray(axis_config, dtype = float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "Axis array must have shape (k, 3) with columns [n, R, Z]."
        )

    return arr





#==============================================================================================================
# OPTIONAL POST-CONSTRUCTION UPDATES
#==============================================================================================================

def apply_optional_equilibrium_updates(
        eq,
        initial_config,
        run_config,
    ):
    resolution_config = run_config.get("resolution", {})

    if initial_config.get("change_resolution_after_load", False):
        eq.change_resolution(
            L = resolution_config.get("L", None),
            M = resolution_config.get("M", None),
            N = resolution_config.get("N", None),
            L_grid = resolution_config.get("L_grid", None),
            M_grid = resolution_config.get("M_grid", None),
            N_grid = resolution_config.get("N_grid", None),
        )

    return eq





#==============================================================================================================
# GENERAL PARSING HELPERS
#==============================================================================================================

def parse_modes_and_params(
        config,
        allowed_mode_keys,
        allowed_param_keys,
        object_name,
    ):
    mode_data = first_existing_key(
        data = config,
        keys = allowed_mode_keys,
        default = None,
    )

    param_data = first_existing_key(
        data = config,
        keys = allowed_param_keys,
        default = None,
    )

    if isinstance(param_data, dict):
        sorted_items = sorted(
            param_data.items(),
            key = lambda item: int(item[0]),
        )
        modes = [int(key) for key, _ in sorted_items]
        params = [float(value) for _, value in sorted_items]
        return modes, params

    if mode_data is None and param_data is not None:
        params = np.asarray(param_data, dtype = float)
        modes = np.arange(params.size)
        return modes, params

    if mode_data is None or param_data is None:
        raise ValueError(
            f"Could not parse modes and params for {object_name}. "
            "Use either {'modes': [...], 'params': [...]} or {'params': {mode: coeff}}."
        )

    modes = np.asarray(mode_data, dtype = int)
    params = np.asarray(param_data, dtype = float)

    if modes.size != params.size:
        raise ValueError(
            f"Mode/parameter length mismatch for {object_name}: "
            f"len(modes) = {modes.size}, len(params) = {params.size}."
        )

    return modes, params





def parse_mode_key(
        mode_key,
    ):
    if isinstance(mode_key, str):
        cleaned = (
            mode_key
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )

        parts = cleaned.split(",")

        if len(parts) != 2:
            raise ValueError(
                f"Could not parse mode key '{mode_key}'. Expected '(m,n)' or 'm,n'."
            )

        return int(parts[0]), int(parts[1])

    if isinstance(mode_key, (list, tuple)) and len(mode_key) == 2:
        return int(mode_key[0]), int(mode_key[1])

    raise ValueError(
        f"Could not parse mode key '{mode_key}'."
    )





def parse_boundary_entry(
        entry,
        coeff_name,
    ):
    if isinstance(entry, dict):
        l = entry.get("l", 0)
        m = entry["m"]
        n = entry["n"]
        coeff = entry.get(coeff_name, entry.get("coeff", entry.get("value", None)))

        if coeff is None:
            raise ValueError(
                f"Boundary entry missing coefficient '{coeff_name}' or 'coeff': {entry}"
            )

        return l, m, n, coeff

    if len(entry) == 4:
        m, n, coeff, *_ = entry
        return 0, m, n, coeff

    if len(entry) == 5:
        l, m, n, coeff, *_ = entry
        return l, m, n, coeff

    raise ValueError(
        "Boundary entries must be dicts, [m, n, coeff], or [l, m, n, coeff]."
    )





def first_existing_key(
        data,
        keys,
        default = None,
    ):
    for key in keys:
        if key in data:
            return data[key]

    return default





def merge_nested_dicts(
        base,
        override,
    ):
    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_nested_dicts(
                base = merged[key],
                override = value,
            )

        else:
            merged[key] = value

    return merged





def drop_none_values(
        data,
    ):
    return {
        key: value
        for key, value in data.items()
        if value is not None
    }

#==============================================================================================================