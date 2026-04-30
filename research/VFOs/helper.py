# helper.py
#===================================================================================================================================================

from pathlib import Path
import importlib.util
import pickle
import re
import traceback

import numpy as np

from desc.geometry import FourierRZToroidalSurface

#===================================================================================================================================================










#===========================================================
# CONFIG HELPERS
#===========================================================

def iota_between_rationals(
        iota_axis: float,
    ):
    """
    Generates bounds for iota between low-order rational surfaces.
    """

    allowed_ranges = [
        (0.0,    0.25),
        (0.25,   0.3333),
        (0.3333, 0.5),
        (0.5,    0.6667),
        (0.6667, 0.75),
        (0.75,   1.0),
        (1.0,    1.25),
        (1.25,   1.3333),
        (1.3333, 1.5),
        (1.5,    1.6667),
        (1.6667, 2.0),
        (2.0,    3.0),
        (3.0,    4.0),
    ]

    for lower, upper in allowed_ranges:
        if lower <= iota_axis <= upper:
            return lower, upper

    raise ValueError("iota_axis is outside all allowed rational intervals.")





def load_config_module(
        config_path,
    ):
    """
    Load a Python config file from disk.
    """

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    spec = importlib.util.spec_from_file_location(
        "run_config",
        config_path,
    )

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config





def prepare_run_directory(
        out_dir,
        config_path,
    ):
    """
    Create the output directory and save the active config.
    """

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    write_config_readme(
        out_dir = out_dir,
        config_path = config_path,
    )

    return out_dir

#===================================================================================================================================================










#===========================================================
# SURFACE SOURCE HELPERS
#===========================================================

def _load_pickle_file(
        filepath,
    ):
    """
    Load a pickle file from disk.
    """

    path = Path(filepath).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)





def _select_surface_points(
        dataset,
        selection_method,
    ):
    """
    Select saved surface descriptions from a dataset.
    """

    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Dataset must be a non-empty list.")

    if selection_method == "all_nested":
        selected = [
            point for point in dataset
            if bool(point.get("labels", {}).get("is_nested", False))
        ]

    elif selection_method == "all_build_ok":
        selected = [
            point for point in dataset
            if bool(point.get("labels", {}).get("build_ok", False))
        ]

    else:
        raise ValueError(
            f"Unsupported selection_method='{selection_method}'."
        )

    if len(selected) == 0:
        raise ValueError(
            f"No surfaces found for selection_method='{selection_method}'."
        )

    return selected





def _surface_from_features(
        features,
        NFP,
    ):
    """
    Reconstruct FourierRZToroidalSurface from saved feature data.
    """

    required_keys = [
        "R_lmn",
        "Z_lmn",
        "modes_R",
        "modes_Z",
    ]

    missing = [
        key for key in required_keys
        if key not in features
    ]

    if missing:
        raise KeyError(
            f"Saved feature block is missing required keys: {missing}"
        )

    return FourierRZToroidalSurface(
        R_lmn = features["R_lmn"],
        modes_R = features["modes_R"],
        Z_lmn = features["Z_lmn"],
        modes_Z = features["modes_Z"],
        NFP = NFP,
    )





def load_surface_pool(
        surface_source_config,
        NFP,
        N_eq,
    ):
    """
    Load and reconstruct the requested surface pool.
    """

    dataset = _load_pickle_file(
        filepath = surface_source_config["dataset_path"],
    )

    selection_method = surface_source_config.get(
        "selection_method",
        "all_nested",
    )

    shuffle = surface_source_config.get(
        "shuffle",
        True,
    )

    shuffle_seed = surface_source_config.get(
        "shuffle_seed",
        None,
    )

    selected_points = _select_surface_points(
        dataset = dataset,
        selection_method = selection_method,
    )

    if shuffle:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(selected_points)

    if N_eq is not None:
        selected_points = selected_points[:N_eq]

    surface_pool = []

    for selected_point in selected_points:
        surface_init = _surface_from_features(
            features = selected_point["features"],
            NFP = NFP,
        )

        surface_pool.append(
            {
                "surface_init": surface_init,
                "selected_point": selected_point,
            }
        )

    return surface_pool

#===================================================================================================================================================










#===========================================================
# OBJECTIVE CONFIG HELPERS
#===========================================================

def resolve_from_context(
        func,
    ):
    """
    Mark a callable as a runtime-context resolver.
    """

    func._resolve_from_context = True

    return func





@resolve_from_context
def _eq(
        context,
    ):
    """
    Resolve active equilibrium from runtime context.
    """

    return context["eq_0"]





def _resolve_value(
        value,
        context,
    ):
    """
    Resolve one value from runtime context if marked.
    """

    if callable(value) and getattr(value, "_resolve_from_context", False):
        return value(context)

    return value





def _resolve_kwargs(
        kwargs,
        context,
    ):
    """
    Resolve all marked runtime values in a kwargs dict.
    """

    return {
        key: _resolve_value(
            value = value,
            context = context,
        )
        for key, value in kwargs.items()
    }





def _parse_toggle_entry(
        toggle,
        key,
        kind,
    ):
    """
    Parse one objective/constraint toggle.
    """

    entry = toggle.get(
        key,
        {
            "use": False,
            "kwargs": {},
        },
    )

    if not isinstance(entry, dict):
        raise TypeError(
            f"{kind} toggle '{key}' must be a dict."
        )

    use = entry.get("use", False)
    kwargs = entry.get("kwargs", {})

    if not isinstance(use, bool):
        raise TypeError(
            f"{kind} toggle '{key}' has non-bool use={use!r}."
        )

    if not isinstance(kwargs, dict):
        raise TypeError(
            f"{kind} toggle '{key}' has non-dict kwargs={kwargs!r}."
        )

    return use, dict(kwargs)





def _append_term(
        term_list,
        toggle,
        registry,
        key,
        context,
        kind,
    ):
    """
    Append one enabled objective or constraint.
    """

    use, user_kwargs = _parse_toggle_entry(
        toggle = toggle,
        key = key,
        kind = kind,
    )

    if not use:
        return

    if key not in registry:
        raise KeyError(
            f"{kind} toggle '{key}' is active, but no registry entry exists."
        )

    wrapper = registry[key]["wrapper"]
    defaults = registry[key].get("defaults", {})

    kwargs = {
        **defaults,
        **user_kwargs,
    }

    kwargs = _resolve_kwargs(
        kwargs = kwargs,
        context = context,
    )

    term_list.append(wrapper(**kwargs))





def _append_terms(
        term_list,
        toggle,
        registry,
        context,
        kind,
    ):
    """
    Append all active terms from a registry.
    """

    for key in registry:
        _append_term(
            term_list = term_list,
            toggle = toggle,
            registry = registry,
            key = key,
            context = context,
            kind = kind,
        )

#===================================================================================================================================================










#===========================================================
# SAVE HELPERS
#===========================================================

def save_equilibrium(
        eq,
        out_dir,
        filename,
    ):
    """
    Save a DESC equilibrium.
    """

    if eq is None:
        return None

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    save_path = out_dir / filename
    eq.save(str(save_path))

    return save_path





def save_failure_trace(
        out_dir,
        filename,
        error,
    ):
    """
    Save traceback information from a failed stage.
    """

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    save_path = out_dir / filename

    with open(save_path, "w") as f:
        f.write(str(error))
        f.write("\n\n")
        f.write(traceback.format_exc())

    return save_path





def write_config_readme(
        out_dir,
        config_path,
    ):
    """
    Save active config.py contents into README.md.
    """

    out_dir = Path(out_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()

    readme_path = out_dir / "README.md"

    with open(config_path, "r") as f:
        config_text = f.read()

    with open(readme_path, "w") as f:
        f.write("```python\n")
        f.write(config_text)

        if not config_text.endswith("\n"):
            f.write("\n")

        f.write("```\n")

    return readme_path

#===================================================================================================================================================