import json
import math
import pickle
from pathlib import Path










#============== SAVE HELPERS =====================================================================================
def save_run_config(
        run_config,
        path,
    ):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    safe_config = make_json_safe(
        value = run_config,
    )

    with open(path, "w") as f:
        json.dump(safe_config, f, indent = 4)










def save_equilibrium(
        eq,
        path,
    ):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    eq.save(str(path))










def save_pickle(
        obj,
        path,
    ):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)
#==============================================================================================================










#============== JSON SAFETY ======================================================================================
def make_json_safe(
        value,
    ):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(
                value = item,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(
                value = item,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(
                value = item,
            )
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if callable(value):
        return get_callable_name(
            value = value,
        )

    if isinstance(value, float):
        if math.isinf(value):
            if value > 0:
                return "inf"

            return "-inf"

        if math.isnan(value):
            return "nan"

        return value

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if hasattr(value, "tolist"):
        converted = value.tolist()

        return make_json_safe(
            value = converted,
        )

    return repr(value)
#==============================================================================================================










#========== get_callable_name ====================================================================================
def get_callable_name(
        value,
    ):
    module = getattr(value, "__module__", None)
    name = getattr(value, "__name__", None)

    if module is None or name is None:
        return repr(value)

    return f"{module}.{name}"
#==============================================================================================================