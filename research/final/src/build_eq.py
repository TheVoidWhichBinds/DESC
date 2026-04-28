from pathlib import Path
from desc.io import load










#============== EQUILIBRIUM LOADING ==============================================================================
def build_initial_equilibrium(
        run_config,
    ):
    initial_config = run_config["initial_equilibrium"]

    if initial_config["source_type"] != "h5":
        raise NotImplementedError(
            f"Initial equilibrium source_type '{initial_config['source_type']}' is not implemented yet."
        )

    eq_obj = load(initial_config["path"])

    if initial_config.get("load_final_member_if_family", True):
        eq_obj = get_final_equilibrium_from_loaded_object(
            eq_obj = eq_obj,
        )

    return eq_obj
#==============================================================================================================










#========== get_final_equilibrium_from_loaded_object =============================================================
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