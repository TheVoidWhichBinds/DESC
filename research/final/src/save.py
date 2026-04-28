import json
import pickle
from pathlib import Path










#============== SAVE HELPERS =====================================================================================
def save_run_config(
        run_config,
        path,
    ):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with open(path, "w") as f:
        json.dump(run_config, f, indent = 4)




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