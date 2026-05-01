# driver.py
#===================================================================================================================================================

from pathlib import Path
import os
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "JAX_PLATFORM_NAME" in os.environ:
    del os.environ["JAX_PLATFORM_NAME"]
from desc import set_device
set_device("gpu")

from eq import run_equilibrium
from opt import run_optimization
from helper import (
    load_config_module,
    prepare_run_directory,
    save_equilibrium,
    save_failure_trace,
    save_optimization_log,
    save_plots_from_run_folder,
)

#===================================================================================================================================================










#===========================================================
# MAIN
#===========================================================

def main():
    """
    Top-level driver.

    This intentionally only performs:
        1. one equilibrium run,
        2. one FLO optimization,
        3. one FNO optimization,
        4. saves for all three results.
    """

    config_path = Path(__file__).resolve().with_name("config.py")
    out_dir_path = Path(__file__).resolve().with_name("outputs")

    config = load_config_module(
        config_path = config_path,
    )

    out_dir = prepare_run_directory(
        out_dir = out_dir_path,
        config_path = config_path,
    )

    eq_config = config.EQ_CONFIG
    opt_config = config.OPT_CONFIG
    optimizer = opt_config["optimizer"]

    print("")
    print("#===========================================================")
    print("# EQUILIBRIUM RUN")
    print("#===========================================================")
    print("")

    try:
        eq_init = run_equilibrium(
            eq_config = eq_config,
        )

        save_equilibrium(
            eq = eq_init,
            out_dir = out_dir,
            filename = "eq_init.h5",
        )

    except Exception as error:
        save_failure_trace(
            out_dir = out_dir,
            filename = "eq_FAILURE.txt",
            error = error,
        )
        raise

    print("")
    print("#===========================================================")
    print("# FLO OPTIMIZATION")
    print("#===========================================================")
    print("")

    try:
        eq_FLO, result_FLO = run_optimization(
            eq_0 = eq_init,
            optimizer = optimizer,
            opt_config = opt_config,
            formulation = "FLO",
        )

        save_equilibrium(
            eq = eq_FLO,
            out_dir = out_dir,
            filename = "opt_FLO.h5",
        )

        save_optimization_log(
            result = result_FLO,
            out_dir = out_dir,
            filename = "FLO_OPT_LOG.txt",
        )

    except Exception as error:
        save_failure_trace(
            out_dir = out_dir,
            filename = "FLO_FAILURE.txt",
            error = error,
        )

    print("")
    print("#===========================================================")
    print("# FNO OPTIMIZATION")
    print("#===========================================================")
    print("")

    try:
        eq_FNO, result_FNO = run_optimization(
            eq_0 = eq_init,
            optimizer = optimizer,
            opt_config = opt_config,
            formulation = "FNO",
        )

        save_equilibrium(
            eq = eq_FNO,
            out_dir = out_dir,
            filename = "opt_FNO.h5",
        )

        save_optimization_log(
            result = result_FNO,
            out_dir = out_dir,
            filename = "FNO_OPT_LOG.txt",
        )

    except Exception as error:
        save_failure_trace(
            out_dir = out_dir,
            filename = "FNO_FAILURE.txt",
            error = error,
        )



    print("")
    print("#===========================================================")
    print("# PLOTS")
    print("#===========================================================")
    print("")


    save_plots_from_run_folder(run_dir = out_dir,)

    
    print("")
    print("#===========================================================")
    print("# COMPLETE")
    print("#===========================================================")
    print("")
    print(f"Saved outputs to: {out_dir}")







if __name__ == "__main__":
    main()

#===================================================================================================================================================