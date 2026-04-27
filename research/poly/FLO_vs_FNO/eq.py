# eq.py

#===================================================================================================================================================
import contextlib
import io
import time
import traceback
import warnings
import sys
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium

from .helper import (
    _extract_last_equilibrium,
    _has_automatic_continuation_failure,
    Tee,
)
#===================================================================================================================================================











#============== EQUILIBRIUM SOLVER ============================================================================================================================
def run_equilibrium(
        eq_config,
    ):
    """
    Builds the raw equilibrium, runs automatic continuation, keeps only the final
    continuation step, captures the full continuation log, and returns a status dict for file-based troubleshooting.

    Returns:
        eq_raw,
        eq_init,
        continuation_status,
        continuation_log
    """

    #-----------------------------------------------
    # Unpacking equilibrium configuration variables:
    surface_init = eq_config["surface_init"]
    pressure_init = eq_config["pressure_init"]
    iota_init = eq_config["iota_init"]
    eq_resolution = eq_config["eq_resolution"]
    #-----------------------------------------------

    continuation_status = {
        "equilibrium_built": False,
        "continuation_returned": False,
        "exception_raised": False,
        "automatic_continuation_failure": False,
        "failure_stage": None,
        "message": None,
        "runtime_seconds": None,
        "num_steps_returned": None,
    }

    continuation_log_buffer = io.StringIO()
    eq_raw = None
    eq_init = None
    caught_warnings = []
    t0 = time.perf_counter()

    #------------------------------------------------------------
    # Build initial equilibrium and solve continuation:
    try:
        L, M, N = eq_resolution

        with contextlib.redirect_stdout(Tee(sys.stdout, continuation_log_buffer)), contextlib.redirect_stderr(Tee(sys.stderr, continuation_log_buffer)):
            eq_raw = Equilibrium(
                L = L,
                M = M,
                N = N,
                surface = surface_init,
                pressure = pressure_init,
                iota = iota_init,
                Psi = 1.0,
                ensure_nested = True,
            )

        continuation_status["equilibrium_built"] = True

        with warnings.catch_warnings(record = True) as caught_warnings:
            warnings.simplefilter("always")

            with contextlib.redirect_stdout(Tee(sys.stdout, continuation_log_buffer)), contextlib.redirect_stderr(Tee(sys.stderr, continuation_log_buffer)):
                continuation_result = solve_continuation_automatic(
                    eq_raw.copy(),
                    verbose = 3,
                    ftol = 1e-4,
                    xtol = 1e-4,
                    gtol = 1e-4,
                )

        continuation_status["runtime_seconds"] = time.perf_counter() - t0

        eq_init, num_steps_returned = _extract_last_equilibrium(
            continuation_result
        )
        continuation_status["num_steps_returned"] = num_steps_returned

        continuation_log = continuation_log_buffer.getvalue()

        if len(caught_warnings) > 0:
            continuation_log += "\n\n# Python warnings captured during continuation\n"
            for warning_item in caught_warnings:
                continuation_log += (
                    f"{warning_item.category.__name__}: "
                    f"{warning_item.message}\n"
                )

        continuation_status["continuation_returned"] = eq_init is not None
        continuation_status["automatic_continuation_failure"] = _has_automatic_continuation_failure(
            continuation_log
        )

        if continuation_status["automatic_continuation_failure"]:
            continuation_status["failure_stage"] = "continuation_warning"
            continuation_status["message"] = "WARNING: Automatic continuation failed"
            eq_init = None

        elif not continuation_status["continuation_returned"]:
            continuation_status["failure_stage"] = "continuation_result"
            continuation_status["message"] = "Continuation returned no equilibrium."

        else:
            continuation_status["message"] = "Continuation completed successfully."

        return eq_raw, eq_init, continuation_status, continuation_log

    except Exception:
        continuation_status["runtime_seconds"] = time.perf_counter() - t0
        continuation_status["exception_raised"] = True

        if continuation_status["equilibrium_built"]:
            continuation_status["failure_stage"] = "continuation_exception"
        else:
            continuation_status["failure_stage"] = "equilibrium_build_exception"

        continuation_log = continuation_log_buffer.getvalue()
        error_trace = traceback.format_exc()

        if len(caught_warnings) > 0:
            continuation_log += "\n\n# Python warnings captured during continuation\n"
            for warning_item in caught_warnings:
                continuation_log += (
                    f"{warning_item.category.__name__}: "
                    f"{warning_item.message}\n"
                )

        continuation_log += "\n\n# Exception traceback\n"
        continuation_log += error_trace

        continuation_status["automatic_continuation_failure"] = _has_automatic_continuation_failure(
            continuation_log
        )

        if continuation_status["automatic_continuation_failure"]:
            continuation_status["message"] = "WARNING: Automatic continuation failed"
        else:
            continuation_status["message"] = error_trace.strip().splitlines()[-1]

        return eq_raw, None, continuation_status, continuation_log
#==============================================================================================================================================================