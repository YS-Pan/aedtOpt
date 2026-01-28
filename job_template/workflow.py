# workflow.py
# the workflow of getting the cost for a single individual.
# the individual is within the population of a optimization algorithm.
# Usually heavily modifed for different optimization task.
# Though identical in different individuals within a optimization task.
# in principle, should be manually coded from ground up for each optimization task.
# For the ease of "coding from ground up", avoid adding contents in this file if possible.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hfss_com import analyze, save_modal, set_para, solver_init, solver_exit, set_hfss_temp_directory
from worker_misc import (
    _append_cost_to_cost_json,
    cost_constraints,
    finalize_cost_json,
    softmax,
    bootstrap_home_dirs,
    dump_runtime_env,
)

BASE_DIR = Path(__file__).resolve().parent
COST_JSON = BASE_DIR / "cost.json"

# Cost keys in the order expected by the optimizer.
# This order defines the mapping from named costs to the optimizer's objective indices.
# batch_eval.py reads only the "costs" array, not these names.
COST_KEYS_IN_ORDER = ("cost_s11", "cost_s22", "cost_s33", "cost_s32", "cost_s31", "cost_constraints")

ERROR_COST = 1.1


def _result_from_npz(npz_path: str | Path, goal: float, worst: float) -> float:
    with np.load(npz_path, allow_pickle=False) as z:
        if "data" in z.files:
            y = np.asarray(z["data"])
        else:
            y = np.asarray(z["real"])  # backward compatibility

        if np.iscomplexobj(y):
            y = np.real(y)

        x = y.astype(float).ravel()

    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"No finite data in NPZ: {npz_path}")

    avg = float(x.mean())
    ext = float(x.max() if goal < worst else x.min())
    return 0.8 * ext + 0.2 * avg


def _cost_modal(hfssApp, expression: str, goalValue: float, worstValue: float, *, cost_json_path: str | Path) -> float:
    key = sys._getframe(1).f_code.co_name
    try:
        path = save_modal(hfssApp, expression)
        result = _result_from_npz(path, goalValue, worstValue)
        cost = softmax(result, goalValue, worstValue)
    except Exception:
        import traceback

        traceback.print_exc()
        cost = softmax(False, goalValue, worstValue)
    _append_cost_to_cost_json(cost_json_path, key, cost)
    return cost


def cost_s11(hfssApp, goalValue: float = -20, worstValue: float = -5, *, cost_json_path: str | Path = COST_JSON) -> float:
    return _cost_modal(hfssApp, "dB(S(1,1))", goalValue, worstValue, cost_json_path=cost_json_path)


def cost_s22(hfssApp, goalValue: float = 0, worstValue: float = -20, *, cost_json_path: str | Path = COST_JSON) -> float:
    return _cost_modal(hfssApp, "dB(S(2,2))", goalValue, worstValue, cost_json_path=cost_json_path)


def cost_s33(hfssApp, goalValue: float = -20, worstValue: float = -5, *, cost_json_path: str | Path = COST_JSON) -> float:
    return _cost_modal(hfssApp, "dB(S(3,3))", goalValue, worstValue, cost_json_path=cost_json_path)


def cost_s32(hfssApp, goalValue: float = -40, worstValue: float = -5, *, cost_json_path: str | Path = COST_JSON) -> float:
    return _cost_modal(hfssApp, "dB(S(3,2))", goalValue, worstValue, cost_json_path=cost_json_path)


def cost_s31(hfssApp, goalValue: float = 0, worstValue: float = -10, *, cost_json_path: str | Path = COST_JSON) -> float:
    return _cost_modal(hfssApp, "dB(S(3,1))", goalValue, worstValue, cost_json_path=cost_json_path)


if __name__ == "__main__":
    bootstrap_home_dirs(BASE_DIR)
    hfssApp = None
    try:
        # Start from a clean, empty JSON object.
        # Costs will be appended as they are computed (no duplicate keys from prefill).
        COST_JSON.write_text("{\n}\n", encoding="utf-8", newline="\n")
        dump_runtime_env()
        hfssApp, *_ = solver_init()
        set_hfss_temp_directory(hfssApp, "TEMP")

        set_para(hfssApp)
        analyze(hfssApp)

        cost_s11(hfssApp)
        cost_s22(hfssApp)
        cost_s33(hfssApp)
        cost_s32(hfssApp)
        cost_s31(hfssApp)

        cost_constraints(cost_json_path=COST_JSON)

    except Exception as e:
        print(f"[workflow] Exception: {e}", file=sys.stderr, flush=True)
    finally:
        # Always finalize cost.json to add the "costs" array for batch_eval.py
        # This ensures machine-readable output even if some costs failed
        finalize_cost_json(COST_JSON, COST_KEYS_IN_ORDER, default_cost=ERROR_COST)

        if hfssApp is not None:
            solver_exit(hfssApp, cleanup_results=False)