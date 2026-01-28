# worker_misc.py
# function libraries for workflow.py
# contains functions that are not related to pyaedt,
# functions here should not be modified within different optimization tasks.

from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_VALID_NAME = re.compile(r"^[A-Za-z_]\w*\Z")


def bootstrap_home_dirs(base: Path) -> None:
    """
    方案B关键点：覆盖式设置，而不是 setdefault。
    确保 TEMP/TMP/APPDATA/LOCALAPPDATA 等都指向作业沙盒可写目录。
    """
    mapping = {
        "USERPROFILE": base / "_home",
        "HOME": base / "_home",
        "APPDATA": base / "_appdata",
        "LOCALAPPDATA": base / "_localappdata",
        "TEMP": base / "_tmp",
        "TMP": base / "_tmp",
        "TMPDIR": base / "_tmp",
    }
    for k, p in mapping.items():
        os.environ[k] = str(p)
        p.mkdir(parents=True, exist_ok=True)


def dump_runtime_env() -> None:
    print(sys.executable, sys.version, file=sys.stderr, flush=True)
    keys = [
        "ANSYSEM_ROOT241",
        "AWP_ROOT241",
        "ANSYSLIC_DIR",
        "PATH",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
    ]
    for k in keys:
        print(f"{k}={os.environ.get(k)}", file=sys.stderr, flush=True)


def _append_cost_to_cost_json(cost_json_path: str | Path, key: str, cost: float) -> None:
    """
    Append a named cost entry to cost.json (for human readability and post-processing).
    The finalize_cost_json function should be called at the end to add the machine-readable "costs" array.
    """
    path = Path(cost_json_path)

    key_json = json.dumps(str(key), ensure_ascii=False).encode("utf-8")
    val_json = json.dumps(float(cost)).encode("utf-8")

    entry_first = b"  " + key_json + b": " + val_json + b"\n"
    entry_next = b"  , " + key_json + b": " + val_json + b"\n"

    if (not path.exists()) or path.stat().st_size == 0:
        path.write_bytes(b"{\n" + entry_first + b"}\n")
        return

    data = path.read_bytes()

    i = len(data) - 1
    while i >= 0 and data[i] in b" \t\r\n":
        i -= 1
    if i < 0:
        path.write_bytes(b"{\n" + entry_first + b"}\n")
        return

    if data[i : i + 1] != b"}":
        raise ValueError(f"{cost_json_path} is not a JSON object (missing closing '}}').")

    idx = i
    prefix = data[:idx]
    suffix = data[idx:]

    has_entries = b":" in prefix
    needs_nl = not prefix.endswith((b"\n", b"\r"))
    nl = b"\n" if needs_nl else b""

    insertion = nl + (entry_next if has_entries else entry_first)
    path.write_bytes(prefix + insertion + suffix)


def finalize_cost_json(
    cost_json_path: str | Path,
    cost_keys_in_order: Sequence[str],
    *,
    default_cost: float = 1.0,
) -> None:
    """
    Add an ordered 'costs' array to cost.json for machine reading by batch_eval.py.
    
    The named cost keys remain in the JSON for human readability and post-processing.
    The 'costs' array is the canonical machine-readable format that batch_eval.py reads.
    
    Args:
        cost_json_path: Path to cost.json
        cost_keys_in_order: Tuple of cost key names in the order expected by the optimizer
        default_cost: Value to use for missing cost keys (error fallback)
    """
    path = Path(cost_json_path)

    if not path.exists() or path.stat().st_size == 0:
        # No costs were written; create a default structure
        data = {key: default_cost for key in cost_keys_in_order}
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {key: default_cost for key in cost_keys_in_order}

    # Build ordered costs array
    costs_array = []
    for key in cost_keys_in_order:
        val = data.get(key, default_cost)
        try:
            costs_array.append(float(val))
        except (TypeError, ValueError):
            costs_array.append(default_cost)

    data["costs"] = costs_array

    # Write back with nice formatting
    path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")


def softmax(result: Any, goal: float, worst: float, *, error_cost: float = 1.1) -> float:
    if result is False or result is None:
        return float(error_cost)
    try:
        r, goal, worst = float(result), float(goal), float(worst)
    except Exception:
        return float(error_cost)
    if not (math.isfinite(r) and math.isfinite(goal) and math.isfinite(worst)) or goal == worst:
        return float(error_cost)

    # Reference form (algebraically simplified):
    # (tanh(4/(worst-goal)*r + 4*worst/(goal-worst) + 2) + 1)/2
    return float((math.tanh(4.0 * (r - worst) / (worst - goal) + 2.0) + 1.0) / 2.0)


def exported_npz_path(expression: str, out_dir: str | Path = "rawData") -> str | None:
    """Return the npz path if an exported file exists whose meta.expression == expression."""
    out = Path(out_dir)
    if not out.is_dir():
        return None
    for p in out.glob("*.npz"):
        try:
            with np.load(p, allow_pickle=False) as z:
                if "meta" in z.files and json.loads(z["meta"].item()).get("expression") == expression:
                    return str(p)
        except Exception:
            pass
    return None


def _as_float_parameter_value(p: Any) -> float:
    v = float(getattr(p, "value"))
    if math.isnan(v) and hasattr(p, "denorm"):
        try:
            v = float(p.denorm(update=False))
        except Exception:
            pass
    return v


def evaluate_constraints(
    module_name: str = "parameters_constraints",
    *,
    extra_env: dict[str, Any] | None = None,
) -> list[float]:
    pc = importlib.import_module(module_name)
    try:
        pc = importlib.reload(pc)
    except Exception:
        pass

    parameters: Iterable[Any] = getattr(pc, "PARAMETERS")
    constraints: Iterable[str] = getattr(pc, "CONSTRAINTS")

    env: dict[str, Any] = {"math": math, "abs": abs, "min": min, "max": max, "pow": pow}
    if extra_env:
        env.update(extra_env)

    for p in parameters:
        name = str(getattr(p, "name"))
        if not _VALID_NAME.match(name):
            raise ValueError(f"Invalid parameter name for eval environment: {name!r}")
        env[name] = _as_float_parameter_value(p)

    values: list[float] = []
    for expr in constraints:
        if not isinstance(expr, str):
            raise TypeError(f"Constraint must be a string expression, got {type(expr).__name__}")
        try:
            values.append(float(eval(expr, {"__builtins__": {}}, env)))
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate constraint {expr!r}: {e}") from e
    return values


def cost_constraints(
    module_name: str = "parameters_constraints",
    *,
    cost_json_path: str | Path = "cost.json",
    key: str = "cost_constraints",
) -> float:
    vals = evaluate_constraints(module_name)
    vals = [v if v <= 0 else 0.0 for v in vals]
    costs = [softmax(v, goal=0.0, worst=-1.0) for v in vals]
    cost = float(sum(costs) / len(costs)) if costs else 0.0
    _append_cost_to_cost_json(cost_json_path, key, cost)
    return cost