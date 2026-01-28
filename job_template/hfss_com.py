# hfss_com.py
# function libraries for workflow.py
# contains functions related to pyaedt.
# functions here should not be modified within different optimization tasks.

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# -----------------------------
# helpers
# -----------------------------
def _scan_single_aedt_file(folder: str = ".") -> str:
    files = sorted(glob.glob(os.path.join(folder, "*.aedt")))
    if not files:
        raise FileNotFoundError(f"No .aedt project found in: {os.path.abspath(folder)}")
    if len(files) != 1:
        raise RuntimeError(f"Multiple .aedt projects found. Please specify projectName: {files}")
    return files[0]


def _sanitize_filename(s: str, max_len: int = 180) -> str:
    """Make a safe filename stem from an expression (sanitizing is allowed)."""
    name = str(s).strip()
    # replace path separators and common illegal filename chars
    name = name.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", name)
    name = name.strip(" .\t")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)

    if not name:
        name = "unnamed"

    stem = os.path.splitext(name)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if stem in reserved:
        name = "_" + name

    return name[:max_len]


def _set_local_variables_low_level(hfss: Any, name_to_value: Dict[str, str]) -> None:
    changed_props: List[Any] = ["NAME:ChangedProps"]
    for name, val in name_to_value.items():
        changed_props.append(["NAME:" + str(name), "Value:=", str(val)])

    hfss._odesign.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:LocalVariableTab",
                ["NAME:PropServers", "LocalVariables"],
                changed_props,
            ],
        ]
    )


def _parse_setup_sweep(spec: Optional[Union[str, Dict[str, str]]]) -> Tuple[Optional[str], Optional[str]]:
    if spec is None:
        return None, None
    if isinstance(spec, dict):
        setup = (spec.get("setup") or spec.get("Setup") or "").strip() or None
        sweep = (spec.get("sweep") or spec.get("Sweep") or "").strip() or None
        return setup, sweep
    text = str(spec).strip()
    if ":" in text:
        setup, sweep = [x.strip() for x in text.split(":", 1)]
        return setup or None, sweep or None
    return text or None, None


def _print_setup_sweep(setup_sweep_name: str) -> None:
    setup, sweep = _parse_setup_sweep(setup_sweep_name)
    print(f"[hfss_com] setup={setup!r} sweep={sweep!r}")


def _infer_unique_setup(hfss: Any, requested_setup: Optional[str]) -> str:
    setups = hfss.get_setups()
    if requested_setup:
        if requested_setup not in setups:
            raise RuntimeError(f"Setup '{requested_setup}' not found. Available: {setups}")
        print(f"[hfss_com] setup={requested_setup!r} sweep={None!r}")
        return requested_setup
    if len(setups) == 1:
        print(f"[hfss_com] setup={setups[0]!r} sweep={None!r}")
        return setups[0]
    raise RuntimeError(f"Cannot infer unique setup. Available setups: {setups}. Please specify analyzeSetup.")


def _infer_setup_sweep_name(hfss: Any) -> str:
    for attr in ("nominal_sweep", "nominal_adaptive"):
        v = getattr(hfss, attr, None)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            _print_setup_sweep(name)
            return name

    names = getattr(hfss, "setup_sweeps_names", None)
    if isinstance(names, (list, tuple)) and len(names) == 1 and str(names[0]).strip():
        name = str(names[0]).strip()
        _print_setup_sweep(name)
        return name
    if isinstance(names, (list, tuple)) and len(names) > 1:
        raise RuntimeError(f"Multiple setup/sweep names found: {names}. Pass setup_sweep_name explicitly.")

    setups = hfss.get_setups()
    if len(setups) == 1:
        setup = setups[0]
        sweeps = hfss.get_sweeps(setup) or []
        if len(sweeps) == 1:
            name = f"{setup} : {sweeps[0]}"
            _print_setup_sweep(name)
            return name
        if len(sweeps) == 0:
            name = setup
            _print_setup_sweep(name)
            return name

    raise RuntimeError("Cannot infer setup_sweep_name. Pass setup_sweep_name explicitly.")


def _iter_any(values: Any) -> List[Any]:
    """Convert unknown 'values' into a safe python list without using truthiness."""
    if values is None:
        return []
    # numpy array
    if isinstance(values, np.ndarray):
        return values.ravel().tolist()
    # common containers
    if isinstance(values, (list, tuple)):
        return list(values)
    # strings should be treated as a single item (not iterated char-by-char)
    if isinstance(values, (str, bytes)):
        return [values]
    # pandas Series / other iterables
    try:
        return list(values)
    except Exception:
        return [values]


def _to_float_array(values: Any) -> np.ndarray:
    """
    Convert an iterable/scalar of mixed values into a float numpy array.
    Robust to numpy arrays and avoids 'values or []' ambiguity.
    """
    seq = _iter_any(values)
    out: List[float] = []
    for v in seq:
        try:
            out.append(float(v))
            continue
        except Exception:
            pass

        # Try to extract numeric token from strings like "30GHz", "1.2mm", etc.
        s = str(v)
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        out.append(float(m.group(0)) if m else float("nan"))

    return np.asarray(out, dtype=float)


def _export_solution_data_npz(
    hfss: Any,
    *,
    expression: str,
    report_category: Optional[str] = None,
    context: Optional[Union[str, Dict[str, Any]]] = None,
    variations: Optional[Dict[str, Union[str, List[str]]]] = None,
    setup_sweep_name: Optional[str] = None,
    out_dir: str = "rawData",
) -> str:
    """
    Export SolutionData for one expression to NPZ.

    Writes ONLY the expression result in simplest form:
      - data: float ndarray (expression values only)
      - axis_<axisname>: float axis arrays when parseable (from SolutionData.intrinsics)
      - meta: json string

    On failure, prints diagnostics and writes a placeholder NPZ with data=[nan].
    """
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    if setup_sweep_name is None:
        setup_sweep_name = _infer_setup_sweep_name(hfss)

    safe_name = _sanitize_filename(expression)
    path = os.path.join(out_dir_abs, f"{safe_name}.npz")

    # Normalize variations into dict[str, list[str]] (PostProcessor3D.get_solution_data expects families dict)
    vrs = None
    if variations is not None:
        vrs = {}
        for k, val in variations.items():
            key = str(k)
            if isinstance(val, (list, tuple, np.ndarray)):
                vrs[key] = [str(x) for x in _iter_any(val)]
            else:
                vrs[key] = [str(val)]

    kwargs: Dict[str, Any] = {"expressions": expression, "setup_sweep_name": setup_sweep_name}
    if report_category:
        kwargs["report_category"] = report_category
    if context is not None:
        kwargs["context"] = context
    if vrs is not None:
        kwargs["variations"] = vrs

    def _write_npz(data: np.ndarray, axis_arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        meta_json = json.dumps(meta, default=str)
        save_kwargs: Dict[str, Any] = {"data": data, "meta": meta_json}

        # include numeric axes only (omit if empty)
        for ax, arr in axis_arrays.items():
            if isinstance(arr, np.ndarray) and arr.size:
                save_kwargs[f"axis_{ax}"] = arr

        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **save_kwargs)
        os.replace(tmp, path)

    def _unwrap_single_expression_result(raw: Any, expr: str) -> Any:
        if not isinstance(raw, dict):
            return raw

        if expr in raw:
            return raw[expr]

        # Match ignoring whitespace differences: "S(1,1)" vs "S(1, 1)"
        target = re.sub(r"\s+", "", str(expr))
        for k in raw.keys():
            if re.sub(r"\s+", "", str(k)) == target:
                return raw[k]

        # Fallback: first value
        try:
            return next(iter(raw.values()))
        except Exception:
            return raw

    def _extract_raw_values(sd: Any, expr: str) -> Any:
        # Newer/alternate API
        fn = getattr(sd, "get_expression_data", None)
        if callable(fn):
            try:
                return fn(expr)
            except TypeError:
                return fn()

        # Common PyAEDT SolutionData API
        for name in ("data_real", "data_magnitude", "data_db20", "data_db10", "data_db"):
            fn2 = getattr(sd, name, None)
            if callable(fn2):
                try:
                    out = fn2(expr)
                except TypeError:
                    out = fn2()
                return _unwrap_single_expression_result(out, expr)

        # Some versions expose data as attributes
        for name in ("data_real", "data_magnitude", "data"):
            out = getattr(sd, name, None)
            if out is not None and not callable(out):
                return _unwrap_single_expression_result(out, expr)

        raise AttributeError(
            f"Cannot extract numeric data for {expr!r} from object type {type(sd).__name__}. "
            f"Available keys/attrs are unexpected."
        )

    def _strip_axis_from_data_if_present(
        data: np.ndarray,
        *,
        axes: List[str],
        axis_arrays: Dict[str, np.ndarray],
        primary_sweep: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Some PyAEDT flows return a 2xN (or Nx2) array where one row/col
        is the sweep axis and the other is the expression value.
        We want NPZ 'data' to contain expression values only.
        """
        if data.ndim != 2:
            return data

        axis_ref: Optional[np.ndarray] = None
        if len(axes) == 1:
            axis_ref = axis_arrays.get(axes[0], None)

        if axis_ref is None and isinstance(primary_sweep, np.ndarray) and primary_sweep.size:
            axis_ref = primary_sweep

        def _allclose(u: np.ndarray, v: np.ndarray) -> bool:
            u = np.asarray(u, dtype=float).ravel()
            v = np.asarray(v, dtype=float).ravel()
            if u.shape != v.shape:
                return False
            return bool(np.allclose(u, v, rtol=1e-8, atol=1e-10, equal_nan=False))

        if axis_ref is not None:
            a = np.asarray(axis_ref, dtype=float).ravel()
            n = int(a.size)

            if data.shape == (2, n):
                if _allclose(data[0, :], a):
                    return np.asarray(data[1, :], dtype=float)
                if _allclose(data[1, :], a):
                    return np.asarray(data[0, :], dtype=float)

            if data.shape == (n, 2):
                if _allclose(data[:, 0], a):
                    return np.asarray(data[:, 1], dtype=float)
                if _allclose(data[:, 1], a):
                    return np.asarray(data[:, 0], dtype=float)

            return data

        # Heuristic fallback for 1D sweeps when axis isn't available
        if len(axes) > 1:
            return data

        def _mono_increasing(v: np.ndarray) -> bool:
            v = np.asarray(v, dtype=float).ravel()
            v = v[np.isfinite(v)]
            if v.size < 3:
                return False
            d = np.diff(v)
            return bool(np.all(d >= 0) and np.any(d > 0))

        if data.shape[0] == 2 and data.shape[1] > 2:
            r0 = data[0, :]
            r1 = data[1, :]
            m0 = _mono_increasing(r0)
            m1 = _mono_increasing(r1)
            if m0 and not m1:
                return np.asarray(r1, dtype=float)
            if m1 and not m0:
                return np.asarray(r0, dtype=float)
            return np.asarray(r1, dtype=float)

        if data.shape[1] == 2 and data.shape[0] > 2:
            c0 = data[:, 0]
            c1 = data[:, 1]
            m0 = _mono_increasing(c0)
            m1 = _mono_increasing(c1)
            if m0 and not m1:
                return np.asarray(c1, dtype=float)
            if m1 and not m0:
                return np.asarray(c0, dtype=float)
            return np.asarray(c1, dtype=float)

        return data

    attempted_kwargs: List[Dict[str, Any]] = []
    try:
        # Build a small set of robust attempts (PyAEDT sometimes returns False on failure)
        seen: set[str] = set()

        def _add_attempt(d: Dict[str, Any]) -> None:
            d2 = dict(d)
            if isinstance(d2.get("variations"), dict):
                d2["variations"] = dict(d2["variations"])  # avoid in-place mutations leaking across attempts
            sig = json.dumps(d2, sort_keys=True, default=str)
            if sig not in seen:
                seen.add(sig)
                attempted_kwargs.append(d2)

        _add_attempt(kwargs)

        # fallback: try setup-only if setup:sweep fails
        setup_only, _sweep = _parse_setup_sweep(setup_sweep_name)
        if setup_only and setup_only != setup_sweep_name:
            _add_attempt({**kwargs, "setup_sweep_name": setup_only})

        # fallback: try without variations
        if "variations" in kwargs:
            d = dict(kwargs)
            d.pop("variations", None)
            _add_attempt(d)
            if setup_only and setup_only != setup_sweep_name:
                _add_attempt({**d, "setup_sweep_name": setup_only})

        # fallback: try alternate category for driven terminal vs driven modal
        if report_category in ("Modal Solution Data", "Terminal Solution Data"):
            alt = "Terminal Solution Data" if report_category == "Modal Solution Data" else "Modal Solution Data"
            _add_attempt({**kwargs, "report_category": alt})

        # fallback: try without report_category (let PyAEDT infer)
        if "report_category" in kwargs:
            d = dict(kwargs)
            d.pop("report_category", None)
            _add_attempt(d)

        sdata = None
        used_kwargs: Dict[str, Any] | None = None
        axes: List[str] = []
        axis_arrays: Dict[str, np.ndarray] = {}
        data: np.ndarray | None = None
        last_err: Exception | None = None

        for kw_try in attempted_kwargs:
            try:
                sd = hfss.post.get_solution_data(**kw_try)
                if sd is None or sd is False:
                    raise RuntimeError(f"get_solution_data returned {sd!r}")

                intr = getattr(sd, "intrinsics", None) or {}
                axes_try = list(intr.keys())

                axis_arrays_try: Dict[str, np.ndarray] = {}
                for ax in axes_try:
                    try:
                        axis_arrays_try[ax] = _to_float_array(intr.get(ax))
                    except Exception:
                        continue

                # Primary sweep axis (helps strip axis even if intrinsics are incomplete)
                primary = None
                try:
                    p = _to_float_array(getattr(sd, "primary_sweep_values", None))
                    primary = p if p.size else None
                except Exception:
                    primary = None

                raw = _extract_raw_values(sd, expression)
                arr = np.asarray(raw)
                if np.iscomplexobj(arr):
                    arr = np.real(arr)
                arr = arr.astype(float, copy=False)

                arr = _strip_axis_from_data_if_present(arr, axes=axes_try, axis_arrays=axis_arrays_try, primary_sweep=primary)

                if not np.isfinite(arr).any():
                    raise RuntimeError("Extracted data has no finite values")

                # Optional reshape if axes imply a grid
                shape = [int(axis_arrays_try[ax].size) for ax in axes_try if ax in axis_arrays_try]
                if shape:
                    expected = int(np.prod(shape))
                    if expected > 0 and arr.size == expected:
                        arr = arr.reshape(shape)

                sdata = sd
                used_kwargs = kw_try
                axes = axes_try
                axis_arrays = axis_arrays_try
                data = arr
                break
            except Exception as e:
                last_err = e
                continue

        if sdata is None or used_kwargs is None or data is None:
            raise RuntimeError(f"All get_solution_data attempts failed. Last error: {type(last_err).__name__}: {last_err}")

        meta = {
            "ok": True,
            "expression": expression,
            "file_stem": safe_name,
            "report_category": used_kwargs.get("report_category"),
            "context": used_kwargs.get("context"),
            "setup_sweep_name": used_kwargs.get("setup_sweep_name"),
            "cwd": os.getcwd(),
            "out_dir": out_dir_abs,
            "axes": axes,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "project": getattr(hfss, "project_name", None),
            "design": getattr(hfss, "design_name", None),
            "get_solution_data_kwargs": used_kwargs,
        }

        _write_npz(data, axis_arrays, meta)

        p = Path(path)
        if (not p.is_file()) or p.stat().st_size == 0:
            print(f"[hfss_com] NPZ NOT saved (missing or empty): {path}")
            print(f"[hfss_com] cwd={os.getcwd()} out_dir={out_dir_abs}")
            print(f"[hfss_com] kwargs={used_kwargs}")

        return path

    except Exception as e:
        print(f"[hfss_com] Export failed for expression={expression!r}")
        print(f"[hfss_com] target_npz={path}")
        print(f"[hfss_com] cwd={os.getcwd()} out_dir={out_dir_abs}")
        if attempted_kwargs:
            print(f"[hfss_com] attempted_kwargs={attempted_kwargs}")
        else:
            print(f"[hfss_com] kwargs={kwargs}")
        print(f"[hfss_com] error={type(e).__name__}: {e}")
        print(traceback.format_exc())

        # Write placeholder so pipeline doesn't die and rawData is not empty.
        try:
            meta = {
                "ok": False,
                "expression": expression,
                "file_stem": safe_name,
                "report_category": report_category,
                "context": context,
                "setup_sweep_name": setup_sweep_name,
                "cwd": os.getcwd(),
                "out_dir": out_dir_abs,
                "get_solution_data_kwargs": kwargs,
                "attempted_kwargs": attempted_kwargs,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            _write_npz(np.asarray([np.nan], dtype=float), {}, meta)
            print(f"[hfss_com] Wrote placeholder NPZ with NaN: {path}")
            return path
        except Exception as e2:
            print(f"[hfss_com] Also failed to write placeholder NPZ: {type(e2).__name__}: {e2}")
            print(traceback.format_exc())
            raise


# =========================================================
# Public API
# =========================================================
def set_hfss_temp_directory(hfssApp: Any, path: str | Path | None = None) -> bool:
    """
    Set AEDT/HFSS temporary directory (PyAEDT: Hfss.set_temporary_directory).

    If `path` is None, uses environment TEMP (fallback TMP).
    """
    if path is None:
        path = os.environ.get("TEMP") or os.environ.get("TMP") or ""
    temp = Path(path).resolve()
    os.makedirs(temp, exist_ok=True)
    return bool(hfssApp.set_temporary_directory(str(temp)))


def solver_init(
    projectName: Optional[str] = None,
    designName: Optional[str] = None,
    *,
    non_graphical: bool = True,
    new_desktop: bool = True,
    close_on_exit: bool = False,
    remove_lock: bool = True,
) -> List[Any]:
    from ansys.aedt.core import Hfss

    if projectName is None:
        projectName = _scan_single_aedt_file(".")

    hfss = Hfss(
        project=projectName,
        design=designName,
        non_graphical=non_graphical,
        new_desktop=new_desktop,
        close_on_exit=close_on_exit,
        remove_lock=remove_lock,
    )

    if designName is None:
        designs = hfss.design_list or []
        if len(designs) != 1:
            raise RuntimeError(f"Project '{projectName}' has {len(designs)} designs: {designs}. Please specify designName.")
        designName = designs[0]

    return [hfss, projectName, designName]


def _load_parameters_py_value_only(path: str) -> Dict[str, str]:
    """
    Load PARAMETERS from a python file (e.g. parameters_constraints.py).
    IMPORTANT: normValue is ignored. Always use .value + .unit.
    """
    import math
    import runpy
    import sys

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    folder = os.path.abspath(os.path.dirname(path))
    sys.path.insert(0, folder)
    try:
        ns = runpy.run_path(path)
    finally:
        try:
            sys.path.remove(folder)
        except ValueError:
            pass

    params = ns.get("PARAMETERS", None)
    if params is None:
        raise ValueError(f"{path!r} does not define PARAMETERS")

    out: Dict[str, str] = {}
    for p in params:
        name = getattr(p, "name", None)
        if not name:
            raise ValueError(f"Invalid parameter object (missing .name): {p!r}")

        unit = str(getattr(p, "unit", "") or "")
        val = getattr(p, "value", None)

        try:
            val_f = float(val)
        except Exception:
            raise ValueError(f"Parameter {name!r} has non-numeric value: {val!r}")

        if math.isnan(val_f):
            raise ValueError(f"Parameter {name!r} value is NaN")

        num = f"{val_f:g}"
        out[str(name)] = f"{num}{unit}" if unit else num

    return out


def set_para(hfssApp: Any, paraFile: str = "parameters_constraints.py") -> bool:
    """Read python paraFile (PARAMETERS) and set HFSS Local Variables (value+unit only)."""
    try:
        name_to_value = _load_parameters_py_value_only(paraFile)
        _set_local_variables_low_level(hfssApp, name_to_value)
        return True
    except Exception as e:
        print(f"[hfss_com.set_para] Failed: {e}")
        print(traceback.format_exc())
        return False


def analyze(
    hfssApp: Any,
    analyzeSetup: Optional[Union[str, Dict[str, str]]] = None,
    CPUcores: int = 4,
    ParallelTasks: int = 1,
    allocGPUs: Optional[int] = None,
) -> bool:
    """Solve a setup. use_auto_settings is ALWAYS False."""
    try:
        req_setup, req_sweep = _parse_setup_sweep(analyzeSetup)
        setup = _infer_unique_setup(hfssApp, req_setup)

        if req_sweep:
            sweeps = hfssApp.get_sweeps(setup) or []
            if sweeps and req_sweep not in sweeps:
                raise RuntimeError(f"Sweep '{req_sweep}' not found under setup '{setup}'. Available: {sweeps}")

        ok = hfssApp.analyze_setup(
            name=setup,
            cores=CPUcores,
            tasks=ParallelTasks,
            gpus=allocGPUs,
            use_auto_settings=False,  # forced
        )
        if not ok:
            raise RuntimeError(f"analyze_setup returned False for setup '{setup}'.")
        return True
    except Exception as e:
        print(f"[hfss_com.analyze] Failed: {e}")
        print(traceback.format_exc())
        return False


# -----------------------------
# rawData export (NumPy .npz)
# filename = sanitized(expression) only (no prefixes/timestamps)
# -----------------------------
def save_modal(hfssApp: Any, expression: str, *, setup_sweep_name: Optional[str] = None) -> str:
    return _export_solution_data_npz(
        hfssApp,
        expression=expression,
        report_category="Modal Solution Data",
        variations={"Freq": ["All"]},
        setup_sweep_name=setup_sweep_name,
    )


def save_nearField(
    hfssApp: Any,
    expression: str,
    *,
    context: str = "Line1",
    variations: Optional[Dict[str, Union[str, List[str]]]] = None,
    setup_sweep_name: Optional[str] = None,
) -> str:
    if variations is None:
        variations = {"NormalizedDistance": ["All"], "Freq": ["All"]}
    return _export_solution_data_npz(
        hfssApp,
        expression=expression,
        report_category="Near Fields",
        context=context,
        variations=variations,
        setup_sweep_name=setup_sweep_name,
    )


def save_farField(
    hfssApp: Any,
    expression: str,
    *,
    context: str = "3D",
    setup_sweep_name: Optional[str] = None,
) -> str:
    return _export_solution_data_npz(
        hfssApp,
        expression=expression,
        report_category="Far Fields",
        context=context,
        variations={"Theta": ["All"], "Phi": ["All"], "Freq": ["All"]},
        setup_sweep_name=setup_sweep_name,
    )


def save_antPara(hfssApp: Any, expression: str, *, setup_sweep_name: Optional[str] = None) -> str:
    return _export_solution_data_npz(
        hfssApp,
        expression=expression,
        report_category="Antenna Parameters",
        variations={"Freq": ["All"]},
        setup_sweep_name=setup_sweep_name,
    )


def solver_exit(hfssApp: Any, *, save_project: bool = True, cleanup_results: bool = True) -> bool:
    ok = True

    project_file = getattr(hfssApp, "project_file", "") or ""
    project_dir = os.path.dirname(os.path.abspath(project_file)) if project_file else os.path.abspath(".")
    project_base = os.path.splitext(os.path.basename(project_file))[0] if project_file else ""

    if save_project:
        try:
            hfssApp.save_project()
        except Exception as e:
            ok = False
            print(f"[hfss_com.solver_exit] save_project failed: {e}")
            print(traceback.format_exc())

    try:
        hfssApp.release_desktop()
    except Exception as e:
        ok = False
        print(f"[hfss_com.solver_exit] release_desktop failed: {e}")
        print(traceback.format_exc())

    if cleanup_results and project_base:
        try:
            aedtresults = os.path.join(project_dir, f"{project_base}.aedtresults")
            if os.path.exists(aedtresults):
                shutil.rmtree(aedtresults, ignore_errors=True)

            pyaedt_folder = os.path.join(project_dir, f"{project_base.replace(' ', '_')}.pyaedt")
            if os.path.exists(pyaedt_folder):
                shutil.rmtree(pyaedt_folder, ignore_errors=True)
        except Exception as e:
            ok = False
            print(f"[hfss_com.solver_exit] cleanup_results failed: {e}")
            print(traceback.format_exc())

    return ok


__all__ = [
    "set_hfss_temp_directory",
    "solver_init",
    "set_para",
    "analyze",
    "save_modal",
    "save_nearField",
    "save_farField",
    "save_antPara",
    "solver_exit",
]