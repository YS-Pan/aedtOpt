from __future__ import annotations

import ast
import copy
import shutil
from pathlib import Path
from typing import Iterable


def _format_float(x: float) -> str:
    """稳定输出 float，便于写回 py 文件。"""
    # 17 位有效数字足够 round-trip 大多数 float
    return format(float(x), ".17g")


def _build_parameters_block(updated_parameters) -> str:
    """
    生成新的 PARAMETERS = (...) 源码块（字符串）。
    写入 value 和 normValue，ranges/unit 保持原样。
    """
    lines: list[str] = []
    lines.append("PARAMETERS = (\n")
    for p in updated_parameters:
        # p.ranges 里可能包含 float 或 (lo, hi) 元组
        ranges_src = repr(p.ranges)
        line = (
            f'    para({p.name!r}, {ranges_src}, '
            f"value={_format_float(p.value)}, "
            f"normValue={_format_float(p.normValue)}, "
            f"unit={p.unit!r}),\n"
        )
        lines.append(line)
    lines.append(")\n")
    return "".join(lines)


def _replace_parameters_assignment(source_text: str, replacement_block: str) -> str:
    """
    用 AST 找到顶层的 PARAMETERS = ... 赋值语句，并用 replacement_block 替换。
    """
    tree = ast.parse(source_text)

    target_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PARAMETERS":
                    target_node = node
                    break
        if target_node is not None:
            break

    if target_node is None:
        raise ValueError("在文件中未找到顶层的 `PARAMETERS = ...` 赋值语句，无法替换。")

    if not hasattr(target_node, "lineno") or not hasattr(target_node, "end_lineno"):
        raise RuntimeError("当前 Python 版本的 AST 缺少 lineno/end_lineno，无法精确替换多行赋值。")

    lines = source_text.splitlines(keepends=True)
    start = target_node.lineno - 1          # 0-based
    end_exclusive = target_node.end_lineno  # end_lineno 是 1-based 且为“包含”，所以这里用作 exclusive 刚好

    new_text = "".join(lines[:start]) + replacement_block + "".join(lines[end_exclusive:])
    return new_text


def clone_and_modify_parameters_file(
    normValueList: list[float],
    cloneFilePath: str | Path,
    *,
    sourceFilePath: str | Path = "parameters_constraints.py",
) -> Path:
    """
    复制并修改 parameters_constraints.py：

    1) from parameters_constraints import PARAMETERS
    2) 深拷贝 PARAMETERS
    3) 用 normValueList 覆盖副本的 normValue，并 denorm 更新 value
    4) 复制源文件到 cloneFilePath
    5) 替换克隆文件中的 PARAMETERS 代码块为新的副本内容
    """
    clone_path = Path(cloneFilePath)
    source_path = Path(sourceFilePath)

    if source_path.resolve() == clone_path.resolve():
        raise ValueError("cloneFilePath 不能与 sourceFilePath 相同。")

    # 1) 导入 PARAMETERS（按你的伪代码：直接从模块导入）
    #    注意：要求调用此函数时，parameters_constraints.py 在当前工作目录或 sys.path 中可导入
    from parameters_constraints import PARAMETERS  # noqa: F401
    from parameters_constraints import PARAMETERS as SRC_PARAMETERS

    if len(normValueList) != len(SRC_PARAMETERS):
        raise ValueError(
            f"normValueList 长度({len(normValueList)}) != PARAMETERS 个数({len(SRC_PARAMETERS)})"
        )

    # 2) 深拷贝 PARAMETERS
    updated = tuple(copy.deepcopy(p) for p in SRC_PARAMETERS)

    # 3) 覆盖 normValue 并 denorm 更新 value
    for p, nv in zip(updated, normValueList):
        p.normValue = float(nv)
        p.denorm(p.normValue, clip=True, update=True)

    # 4) 复制源文件到 cloneFilePath
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, clone_path)

    # 5) 替换克隆文件中的 PARAMETERS
    original_text = clone_path.read_text(encoding="utf-8")
    replacement_block = _build_parameters_block(updated)
    new_text = _replace_parameters_assignment(original_text, replacement_block)
    clone_path.write_text(new_text, encoding="utf-8")

    return clone_path


def _load_module_from_file(py_file: Path, module_name: str = "_cloned_parameters_constraints"):
    """用于测试：从指定 .py 文件路径动态加载模块。"""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法为 {py_file} 创建 import spec")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    # ====== 测试输入 ======
    norm_values = [0.10, 0.35, 0.80, 0.60]

    # 建议克隆到同目录，避免 import parameters_constraints_class 找不到
    cloned_path = Path("parameters_constraints_CLONE.py")

    out = clone_and_modify_parameters_file(
        normValueList=norm_values,
        cloneFilePath=cloned_path,
        sourceFilePath="parameters_constraints.py",
    )
    print(f"已生成克隆文件: {out.resolve()}")

    # ====== 验证：动态加载克隆文件并打印参数 ======
    m = _load_module_from_file(out, module_name="_pc_clone")
    print("克隆文件中的 PARAMETERS：")
    for p in m.PARAMETERS:
        print(f"  {p.name:>8s}  normValue={p.normValue:.6g}  value={p.value:.6g}  unit={p.unit!r}  ranges={p.ranges}")