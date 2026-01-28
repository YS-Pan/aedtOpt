# misc.py
from __future__ import annotations

from math import comb


def population_size(expectedSize: int, numberOfGoals: int, p_cap: int = 200) -> tuple[int, int]:
    """
    根据 NSGA-III 的 uniform reference points 规则，修正期望种群规模 expectedSize，
    得到最接近且“可被 NSGA-III 接受”的 POP_SIZE。

    定义：
      M = numberOfGoals (目标数)
      P = divisions
      H = C(M+P-1, P)  (reference points 数量)

    选择策略：
      - 找到最小 P 使得 H >= expectedSize
      - 在 P 和 P-1 对应的 H 中选择离 expectedSize 更近者
      - 若距离相同，优先选择 H >= expectedSize（即不小于期望规模）

    返回：
      (POP_SIZE, P)
    """
    mu = int(expectedSize)
    m = int(numberOfGoals)

    if mu <= 0:
        raise ValueError("expectedSize 必须为正整数")
    if m < 2:
        raise ValueError("numberOfGoals 必须 >= 2（NSGA-III 面向多目标）")
    if p_cap < 1:
        raise ValueError("p_cap 必须 >= 1")

    def H(p: int) -> int:
        return comb(m + p - 1, p)

    p = 1
    h = H(p)
    while h < mu and p < p_cap:
        p += 1
        h = H(p)

    # 触顶仍达不到 mu：只能返回触顶结果（仍是合法的 H）
    if p >= p_cap and h < mu:
        return h, p

    if p == 1:
        return h, p

    p_hi, h_hi = p, h
    p_lo, h_lo = p - 1, H(p - 1)

    d_hi = abs(h_hi - mu)
    d_lo = abs(h_lo - mu)

    if d_lo < d_hi:
        return h_lo, p_lo
    if d_hi < d_lo:
        return h_hi, p_hi

    # tie -> prefer H >= mu
    return h_hi, p_hi




