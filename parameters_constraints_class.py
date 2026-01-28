"""
Datatype and helper methods for optimization parameters.

A parameter ("para") supports mixed domains via `ranges`:

- Each element of `ranges` is either:
  - a 2-tuple (lo, hi) meaning a continuous interval, OR
  - a number meaning a discrete allowed value.

IMPORTANT:
Because `ranges` is allowed to contain overlapping elements (overlapping intervals,
duplicated discrete values, or overlap between discrete values and intervals),
the mapping from a real `value` back to a normalized `normValue` in [0, 1] is
not necessarily unique. Therefore this module intentionally does NOT provide
a `norm()` function.

Always treat `normValue` as the source of truth, and use `denorm()` to obtain
the corresponding real `value`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

RangeElem = Union[float, tuple[float, float]]


def _coerce_range_elem(elem) -> RangeElem:
    # Continuous range: tuple/list of length 2
    if isinstance(elem, (tuple, list)):
        if len(elem) != 2:
            raise ValueError(f"Range tuple must have length 2, got {elem!r}")
        lo = float(elem[0])
        hi = float(elem[1])
        return (lo, hi)

    # Discrete value: number-like
    return float(elem)


@dataclass
class para:
    name: str
    ranges: tuple[RangeElem, ...]
    value: float = float("nan")
    normValue: float = float("nan")
    unit: str = ""

    def __post_init__(self) -> None:
        # Ensure ranges is a tuple
        if not isinstance(self.ranges, tuple):
            self.ranges = tuple(self.ranges)  # type: ignore[assignment]

        if len(self.ranges) == 0:
            raise ValueError(f"Parameter {self.name!r} must have at least one range element")

        # Convert all numeric content to float
        self.ranges = tuple(_coerce_range_elem(e) for e in self.ranges)
        self.value = float(self.value)
        self.normValue = float(self.normValue)
        self.unit = str(self.unit)

    @property
    def range_count(self) -> int:
        return len(self.ranges)

    def denorm(self, normValue: float | None = None, *, clip: bool = True, update: bool = True) -> float:
        """
        Map a normalized value in [0, 1] back to a real value based on `ranges`.

        Convention:
        - Split [0,1] evenly into N bins (N = len(ranges))
        - Pick bin index = floor(norm * N) (with norm=1 mapped to last bin)
        - Within-bin position = fractional part of (norm * N)
        - If chosen bin is:
          - tuple(lo, hi): interpolate lo + (hi-lo)*pos
          - float: return that float

        Args:
            normValue: if None, uses self.normValue
            clip: if True, clamp normValue into [0, 1]
            update: if True, store result into self.value and (clipped) norm into self.normValue

        Returns:
            The denormalized real value (float).
        """
        if normValue is None:
            normValue = self.normValue

        n = self.range_count
        if n <= 0:
            raise ValueError("ranges is empty")

        x = float(normValue)
        if clip:
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0

        scaled = x * n

        # Handle x==1.0: scaled==n, which would overflow without clamping
        idx = int(scaled)
        if idx >= n:
            idx = n - 1
            pos = 1.0
        else:
            pos = scaled - idx  # in [0,1)

        elem = self.ranges[idx]
        if isinstance(elem, tuple):
            lo, hi = elem
            real = lo + (hi - lo) * pos
        else:
            real = float(elem)

        if update:
            self.normValue = x
            self.value = real
        return real
    
