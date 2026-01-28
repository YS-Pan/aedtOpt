"""
Parameter and constraint definitions for an optimization program.

- PARAMETERS: tuple of `para` objects
- CONSTRAINTS: tuple of strings; each string is an expression that must be > 0

Notes:
- Constraints are intentionally kept as strings.
- Constraint evaluation should provide needed names such as `math`, `abs`, etc.
"""

from __future__ import annotations

from parameters_constraints_class import para

PARAMETERS = (
    para('choke1H', ((3, 15),), value=8, normValue=float("nan"), unit='mm'),
    para('choke1R1', ((6, 11),), value=7, normValue=float("nan"), unit='mm'),
    para('choke1R2', ((6, 11),), value=10, normValue=float("nan"), unit='mm'),
    para('choke2Gap', ((-4, 10),), value=5, normValue=float("nan"), unit='mm'),
    para('choke2H', ((3, 15),), value=8, normValue=float("nan"), unit='mm'),
    para('choke2R1', ((6, 11),), value=8, normValue=float("nan"), unit='mm'),
    para('choke2R2', ((6, 11),), value=10, normValue=float("nan"), unit='mm'),
    para('choke3Gap', ((-4, 10),), value=5, normValue=float("nan"), unit='mm'),
    para('choke3H', ((3, 15),), value=8, normValue=float("nan"), unit='mm'),
    para('choke3R1', ((6, 11),), value=11, normValue=float("nan"), unit='mm'),
    para('choke3R2', ((6, 11),), value=7, normValue=float("nan"), unit='mm'),
    para('choke4Gap', ((-4, 10),), value=6, normValue=float("nan"), unit='mm'),
    para('choke4H', ((3, 15),), value=8, normValue=float("nan"), unit='mm'),
    para('choke4R1', ((6, 11),), value=11, normValue=float("nan"), unit='mm'),
    para('choke4R2', ((6, 11),), value=8, normValue=float("nan"), unit='mm'),
    para('inputD1', ((-25, -5),), value=-12, normValue=float("nan"), unit='mm'),
    para('inputD2', ((-25, -10),), value=-13, normValue=float("nan"), unit='mm'),
    para('inputSizeY', ((6.0999999999999996, 20),), value=14, normValue=float("nan"), unit='mm'),
    para('inputsizeZ', ((6.0999999999999996, 20),), value=14, normValue=float("nan"), unit='mm'),
    para('ReflectL', ((6, 20),), value=10.775254222308501, normValue=float("nan"), unit='mm'),
    para('ReflectY', ((5, 12),), value=7.80289423209096, normValue=float("nan"), unit='mm'),
    para('ReflectZ', ((5, 12),), value=7.4094599678444402, normValue=float("nan"), unit='mm'),
)

CONSTRAINTS = (
    "-(choke1H + choke2H + choke3H + choke4H - 30)*0.1",
)

