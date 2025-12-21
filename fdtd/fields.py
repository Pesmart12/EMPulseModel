# fdtd/fields.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .grid import Grid2D


@dataclass
class FieldsTMz:
    """
    TMz fields on a 2D Yee grid:
      - Ez on (i, j)
      - Hx on (i, j) (conceptually staggered in y)
      - Hy on (i, j) (conceptually staggered in x)

    We store all as (nx, ny) arrays for simplicity and use
    interior differences consistent with Yee updates.
    """
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray

    @classmethod
    def zeros(cls, grid: Grid2D, dtype=np.float64) -> "FieldsTMz":
        shape = (grid.nx, grid.ny)
        return cls(
            Ez=np.zeros(shape, dtype=dtype),
            Hx=np.zeros(shape, dtype=dtype),
            Hy=np.zeros(shape, dtype=dtype),
        )

    def reset(self) -> None:
        self.Ez.fill(0.0)
        self.Hx.fill(0.0)
        self.Hy.fill(0.0)
