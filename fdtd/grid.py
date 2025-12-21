# fdtd/grid.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """
    2D FDTD grid for TMz mode (Ez, Hx, Hy).

    Coordinates:
      - x index i in [0, nx-1]
      - y index j in [0, ny-1]

    Spatial steps:
      - dx, dy in meters
    Time step:
      - dt in seconds, must satisfy CFL for stability.
    """
    nx: int
    ny: int
    dx: float
    dy: float
    dt: float

    # Material properties (defaults: free space)
    eps0: float = 8.854187817e-12
    mu0: float = 4.0e-7 * np.pi

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be >= 3.")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("dx and dy must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

    @property
    def c0(self) -> float:
        return 1.0 / np.sqrt(self.mu0 * self.eps0)

    def cfl_dt_max(self) -> float:
        """
        2D CFL limit for standard Yee scheme:
          dt <= 1 / (c * sqrt( (1/dx^2) + (1/dy^2) ))
        """
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        return 1.0 / (self.c0 * np.sqrt(inv_dx2 + inv_dy2))

    def assert_cfl(self, safety: float = 0.99) -> None:
        """
        Raises if dt violates CFL * safety factor.
        """
        dt_max = self.cfl_dt_max() * safety
        if self.dt > dt_max:
            raise ValueError(
                f"Time step dt={self.dt:.3e} exceeds CFL limit (with safety={safety}): "
                f"{dt_max:.3e}. Reduce dt or increase dx/dy."
            )
