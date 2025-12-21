# fdtd/boundaries.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .grid import Grid2D


@dataclass
class MurABC:
    """
    First-order Mur absorbing boundary condition (ABC) for Ez.

    This is a simple, lightweight ABC that reduces reflections at the edges.
    It is not as strong as PML, but it's a good starting point.

    We store previous Ez values along each boundary to apply the Mur update.

    References:
      - Standard FDTD texts (e.g., Taflove).
    """
    grid: Grid2D

    def __post_init__(self) -> None:
        nx, ny = self.grid.nx, self.grid.ny
        # Previous time-step boundary values (copy of Ez at previous step)
        self._Ez_prev = np.zeros((nx, ny), dtype=np.float64)

        # Precompute Mur coefficients for x and y boundaries
        c = self.grid.c0
        dt = self.grid.dt
        dx = self.grid.dx
        dy = self.grid.dy

        self._coef_x = (c * dt - dx) / (c * dt + dx)
        self._coef_y = (c * dt - dy) / (c * dt + dy)

    def store_previous(self, Ez: np.ndarray) -> None:
        """
        Call once per time step BEFORE Ez is updated, or right after,
        as long as you're consistent. Here we expect:
          - update H
          - update Ez interior
          - apply Mur using Ez_prev (previous full Ez)
          - then store Ez as Ez_prev for next step
        """
        np.copyto(self._Ez_prev, Ez)

    def apply(self, Ez: np.ndarray) -> None:
        """
        Apply Mur ABC to Ez on all four boundaries using self._Ez_prev.
        """
        nx, ny = self.grid.nx, self.grid.ny
        Ez_prev = self._Ez_prev

        cx = self._coef_x
        cy = self._coef_y

        # Left boundary (i=0): uses i=1
        # Ez(0,j)^{n+1} = Ez(1,j)^n + cx*(Ez(1,j)^{n+1} - Ez(0,j)^n)
        Ez[0, 1:-1] = Ez_prev[1, 1:-1] + cx * (Ez[1, 1:-1] - Ez_prev[0, 1:-1])

        # Right boundary (i=nx-1): uses i=nx-2
        Ez[nx - 1, 1:-1] = Ez_prev[nx - 2, 1:-1] + cx * (
            Ez[nx - 2, 1:-1] - Ez_prev[nx - 1, 1:-1]
        )

        # Bottom boundary (j=0): uses j=1
        Ez[1:-1, 0] = Ez_prev[1:-1, 1] + cy * (Ez[1:-1, 1] - Ez_prev[1:-1, 0])

        # Top boundary (j=ny-1): uses j=ny-2
        Ez[1:-1, ny - 1] = Ez_prev[1:-1, ny - 2] + cy * (
            Ez[1:-1, ny - 2] - Ez_prev[1:-1, ny - 1]
        )

        # Corners: simple copy (good enough for Mur 1st order baseline)
        Ez[0, 0] = Ez[1, 1]
        Ez[0, ny - 1] = Ez[1, ny - 2]
        Ez[nx - 1, 0] = Ez[nx - 2, 1]
        Ez[nx - 1, ny - 1] = Ez[nx - 2, ny - 2]
