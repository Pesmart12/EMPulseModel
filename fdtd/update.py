# fdtd/update.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .grid import Grid2D
from .fields import FieldsTMz
from .boundaries import MurABC
from .source import PointSource


@dataclass
class Material2D:
    """
    Simple homogeneous / inhomogeneous material maps.

    For TMz, we use:
      - eps (permittivity) on Ez nodes
      - mu  (permeability) on H nodes
      - sigma (conductivity) on Ez nodes (optional loss)

    Arrays shape: (nx, ny)
    """
    eps: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def homogeneous(cls, grid: Grid2D, eps_r: float = 1.0, mu_r: float = 1.0, sigma: float = 0.0) -> "Material2D":
        eps = np.full((grid.nx, grid.ny), grid.eps0 * eps_r, dtype=np.float64)
        mu = np.full((grid.nx, grid.ny), grid.mu0 * mu_r, dtype=np.float64)
        sig = np.full((grid.nx, grid.ny), sigma, dtype=np.float64)
        return cls(eps=eps, mu=mu, sigma=sig)


@dataclass
class FDTDSolverTMz:
    """
    2D TMz FDTD solver.

    Update equations (Yee):
      Hx^{n+1/2} = Hx^{n-1/2} - (dt/mu) * (dEz/dy)
      Hy^{n+1/2} = Hy^{n-1/2} + (dt/mu) * (dEz/dx)

      Ez^{n+1} = Ez^{n} + (dt/eps) * (dHy/dx - dHx/dy)  (lossless)

    With conductivity sigma, we use a common lossy update form:
      Ez^{n+1} = Ca * Ez^{n} + Cb * (curl_H)
    where:
      Ca = (1 - sigma*dt/(2*eps)) / (1 + sigma*dt/(2*eps))
      Cb = (dt/eps) / (1 + sigma*dt/(2*eps))
    """
    grid: Grid2D
    material: Material2D
    fields: FieldsTMz
    boundary: MurABC | None = None

    def __post_init__(self) -> None:
        self.grid.assert_cfl(safety=0.99)

        # Precompute coefficients
        self._dt = self.grid.dt
        self._dx = self.grid.dx
        self._dy = self.grid.dy

        eps = self.material.eps
        mu = self.material.mu
        sigma = self.material.sigma

        # H update coefficient (assume mu map same grid shape)
        self._ch = self._dt / mu

        # Ez lossy coefficients
        # Avoid division by zero if eps is weird
        denom = 1.0 + (sigma * self._dt) / (2.0 * eps)
        self._Ca = (1.0 - (sigma * self._dt) / (2.0 * eps)) / denom
        self._Cb = (self._dt / eps) / denom

    def step_h(self) -> None:
        Ez = self.fields.Ez
        Hx = self.fields.Hx
        Hy = self.fields.Hy

        ch = self._ch
        dy = self._dy
        dx = self._dx

        # Hx update uses dEz/dy
        # Hx[i, j] <- Hx[i, j] - (dt/mu)* (Ez[i, j+1] - Ez[i, j]) / dy
        Hx[:, 0:-1] -= ch[:, 0:-1] * (Ez[:, 1:] - Ez[:, 0:-1]) / dy

        # Hy update uses dEz/dx
        # Hy[i, j] <- Hy[i, j] + (dt/mu)* (Ez[i+1, j] - Ez[i, j]) / dx
        Hy[0:-1, :] += ch[0:-1, :] * (Ez[1:, :] - Ez[0:-1, :]) / dx

        # (Last row/col of H remain as-is; boundaries handled by Ez ABC)

    def step_e(self) -> None:
        Ez = self.fields.Ez
        Hx = self.fields.Hx
        Hy = self.fields.Hy

        Ca = self._Ca
        Cb = self._Cb
        dy = self._dy
        dx = self._dx

        # curl_H = (dHy/dx - dHx/dy)
        # dHy/dx at Ez interior: (Hy[i,j] - Hy[i-1,j]) / dx
        # dHx/dy at Ez interior: (Hx[i,j] - Hx[i,j-1]) / dy
        curl_h = np.zeros_like(Ez)

        curl_h[1:, :] += (Hy[1:, :] - Hy[0:-1, :]) / dx
        curl_h[:, 1:] -= (Hx[:, 1:] - Hx[:, 0:-1]) / dy

        # Update Ez everywhere (interior correct; boundaries later overwritten by ABC)
        Ez[:] = Ca * Ez + Cb * curl_h

    def apply_source(self, source: PointSource | None, t: float, value: float | None = None) -> None:
        """
        Apply a source to Ez.
        If value is provided, uses it directly; otherwise caller supplies computed signal.
        """
        if source is None:
            return
        if value is None:
            raise ValueError("apply_source requires 'value' if a source is provided.")
        source.apply(self.fields.Ez, value)

    def step(self, *, source: PointSource | None = None, source_value: float | None = None) -> None:
        """
        Advance one full time step:
          1) Update H from Ez
          2) Store Ez_prev (for Mur)
          3) Update Ez from H
          4) Apply source
          5) Apply ABC (Mur) to Ez
          6) Store Ez_prev for next step (optional pattern)

        Note: We store previous Ez BEFORE updating Ez, because Mur needs Ez^n.
        """
        # 1) Update H
        self.step_h()

        # 2) Save Ez^n for ABC
        if self.boundary is not None:
            self.boundary.store_previous(self.fields.Ez)

        # 3) Update Ez (interior)
        self.step_e()

        # 4) Apply source (acts on Ez^n+1)
        if source is not None:
            if source_value is None:
                raise ValueError("source_value must be provided when source is not None.")
            source.apply(self.fields.Ez, source_value)

        # 5) Apply ABC to Ez boundary
        if self.boundary is not None:
            self.boundary.apply(self.fields.Ez)
