# empulse1d/fdtd/mur.py
"""
Mur absorbing boundary condition (ABC) for 1D FDTD.
We use the classic 1st-order Mur boundary update for the E field at the two ends
of a 1D domain.

Assumptions:
- Uniform spatial grid with spacing dz
- Uniform time step dt
- Homogeneous wave speed c in the domain (for MVP we use free-space c)

Notation:
- E is stored at integer grid indices i = 0..Nz-1
- H is staggered in a Yee grid but Mur is applied to E endpoints only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MurABC:
    """Container for Mur ABC parameters."""
    c: float  # wave speed [m/s]
    dt: float  # timestep [s]
    dz: float  # spatial step [m]


def mur_coefficient(c: float, dt: float, dz: float) -> float:
    """
    Compute the Mur coefficient k
    k = (c*dt - dz) / (c*dt + dz)
    """
    if c <= 0:
        raise ValueError(f"c must be > 0")
    if dt <= 0:
        raise ValueError(f"dt must be > 0")
    if dz <= 0:
        raise ValueError(f"dz must be > 0")
    return (c * dt - dz) / (c * dt + dz)


def apply_mur_abc(E: np.ndarray, E_prev: np.ndarray, c: float, dt: float, dz: float) -> None:
    """
    Apply 1st-order Mur ABC in-place to the boundary points of E.
    E: E field at time n+1 (after interior update). Shape (Nz,)
    E_prev: E field at time n (before update). Shape (Nz,)
    """
    if E.ndim != 1 or E_prev.ndim != 1:
        raise ValueError("E and E_prev must be 1D arrays.")
    if E.shape != E_prev.shape:
        raise ValueError(f"E and E_prev shapes must match, got {E.shape} vs {E_prev.shape}.")
    if E.size < 3:
        raise ValueError("Mur ABC requires Nz >= 3.")

    k = mur_coefficient(c, dt, dz)

    # Left boundary (i=0)
    # E[0]^{n+1} = E[1]^n + k * (E[1]^{n+1} - E[0]^n)
    E0_new = E_prev[1] + k * (E[1] - E_prev[0])

    # Right boundary (i=Nz-1)
    # E[N-1]^{n+1} = E[N-2]^n + k * (E[N-2]^{n+1} - E[N-1]^n)
    ENm1_new = E_prev[-2] + k * (E[-2] - E_prev[-1])

    E[0] = E0_new
    E[-1] = ENm1_new


def apply_mur_abc_with_obj(E: np.ndarray, E_prev: np.ndarray, mur: MurABC) -> None:
    """Convenience wrapper using a MurABC dataclass."""
    apply_mur_abc(E=E, E_prev=E_prev, c=mur.c, dt=mur.dt, dz=mur.dz)