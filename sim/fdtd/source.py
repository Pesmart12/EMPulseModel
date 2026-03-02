# empulse1d/fdtd/source.py
"""
We model a "point dipole" in 1D as a localized impressed source injected into
the E-field update at a single grid index i0 (corresponding to position z0).

This is inverse-friendly: the source location is parameterized by i0 (or z0),
and the forward model returns boundary E(t) traces.

Design goals:
- Keep injection explicit (no hidden globals)
- Support Ricker wavelet time function
- Allow unknown amplitude A to be fit in the inverse step (so forward can run with A=1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .wavelet import ricker_wavelet, RickerParams


@dataclass(frozen=True)
class PointSource:
    #Point source definition 
    
    i0: int # grid index of the source (0 <= i0 <= Nz-1). In practice keep away from boundaries.
    kind: Literal["soft_e"] = "soft_e" #i njection mode (currently only "soft_e" supported, meaning additive injection into E[i0])
    amp: float = 1.0 # amplitude scale (often 1.0 for forward runs; fit later in inverse)
    ricker: Optional[RickerParams] = None # Ricker wavelet parameters (f0, t0) defining the source time function. Must be provided for Ricker injection.

    def validate(self, Nz: int) -> None:
        if not (0 <= self.i0 < Nz):
            raise ValueError(f"Source i0 must be in [0, {Nz-1}], got {self.i0}")
        if self.amp == 0:
            # allowed, but usually accidental
            raise ValueError("Source amp is 0; did you mean to disable the source?")
        if self.ricker is None:
            raise ValueError("PointSource.ricker must be provided (RickerParams).")
        if self.ricker.f0 <= 0:
            raise ValueError(f"Ricker f0 must be > 0, got {self.ricker.f0}")


def z_to_index(z0: float, dz: float, Nz: int) -> int:
    # Convert a physical position z0 [m] to the nearest grid index.
    if dz <= 0:
        raise ValueError(f"dz must be > 0, got {dz}")
    if Nz < 1:
        raise ValueError(f"Nz must be >= 1, got {Nz}")
    i0 = int(np.rint(z0 / dz))
    return int(np.clip(i0, 0, Nz - 1))


def inject_point_source(E: np.ndarray, *, i0: int, value: float, kind: Literal["soft_e"] = "soft_e",) -> None:
    #Inject a point source into the E-field array in-place.

    if E.ndim != 1:
        raise ValueError("E must be a 1D array.")
    if not (0 <= i0 < E.size):
        raise ValueError(f"i0 must be within E array bounds [0, {E.size-1}], got {i0}")

    if kind == "soft_e":
        E[i0] += float(value)
    else:
        raise ValueError(f"Unknown source injection kind: {kind}")


def source_time_function_ricker(t: float, params: RickerParams) -> float:
    # Evaluate the Ricker wavelet source time function at scalar time t
    return float(ricker_wavelet(t=t, f0=params.f0, t0=params.t0, amp=params.amp))


def inject_ricker_point_source(E: np.ndarray, *, src: PointSource, t: float,) -> None:
    #Convenience: evaluate Ricker wavelet at time t and inject at src.i0.
    if src.ricker is None:
        raise ValueError("src.ricker must not be None for Ricker injection.")
    val = src.amp * source_time_function_ricker(t=t, params=src.ricker)
    inject_point_source(E, i0=src.i0, value=val, kind=src.kind)