# fdtd/source.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PointSource:
    """
    Simple point source that adds to Ez at (ix, iy).

    mode:
      - "add": Ez[ix, iy] += value(t)
      - "hard": Ez[ix, iy]  = value(t)
    """
    ix: int
    iy: int
    mode: str = "add"

    def apply(self, Ez: np.ndarray, value: float) -> None:
        if self.mode == "add":
            Ez[self.ix, self.iy] += value
        elif self.mode == "hard":
            Ez[self.ix, self.iy] = value
        else:
            raise ValueError(f"Unknown source mode: {self.mode}")


def gaussian_pulse(t: float, t0: float, spread: float, amplitude: float = 1.0) -> float:
    """
    Gaussian pulse centered at t0:
      amplitude * exp(-((t - t0)/spread)^2)
    """
    x = (t - t0) / spread
    return float(amplitude * np.exp(-(x * x)))


def ricker_wavelet(t: float, f0: float, t0: float, amplitude: float = 1.0) -> float:
    """
    Ricker (Mexican hat) wavelet commonly used as a band-limited pulse.
    f0: center frequency (Hz)
    t0: time shift (s)
    """
    tau = t - t0
    a = (np.pi * f0 * tau)
    val = (1.0 - 2.0 * a * a) * np.exp(-(a * a))
    return float(amplitude * val)
