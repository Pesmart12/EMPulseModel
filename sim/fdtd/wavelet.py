# empulse1d/fdtd/wavelet.py
"""
Wavelet utilities for EMPulse-1D (Ricker wavelet generation and time axis creation)
Conventions:
- t is in seconds
- f0 is in Hz
- t0 is in seconds (time shift to center the wavelet)
- amp is unitless scale factor (overall amplitude)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class RickerParams:
    #Parameters for a Ricker wavelet
    f0: float  # center frequency [Hz]
    t0: float  # time shift [s]
    amp: float = 1.0  # amplitude scale


def ricker_wavelet(t: Union[float, np.ndarray], f0: float, t0: float, amp: float = 1.0) -> Union[float, np.ndarray]:
    """
    Returns Wavelet value(s) with same shape as t
    Choose t0 so the wavelet starts near ~0 for t>=0 in your simulation window
    """
    if f0 <= 0:
        raise ValueError(f"f0 must be > 0, got {f0}")

    """ compute the Ricker wavelet using the formula:
      y(t) = (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-(pi * f0 * (t - t0))^2)
    """

    # Convert t to numpy array for vectorized computation, but keep track of original shape
    tt = np.asarray(t, dtype=float)
    # Compute the Ricker wavelet values
    a = np.pi * f0 * (tt - t0)
    # Ricker wavelet formula
    y = (1.0 - 2.0 * a**2) * np.exp(-a**2)
    # Scale by amplitude
    y = amp * y

    # Return python float if input was scalar-like
    if np.isscalar(t):
        return float(np.asarray(y))
    return y


def ricker_from_params(t: Union[float, np.ndarray], params: RickerParams) -> Union[float, np.ndarray]:
    """Convenience wrapper to evaluate a Ricker wavelet from a RickerParams dataclass."""
    return ricker_wavelet(t=t, f0=params.f0, t0=params.t0, amp=params.amp)


def make_time_axis(dt: float, nt: int, t_start: float = 0.0) -> np.ndarray:
    """
    Create a uniformly-spaced time axis.
    Args:
        dt: timestep [s], must be > 0
        nt: number of timesteps, must be >= 1
        t_start: starting time [s]
    Returns t (numpy array of shape (nt,))
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if nt < 1:
        raise ValueError(f"nt must be >= 1, got {nt}")
    return t_start + dt * np.arange(nt, dtype=float)