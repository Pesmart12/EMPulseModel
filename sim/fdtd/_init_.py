"""
empulse1d.fdtd

Core 1D FDTD components for forward EM simulation.

This submodule implements:
- 1D Yee-grid Maxwell updates (E, H)
- Ricker wavelet source injection
- Mur absorbing boundary conditions
- Small utilities for grid / indexing

The forward solver is intentionally simple and inverse-friendly.
"""

from .solver import run_fdtd
from .wavelet import ricker_wavelet
from .source import inject_point_source
from .mur import apply_mur_abc

__all__ = [
    "run_fdtd",
    "ricker_wavelet",
    "inject_point_source",
    "apply_mur_abc",
]