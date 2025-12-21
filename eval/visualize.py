from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    """
    Visualization configuration.
    """
    title: str = ""
    show_colorbar: bool = True
    origin: str = "lower"  # convenient for (x,y) grid interpretation
    interpolation: str = "nearest"
    interior_margin: int = 0


def _apply_margin(arr: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return arr
    nx, ny = arr.shape
    if 2 * margin >= nx or 2 * margin >= ny:
        raise ValueError("Margin too large for array.")
    return arr[margin:nx - margin, margin:ny - margin]


def plot_field(
    field: np.ndarray,
    *,
    cfg: PlotConfig = PlotConfig(),
    mark: Optional[Tuple[int, int]] = None,
    mark2: Optional[Tuple[int, int]] = None,
    mark_labels: Tuple[str, str] = ("Focus", "Source"),
) -> None:
    """
    Plot a 2D field (Ez snapshot, focus_map, etc).

    Parameters
    ----------
    field:
      2D numpy array

    mark:
      Optional (ix, iy) point to mark (e.g., predicted focus)

    mark2:
      Optional second point to mark (e.g., true source)

    interior_margin:
      If cfg.interior_margin > 0, crops the plotted field to ignore ABC artifacts.
      Note: mark coordinates are still in full-grid indices; we shift them accordingly.
    """
    if field.ndim != 2:
        raise ValueError("field must be 2D.")

    margin = cfg.interior_margin
    cropped = _apply_margin(field, margin)

    plt.figure()
    im = plt.imshow(
        cropped.T,  # transpose so x is horizontal axis, y vertical
        origin=cfg.origin,
        interpolation=cfg.interpolation,
        aspect="auto",
    )
    plt.title(cfg.title)

    if cfg.show_colorbar:
        plt.colorbar(im, shrink=0.85)

    def _shift(p: Tuple[int, int]) -> Tuple[float, float]:
        return (p[0] - margin, p[1] - margin)

    # mark points if provided
    if mark is not None:
        x, y = _shift(mark)
        plt.scatter([x], [y], marker="x")
        plt.text(x + 1, y + 1, mark_labels[0])

    if mark2 is not None:
        x, y = _shift(mark2)
        plt.scatter([x], [y], marker="o")
        plt.text(x + 1, y + 1, mark_labels[1])

    plt.xlabel("x (grid index)")
    plt.ylabel("y (grid index)")
    plt.tight_layout()
    plt.show()


def plot_sensor_signals(
    recordings: np.ndarray,
    *,
    dt: float,
    max_sensors: int = 8,
    title: str = "Sensor recordings (Ez vs time)",
) -> None:
    """
    Plot up to max_sensors sensor time series.

    recordings shape: (num_sensors, num_steps)
    """
    if recordings.ndim != 2:
        raise ValueError("recordings must be 2D (num_sensors, num_steps).")

    num_sensors, num_steps = recordings.shape
    t = np.arange(num_steps) * dt

    plt.figure()
    k_show = min(num_sensors, max_sensors)
    for k in range(k_show):
        plt.plot(t, recordings[k, :], label=f"S{k}")

    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("Ez (arb.)")
    if k_show > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()
