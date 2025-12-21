from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class FocusMetrics:
    """
    Summary metrics for time-reversal focusing.

    focus_idx:
      Argmax location in focus_map, in grid indices (ix, iy)

    focus_value:
      Value of focus_map at focus_idx (depends on how focus_map was constructed)

    distance_cells:
      Euclidean distance in grid cells between estimated focus and true source

    distance_m:
      Euclidean distance in meters between estimated focus and true source

    focus_ratio:
      (peak value) / (median value) over the interior region (robust "contrast")

    fwhm_cells:
      Approximate full-width at half maximum, computed from the area above half max
      and converted to an equivalent radius (in grid cells). Useful single-number
      "sharpness" indicator for the focus spot.
    """
    focus_idx: Tuple[int, int]
    focus_value: float
    distance_cells: float
    distance_m: float
    focus_ratio: float
    fwhm_cells: Optional[float]


def _interior_slice(nx: int, ny: int, margin: int) -> Tuple[slice, slice]:
    if margin < 0:
        raise ValueError("margin must be >= 0.")
    if 2 * margin >= nx or 2 * margin >= ny:
        raise ValueError("margin too large for grid.")
    return (slice(margin, nx - margin), slice(margin, ny - margin))


def compute_focus_metrics(
    *,
    focus_map: np.ndarray,
    true_source_idx: Tuple[int, int],
    dx: float,
    dy: float,
    interior_margin: int = 4,
    compute_fwhm: bool = True,
) -> FocusMetrics:
    """
    Compute robust focusing metrics from a focus_map.

    Parameters
    ----------
    focus_map:
      2D array (nx, ny) — typically max_t(|Ez|) or max_t(Ez^2).

    true_source_idx:
      (ix, iy) index location of the true source.

    dx, dy:
      Grid spacings in meters.

    interior_margin:
      Ignore boundaries when computing background statistics (to reduce ABC artifacts).

    compute_fwhm:
      If True, estimate an equivalent FWHM radius (in cells) based on area above half max.
    """
    if focus_map.ndim != 2:
        raise ValueError("focus_map must be 2D.")

    nx, ny = focus_map.shape
    # Estimated focus
    flat_idx = int(np.argmax(focus_map))
    fx, fy = np.unravel_index(flat_idx, focus_map.shape)
    focus_value = float(focus_map[fx, fy])

    # Distances
    tx, ty = true_source_idx
    dist_cells = float(np.sqrt((fx - tx) ** 2 + (fy - ty) ** 2))
    dist_m = float(np.sqrt(((fx - tx) * dx) ** 2 + ((fy - ty) * dy) ** 2))

    # Contrast: peak / median background (interior)
    slx, sly = _interior_slice(nx, ny, interior_margin)
    interior = focus_map[slx, sly]
    med = float(np.median(interior))
    focus_ratio = float(focus_value / (med + 1e-12))

    # FWHM estimate
    fwhm_cells = None
    if compute_fwhm and focus_value > 0:
        half = 0.5 * focus_value
        mask = interior >= half
        area_cells = int(np.count_nonzero(mask))

        # Equivalent radius r where area = pi r^2
        if area_cells > 0:
            r = np.sqrt(area_cells / np.pi)
            # FWHM "diameter" ~ 2r; we report diameter in cells (common in imaging)
            fwhm_cells = float(2.0 * r)

    return FocusMetrics(
        focus_idx=(int(fx), int(fy)),
        focus_value=focus_value,
        distance_cells=dist_cells,
        distance_m=dist_m,
        focus_ratio=focus_ratio,
        fwhm_cells=fwhm_cells,
    )


def compare_focus_to_truth(
    *,
    focus_idx: Tuple[int, int],
    true_source_idx: Tuple[int, int],
    dx: float,
    dy: float,
) -> Dict[str, float]:
    """
    Lightweight helper if you only want error numbers.

    Returns
    -------
    dict with:
      - distance_cells
      - distance_m
    """
    fx, fy = focus_idx
    tx, ty = true_source_idx
    dist_cells = float(np.sqrt((fx - tx) ** 2 + (fy - ty) ** 2))
    dist_m = float(np.sqrt(((fx - tx) * dx) ** 2 + ((fy - ty) * dy) ** 2))
    return {"distance_cells": dist_cells, "distance_m": dist_m}
