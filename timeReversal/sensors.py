
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class Sensor:
    """
    A single point sensor that records Ez at a grid index (ix, iy).
    """
    ix: int
    iy: int
    name: str = ""


@dataclass
class SensorArray:
    """
    A collection of point sensors and their recorded time series.

    recordings shape: (num_sensors, num_steps)
    """
    sensors: List[Sensor]
    recordings: Optional[np.ndarray] = None

    def allocate(self, num_steps: int, dtype=np.float64) -> None:
        self.recordings = np.zeros((len(self.sensors), num_steps), dtype=dtype)

    def record(self, Ez: np.ndarray, step_idx: int) -> None:
        if self.recordings is None:
            raise RuntimeError("SensorArray.recordings is None. Call allocate(num_steps) first.")
        for k, s in enumerate(self.sensors):
            self.recordings[k, step_idx] = Ez[s.ix, s.iy]

    def time_reverse(self) -> np.ndarray:
        if self.recordings is None:
            raise RuntimeError("No recordings to time-reverse.")
        return self.recordings[:, ::-1].copy()

    @property
    def positions(self) -> List[Tuple[int, int]]:
        return [(s.ix, s.iy) for s in self.sensors]


def sensors_on_ring(
    nx: int,
    ny: int,
    center: Tuple[int, int],
    radius: int,
    num_sensors: int,
    name_prefix: str = "S",
) -> SensorArray:
    """
    Place sensors approximately on a ring in index space.
    This is convenient for time-reversal focusing demos.

    Parameters
    ----------
    center: (cx, cy) in grid indices
    radius: ring radius in grid cells
    num_sensors: number of sensors to place

    Notes
    -----
    - Points are rounded to nearest integer indices.
    - We clip to stay inside [1..nx-2], [1..ny-2] to avoid boundary cells.
    - If rounding causes duplicates, we keep them (simple baseline behavior).
    """
    cx, cy = center
    sensors: List[Sensor] = []
    for k in range(num_sensors):
        theta = 2.0 * np.pi * k / num_sensors
        ix = int(round(cx + radius * np.cos(theta)))
        iy = int(round(cy + radius * np.sin(theta)))

        ix = int(np.clip(ix, 1, nx - 2))
        iy = int(np.clip(iy, 1, ny - 2))

        sensors.append(Sensor(ix=ix, iy=iy, name=f"{name_prefix}{k:02d}"))
    return SensorArray(sensors=sensors)


def sensors_on_line(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    num_sensors: int,
    nx: int,
    ny: int,
    name_prefix: str = "S",
) -> SensorArray:
    """
    Place sensors along a line segment in index space.

    Parameters
    ----------
    (x0, y0) -> (x1, y1): endpoints in grid indices
    num_sensors: number of sensors
    """
    xs = np.linspace(x0, x1, num_sensors)
    ys = np.linspace(y0, y1, num_sensors)
    sensors: List[Sensor] = []
    for k, (x, y) in enumerate(zip(xs, ys)):
        ix = int(np.clip(int(round(x)), 1, nx - 2))
        iy = int(np.clip(int(round(y)), 1, ny - 2))
        sensors.append(Sensor(ix=ix, iy=iy, name=f"{name_prefix}{k:02d}"))
    return SensorArray(sensors=sensors)
