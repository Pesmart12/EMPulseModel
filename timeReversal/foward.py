from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import numpy as np

from fdtd.grid import Grid2D
from fdtd.fields import FieldsTMz
from fdtd.update import Material2D, FDTDSolverTMz
from fdtd.boundaries import MurABC
from fdtd.source import PointSource


@dataclass
class ForwardResult:
    """
    Holds the output of a forward time-reversal experiment:
    - sensor recordings (SensorArray.recordings)
    - optional snapshots / metadata
    """
    recordings: np.ndarray  # shape: (num_sensors, num_steps)
    dt: float
    num_steps: int
    meta: Dict[str, Any]


def run_forward(
    *,
    grid: Grid2D,
    material: Material2D,
    source: PointSource,
    source_signal: Callable[[float], float],
    sensors,  # SensorArray
    num_steps: int,
    use_mur_abc: bool = True,
    record_after_step: bool = True,
    snapshot_stride: Optional[int] = None,
) -> ForwardResult:
    """
    Forward propagation:
      - run FDTD with a point source
      - record Ez at sensor locations

    Parameters
    ----------
    record_after_step:
      If True, record after each full solver.step (Ez at n+1).
      If False, record before step (Ez at n).
      Either is fine as long as you are consistent between forward and reverse.

    snapshot_stride:
      If set (e.g., 20), store Ez snapshots every stride steps in meta["snapshots"].
    """
    fields = FieldsTMz.zeros(grid)
    boundary = MurABC(grid) if use_mur_abc else None
    solver = FDTDSolverTMz(grid=grid, material=material, fields=fields, boundary=boundary)

    sensors.allocate(num_steps)

    snapshots = []
    for n in range(num_steps):
        t = n * grid.dt

        if not record_after_step:
            sensors.record(fields.Ez, n)

        solver.step(source=source, source_value=source_signal(t))

        if record_after_step:
            sensors.record(fields.Ez, n)

        if snapshot_stride is not None and (n % snapshot_stride == 0):
            snapshots.append(fields.Ez.copy())

    meta = {
        "source": {"ix": source.ix, "iy": source.iy, "mode": source.mode},
        "sensor_positions": sensors.positions,
        "use_mur_abc": use_mur_abc,
        "record_after_step": record_after_step,
    }
    if snapshot_stride is not None:
        meta["snapshot_stride"] = snapshot_stride
        meta["snapshots"] = snapshots

    return ForwardResult(
        recordings=sensors.recordings.copy(),
        dt=grid.dt,
        num_steps=num_steps,
        meta=meta,
    )
