from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from fdtd.grid import Grid2D
from fdtd.fields import FieldsTMz
from fdtd.update import Material2D, FDTDSolverTMz
from fdtd.boundaries import MurABC
from fdtd.source import PointSource


@dataclass
class ReverseResult:
    """
    Output of time-reversal run.
    - focus_map: max over time of |Ez| (or Ez^2) at each grid point
    - focus_idx: (ix, iy) where focus_map is maximum
    - focus_value: max value at focus_idx
    - optional snapshots
    """
    focus_map: np.ndarray
    focus_idx: tuple[int, int]
    focus_value: float
    meta: Dict[str, Any]


def _inject_time_reversal_sources(
    Ez: np.ndarray,
    sensor_positions: list[tuple[int, int]],
    values: np.ndarray,
    mode: str = "add",
) -> None:
    """
    Inject sensor signals into Ez at each sensor location.
    mode "add" is typical for TRM-style re-emission.
    """
    if values.shape[0] != len(sensor_positions):
        raise ValueError("values length must match number of sensors.")
    for k, (ix, iy) in enumerate(sensor_positions):
        if mode == "add":
            Ez[ix, iy] += values[k]
        elif mode == "hard":
            Ez[ix, iy] = values[k]
        else:
            raise ValueError(f"Unknown injection mode: {mode}")


def run_time_reversal(
    *,
    grid: Grid2D,
    material: Material2D,
    recordings: np.ndarray,  # shape: (num_sensors, num_steps)
    sensor_positions: list[tuple[int, int]],
    use_mur_abc: bool = True,
    injection_mode: str = "add",
    focus_metric: str = "abs",   # "abs" or "energy"
    snapshot_stride: Optional[int] = None,
) -> ReverseResult:
    """
    Time-reversal propagation:
      - reverse each sensor recording in time
      - re-inject as sources while running FDTD
      - compute a focusing map

    focus_metric:
      - "abs": accumulate max |Ez|
      - "energy": accumulate max (Ez^2)

    Notes
    -----
    This is the clean baseline version. Later, you can:
      - scale injections by impedance / calibration factors
      - inject H components too
      - use PML instead of Mur
    """
    if recordings.ndim != 2:
        raise ValueError("recordings must be a 2D array (num_sensors, num_steps).")
    num_sensors, num_steps = recordings.shape

    if len(sensor_positions) != num_sensors:
        raise ValueError("sensor_positions length must match recordings first dimension.")

    # Time reverse signals
    tr_signals = recordings[:, ::-1].copy()

    fields = FieldsTMz.zeros(grid)
    boundary = MurABC(grid) if use_mur_abc else None
    solver = FDTDSolverTMz(grid=grid, material=material, fields=fields, boundary=boundary)

    focus_map = np.zeros((grid.nx, grid.ny), dtype=np.float64)
    snapshots = []

    for n in range(num_steps):
        # Advance the solver by one step WITHOUT a single point source.
        solver.step(source=None, source_value=None)

        # Inject the time-reversed sensor signals at this time index
        _inject_time_reversal_sources(
            Ez=fields.Ez,
            sensor_positions=sensor_positions,
            values=tr_signals[:, n],
            mode=injection_mode,
        )

        # Optionally apply ABC again (helps when injection touches boundaries)
        if solver.boundary is not None:
            solver.boundary.apply(fields.Ez)

        # Update focusing metric
        if focus_metric == "abs":
            np.maximum(focus_map, np.abs(fields.Ez), out=focus_map)
        elif focus_metric == "energy":
            np.maximum(focus_map, fields.Ez * fields.Ez, out=focus_map)
        else:
            raise ValueError("focus_metric must be 'abs' or 'energy'.")

        if snapshot_stride is not None and (n % snapshot_stride == 0):
            snapshots.append(fields.Ez.copy())

    # Find focus location
    flat_idx = int(np.argmax(focus_map))
    ix, iy = np.unravel_index(flat_idx, focus_map.shape)
    focus_value = float(focus_map[ix, iy])

    meta: Dict[str, Any] = {
        "num_steps": num_steps,
        "num_sensors": num_sensors,
        "sensor_positions": sensor_positions,
        "use_mur_abc": use_mur_abc,
        "injection_mode": injection_mode,
        "focus_metric": focus_metric,
    }
    if snapshot_stride is not None:
        meta["snapshot_stride"] = snapshot_stride
        meta["snapshots"] = snapshots

    return ReverseResult(
        focus_map=focus_map,
        focus_idx=(int(ix), int(iy)),
        focus_value=focus_value,
        meta=meta,
    )
