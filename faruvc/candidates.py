"""Candidate lamp positions/orientations for the optimiser to choose from.

Two placement modes (per the brief):
  * ``downlight``    : lamps on the ceiling, aimed straight down.
  * ``corner_edge``  : downlights PLUS wall-mounted and corner-mounted lamps near the
                       ceiling, tilted down into the room.

Each candidate is a :class:`~faruvc.field.LampInstance` (all sharing one Photometry).
"""

from __future__ import annotations

import numpy as np

from .field import LampInstance
from .geometry import Room
from .photometry import Photometry

DOWN = np.array([0.0, 0.0, -1.0])


def generate_candidates(
    room: Room,
    photometry: Photometry,
    mode: str = "downlight",
    *,
    ceiling_offset: float = 0.05,
    grid_spacing: float = 0.6,
    wall_inset: float = 0.3,
    wall_mount_height: float | None = None,
    wall_spacing: float = 1.0,
    wall_tilt_deg: float = 30.0,
    corner_tilt_deg: float = 30.0,
) -> list[LampInstance]:
    """Return a list of candidate LampInstances.

    Parameters control the discretisation; finer grids give better optima but a
    larger ILP.
    """
    z_ceil = room.height - ceiling_offset
    candidates: list[LampInstance] = []

    # --- ceiling downlights ------------------------------------------------
    for xy in _interior_grid(room, grid_spacing, wall_inset):
        candidates.append(LampInstance(photometry, pos=(xy[0], xy[1], z_ceil), aim=DOWN))

    if mode == "downlight":
        return candidates

    # --- corner/edge lamps -------------------------------------------------
    h = wall_mount_height if wall_mount_height is not None else z_ceil
    # Wall mounts: spaced along each edge, aimed along the inward normal, tilted down.
    for p0, p1 in room.edges():
        n_in = _inward_normal(room, p0, p1)
        edge = p1 - p0
        length = np.linalg.norm(edge)
        if length < 1e-6:
            continue
        npts = max(1, int(round(length / wall_spacing)))
        for k in range(npts):
            t = (k + 0.5) / npts
            base = p0 + t * edge + n_in * wall_inset
            aim = _tilt_down(n_in, wall_tilt_deg)
            candidates.append(LampInstance(photometry, pos=(base[0], base[1], h), aim=aim))

    # Corner mounts: aimed along the inward angle bisector, tilted down.
    v = room.vertices
    n = len(v)
    for i in range(n):
        prev, cur, nxt = v[(i - 1) % n], v[i], v[(i + 1) % n]
        e1 = _unit(prev - cur)
        e2 = _unit(nxt - cur)
        bis = _unit(e1 + e2)  # points into the room for a convex corner
        # ensure it points inward
        if not room.contains((cur + bis * 0.1)[None, :])[0]:
            bis = -bis
        base = cur + bis * (wall_inset * 1.5)
        aim = _tilt_down(bis, corner_tilt_deg)
        candidates.append(LampInstance(photometry, pos=(base[0], base[1], h), aim=aim))

    return candidates


# --- helpers --------------------------------------------------------------
def _interior_grid(room: Room, spacing: float, inset: float) -> np.ndarray:
    (xmn, ymn), (xmx, ymx) = room.bbox
    xs = np.arange(xmn + inset, xmx - inset + 1e-9, spacing)
    ys = np.arange(ymn + inset, ymx - inset + 1e-9, spacing)
    if len(xs) == 0:
        xs = np.array([(xmn + xmx) / 2])
    if len(ys) == 0:
        ys = np.array([(ymn + ymx) / 2])
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    return flat[room.contains(flat)]


def _inward_normal(room: Room, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    edge = _unit(p1 - p0)
    normal = np.array([-edge[1], edge[0]])      # rotate +90°
    mid = (p0 + p1) / 2
    if not room.contains((mid + normal * 0.1)[None, :])[0]:
        normal = -normal
    return normal


def _tilt_down(horiz_dir2d: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Build a 3D aim: horizontal direction tilted downward by tilt_deg."""
    h = _unit(horiz_dir2d)
    t = np.radians(tilt_deg)
    return np.array([h[0] * np.cos(t), h[1] * np.cos(t), -np.sin(t)])


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v
