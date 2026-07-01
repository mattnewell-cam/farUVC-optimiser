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
    edge_inset: float = 0.1,
    wall_mount_height: float | None = None,
    wall_spacing: float = 1.5,
    tilts: tuple[float, ...] = (25.0, 40.0),
    aim_azimuths: int = 3,
    azimuth_spread_deg: float = 45.0,
) -> list[LampInstance]:
    """Return a list of candidate LampInstances.

    Parameters control the discretisation; finer grids give better optima but a
    larger ILP.
    """
    z_ceil = room.height - ceiling_offset
    candidates: list[LampInstance] = []

    # --- ceiling downlights ------------------------------------------------
    # Downlights are inset from the walls (wall_inset): a ceiling fixture can't hang in
    # the wall, and a downlight against a wall wastes half its cone. Wall/corner MOUNTS,
    # by contrast, physically sit ON the wall/corner -> only a token edge_inset to keep
    # them inside the polygon (and off the 1/r^2 singularity). In corner_edge mode the
    # swept edge/corner aims dominate, so we thin the downlight grid to keep the candidate
    # count (and solve time) down.
    dl_spacing = grid_spacing if mode == "downlight" else max(grid_spacing, 1.0)
    for xy in _interior_grid(room, dl_spacing, wall_inset):
        candidates.append(LampInstance(photometry, pos=(xy[0], xy[1], z_ceil), aim=DOWN,
                                       meta={"mount": "ceiling", "h": z_ceil}))

    if mode == "downlight":
        return candidates

    # --- corner/edge lamps -------------------------------------------------
    h = wall_mount_height if wall_mount_height is not None else z_ceil
    # Wall/corner mounts sweep AIM ANGLE rather than using one fixed aim: at each position
    # we fan the horizontal direction across the inward half-space and try several tilts.
    # (No "aim at the far corner" heuristic -- that's brittle for concave/complex shapes;
    # sweeping is shape-agnostic. Adding aims is cheap: the field maths is vectorised, and
    # avg fluence is separable so the solver handles the extra candidates well.)
    for p0, p1 in room.edges():
        n_in = _inward_normal(room, p0, p1)
        edge = p1 - p0
        length = np.linalg.norm(edge)
        if length < 1e-6:
            continue
        npts = max(1, int(round(length / wall_spacing)))
        for k in range(npts):
            t = (k + 0.5) / npts
            base = p0 + t * edge + n_in * edge_inset
            for off, td in _sweep(aim_azimuths, azimuth_spread_deg, tilts):
                aim = _tilt_down(_rot2d(n_in, off), td)
                meta = {"mount": "edge", "p0": p0.copy(), "p1": p1.copy(),
                        "n_in": n_in.copy(), "inset": edge_inset, "h": h,
                        "t": t, "tilt": td, "az": off}
                candidates.append(LampInstance(photometry, pos=(base[0], base[1], h),
                                               aim=aim, meta=meta))

    # Corner mounts: fan around the inward angle bisector.
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
        base = cur + bis * edge_inset
        for off, td in _sweep(aim_azimuths, azimuth_spread_deg, tilts):
            aim = _tilt_down(_rot2d(bis, off), td)
            meta = {"mount": "corner", "base": base.copy(), "bis": bis.copy(), "h": h,
                    "tilt": td, "az": off}
            candidates.append(LampInstance(photometry, pos=(base[0], base[1], h),
                                           aim=aim, meta=meta))

    return candidates


def _sweep(n_az, spread_deg, tilts):
    """Yield (azimuth_offset_rad, tilt_deg) over the horizontal fan × tilts."""
    offs = [0.0] if n_az <= 1 else np.radians(np.linspace(-spread_deg, spread_deg, n_az))
    for off in offs:
        for td in tilts:
            yield float(off), float(td)


def _rot2d(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


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
