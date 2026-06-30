"""Room geometry: a 2.5D extruded polygon and the sampling grids used for evaluation.

A :class:`Room` is a horizontal floor polygon (metres) extruded to a ceiling height.
We build:
  * a volume grid filling the room  -> average fluence (germicidal target),
  * a horizontal plane grid at occupant height -> eye/skin exposure checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Room:
    vertices: np.ndarray   # (N, 2) polygon in metres, CCW or CW
    height: float          # ceiling height, metres

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must be (N, 2)")
        if len(self.vertices) < 3:
            raise ValueError("a room needs at least 3 vertices")

    # --- basic measures ---------------------------------------------------
    @property
    def bbox(self):
        mn = self.vertices.min(axis=0)
        mx = self.vertices.max(axis=0)
        return mn, mx

    @property
    def area(self) -> float:
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def volume(self) -> float:
        return self.area * self.height

    # --- point-in-polygon (vectorized ray casting) ------------------------
    def contains(self, pts_xy: np.ndarray) -> np.ndarray:
        """Boolean mask of which (M,2) points lie inside the polygon."""
        p = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
        x, y = p[:, 0], p[:, 1]
        vx, vy = self.vertices[:, 0], self.vertices[:, 1]
        inside = np.zeros(len(p), dtype=bool)
        n = len(self.vertices)
        j = n - 1
        for i in range(n):
            xi, yi, xj, yj = vx[i], vy[i], vx[j], vy[j]
            cond = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
            )
            inside ^= cond
            j = i
        return inside

    # --- sampling grids ---------------------------------------------------
    def volume_grid(self, spacing: float = 0.3, z_margin: float = 0.05) -> np.ndarray:
        """(M,3) points on a regular lattice inside the room volume.

        Spacing in metres; a small z margin keeps samples off the exact floor/ceiling.
        """
        (xmn, ymn), (xmx, ymx) = self.bbox
        xs = _axis(xmn, xmx, spacing)
        ys = _axis(ymn, ymx, spacing)
        zs = _axis(z_margin, self.height - z_margin, spacing)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        flat = np.column_stack([gx.ravel(), gy.ravel()])
        mask = self.contains(flat)
        flat = flat[mask]
        pts = np.repeat(flat, len(zs), axis=0)
        zcol = np.tile(zs, len(flat))
        return np.column_stack([pts, zcol])

    def plane_grid(self, z: float, spacing: float = 0.25) -> np.ndarray:
        """(M,3) points inside the polygon at a fixed height z."""
        (xmn, ymn), (xmx, ymx) = self.bbox
        xs = _axis(xmn, xmx, spacing)
        ys = _axis(ymn, ymx, spacing)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        flat = np.column_stack([gx.ravel(), gy.ravel()])
        flat = flat[self.contains(flat)]
        return np.column_stack([flat, np.full(len(flat), float(z))])

    # --- features for candidate lamp placement ----------------------------
    def edges(self):
        """Yield (p0, p1) wall segment endpoints (2D)."""
        v = self.vertices
        for i in range(len(v)):
            yield v[i], v[(i + 1) % len(v)]

    def corners(self) -> np.ndarray:
        return self.vertices.copy()

    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)


def _axis(lo: float, hi: float, spacing: float) -> np.ndarray:
    if hi <= lo:
        return np.array([(lo + hi) / 2.0])
    n = max(2, int(np.ceil((hi - lo) / spacing)) + 1)
    return np.linspace(lo, hi, n)
