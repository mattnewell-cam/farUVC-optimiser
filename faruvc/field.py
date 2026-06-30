"""Fluence and exposure fields from placed lamps.

A :class:`LampInstance` is a :class:`~faruvc.photometry.Photometry` distribution put
at a position with an aim direction. Its core output, for a set of field points, is:
  * ``m``   : the scalar magnitude I(v,h)/r^2 of each point's irradiance from the lamp
              (µW/cm²), and
  * ``dhat``: the unit direction from each point toward the lamp.

Every safety/efficacy quantity is a linear combination of per-lamp terms, which is
what lets the optimiser treat lamp selection as an integer-linear program:

  * fluence rate (scalar/omnidirectional) :  E0 = Σ m_i
  * planar irradiance on a surface n̂      :  E_n = Σ m_i · max(0, dhat_i·n̂)   [skin]
  * eye worst case                        :  max over horizontal view dirs of E_n

Units: IES intensity is mW/sr, distances are metres, so I/r² is mW/m². We convert to
µW/cm² (×0.1) throughout, matching how Illuminate/guv-calcs report exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .photometry import Photometry

MW_PER_M2_TO_UW_PER_CM2 = 0.1  # 1 mW/m² = 0.1 µW/cm²
_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class LampInstance:
    photometry: Photometry
    pos: np.ndarray                       # (3,) metres
    aim: np.ndarray = dc_field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    dimming: float = 1.0                  # 0..1 output scale

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float).reshape(3)
        a = np.asarray(self.aim, dtype=float).reshape(3)
        self.aim = a / (np.linalg.norm(a) + 1e-300)
        # Build a stable azimuth reference frame (u, w) spanning the plane ⊥ aim.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, self.aim)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, self.aim) * self.aim
        self._u = u / (np.linalg.norm(u) + 1e-300)
        self._w = np.cross(self.aim, self._u)

    # --- core ------------------------------------------------------------
    def scalar_and_dir(self, points: np.ndarray):
        """Return (m, dhat) for (M,3) field points.

        m    : (M,) scalar irradiance magnitude in µW/cm²
        dhat : (M,3) unit vectors pointing from each field point toward the lamp
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        d = self.pos[None, :] - pts            # point -> lamp
        r = np.linalg.norm(d, axis=1)
        r_safe = np.maximum(r, 1e-6)
        dhat = d / r_safe[:, None]

        ray = -dhat                            # lamp -> point
        cos_v = np.clip(np.einsum("ij,j->i", ray, self.aim), -1.0, 1.0)
        vert = np.degrees(np.arccos(cos_v))
        horiz = np.degrees(np.arctan2(
            np.einsum("ij,j->i", ray, self._w),
            np.einsum("ij,j->i", ray, self._u),
        ))
        intensity = self.photometry.intensity(vert, horiz)        # mW/sr
        m = self.dimming * intensity / (r_safe ** 2) * MW_PER_M2_TO_UW_PER_CM2
        return m, dhat

    # --- linear quantities (per lamp) ------------------------------------
    def fluence(self, points: np.ndarray) -> np.ndarray:
        """Scalar fluence-rate contribution (µW/cm²) at each point."""
        m, _ = self.scalar_and_dir(points)
        return m

    def planar(self, points: np.ndarray, normal=_Z) -> np.ndarray:
        """Planar irradiance (µW/cm²) onto a surface with the given outward normal."""
        m, dhat = self.scalar_and_dir(points)
        n = np.asarray(normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-300)
        cos = np.clip(np.einsum("ij,j->i", dhat, n), 0.0, None)
        return m * cos

    def eye_terms(self, points: np.ndarray, azimuths: np.ndarray) -> np.ndarray:
        """Planar irradiance for a set of horizontal viewing directions.

        Returns (M, A): for each point and each azimuth (eye looking horizontally in
        that compass direction), the clamped planar irradiance. The eye worst case is
        the max over azimuths; imposing the limit per-azimuth keeps the optimiser linear.
        """
        m, dhat = self.scalar_and_dir(points)
        az = np.asarray(azimuths, dtype=float)
        normals = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)  # (A,3)
        cos = np.clip(dhat @ normals.T, 0.0, None)        # (M,A)
        return m[:, None] * cos


# --------------------------------------------------------------------------
# Aggregate fields over many lamps (for reporting/validation, not the optimiser)
# --------------------------------------------------------------------------
def fluence_field(lamps, points) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    total = np.zeros(len(pts))
    for lamp in lamps:
        total += lamp.fluence(pts)
    return total


def planar_field(lamps, points, normal=_Z) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    total = np.zeros(len(pts))
    for lamp in lamps:
        total += lamp.planar(pts, normal=normal)
    return total


def eye_field(lamps, points, n_az: int = 16) -> np.ndarray:
    """Worst-case (over horizontal view azimuths) eye-plane irradiance per point."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    az = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    acc = np.zeros((len(pts), n_az))
    for lamp in lamps:
        acc += lamp.eye_terms(pts, az)
    return acc.max(axis=1)


# --------------------------------------------------------------------------
# Standard-specific exposure calc modes (mirrors guv-calcs PlaneCalcMode).
#
# Each mode reduces to a set of receiver "samples". The exposure at a point is the
# max over samples of the summed per-lamp contribution; every sample is linear in the
# lamps, so the optimiser can constrain each sample <= cap.
#   fluence_rate  : omnidirectional scalar (a single sample = sum of magnitudes)
#   planar_normal : irradiance on the horizontal plane normal (downwelling)
#   planar_max    : worst-orientation planar irradiance (sampled over a hemisphere)
#   eye_worst_case: worst horizontal viewing direction (vertical plane)
# --------------------------------------------------------------------------
def mode_normals(mode: str, n_az: int = 16):
    """Receiver normals for a calc mode, or None for the scalar fluence mode."""
    if mode == "fluence_rate":
        return None
    if mode == "planar_normal":
        return _Z[None, :].copy()
    if mode == "planar_max":
        return _hemisphere_normals()
    if mode == "eye_worst_case":
        az = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
        return np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)
    raise ValueError(f"unknown calc mode {mode!r}")


def lamp_exposure_terms(lamp: "LampInstance", points, mode: str, n_az: int = 16):
    """(M, K) per-point, per-sample linear contributions of one lamp for a mode."""
    m, dhat = lamp.scalar_and_dir(points)
    normals = mode_normals(mode, n_az)
    if normals is None:                         # fluence_rate: omnidirectional
        return m[:, None]
    cos = np.clip(dhat @ normals.T, 0.0, None)  # (M, K)
    return m[:, None] * cos


def exposure_field(lamps, points, mode: str, n_az: int = 16) -> np.ndarray:
    """Realised exposure (µW/cm²) per point under a given standard calc mode."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    terms = None
    for lamp in lamps:
        t = lamp_exposure_terms(lamp, pts, mode, n_az)
        terms = t if terms is None else terms + t
    if terms is None:
        return np.zeros(len(pts))
    return terms.max(axis=1)


def _hemisphere_normals(n_az: int = 6) -> np.ndarray:
    """Unit normals sampling the upper hemisphere (for worst-orientation planar-max).

    Zenith plus rings at 45/90° polar angle. All room lamps sit above the occupant
    plane, so the worst receiving orientation lies in this hemisphere. Kept coarse
    (13 normals) for optimiser speed; the cap margin absorbs the discretisation.
    """
    normals = [np.array([0.0, 0.0, 1.0])]
    for polar in (45.0, 90.0):
        p = np.radians(polar)
        for k in range(n_az):
            a = 2 * np.pi * k / n_az
            normals.append(np.array([np.sin(p) * np.cos(a),
                                     np.sin(p) * np.sin(a),
                                     np.cos(p)]))
    return np.array(normals)


def column_fluence_grid(room, lamps, spacing: float = 0.15, z_spacing: float = 0.3):
    """Top-down fluence coverage map for visualisation.

    Returns (values, extent) where ``values`` is an (ny, nx) array of the column-
    averaged fluence rate (µW/cm²) at each (x, y) inside the polygon (NaN outside),
    and ``extent`` is (xmin, xmax, ymin, ymax). Column-averaging over the room height
    gives an intuitive "how well covered is this spot" picture.
    """
    (xmn, ymn), (xmx, ymx) = room.bbox
    xs = np.arange(xmn, xmx + 1e-9, spacing)
    ys = np.arange(ymn, ymx + 1e-9, spacing)
    zs = np.arange(0.1, room.height - 0.05, z_spacing)
    if len(zs) == 0:
        zs = np.array([room.height / 2])
    gx, gy = np.meshgrid(xs, ys, indexing="xy")        # (ny, nx)
    flat_xy = np.column_stack([gx.ravel(), gy.ravel()])
    inside = room.contains(flat_xy)

    vals = np.full(len(flat_xy), np.nan)
    if inside.any():
        in_xy = flat_xy[inside]
        # Build (Npts*Nz, 3) and average fluence over z for each xy.
        pts = np.repeat(in_xy, len(zs), axis=0)
        zcol = np.tile(zs, len(in_xy))
        pts3 = np.column_stack([pts, zcol])
        f = fluence_field(lamps, pts3).reshape(len(in_xy), len(zs)).mean(axis=1)
        vals[inside] = f
    return vals.reshape(gy.shape), (float(xmn), float(xmx), float(ymn), float(ymx))
