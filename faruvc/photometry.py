"""Lamp photometry: parse IESNA LM-63 files and provide a vectorized intensity model.

For far-UVC luminaires the IES "candela" values are radiant intensity in mW/sr
(the OSLUV B1 file declares ``[_INTENSITYUNITS] mW/sr``). We treat the photometric
distribution as radiant intensity I(vertical_angle, horizontal_angle) and provide a
periodic-in-azimuth, clamped-in-elevation interpolator.

Angle conventions (IES Type C goniometry, as used by the OSLUV B1 file):
  * vertical angle  v in [0, 90]: 0 = straight along the lamp's aim axis (nadir for a
    downlight), increasing toward the lamp's horizontal plane.
  * horizontal angle h in [0, 360): azimuth around the aim axis.
Directions with v greater than the measured maximum get zero intensity (a downlight
emits nothing above its own equator).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Photometry:
    """Radiant-intensity distribution of a luminaire.

    Attributes
    ----------
    vert_angles : (Nv,) array of vertical angles in degrees, ascending.
    horiz_angles : (Nh,) array of horizontal angles in degrees, ascending.
    candela : (Nv, Nh) array of radiant intensity in mW/sr (already scaled by the
        IES candela multiplier).
    meta : parsed IES keywords/header values.
    """

    vert_angles: np.ndarray
    horiz_angles: np.ndarray
    candela: np.ndarray  # shape (Nv, Nh), mW/sr
    meta: dict

    # --- construction -----------------------------------------------------
    @classmethod
    def from_ies(cls, path: str | Path) -> "Photometry":
        return parse_ies(Path(path))

    # --- intensity lookup -------------------------------------------------
    def __post_init__(self) -> None:
        v = np.asarray(self.vert_angles, dtype=float)
        h = np.asarray(self.horiz_angles, dtype=float)
        c = np.asarray(self.candela, dtype=float)

        # Make azimuth explicitly periodic so interpolation wraps cleanly. If the
        # file already spans a full turn (0..360) the endpoints coincide; otherwise
        # (e.g. quadrant/half symmetry) mirror/extend to a full 0..360 turn.
        h_full, c_full = _expand_azimuth(h, c)

        self._interp = RegularGridInterpolator(
            (v, h_full), c_full, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        self._vmax = float(v.max())

    def intensity(self, vert_deg, horiz_deg):
        """Radiant intensity (mW/sr) at the given lamp-local angles (vectorized).

        ``vert_deg`` and ``horiz_deg`` may be scalars or broadcastable arrays.
        """
        vert = np.asarray(vert_deg, dtype=float)
        horiz = np.mod(np.asarray(horiz_deg, dtype=float), 360.0)
        out = self._interp(np.stack([np.broadcast_arrays(vert, horiz)[0].ravel(),
                                     np.broadcast_arrays(vert, horiz)[1].ravel()], axis=-1))
        out = out.reshape(np.broadcast_shapes(vert.shape, horiz.shape))
        # Hard zero above the measured elevation (no spurious extrapolated emission).
        out = np.where(vert > self._vmax + 1e-9, 0.0, out)
        return out

    # --- derived quantities ----------------------------------------------
    def total_power_mw(self) -> float:
        """Integrate intensity over the measured solid angle -> radiant power (mW).

        Uses P = integral I(v,h) sin(v) dv dh over the sampled grid (trapezoidal).
        Type C: dOmega = sin(v) dv dh with v measured from the aim axis.
        """
        v = np.deg2rad(self.vert_angles)
        h = np.deg2rad(self.horiz_angles)
        integrand = self.candela * np.sin(v)[:, None]  # (Nv, Nh)
        # integrate over h then v
        over_h = np.trapezoid(integrand, h, axis=1)
        return float(np.trapezoid(over_h, v))

    @property
    def peak_candela(self) -> float:
        return float(self.candela.max())


# --------------------------------------------------------------------------
# IES LM-63 parsing
# --------------------------------------------------------------------------
def parse_ies(path: Path) -> Photometry:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Collect [KEYWORD] metadata up to the TILT line.
    meta: dict = {}
    tilt_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.upper().startswith("TILT="):
            meta["TILT"] = s.split("=", 1)[1].strip()
            tilt_idx = i
            break
        if s.startswith("[") and "]" in s:
            key, _, val = s[1:].partition("]")
            meta[key.strip()] = val.strip()
    if tilt_idx is None:
        raise ValueError(f"{path}: no TILT= line found; not a valid IES file")
    if meta.get("TILT", "NONE").upper() != "NONE":
        raise NotImplementedError(f"{path}: TILT={meta['TILT']} not supported")

    # Everything after the TILT line is whitespace-separated numbers.
    nums = _tokenize_floats(lines[tilt_idx + 1:])
    it = iter(nums)

    def take(n):
        return [next(it) for _ in range(n)]

    (n_lamps, lumens_per_lamp, cand_mult,
     n_vert, n_horiz, phot_type, units_type,
     width, length, height) = take(10)
    n_vert, n_horiz = int(round(n_vert)), int(round(n_horiz))
    ballast, _future, input_watts = take(3)

    vert = np.array(take(n_vert), dtype=float)
    horiz = np.array(take(n_horiz), dtype=float)
    raw = np.array(take(n_vert * n_horiz), dtype=float)

    # LM-63 ordering: for each horizontal angle, all vertical angles.
    candela = raw.reshape(n_horiz, n_vert).T * float(cand_mult)  # -> (Nv, Nh)

    meta.update(
        n_lamps=int(round(n_lamps)),
        lumens_per_lamp=lumens_per_lamp,
        candela_multiplier=cand_mult,
        photometric_type=int(round(phot_type)),
        units_type=int(round(units_type)),
        luminous_dims=(width, length, height),
        ballast_factor=ballast,
        input_watts=input_watts,
        source=str(path),
    )
    return Photometry(vert_angles=vert, horiz_angles=horiz, candela=candela, meta=meta)


def _tokenize_floats(lines) -> list:
    out = []
    for line in lines:
        for tok in line.replace(",", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                # Trailing comments or stray tokens are ignored.
                continue
    return out


def _expand_azimuth(h: np.ndarray, c: np.ndarray):
    """Return azimuth angles + candela covering a full 0..360 turn for interpolation.

    Handles the common IES symmetry encodings:
      * full sphere already present (last angle == 360): use as-is.
      * single plane (Nh == 1, axially symmetric): replicate at 0 and 360.
      * quadrant/half symmetry (last angle 90 or 180): mirror out to 360.
    """
    h = np.asarray(h, dtype=float)
    c = np.asarray(c, dtype=float)
    if h.size == 1:
        return np.array([0.0, 360.0]), np.concatenate([c, c], axis=1)
    last = h[-1]
    if np.isclose(last, 360.0):
        return h, c
    if np.isclose(last, 180.0):
        # mirror (180,360) from (180,0)
        mirror_h = 360.0 - h[-2::-1]
        mirror_c = c[:, -2::-1]
        return np.concatenate([h, mirror_h]), np.concatenate([c, mirror_c], axis=1)
    if np.isclose(last, 90.0):
        h2 = np.concatenate([h, 180.0 - h[-2::-1]])
        c2 = np.concatenate([c, c[:, -2::-1]], axis=1)
        h3 = np.concatenate([h2, 360.0 - h2[-2::-1]])
        c3 = np.concatenate([c2, c2[:, -2::-1]], axis=1)
        return h3, c3
    # Fallback: assume already a usable span; close the loop to 360.
    return np.concatenate([h, [360.0]]), np.concatenate([c, c[:, :1]], axis=1)
