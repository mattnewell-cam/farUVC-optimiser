"""Photobiological exposure limits (ACGIH / ICNIRP) for far-UVC.

This mirrors OSLUV `guv-calcs` exactly so our numbers match Illuminate:

  * A standard supplies a spectral *weighting* curve S(λ) (the action spectrum).
  * For a lamp with relative spectral power i(λ), the effective weighting is
        s = Σ S(λ)·i(λ)/Σ i(λ)            (Riemann sum, ``_sum_spectrum``)
    with S(λ) log-interpolated onto the lamp's wavelengths (``_log_interp``).
  * The 8-hour threshold limit value is  TLV = 3 / s   [mJ/cm² per 8 h]
    (the "3" is the standards' effective-dose limit).
  * Sustained-exposure cap:  E_max = TLV / 28800 s · 1000  [µW/cm²].

Capping raw irradiance at ``E_max`` is equivalent to guv-calcs' "weighted dose ≤ 3".

Bundled data (``data/``): ``UV Spectral Weighting Curves.csv`` and the per-lamp
spectrum CSV (e.g. ``lamp_data/ushio_b1.csv``), both taken from guv-calcs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"
_WEIGHTS_CSV = _DATA / "UV Spectral Weighting Curves.csv"

EXPOSURE_SECONDS_8H = 8 * 60 * 60  # 28800
_EFFECTIVE_DOSE_LIMIT = 3.0        # mJ/cm² effective, per the standards


class Standard(Enum):
    """Exposure standards. The TLV *limits* come from the action-spectrum curve; the
    *assessment geometry* (which calc mode, what plane height) is the recommended-
    practice layer and differs between standards — notably UL 8802 evaluates the eye
    with fluence rate and the skin with worst-orientation planar-max."""
    RP27_1 = "rp27_1"   # ANSI/IES RP 27.1-22 (ACGIH limits)
    UL8802 = "ul8802"   # UL 8802 (ACGIH limits, different assessment geometry)
    ICNIRP = "icnirp"   # IEC 62471-6:2022 (ICNIRP limits)

    @property
    def label(self) -> str:
        return {
            Standard.RP27_1: "RP 27.1-22 (ACGIH)",
            Standard.UL8802: "UL 8802 (ACGIH)",
            Standard.ICNIRP: "IEC 62471-6:2022 (ICNIRP)",
        }[self]

    @classmethod
    def from_token(cls, token: str) -> "Standard":
        t = str(token).strip().lower().replace("-", "").replace(".", "").replace(" ", "")
        aliases = {
            "acgih": cls.RP27_1, "rp27": cls.RP27_1, "rp271": cls.RP27_1,
            "rp2712": cls.RP27_1, "rp27122": cls.RP27_1, "ansi": cls.RP27_1,
            "ul8802": cls.UL8802, "ul": cls.UL8802,
            "icnirp": cls.ICNIRP, "iec": cls.ICNIRP, "iec624716": cls.ICNIRP,
        }
        if t in aliases:
            return aliases[t]
        return cls(token)  # try raw enum value

    def weight_column(self, target: str) -> str:
        """CSV column name for this standard's eye/skin action spectrum."""
        if self in (Standard.RP27_1, Standard.UL8802):
            # Both use the ACGIH (RP 27.1-22) action spectra -> identical TLVs.
            return ("ANSI IES RP 27.1-22 (Eye)" if target == "eye"
                    else "ANSI IES RP 27.1-22 (Skin)")
        return "IEC 62471-6:2022 (Eye/Skin)"   # ICNIRP: one eye/skin curve

    @property
    def zone(self) -> "ZoneConfig":
        """Assessment geometry: calc mode for eye & skin and the plane height (m)."""
        return _ZONE_CONFIG[self]


@dataclass(frozen=True)
class ZoneConfig:
    height_m: float
    eye_mode: str    # one of: eye_worst_case, fluence_rate
    skin_mode: str   # one of: planar_normal, planar_max


_ZONE_CONFIG = {
    Standard.RP27_1: ZoneConfig(height_m=1.8, eye_mode="eye_worst_case", skin_mode="planar_normal"),
    Standard.UL8802: ZoneConfig(height_m=1.9, eye_mode="fluence_rate",  skin_mode="planar_max"),
    Standard.ICNIRP: ZoneConfig(height_m=1.8, eye_mode="eye_worst_case", skin_mode="planar_normal"),
}


@dataclass(frozen=True)
class ExposureLimits:
    """Sustained-irradiance caps (µW/cm²) and 8-h TLVs (mJ/cm²) for a lamp+standard."""
    standard: Standard
    eye_tlv_mj: float
    skin_tlv_mj: float

    @property
    def eye_uw(self) -> float:
        return _tlv_to_uw(self.eye_tlv_mj)

    @property
    def skin_uw(self) -> float:
        return _tlv_to_uw(self.skin_tlv_mj)


# --- public API -----------------------------------------------------------
def limits_for_spectrum(spectrum_csv: str | Path, standard: Standard) -> ExposureLimits:
    """Compute eye/skin limits for a lamp spectrum CSV under the given standard."""
    wl, inten = _load_spectrum(Path(spectrum_csv))
    eye = _tlv(wl, inten, standard.weight_column("eye"))
    skin = _tlv(wl, inten, standard.weight_column("skin"))
    return ExposureLimits(standard=standard, eye_tlv_mj=eye, skin_tlv_mj=skin)


# --- core math (mirrors guv-calcs) ----------------------------------------
def _tlv(spec_wl: np.ndarray, spec_i: np.ndarray, weight_column: str) -> float:
    wwl, wval = _weight_curve(weight_column)
    w = _log_interp(spec_wl, wwl, wval)
    i_norm = spec_i / _sum_spectrum(spec_wl, spec_i)
    s = _sum_spectrum(spec_wl, w * i_norm)
    return _EFFECTIVE_DOSE_LIMIT / s


def _tlv_to_uw(tlv_mj: float) -> float:
    return tlv_mj / EXPOSURE_SECONDS_8H * 1000.0


def _sum_spectrum(wl: np.ndarray, i: np.ndarray) -> float:
    wl = np.asarray(wl, float)
    i = np.asarray(i, float)
    return float(np.sum(i[1:] * np.diff(wl)))


def _log_interp(wvs, weight_wvs, weight_values) -> np.ndarray:
    logterp = np.interp(np.asarray(wvs, float),
                        np.asarray(weight_wvs, float),
                        np.log10(np.asarray(weight_values, float)))
    return np.power(10.0, logterp)


# --- data loading ---------------------------------------------------------
_weight_cache: dict | None = None


def _weight_curve(column: str):
    global _weight_cache
    if _weight_cache is None:
        _weight_cache = _load_weight_table(_WEIGHTS_CSV)
    wl = _weight_cache["Wavelength (nm)"]
    return wl, _weight_cache[column]


def _load_weight_table(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        header = [h.strip() for h in next(rd)]
        cols: dict[str, list] = {h: [] for h in header}
        for row in rd:
            if not row or not row[0].strip():
                continue
            for h, v in zip(header, row):
                cols[h].append(float(v))
    return {h: np.array(v) for h, v in cols.items()}


def _load_spectrum(path: Path):
    wl, inten = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd, None)  # header
        for row in rd:
            if len(row) < 2:
                continue
            try:
                wl.append(float(row[0])); inten.append(float(row[1]))
            except ValueError:
                continue
    return np.array(wl), np.array(inten)
