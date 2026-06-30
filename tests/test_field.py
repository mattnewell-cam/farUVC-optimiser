"""Core physics checks. Run: python -m pytest tests/ -q  (or python tests/test_field.py)

The key correctness anchors:
  * IES parsing + intensity match the raw B1 file,
  * scalar fluence obeys the inverse-square law analytically,
  * exposure limits reproduce guv-calcs' authoritative B1 TLVs.

A heavier cross-check against guv_calcs (if installed) lives in
``test_matches_guv_calcs`` and is skipped when the package is absent.
"""

import numpy as np

from faruvc.field import LampInstance
from faruvc.geometry import Room
from faruvc.photometry import Photometry
from faruvc.regs import Standard, limits_for_spectrum

IES = "data/lamp_data/ushio_b1.ies"
SPECTRUM = "data/lamp_data/ushio_b1.csv"


def test_inverse_square_on_axis():
    phot = Photometry.from_ies(IES)
    lamp = LampInstance(phot, pos=(0, 0, 2.0))   # downlight
    I0 = float(phot.intensity(0, 0))             # mW/sr
    for r in (0.5, 1.0, 1.5):
        p = np.array([[0.0, 0.0, 2.0 - r]])      # directly below at distance r
        expected = I0 / r**2 * 0.1               # mW/m² -> µW/cm²
        assert abs(float(lamp.fluence(p)[0]) - expected) < 1e-6


def test_downlight_eye_is_zero_below():
    phot = Photometry.from_ies(IES)
    lamp = LampInstance(phot, pos=(0, 0, 3.0))
    p = np.array([[0.0, 0.0, 1.8]])
    # Directly under a downlight the eye (horizontal view) sees nothing; skin sees all.
    from faruvc.field import eye_field, planar_field
    assert float(eye_field([lamp], p)[0]) < 1e-9
    assert float(planar_field([lamp], p)[0]) > 1.0


def test_b1_limits_match_authoritative():
    acgih = limits_for_spectrum(SPECTRUM, Standard.RP27_1)
    icnirp = limits_for_spectrum(SPECTRUM, Standard.ICNIRP)
    assert abs(acgih.eye_uw - 5.2311) < 1e-3
    assert abs(acgih.skin_uw - 15.9184) < 1e-3
    assert abs(icnirp.eye_uw - 0.8020) < 1e-3
    assert abs(icnirp.skin_uw - icnirp.eye_uw) < 1e-9   # ICNIRP: one eye/skin curve


def test_matches_guv_calcs():
    """Our scalar fluence must equal guv_calcs.Lamp.irradiance_at (µW/cm²)."""
    try:
        from guv_calcs import Lamp
    except Exception:
        import pytest
        pytest.skip("guv_calcs not installed (dev oracle only)")
    gl = Lamp(lamp_id="b1"); gl.load_ies(IES)
    phot = Photometry.from_ies(IES)
    for theta in (0, 20, 40, 60):
        for r in (1.0, 2.0):
            guv = gl.irradiance_at(theta, 0.0, r)
            ours = float(phot.intensity(theta, 0)) / r**2 * 0.1
            assert abs(guv - ours) < 1e-4 * max(1.0, guv)


if __name__ == "__main__":
    test_inverse_square_on_axis()
    test_downlight_eye_is_zero_below()
    test_b1_limits_match_authoritative()
    test_matches_guv_calcs()
    print("all core physics checks passed")
