"""faruvc — far-UVC (222 nm) luminaire-placement optimiser.

A lean, vectorized core for:
  * loading lamp photometry from IES files (photometry.py),
  * computing fluence-rate and skin/eye irradiance fields in a 2.5D room (field.py),
  * encoding ACGIH / ICNIRP exposure limits (regs.py),
  * optimising the minimum number of lamps + arrangement (optimize.py).

Physics is validated against OSLUV's open-source `guv-calcs` engine; lamp IES data
comes from that project. See README.md.
"""

__version__ = "0.1.0"
