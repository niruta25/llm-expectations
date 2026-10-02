"""Turning what a judge said into a probability — or admitting you have not.

The mistake this package exists to prevent is treating a number a language
model emitted as a probability. A judge reporting "confidence 0.9" is stating a
feeling; ranking on it and calling the result a risk score is exactly the
trust-laundering this library is built to catch (DESIGN.md §7).

So the passthrough is a calibrator too. It has the same shape as a real one,
reports ``is_calibrated = False``, and every number downstream of it carries
that stamp.
"""

from __future__ import annotations

from .base import Calibrator, FitReport, StaleCalibration, fingerprint
from .identity import IdentityCalibrator

__all__ = [
    "Calibrator",
    "FitReport",
    "IdentityCalibrator",
    "StaleCalibration",
    "fingerprint",
]
