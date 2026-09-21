"""OKAFF online changepoint detection."""
from .detector import OKAFF
from .adaptive_thresholds import OKAFFAdaptiveThreshold, okaff_gaussian_theory_moments
from .features import estimate_gaussian_gamma, make_rff
from .fixed_thresholds import (
    gaussian_kernel_theory_terms, lambda_band_coefficients, theoretical_rejection_band,
)
from .single import detect_first_change
from .multiple import detect_multiple_changes

__version__ = "0.1.0"
__all__ = [
    "OKAFF", "OKAFFAdaptiveThreshold", "okaff_gaussian_theory_moments",
    "estimate_gaussian_gamma", "make_rff", "gaussian_kernel_theory_terms",
    "lambda_band_coefficients", "theoretical_rejection_band",
    "detect_first_change", "detect_multiple_changes",
]
