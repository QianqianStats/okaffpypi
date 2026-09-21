"""Detect the first change in one numeric data stream with OKAFF.

The input CSV must contain more rows than the selected burn-in (default 50)
and only numeric feature columns. Monitoring begins after the burn-in and
stops at the first two-sided threshold alarm.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_thresholds import OKAFFAdaptiveThreshold, okaff_gaussian_theory_moments
from .detector import OKAFF
from .features import estimate_gaussian_gamma, make_rff

BURN_IN = 50
DEFAULT_QUANTILE = 0.92
DEFAULT_N_RFF = 500
DEFAULT_SEED = 2026
LAMBDA0 = 0.999
LAMBDA1 = 0.999
ETA = 1e-3
DEFAULT_CLIP = (1e-3, 0.999)
REFERENCE_THRESHOLD_RATE = 0.1
MONITORING_THRESHOLD_RATE = 0.01



def _validate_detector_options(lambda0, lambda1, eta, clip, reference_threshold_rate, monitoring_threshold_rate):
    def finite_number(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")

    for name, value in (("lambda0", lambda0), ("lambda1", lambda1)):
        finite_number(value, name)
        if not 0 <= value < 1:
            raise ValueError(f"{name} must satisfy 0 <= {name} < 1")
    finite_number(eta, "eta")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    try:
        lower, upper = clip
    except (TypeError, ValueError):
        raise ValueError("clip must contain two bounds") from None
    finite_number(lower, "clip lower bound")
    finite_number(upper, "clip upper bound")
    if not 0 <= lower < upper <= 1:
        raise ValueError("clip must satisfy 0 <= lower < upper <= 1")
    for name, value in (("reference_threshold_rate", reference_threshold_rate),
                        ("monitoring_threshold_rate", monitoring_threshold_rate)):
        finite_number(value, name)
        if not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1")


def _validate_thresholding_method(thresholding_method):
    if thresholding_method not in ("adaptive", "fixed"):
        raise ValueError("thresholding_method must be adaptive or fixed")


def _add_detector_arguments(parser):
    parser.add_argument("--thresholding-method", choices=("adaptive", "fixed"), default="adaptive", help="Two-sided adaptive bounds or bounds frozen after burn-in (default adaptive)")
    parser.add_argument("--lambda0", type=float, default=LAMBDA0, help="Initial forgetting factor (default 0.999)")
    parser.add_argument("--lambda1", type=float, default=LAMBDA1, help="Forgetting factor after the first observation (default 0.999)")
    parser.add_argument("--eta", type=float, default=ETA, help="Forgetting-factor learning rate (default 0.001)")
    parser.add_argument("--clip", type=float, nargs=2, default=DEFAULT_CLIP, metavar=("LOWER", "UPPER"), help="Forgetting-factor bounds (default 0.001 0.999)")
    parser.add_argument("--reference-threshold-rate", type=float, default=REFERENCE_THRESHOLD_RATE, help="Threshold update rate during burn-in (default 0.1)")
    parser.add_argument("--monitoring-threshold-rate", type=float, default=MONITORING_THRESHOLD_RATE, help="Threshold update rate during monitoring (default 0.01)")


def _detector_options(args):
    return {name: getattr(args, name) for name in (
        "lambda0", "lambda1", "eta", "clip", "reference_threshold_rate", "monitoring_threshold_rate"
    )}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path, help="CSV containing one numeric observation per row")
    parser.add_argument("--burn-in", type=int, default=BURN_IN, help="Reference observations (minimum 2; default 50)")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--n-rff", type=int, default=DEFAULT_N_RFF)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-csv", type=Path, help="Optional one-row detection summary CSV")
    _add_detector_arguments(parser)
    args = parser.parse_args()
    try:
        _validate_detector_options(**_detector_options(args))
    except ValueError as error:
        parser.error(str(error))
    if not 0.5 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.5 and 1")
    if args.n_rff <= 0:
        parser.error("--n-rff must be a positive integer")
    if args.burn_in < 2:
        parser.error("--burn-in must be an integer of at least 2")
    return args


def _validate_burn_in(burn_in):
    if isinstance(burn_in, (bool, np.bool_)) or not isinstance(burn_in, (int, np.integer)) or burn_in < 2:
        raise ValueError("burn_in must be an integer of at least 2")


def load_stream(path, *, burn_in=BURN_IN):
    _validate_burn_in(burn_in)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.shape[1] == 0:
        raise ValueError("Input CSV has no feature columns")
    try:
        stream = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Every input column must be numeric") from error
    if len(stream) <= burn_in:
        raise ValueError(
            f"Input stream must contain more than {burn_in} observations; got {len(stream)}"
        )
    if not np.isfinite(stream).all():
        raise ValueError("Input CSV contains missing or non-finite values")
    return stream


def _validate_stream(stream, quantile, n_rff, burn_in=BURN_IN):
    _validate_burn_in(burn_in)
    stream = np.asarray(stream, dtype=float)
    if stream.ndim == 1:
        stream = stream[:, None]
    if stream.ndim != 2 or stream.shape[1] == 0 or len(stream) <= burn_in:
        raise ValueError(f"stream must have more than {burn_in} rows and at least one feature")
    if not np.isfinite(stream).all():
        raise ValueError("stream must contain only finite values")
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be between 0.5 and 1")
    if isinstance(n_rff, bool) or int(n_rff) != n_rff or n_rff <= 0:
        raise ValueError("n_rff must be a positive integer")
    return stream


def detect_first_change(stream, *, quantile=DEFAULT_QUANTILE, n_rff=DEFAULT_N_RFF, seed=DEFAULT_SEED, burn_in=BURN_IN,
    lambda0=LAMBDA0, lambda1=LAMBDA1, eta=ETA, clip=DEFAULT_CLIP,
    reference_threshold_rate=REFERENCE_THRESHOLD_RATE,
    monitoring_threshold_rate=MONITORING_THRESHOLD_RATE,
    thresholding_method="adaptive",
):
    """Return the first alarm after burn_in reference rows (default 50, minimum 2).

    lambda0/lambda1 are forgetting factors in [0, 1); eta is nonnegative.
    clip bounds them in [0, 1]. Threshold rates are in (0, 1), for burn-in
    and monitoring respectively. Theory uses the clipped initial lambda0.

    thresholding_method="fixed" freezes the learned two-sided interval after
    burn-in; "adaptive" (default) continues updating it while monitoring.

    Positions are one-based detection times; the stream must exceed burn_in rows.
    """
    _validate_thresholding_method(thresholding_method)
    _validate_detector_options(lambda0, lambda1, eta, clip, reference_threshold_rate, monitoring_threshold_rate)
    stream = _validate_stream(stream, quantile, n_rff, burn_in)
    reference = stream[:burn_in]
    dimension = stream.shape[1]
    gamma = estimate_gaussian_gamma(reference)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False, ddof=1))
    theory_a, theory_sv = okaff_gaussian_theory_moments(
        gamma=gamma,
        covariance=covariance,
        lambda_value=float(np.clip(lambda0, *clip)),
    )

    detector = OKAFF(
        lambda0=lambda0,
        lambda1=lambda1,
        eta=eta,
        clip=clip,
        feat_func=make_rff(
            np.random.default_rng(seed),
            gamma=gamma,
            dimension=dimension,
            n_rff=n_rff,
        ),
        thresholding_method="fixed",
        fixed_threshold=(-np.inf, np.inf),
        store_values=False,
    )
    threshold = OKAFFAdaptiveThreshold(
        alpha=reference_threshold_rate,
        quantile=quantile,
        t0=burn_in + 1,
        initialization="theory_A_SV",
        theory_A=theory_a,
        theory_SV=theory_sv,
    )

    for observation in reference:
        detector.update(observation)
        threshold.update(detector.statistic)

    if thresholding_method == "fixed":
        threshold.freeze()
    else:
        threshold.alpha = monitoring_threshold_rate
    for zero_based_index in range(burn_in, len(stream)):
        detector.update(stream[zero_based_index])
        statistic = float(detector.statistic)
        alarm = threshold.update(statistic)
        if alarm:
            lower, upper = threshold.interval
            return {
                "detected": True,
                "detected_change_point": zero_based_index + 1,
                "monitoring_run_length": zero_based_index + 1 - burn_in,
                "statistic": statistic,
                "threshold_lower": lower,
                "threshold_upper": upper,
                "q": quantile,
                "burn_in": burn_in,
                "n_observations": len(stream),
                "dimension": dimension,
                "n_rff": n_rff,
            }

    lower, upper = threshold.interval
    return {
        "detected": False,
        "detected_change_point": pd.NA,
        "monitoring_run_length": len(stream) - burn_in,
        "statistic": float(detector.statistic),
        "threshold_lower": lower,
        "threshold_upper": upper,
        "q": quantile,
        "burn_in": burn_in,
        "n_observations": len(stream),
        "dimension": dimension,
        "n_rff": n_rff,
    }


def main():
    args = parse_args()
    result = detect_first_change(
        load_stream(args.data_csv, burn_in=args.burn_in),
        quantile=args.quantile,
        n_rff=args.n_rff,
        seed=args.seed,
        burn_in=args.burn_in,
        thresholding_method=args.thresholding_method,
        **_detector_options(args),
    )
    result_frame = pd.DataFrame([result])
    print(result_frame.to_string(index=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
