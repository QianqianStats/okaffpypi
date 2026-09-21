"""Detect multiple changes in one numeric data stream with OKAFF.

The initial burn-in (default 50 observations) establishes the reference. After
each alarm, the detector resets and uses the same number of observations to
learn the new baseline before monitoring resumes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .single import (
    _validate_stream,
    _validate_thresholding_method,
    _validate_detector_options,
    _add_detector_arguments,
    _detector_options,
    DEFAULT_CLIP,
    BURN_IN,
    DEFAULT_N_RFF,
    DEFAULT_QUANTILE,
    DEFAULT_SEED,
    ETA,
    LAMBDA0,
    LAMBDA1,
    MONITORING_THRESHOLD_RATE,
    REFERENCE_THRESHOLD_RATE,
    OKAFF,
    OKAFFAdaptiveThreshold,
    estimate_gaussian_gamma,
    load_stream,
    make_rff,
    okaff_gaussian_theory_moments,
)

RESULT_COLUMNS = [
    "change_number",
    "detected_change_point",
    "monitoring_run_length",
    "burn_in_start",
    "burn_in_end",
    "statistic",
    "threshold_lower",
    "threshold_upper",
    "alarm_side",
    "q",
    "burn_in",
    "dimension",
    "n_rff",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path, help="CSV containing one numeric observation per row")
    parser.add_argument("--burn-in", type=int, default=BURN_IN, help="Reference observations (minimum 2; default 50)")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--n-rff", type=int, default=DEFAULT_N_RFF)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-csv", type=Path, help="Optional detected-change summary CSV")
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


def initialize_monitor(reference, *, quantile, n_rff, seed,
    lambda0=LAMBDA0, lambda1=LAMBDA1, eta=ETA, clip=DEFAULT_CLIP,
    reference_threshold_rate=REFERENCE_THRESHOLD_RATE,
    monitoring_threshold_rate=MONITORING_THRESHOLD_RATE,
    thresholding_method="adaptive",
):
    _validate_thresholding_method(thresholding_method)
    _validate_detector_options(lambda0, lambda1, eta, clip, reference_threshold_rate, monitoring_threshold_rate)
    dimension = reference.shape[1]
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
        t0=len(reference) + 1,
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
    return detector, threshold


def detect_multiple_changes(stream, *, quantile=DEFAULT_QUANTILE, n_rff=DEFAULT_N_RFF, seed=DEFAULT_SEED, burn_in=BURN_IN,
    lambda0=LAMBDA0, lambda1=LAMBDA1, eta=ETA, clip=DEFAULT_CLIP,
    reference_threshold_rate=REFERENCE_THRESHOLD_RATE,
    monitoring_threshold_rate=MONITORING_THRESHOLD_RATE,
    thresholding_method="adaptive",
):
    """Return alarms, relearning burn_in reference rows initially and after each alarm.

    Detector and threshold options match detect_first_change and are reused
    after every alarm. Theory uses the clipped initial lambda0.

    thresholding_method="fixed" freezes each baseline interval after burn-in;
    "adaptive" (default) continues updating it while monitoring.

    burn_in defaults to 50 and must be an integer of at least 2. The stream
    must exceed burn_in rows; tails without a monitoring observation are skipped.
    """
    _validate_thresholding_method(thresholding_method)
    _validate_detector_options(lambda0, lambda1, eta, clip, reference_threshold_rate, monitoring_threshold_rate)
    stream = _validate_stream(stream, quantile, n_rff, burn_in)
    results = []
    burn_in_start = 0
    segment_number = 0
    n_observations, dimension = stream.shape

    while burn_in_start + burn_in < n_observations:
        segment_number += 1
        burn_in_end = burn_in_start + burn_in
        reference = stream[burn_in_start:burn_in_end]
        detector, threshold = initialize_monitor(
            reference,
            quantile=quantile,
            n_rff=n_rff,
            seed=np.random.SeedSequence([seed, segment_number]),
            lambda0=lambda0, lambda1=lambda1, eta=eta, clip=clip,
            reference_threshold_rate=reference_threshold_rate,
            monitoring_threshold_rate=monitoring_threshold_rate,
            thresholding_method=thresholding_method,
        )

        alarm_found = False
        for index in range(burn_in_end, n_observations):
            detector.update(stream[index])
            statistic = float(detector.statistic)
            if threshold.update(statistic):
                lower, upper = threshold.interval
                results.append(
                    {
                        "change_number": len(results) + 1,
                        "detected_change_point": index + 1,
                        "monitoring_run_length": index - burn_in_end + 1,
                        "burn_in_start": burn_in_start + 1,
                        "burn_in_end": burn_in_end,
                        "statistic": statistic,
                        "threshold_lower": lower,
                        "threshold_upper": upper,
                        "alarm_side": "lower" if statistic < lower else "upper",
                        "q": quantile,
                        "burn_in": burn_in,
                        "dimension": dimension,
                        "n_rff": n_rff,
                    }
                )
                burn_in_start = index + 1
                alarm_found = True
                break

        if not alarm_found:
            break

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def main():
    args = parse_args()
    stream = load_stream(args.data_csv, burn_in=args.burn_in)
    results = detect_multiple_changes(
        stream,
        quantile=args.quantile,
        n_rff=args.n_rff,
        seed=args.seed,
        burn_in=args.burn_in,
        thresholding_method=args.thresholding_method,
        **_detector_options(args),
    )

    if results.empty:
        print(f"No change point was detected after the initial {args.burn_in}-observation burn-in.")
    else:
        print(results.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
