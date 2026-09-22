"""Gaussian bandwidth estimation and random Fourier features."""
import math
import numpy as np


def estimate_gaussian_gamma(reference_sample, max_len=500):
    """Median bandwidth heuristic, preserving the original seeded sampling.

    Samples up to ``max_len`` rows with replacement using seed 1234.
    Constant/degenerate references cannot define a bandwidth and raise ValueError.
    """
    sample = np.asarray(reference_sample, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    if sample.ndim != 2 or sample.shape[0] < 2 or sample.shape[1] == 0:
        raise ValueError("reference_sample must have at least two rows and one feature")
    if not np.isfinite(sample).all():
        raise ValueError("reference_sample must be finite")
    if isinstance(max_len, bool) or int(max_len) != max_len or max_len < 2:
        raise ValueError("max_len must be an integer >= 2")
    selected = np.random.default_rng(1234).choice(sample, min(len(sample), int(max_len)))
    distances = [np.linalg.norm(selected[i] - selected[j], ord=2)**2
                 for i in range(len(selected)) for j in range(i + 1, len(selected))]
    median = float(np.median(distances))
    if median <= 0 or not np.isfinite(median):
        raise ValueError("reference sample has a zero or invalid median distance")
    return float(1 / (2 * np.sqrt(median * 0.5)**2))


def make_rff(rng, *, gamma, dimension, n_rff):
    frequencies = rng.normal(
        scale=math.sqrt(2.0 * gamma),
        size=(n_rff, dimension),
    )

    def feature(observation):
        projection = np.einsum(
            "d,md->m",
            np.asarray(observation, dtype=float),
            frequencies,
            optimize=False,
        )
        return np.concatenate((np.cos(projection), np.sin(projection))) / math.sqrt(n_rff)

    return feature


