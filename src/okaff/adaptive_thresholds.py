from __future__ import annotations

from math import sqrt
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


def okaff_gaussian_theory_moments(
    gamma: float,
    covariance: np.ndarray,
    lambda_value: float,
) -> Tuple[float, float]:
    """Return the theoretical OKAFF center ``A`` and scale ``SV``.

    The Gaussian kernel is ``exp(-gamma * ||x-y||^2)``
    """
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value < 1.0:
        raise ValueError("lambda_value must satisfy 0 <= lambda_value < 1")

    covariance = np.asarray(covariance, dtype=float)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("covariance must be a finite square matrix")

    sigma_squared = 1.0 / (2.0 * float(gamma))
    identity = np.eye(covariance.shape[0])

    def inverse_sqrt_determinant(matrix: np.ndarray) -> float:
        sign, log_determinant = np.linalg.slogdet(matrix)
        if sign <= 0:
            raise ValueError("theory matrix must be positive definite")
        return float(np.exp(-0.5 * log_determinant))

    term1 = inverse_sqrt_determinant(
        identity + covariance / sigma_squared
    )
    theta_k = inverse_sqrt_determinant(
        identity + 2.0 * covariance / sigma_squared
    )
    term3 = inverse_sqrt_determinant(
        identity + 3.0 * covariance / sigma_squared
    )
    term4 = inverse_sqrt_determinant(
        identity + 4.0 * covariance / sigma_squared
    )
    term13 = term1 * term3
    theta_squared = theta_k**2
    eta_k = max(0.0, term13 - theta_squared)
    zeta_k = max(0.0, term4 - 2.0 * term13 + theta_squared)

    lam = float(lambda_value)
    delta_lambda = (1.0 - lam) / (1.0 + lam)
    rho_lambda = (1.0 - lam) ** 2 / (1.0 + lam + lam**2)
    kappa_lambda = (
        (1.0 - lam) ** 3
        / (1.0 + lam + lam**2 + lam**3)
    )

    theory_a = (
        delta_lambda + (1.0 - delta_lambda) * theta_k
    )
    theory_sv_squared = (
        4.0
        * (delta_lambda - 2.0 * rho_lambda + kappa_lambda)
        * eta_k
        + 2.0
        * (delta_lambda**2 - kappa_lambda)
        * zeta_k
    )
    theory_sv = sqrt(max(0.0, theory_sv_squared))
    return float(theory_a), float(theory_sv)


class OKAFFAdaptiveThreshold:
    """Two-sided exponentially weighted threshold for OKAFF.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        quantile: float = 0.95,
        t0: int = 50,
        initialization: str = "zero",
        theory_A: Optional[float] = None,
        theory_SV: Optional[float] = None,
    ) -> None:
        _validate_rate(alpha, "alpha")
        _validate_quantile(quantile)
        _validate_t0(t0)
        if initialization not in {"zero", "theory_A_SV"}:
            raise ValueError(
                "initialization must be 'zero' or 'theory_A_SV'"
            )
        if initialization == "theory_A_SV":
            if theory_A is None or theory_SV is None:
                raise ValueError(
                    "theory_A and theory_SV are required when "
                    "initialization='theory_A_SV'"
                )
            theory_A = _validate_statistic(theory_A)
            theory_SV = _validate_statistic(theory_SV)
            if theory_SV < 0.0:
                raise ValueError("theory_SV must be nonnegative")

        self.alpha = float(alpha)
        self.quantile = float(quantile)
        self.multiplier = float(norm.ppf(quantile))
        self.t0 = int(t0)
        self.initialization = initialization
        self.theory_A = None if theory_A is None else float(theory_A)
        self.theory_SV = None if theory_SV is None else float(theory_SV)
        self.reset()

    def update(self, statistic: float) -> bool:
        """Update the threshold and return whether this time step alarms."""
        value = _validate_statistic(statistic)
        self.time += 1

        if self.frozen:
            self.alarm = bool(self.time >= self.t0 and (value < self.lower or value > self.upper))
            return self.alarm

        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * value
        self.second_moment = (
            (1.0 - self.alpha) * self.second_moment
            + self.alpha * value**2
        )
        self.variance = max(0.0, self.second_moment - self.mean**2)
        deviation = self.multiplier * sqrt(self.variance)
        self.lower = self.mean - deviation
        self.upper = self.mean + deviation

        self.alarm = bool(
            self.time >= self.t0
            and (value < self.lower or value > self.upper)
        )
        return self.alarm

    def freeze(self) -> None:
        """Keep the current two-sided interval fixed on subsequent updates.

        Call after the last burn-in observation. reset() resumes learning.
        """
        self.frozen = True

    @property
    def interval(self) -> Tuple[float, float]:
        """Current acceptance interval ``(lower, upper)``."""
        return self.lower, self.upper

    @property
    def threshold(self) -> Tuple[float, float]:
        """Alias for :attr:`interval`."""
        return self.interval

    def reset(self) -> None:
        """Restore the initial moment and alarm states."""
        self.frozen = False
        self.time = 0
        if self.initialization == "zero":
            self.mean = 0.0
            self.second_moment = 0.0
            self.variance = 0.0
            self.lower = 0.0
            self.upper = 0.0
        else:
            self.mean = self.theory_A
            self.second_moment = self.theory_A**2 + self.theory_SV**2
            self.variance = self.theory_SV**2
            deviation = self.multiplier * self.theory_SV
            self.lower = self.theory_A - deviation
            self.upper = self.theory_A + deviation
        self.alarm = False


def _validate_rate(rate: float, name: str) -> None:
    if not np.isfinite(rate) or not 0.0 < rate < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")


def _validate_quantile(quantile: float) -> None:
    if not np.isfinite(quantile) or not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0.5 and 1")


def _validate_t0(t0: int) -> None:
    if isinstance(t0, bool) or int(t0) != t0 or t0 < 1:
        raise ValueError("t0 must be a positive integer")


def _validate_statistic(
    statistic: float,
    require_nonnegative: bool = True,
) -> float:
    value = float(statistic)
    if not np.isfinite(value):
        raise ValueError("statistic must be finite")
    if require_nonnegative and value < 0.0:
        raise ValueError("statistic must be nonnegative")
    return value


