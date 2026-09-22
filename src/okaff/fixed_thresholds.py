"""Fixed rejection thresholds for the Gaussian-kernel OKAFF experiment."""

import numpy as np


def _logdet_spd(matrix, *, name="matrix"):
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError(
            f"{name} must be positive definite; slogdet sign={sign}."
        )
    return float(logdet)


def gaussian_kernel_theory_terms(Sigma_d, sigmasq):
    """Return theta_K, eta_K, and zeta_K for a Gaussian kernel."""
    Sigma_d = np.asarray(Sigma_d, dtype=float)
    if Sigma_d.ndim != 2 or Sigma_d.shape[0] != Sigma_d.shape[1]:
        raise ValueError("Sigma_d must be a square matrix.")
    if sigmasq <= 0:
        raise ValueError("sigmasq must be positive.")

    identity = np.eye(Sigma_d.shape[0])
    logdet_1 = _logdet_spd(
        identity + Sigma_d / sigmasq, name="I + Sigma/sigma^2"
    )
    logdet_2 = _logdet_spd(
        identity + 2.0 * Sigma_d / sigmasq,
        name="I + 2 Sigma/sigma^2",
    )
    logdet_3 = _logdet_spd(
        identity + 3.0 * Sigma_d / sigmasq,
        name="I + 3 Sigma/sigma^2",
    )
    logdet_4 = _logdet_spd(
        identity + 4.0 * Sigma_d / sigmasq,
        name="I + 4 Sigma/sigma^2",
    )

    theta_K = float(np.exp(-0.5 * logdet_2))
    term13 = float(np.exp(-0.5 * (logdet_1 + logdet_3)))
    term2 = float(np.exp(-logdet_2))
    term4 = float(np.exp(-0.5 * logdet_4))
    eta_K = term13 - term2
    zeta_K = term4 - 2.0 * term13 + term2

    if eta_K < 0 and abs(eta_K) < 1e-14:
        eta_K = 0.0
    if zeta_K < 0 and abs(zeta_K) < 1e-14:
        zeta_K = 0.0
    if eta_K < 0:
        raise ValueError(f"Theoretical eta_K is negative: {eta_K}.")
    if zeta_K < 0:
        raise ValueError(f"Theoretical zeta_K is negative: {zeta_K}.")

    return float(theta_K), float(eta_K), float(zeta_K)


def lambda_band_coefficients(lambda_value):
    """Return delta, rho, and kappa for the selected lambda."""
    if not 0.0 <= lambda_value < 1.0:
        raise ValueError("lambda_value must satisfy 0 <= lambda_value < 1.")

    lam = float(lambda_value)
    delta_lambda = (1.0 - lam) / (1.0 + lam)
    rho_lambda = (1.0 - lam) ** 2 / (1.0 + lam + lam**2)
    kappa_lambda = (1.0 - lam) ** 3 / (
        1.0 + lam + lam**2 + lam**3
    )
    return float(delta_lambda), float(rho_lambda), float(kappa_lambda)


def theoretical_rejection_band(
    *, L, lambda_value, theta_K, eta_K, zeta_K
):
    """Calculate the lower and upper fixed rejection thresholds."""
    if L <= 0:
        raise ValueError("L must be positive.")

    delta_lambda, rho_lambda, kappa_lambda = lambda_band_coefficients(
        float(lambda_value)
    )
    center = delta_lambda + (1.0 - delta_lambda) * float(theta_K)
    sv2 = (
        4.0
        * (delta_lambda - 2.0 * rho_lambda + kappa_lambda)
        * float(eta_K)
        + 2.0
        * (delta_lambda**2 - kappa_lambda)
        * float(zeta_K)
    )
    if sv2 < 0 and abs(sv2) < 1e-14:
        sv2 = 0.0
    if sv2 < 0:
        raise ValueError(f"Theoretical SV^2 is negative: {sv2}.")

    standard_value = float(np.sqrt(sv2))
    half_width = float(L) * standard_value
    lower = center - half_width
    upper = center + half_width
    return {
        "lo": float(lower),
        "hi": float(upper),
        "center": float(center),
        "SV": standard_value,
        "half_width": float(half_width),
        "lambda_value": float(lambda_value),
        "delta_lambda": delta_lambda,
        "rho_lambda": rho_lambda,
        "kappa_lambda": kappa_lambda,
        "theta_K": float(theta_K),
        "eta_K": float(eta_K),
        "zeta_K": float(zeta_K),
    }


