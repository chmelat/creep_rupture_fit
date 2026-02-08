"""
Larson-Miller creep model.

Larson-Miller equation:
    log10(tr) = (a0 + a1*log10(sigma) + a2*log10^2(sigma) + ...) / T - C

where:
    tr = time to rupture [h]
    sigma = stress [MPa]
    T = temperature [K]
    C = Larson-Miller constant
    a0, a1, a2, ... = polynomial coefficients
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist


@dataclass
class LMParams:
    """Larson-Miller model parameters."""
    C: float
    coeffs: list[float]
    C_fixed: bool = False

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    @property
    def n_fitted(self) -> int:
        """Number of fitted parameters."""
        return len(self.coeffs) if self.C_fixed else len(self.coeffs) + 1

    def as_array(self) -> np.ndarray:
        """Return fitted parameters as array (excludes C if fixed)."""
        if self.C_fixed:
            return np.array(self.coeffs)
        return np.array([self.C] + self.coeffs)

    def predict_log_tr(self, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Calculate log10(tr) for given sigma and T."""
        log_sigma = np.log10(sigma)
        poly_sum = sum(ai * (log_sigma ** i) for i, ai in enumerate(self.coeffs))
        return poly_sum / T - self.C


@dataclass
class LMFitResult:
    """Result of Larson-Miller fit."""
    params: LMParams
    mse: float
    r_squared: float
    n_points: int
    success: bool


@dataclass
class CIResult:
    """Confidence interval results."""
    se: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    dof: int
    C_fixed: bool
    method: str = "asymptotic"
    n_bootstrap: int | None = None
    n_successful: int | None = None


def compute_residuals(params: LMParams, sigma: np.ndarray, T: np.ndarray,
                      tr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """
    Compute residuals and fit statistics.

    Returns:
        (residuals, ss_res, ss_tot, mse)
    """
    log_tr_calc = params.predict_log_tr(sigma, T)
    log_tr_exp = np.log10(tr)

    residuals = log_tr_exp - log_tr_calc
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_tr_exp - np.mean(log_tr_exp)) ** 2)
    mse = ss_res / len(tr)

    return residuals, ss_res, ss_tot, mse


def _objective_function(param_array: np.ndarray, sigma: np.ndarray, T: np.ndarray,
                        tr: np.ndarray, fix_c: float | None) -> float:
    """Objective function for optimization (MSE in log space)."""
    if fix_c is not None:
        C, coeffs = fix_c, param_array
    else:
        C, coeffs = param_array[0], param_array[1:]

    params = LMParams(C=C, coeffs=list(coeffs), C_fixed=(fix_c is not None))
    _, _, _, mse = compute_residuals(params, sigma, T, tr)
    return mse


def fit_larson_miller(sigma: np.ndarray, T: np.ndarray, tr: np.ndarray,
                      order: int = 1, fix_c: float | None = None) -> LMFitResult:
    """
    Fit Larson-Miller equation to experimental data.

    Args:
        sigma: stress values [MPa]
        T: temperature values [K]
        tr: time to rupture [h]
        order: polynomial order (1 or 2)
        fix_c: if not None, fix C to this value

    Returns:
        LMFitResult with fitted parameters and statistics
    """
    # Initial guesses:
    # - C ~ 20: typical LM constant for steels (literature range 15-25)
    # - a0 ~ 20000: typical P_LM value at intermediate stress
    # - higher order coeffs = 0: let optimizer find them
    if fix_c is not None:
        x0 = [20000] + [0] * order
    else:
        x0 = [20, 20000] + [0] * order

    # Nelder-Mead: derivative-free, robust for this smooth 3-4 param problem
    # maxiter=10000: enough for convergence even with poor initial guess
    # xatol/fatol=1e-8: tight tolerance for reproducible results
    result = minimize(
        _objective_function,
        x0,
        args=(sigma, T, tr, fix_c),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
    )

    if fix_c is not None:
        C, coeffs = fix_c, list(result.x)
    else:
        C, coeffs = result.x[0], list(result.x[1:])

    params = LMParams(C=C, coeffs=coeffs, C_fixed=(fix_c is not None))
    _, ss_res, ss_tot, mse = compute_residuals(params, sigma, T, tr)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return LMFitResult(
        params=params,
        mse=mse,
        r_squared=r_squared,
        n_points=len(tr),
        success=result.success
    )


def predict_stress_for_tr(tr_target: float, T: float, params: LMParams,
                          sigma_range: tuple = (0.1, 1000)) -> float | None:
    # sigma_range (0.1, 1000) MPa: practical engineering range
    # - 0.1 MPa: below any realistic creep stress
    # - 1000 MPa: above yield strength of most steels
    """
    Predict stress for a given time to rupture and temperature.

    Solves: P_LM(sigma) = T * (C + log10(tr_target))
    """
    log_tr_target = np.log10(tr_target)
    k = T * (params.C + log_tr_target)

    coeffs = params.coeffs
    order = params.order

    if order == 1:
        a0, a1 = coeffs
        if a1 == 0:
            return None
        x = (k - a0) / a1
        sigma = 10 ** x

    elif order == 2:
        a0, a1, a2 = coeffs

        if a2 == 0:
            if a1 == 0:
                return None
            x = (k - a0) / a1
            sigma = 10 ** x
        else:
            D = a1**2 - 4 * a2 * (a0 - k)
            if D < 0:
                return None
            elif D == 0:
                x = -a1 / (2 * a2)
                sigma = 10 ** x
            else:
                sqrt_D = np.sqrt(D)
                x1 = (-a1 + sqrt_D) / (2 * a2)
                x2 = (-a1 - sqrt_D) / (2 * a2)
                sigma1, sigma2 = 10 ** x1, 10 ** x2
                # Choose root on the physically correct branch:
                # dP_LM/dx < 0 (higher stress → shorter life)
                candidates = []
                for x, s in [(x1, sigma1), (x2, sigma2)]:
                    if s > 0 and (a1 + 2 * a2 * x) < 0:
                        candidates.append(s)
                if not candidates:
                    # Fallback: no root on descending branch, take any positive
                    candidates = [s for s in (sigma1, sigma2) if s > 0]
                if not candidates:
                    return None
                sigma = min(candidates)
    else:
        raise ValueError(f"Analytical solution not implemented for order {order}")

    if sigma_range[0] <= sigma <= sigma_range[1]:
        return sigma
    return None


def compute_jacobian(params: LMParams, sigma: np.ndarray, T: np.ndarray,
                     rel_eps: float = 1e-6, abs_eps: float = 1e-10) -> np.ndarray:
    """Compute Jacobian matrix numerically with adaptive step size.

    rel_eps=1e-6: relative step ~sqrt(machine epsilon) for float64,
                  balances truncation vs roundoff error
    abs_eps=1e-10: absolute floor for near-zero parameters
    """
    n = len(sigma)
    p = params.n_fitted
    J = np.zeros((n, p))

    param_array = params.as_array()

    for j in range(p):
        # Adaptive step size based on parameter magnitude
        eps_j = max(abs(param_array[j]) * rel_eps, abs_eps)

        params_plus = param_array.copy()
        params_plus[j] += eps_j
        params_minus = param_array.copy()
        params_minus[j] -= eps_j

        if params.C_fixed:
            y_plus = LMParams(params.C, list(params_plus), True).predict_log_tr(sigma, T)
            y_minus = LMParams(params.C, list(params_minus), True).predict_log_tr(sigma, T)
        else:
            y_plus = LMParams(params_plus[0], list(params_plus[1:]), False).predict_log_tr(sigma, T)
            y_minus = LMParams(params_minus[0], list(params_minus[1:]), False).predict_log_tr(sigma, T)

        J[:, j] = (y_plus - y_minus) / (2 * eps_j)

    return J


def compute_confidence_intervals(fit: LMFitResult, sigma: np.ndarray, T: np.ndarray,
                                  tr: np.ndarray, confidence: float = 0.95) -> CIResult | None:
    """
    Compute confidence intervals using asymptotic approximation.

    Uses Jacobian-based covariance matrix: Cov = s^2 * (J'J)^-1
    """
    n = len(tr)
    p = fit.params.n_fitted
    dof = n - p

    if dof <= 0:
        return None

    J = compute_jacobian(fit.params, sigma, T)
    _, ss_res, _, _ = compute_residuals(fit.params, sigma, T, tr)
    s2 = ss_res / dof

    JtJ = J.T @ J

    # Check matrix conditioning - poorly conditioned matrix gives unreliable CI
    # Threshold 1e12: conservative limit, ~4 digits of precision lost
    # (machine epsilon ~1e-16, so 1e12 * eps ~1e-4 relative error)
    cond = np.linalg.cond(JtJ)
    if cond > 1e12:
        return None

    try:
        JtJ_inv = np.linalg.inv(JtJ)
        Cov = s2 * JtJ_inv
    except np.linalg.LinAlgError:
        return None

    se = np.sqrt(np.diag(Cov))

    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, dof)
    param_array = fit.params.as_array()
    ci_lower = param_array - t_crit * se
    ci_upper = param_array + t_crit * se

    return CIResult(
        se=se.tolist(),
        ci_lower=ci_lower.tolist(),
        ci_upper=ci_upper.tolist(),
        dof=dof,
        C_fixed=fit.params.C_fixed,
        method="asymptotic"
    )


def bootstrap_confidence_intervals(sigma: np.ndarray, T: np.ndarray, tr: np.ndarray,
                                    order: int, fix_c: float | None = None,
                                    n_bootstrap: int = 200,
                                    confidence: float = 0.95,
                                    seed: int | None = None) -> CIResult | None:
    """Compute confidence intervals using bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(tr)
    param_samples = []
    C_fixed = (fix_c is not None)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        try:
            result = fit_larson_miller(sigma[idx], T[idx], tr[idx], order, fix_c)
            if result.success:
                param_samples.append(result.params.as_array())
        except Exception:
            continue

    # Minimum 10 successful fits required for meaningful percentile estimation
    # (95% CI needs at least 2.5th and 97.5th percentiles)
    if len(param_samples) < 10:
        return None

    param_samples = np.array(param_samples)

    alpha = 1 - confidence
    ci_lower = np.percentile(param_samples, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(param_samples, 100 * (1 - alpha / 2), axis=0)
    se = np.std(param_samples, axis=0)

    return CIResult(
        se=se.tolist(),
        ci_lower=ci_lower.tolist(),
        ci_upper=ci_upper.tolist(),
        dof=n - (len(param_samples[0])),
        C_fixed=C_fixed,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        n_successful=len(param_samples)
    )


