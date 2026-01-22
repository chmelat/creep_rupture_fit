"""
Wilshire creep model.

Wilshire equation:
    σ/σ_TS = exp(-k * (tr * exp(-Q/RT))^u)

where:
    σ = stress [MPa]
    σ_TS = tensile strength at creep temperature [MPa]
    tr = time to rupture [h]
    T = temperature [K]
    R = 8.314 J/K/mol (universal gas constant)
    Q = activation energy [J/mol]
    k, u = model parameters

Wilshire parameter: PW = ln(tr * exp(-Q/RT)) = ln(tr) - Q/(R*T)

Multi-region approach:
    For steel T23, two regions are typically used:
    - Region 1 (PW < -23): k = 148.8, u = 0.2451
    - Region 2 (PW >= -23): k = 8.904, u = 0.1218
    - Q = 198000 J/mol (common for both regions)
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


R_GAS = 8.314  # Universal gas constant [J/K/mol]


@dataclass
class TensileData:
    """Tensile strength data as function of temperature."""
    T_kelvin: np.ndarray
    sigma_TS: np.ndarray
    _interpolator: interp1d = field(default=None, repr=False)

    def __post_init__(self):
        """Create interpolator after initialization."""
        if self._interpolator is None:
            self._interpolator = interp1d(
                self.T_kelvin, self.sigma_TS,
                kind='linear', fill_value='extrapolate'
            )

    @property
    def T_min(self) -> float:
        """Minimum temperature in data."""
        return float(np.min(self.T_kelvin))

    @property
    def T_max(self) -> float:
        """Maximum temperature in data."""
        return float(np.max(self.T_kelvin))

    def interpolate(self, T: float | np.ndarray, warn_extrapolation: bool = True) -> float | np.ndarray:
        """
        Interpolate σ_TS for given temperature(s).

        Args:
            T: temperature(s) in Kelvin
            warn_extrapolation: if True, warn when extrapolating outside data range
        """
        T_arr = np.atleast_1d(T)

        if warn_extrapolation:
            below = T_arr < self.T_min
            above = T_arr > self.T_max
            if np.any(below) or np.any(above):
                out_of_range = T_arr[below | above]
                warnings.warn(
                    f"Extrapolating tensile strength outside data range "
                    f"[{self.T_min - 273.15:.0f}, {self.T_max - 273.15:.0f}] °C. "
                    f"Temperatures outside range: {out_of_range - 273.15} °C",
                    UserWarning
                )

        result = self._interpolator(T)

        # Check for non-physical values
        if np.any(result <= 0):
            raise ValueError(
                "Interpolated tensile strength is non-positive. "
                "Check temperature range or tensile data."
            )

        return result

    @classmethod
    def from_file(cls, filepath: str, temp_unit: str = 'C') -> 'TensileData':
        """Load tensile data from CSV file."""
        T_list, sigma_list = [], []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        T_list.append(float(parts[0].strip()))
                        sigma_list.append(float(parts[1].strip()))
                    except ValueError:
                        continue

        if not T_list:
            raise ValueError("No valid data found in tensile data file")

        T = np.array(T_list)
        sigma_TS = np.array(sigma_list)

        # Convert to Kelvin if needed
        if temp_unit.upper() == 'C':
            T = T + 273.15

        return cls(T_kelvin=T, sigma_TS=sigma_TS)


@dataclass
class WSHRegion:
    """Parameters for one region in multi-region Wilshire model."""
    k: float
    u: float
    PW_min: float = -np.inf
    PW_max: float = np.inf
    Q: float | None = None  # per-region Q (None = use global)

    def contains(self, PW: float | np.ndarray) -> bool | np.ndarray:
        """Check if PW is in this region."""
        return (PW >= self.PW_min) & (PW < self.PW_max)


@dataclass
class WSHParams:
    """Wilshire model parameters."""
    Q: float  # activation energy [J/mol]
    regions: list[WSHRegion]  # parameters for each region
    R: float = R_GAS  # gas constant

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def get_region(self, PW: float) -> WSHRegion:
        """Get region for given PW value."""
        for region in self.regions:
            if region.contains(PW):
                return region
        # Return last region as fallback
        return self.regions[-1]


@dataclass
class WSHFitResult:
    """Result of Wilshire fit."""
    params: WSHParams
    mse: float
    r_squared: float
    n_points: int
    success: bool
    breakpoints: list[float] | None = None


@dataclass
class WSHRegionCI:
    """CI for a single Wilshire region."""
    k_se: float
    k_ci_lower: float
    k_ci_upper: float
    u_se: float
    u_ci_lower: float
    u_ci_upper: float
    # Per-region Q (if --per-region-q)
    Q_se: float | None = None
    Q_ci_lower: float | None = None
    Q_ci_upper: float | None = None


@dataclass
class WSHCIResult:
    """Confidence interval results for Wilshire model."""
    Q_se: float
    Q_ci_lower: float
    Q_ci_upper: float
    Q_fixed: bool
    regions: list[WSHRegionCI]
    dof: int
    method: str = "bootstrap"
    n_bootstrap: int | None = None
    n_successful: int | None = None
    breakpoint_ci: list[tuple[float, float]] | None = None


def compute_wilshire_parameter(tr: np.ndarray, T: np.ndarray, Q: float,
                                R: float = R_GAS) -> np.ndarray:
    """
    Compute Wilshire parameter.

    PW = ln(tr * exp(-Q/RT)) = ln(tr) - Q/(R*T)

    Args:
        tr: time to rupture [h]
        T: temperature [K]
        Q: activation energy [J/mol]
        R: gas constant [J/K/mol]

    Returns:
        Wilshire parameter PW
    """
    return np.log(tr) - Q / (R * T)


def _compute_y(sigma: np.ndarray, sigma_TS: np.ndarray, validate: bool = True) -> np.ndarray:
    """
    Compute transformed variable y = ln(-ln(σ/σ_TS)).

    This linearizes the Wilshire equation:
        y = ln(k) + u * PW

    Args:
        sigma: stress values [MPa]
        sigma_TS: tensile strength values [MPa]
        validate: if True, raise error when σ >= σ_TS

    Raises:
        ValueError: if any σ >= σ_TS (physically impossible for creep)
    """
    ratio = sigma / sigma_TS

    if validate:
        invalid = ratio >= 1.0
        if np.any(invalid):
            invalid_idx = np.where(invalid)[0]
            raise ValueError(
                f"Stress must be less than tensile strength (σ < σ_TS). "
                f"Found {np.sum(invalid)} invalid points at indices {invalid_idx.tolist()}: "
                f"σ = {sigma[invalid]}, σ_TS = {sigma_TS[invalid]}"
            )

    # Clip to avoid numerical issues (only for values very close to bounds)
    # 1e-10: ~exp(-23), well beyond any physical σ/σ_TS ratio
    # 1-1e-10: prevents log(0) when σ ≈ σ_TS
    ratio = np.clip(ratio, 1e-10, 1 - 1e-10)
    return np.log(-np.log(ratio))


def _fit_single_region(PW: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit single region using linear regression.

    y = ln(k) + u * PW  =>  y = intercept + slope * PW

    Returns:
        (k, u, mse)
    """
    n = len(PW)
    if n < 2:
        return np.nan, np.nan, np.inf

    # Linear regression
    A = np.vstack([np.ones(n), PW]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = result[0]

    k = np.exp(intercept)
    u = slope

    y_pred = intercept + slope * PW
    mse = np.mean((y - y_pred) ** 2)

    return k, u, mse


def fit_piecewise_linear(PW: np.ndarray, y: np.ndarray,
                          breakpoints: list[float]) -> tuple[float, list[tuple[float, float]]]:
    """
    Fit piecewise linear function.

    Args:
        PW: Wilshire parameter values (sorted)
        y: transformed stress values ln(-ln(σ/σ_TS))
        breakpoints: list of PW values where breaks occur

    Returns:
        (total_mse, list of (k, u) for each region)
    """
    # Sort breakpoints and add boundaries
    bounds = [-np.inf] + sorted(breakpoints) + [np.inf]
    n_regions = len(bounds) - 1

    region_params = []
    total_ss_res = 0
    total_n = 0

    for i in range(n_regions):
        mask = (PW >= bounds[i]) & (PW < bounds[i + 1])
        if np.sum(mask) < 2:
            # Not enough points in region
            region_params.append((np.nan, np.nan))
            continue

        PW_region = PW[mask]
        y_region = y[mask]

        k, u, _ = _fit_single_region(PW_region, y_region)
        region_params.append((k, u))

        # Compute residuals
        y_pred = np.log(k) + u * PW_region
        total_ss_res += np.sum((y_region - y_pred) ** 2)
        total_n += len(PW_region)

    total_mse = total_ss_res / total_n if total_n > 0 else np.inf
    return total_mse, region_params


def compute_bic(n_points: int, mse: float, n_params: int) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = n * ln(MSE) + k * ln(n)

    Lower BIC is better.
    """
    if mse <= 0 or n_points <= 0:
        return np.inf
    return n_points * np.log(mse) + n_params * np.log(n_points)


def detect_breakpoints(PW: np.ndarray, y: np.ndarray,
                        max_regions: int = 3,
                        min_points_per_region: int = 5) -> list[float]:
    # min_points_per_region=5: minimum for reliable linear regression
    # (2 params per region, need overdetermined system + some robustness)
    """
    Automatically detect breakpoints using BIC criterion.

    Algorithm:
        1. For n_regions = 1, 2, ..., max_regions:
           - Find optimal breakpoint positions
           - Fit piecewise linear regression
           - Compute BIC
        2. Select model with lowest BIC

    Args:
        PW: Wilshire parameter values
        y: ln(-ln(σ/σ_TS)) values
        max_regions: maximum number of regions to try
        min_points_per_region: minimum points required per region

    Returns:
        List of breakpoint PW values (empty for single region)
    """
    n = len(PW)
    if n < 2 * min_points_per_region:
        return []  # Not enough data for multiple regions

    # Sort data
    sort_idx = np.argsort(PW)
    PW_sorted = PW[sort_idx]
    y_sorted = y[sort_idx]

    best_bic = np.inf
    best_breakpoints = []

    # Single region (no breakpoints)
    k, u, mse = _fit_single_region(PW_sorted, y_sorted)
    n_params = 2  # k, u
    bic = compute_bic(n, mse, n_params)
    if bic < best_bic:
        best_bic = bic
        best_breakpoints = []

    # Two regions (one breakpoint)
    if max_regions >= 2 and n >= 2 * min_points_per_region:
        # Search for optimal breakpoint
        candidate_idx = range(min_points_per_region, n - min_points_per_region)

        for i in candidate_idx:
            bp = (PW_sorted[i - 1] + PW_sorted[i]) / 2
            mse, _ = fit_piecewise_linear(PW_sorted, y_sorted, [bp])
            n_params = 4  # k1, u1, k2, u2
            bic = compute_bic(n, mse, n_params)

            if bic < best_bic:
                best_bic = bic
                best_breakpoints = [bp]

    # Three regions (two breakpoints)
    if max_regions >= 3 and n >= 3 * min_points_per_region:
        candidate_idx = list(range(min_points_per_region, n - 2 * min_points_per_region))

        for i, idx1 in enumerate(candidate_idx[:-min_points_per_region]):
            for idx2 in candidate_idx[i + min_points_per_region:]:
                bp1 = (PW_sorted[idx1 - 1] + PW_sorted[idx1]) / 2
                bp2 = (PW_sorted[idx2 - 1] + PW_sorted[idx2]) / 2
                mse, _ = fit_piecewise_linear(PW_sorted, y_sorted, [bp1, bp2])
                n_params = 6  # k1, u1, k2, u2, k3, u3
                bic = compute_bic(n, mse, n_params)

                if bic < best_bic:
                    best_bic = bic
                    best_breakpoints = [bp1, bp2]

    return sorted(best_breakpoints)


def _estimate_Q(sigma: np.ndarray, T: np.ndarray, tr: np.ndarray,
                 sigma_TS: np.ndarray, Q_range: tuple = (100000, 400000)) -> float:
    # Q_range 100-400 kJ/mol covers all common engineering alloys:
    # - Al alloys: 130-150 kJ/mol
    # - Carbon/Cr-Mo steels: 180-280 kJ/mol
    # - Austenitic steels: 250-350 kJ/mol
    # - Ni superalloys: 280-400 kJ/mol
    """
    Estimate activation energy Q by minimizing MSE of single-region fit.

    This provides a starting point for multi-region fitting.
    """
    y = _compute_y(sigma, sigma_TS)

    def objective(Q):
        PW = compute_wilshire_parameter(tr, T, Q)
        _, _, mse = _fit_single_region(PW, y)
        return mse

    result = minimize_scalar(objective, bounds=Q_range, method='bounded')
    return result.x


def _estimate_Q_per_region(sigma: np.ndarray, T: np.ndarray, tr: np.ndarray,
                            sigma_TS: np.ndarray, mask: np.ndarray,
                            Q_range: tuple = (100000, 400000),
                            min_points: int = 5) -> float | None:
    """
    Estimate Q for a single region.

    Args:
        sigma, T, tr, sigma_TS: data arrays
        mask: boolean array selecting points for this region
        Q_range: search range for Q
        min_points: minimum points required to estimate Q

    Returns:
        Estimated Q or None if not enough points
    """
    if np.sum(mask) < min_points:
        return None

    sigma_r = sigma[mask]
    T_r = T[mask]
    tr_r = tr[mask]
    sigma_TS_r = sigma_TS[mask]

    y = _compute_y(sigma_r, sigma_TS_r)

    def objective(Q):
        PW = compute_wilshire_parameter(tr_r, T_r, Q)
        _, _, mse = _fit_single_region(PW, y)
        return mse

    result = minimize_scalar(objective, bounds=Q_range, method='bounded')
    return result.x


def fit_wilshire(sigma: np.ndarray, T: np.ndarray, tr: np.ndarray,
                  tensile_data: TensileData,
                  region_boundaries: list[float] | None = None,
                  fix_Q: float | None = None,
                  auto_detect: bool = True,
                  max_regions: int = 3,
                  per_region_Q: bool = False) -> WSHFitResult:
    """
    Fit Wilshire equation to experimental data.

    Args:
        sigma: stress values [MPa]
        T: temperature values [K]
        tr: time to rupture [h]
        tensile_data: TensileData object with σ_TS vs T
        region_boundaries: manual breakpoint PW values (None = auto-detect)
        fix_Q: fixed activation energy [J/mol] (None = estimate)
        auto_detect: whether to auto-detect breakpoints if region_boundaries is None
        max_regions: maximum number of regions for auto-detection
        per_region_Q: if True, estimate separate Q for each region

    Returns:
        WSHFitResult with fitted parameters
    """
    # Get tensile strength at test temperatures
    sigma_TS = tensile_data.interpolate(T)

    # Compute transformed variable
    y = _compute_y(sigma, sigma_TS)

    # Estimate or use fixed Q (global Q, always needed for breakpoint detection)
    if fix_Q is not None:
        Q = fix_Q
        if per_region_Q:
            warnings.warn(
                "Both --fix-q and --per-region-q specified. "
                "Using fixed Q for initial breakpoint detection, "
                "then estimating per-region Q.",
                UserWarning
            )
    else:
        Q = _estimate_Q(sigma, T, tr, sigma_TS)

    # Compute Wilshire parameter (using global Q for breakpoint detection)
    PW = compute_wilshire_parameter(tr, T, Q)

    # Determine breakpoints
    if region_boundaries is not None:
        breakpoints = sorted(region_boundaries)
    elif auto_detect:
        breakpoints = detect_breakpoints(PW, y, max_regions=max_regions)
    else:
        breakpoints = []

    # Build region masks based on PW (using global Q)
    bounds = [-np.inf] + breakpoints + [np.inf]
    n_regions = len(bounds) - 1
    region_masks = []
    for i in range(n_regions):
        mask = (PW >= bounds[i]) & (PW < bounds[i + 1])
        region_masks.append(mask)

    # Fit regions
    if per_region_Q and n_regions > 1:
        # Per-region Q estimation
        regions = []
        total_ss_res = 0
        total_n = 0

        for i, mask in enumerate(region_masks):
            if np.sum(mask) < 2:
                continue

            # Estimate Q for this region
            Q_region = _estimate_Q_per_region(sigma, T, tr, sigma_TS, mask)

            if Q_region is None:
                # Not enough points, use global Q
                Q_region = Q

            # Compute PW with region-specific Q
            PW_region = compute_wilshire_parameter(tr[mask], T[mask], Q_region)
            y_region = y[mask]

            # Fit k, u for this region
            k, u, _ = _fit_single_region(PW_region, y_region)

            if not np.isnan(k):
                regions.append(WSHRegion(
                    k=k, u=u,
                    PW_min=bounds[i], PW_max=bounds[i + 1],
                    Q=Q_region
                ))

                # Compute residuals
                y_pred_region = np.log(k) + u * PW_region
                total_ss_res += np.sum((y_region - y_pred_region) ** 2)
                total_n += len(y_region)

        mse = total_ss_res / total_n if total_n > 0 else np.inf

        if not regions:
            # Fallback to single region with global Q
            k, u, mse = _fit_single_region(PW, y)
            regions = [WSHRegion(k=k, u=u)]
            breakpoints = []

    else:
        # Single Q for all regions (default behavior)
        mse, region_params = fit_piecewise_linear(PW, y, breakpoints)

        regions = []
        for i, (k, u) in enumerate(region_params):
            if not np.isnan(k):
                regions.append(WSHRegion(k=k, u=u, PW_min=bounds[i], PW_max=bounds[i + 1]))

        if not regions:
            # Fallback to single region
            k, u, mse = _fit_single_region(PW, y)
            regions = [WSHRegion(k=k, u=u)]
            breakpoints = []

    params = WSHParams(Q=Q, regions=regions)

    # Compute R-squared
    # For per-region Q, we need to use region-specific PW values
    y_pred = np.zeros_like(y)
    for region in regions:
        Q_eff = region.Q if region.Q is not None else Q
        PW_for_region = compute_wilshire_parameter(tr, T, Q_eff)
        mask = region.contains(PW)  # Use global PW for region selection
        y_pred[mask] = np.log(region.k) + region.u * PW_for_region[mask]

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return WSHFitResult(
        params=params,
        mse=mse,
        r_squared=r_squared,
        n_points=len(tr),
        success=True,
        breakpoints=breakpoints
    )


def predict_stress_wsh(tr_target: float, T: float, params: WSHParams,
                        tensile_data: TensileData) -> float | None:
    """
    Predict stress for given time to rupture and temperature.

    σ = σ_TS * exp(-k * (tr * exp(-Q/RT))^u)

    The equation requires solving for σ given PW, which may span multiple regions.
    We use iteration to find the correct region.
    """
    sigma_TS = float(tensile_data.interpolate(T))

    # Use global Q for region selection
    PW = float(compute_wilshire_parameter(np.array([tr_target]), np.array([T]), params.Q)[0])

    region = params.get_region(PW)

    # Use region-specific Q if available, otherwise global Q
    Q_eff = region.Q if region.Q is not None else params.Q
    PW_eff = float(compute_wilshire_parameter(np.array([tr_target]), np.array([T]), Q_eff)[0])

    # σ/σ_TS = exp(-k * exp(u * PW))
    # Using PW = ln(tr) - Q/(R*T), we have:
    # exp(PW) = tr * exp(-Q/RT)
    # So: σ/σ_TS = exp(-k * exp(u * PW))
    exponent = -region.k * np.exp(region.u * PW_eff)
    ratio = np.exp(exponent)

    sigma = sigma_TS * ratio

    if sigma <= 0 or sigma > sigma_TS:
        return None

    return sigma


def predict_log_tr_wsh(sigma: np.ndarray, T: np.ndarray, params: WSHParams,
                        tensile_data: TensileData,
                        max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
    # max_iter=10: typically converges in 2-3 iterations for well-separated regions
    # tol=1e-6: PW precision ~6 digits, sufficient for time predictions
    """
    Predict log10(tr) for given stress and temperature.

    From: σ/σ_TS = exp(-k * exp(u * PW))
    We get: PW = (y - ln(k)) / u  where y = ln(-ln(σ/σ_TS))
    Then: log10(tr) = (PW + Q/(R*T)) / ln(10)

    For multi-region models, we iterate to find the correct region for each point.
    For per-region Q, each region uses its own Q value.

    Args:
        sigma: stress values [MPa]
        T: temperature values [K]
        params: Wilshire model parameters
        tensile_data: tensile strength data
        max_iter: maximum iterations for region convergence
        tol: tolerance for PW convergence

    Returns:
        log10(tr) predictions
    """
    sigma_TS = tensile_data.interpolate(T, warn_extrapolation=False)
    y = _compute_y(sigma, sigma_TS, validate=False)  # y = ln(-ln(σ/σ_TS))

    n = len(sigma)
    PW = np.zeros(n)
    Q_arr = np.full(n, params.Q)  # Q value for each point

    # Single region - no iteration needed
    if params.n_regions == 1:
        region = params.regions[0]
        PW = (y - np.log(region.k)) / region.u
        if region.Q is not None:
            Q_arr[:] = region.Q
    else:
        # Multi-region: iterate to convergence
        # Initial guess using middle region
        mid_region = params.regions[len(params.regions) // 2]
        PW_prev = (y - np.log(mid_region.k)) / mid_region.u

        for iteration in range(max_iter):
            PW_new = np.zeros(n)

            for i in range(n):
                region = params.get_region(PW_prev[i])
                PW_new[i] = (y[i] - np.log(region.k)) / region.u
                # Store per-region Q
                Q_arr[i] = region.Q if region.Q is not None else params.Q

            # Check convergence
            if np.max(np.abs(PW_new - PW_prev)) < tol:
                PW = PW_new
                break

            PW_prev = PW_new
        else:
            # Max iterations reached - use last values
            PW = PW_new
            warnings.warn(
                f"predict_log_tr_wsh: max iterations ({max_iter}) reached "
                f"without convergence (tol={tol})",
                UserWarning
            )

    # Convert PW to log10(tr)
    # PW = ln(tr) - Q/(R*T)  =>  ln(tr) = PW + Q/(R*T)
    # Use per-region Q if available
    ln_tr = PW + Q_arr / (params.R * T)
    log_tr = ln_tr / np.log(10)

    return log_tr


def bootstrap_confidence_intervals_wsh(
    sigma: np.ndarray,
    T: np.ndarray,
    tr: np.ndarray,
    tensile_data: TensileData,
    region_boundaries: list[float] | None = None,
    fix_Q: float | None = None,
    per_region_Q: bool = False,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int | None = None,
) -> WSHCIResult | None:
    """
    Bootstrap confidence intervals for Wilshire model.

    Args:
        sigma: stress values [MPa]
        T: temperature values [K]
        tr: time to rupture [h]
        tensile_data: TensileData object with σ_TS vs T
        region_boundaries: manual breakpoint PW values (None = auto-detect)
        fix_Q: fixed activation energy [J/mol] (None = estimate)
        per_region_Q: if True, estimate separate Q for each region
        n_bootstrap: number of bootstrap iterations (default: 200)
        confidence: confidence level (default: 0.95)
        seed: random seed for reproducibility

    Returns:
        WSHCIResult with confidence intervals, or None if failed
    """
    rng = np.random.default_rng(seed)
    n = len(tr)
    Q_fixed = (fix_Q is not None)

    # Perform original fit to determine n_regions and breakpoints
    original_fit = fit_wilshire(
        sigma, T, tr, tensile_data,
        region_boundaries=region_boundaries,
        fix_Q=fix_Q,
        per_region_Q=per_region_Q
    )
    n_regions = original_fit.params.n_regions
    original_breakpoints = original_fit.breakpoints or []

    # Storage for bootstrap samples
    Q_samples = []
    region_samples = [{'k': [], 'u': [], 'Q': []} for _ in range(n_regions)]

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)

        try:
            # For bootstrap: use fixed breakpoints from original fit if auto-detected
            # This ensures consistent region structure across samples
            bp_for_fit = original_breakpoints if region_boundaries is None and original_breakpoints else region_boundaries

            result = fit_wilshire(
                sigma[idx], T[idx], tr[idx], tensile_data,
                region_boundaries=bp_for_fit,
                fix_Q=fix_Q,
                per_region_Q=per_region_Q
            )

            if not result.success:
                continue

            # Only accept samples with same number of regions
            if result.params.n_regions != n_regions:
                continue

            # Store Q (global)
            Q_samples.append(result.params.Q)

            # Store region parameters
            for i, region in enumerate(result.params.regions):
                region_samples[i]['k'].append(region.k)
                region_samples[i]['u'].append(region.u)
                if region.Q is not None:
                    region_samples[i]['Q'].append(region.Q)

        except Exception:
            continue

    # Require at least 10 successful samples for meaningful percentile estimation
    # (95% CI needs 2.5th and 97.5th percentiles)
    if len(Q_samples) < 10:
        return None

    Q_samples = np.array(Q_samples)
    alpha = 1 - confidence

    # Helper for robust SE using IQR
    def robust_se(arr: np.ndarray) -> float:
        """Compute robust SE using IQR, scaled to match std for normal distribution."""
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        return float(iqr / 1.35)  # IQR ≈ 1.35σ for normal distribution

    # Compute Q CI (use robust SE for consistency)
    Q_se = robust_se(Q_samples)
    Q_ci_lower = float(np.percentile(Q_samples, 100 * alpha / 2))
    Q_ci_upper = float(np.percentile(Q_samples, 100 * (1 - alpha / 2)))

    # Compute CI for each region
    regions_ci = []
    for i in range(n_regions):
        k_arr = np.array(region_samples[i]['k'])
        u_arr = np.array(region_samples[i]['u'])

        if len(k_arr) < 10:
            # Not enough samples for this region
            regions_ci.append(WSHRegionCI(
                k_se=np.nan, k_ci_lower=np.nan, k_ci_upper=np.nan,
                u_se=np.nan, u_ci_lower=np.nan, u_ci_upper=np.nan
            ))
            continue

        # Use robust SE for k (log-normal distributed, sensitive to outliers)
        k_se = robust_se(k_arr)
        k_ci_lower = float(np.percentile(k_arr, 100 * alpha / 2))
        k_ci_upper = float(np.percentile(k_arr, 100 * (1 - alpha / 2)))

        # u is more normally distributed, but use robust SE for consistency
        u_se = robust_se(u_arr)
        u_ci_lower = float(np.percentile(u_arr, 100 * alpha / 2))
        u_ci_upper = float(np.percentile(u_arr, 100 * (1 - alpha / 2)))

        # Per-region Q CI (if applicable)
        Q_region_se = None
        Q_region_ci_lower = None
        Q_region_ci_upper = None
        if region_samples[i]['Q']:
            Q_region_arr = np.array(region_samples[i]['Q'])
            if len(Q_region_arr) >= 10:
                Q_region_se = robust_se(Q_region_arr)
                Q_region_ci_lower = float(np.percentile(Q_region_arr, 100 * alpha / 2))
                Q_region_ci_upper = float(np.percentile(Q_region_arr, 100 * (1 - alpha / 2)))

        regions_ci.append(WSHRegionCI(
            k_se=k_se, k_ci_lower=k_ci_lower, k_ci_upper=k_ci_upper,
            u_se=u_se, u_ci_lower=u_ci_lower, u_ci_upper=u_ci_upper,
            Q_se=Q_region_se, Q_ci_lower=Q_region_ci_lower, Q_ci_upper=Q_region_ci_upper
        ))

    # Degrees of freedom
    n_params = 1 + 2 * n_regions  # Q + (k, u) per region
    if Q_fixed:
        n_params -= 1
    if per_region_Q and n_regions > 1:
        n_params += n_regions - 1  # Additional Q per region (minus global)
    dof = n - n_params

    return WSHCIResult(
        Q_se=Q_se,
        Q_ci_lower=Q_ci_lower,
        Q_ci_upper=Q_ci_upper,
        Q_fixed=Q_fixed,
        regions=regions_ci,
        dof=dof,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        n_successful=len(Q_samples),
        breakpoint_ci=None  # Not meaningful (breakpoints fixed during bootstrap)
    )


