#!/usr/bin/env python3
"""
CLI utility for fitting creep rupture equations.

Supported models:
- Larson-Miller (LM): log10(tr) = (a0 + a1*log10(sigma) + ...) / T - C
- Wilshire (WSH): σ/σ_TS = exp(-k * (tr * exp(-Q/RT))^u)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import numpy as np

# Add script directory to path for module imports when running as script
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from models.lm import (  # noqa: E402
    LMFitResult, CIResult,
    fit_larson_miller, predict_stress_for_tr,
    compute_confidence_intervals, bootstrap_confidence_intervals
)
from models.wsh import (  # noqa: E402
    TensileData, WSHFitResult, WSHCIResult, fit_wilshire, predict_stress_wsh, compute_wilshire_parameter,
    bootstrap_confidence_intervals_wsh
)
from version import __version__, get_version_string  # noqa: E402


@dataclass
class Prediction:
    """Single stress prediction."""
    T_celsius: float
    T_kelvin: float
    tr_target: float
    sigma: float | None


# =============================================================================
# Input parsing and validation
# =============================================================================

def detect_delimiter(line: str) -> str | None:
    """Auto-detect delimiter from a data line."""
    for delim in ['\t', ';', ',']:
        if delim in line:
            return delim
    return None  # None means split on whitespace


def validate_data(sigma: np.ndarray, T: np.ndarray, tr: np.ndarray) -> None:
    """Validate input data, raise ValueError if invalid."""
    if len(sigma) == 0:
        raise ValueError("No data points")

    if len(sigma) != len(T) or len(sigma) != len(tr):
        raise ValueError("Arrays must have same length")

    if np.any(sigma <= 0):
        raise ValueError("Stress (sigma) must be positive")

    if np.any(T <= 0):
        raise ValueError("Temperature must be positive (in Kelvin)")

    if np.any(tr <= 0):
        raise ValueError("Time to rupture must be positive")

    if np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(T)) or np.any(~np.isfinite(tr)):
        raise ValueError("Data contains non-finite values (inf/nan)")


def parse_input_file(filepath: str, delimiter: str = None, temp_unit: str = 'C') -> tuple:
    """
    Parse input data file.

    Expected columns: sigma [MPa], T [C or K], tr [h]

    Returns:
        tuple: (sigma, T_kelvin, tr) as numpy arrays
    """
    sigma_list, temp_list, tr_list = [], [], []
    skipped_lines = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    auto_delim = None

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        if auto_delim is None and delimiter is None:
            auto_delim = detect_delimiter(line)

        delim = delimiter if delimiter is not None else auto_delim

        parts = line.split() if delim is None else [p.strip() for p in line.split(delim)]

        try:
            values = [float(p) for p in parts if p]
            if len(values) >= 3:
                sigma_list.append(values[0])
                temp_list.append(values[1])
                tr_list.append(values[2])
            else:
                skipped_lines.append(line_num)
        except ValueError:
            skipped_lines.append(line_num)

    if not sigma_list:
        raise ValueError("No valid data found in input file")

    if skipped_lines:
        import warnings
        warnings.warn(
            f"Skipped {len(skipped_lines)} non-parseable line(s) "
            f"in {filepath}: {skipped_lines}",
            UserWarning
        )

    sigma = np.array(sigma_list)
    temp = np.array(temp_list)
    tr = np.array(tr_list)

    T_kelvin = temp + 273.15 if temp_unit.upper() == 'C' else temp

    validate_data(sigma, T_kelvin, tr)

    return sigma, T_kelvin, tr


# =============================================================================
# Output formatting - Larson-Miller
# =============================================================================

def format_param_with_ci(name: str, value: float, se: float, ci_lo: float, ci_hi: float,
                         precision: int = 2) -> str:
    """Format a single parameter with CI."""
    if precision == 2:
        return f"  {name} = {value:.2f} +/- {se:.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]"
    return f"  {name} = {value:.1f} +/- {se:.1f}  [{ci_lo:.1f}, {ci_hi:.1f}]"


def format_output_text_lm(fit: LMFitResult, predictions: list[Prediction] | None = None,
                          ci: CIResult | None = None) -> str:
    """Format LM output as human-readable text."""
    lines = [f"=== Larson-Miller Fit (order {fit.params.order}) ===", ""]

    if ci:
        lines.append("Fitted parameters (95% CI):")
        idx = 0

        if ci.C_fixed:
            lines.append(f"  C  = {fit.params.C:.2f} (fixed)")
        else:
            lines.append(format_param_with_ci("C ", fit.params.C, ci.se[0], ci.ci_lower[0], ci.ci_upper[0]))
            idx = 1

        for i, coeff in enumerate(fit.params.coeffs):
            lines.append(format_param_with_ci(f"a{i}", coeff, ci.se[idx + i], ci.ci_lower[idx + i], ci.ci_upper[idx + i], precision=1))
    else:
        lines.append("Fitted parameters:")
        lines.append(f"  C  = {fit.params.C:.2f}")
        for i, coeff in enumerate(fit.params.coeffs):
            lines.append(f"  a{i} = {coeff:.1f}")

    lines.extend(["", "Fit quality:"])
    lines.append(f"  Err (MSE log10(tr)) = {fit.mse:.4f}")
    lines.append(f"  R^2 = {fit.r_squared:.4f}")
    lines.append(f"  N  = {fit.n_points}")

    if ci:
        lines.append(f"  DOF = {ci.dof}")
        if ci.method == "bootstrap":
            lines.append(f"  Bootstrap: {ci.n_successful}/{ci.n_bootstrap} successful")

    if predictions:
        lines.extend(["", f"Predictions for tr = {predictions[0].tr_target:.0f} h:"])
        for pred in predictions:
            if pred.sigma is not None:
                lines.append(f"  T = {pred.T_celsius:.0f} C: sigma = {pred.sigma:.1f} MPa")
            else:
                lines.append(f"  T = {pred.T_celsius:.0f} C: no solution in range")

    return '\n'.join(lines)


# =============================================================================
# Output formatting - Wilshire
# =============================================================================

def format_output_text_wsh(fit: WSHFitResult, predictions: list[Prediction] | None = None,
                           ci: WSHCIResult | None = None) -> str:
    """Format WSH output as human-readable text."""
    params = fit.params
    lines = [f"=== Wilshire Fit ({params.n_regions} region{'s' if params.n_regions > 1 else ''}) ===", ""]

    # Check if any region has per-region Q
    has_per_region_q = any(r.Q is not None for r in params.regions)

    if ci:
        lines.append("Fitted parameters (95% CI):")
        # Global Q
        if ci.Q_fixed:
            lines.append(f"  Q  = {params.Q:.0f} J/mol (fixed)")
        else:
            lines.append(f"  Q  = {params.Q:.0f} +/- {ci.Q_se:.0f}  [{ci.Q_ci_lower:.0f}, {ci.Q_ci_upper:.0f}] J/mol")
        lines.append("")

        # Regions with CI
        for i, region in enumerate(params.regions):
            region_ci = ci.regions[i] if i < len(ci.regions) else None

            if params.n_regions > 1:
                pw_min_str = f"{region.PW_min:.1f}" if region.PW_min != -np.inf else "-inf"
                pw_max_str = f"{region.PW_max:.1f}" if region.PW_max != np.inf else "inf"
                lines.append(f"  Region {i+1} (PW in [{pw_min_str}, {pw_max_str})):")

            if region_ci and not np.isnan(region_ci.k_se):
                lines.append(f"    k = {region.k:.4f} +/- {region_ci.k_se:.4f}  [{region_ci.k_ci_lower:.4f}, {region_ci.k_ci_upper:.4f}]")
                lines.append(f"    u = {region.u:.4f} +/- {region_ci.u_se:.4f}  [{region_ci.u_ci_lower:.4f}, {region_ci.u_ci_upper:.4f}]")
                if region_ci.Q_se is not None:
                    lines.append(f"    Q = {region.Q:.0f} +/- {region_ci.Q_se:.0f}  [{region_ci.Q_ci_lower:.0f}, {region_ci.Q_ci_upper:.0f}] J/mol")
            else:
                lines.append(f"    k = {region.k:.4f}")
                lines.append(f"    u = {region.u:.4f}")
                if region.Q is not None:
                    lines.append(f"    Q = {region.Q:.0f} J/mol")

        if fit.breakpoints:
            lines.append("")
            lines.append(f"  Breakpoints: {', '.join(f'{bp:.2f}' for bp in fit.breakpoints)}")
    else:
        lines.append("Fitted parameters:")
        if not has_per_region_q:
            lines.append(f"  Q  = {params.Q:.0f} J/mol ({params.Q/1000:.1f} kJ/mol)")
        else:
            lines.append(f"  Q (global) = {params.Q:.0f} J/mol ({params.Q/1000:.1f} kJ/mol)")
        lines.append("")

        for i, region in enumerate(params.regions):
            if params.n_regions > 1:
                pw_min_str = f"{region.PW_min:.1f}" if region.PW_min != -np.inf else "-inf"
                pw_max_str = f"{region.PW_max:.1f}" if region.PW_max != np.inf else "inf"
                lines.append(f"  Region {i+1} (PW in [{pw_min_str}, {pw_max_str})):")
            lines.append(f"    k = {region.k:.4f}")
            lines.append(f"    u = {region.u:.4f}")
            if region.Q is not None:
                lines.append(f"    Q = {region.Q:.0f} J/mol ({region.Q/1000:.1f} kJ/mol)")

        if fit.breakpoints:
            lines.append("")
            lines.append(f"  Breakpoints: {', '.join(f'{bp:.2f}' for bp in fit.breakpoints)}")

    lines.extend(["", "Fit quality:"])
    lines.append(f"  Err (MSE ln space) = {fit.mse:.4f}")
    lines.append(f"  R^2 = {fit.r_squared:.4f}")
    lines.append(f"  N  = {fit.n_points}")

    if ci:
        lines.append(f"  DOF = {ci.dof}")
        if ci.method == "bootstrap":
            lines.append(f"  Bootstrap: {ci.n_successful}/{ci.n_bootstrap} successful")

    if predictions:
        lines.extend(["", f"Predictions for tr = {predictions[0].tr_target:.0f} h:"])
        for pred in predictions:
            if pred.sigma is not None:
                lines.append(f"  T = {pred.T_celsius:.0f} C: sigma = {pred.sigma:.1f} MPa")
            else:
                lines.append(f"  T = {pred.T_celsius:.0f} C: no solution in range")

    return '\n'.join(lines)


# =============================================================================
# Output formatting - JSON
# =============================================================================

def format_output_json_lm(fit: LMFitResult, predictions: list[Prediction] | None = None,
                          ci: CIResult | None = None) -> str:
    """Format LM output as JSON."""
    output = {
        'model': 'larson-miller',
        'fit': {
            'C': fit.params.C,
            'coeffs': fit.params.coeffs,
            'C_fixed': fit.params.C_fixed,
            'order': fit.params.order,
            'mse': fit.mse,
            'r_squared': fit.r_squared,
            'n_points': fit.n_points,
            'success': fit.success
        },
        'predictions': [
            {'T_celsius': p.T_celsius, 'T_kelvin': p.T_kelvin,
             'tr_target': p.tr_target, 'sigma': p.sigma}
            for p in predictions
        ] if predictions else None,
        'confidence_intervals': {
            'se': ci.se,
            'ci_lower': ci.ci_lower,
            'ci_upper': ci.ci_upper,
            'dof': ci.dof,
            'C_fixed': ci.C_fixed,
            'method': ci.method,
            'n_bootstrap': ci.n_bootstrap,
            'n_successful': ci.n_successful
        } if ci else None
    }
    return json.dumps(output, indent=2)


def format_output_json_wsh(fit: WSHFitResult, predictions: list[Prediction] | None = None,
                           ci: WSHCIResult | None = None) -> str:
    """Format WSH output as JSON."""
    output = {
        'model': 'wilshire',
        'fit': {
            'Q': fit.params.Q,
            'Q_kJ_mol': fit.params.Q / 1000,
            'regions': [
                {
                    'k': r.k,
                    'u': r.u,
                    'PW_min': r.PW_min if r.PW_min != -np.inf else None,
                    'PW_max': r.PW_max if r.PW_max != np.inf else None,
                    'Q': r.Q,
                    'Q_kJ_mol': r.Q / 1000 if r.Q is not None else None
                }
                for r in fit.params.regions
            ],
            'breakpoints': fit.breakpoints,
            'mse': fit.mse,
            'r_squared': fit.r_squared,
            'n_points': fit.n_points,
            'success': fit.success
        },
        'predictions': [
            {'T_celsius': p.T_celsius, 'T_kelvin': p.T_kelvin,
             'tr_target': p.tr_target, 'sigma': p.sigma}
            for p in predictions
        ] if predictions else None,
        'confidence_intervals': {
            'Q': {
                'se': ci.Q_se,
                'ci_lower': ci.Q_ci_lower,
                'ci_upper': ci.Q_ci_upper,
                'fixed': ci.Q_fixed
            },
            'regions': [
                {
                    'k_se': r.k_se if not np.isnan(r.k_se) else None,
                    'k_ci_lower': r.k_ci_lower if not np.isnan(r.k_ci_lower) else None,
                    'k_ci_upper': r.k_ci_upper if not np.isnan(r.k_ci_upper) else None,
                    'u_se': r.u_se if not np.isnan(r.u_se) else None,
                    'u_ci_lower': r.u_ci_lower if not np.isnan(r.u_ci_lower) else None,
                    'u_ci_upper': r.u_ci_upper if not np.isnan(r.u_ci_upper) else None,
                    'Q_se': r.Q_se,
                    'Q_ci_lower': r.Q_ci_lower,
                    'Q_ci_upper': r.Q_ci_upper
                }
                for r in ci.regions
            ],
            'breakpoints_ci': None,  # Not meaningful (fixed during bootstrap)
            'method': ci.method,
            'n_bootstrap': ci.n_bootstrap,
            'n_successful': ci.n_successful,
            'dof': ci.dof
        } if ci else None
    }
    return json.dumps(output, indent=2)


# =============================================================================
# Output formatting - CSV
# =============================================================================

def format_output_csv_lm(fit: LMFitResult, predictions: list[Prediction] | None = None,
                         ci: CIResult | None = None) -> str:
    """Format LM output as CSV."""
    lines = []

    if ci:
        lines.append("# Fitted parameters (95% CI)")
        lines.append("parameter,value,se,ci_lower,ci_upper")
        idx = 0

        if ci.C_fixed:
            lines.append(f"C,{fit.params.C},,,fixed")
        else:
            lines.append(f"C,{fit.params.C},{ci.se[0]},{ci.ci_lower[0]},{ci.ci_upper[0]}")
            idx = 1

        for i, coeff in enumerate(fit.params.coeffs):
            lines.append(f"a{i},{coeff},{ci.se[idx + i]},{ci.ci_lower[idx + i]},{ci.ci_upper[idx + i]}")
    else:
        lines.append("# Fitted parameters")
        lines.append("parameter,value")
        lines.append(f"C,{fit.params.C}")
        for i, coeff in enumerate(fit.params.coeffs):
            lines.append(f"a{i},{coeff}")

    lines.append(f"MSE,{fit.mse}")
    lines.append(f"R2,{fit.r_squared}")

    if predictions:
        lines.extend(["", "# Predictions", "T_celsius,tr_hours,sigma_MPa"])
        for pred in predictions:
            sigma_str = f"{pred.sigma:.2f}" if pred.sigma is not None else "N/A"
            lines.append(f"{pred.T_celsius},{pred.tr_target},{sigma_str}")

    return '\n'.join(lines)


def format_output_csv_wsh(fit: WSHFitResult, predictions: list[Prediction] | None = None,
                          ci: WSHCIResult | None = None) -> str:
    """Format WSH output as CSV."""
    lines = []

    if ci:
        lines.append("# Wilshire fitted parameters (95% CI)")
        lines.append("parameter,value,se,ci_lower,ci_upper")

        if ci.Q_fixed:
            lines.append(f"Q,{fit.params.Q},,,fixed")
        else:
            lines.append(f"Q,{fit.params.Q},{ci.Q_se},{ci.Q_ci_lower},{ci.Q_ci_upper}")

        lines.append(f"n_regions,{fit.params.n_regions},,,")

        for i, region in enumerate(fit.params.regions):
            prefix = f"region{i+1}_" if fit.params.n_regions > 1 else ""
            region_ci = ci.regions[i] if i < len(ci.regions) else None

            if region_ci and not np.isnan(region_ci.k_se):
                lines.append(f"{prefix}k,{region.k},{region_ci.k_se},{region_ci.k_ci_lower},{region_ci.k_ci_upper}")
                lines.append(f"{prefix}u,{region.u},{region_ci.u_se},{region_ci.u_ci_lower},{region_ci.u_ci_upper}")
            else:
                lines.append(f"{prefix}k,{region.k},,,")
                lines.append(f"{prefix}u,{region.u},,,")

            if fit.params.n_regions > 1:
                lines.append(f"{prefix}PW_min,{region.PW_min},,,")
                lines.append(f"{prefix}PW_max,{region.PW_max},,,")

            if region.Q is not None:
                if region_ci and region_ci.Q_se is not None:
                    lines.append(f"{prefix}Q,{region.Q},{region_ci.Q_se},{region_ci.Q_ci_lower},{region_ci.Q_ci_upper}")
                else:
                    lines.append(f"{prefix}Q,{region.Q},,,")
    else:
        lines.append("# Wilshire fitted parameters")
        lines.append("parameter,value")
        lines.append(f"Q,{fit.params.Q}")
        lines.append(f"n_regions,{fit.params.n_regions}")

        for i, region in enumerate(fit.params.regions):
            prefix = f"region{i+1}_" if fit.params.n_regions > 1 else ""
            lines.append(f"{prefix}k,{region.k}")
            lines.append(f"{prefix}u,{region.u}")
            if fit.params.n_regions > 1:
                lines.append(f"{prefix}PW_min,{region.PW_min}")
                lines.append(f"{prefix}PW_max,{region.PW_max}")
            if region.Q is not None:
                lines.append(f"{prefix}Q,{region.Q}")

    lines.append(f"MSE,{fit.mse}")
    lines.append(f"R2,{fit.r_squared}")

    if predictions:
        lines.extend(["", "# Predictions", "T_celsius,tr_hours,sigma_MPa"])
        for pred in predictions:
            sigma_str = f"{pred.sigma:.2f}" if pred.sigma is not None else "N/A"
            lines.append(f"{pred.T_celsius},{pred.tr_target},{sigma_str}")

    return '\n'.join(lines)


# =============================================================================
# Plotting
# =============================================================================

def create_plot(sigma: np.ndarray, T_kelvin: np.ndarray, tr: np.ndarray,
                predict_fn: Callable[[float, float], float | None], title: str,
                predict_temps_kelvin: list | None = None,
                output_file: str | None = None, show: bool = True) -> None:
    """Create plot of experimental data and fitted curves.

    Args:
        predict_fn: Callable(tr_val, T_k) -> stress or None
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot", file=sys.stderr)
        return

    unique_temps = np.unique(T_kelvin)
    if predict_temps_kelvin is None:
        predict_temps_kelvin = unique_temps

    unique_temps = np.asarray(unique_temps)
    predict_temps_kelvin = np.asarray(predict_temps_kelvin)

    # Create unified list of all temperatures (merge with tolerance)
    all_temps = list(unique_temps)
    for T_p in predict_temps_kelvin:
        if not np.any(np.isclose(T_p, all_temps, rtol=1e-9)):
            all_temps.append(T_p)
    all_temps = sorted(all_temps)

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_temps)))
    temp_to_color = {T_k: colors[i] for i, T_k in enumerate(all_temps)}

    fig, ax = plt.subplots(figsize=(10, 7))
    # tr_range: 1 to 10^6 hours covers short-term tests to design life (100k h)
    tr_range = np.logspace(0, 6, 100)

    def temp_in_array(T, arr):
        """Check if temperature T is in array with tolerance."""
        return np.any(np.isclose(T, arr, rtol=1e-9))

    # Plot both data and fit for each temperature together (for better legend ordering)
    for T_k in all_temps:
        T_c = T_k - 273.15
        color = temp_to_color[T_k]

        # Experimental data points
        if temp_in_array(T_k, unique_temps):
            mask = np.isclose(T_kelvin, T_k, rtol=1e-9)
            ax.scatter(np.log10(tr[mask]), sigma[mask],
                       color=color, label=f'{T_c:.0f} °C', alpha=0.7, s=50)

        # Fitted curve
        if temp_in_array(T_k, predict_temps_kelvin):
            sigma_pred, tr_valid = [], []
            for tr_val in tr_range:
                s = predict_fn(tr_val, T_k)
                # 1-1000 MPa: practical engineering stress range for plotting
                if s is not None and 1 < s < 1000:
                    sigma_pred.append(s)
                    tr_valid.append(tr_val)

            if sigma_pred:
                has_data = temp_in_array(T_k, unique_temps)
                label = None if has_data else f'{T_c:.0f} °C'
                ax.plot(np.log10(tr_valid), sigma_pred, '-',
                        color=color, linewidth=2, label=label)

    ax.set_xlabel('log₁₀(tr) [h]', fontsize=12)
    ax.set_ylabel('σ [MPa]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}", file=sys.stderr)

    if show and not output_file:
        plt.show()


def create_diagnostic_plot_wsh(sigma: np.ndarray, T_kelvin: np.ndarray, tr: np.ndarray,
                                fit: WSHFitResult, tensile_data: TensileData,
                                output_file: str | None = None, show: bool = True) -> None:
    """Create diagnostic plot for Wilshire model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping diagnostic plot", file=sys.stderr)
        return

    sigma_TS = tensile_data.interpolate(T_kelvin, warn_extrapolation=False)
    # Compute y = ln(-ln(σ/σ_TS)) - the linearized Wilshire transform
    ratio = np.clip(sigma / sigma_TS, 1e-10, 1 - 1e-10)
    y = np.log(-np.log(ratio))

    # Check if any region has per-region Q
    has_per_region_q = any(r.Q is not None for r in fit.params.regions)

    # For per-region Q, we need to plot each region with its own PW coordinates
    # Global PW is still used for determining region membership
    PW_global = compute_wilshire_parameter(tr, T_kelvin, fit.params.Q)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot data points colored by temperature
    unique_temps = np.unique(T_kelvin)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps)))

    if has_per_region_q:
        # For per-region Q: plot each region's data using region-specific PW
        # Build region masks first
        bounds = [-np.inf] + (fit.breakpoints or []) + [np.inf]

        for i, region in enumerate(fit.params.regions):
            Q_eff = region.Q if region.Q is not None else fit.params.Q
            PW_region = compute_wilshire_parameter(tr, T_kelvin, Q_eff)

            # Mask for this region (using global PW for region selection)
            region_mask = (PW_global >= bounds[i]) & (PW_global < bounds[i + 1])

            for T_k, color in zip(unique_temps, colors):
                temp_mask = np.isclose(T_kelvin, T_k, rtol=1e-9)
                combined_mask = region_mask & temp_mask
                if np.any(combined_mask):
                    T_c = T_k - 273.15
                    # Only add label once per temperature
                    label = f'{T_c:.0f} °C' if i == 0 else None
                    ax.scatter(PW_region[combined_mask], y[combined_mask],
                               color=color, alpha=0.7, s=50, label=label)
    else:
        # Single Q: plot all data using global PW
        for T_k, color in zip(unique_temps, colors):
            mask = np.isclose(T_kelvin, T_k, rtol=1e-9)
            T_c = T_k - 273.15
            ax.scatter(PW_global[mask], y[mask], color=color, alpha=0.7, s=50, label=f'{T_c:.0f} °C')

    # Plot breakpoints (only meaningful for single Q)
    if not has_per_region_q:
        for bp in (fit.breakpoints or []):
            ax.axvline(x=bp, color='red', linestyle='--', linewidth=2)

    # Plot fitted lines for each region
    line_colors = ['darkgreen', 'darkorange', 'purple']
    bounds = [-np.inf] + (fit.breakpoints or []) + [np.inf]

    for i, region in enumerate(fit.params.regions):
        Q_eff = region.Q if region.Q is not None else fit.params.Q
        PW_region = compute_wilshire_parameter(tr, T_kelvin, Q_eff)

        # Get PW range for this region's data
        region_mask = (PW_global >= bounds[i]) & (PW_global < bounds[i + 1])
        if np.any(region_mask):
            PW_min_r = np.min(PW_region[region_mask])
            PW_max_r = np.max(PW_region[region_mask])
        else:
            PW_min_r = np.min(PW_region)
            PW_max_r = np.max(PW_region)

        margin = 0.1 * (PW_max_r - PW_min_r) if PW_max_r > PW_min_r else 1.0
        x_range = np.linspace(PW_min_r - margin, PW_max_r + margin, 100)

        y_fit = np.log(region.k) + region.u * x_range
        color = line_colors[i % len(line_colors)]

        if has_per_region_q and region.Q is not None:
            label = f'R{i+1}: k={region.k:.3f}, u={region.u:.4f}, Q={region.Q/1000:.0f} kJ/mol'
        else:
            label = f'k={region.k:.3f}, u={region.u:.4f}'
        ax.plot(x_range, y_fit, '-', color=color, linewidth=2, label=label)

    ax.set_xlabel('PW = ln(tr) - Q/(RT)', fontsize=12)
    ax.set_ylabel('ln(-ln(σ/σ_TS))', fontsize=12)

    if has_per_region_q:
        ax.set_title('Wilshire Diagnostic Plot (per-region Q)', fontsize=14)
    else:
        ax.set_title(f'Wilshire Diagnostic Plot (Q = {fit.params.Q/1000:.0f} kJ/mol)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Diagnostic plot saved to: {output_file}", file=sys.stderr)

    if show and not output_file:
        plt.show()


def show_all_plots() -> None:
    """Display all created plots at once."""
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except ImportError:
        pass


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'Fit creep rupture equations to experimental data. {get_version_string()}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.txt                          # Basic LM fit (order 1)
  %(prog)s data.txt -n 1 --fix-c 20          # LM order 1 with fixed C=20
  %(prog)s data.txt --predict-tr 100000      # Fit with predictions for 100000 h
  %(prog)s data.txt --no-confidence          # Fit without CI
  %(prog)s data.txt --bootstrap 1000         # Fit with 95%% CI (bootstrap)

  # Wilshire model
  %(prog)s data.txt --model wsh --tensile-data tensile.csv
  %(prog)s data.txt --model wsh --tensile-data tensile.csv --fix-q 198000
  %(prog)s data.txt --model wsh --tensile-data tensile.csv --wsh-regions=-23
        """
    )

    # Common arguments
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('input_file', help='Input data file (sigma, T, tr)')
    parser.add_argument('--model', type=str, default='lm', choices=['lm', 'wsh'],
                        help='Creep model to use (default: lm)')
    parser.add_argument('--predict-tr', type=float, metavar='TIME',
                        help='Predict sigma for given time to rupture [h]')
    parser.add_argument('--predict-temps', type=str, default='500,550,600,650',
                        help='Temperatures for prediction [C], comma-separated')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--plot-output', type=str, metavar='FILE',
                        help='Save plot to file (PNG/PDF)')
    parser.add_argument('--output-format', type=str, default='text',
                        choices=['text', 'json', 'csv'],
                        help='Output format (default: text)')
    parser.add_argument('--delimiter', type=str,
                        help='Explicit delimiter (otherwise auto-detect)')
    parser.add_argument('--temp-unit', type=str, default='C', choices=['K', 'C'],
                        help='Temperature unit in input data (default: C)')

    # Confidence intervals (shared for both models)
    ci_group = parser.add_argument_group('Confidence intervals')
    ci_group.add_argument('--no-confidence', action='store_true',
                          help='Disable confidence intervals')
    ci_group.add_argument('--bootstrap', type=int, metavar='N', nargs='?', const=200, default=None,
                          help='Use bootstrap CI (default for WSH; optional N iterations, default 200)')
    ci_group.add_argument('--seed', type=int, metavar='SEED',
                          help='Random seed for reproducibility')

    # Larson-Miller specific
    lm_group = parser.add_argument_group('Larson-Miller options')
    lm_group.add_argument('-n', '--order', type=int, default=1, choices=[1, 2],
                          help='Order of LM equation (1 or 2), default=1')
    lm_group.add_argument('--fix-c', type=float, metavar='VALUE',
                          help='Fix constant C to specified value')

    # Wilshire specific
    wsh_group = parser.add_argument_group('Wilshire options')
    wsh_group.add_argument('--tensile-data', type=str, metavar='FILE',
                           help='File with tensile strength data (T, sigma_TS)')
    wsh_group.add_argument('--wsh-regions', type=str, metavar='BOUNDS',
                           help='PW breakpoints for multi-region fit (e.g., "-23" or "-25,-20")')
    wsh_group.add_argument('--fix-q', type=float, metavar='VALUE',
                           help='Fix activation energy Q [J/mol]')
    wsh_group.add_argument('--per-region-q', action='store_true',
                           help='Estimate separate Q for each region (default: single Q)')
    wsh_group.add_argument('--wsh-diagnostic', type=str, metavar='FILE', nargs='?', const='',
                           help='Generate Wilshire diagnostic plot (optionally save to FILE)')

    args = parser.parse_args()

    # Parse and validate input
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        sigma, T_kelvin, tr = parse_input_file(
            args.input_file,
            delimiter=args.delimiter,
            temp_unit=args.temp_unit
        )
    except Exception as e:
        print(f"Error parsing input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse prediction temperatures
    predict_temps_c = None
    try:
        predict_temps_c = [float(t.strip()) for t in args.predict_temps.split(',')]
    except ValueError as e:
        print(f"Error: Invalid temperature format in --predict-temps: {e}", file=sys.stderr)
        sys.exit(1)

    # Model-specific processing
    if args.model == 'lm':
        # Larson-Miller fit
        try:
            fit = fit_larson_miller(sigma, T_kelvin, tr, order=args.order, fix_c=args.fix_c)
        except Exception as e:
            print(f"Error during fitting: {e}", file=sys.stderr)
            sys.exit(1)

        # Confidence intervals
        ci = None
        if not args.no_confidence:
            if args.bootstrap is not None:
                ci = bootstrap_confidence_intervals(
                    sigma, T_kelvin, tr, order=args.order, fix_c=args.fix_c,
                    n_bootstrap=args.bootstrap, seed=args.seed
                )
                if ci is None:
                    print("Warning: Bootstrap CI failed (not enough successful fits)", file=sys.stderr)
            else:
                # Asymptotic CI: fast and accurate for LM's 3-4 parameters
                ci = compute_confidence_intervals(fit, sigma, T_kelvin, tr)
                if ci is None:
                    print("Warning: Asymptotic CI failed (ill-conditioned matrix or insufficient DOF)", file=sys.stderr)

        # Predictions
        predictions = None
        if args.predict_tr:
            predictions = [
                Prediction(
                    T_celsius=T_c,
                    T_kelvin=T_c + 273.15,
                    tr_target=args.predict_tr,
                    sigma=predict_stress_for_tr(args.predict_tr, T_c + 273.15, fit.params)
                )
                for T_c in predict_temps_c
            ]

        # Output
        formatters = {
            'json': format_output_json_lm,
            'csv': format_output_csv_lm,
            'text': format_output_text_lm
        }
        print(formatters[args.output_format](fit, predictions, ci))

        # Plot
        if not args.no_plot or args.plot_output:
            predict_temps_k = [T_c + 273.15 for T_c in predict_temps_c] if args.predict_tr else None
            create_plot(
                sigma, T_kelvin, tr,
                lambda tr_val, T_k: predict_stress_for_tr(tr_val, T_k, fit.params),
                f'Larson-Miller Fit (order {fit.params.order})',
                predict_temps_k, args.plot_output)

    else:  # Wilshire model
        # Validate tensile data
        if not args.tensile_data:
            print("Error: --tensile-data is required for Wilshire model", file=sys.stderr)
            sys.exit(1)

        if not Path(args.tensile_data).exists():
            print(f"Error: Tensile data file not found: {args.tensile_data}", file=sys.stderr)
            sys.exit(1)

        try:
            tensile_data = TensileData.from_file(args.tensile_data, temp_unit=args.temp_unit)
        except Exception as e:
            print(f"Error parsing tensile data: {e}", file=sys.stderr)
            sys.exit(1)

        # Parse region boundaries
        region_boundaries = None
        if args.wsh_regions:
            try:
                region_boundaries = [float(b.strip()) for b in args.wsh_regions.split(',')]
            except ValueError as e:
                print(f"Error: Invalid --wsh-regions format: {e}", file=sys.stderr)
                sys.exit(1)

        # Wilshire fit
        try:
            fit = fit_wilshire(
                sigma, T_kelvin, tr,
                tensile_data=tensile_data,
                region_boundaries=region_boundaries,
                fix_Q=args.fix_q,
                per_region_Q=args.per_region_q
            )
        except Exception as e:
            print(f"Error during fitting: {e}", file=sys.stderr)
            sys.exit(1)

        # Confidence intervals (WSH always uses bootstrap)
        ci = None
        if not args.no_confidence:
            n_boot = args.bootstrap if args.bootstrap is not None else 200
            ci = bootstrap_confidence_intervals_wsh(
                sigma, T_kelvin, tr,
                tensile_data=tensile_data,
                region_boundaries=region_boundaries,
                fix_Q=args.fix_q,
                per_region_Q=args.per_region_q,
                n_bootstrap=n_boot,
                seed=args.seed
            )
            if ci is None:
                print("Warning: Bootstrap CI failed (not enough successful fits)", file=sys.stderr)

        # Predictions
        predictions = None
        if args.predict_tr:
            predictions = [
                Prediction(
                    T_celsius=T_c,
                    T_kelvin=T_c + 273.15,
                    tr_target=args.predict_tr,
                    sigma=predict_stress_wsh(args.predict_tr, T_c + 273.15, fit.params, tensile_data)
                )
                for T_c in predict_temps_c
            ]

        # Output
        formatters = {
            'json': format_output_json_wsh,
            'csv': format_output_csv_wsh,
            'text': format_output_text_wsh
        }
        print(formatters[args.output_format](fit, predictions, ci))

        # Determine if we need to show multiple plots
        show_diagnostic = args.wsh_diagnostic is not None
        show_main = not args.no_plot or args.plot_output
        show_multiple = show_diagnostic and show_main and not args.wsh_diagnostic and not args.plot_output

        # Diagnostic plot
        if show_diagnostic:
            diag_file = args.wsh_diagnostic if args.wsh_diagnostic else None
            create_diagnostic_plot_wsh(sigma, T_kelvin, tr, fit, tensile_data,
                                       diag_file, show=not show_multiple)

        # Main plot
        if show_main:
            predict_temps_k = [T_c + 273.15 for T_c in predict_temps_c] if args.predict_tr else None
            n_reg = fit.params.n_regions
            create_plot(
                sigma, T_kelvin, tr,
                lambda tr_val, T_k: predict_stress_wsh(tr_val, T_k, fit.params, tensile_data),
                f'Wilshire Fit ({n_reg} region{"s" if n_reg > 1 else ""})',
                predict_temps_k, args.plot_output, show=not show_multiple)

        # Show all plots at once if multiple interactive plots
        if show_multiple:
            show_all_plots()


if __name__ == '__main__':
    main()
