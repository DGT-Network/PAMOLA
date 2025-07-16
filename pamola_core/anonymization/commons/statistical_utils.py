"""
PAMOLA.CORE - Privacy-Aware Machine Learning Analytics
Statistical Utilities for Noise Operations
=========================================

Package:       pamola_core.anonymization.commons
Module:        statistical_utils.py
Version:       1.0.1
Last Updated:  2025-06-16
License:       BSD 3-Clause
Status:        Stable

Description:
    This module provides statistical utilities specifically designed for noise
    operations in the PAMOLA anonymization framework. It extends the general
    statistical metrics with noise-specific analysis functions.

    The module focuses on:
    - Noise quality metrics (SNR, distribution analysis)
    - Utility preservation metrics
    - Temporal pattern analysis for datetime noise
    - Statistical property preservation

Dependencies:
    - pamola_core.utils.statistical_metrics: Base statistical functions
    - numpy: Numerical computations
    - pandas: Data manipulation
    - scipy: Statistical tests

Design Philosophy:
    - Reuse existing statistical functions from pamola_core.utils.statistical_metrics
    - Add only noise-specific functionality
    - Maintain consistency with PAMOLA framework patterns
    - Optimize for large-scale data processing

Authors:
    PAMOLA Core Team

Changelog:
    1.0.1 (2025-06-16):
        - Fixed type annotations for numpy operations
        - Added parameter validation
        - Added custom exception class
        - Improved type safety with explicit float conversions
    1.0.0 (2025-06-16):
        - Initial implementation
        - Added noise quality metrics
        - Added utility preservation functions
        - Added temporal analysis utilities

TODO:
    - Add calculate_utility_metrics() as per REQ-UNIFORM-007
    - Add calculate_correlation_preservation() for multi-field analysis
    - Add estimate_information_loss() for privacy-utility tradeoff
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats

# Import base statistical functions from core utils
from pamola_core.utils.statistical_metrics import (
    calculate_gini_coefficient,
    calculate_shannon_entropy,
    get_distribution_summary,
    EPSILON
)

# Configure module logger
logger = logging.getLogger(__name__)


# Custom exceptions
class StatisticalUtilsError(Exception):
    """Base exception for statistical utilities."""
    pass


class InvalidParameterError(StatisticalUtilsError):
    """Raised when invalid parameters are provided."""
    pass


def calculate_signal_to_noise_ratio(
        original: Union[pd.Series, np.ndarray],
        noisy: Union[pd.Series, np.ndarray],
        method: str = "standard"
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) for noisy data.

    SNR measures the level of desired signal relative to the level of noise.
    Higher values indicate better signal preservation.

    Parameters:
    -----------
    original : Union[pd.Series, np.ndarray]
        Original signal values
    noisy : Union[pd.Series, np.ndarray]
        Noisy signal values
    method : str, optional
        Calculation method:
        - "standard": 20 * log10(signal_std / noise_std) in dB
        - "power": 10 * log10(signal_power / noise_power) in dB
        - "ratio": Simple ratio signal_std / noise_std

    Returns:
    --------
    float
        SNR value (in dB for standard/power methods)

    Raises:
    -------
    InvalidParameterError
        If method is unknown

    Examples:
    ---------
    >>> original = pd.Series([100, 102, 98, 101, 99])
    >>> noisy = pd.Series([101, 100, 97, 103, 98])
    >>> snr = calculate_signal_to_noise_ratio(original, noisy)
    >>> print(f"SNR: {snr:.2f} dB")
    """
    # Convert to numpy arrays with explicit float dtype
    orig = np.asarray(original, dtype=float)
    noised = np.asarray(noisy, dtype=float)

    # Calculate noise
    noise = noised - orig

    # Remove NaN values
    mask = ~(np.isnan(orig) | np.isnan(noised))
    signal = orig[mask]
    noise_clean = noise[mask]

    if signal.size == 0:
        logger.warning("No valid values for SNR calculation")
        return 0.0

    if method == "standard":
        # Convert numpy types to Python float
        s_std: float = float(np.std(signal))
        n_std: float = float(np.std(noise_clean))

        if n_std < EPSILON:
            return float('inf')  # No noise

        return 20.0 * math.log10(s_std / n_std)

    elif method == "power":
        # Convert numpy types to Python float
        s_pow: float = float(np.mean(signal ** 2))
        n_pow: float = float(np.mean(noise_clean ** 2))

        if n_pow < EPSILON:
            return float('inf')  # No noise

        return 10.0 * math.log10(s_pow / n_pow)

    elif method == "ratio":
        # Convert numpy types to Python float
        s_std: float = float(np.std(signal))
        n_std: float = float(np.std(noise_clean))

        if n_std < EPSILON:
            return float('inf')  # No noise

        return s_std / n_std

    else:
        raise InvalidParameterError(f"Unknown method: {method}. Use 'standard', 'power', or 'ratio'")


def analyze_noise_uniformity(
        noise_values: Union[pd.Series, np.ndarray],
        expected_min: float,
        expected_max: float,
        n_bins: int = 20
) -> Dict[str, Any]:
    """
    Analyze whether noise follows expected uniform distribution.

    Parameters:
    -----------
    noise_values : Union[pd.Series, np.ndarray]
        Actual noise values (difference between noisy and original)
    expected_min : float
        Expected minimum noise value
    expected_max : float
        Expected maximum noise value
    n_bins : int, optional
        Number of bins for chi-square test (default: 20)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - uniformity_test: Chi-square and KS test results
        - actual_range: Actual min/max values
        - distribution_metrics: Skewness, kurtosis, etc.

    Raises:
    -------
    InvalidParameterError
        If expected_min >= expected_max or no valid noise values

    Examples:
    ---------
    >>> noise = np.random.uniform(-5, 5, 1000)
    >>> analysis = analyze_noise_uniformity(noise, -5, 5)
    >>> print(f"Uniformity p-value: {analysis['uniformity_test']['chi2_p_value']:.4f}")
    """
    # Validate parameters
    if expected_min >= expected_max:
        raise InvalidParameterError(f"expected_min ({expected_min}) must be less than expected_max ({expected_max})")

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise InvalidParameterError("n_bins must be a positive integer")

    # Convert to numpy array with float dtype
    values = np.asarray(noise_values, dtype=float)

    # Remove NaN values
    noise_clean = values[~np.isnan(values)]

    if noise_clean.size == 0:
        raise InvalidParameterError("No valid noise values provided")

    # Chi-square test for uniformity
    hist, _ = np.histogram(noise_clean, bins=n_bins, range=(expected_min, expected_max))
    # Calculate expected frequency per bin assuming uniform distribution.
    # Note: we use hist.sum() instead of noise_clean.size to ensure we only count
    # values that actually fall within the specified [expected_min, expected_max] range.
    # Some values may fall outside this range and get excluded by np.histogram,
    # so hist.sum() gives the correct total count considered in the histogram.
    expected_freq = hist.sum() / n_bins
    expected = np.full(n_bins, expected_freq)

    # Perform chi-square test
    chi2_stat, p_val_chi2 = stats.chisquare(hist, expected)

    # Kolmogorov-Smirnov test as alternative
    ks_stat, p_val_ks = stats.kstest(
        noise_clean,
        'uniform',
        args=(expected_min, expected_max - expected_min)
    )

    # Calculate actual range
    actual_min: float = float(noise_clean.min())
    actual_max: float = float(noise_clean.max())

    # Distribution metrics with explicit float conversion
    distribution_metrics = {
        "mean": float(np.mean(noise_clean)),
        "std": float(np.std(noise_clean)),
        "skewness": float(stats.skew(noise_clean)),
        "kurtosis": float(stats.kurtosis(noise_clean)),
        "expected_mean": (expected_min + expected_max) / 2.0,
        "expected_std": (expected_max - expected_min) / math.sqrt(12.0)  # Uniform std
    }

    results = {
        "uniformity_test": {
            "chi2_statistic": float(chi2_stat),
            "chi2_p_value": float(p_val_chi2),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(p_val_ks),
            "is_uniform_chi2": p_val_chi2 > 0.05,
            "is_uniform_ks": p_val_ks > 0.05
        },
        "actual_range": {
            "min": actual_min,
            "max": actual_max,
            "expected_min": expected_min,
            "expected_max": expected_max,
            "within_bounds": (actual_min >= expected_min) and (actual_max <= expected_max)
        },
        "distribution_metrics": distribution_metrics
    }

    return results


def calculate_utility_preservation(
        original: pd.Series,
        noisy: pd.Series,
        metrics: List[str] = None
) -> Dict[str, float]:
    """
    Calculate utility preservation metrics after noise addition.

    Measures how well statistical properties are preserved after adding noise.

    Parameters:
    -----------
    original : pd.Series
        Original data
    noisy : pd.Series
        Data after noise addition
    metrics : List[str], optional
        Specific metrics to calculate. Default: all available

    Returns:
    --------
    Dict[str, float]
        Preservation metrics (values closer to 1 indicate better preservation)

    Examples:
    ---------
    >>> original = pd.Series([1, 2, 3, 4, 5])
    >>> noisy = pd.Series([1.1, 1.9, 3.2, 3.8, 5.1])
    >>> preservation = calculate_utility_preservation(original, noisy)
    >>> print(f"Mean preservation: {preservation['mean_preservation']:.4f}")
    """
    if metrics is None:
        metrics = ["mean", "std", "median", "iqr", "correlation", "rank_correlation"]

    results = {}

    # Remove NaN values for paired analysis
    mask = ~(original.isna() | noisy.isna())
    orig_clean = original[mask]
    noisy_clean = noisy[mask]

    if len(orig_clean) == 0:
        return {m + "_preservation": np.nan for m in metrics}

    # Mean preservation
    if "mean" in metrics:
        orig_mean = orig_clean.mean()
        if abs(orig_mean) > EPSILON:
            results["mean_preservation"] = 1 - abs(orig_mean - noisy_clean.mean()) / abs(orig_mean)
        else:
            results["mean_preservation"] = 1.0 if abs(noisy_clean.mean()) < EPSILON else 0.0

    # Standard deviation preservation
    if "std" in metrics:
        orig_std = orig_clean.std()
        noisy_std = noisy_clean.std()
        if orig_std > EPSILON:
            results["std_preservation"] = min(noisy_std / orig_std, orig_std / noisy_std)
        else:
            results["std_preservation"] = 1.0 if noisy_std < EPSILON else 0.0

    # Median preservation
    if "median" in metrics:
        orig_median = orig_clean.median()
        if abs(orig_median) > EPSILON:
            results["median_preservation"] = 1 - abs(orig_median - noisy_clean.median()) / abs(orig_median)
        else:
            results["median_preservation"] = 1.0 if abs(noisy_clean.median()) < EPSILON else 0.0

    # Interquartile range preservation
    if "iqr" in metrics:
        orig_iqr = orig_clean.quantile(0.75) - orig_clean.quantile(0.25)
        noisy_iqr = noisy_clean.quantile(0.75) - noisy_clean.quantile(0.25)
        if orig_iqr > EPSILON:
            results["iqr_preservation"] = min(noisy_iqr / orig_iqr, orig_iqr / noisy_iqr)
        else:
            results["iqr_preservation"] = 1.0 if noisy_iqr < EPSILON else 0.0

    # Pearson correlation
    if "correlation" in metrics:
        if len(orig_clean) > 1:
            corr = orig_clean.corr(noisy_clean)
            results["correlation"] = corr if not np.isnan(corr) else 0.0
        else:
            results["correlation"] = np.nan

    # Spearman rank correlation
    if "rank_correlation" in metrics:
        if len(orig_clean) > 1:
            rank_corr = orig_clean.corr(noisy_clean, method='spearman')
            results["rank_correlation"] = rank_corr if not np.isnan(rank_corr) else 0.0
        else:
            results["rank_correlation"] = np.nan

    return results


def analyze_temporal_noise_impact(
        original_timestamps: pd.Series,
        noisy_timestamps: pd.Series
) -> Dict[str, Any]:
    """
    Analyze the impact of noise on temporal data.

    Specific analysis for datetime fields after temporal noise addition.

    Parameters:
    -----------
    original_timestamps : pd.Series
        Original datetime values
    noisy_timestamps : pd.Series
        Datetime values after noise addition

    Returns:
    --------
    Dict[str, Any]
        Temporal impact metrics

    Examples:
    ---------
    >>> original = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03'])
    >>> noisy = pd.to_datetime(['2025-01-01 12:00', '2025-01-02 08:00', '2025-01-02 20:00'])
    >>> impact = analyze_temporal_noise_impact(pd.Series(original), pd.Series(noisy))
    """
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(original_timestamps):
        original_timestamps = pd.to_datetime(original_timestamps)
    if not pd.api.types.is_datetime64_any_dtype(noisy_timestamps):
        noisy_timestamps = pd.to_datetime(noisy_timestamps)

    # Calculate time shifts
    shifts = noisy_timestamps - original_timestamps
    shifts_seconds = shifts.dt.total_seconds()

    # Remove NaN values
    valid_mask = ~(shifts_seconds.isna())
    shifts_clean = shifts_seconds[valid_mask]

    if len(shifts_clean) == 0:
        return {
            "shift_statistics": {},
            "pattern_preservation": {},
            "ordering_preservation": {}
        }

    # Shift statistics
    shift_stats = {
        "mean_shift_hours": float(shifts_clean.mean() / 3600),
        "std_shift_hours": float(shifts_clean.std() / 3600),
        "min_shift_hours": float(shifts_clean.min() / 3600),
        "max_shift_hours": float(shifts_clean.max() / 3600),
        "median_shift_hours": float(shifts_clean.median() / 3600),
        "zero_shifts": int((shifts_clean == 0).sum()),
        "forward_shifts": int((shifts_clean > 0).sum()),
        "backward_shifts": int((shifts_clean < 0).sum())
    }

    # Pattern preservation
    pattern_preservation = {}

    # Day of week preservation
    original_dow = original_timestamps[valid_mask].dt.dayofweek
    noisy_dow = noisy_timestamps[valid_mask].dt.dayofweek
    pattern_preservation["weekday_preserved"] = float((original_dow == noisy_dow).mean())

    # Hour of day preservation
    original_hour = original_timestamps[valid_mask].dt.hour
    noisy_hour = noisy_timestamps[valid_mask].dt.hour
    pattern_preservation["hour_preserved"] = float((original_hour == noisy_hour).mean())

    # Weekend preservation
    original_weekend = original_dow.isin([5, 6])
    noisy_weekend = noisy_dow.isin([5, 6])
    pattern_preservation["weekend_preserved"] = float((original_weekend == noisy_weekend).mean())

    # Ordering preservation
    if len(original_timestamps) > 1:
        # Check if temporal ordering is preserved
        original_order = original_timestamps.argsort()
        noisy_order = noisy_timestamps.argsort()
        ordering_preserved = (original_order == noisy_order).all()

        # Kendall's tau for ordering correlation
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(original_order, noisy_order)

        ordering_preservation = {
            "order_fully_preserved": bool(ordering_preserved),
            "kendall_tau": float(tau),
            "kendall_p_value": float(p_value),
            "inversions": int(((noisy_order != original_order).sum()) / 2)
        }
    else:
        ordering_preservation = {
            "order_fully_preserved": True,
            "kendall_tau": 1.0,
            "kendall_p_value": 1.0,
            "inversions": 0
        }

    return {
        "shift_statistics": shift_stats,
        "pattern_preservation": pattern_preservation,
        "ordering_preservation": ordering_preservation
    }


def calculate_noise_distribution_fit(
        noise_values: Union[pd.Series, np.ndarray],
        distribution: str = "uniform",
        params: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Test how well noise fits expected distribution.

    Parameters:
    -----------
    noise_values : Union[pd.Series, np.ndarray]
        Actual noise values
    distribution : str
        Expected distribution: "uniform", "normal", "laplace"
    params : Dict[str, float]
        Distribution parameters (e.g., {"loc": 0, "scale": 1})

    Returns:
    --------
    Dict[str, Any]
        Goodness-of-fit test results
    """
    # Convert to numpy array
    if isinstance(noise_values, pd.Series):
        noise_values = noise_values.values

    # Remove NaN values
    noise_clean = noise_values[~np.isnan(noise_values)]

    if len(noise_clean) == 0:
        return {"error": "No valid noise values"}

    # Default parameters
    if params is None:
        if distribution == "uniform":
            params = {"loc": noise_clean.min(), "scale": noise_clean.max() - noise_clean.min()}
        elif distribution == "normal":
            params = {"loc": noise_clean.mean(), "scale": noise_clean.std()}
        elif distribution == "laplace":
            params = {"loc": noise_clean.mean(), "scale": noise_clean.std() / np.sqrt(2)}

    # Perform goodness-of-fit tests
    if distribution == "uniform":
        ks_stat, ks_pvalue = stats.kstest(noise_clean, 'uniform', args=(params["loc"], params["scale"]))
    elif distribution == "normal":
        ks_stat, ks_pvalue = stats.kstest(noise_clean, 'norm', args=(params["loc"], params["scale"]))
    elif distribution == "laplace":
        ks_stat, ks_pvalue = stats.kstest(noise_clean, 'laplace', args=(params["loc"], params["scale"]))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # Additional tests
    anderson_result = None
    if distribution in ["normal", "uniform"]:
        try:
            anderson_result = stats.anderson(noise_clean, dist=distribution.replace("normal", "norm"))
        except:
            pass

    results = {
        "distribution": distribution,
        "parameters": params,
        "ks_test": {
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "reject_null": ks_pvalue < 0.05
        },
        "sample_size": len(noise_clean)
    }

    if anderson_result:
        results["anderson_test"] = {
            "statistic": float(anderson_result.statistic),
            "critical_values": anderson_result.critical_values.tolist(),
            "significance_levels": anderson_result.significance_level.tolist()
        }

    return results


def calculate_multifield_noise_correlation(
        noise_dict: Dict[str, Union[pd.Series, np.ndarray]]
) -> pd.DataFrame:
    """
    Calculate correlation between noise added to different fields.

    Important for ensuring independence of noise across fields.

    Parameters:
    -----------
    noise_dict : Dict[str, Union[pd.Series, np.ndarray]]
        Dictionary mapping field names to noise values

    Returns:
    --------
    pd.DataFrame
        Correlation matrix between noise fields

    Examples:
    ---------
    >>> noise_dict = {
    ...     "age": np.random.uniform(-5, 5, 100),
    ...     "salary": np.random.uniform(-1000, 1000, 100),
    ...     "score": np.random.uniform(-0.1, 0.1, 100)
    ... }
    >>> corr = calculate_multifield_noise_correlation(noise_dict)
    """
    # Convert all to Series
    noise_df = pd.DataFrame()
    for field, noise_values in noise_dict.items():
        if isinstance(noise_values, np.ndarray):
            noise_df[field] = pd.Series(noise_values)
        else:
            noise_df[field] = noise_values

    # Calculate correlation matrix
    return noise_df.corr()


def get_noise_quality_summary(
        original: pd.Series,
        noisy: pd.Series,
        expected_noise_range: Tuple[float, float] = None,
        noise_type: str = "uniform"
) -> Dict[str, Any]:
    """
    Get comprehensive noise quality summary.

    Combines multiple noise quality metrics into a single report.

    Parameters:
    -----------
    original : pd.Series
        Original data
    noisy : pd.Series
        Data after noise addition
    expected_noise_range : Tuple[float, float], optional
        Expected (min, max) for uniform noise
    noise_type : str
        Type of noise: "uniform", "normal", "laplace"

    Returns:
    --------
    Dict[str, Any]
        Comprehensive noise quality report
    """
    # Calculate actual noise
    noise = noisy - original

    # Basic metrics
    summary = {
        "noise_type": noise_type,
        "total_records": len(original),
        "valid_records": (~noise.isna()).sum()
    }

    # SNR metrics
    summary["signal_to_noise"] = {
        "snr_db": calculate_signal_to_noise_ratio(original, noisy, method="standard"),
        "snr_ratio": calculate_signal_to_noise_ratio(original, noisy, method="ratio")
    }

    # Utility preservation
    summary["utility_preservation"] = calculate_utility_preservation(original, noisy)

    # Distribution analysis
    if noise_type == "uniform" and expected_noise_range:
        summary["uniformity_analysis"] = analyze_noise_uniformity(
            noise, expected_noise_range[0], expected_noise_range[1]
        )

    # Distribution fit
    summary["distribution_fit"] = calculate_noise_distribution_fit(noise, noise_type)

    # Use base statistical metrics
    summary["noise_distribution"] = get_distribution_summary(
        noise,
        include_gini=True,
        include_concentration=False,
        include_entropy=True
    )

    return summary


# TODO: Functions to be implemented as per REQ-UNIFORM-007
def calculate_utility_metrics(
        original_data: pd.Series,
        transformed_data: pd.Series,
        metric_set: str = "standard"
) -> Dict[str, float]:
    """
    Calculate utility preservation metrics (placeholder for future implementation).

    This function will provide comprehensive utility metrics as specified
    in the Noise Sub-SRS requirements.

    Parameters:
    -----------
    original_data : pd.Series
        Original data before transformation
    transformed_data : pd.Series
        Data after transformation
    metric_set : str
        Set of metrics to calculate: "minimal", "standard", "detailed"

    Returns:
    --------
    Dict[str, float]
        Utility metrics
    """
    # TODO: Implement as per specification
    raise NotImplementedError("Function will be implemented as part of REQ-UNIFORM-007")


def calculate_correlation_preservation(
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        fields: List[str]
) -> Dict[str, float]:
    """
    Calculate correlation preservation between multiple fields (placeholder).

    This function will analyze how well correlations between fields
    are preserved after noise addition.

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataframe
    transformed_df : pd.DataFrame
        Transformed dataframe
    fields : List[str]
        Fields to analyze

    Returns:
    --------
    Dict[str, float]
        Correlation preservation metrics
    """
    # TODO: Implement multi-field correlation analysis
    raise NotImplementedError("Function will be implemented for multi-field noise analysis")


def estimate_information_loss(
        original_data: pd.Series,
        noisy_data: pd.Series,
        epsilon: float = None
) -> Dict[str, float]:
    """
    Estimate information loss due to noise (placeholder).

    This function will quantify the privacy-utility tradeoff
    by estimating information loss.

    Parameters:
    -----------
    original_data : pd.Series
        Original data
    noisy_data : pd.Series
        Noisy data
    epsilon : float, optional
        Privacy parameter for differential privacy

    Returns:
    --------
    Dict[str, float]
        Information loss metrics
    """
    # TODO: Implement information-theoretic metrics
    raise NotImplementedError("Function will be implemented for privacy-utility analysis")