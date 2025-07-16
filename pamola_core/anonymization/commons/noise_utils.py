"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Noise Operation Utilities
Package:       pamola_core.anonymization.commons
Version:       1.0.0
Status:        development
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
   Utility functions and classes for noise addition operations in PAMOLA.CORE.
   This module provides comprehensive support for secure random generation,
   noise impact analysis, distribution preservation metrics, and practical
   noise parameter recommendations.

   The module supports both cryptographically secure and standard random
   generation, making it suitable for both privacy-critical applications
   and performance-optimized scenarios.

Key Features:
   - Thread-safe secure random number generation with multiple distributions
   - Comprehensive noise impact and utility metrics calculation
   - Distribution preservation analysis with statistical tests
   - Intelligent noise range recommendations based on signal-to-noise ratio
   - Validation utilities for noise bounds and parameters
   - Support for numeric and temporal noise generation

Dependencies:
   - numpy: Numerical operations and random generation
   - pandas: Series and DataFrame operations
   - scipy: Statistical tests and distributions
   - secrets: Cryptographically secure random generation
   - threading: Thread safety for random generators
   - typing: Type hints and annotations

Usage Example:
   >>> from pamola_core.anonymization.commons.noise_utils import (
   ...     SecureRandomGenerator, calculate_noise_impact, suggest_noise_range
   ... )
   >>>
   >>> # Create secure random generator
   >>> rng = SecureRandomGenerator(use_secure=True)
   >>> noise = rng.uniform(-10, 10, size=1000)
   >>>
   >>> # Analyze noise impact
   >>> original = pd.Series(np.random.randn(1000))
   >>> noisy = original + noise
   >>> impact = calculate_noise_impact(original, noisy)
   >>> print(f"RMSE: {impact['rmse']:.3f}")
   >>>
   >>> # Get noise range recommendation
   >>> suggested_range = suggest_noise_range(original, target_snr=10.0)
   >>> print(f"Suggested range: ±{suggested_range:.3f}")

Performance Considerations:
   - SecureRandomGenerator using secrets module is slower than NumPy
   - For large-scale operations, consider use_secure=False with proper seeding
   - Distribution calculations cache results where possible
   - Statistical tests use sampling for very large datasets

Security Notes:
   - Use use_secure=True for privacy-critical applications
   - Standard NumPy random is suitable for non-sensitive applications
   - Always validate noise bounds to prevent information leakage
   - Consider differential privacy extensions for formal guarantees

Future Extensions:
   - Differential privacy mechanisms (Laplace, Gaussian)
   - Adaptive noise based on local sensitivity
   - Multi-dimensional correlated noise
   - Query-specific noise calibration
"""

import logging
import secrets
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Import validation exceptions with correct signatures
from pamola_core.anonymization.commons.validation.exceptions import (
    InvalidParameterError
)

# Configure module logger
logger = logging.getLogger(__name__)

# Constants for noise operations
NOISE_DISTRIBUTIONS = ['uniform', 'normal', 'laplace', 'exponential']
TEMPORAL_UNITS = ['seconds', 'minutes', 'hours', 'days', 'weeks']
DEFAULT_HISTOGRAM_BINS = 20
MAX_SAMPLE_SIZE = 10000  # For performance in statistical tests

# Time conversion constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
SECONDS_PER_WEEK = 604800

# Statistical constants
EPSILON = 1e-10  # Small value to avoid division by zero
PI = np.pi
SQRT_2 = np.sqrt(2)
SQRT_12 = np.sqrt(12)  # For uniform distribution variance

# Distribution parameters
UNIFORM_RANGE_DEFAULT = 1.0
NORMAL_LOC_DEFAULT = 0.0
NORMAL_SCALE_DEFAULT = 1.0
LAPLACE_LOC_DEFAULT = 0.0
LAPLACE_SCALE_DEFAULT = 1.0
EXPONENTIAL_SCALE_DEFAULT = 1.0

# Box-Muller constants
BOX_MULLER_CONST = -2.0
BOX_MULLER_ANGLE = 2.0

# Percentile defaults
DEFAULT_PERCENTILES = [10, 25, 50, 75, 90]
PERCENTILE_CHECK_DEFAULT = 99.0
PERCENTILE_LOWER_CHECK = 0.01
PERCENTILE_UPPER_CHECK = 0.99

# Threshold constants
SKEWNESS_THRESHOLD = 1.0
VIOLATION_THRESHOLD_PCT = 5.0
NOISE_RATIO_WARNING = 0.5
NOISE_RATIO_EXCESSIVE = 1.0
UTILITY_RATIO_THRESHOLD = 0.3
CORRELATION_LOW_THRESHOLD = 0.7
DISTRIBUTION_PVALUE_THRESHOLD = 0.05
UTILITY_SCORE_LOW = 0.5

# SNR conversion
DB_CONVERSION_FACTOR = 10.0
DB_LOG_BASE = 10

# Default noise parameters
DEFAULT_TARGET_SNR = 10.0
MULTIPLICATIVE_DEFAULT_RANGE = 0.1  # 10% variation
NOISE_SCALE_DIVISOR = 3.0  # For non-uniform distributions


class SecureRandomGenerator:
    """
    Thread-safe random number generator supporting both secure and fast generation.

    This class provides a unified interface for random number generation that can
    switch between cryptographically secure (using secrets module) and standard
    NumPy random generation based on security requirements.

    Attributes:
        use_secure (bool): Whether to use cryptographically secure generation
        seed (Optional[int]): Random seed for reproducibility (ignored if use_secure=True)
        _lock (threading.Lock): Thread safety lock
        _rng: Internal random generator (secrets.SystemRandom or np.random.Generator)

    Example:
        >>> # Secure generation for privacy
        >>> secure_rng = SecureRandomGenerator(use_secure=True)
        >>> secure_values = secure_rng.uniform(0, 1, size=100)
        >>>
        >>> # Fast generation with seed for reproducibility
        >>> fast_rng = SecureRandomGenerator(use_secure=False, seed=42)
        >>> reproducible_values = fast_rng.uniform(0, 1, size=100)
    """

    def __init__(self, use_secure: bool = True, seed: Optional[int] = None):
        """
        Initialize random generator with specified security level.

        Args:
            use_secure: If True, use cryptographically secure random generation
            seed: Random seed for reproducibility (only used if use_secure=False)
        """
        self.use_secure = use_secure
        self.seed = seed
        self._lock = threading.Lock()

        if use_secure:
            self._rng = secrets.SystemRandom()
            if seed is not None:
                logger.warning("Seed ignored when use_secure=True")
        else:
            self._rng = np.random.default_rng(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0,
                size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
        """
        Generate uniform random values in [low, high).

        Args:
            low: Lower boundary (inclusive)
            high: Upper boundary (exclusive)
            size: Output shape. If None, returns scalar

        Returns:
            Random value(s) from uniform distribution

        Raises:
            InvalidParameterError: If low >= high
        """
        if low >= high:
            raise InvalidParameterError(
                param_name="low",
                param_value=low,
                reason=f"must be less than 'high' ({high})"
            )

        with self._lock:
            if isinstance(self._rng, secrets.SystemRandom):
                if size is None:
                    return self._rng.uniform(low, high)
                # Handle multi-dimensional sizes
                if isinstance(size, tuple):
                    total_size = np.prod(size)
                    values = np.array([
                        self._rng.uniform(low, high) for _ in range(total_size)
                    ])
                    return values.reshape(size)
                else:
                    return np.array([
                        self._rng.uniform(low, high) for _ in range(size)
                    ])
            else:
                return self._rng.uniform(low, high, size)

    def normal(self, loc: float = 0.0, scale: float = 1.0,
               size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
        """
        Generate normal (Gaussian) random values.

        Args:
            loc: Mean of the distribution
            scale: Standard deviation (must be positive)
            size: Output shape. If None, returns scalar

        Returns:
            Random value(s) from normal distribution

        Raises:
            InvalidParameterError: If scale <= 0
        """
        if scale <= 0:
            raise InvalidParameterError(
                param_name="scale",
                param_value=scale,
                reason="standard deviation must be positive"
            )

        with self._lock:
            if isinstance(self._rng, secrets.SystemRandom):
                if size is None:
                    # Box-Muller transform for single value
                    u1 = self._rng.random()
                    u2 = self._rng.random()
                    z0 = np.sqrt(BOX_MULLER_CONST * np.log(u1)) * np.cos(BOX_MULLER_ANGLE * PI * u2)
                    return loc + scale * z0

                # Generate for array
                if isinstance(size, tuple):
                    total_size = np.prod(size)
                else:
                    total_size = size

                # Box-Muller transform for efficiency
                n_pairs = (total_size + 1) // 2
                u1 = np.array([self._rng.random() for _ in range(n_pairs)])
                u2 = np.array([self._rng.random() for _ in range(n_pairs)])

                z0 = np.sqrt(BOX_MULLER_CONST * np.log(u1)) * np.cos(BOX_MULLER_ANGLE * PI * u2)
                z1 = np.sqrt(BOX_MULLER_CONST * np.log(u1)) * np.sin(BOX_MULLER_ANGLE * PI * u2)

                # Interleave z0 and z1
                values = np.empty(2 * n_pairs)
                values[0::2] = z0
                values[1::2] = z1

                # Trim to requested size and reshape
                values = values[:total_size] * scale + loc
                if isinstance(size, tuple):
                    return values.reshape(size)
                return values
            else:
                return self._rng.normal(loc, scale, size)

    def laplace(self, loc: float = 0.0, scale: float = 1.0,
                size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
        """
        Generate Laplace random values (for differential privacy).

        Args:
            loc: Location parameter (median)
            scale: Scale parameter (must be positive)
            size: Output shape. If None, returns scalar

        Returns:
            Random value(s) from Laplace distribution

        Raises:
            InvalidParameterError: If scale <= 0
        """
        if scale <= 0:
            raise InvalidParameterError(
                param_name="scale",
                param_value=scale,
                reason="must be positive for Laplace distribution"
            )

        with self._lock:
            if isinstance(self._rng, secrets.SystemRandom):
                if size is None:
                    # Inverse transform sampling
                    u = self._rng.uniform(-0.5, 0.5)
                    return loc - scale * np.sign(u) * np.log(1 - 2 * abs(u))

                # Generate for array
                if isinstance(size, tuple):
                    total_size = np.prod(size)
                else:
                    total_size = size

                u = np.array([
                    self._rng.uniform(-0.5, 0.5) for _ in range(total_size)
                ])
                values = loc - scale * np.sign(u) * np.log(1 - 2 * np.abs(u))

                if isinstance(size, tuple):
                    return values.reshape(size)
                return values
            else:
                return self._rng.laplace(loc, scale, size)

    def choice(self, a: Union[int, np.ndarray], size: Optional[int] = None,
               replace: bool = True, p: Optional[np.ndarray] = None) -> Union[Any, np.ndarray]:
        """
        Generate random sample from given array or range.

        Args:
            a: If int, sample from range(a). If array, sample from array
            size: Number of samples. If None, returns single value
            replace: Whether to sample with replacement
            p: Probabilities for each element (must sum to 1)

        Returns:
            Random sample(s) from the input
        """
        with self._lock:
            if isinstance(self._rng, secrets.SystemRandom):
                # Convert to list for easier handling
                if isinstance(a, int):
                    population = list(range(a))
                else:
                    population = list(a)

                if p is not None:
                    # Weighted sampling using cumulative probabilities
                    cumsum = np.cumsum(p)
                    if size is None:
                        r = self._rng.random()
                        idx = np.searchsorted(cumsum, r)
                        return population[idx]
                    else:
                        indices = [
                            np.searchsorted(cumsum, self._rng.random())
                            for _ in range(size)
                        ]
                        return np.array([population[i] for i in indices])
                else:
                    # Uniform sampling
                    if size is None:
                        return self._rng.choice(population)
                    else:
                        if replace:
                            return np.array([
                                self._rng.choice(population) for _ in range(size)
                            ])
                        else:
                            return np.array(self._rng.sample(population, size))
            else:
                return self._rng.choice(a, size, replace, p)


def calculate_noise_impact(original: pd.Series,
                           noisy: pd.Series,
                           sample_size: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate comprehensive metrics measuring the impact of noise on data utility.

    This function computes various error metrics and correlations to quantify
    how much the noise has affected the original data. It's useful for
    evaluating the privacy-utility tradeoff.

    Args:
        original: Original data series
        noisy: Data series after noise addition
        sample_size: If provided, sample data for faster computation on large datasets

    Returns:
        Dictionary containing:
            - rmse: Root Mean Square Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error
            - correlation: Pearson correlation coefficient
            - rank_correlation: Spearman rank correlation
            - relative_error: Mean relative error
            - max_absolute_error: Maximum absolute difference
            - snr: Signal-to-Noise Ratio (in dB)

    Example:
        >>> original = pd.Series([1, 2, 3, 4, 5])
        >>> noisy = pd.Series([1.1, 1.9, 3.2, 3.8, 5.1])
        >>> impact = calculate_noise_impact(original, noisy)
        >>> print(f"SNR: {impact['snr']:.1f} dB")
    """
    # Remove nulls for comparison
    mask = original.notna() & noisy.notna()
    orig_clean = original[mask]
    noisy_clean = noisy[mask]

    if len(orig_clean) == 0:
        logger.warning("No valid data points for noise impact calculation")
        return {}

    # Sample if dataset is large
    if sample_size and len(orig_clean) > sample_size:
        idx = np.random.choice(len(orig_clean), sample_size, replace=False)
        orig_clean = orig_clean.iloc[idx]
        noisy_clean = noisy_clean.iloc[idx]

    # Calculate noise
    noise = noisy_clean - orig_clean

    # Basic error metrics
    rmse = float(np.sqrt(np.mean(noise ** 2)))
    mae = float(np.mean(np.abs(noise)))

    # Relative error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs(noise) / (np.abs(orig_clean) + EPSILON)
        relative_error = float(np.mean(relative_errors[np.isfinite(relative_errors)]))

        # MAPE (Mean Absolute Percentage Error)
        percentage_errors = 100 * np.abs(noise) / (np.abs(orig_clean) + EPSILON)
        mape = float(np.mean(percentage_errors[np.isfinite(percentage_errors)]))

    # Correlation metrics
    correlation = float(orig_clean.corr(noisy_clean))
    rank_correlation = float(orig_clean.corr(noisy_clean, method='spearman'))

    # Maximum error
    max_absolute_error = float(np.max(np.abs(noise)))

    # Signal-to-Noise Ratio (in dB)
    signal_power = float(np.var(orig_clean))
    noise_power = float(np.var(noise))
    if noise_power > 0:
        snr_db = DB_CONVERSION_FACTOR * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "correlation": correlation,
        "rank_correlation": rank_correlation,
        "relative_error": relative_error,
        "max_absolute_error": max_absolute_error,
        "snr": snr_db
    }


def calculate_distribution_preservation(original: pd.Series,
                                        noisy: pd.Series,
                                        n_bins: int = DEFAULT_HISTOGRAM_BINS,
                                        percentiles: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Analyze how well noise addition preserves the data distribution.

    This function performs comprehensive statistical tests and comparisons
    to evaluate whether the noisy data maintains similar distributional
    properties to the original data.

    Args:
        original: Original data series
        noisy: Data series after noise addition
        n_bins: Number of bins for histogram comparison
        percentiles: List of percentiles to compare (default: [10, 25, 50, 75, 90])

    Returns:
        Dictionary containing:
            - ks_statistic: Kolmogorov-Smirnov test statistic
            - ks_pvalue: p-value for KS test
            - wasserstein_distance: Earth mover's distance between distributions
            - histogram_distance: L1 distance between normalized histograms
            - mean_shift: Difference in means
            - std_ratio: Ratio of standard deviations
            - skewness_diff: Difference in skewness
            - kurtosis_diff: Difference in kurtosis
            - percentile_shifts: Dict of percentile differences

    Example:
        >>> dist_metrics = calculate_distribution_preservation(original, noisy)
        >>> if dist_metrics['ks_pvalue'] > 0.05:
        ...     print("Distributions are statistically similar")
    """
    # Clean data
    orig_clean = original.dropna()
    noisy_clean = noisy.dropna()

    if len(orig_clean) == 0 or len(noisy_clean) == 0:
        logger.warning("Insufficient data for distribution analysis")
        return {}

    # Sample if too large for statistical tests
    if len(orig_clean) > MAX_SAMPLE_SIZE:
        orig_clean = orig_clean.sample(MAX_SAMPLE_SIZE, random_state=42)
    if len(noisy_clean) > MAX_SAMPLE_SIZE:
        noisy_clean = noisy_clean.sample(MAX_SAMPLE_SIZE, random_state=42)

    # KS test for distribution similarity
    ks_stat, ks_pvalue = stats.ks_2samp(orig_clean, noisy_clean)

    # Wasserstein distance (Earth Mover's Distance)
    wasserstein_dist = stats.wasserstein_distance(orig_clean, noisy_clean) #type: ignore

    # Histogram comparison
    # Create common bins based on combined range
    combined = pd.concat([orig_clean, noisy_clean])
    bins = np.histogram_bin_edges(combined, bins=n_bins)

    hist_orig, _ = np.histogram(orig_clean, bins=bins, density=True)
    hist_noisy, _ = np.histogram(noisy_clean, bins=bins, density=True)

    # L1 distance between histograms
    histogram_distance = np.sum(np.abs(hist_orig - hist_noisy)) * (bins[1] - bins[0])

    # Basic statistics comparison
    mean_shift = float(noisy_clean.mean() - orig_clean.mean())
    std_orig = orig_clean.std()
    std_noisy = noisy_clean.std()
    std_ratio = float(std_noisy / std_orig) if std_orig > 0 else float('inf')

    # Higher moments
    skewness_diff = float(noisy_clean.skew() - orig_clean.skew())
    kurtosis_diff = float(noisy_clean.kurtosis() - orig_clean.kurtosis())

    # Percentile analysis
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    percentile_shifts = {}
    for p in percentiles:
        orig_p = orig_clean.quantile(p / 100)
        noisy_p = noisy_clean.quantile(p / 100)
        percentile_shifts[f"p{p}"] = float(noisy_p - orig_p)

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": float(wasserstein_dist),
        "histogram_distance": float(histogram_distance),
        "mean_shift": mean_shift,
        "std_ratio": std_ratio,
        "skewness_diff": skewness_diff,
        "kurtosis_diff": kurtosis_diff,
        "percentile_shifts": percentile_shifts
    }


def suggest_noise_range(series: pd.Series,
                        target_snr: float = DEFAULT_TARGET_SNR,
                        noise_type: str = "additive",
                        distribution: str = "uniform") -> float:
    """
    Suggest appropriate noise range based on target signal-to-noise ratio.

    This function analyzes the data distribution and recommends a noise range
    that achieves the desired SNR while maintaining data utility.

    Args:
        series: Original data series
        target_snr: Desired signal-to-noise ratio (in linear scale, not dB)
        noise_type: Type of noise application ('additive' or 'multiplicative')
        distribution: Noise distribution ('uniform', 'normal', 'laplace')

    Returns:
        Suggested noise range/parameter value

    Raises:
        InvalidParameterError: If distribution is not recognized

    Example:
        >>> data = pd.Series(np.random.normal(100, 15, 1000))
        >>> # Get range for 20dB SNR (linear SNR = 10)
        >>> noise_range = suggest_noise_range(data, target_snr=10.0)
        >>> print(f"Suggested uniform noise range: ±{noise_range:.2f}")
    """
    if distribution not in NOISE_DISTRIBUTIONS:
        raise InvalidParameterError(
            param_name="distribution",
            param_value=distribution,
            reason=f"must be one of {NOISE_DISTRIBUTIONS}"
        )

    # Calculate signal statistics
    signal_std = series.std()
    signal_var = signal_std ** 2

    # Target noise variance
    target_noise_var = signal_var / target_snr
    target_noise_std = np.sqrt(target_noise_var)

    if noise_type == "additive":
        # Calculate range based on distribution
        if distribution == "uniform":
            # For uniform distribution: var = range²/12
            suggested_range = target_noise_std * SQRT_12
        elif distribution == "normal":
            # For normal: parameter is standard deviation
            suggested_range = target_noise_std
        elif distribution == "laplace":
            # For Laplace: var = 2*scale²
            suggested_range = target_noise_std / SQRT_2
        elif distribution == "exponential":
            # For exponential: var = scale²
            suggested_range = target_noise_std
        else:
            # Default to uniform
            suggested_range = target_noise_std * SQRT_12

    else:  # multiplicative
        # For multiplicative noise, we need relative noise
        signal_mean = series.mean()
        if abs(signal_mean) > EPSILON:
            # Scale by mean for multiplicative noise
            relative_std = target_noise_std / abs(signal_mean)

            if distribution == "uniform":
                suggested_range = relative_std * SQRT_12
            elif distribution == "normal":
                suggested_range = relative_std
            else:
                suggested_range = relative_std
        else:
            logger.warning("Mean is near zero, multiplicative noise may not be appropriate")
            suggested_range = MULTIPLICATIVE_DEFAULT_RANGE  # Default 10% variation

    return float(suggested_range)


def validate_noise_bounds(series: pd.Series,
                          noise_range: Union[float, Tuple[float, float]],
                          output_min: Optional[float] = None,
                          output_max: Optional[float] = None,
                          percentile_check: float = PERCENTILE_CHECK_DEFAULT) -> Dict[str, Any]:
    """
    Validate that noise bounds are reasonable for the data.

    This function checks whether the specified noise parameters and output bounds
    are appropriate for the data distribution, helping prevent excessive noise
    or overly restrictive bounds.

    Args:
        series: Original data series
        noise_range: Noise range (symmetric float or (min, max) tuple)
        output_min: Minimum allowed output value
        output_max: Maximum allowed output value
        percentile_check: Percentile to check for bound violations (default: 99%)

    Returns:
        Dictionary containing:
            - valid: Whether bounds are reasonable
            - warnings: List of warning messages
            - noise_to_data_ratio: Ratio of noise range to data range
            - expected_violations: Estimated percentage of values hitting bounds
            - recommendations: Suggested adjustments

    Example:
        >>> validation = validate_noise_bounds(
        ...     data, noise_range=50, output_min=0, output_max=200 #type: ignore
        ... )
        >>> if not validation['valid']:
        ...     print("Warnings:", validation['warnings'])
    """
    data_min = series.min()
    data_max = series.max()
    data_range = data_max - data_min
    data_std = series.std()

    # Calculate noise magnitude
    if isinstance(noise_range, tuple):
        noise_min, noise_max = noise_range
        noise_magnitude = max(abs(noise_min), abs(noise_max))
        is_symmetric = abs(noise_min + noise_max) < 1e-10
    else:
        noise_magnitude = abs(noise_range)
        noise_min, noise_max = -noise_magnitude, noise_magnitude
        is_symmetric = True

    warnings = []
    recommendations = []

    # Check if noise is too large relative to data
    noise_to_data_ratio = noise_magnitude / data_range if data_range > 0 else float('inf')

    if noise_to_data_ratio > NOISE_RATIO_EXCESSIVE:
        warnings.append(
            f"Noise range ({noise_magnitude:.2f}) exceeds data range ({data_range:.2f})"
        )
        recommendations.append(
            f"Consider reducing noise range to {UTILITY_RATIO_THRESHOLD * data_range:.2f} for better utility"
        )
    elif noise_to_data_ratio > NOISE_RATIO_WARNING:
        warnings.append(
            f"Noise range is {noise_to_data_ratio:.1%} of data range (may be excessive)"
        )

    # Check output bounds
    expected_violations = {"lower": 0.0, "upper": 0.0}

    if output_min is not None or output_max is not None:
        # Estimate bound violations using percentiles
        percentiles = series.quantile([PERCENTILE_LOWER_CHECK, 0.05, 0.95, PERCENTILE_UPPER_CHECK])

        if output_min is not None:
            # Check lower bound
            worst_case_min = data_min + noise_min
            if worst_case_min < output_min:
                # Estimate violation probability
                violation_point = output_min - noise_min
                pct_below = (series < violation_point).mean() * 100
                expected_violations["lower"] = pct_below

                if pct_below > VIOLATION_THRESHOLD_PCT:
                    warnings.append(
                        f"Approximately {pct_below:.1f}% of values may hit lower bound"
                    )
                    recommendations.append(
                        f"Consider lowering output_min to {percentiles[PERCENTILE_LOWER_CHECK] + noise_min:.2f}"
                    )

            if output_min > data_min:
                warnings.append(
                    f"Output minimum ({output_min:.2f}) exceeds data minimum ({data_min:.2f})"
                )

        if output_max is not None:
            # Check upper bound
            worst_case_max = data_max + noise_max
            if worst_case_max > output_max:
                # Estimate violation probability
                violation_point = output_max - noise_max
                pct_above = (series > violation_point).mean() * 100
                expected_violations["upper"] = pct_above

                if pct_above > VIOLATION_THRESHOLD_PCT:
                    warnings.append(
                        f"Approximately {pct_above:.1f}% of values may hit upper bound"
                    )
                    recommendations.append(
                        f"Consider raising output_max to {percentiles[PERCENTILE_UPPER_CHECK] + noise_max:.2f}"
                    )

            if output_max < data_max:
                warnings.append(
                    f"Output maximum ({output_max:.2f}) is less than data maximum ({data_max:.2f})"
                )

    # Check for distribution compatibility
    if not is_symmetric:
        skewness = series.skew()
        if abs(skewness) > SKEWNESS_THRESHOLD and np.sign(skewness) != np.sign(noise_min + noise_max):
            warnings.append(
                "Asymmetric noise direction conflicts with data skewness"
            )
            recommendations.append(
                "Consider using symmetric noise or adjusting noise direction"
            )

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "recommendations": recommendations,
        "noise_to_data_ratio": noise_to_data_ratio,
        "expected_violations": expected_violations,
        "data_bounds": {"min": float(data_min), "max": float(data_max)},
        "data_stats": {
            "mean": float(series.mean()),
            "std": float(data_std),
            "skewness": float(series.skew())
        },
        "noise_magnitude": noise_magnitude
    }


def generate_numeric_noise(size: int,
                           distribution: str = "uniform",
                           params: Optional[Dict[str, float]] = None,
                           secure: bool = True,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Generate numeric noise values with specified distribution.

    Convenience function for generating noise arrays with various distributions
    and parameters. Supports both secure and reproducible generation.

    Args:
        size: Number of noise values to generate
        distribution: Distribution type ('uniform', 'normal', 'laplace', 'exponential')
        params: Distribution parameters (see examples)
        secure: Whether to use cryptographically secure generation
        seed: Random seed (only used if secure=False)

    Returns:
        Array of noise values

    Raises:
        InvalidParameterError: If size <= 0 or distribution not recognized

    Examples:
        >>> # Uniform noise in [-10, 10]
        >>> noise = generate_numeric_noise(1000, 'uniform', {'low': -10, 'high': 10})
        >>>
        >>> # Normal noise with mean=0, std=5
        >>> noise = generate_numeric_noise(1000, 'normal', {'loc': 0, 'scale': 5}) #type: ignore
        >>>
        >>> # Laplace noise for differential privacy
        >>> noise = generate_numeric_noise(1000, 'laplace', {'scale': 2.0}) #type: ignore
    """
    if size <= 0:
        raise InvalidParameterError(
            param_name="size",  # ← NEW
            param_value=size,
            reason="size must be positive"
        )

    if distribution not in NOISE_DISTRIBUTIONS:
        raise InvalidParameterError(
            param_name="distribution",  # ← NEW
            param_value=distribution,
            reason=f"must be one of {NOISE_DISTRIBUTIONS}"
        )

    # Default parameters
    if params is None:
        params = {}

    # Create generator
    rng = SecureRandomGenerator(use_secure=secure, seed=seed)

    # Generate noise based on distribution
    if distribution == "uniform":
        low = params.get('low', -UNIFORM_RANGE_DEFAULT)
        high = params.get('high', UNIFORM_RANGE_DEFAULT)
        return rng.uniform(low, high, size)

    elif distribution == "normal":
        loc = params.get('loc', NORMAL_LOC_DEFAULT)
        scale = params.get('scale', NORMAL_SCALE_DEFAULT)
        return rng.normal(loc, scale, size)

    elif distribution == "laplace":
        loc = params.get('loc', LAPLACE_LOC_DEFAULT)
        scale = params.get('scale', LAPLACE_SCALE_DEFAULT)
        return rng.laplace(loc, scale, size)


    elif distribution == "exponential":

        scale = params.get('scale', EXPONENTIAL_SCALE_DEFAULT)

        if params.get('symmetric', False):

            exp_values = np.random.exponential(scale, size)
            signs = rng.choice(np.array([-1, 1], dtype=int), size=size)
            return exp_values * signs

        else:

            return np.random.exponential(scale, size)

    else:
        # Fallback to uniform
        return rng.uniform(-UNIFORM_RANGE_DEFAULT, UNIFORM_RANGE_DEFAULT, size)


def generate_temporal_noise(size: int,
                            range_value: float,
                            unit: str = "days",
                            distribution: str = "uniform",
                            secure: bool = True,
                            seed: Optional[int] = None) -> pd.TimedeltaIndex:
    """
    Generate temporal noise values as time deltas.

    Creates time shift values for adding noise to datetime fields, with
    support for different time units and distributions.

    Args:
        size: Number of time deltas to generate
        range_value: Maximum shift in specified unit
        unit: Time unit ('seconds', 'minutes', 'hours', 'days', 'weeks')
        distribution: Distribution type (currently only 'uniform' is fully supported)
        secure: Whether to use cryptographically secure generation
        seed: Random seed (only used if secure=False)

    Returns:
        TimedeltaIndex with random time shifts

    Raises:
        InvalidParameterError: If unit not recognized

    Example:
        >>> # Generate random shifts of ±7 days
        >>> shifts = generate_temporal_noise(100, range_value=7, unit='days')
        >>>
        >>> # Apply to datetime series
        >>> noisy_dates = original_dates + shifts #type: ignore
    """
    if unit not in TEMPORAL_UNITS:
        raise InvalidParameterError(
            param_name="unit",
            param_value=unit,
            reason=f"must be one of {TEMPORAL_UNITS}"
        )

    # Convert range to seconds
    unit_multipliers = {
        'seconds': 1,
        'minutes': SECONDS_PER_MINUTE,
        'hours': SECONDS_PER_HOUR,
        'days': SECONDS_PER_DAY,
        'weeks': SECONDS_PER_WEEK
    }

    range_seconds = range_value * unit_multipliers[unit]

    # Generate noise values
    if distribution == "uniform":
        noise_seconds = generate_numeric_noise(
            size, 'uniform',
            {'low': -range_seconds, 'high': range_seconds},
            secure=secure, seed=seed
        )
    else:
        # For other distributions, generate and scale
        noise_values = generate_numeric_noise(
            size, distribution,
            {'scale': range_seconds / NOISE_SCALE_DIVISOR},  # Rough scaling
            secure=secure, seed=seed
        )
        noise_seconds = np.clip(noise_values, -range_seconds, range_seconds)

    # Convert to TimedeltaIndex
    return pd.to_timedelta(noise_seconds, unit='s')


def analyze_noise_effectiveness(original: pd.Series,
                                noisy: pd.Series,
                                privacy_metric: str = "snr",
                                target_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze the effectiveness of noise addition for privacy protection.

    This function evaluates whether the noise addition achieves the desired
    privacy level while maintaining acceptable utility.

    Args:
        original: Original data series
        noisy: Data series after noise addition
        privacy_metric: Metric to use ('snr', 'rmse', 'correlation')
        target_value: Target value for the privacy metric

    Returns:
        Dictionary containing:
            - effective: Whether noise meets privacy requirements
            - actual_value: Actual metric value achieved
            - target_value: Target value (if provided)
            - utility_score: Overall utility preservation score (0-1)
            - recommendations: List of suggestions for improvement

    Example:
        >>> analysis = analyze_noise_effectiveness(
        ...     original, noisy, privacy_metric='snr', target_value=10.0
        ... )
        >>> print(f"Privacy effective: {analysis['effective']}")
    """
    # Calculate comprehensive metrics
    impact = calculate_noise_impact(original, noisy)
    distribution = calculate_distribution_preservation(original, noisy)

    # Evaluate privacy metric
    if privacy_metric == "snr":
        actual_value = 10 ** (impact['snr'] / DB_CONVERSION_FACTOR)  # Convert from dB to linear
        effective = actual_value <= target_value if target_value else True
        metric_direction = "lower"
    elif privacy_metric == "rmse":
        actual_value = impact['rmse']
        effective = actual_value >= target_value if target_value else True
        metric_direction = "higher"
    elif privacy_metric == "correlation":
        actual_value = impact['correlation']
        effective = actual_value <= target_value if target_value else True
        metric_direction = "lower"
    else:
        actual_value = impact.get(privacy_metric, 0)
        effective = True
        metric_direction = "unknown"

    # Calculate utility score (0-1, higher is better)
    utility_components = [
        min(1.0, impact['correlation']),  # Correlation preservation
        min(1.0, 1.0 / (1.0 + impact['relative_error'])),  # Low relative error
        min(1.0, 1.0 / (1.0 + abs(distribution['mean_shift']))),  # Mean preservation
        min(1.0, 2.0 - abs(distribution['std_ratio'] - 1.0)),  # Std preservation
        min(1.0, distribution['ks_pvalue'] * DB_CONVERSION_FACTOR)  # Distribution similarity
    ]
    utility_score = np.mean(utility_components)

    # Generate recommendations
    recommendations = []

    if not effective and target_value:
        if metric_direction == "lower" and actual_value > target_value:
            recommendations.append(
                f"Increase noise magnitude to achieve {privacy_metric} ≤ {target_value}"
            )
        elif metric_direction == "higher" and actual_value < target_value:
            recommendations.append(
                f"Increase noise magnitude to achieve {privacy_metric} ≥ {target_value}"
            )

    if utility_score < UTILITY_SCORE_LOW:
        recommendations.append("Consider reducing noise magnitude to improve utility")

    if distribution['ks_pvalue'] < DISTRIBUTION_PVALUE_THRESHOLD:
        recommendations.append(
            "Noise significantly alters distribution - consider distribution-preserving noise"
        )

    if impact['correlation'] < CORRELATION_LOW_THRESHOLD:
        recommendations.append(
            "Low correlation with original data - verify noise range is appropriate"
        )

    return {
        "effective": effective,
        "actual_value": float(actual_value),
        "target_value": target_value,
        "privacy_metric": privacy_metric,
        "utility_score": float(utility_score),
        "utility_components": {
            "correlation": float(utility_components[0]),
            "relative_error": float(utility_components[1]),
            "mean_preservation": float(utility_components[2]),
            "std_preservation": float(utility_components[3]),
            "distribution_similarity": float(utility_components[4])
        },
        "recommendations": recommendations
    }


def create_noise_report(original: pd.Series,
                        noisy: pd.Series,
                        noise_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive noise analysis report.

    Generates a detailed report summarizing all aspects of noise addition,
    including configuration, impact metrics, distribution analysis, and
    effectiveness evaluation.

    Args:
        original: Original data series
        noisy: Data series after noise addition
        noise_params: Dictionary of noise parameters used

    Returns:
        Comprehensive report dictionary

    Example:
        >>> report = create_noise_report(
        ...     original, noisy,
        ...     {'type': 'uniform', 'range': 10.0, 'distribution': 'uniform'}
        ... )
        >>> print(json.dumps(report, indent=2)) #type: ignore
    """
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "configuration": noise_params,
        "data_summary": {
            "original": {
                "count": int(original.count()),
                "mean": float(original.mean()),
                "std": float(original.std()),
                "min": float(original.min()),
                "max": float(original.max())
            },
            "noisy": {
                "count": int(noisy.count()),
                "mean": float(noisy.mean()),
                "std": float(noisy.std()),
                "min": float(noisy.min()),
                "max": float(noisy.max())
            }
        },
        "impact_metrics": calculate_noise_impact(original, noisy),
        "distribution_analysis": calculate_distribution_preservation(original, noisy),
        "effectiveness": analyze_noise_effectiveness(
            original, noisy,
            privacy_metric=noise_params.get('privacy_metric', 'snr'),
            target_value=noise_params.get('target_value')
        )
    }

    # Add actual noise statistics
    actual_noise = noisy - original
    report["actual_noise_stats"] = {
        "mean": float(actual_noise.mean()),
        "std": float(actual_noise.std()),
        "min": float(actual_noise.min()),
        "max": float(actual_noise.max()),
        "zero_noise_count": int((actual_noise == 0).sum())
    }

    return report


# Module metadata
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"

# Export public API
__all__ = [
    # Main classes
    'SecureRandomGenerator',

    # Analysis functions
    'calculate_noise_impact',
    'calculate_distribution_preservation',
    'suggest_noise_range',
    'validate_noise_bounds',

    # Generation functions
    'generate_numeric_noise',
    'generate_temporal_noise',

    # Analysis and reporting
    'analyze_noise_effectiveness',
    'create_noise_report',

    # Constants
    'NOISE_DISTRIBUTIONS',
    'TEMPORAL_UNITS',
    'DEFAULT_HISTOGRAM_BINS',
    'MAX_SAMPLE_SIZE'
]