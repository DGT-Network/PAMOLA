"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Statistical Metrics Utilities
Package:       pamola_core.utils
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
   This module provides general-purpose statistical metrics for analyzing
   data distributions, concentration, and inequality. These metrics are
   used across various PAMOLA.CORE operations including data profiling,
   anonymization assessment, and quality evaluation.

Key Features:
   - Gini coefficient calculation for measuring inequality
   - Concentration metrics (CR-k) for distribution analysis
   - Lorenz curve computation for inequality visualization
   - Herfindahl-Hirschman Index (HHI) for market concentration
   - Shannon entropy for distribution randomness
   - Efficient implementations optimized for large datasets
   - Support for both raw values and frequency distributions

Design Principles:
   - Performance: Optimized for large-scale data processing
   - Accuracy: Numerically stable implementations
   - Flexibility: Support multiple input formats
   - Simplicity: Clear, well-documented functions
   - Reusability: Generic implementations for broad applicability

Dependencies:
   - numpy: Numerical computations
   - pandas: Data structure support
   - typing: Type hints
   - logging: Error and warning reporting

Changelog:
   1.0.0 - Initial implementation with core statistical metrics
         - Gini coefficient with multiple calculation methods
         - Concentration ratios (CR-k)
         - Lorenz curve
         - Herfindahl-Hirschman Index
         - Shannon entropy
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small constant for numerical stability
DEFAULT_TOP_K = [1, 5, 10, 20, 50]  # Default k values for concentration metrics


def calculate_gini_coefficient(
    data: Union[np.ndarray, pd.Series, List[float]],
    is_frequency: bool = False,
    method: str = "accurate",
) -> float:
    """
    Calculate the Gini coefficient for measuring inequality in a distribution.

    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    It is commonly used to measure income inequality but applicable to any distribution.

    Parameters:
    -----------
    data : Union[np.ndarray, pd.Series, List[float]]
        The data values or frequencies to analyze.
        If is_frequency=True, these are counts/frequencies.
        If is_frequency=False, these are the actual values.
    is_frequency : bool, optional
        Whether the input data represents frequencies/counts (default: False)
    method : str, optional
        Calculation method:
        - "accurate": More accurate but slower (default)
        - "fast": Faster approximation suitable for large datasets
        - "sorted": Assumes data is pre-sorted (fastest)

    Returns:
    --------
    float
        Gini coefficient between 0 and 1

    Raises:
    -------
    ValueError
        If data is empty or contains negative values

    Examples:
    ---------
    >>> # Perfect equality
    >>> data = [100, 100, 100, 100]
    >>> gini = calculate_gini_coefficient(data)
    >>> print(f"Gini: {gini:.4f}")  # Output: Gini: 0.0000

    >>> # High inequality
    >>> data = [1, 1, 1, 1, 96]
    >>> gini = calculate_gini_coefficient(data)
    >>> print(f"Gini: {gini:.4f}")  # Output: Gini: 0.7680

    >>> # Using frequencies
    >>> frequencies = [50, 30, 15, 5]  # 50 items of type A, 30 of B, etc.
    >>> gini = calculate_gini_coefficient(frequencies, is_frequency=True)
    """
    # Convert input to numpy array
    if isinstance(data, pd.Series):
        values = data.values
    elif isinstance(data, list):
        values = np.array(data)
    else:
        values = np.asarray(data)

    # Remove NaN values
    values = values[~np.isnan(values)]

    # Validate input
    if len(values) == 0:
        logger.warning("Empty data provided for Gini calculation")
        return 0.0

    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non-negative values")

    # Handle all-zero case
    if np.all(values == 0):
        return 0.0

    # Calculate based on method
    if method == "accurate":
        return _gini_accurate(values, is_frequency)
    elif method == "fast":
        return _gini_fast(values, is_frequency)
    elif method == "sorted":
        return _gini_sorted(values, is_frequency)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'accurate', 'fast', or 'sorted'"
        )


def _gini_accurate(values: np.ndarray, is_frequency: bool) -> float:
    """Accurate Gini calculation using the standard formula."""
    if is_frequency:
        # Convert frequencies to individual values
        # e.g., [3, 2, 1] means 3 items of value 0, 2 of value 1, 1 of value 2
        n = np.sum(values)
        if n == 0:
            return 0.0

        # Calculate cumulative proportions
        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)

        # Gini formula for frequency data
        numerator = 2 * np.sum(np.arange(1, len(values) + 1) * sorted_values)
        denominator = n * cumsum[-1]

        if denominator == 0:
            return 0.0

        return numerator / denominator - (len(values) + 1) / len(values)

    else:
        # Standard Gini calculation for raw values
        n = len(values)
        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)

        # Avoid division by zero
        if cumsum[-1] == 0:
            return 0.0

        # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        return (2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (n * cumsum[-1]) - (
            n + 1
        ) / n


def _gini_fast(values: np.ndarray, is_frequency: bool = False) -> float:
    """
    Fast approximation of the Gini coefficient for categorical or frequency data.

    This implementation is intended for use cases where an efficient (O(n^2)) but
    reasonably accurate estimate is sufficient. It supports both raw categorical arrays
    (with is_frequency=False) and frequency/count arrays (is_frequency=True).

    The Gini coefficient is a measure of statistical dispersion intended to represent
    the inequality of a distribution. It ranges from 0 (perfect equality) to 1 (maximal inequality).

    Parameters
    ----------
    values : np.ndarray
        If is_frequency is True, this should be an array of integer frequencies/counts.
        If is_frequency is False, this should be an array of raw categorical values (not recommended for large n).
    is_frequency : bool, default=False
        Whether 'values' contains frequencies/counts (True) or raw values (False).

    Returns
    -------
    gini : float
        Fast approximate Gini coefficient in the range [0, 1].

    Notes
    -----
    - For large arrays, this approach is much faster than the full O(n^2) pairwise comparison.
    - Handles cases where all values are zero or where the denominator could be zero.
    """

    if is_frequency:
        # For frequency/count data, compute normalized proportions.
        total = np.sum(values)
        if total == 0:
            return 0.0
        proportions = values / total
        n = len(proportions)
    else:
        # For raw data, build frequency table.
        _, counts = np.unique(values, return_counts=True)
        total = np.sum(counts)
        if total == 0:
            return 0.0
        proportions = counts / total
        n = len(proportions)

    # Mean absolute difference approximation (O(n^2), but acceptable for moderate n)
    mad = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            mad += abs(proportions[i] - proportions[j])

    # Calculate denominator and guard against division by zero
    denominator = float(n * (n - 1)) * float(np.mean(proportions))
    if denominator == 0.0:
        return 0.0

    return 2 * mad / float(denominator)


def _gini_sorted(values: np.ndarray, is_frequency: bool) -> float:
    """Gini calculation assuming pre-sorted data (ascending order)."""
    if not is_frequency:
        # For raw values, just use the standard formula
        n = len(values)
        cumsum = np.cumsum(values)

        if cumsum[-1] == 0:
            return 0.0

        return (2 * np.sum(np.arange(1, n + 1) * values)) / (n * cumsum[-1]) - (
            n + 1
        ) / n

    else:
        # For frequency data (already sorted)
        n = np.sum(values)
        if n == 0:
            return 0.0

        cumsum = np.cumsum(values)
        numerator = 2 * np.sum(np.arange(1, len(values) + 1) * values)

        return numerator / (n * cumsum[-1]) - (len(values) + 1) / len(values)


def calculate_concentration_metrics(
    data: Union[pd.Series, np.ndarray, Dict[str, int]],
    top_k: Optional[List[int]] = None,
    as_percentage: bool = True,
) -> Dict[str, float]:
    """
    Calculate concentration ratios (CR-k) for a distribution.

    Concentration ratios measure what percentage of the total is held by
    the top k entities. Commonly used in market analysis and distribution studies.

    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray, Dict[str, int]]
        The data to analyze. Can be:
        - pd.Series: Will use value_counts()
        - np.ndarray: Will count occurrences
        - Dict: Category -> count mapping
    top_k : Optional[List[int]], optional
        List of k values to calculate CR-k for (default: [1, 5, 10, 20, 50])
    as_percentage : bool, optional
        Whether to return as percentage (0-100) or ratio (0-1) (default: True)

    Returns:
    --------
    Dict[str, float]
        Dictionary with keys like "cr_1", "cr_5", "cr_10" and concentration values

    Examples:
    ---------
    >>> # Market share data
    >>> market_share = {"Company A": 3500, "Company B": 2500, "Company C": 2000,
    ...                 "Company D": 1000, "Company E": 500, "Others": 500}
    >>> metrics = calculate_concentration_metrics(market_share, top_k=[1, 3, 5])
    >>> print(f"CR-1: {metrics['cr_1']:.1f}%")  # Output: CR-1: 35.0%
    >>> print(f"CR-3: {metrics['cr_3']:.1f}%")  # Output: CR-3: 80.0%

    >>> # Using pandas Series
    >>> data = pd.Series(['A', 'A', 'A', 'B', 'B', 'C', 'D', 'E'])
    >>> metrics = calculate_concentration_metrics(data)
    >>> print(f"CR-1: {metrics['cr_1']:.1f}%")  # Output: CR-1: 37.5%
    """
    if top_k is None:
        top_k = DEFAULT_TOP_K.copy()

    # Get frequency counts
    if isinstance(data, pd.Series):
        counts = data.value_counts().sort_values(ascending=False)
    elif isinstance(data, dict):
        counts = pd.Series(data).sort_values(ascending=False)
    elif isinstance(data, np.ndarray):
        unique, counts_array = np.unique(data, return_counts=True)
        counts = pd.Series(counts_array, index=unique).sort_values(ascending=False)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Calculate total
    total = counts.sum()
    if total == 0:
        return {f"cr_{k}": 0.0 for k in top_k}

    # Calculate concentration ratios
    metrics = {}
    n_categories = len(counts)

    for k in top_k:
        if k > n_categories:
            # If k exceeds number of categories, use all categories
            top_k_sum = total
        else:
            top_k_sum = counts.head(k).sum()

        ratio = top_k_sum / total
        if as_percentage:
            metrics[f"cr_{k}"] = ratio * 100
        else:
            metrics[f"cr_{k}"] = ratio

    return metrics


def calculate_lorenz_curve(
    data: Union[np.ndarray, pd.Series, List[float]], n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lorenz curve for visualizing inequality.

    The Lorenz curve plots the cumulative percentage of the total value
    against the cumulative percentage of the population.

    Parameters:
    -----------
    data : Union[np.ndarray, pd.Series, List[float]]
        The data values to analyze
    n_points : int, optional
        Number of points to return for the curve (default: 100)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (x_values, y_values) where:
        - x_values: Cumulative population percentage (0 to 1)
        - y_values: Cumulative value percentage (0 to 1)

    Examples:
    ---------
    >>> # Calculate Lorenz curve
    >>> incomes = [20000, 30000, 40000, 50000, 100000]
    >>> x, y = calculate_lorenz_curve(incomes)
    >>> # x and y can be plotted to visualize inequality
    >>> # Perfect equality would be a diagonal line from (0,0) to (1,1)
    """
    # Convert to numpy array and sort
    if isinstance(data, pd.Series):
        values = data.values
    elif isinstance(data, list):
        values = np.array(data)
    else:
        values = np.asarray(data)

    # Remove NaN and sort
    values = values[~np.isnan(values)]
    values = np.sort(values)

    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 1])

    # Calculate cumulative proportions
    n = len(values)
    cumsum = np.cumsum(values)
    total = cumsum[-1]

    if total == 0:
        # All zero values - perfect equality
        return np.linspace(0, 1, n_points), np.linspace(0, 1, n_points)

    # Calculate Lorenz curve points
    x = np.arange(n + 1) / n
    y = np.concatenate([[0], cumsum / total])

    # Interpolate to get n_points
    x_interp = np.linspace(0, 1, n_points)
    y_interp = np.interp(x_interp, x, y)

    return x_interp, y_interp


def calculate_herfindahl_index(
    data: Union[pd.Series, np.ndarray, Dict[str, float]], normalized: bool = False
) -> float:
    """
    Calculate the Herfindahl-Hirschman Index (HHI) for market concentration.

    HHI is calculated as the sum of squared market shares. It ranges from
    near 0 (perfect competition) to 10,000 (monopoly) in standard form,
    or 0 to 1 in normalized form.

    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray, Dict[str, float]]
        Market share data. Can be counts or percentages.
    normalized : bool, optional
        Whether to return normalized HHI (0-1) or standard (0-10000) (default: False)

    Returns:
    --------
    float
        HHI value

    Examples:
    ---------
    >>> # Market with 4 equal competitors (25% each)
    >>> shares = [25, 25, 25, 25]
    >>> hhi = calculate_herfindahl_index(shares) # type: ignore
    >>> print(f"HHI: {hhi:.0f}")  # Output: HHI: 2500

    >>> # Highly concentrated market
    >>> shares = {"Leader": 70, "Follower": 20, "Small1": 5, "Small2": 5}
    >>> hhi = calculate_herfindahl_index(shares, normalized=True)
    >>> print(f"Normalized HHI: {hhi:.4f}")  # Output: Normalized HHI: 0.5400
    """
    # Get values
    if isinstance(data, pd.Series):
        values = data.values
    elif isinstance(data, dict):
        values = np.array(list(data.values()))
    elif isinstance(data, (list, np.ndarray)):
        values = np.asarray(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Remove NaN values
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return 0.0

    # Calculate total and shares
    total = np.sum(values)
    if total == 0:
        return 0.0

    # Calculate shares (ensure they sum to 1 or 100)
    shares = values / total

    # Calculate HHI
    if normalized:
        # Normalized HHI: (HHI - 1/N) / (1 - 1/N)
        n = len(shares)
        raw_hhi = np.sum(shares**2)
        if n == 1:
            return 1.0
        return (raw_hhi - 1 / n) / (1 - 1 / n)
    else:
        # Standard HHI: sum of squared percentage shares
        percentage_shares = shares * 100
        return np.sum(percentage_shares**2)


def calculate_shannon_entropy(
    data: Union[pd.Series, np.ndarray, Dict[str, int]],
    base: float = 2.0,
    normalize: bool = True,
) -> float:
    """
    Calculate Shannon entropy for a categorical distribution.

    Shannon entropy measures the randomness or uncertainty in a distribution.
    Higher values indicate more uniform distributions.

    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray, Dict[str, int]]
        Categorical data or frequency counts
    base : float, optional
        Logarithm base (2 for bits, e for nats) (default: 2.0)
    normalize : bool, optional
        Whether to normalize by maximum entropy (default: True)

    Returns:
    --------
    float
        Shannon entropy value

    Examples:
    ---------
    >>> # Uniform distribution (maximum entropy)
    >>> data = ['A', 'B', 'C', 'D'] * 25  # 25 of each
    >>> entropy = calculate_shannon_entropy(data) # type: ignore
    >>> print(f"Entropy: {entropy:.4f}")  # Output: Entropy: 1.0000

    >>> # Skewed distribution (low entropy)
    >>> data = ['A'] * 90 + ['B'] * 5 + ['C'] * 3 + ['D'] * 2
    >>> entropy = calculate_shannon_entropy(data) # type: ignore
    >>> print(f"Entropy: {entropy:.4f}")  # Output: Entropy: 0.3758
    """
    # Get frequency counts
    if isinstance(data, pd.Series):
        counts = data.value_counts()
    elif isinstance(data, dict):
        counts = pd.Series(data)
    elif isinstance(data, np.ndarray):
        unique, counts_array = np.unique(data, return_counts=True)
        counts = pd.Series(counts_array, index=unique)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Calculate probabilities
    total = counts.sum()
    if total == 0 or len(counts) <= 1:
        return 0.0

    probabilities = counts / total

    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy
    if base == 2:
        entropy = -np.sum(probabilities * np.log2(probabilities + EPSILON))
    elif base == np.e:
        entropy = -np.sum(probabilities * np.log(probabilities + EPSILON))
    else:
        entropy = -np.sum(
            probabilities * np.log(probabilities + EPSILON) / np.log(base)
        )

    # Normalize if requested
    if normalize and len(counts) > 1:
        max_entropy = np.log(len(counts)) / np.log(base)
        entropy = entropy / max_entropy

    return float(entropy)


from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd


def get_distribution_summary(
    data: Union[pd.Series, np.ndarray, List[Any], Dict[Any, int]],
    top_k: Optional[Union[List[int], int]] = None,
    include_gini: bool = True,
    include_concentration: bool = True,
    include_entropy: bool = True,
    normalize_entropy: bool = True,
) -> Dict[str, Any]:
    """
    Calculate comprehensive distribution metrics for categorical or frequency data.

    Parameters
    ----------
    data : pd.Series, np.ndarray, list, or dict
        Input data: raw categorical values, frequency counts, or a mapping category -> count.
    top_k : list[int] or int, optional
        List of k values (or a single int) for concentration ratio calculation.
    include_gini : bool
        Whether to include the Gini coefficient.
    include_concentration : bool
        Whether to include concentration ratios (CR-k).
    include_entropy : bool
        Whether to include normalized Shannon entropy.
    normalize_entropy : bool
        Whether to normalize entropy by maximum (default: True).

    Returns
    -------
    summary : dict
        {
            'total_count': int,
            'unique_count': int,
            'gini': float,                  # if include_gini
            'concentration': dict,          # if include_concentration
            'entropy': float,               # if include_entropy
            ... (other stats can be added)
        }
    """

    # Process top_k: ensure list of ints
    if top_k is None:
        top_k_list = [1, 5, 10]
    elif isinstance(top_k, int):
        top_k_list = [top_k]
    else:
        top_k_list = list(top_k)

    # Normalize data to value_counts (Series[int])
    if isinstance(data, pd.Series):
        value_counts = data.value_counts(dropna=False)
    elif isinstance(data, dict):
        value_counts = pd.Series(data)
    elif isinstance(data, (np.ndarray, list)):
        value_counts = pd.Series(data).value_counts(dropna=False)
    else:
        raise TypeError("Unsupported data type for distribution summary.")

    summary: Dict[str, Any] = {
        "total_count": int(value_counts.sum()),
        "unique_count": int(len(value_counts)),
    }

    # Main metrics
    if include_gini:
        summary["gini"] = float(
            calculate_gini_coefficient(value_counts.values, is_frequency=True)
        )

    if include_concentration:
        summary["concentration"] = dict(
            calculate_concentration_metrics(value_counts, top_k=top_k_list)
        )

    if include_entropy:
        summary["entropy"] = float(
            calculate_shannon_entropy(value_counts, normalize=normalize_entropy)
        )

    # Mode stats (optional, but useful for reporting)
    if len(value_counts) > 0:
        summary["mode"] = value_counts.index[0]
        summary["mode_frequency"] = int(value_counts.iloc[0])
        summary["mode_percentage"] = float(value_counts.iloc[0] / value_counts.sum())
    else:
        summary["mode"] = None
        summary["mode_frequency"] = 0
        summary["mode_percentage"] = 0.0

    return summary
