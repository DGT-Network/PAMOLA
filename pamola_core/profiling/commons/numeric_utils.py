"""
Utility functions for numeric data analysis.

This module provides utility functions for analyzing numeric data, including
statistical calculations, distribution analysis, outlier detection, and normality testing.
These functions are used by the NumericAnalyzer class and other components that need
to perform numeric analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure logger for this module
logger = logging.getLogger(__name__)


def calculate_skewness(data: Union[pd.Series, List, np.ndarray], min_samples: int = 3) -> float:
    """
    Safely calculate skewness for a numeric dataset.

    Parameters:
    -----------
    data : array-like
        Numeric data to calculate skewness
    min_samples : int
        Minimum number of samples required for calculation

    Returns:
    --------
    float
        Skewness value or 0.0 if calculation fails
    """
    try:
        # Convert to a 1D numpy array to ensure compatibility
        data_array = np.asarray(data, dtype=float).flatten()

        # Check if we have enough samples
        if len(data_array) >= min_samples:
            # Calculate skewness manually if stats.skew is not working
            # This is the Fisher-Pearson formula for unbiased skewness
            n = len(data_array)
            mean = np.mean(data_array)
            deviation = data_array - mean
            variance = np.var(data_array, ddof=1)  # Use unbiased variance

            if variance == 0:
                return 0.0

            # Calculate the third moment
            m3 = np.sum(deviation ** 3) / n

            # Unbiased skewness formula
            skewness = m3 / (variance ** (3 / 2)) * np.sqrt(n * (n - 1)) / (n - 2)

            return float(skewness)
        return 0.0
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating skewness: {e}")
        # Return 0.0 for any calculation errors
        return 0.0


def calculate_kurtosis(data: Union[pd.Series, List, np.ndarray], min_samples: int = 4) -> float:
    """
    Safely calculate kurtosis for a numeric dataset.

    Parameters:
    -----------
    data : array-like
        Numeric data to calculate kurtosis
    min_samples : int
        Minimum number of samples required for calculation

    Returns:
    --------
    float
        Kurtosis value or 0.0 if calculation fails
    """
    try:
        # Convert to a plain Python list to avoid type issues with scipy.stats
        if hasattr(data, 'tolist'):
            data = data.tolist()

        # Check if we have enough data and it's all numeric
        if len(data) > min_samples and all(isinstance(x, (int, float)) for x in data):
            return float(stats.kurtosis(data))
        return 0.0
    except (TypeError, ValueError) as e:
        logger.warning(f"Error calculating kurtosis: {e}")
        return 0.0


def count_values_by_condition(data: pd.Series, condition_type: str, near_zero_threshold: float = 1e-10) -> int:
    """
    Safely count values in a dataset based on specified condition.

    Parameters:
    -----------
    data : pd.Series
        Numeric data to apply condition
    condition_type : str
        Type of condition: 'zero', 'positive', 'negative', 'near_zero'
    near_zero_threshold : float, optional
        Threshold for near-zero detection

    Returns:
    --------
    int
        Count of values meeting the condition
    """
    try:
        # Handle non-Series data
        if not isinstance(data, pd.Series):
            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            else:
                return 0

        # Apply appropriate condition with explicit conversion
        if condition_type == 'zero':
            mask = data == 0
            return int(mask.sum()) if hasattr(mask, 'sum') else 0
        elif condition_type == 'positive':
            mask = data > 0
            return int(mask.sum()) if hasattr(mask, 'sum') else 0
        elif condition_type == 'negative':
            mask = data < 0
            return int(mask.sum()) if hasattr(mask, 'sum') else 0
        elif condition_type == 'near_zero':
            mask = (data.abs() > 0) & (data.abs() < near_zero_threshold)
            return int(mask.sum()) if hasattr(mask, 'sum') else 0
        else:
            return 0
    except Exception as e:
        logger.warning(f"Error counting values by condition: {e}")
        return 0


def calculate_basic_stats(data: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic statistics for numeric data in a single pass.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data

    Returns:
    --------
    Dict[str, Any]
        Dictionary of basic statistics
    """
    try:
        # Use agg for efficient single-pass calculation
        stats_dict = data.agg(['min', 'max', 'mean', 'median', 'std', 'var', 'sum']).to_dict()

        # Ensure proper types
        result = {
            'min': float(stats_dict['min']),
            'max': float(stats_dict['max']),
            'mean': float(stats_dict['mean']),
            'median': float(stats_dict['median']),
            'std': float(stats_dict['std']),
            'variance': float(stats_dict['var']),
            'sum': float(stats_dict['sum'])
        }
        return result
    except Exception as e:
        logger.warning(f"Error calculating basic statistics: {e}")
        return {
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'variance': None,
            'sum': None
        }


def calculate_extended_stats(data: pd.Series, near_zero_threshold: float = 1e-10) -> Dict[str, Any]:
    """
    Calculate extended statistics for numeric data including percentiles and value counts.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    near_zero_threshold : float
        Threshold for near-zero detection

    Returns:
    --------
    Dict[str, Any]
        Dictionary of extended statistics
    """
    try:
        # Basic stats
        basic_stats = calculate_basic_stats(data)

        # Count
        valid_count = len(data)

        # Skewness and kurtosis
        skewness = calculate_skewness(data)
        kurtosis = calculate_kurtosis(data)

        # Value counts
        zero_count = count_values_by_condition(data, 'zero')
        near_zero_count = count_values_by_condition(data, 'near_zero', near_zero_threshold)
        negative_count = count_values_by_condition(data, 'negative')
        positive_count = count_values_by_condition(data, 'positive')

        # Combine results
        result = {
            'count': int(valid_count),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'zero_count': int(zero_count),
            'zero_percentage': float(round(zero_count / valid_count * 100, 2)) if valid_count > 0 else 0.0,
            'near_zero_count': int(near_zero_count),
            'near_zero_percentage': float(round(near_zero_count / valid_count * 100, 2)) if valid_count > 0 else 0.0,
            'negative_count': int(negative_count),
            'negative_percentage': float(round(negative_count / valid_count * 100, 2)) if valid_count > 0 else 0.0,
            'positive_count': int(positive_count),
            'positive_percentage': float(round(positive_count / valid_count * 100, 2)) if valid_count > 0 else 0.0,
        }

        # Update with basic stats
        result.update(basic_stats)

        return result
    except Exception as e:
        logger.warning(f"Error calculating extended statistics: {e}")
        return {}


def calculate_percentiles(data: pd.Series, percentiles: List[float] = None) -> Dict[str, float]:
    """
    Calculate percentiles for numeric data.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    percentiles : List[float], optional
        List of percentiles to calculate (0-100)

    Returns:
    --------
    Dict[str, float]
        Dictionary of percentiles
    """
    if percentiles is None:
        percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]

    try:
        percentile_values = data.quantile([p / 100 for p in percentiles])
        # Map percentiles to their values with proper indexing
        return {f'p{p}': float(percentile_values.loc[p / 100]) for p in percentiles}
    except Exception as e:
        logger.warning(f"Error calculating percentiles: {e}")
        # Return 0.0 instead of None to maintain the Dict[str, float] type
        return {f'p{p}': 0.0 for p in percentiles}


def calculate_histogram(data: pd.Series, bins: int = 10) -> Dict[str, Any]:
    """
    Calculate histogram data for numeric values.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    bins : int
        Number of bins for histogram

    Returns:
    --------
    Dict[str, Any]
        Dictionary with histogram data
    """
    try:
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)]

        return {
            'bins': bin_labels,
            'counts': [int(count) for count in hist],
            'bin_edges': [float(edge) for edge in bin_edges]
        }
    except Exception as e:
        logger.warning(f"Error calculating histogram: {e}")
        return {'bins': [], 'counts': [], 'bin_edges': []}


def detect_outliers(data: pd.Series, iqr_factor: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using the IQR method.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    iqr_factor : float
        Factor to multiply IQR by for determining outlier boundaries

    Returns:
    --------
    Dict[str, Any]
        Dictionary with outlier information
    """
    try:
        q1 = float(data.quantile(0.25))
        q3 = float(data.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - (iqr_factor * iqr)
        upper_bound = q3 + (iqr_factor * iqr)

        outliers_mask = (data < lower_bound) | (data > upper_bound)
        outlier_count = int(outliers_mask.sum()) if hasattr(outliers_mask, 'sum') else 0
        valid_count = len(data)

        result = {
            'iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'count': outlier_count,
            'percentage': float(round(outlier_count / valid_count * 100, 2)) if valid_count > 0 else 0.0
        }

        # Sample outliers if any exist
        if outlier_count > 0:
            try:
                outlier_sample = data[outliers_mask].sample(min(10, outlier_count)).tolist()
                result['sample'] = outlier_sample
            except:
                # Handle case where sampling fails
                outlier_values = data[outliers_mask].tolist()
                result['sample'] = outlier_values[:min(10, outlier_count)]

        return result
    except Exception as e:
        logger.warning(f"Error detecting outliers: {e}")
        return {
            'iqr': None,
            'lower_bound': None,
            'upper_bound': None,
            'count': 0,
            'percentage': 0.0
        }


def test_normality(data: pd.Series, test_method: str = 'all') -> Dict[str, Any]:
    """
    Test the normality of a numeric dataset.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    test_method : str
        Method to use for testing: 'shapiro', 'anderson', 'ks', or 'all'

    Returns:
    --------
    Dict[str, Any]
        Dictionary with normality test results
    """
    result: Dict[str, Any] = {}

    # Limit sample size for the tests (some tests don't work well with very large samples)
    max_sample_size = 5000
    if len(data) > max_sample_size:
        sample = data.sample(max_sample_size, random_state=42)
    else:
        sample = data

    try:
        # Shapiro-Wilk test
        if test_method in ('shapiro', 'all'):
            try:
                shapiro_stat, shapiro_p = stats.shapiro(sample)
                result['shapiro'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'normal': shapiro_p > 0.05
                }
            except Exception as e:
                logger.warning(f"Error performing Shapiro-Wilk test: {e}")
                result['shapiro'] = {'error': str(e)}

        # Anderson-Darling test
        if test_method in ('anderson', 'all'):
            try:
                anderson_result = stats.anderson(sample, dist='norm')
                result['anderson'] = {
                    'statistic': float(anderson_result.statistic),
                    'critical_values': [float(cv) for cv in anderson_result.critical_values],
                    'significance_levels': [float(sl) for sl in anderson_result.significance_level],
                    # If statistic > critical value at 5% significance, then not normal
                    'normal': anderson_result.statistic < anderson_result.critical_values[2]  # Index 2 is 5% level
                }
            except Exception as e:
                logger.warning(f"Error performing Anderson-Darling test: {e}")
                result['anderson'] = {'error': str(e)}

        # Kolmogorov-Smirnov test
        if test_method in ('ks', 'all'):
            try:
                # Standardize the data for comparison with standard normal
                data_std = (sample - sample.mean()) / sample.std()
                ks_stat, ks_p = stats.kstest(data_std, 'norm')
                result['ks'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'normal': ks_p > 0.05
                }
            except Exception as e:
                logger.warning(f"Error performing Kolmogorov-Smirnov test: {e}")
                result['ks'] = {'error': str(e)}

        # Calculate skewness and kurtosis again for convenience
        result['skewness'] = float(calculate_skewness(sample))
        result['kurtosis'] = float(calculate_kurtosis(sample))

        # Overall assessment
        normal_results = [v.get('normal', False) for k, v in result.items()
                          if k in ('shapiro', 'anderson', 'ks') and isinstance(v, dict) and 'normal' in v]

        if normal_results:
            result['is_normal'] = all(normal_results)
            result['normal_test_count'] = len(normal_results)
            result['normal_test_passed'] = sum(1 for x in normal_results if x)

        return result

    except Exception as e:
        logger.warning(f"Error testing normality: {e}")
        return {
            'error': str(e),
            'is_normal': False
        }


def analyze_numeric_chunk(chunk_df: pd.DataFrame, field_name: str,
                          near_zero_threshold: float = 1e-10) -> Dict[str, Any]:
    """
    Analyze a chunk of numeric data.

    Parameters:
    -----------
    chunk_df : pd.DataFrame
        DataFrame chunk to analyze
    field_name : str
        Name of the field to analyze
    near_zero_threshold : float
        Threshold for near-zero detection

    Returns:
    --------
    Dict[str, Any]
        Dictionary with analysis results
    """
    if field_name not in chunk_df.columns:
        return {}  # Return empty dictionary instead of None

    chunk_series = pd.to_numeric(chunk_df[field_name].dropna(), errors='coerce')
    if len(chunk_series) == 0:
        return {}  # Return empty dictionary instead of None

    # Basic stats using aggregation
    chunk_basic_stats = chunk_series.agg(['count', 'min', 'max', 'mean', 'median', 'std', 'var', 'sum']).to_dict()

    # Ensure all values are properly converted to appropriate types
    for key, value in chunk_basic_stats.items():
        if key == 'count':
            chunk_basic_stats[key] = int(value)
        else:
            chunk_basic_stats[key] = float(value)

    # Skewness and kurtosis
    chunk_basic_stats['skewness'] = calculate_skewness(chunk_series)
    chunk_basic_stats['kurtosis'] = calculate_kurtosis(chunk_series)

    # Count special values
    chunk_basic_stats['zero_count'] = count_values_by_condition(chunk_series, 'zero')
    chunk_basic_stats['negative_count'] = count_values_by_condition(chunk_series, 'negative')
    chunk_basic_stats['positive_count'] = count_values_by_condition(chunk_series, 'positive')
    chunk_basic_stats['near_zero_count'] = count_values_by_condition(chunk_series, 'near_zero', near_zero_threshold)

    return chunk_basic_stats


def combine_chunk_results(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine results from multiple chunks.

    Parameters:
    -----------
    chunk_results : List[Dict[str, Any]]
        List of chunk analysis results

    Returns:
    --------
    Dict[str, Any]
        Combined statistics
    """
    # Filter out None results
    filtered_results = []
    for r in chunk_results:
        if r is not None:
            filtered_results.append(r)

    # If no valid results, return default values
    if not filtered_results:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'variance': None,
            'sum': None,
            'skewness': None,
            'kurtosis': None,
            'zero_count': 0,
            'zero_percentage': 0.0,
            'near_zero_count': 0,
            'near_zero_percentage': 0.0,
            'negative_count': 0,
            'negative_percentage': 0.0,
            'positive_count': 0,
            'positive_percentage': 0.0
        }

    # Calculate total count
    total_count = 0
    for r in filtered_results:
        count = r.get('count', 0)
        if count is not None:
            total_count += count

    # Initialize result dictionary
    result_dict = {}
    result_dict['count'] = total_count

    # Handle min values
    min_values = []
    for r in filtered_results:
        min_val = r.get('min')
        if min_val is not None:
            min_values.append(min_val)

    if min_values:
        min_value = min(min_values)
        result_dict['min'] = min_value
    else:
        result_dict['min'] = None

    # Handle max values
    max_values = []
    for r in filtered_results:
        max_val = r.get('max')
        if max_val is not None:
            max_values.append(max_val)

    if max_values:
        max_value = max(max_values)
        result_dict['max'] = max_value
    else:
        result_dict['max'] = None

    # Handle mean
    if total_count > 0:
        mean_sum = 0.0
        for r in filtered_results:
            mean_val = r.get('mean')
            count_val = r.get('count')
            if mean_val is not None and count_val is not None:
                mean_sum += mean_val * count_val
        result_dict['mean'] = mean_sum / total_count
    else:
        result_dict['mean'] = None

    # Handle median (approximation)
    if filtered_results and len(filtered_results) > 0:
        middle_idx = len(filtered_results) // 2
        median_val = filtered_results[middle_idx].get('median')
        result_dict['median'] = median_val
    else:
        result_dict['median'] = None

    # Handle std
    if total_count > 0:
        std_sum = 0.0
        for r in filtered_results:
            std_val = r.get('std')
            count_val = r.get('count')
            if std_val is not None and count_val is not None:
                std_sum += std_val * count_val
        result_dict['std'] = std_sum / total_count
    else:
        result_dict['std'] = None

    # Handle variance
    if total_count > 0:
        var_sum = 0.0
        for r in filtered_results:
            var_val = r.get('var')
            count_val = r.get('count')
            if var_val is not None and count_val is not None:
                var_sum += var_val * count_val
        result_dict['variance'] = var_sum / total_count
    else:
        result_dict['variance'] = None

    # Handle sum
    sum_total = 0.0
    sum_available = False
    for r in filtered_results:
        sum_val = r.get('sum')
        if sum_val is not None:
            sum_total += sum_val
            sum_available = True

    result_dict['sum'] = sum_total if sum_available else None

    # Handle skewness
    if total_count > 0:
        skew_sum = 0.0
        for r in filtered_results:
            skew_val = r.get('skewness')
            count_val = r.get('count')
            if skew_val is not None and count_val is not None:
                skew_sum += skew_val * count_val
        result_dict['skewness'] = skew_sum / total_count
    else:
        result_dict['skewness'] = None

    # Handle kurtosis
    if total_count > 0:
        kurt_sum = 0.0
        for r in filtered_results:
            kurt_val = r.get('kurtosis')
            count_val = r.get('count')
            if kurt_val is not None and count_val is not None:
                kurt_sum += kurt_val * count_val
        result_dict['kurtosis'] = kurt_sum / total_count
    else:
        result_dict['kurtosis'] = None

    # Handle count values
    zero_count = 0
    negative_count = 0
    positive_count = 0
    near_zero_count = 0

    for r in filtered_results:
        zero_val = r.get('zero_count')
        negative_val = r.get('negative_count')
        positive_val = r.get('positive_count')
        near_zero_val = r.get('near_zero_count')

        if zero_val is not None:
            zero_count += zero_val
        if negative_val is not None:
            negative_count += negative_val
        if positive_val is not None:
            positive_count += positive_val
        if near_zero_val is not None:
            near_zero_count += near_zero_val

    result_dict['zero_count'] = zero_count
    result_dict['negative_count'] = negative_count
    result_dict['positive_count'] = positive_count
    result_dict['near_zero_count'] = near_zero_count

    # Calculate percentages
    if total_count > 0:
        result_dict['zero_percentage'] = round((zero_count / total_count) * 100, 2)
        result_dict['negative_percentage'] = round((negative_count / total_count) * 100, 2)
        result_dict['positive_percentage'] = round((positive_count / total_count) * 100, 2)
        result_dict['near_zero_percentage'] = round((near_zero_count / total_count) * 100, 2)
    else:
        result_dict['zero_percentage'] = 0.0
        result_dict['negative_percentage'] = 0.0
        result_dict['positive_percentage'] = 0.0
        result_dict['near_zero_percentage'] = 0.0

    return result_dict


def create_empty_stats() -> Dict[str, Any]:
    """
    Create empty statistics dictionary for when there's no valid data.

    Returns:
    --------
    Dict[str, Any]
        Empty statistics dictionary
    """
    return {
        'count': 0,
        'min': None,
        'max': None,
        'mean': None,
        'median': None,
        'std': None,
        'variance': None,
        'sum': None,
        'skewness': None,
        'kurtosis': None,
        'percentiles': {},
        'histogram': {'bins': [], 'counts': [], 'bin_edges': []},
        'zero_count': 0,
        'zero_percentage': 0,
        'near_zero_count': 0,
        'near_zero_percentage': 0,
        'negative_count': 0,
        'negative_percentage': 0,
        'positive_count': 0,
        'positive_percentage': 0,
        'normality': {
            'is_normal': False,
            'tests': {}
        }
    }


def prepare_numeric_data(df: pd.DataFrame, field_name: str) -> Tuple[pd.Series, int, int]:
    """
    Prepare numeric data for analysis by handling nulls and converting types.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field to prepare

    Returns:
    --------
    Tuple[pd.Series, int, int]
        A tuple containing:
        - The prepared numeric series with valid data only
        - Count of null values
        - Count of non-null values
    """
    # Count nulls
    null_count = df[field_name].isna().sum()
    non_null_count = len(df) - null_count

    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(df[field_name], errors='coerce')

    # Drop NaN values for valid data
    valid_data = numeric_series.dropna()

    return valid_data, null_count, non_null_count


def handle_large_dataframe(df: pd.DataFrame, field_name: str,
                           analyze_func, chunk_size: int = 10000,
                           **kwargs) -> Dict[str, Any]:
    """
    Process a large DataFrame in chunks to analyze numeric data.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    field_name : str
        The name of the field to analyze
    analyze_func : callable
        Function to analyze each chunk
    chunk_size : int
        Size of chunks to process
    **kwargs :
        Additional parameters to pass to the analyze function

    Returns:
    --------
    Dict[str, Any]
        Combined results of the analysis
    """
    # Split DataFrame into chunks
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    chunk_results = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]

        # Analyze chunk
        chunk_result = analyze_func(chunk_df, field_name, **kwargs)
        if chunk_result:  # Only add non-empty results
            chunk_results.append(chunk_result)

    # Combine results from all chunks
    return combine_chunk_results(chunk_results)