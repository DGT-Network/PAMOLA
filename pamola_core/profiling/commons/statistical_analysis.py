"""
Statistical analysis utilities for the HHR anonymization project.

This module provides functions for statistical analysis, including normality testing,
outlier detection, and distribution analysis that can be applied to various types of data.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from pamola_core.utils.logging import configure_logging

# Configure logger using the custom logging utility
logger = configure_logging(level=logging.INFO)


def calculate_distribution_metrics(data: pd.Series) -> Dict[str, float]:
    """
    Calculate various distribution metrics for numeric data.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data

    Returns:
    --------
    Dict[str, float]
        Dictionary with distribution metrics
    """
    try:
        from core.profiling.commons.numeric_utils import calculate_skewness, calculate_kurtosis

        result = {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'variance': float(data.var()),
            'skewness': calculate_skewness(data),
            'kurtosis': calculate_kurtosis(data),
            'range': float(data.max() - data.min())
        }

        # Calculate coefficient of variation
        if result['mean'] != 0:
            result['cv'] = (result['std'] / abs(result['mean'])) * 100
        else:
            result['cv'] = np.nan

        return result
    except Exception as e:
        logger.warning(f"Error calculating distribution metrics: {e}")
        return {}


def test_normality(
        data: pd.Series,
        test_method: str = 'all',
        sample_limit: int = 5000
) -> Dict[str, Any]:
    """
    Test the normality of a dataset using various methods.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    test_method : str
        Method to use for testing: 'shapiro', 'anderson', 'ks', or 'all'
    sample_limit : int
        Maximum sample size for tests (some tests don't work well with very large samples)

    Returns:
    --------
    Dict[str, Any]
        Dictionary with normality test results
    """
    result: Dict[str, Any] = {}

    # Limit sample size for the tests
    if len(data) > sample_limit:
        sample = data.sample(sample_limit, random_state=42)
    else:
        sample = data

    try:
        # Import needed functions
        from core.profiling.commons.numeric_utils import calculate_skewness, calculate_kurtosis

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

        # Calculate skewness and kurtosis
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


def detect_outliers(
        data: pd.Series,
        method: str = 'iqr',
        **params
) -> Dict[str, Any]:
    """
    Detect outliers in numeric data using specified method.

    Parameters:
    -----------
    data : pd.Series
        Series of numeric data
    method : str
        Method to use for detection: 'iqr', 'zscore', 'modified_zscore'
    **params : dict
        Additional parameters specific to each method:
        - For 'iqr': iqr_factor (default: 1.5)
        - For 'zscore': threshold (default: 3.0)
        - For 'modified_zscore': threshold (default: 3.5)

    Returns:
    --------
    Dict[str, Any]
        Dictionary with outlier information
    """
    if method == 'iqr':
        return detect_outliers_iqr(data, **params)
    elif method == 'zscore':
        return detect_outliers_zscore(data, **params)
    elif method == 'modified_zscore':
        return detect_outliers_modified_zscore(data, **params)
    else:
        logger.warning(f"Unknown outlier detection method: {method}. Using IQR method.")
        return detect_outliers_iqr(data, **params)


def detect_outliers_iqr(
        data: pd.Series,
        iqr_factor: float = 1.5
) -> Dict[str, Any]:
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
            'method': 'iqr',
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
        logger.warning(f"Error detecting outliers using IQR method: {e}")
        return {
            'method': 'iqr',
            'iqr': None,
            'lower_bound': None,
            'upper_bound': None,
            'count': 0,
            'percentage': 0.0
        }

        def detect_outliers_zscore(
                data: pd.Series,
                threshold: float = 3.0
        ) -> Dict[str, Any]:
            """
            Detect outliers using the Z-score method.

            Parameters:
            -----------
            data : pd.Series
                Series of numeric data
            threshold : float
                Z-score threshold for outlier detection

            Returns:
            --------
            Dict[str, Any]
                Dictionary with outlier information
            """
            try:
                # Calculate z-scores
                mean = data.mean()
                std = data.std()

                if std == 0:
                    return {
                        'method': 'zscore',
                        'threshold': threshold,
                        'count': 0,
                        'percentage': 0.0,
                        'error': 'Standard deviation is zero, cannot calculate z-scores'
                    }

                z_scores = (data - mean) / std

                # Identify outliers
                outliers_mask = abs(z_scores) > threshold
                outlier_count = int(outliers_mask.sum()) if hasattr(outliers_mask, 'sum') else 0
                valid_count = len(data)

                result = {
                    'method': 'zscore',
                    'threshold': float(threshold),
                    'count': outlier_count,
                    'percentage': float(round(outlier_count / valid_count * 100, 2)) if valid_count > 0 else 0.0
                }

                # Sample outliers if any exist
                if outlier_count > 0:
                    try:
                        outlier_indices = outliers_mask[outliers_mask].index
                        outlier_values = data.loc[outlier_indices]
                        outlier_zscores = z_scores.loc[outlier_indices]

                        # Combine values and z-scores
                        outlier_sample = []
                        for i in range(min(10, len(outlier_indices))):
                            idx = outlier_indices[i]
                            outlier_sample.append({
                                'value': float(outlier_values.iloc[i]),
                                'z_score': float(outlier_zscores.iloc[i])
                            })

                        result['sample'] = outlier_sample
                    except:
                        # Handle case where sampling fails
                        result['sample'] = []

                return result
            except Exception as e:
                logger.warning(f"Error detecting outliers using Z-score method: {e}")
                return {
                    'method': 'zscore',
                    'threshold': threshold,
                    'count': 0,
                    'percentage': 0.0,
                    'error': str(e)
                }

        def detect_outliers_modified_zscore(
                data: pd.Series,
                threshold: float = 3.5
        ) -> Dict[str, Any]:
            """
            Detect outliers using the modified Z-score method.

            This method is more robust to outliers in the calculation of the score itself
            by using median and MAD instead of mean and standard deviation.

            Parameters:
            -----------
            data : pd.Series
                Series of numeric data
            threshold : float
                Modified Z-score threshold for outlier detection

            Returns:
            --------
            Dict[str, Any]
                Dictionary with outlier information
            """
            try:
                # Calculate modified z-scores
                median = data.median()
                # Median Absolute Deviation
                mad = (data - median).abs().median() * 1.4826  # Consistency factor for normal distribution

                if mad == 0:
                    return {
                        'method': 'modified_zscore',
                        'threshold': threshold,
                        'count': 0,
                        'percentage': 0.0,
                        'error': 'Median Absolute Deviation is zero, cannot calculate modified z-scores'
                    }

                modified_z_scores = 0.6745 * (data - median) / mad

                # Identify outliers
                outliers_mask = abs(modified_z_scores) > threshold
                outlier_count = int(outliers_mask.sum()) if hasattr(outliers_mask, 'sum') else 0
                valid_count = len(data)

                result = {
                    'method': 'modified_zscore',
                    'threshold': float(threshold),
                    'median': float(median),
                    'mad': float(mad),
                    'count': outlier_count,
                    'percentage': float(round(outlier_count / valid_count * 100, 2)) if valid_count > 0 else 0.0
                }

                # Sample outliers if any exist
                if outlier_count > 0:
                    try:
                        outlier_indices = outliers_mask[outliers_mask].index
                        outlier_values = data.loc[outlier_indices]
                        outlier_zscores = modified_z_scores.loc[outlier_indices]

                        # Combine values and modified z-scores
                        outlier_sample = []
                        for i in range(min(10, len(outlier_indices))):
                            idx = outlier_indices[i]
                            outlier_sample.append({
                                'value': float(outlier_values.iloc[i]),
                                'modified_z_score': float(outlier_zscores.iloc[i])
                            })

                        result['sample'] = outlier_sample
                    except:
                        # Handle case where sampling fails
                        result['sample'] = []

                return result
            except Exception as e:
                logger.warning(f"Error detecting outliers using modified Z-score method: {e}")
                return {
                    'method': 'modified_zscore',
                    'threshold': threshold,
                    'count': 0,
                    'percentage': 0.0,
                    'error': str(e)
                }