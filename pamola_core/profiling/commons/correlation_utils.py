"""
Utility functions for analyzing correlations between fields in the project.

This module provides pure analytical functions for correlation calculation,
separate from operation logic, focusing on metrics calculation, relationship
detection, and data preparation. It supports various correlation methods for
different data types (numerical, categorical, mixed).

Core functions:
- calculate_correlation: Calculate correlation between two fields
- calculate_cramers_v: Calculate Cramer's V for categorical-categorical correlations
- calculate_correlation_ratio: Calculate correlation ratio for categorical-numeric correlations
- interpret_correlation: Provide textual interpretation of correlation coefficients
- create_correlation_matrix: Calculate correlation matrix for multiple fields
"""

import logging
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr, pearsonr, spearmanr

# Configure logger
logger = logging.getLogger(__name__)


def analyze_correlation(
        df: pd.DataFrame,
        field1: str,
        field2: str,
        mvf_parser: Optional[str] = None,
        null_handling: str = 'drop',
        method: Optional[str] = None,
        task_logger: Optional[logging.Logger] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    Analyze correlation between two fields in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field
    mvf_parser : str, optional
        String lambda to parse multi-valued fields (MVF) if applicable.
    null_handling : str, optional
        How to handle null values:
        - 'drop': remove rows with nulls
        - 'fill': fill nulls with a specified value (default NaN)
        - 'pairwise': use pairwise deletion for correlation calculation
    method : str, optional
        Correlation method to use. If None, automatically selected based on data types.
        Options: 'pearson', 'spearman', 'cramers_v', 'correlation_ratio', 'point_biserial'
    **kwargs : dict

    Returns:
    --------
    Dict[str, Any]
        Results of the correlation analysis including:
        - field1: name of first field
        - field2: name of second field
        - method: correlation method used
        - correlation_coefficient: calculated correlation value
        - p_value: statistical significance (if applicable)
        - interpretation: text interpretation of correlation
        - sample_size: number of samples used
        - null_stats: information about null values
        - plot_data: data for visualization (if requested)
    """
    if task_logger is not None:
        logger = task_logger

    # Validate fields
    if field1 not in df.columns or field2 not in df.columns:
        error_message = f"Field not found: "
        if field1 not in df.columns:
            error_message += field1
        if field2 not in df.columns:
            error_message += f" {field2}" if field1 not in df.columns else field2
        return {'error': error_message}

    # Handle MVF fields if parser provided
    df_clean = df[[field1, field2]].copy()
    if mvf_parser is not None:
        mvf_lamdable = eval(mvf_parser) if isinstance(mvf_parser, str) else mvf_parser
        if callable(mvf_lamdable):
            df_clean = prepare_mvf_fields(df_clean, field1, field2, mvf_lamdable)

    # Determine field types
    is_numeric1 = pd.api.types.is_numeric_dtype(df_clean[field1])
    is_numeric2 = pd.api.types.is_numeric_dtype(df_clean[field2])

    # Handle null values
    df_clean, null_stats = handle_null_values(df_clean, null_handling)

    # If no data left after handling nulls, return error
    if len(df_clean) == 0:
        return {
            'error': "No data available after handling null values",
            'null_stats': null_stats
        }

    # Calculate correlation based on field types
    correlation_info = calculate_correlation(
        df_clean, field1, field2, method, logger
    )

    # Prepare basic statistics
    stats = {
        'field1': field1,
        'field2': field2,
        'field1_type': 'numeric' if is_numeric1 else 'categorical',
        'field2_type': 'numeric' if is_numeric2 else 'categorical',
        'method': correlation_info['method'],
        'correlation_coefficient': correlation_info['coefficient'],
        'p_value': correlation_info.get('p_value'),
        'sample_size': len(df_clean),
        'interpretation': interpret_correlation(
            correlation_info['coefficient'], correlation_info['method']
        ),
        'null_stats': null_stats
    }

    # Include plot data if requested
    plot_data = prepare_plot_data(df_clean, field1, field2, is_numeric1, is_numeric2)
    stats['plot_data'] = plot_data

    return stats


def analyze_correlation_matrix(
        df: pd.DataFrame,
        fields: List[str],
        methods: Optional[Dict[str, str]] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    Create a correlation matrix for multiple fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    fields : List[str]
        List of field names to include in the correlation matrix
    methods : Dict[str, str], optional
        Dictionary mapping field pairs to correlation methods
    **kwargs : dict
        Additional parameters:
        - mvf_parser: function to parse multi-valued fields
        - null_handling: how to handle null values ('drop', 'fill', 'pairwise')
        - min_threshold: minimum correlation threshold for significant correlations
        - max_fields: maximum number of fields to include

    Returns:
    --------
    Dict[str, Any]
        Dictionary with correlation matrix and supporting information
    """
    # Check if fields exist in DataFrame
    missing_fields = [field for field in fields if field not in df.columns]
    if missing_fields:
        return {'error': f"Fields not found: {', '.join(missing_fields)}"}

    # Extract parameters
    mvf_parser = kwargs.get('mvf_parser', None)
    null_handling = kwargs.get('null_handling', 'drop')
    min_threshold = kwargs.get('min_threshold', 0.3)
    max_fields = kwargs.get('max_fields', 20)

    # Limit number of fields if necessary
    if len(fields) > max_fields:
        logger.warning(f"Too many fields ({len(fields)}) for correlation matrix. "
                       f"Limiting to {max_fields} fields.")
        fields = fields[:max_fields]

    # Initialize correlation matrix and methods dictionary
    corr_matrix = pd.DataFrame(index=fields, columns=fields)
    result_methods = {}
    p_values = {}

    # Fill correlation matrix
    for i, field1 in enumerate(fields):
        for j, field2 in enumerate(fields):
            # Same field, correlation = 1
            if i == j:
                corr_matrix.loc[field1, field2] = 1.0
                result_methods[f"{field1}_{field2}"] = "self"
            # Only calculate for upper triangle, copy for lower triangle
            elif i < j:
                # Get correlation method if specified
                method = None
                if methods:
                    method = methods.get(f"{field1}_{field2}")

                try:
                    # Calculate correlation
                    result = analyze_correlation(
                        df,
                        field1,
                        field2,
                        method=method,
                        mvf_parser=mvf_parser,
                        null_handling=null_handling,
                        include_plots=False
                    )

                    # Extract results
                    if 'error' in result:
                        corr_matrix.loc[field1, field2] = np.nan
                        result_methods[f"{field1}_{field2}"] = "error"
                        logger.warning(f"Error analyzing correlation between {field1} and {field2}: "
                                       f"{result['error']}")
                    else:
                        corr_value = result['correlation_coefficient']
                        corr_matrix.loc[field1, field2] = corr_value
                        corr_matrix.loc[field2, field1] = corr_value  # Mirror value
                        result_methods[f"{field1}_{field2}"] = result['method']
                        result_methods[f"{field2}_{field1}"] = result['method']
                        if 'p_value' in result and result['p_value'] is not None:
                            p_values[f"{field1}_{field2}"] = result['p_value']
                            p_values[f"{field2}_{field1}"] = result['p_value']
                except Exception as e:
                    logger.error(f"Error calculating correlation between {field1} and {field2}: {e}")
                    corr_matrix.loc[field1, field2] = np.nan
                    corr_matrix.loc[field2, field1] = np.nan
                    result_methods[f"{field1}_{field2}"] = "error"
                    result_methods[f"{field2}_{field1}"] = "error"

    # Find significant correlations
    significant_correlations = find_significant_correlations(
        corr_matrix, min_threshold, p_values
    )

    # Create results dictionary
    matrix_dict = corr_matrix.to_dict()

    # The default converter to dict doesn't handle NaN values correctly in some contexts
    # Convert any NaN values to None for better JSON serialization
    for row in matrix_dict:
        for col in matrix_dict[row]:
            if isinstance(matrix_dict[row][col], float) and np.isnan(matrix_dict[row][col]):
                matrix_dict[row][col] = None

    results = {
        'correlation_matrix': matrix_dict,
        'methods': result_methods,
        'significant_correlations': significant_correlations,
        'p_values': p_values,
        'min_threshold': min_threshold,
        'fields_analyzed': len(fields)
    }

    return results


def detect_correlation_type(df: pd.DataFrame, field1: str, field2: str) -> str:
    """
    Automatically detect the most appropriate correlation method for two fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field

    Returns:
    --------
    str
        Recommended correlation method
    """
    # Determine field types
    is_numeric1 = pd.api.types.is_numeric_dtype(df[field1])
    is_numeric2 = pd.api.types.is_numeric_dtype(df[field2])

    # Check if binary (only 2 unique values after removing nulls)
    is_binary1 = df[field1].nunique() == 2
    is_binary2 = df[field2].nunique() == 2

    # Select method based on types
    if is_numeric1 and is_numeric2:
        return 'pearson'
    elif (not is_numeric1) and (not is_numeric2):
        return 'cramers_v'
    elif is_numeric1 and not is_numeric2:
        if is_binary2:
            return 'point_biserial'
        else:
            return 'correlation_ratio'
    elif not is_numeric1 and is_numeric2:
        if is_binary1:
            return 'point_biserial'
        else:
            return 'correlation_ratio'

    # Default fallback
    return 'pearson'


def calculate_correlation(df: pd.DataFrame,
                          field1: str,
                          field2: str,
                          method: Optional[str] = None,
                          task_logger: Optional[logging.Logger] = None
                          ) -> Dict[str, Any]:
    """
    Calculate correlation between two fields based on their types.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field
    method : str, optional
        Override automatic method selection

    Returns:
    --------
    Dict[str, Any]
        Correlation information including coefficient, method, and p-value if applicable
    """
    if task_logger is not None:
        logger = task_logger
    # Determine field types
    is_numeric1 = pd.api.types.is_numeric_dtype(df[field1])
    is_numeric2 = pd.api.types.is_numeric_dtype(df[field2])

    # Select method based on types if not specified
    if method is None:
        method = detect_correlation_type(df, field1, field2)

    # Calculate correlation based on selected method
    correlation_info = {
        'method': method,
        'coefficient': None,
        'p_value': None
    }

    try:
        if method == 'pearson':
            if not (is_numeric1 and is_numeric2):
                logger.error("Pearson correlation requires both fields to be numeric.")
            
            x = pd.to_numeric(df[field1], errors='coerce')
            y = pd.to_numeric(df[field2], errors='coerce')

            coef, p_value = pearsonr(x, y)
            correlation_info['coefficient'] = coef
            correlation_info['p_value'] = p_value

        elif method == 'spearman':
            if not (is_numeric1 and is_numeric2):
                logger.error("Spearman correlation requires both fields to be numeric.")
            
            x = pd.to_numeric(df[field1], errors='coerce')
            y = pd.to_numeric(df[field2], errors='coerce')

            coef, p_value = spearmanr(x, y)
            correlation_info['coefficient'] = coef
            correlation_info['p_value'] = p_value

        elif method == 'cramers_v':
            coef = calculate_cramers_v(df[field1], df[field2])
            correlation_info['coefficient'] = coef

        elif method == 'point_biserial':
            if is_numeric1 and not is_numeric2:
                coef, p_value = calculate_point_biserial(df[field2], pd.to_numeric(df[field1], errors='coerce'))
            else:
                coef, p_value = calculate_point_biserial(df[field1], pd.to_numeric(df[field2], errors='coerce'))
            correlation_info['coefficient'] = coef
            correlation_info['p_value'] = p_value

        elif method == 'correlation_ratio':
            if is_numeric1 and not is_numeric2:
                coef = calculate_correlation_ratio(df[field2], df[field1])
            else:
                coef = calculate_correlation_ratio(df[field1], df[field2])
            correlation_info['coefficient'] = coef

        else:
            # Fallback to a simple method
            logger.warning(f"Unknown correlation method: {method}. Using default.")
            if is_numeric1 and is_numeric2:
                x = pd.to_numeric(df[field1], errors='coerce')
                y = pd.to_numeric(df[field2], errors='coerce')

                coef = x.corr(y)
                correlation_info['method'] = 'pearson'
                correlation_info['coefficient'] = coef
            else:
                correlation_info['method'] = 'unknown'
                correlation_info['coefficient'] = 0.0

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        correlation_info['method'] = 'error'
        correlation_info['coefficient'] = 0.0
        correlation_info['error'] = str(e)
    
    # Clean up NaN values
    if correlation_info['coefficient'] is not None and np.isnan(correlation_info['coefficient']):
        correlation_info['coefficient'] = 0.0
    if correlation_info['p_value'] is not None and np.isnan(correlation_info['p_value']):
        correlation_info['p_value'] = 0.0
    if correlation_info['coefficient'] is not None:
        correlation_info['coefficient'] = float(correlation_info['coefficient'])

    return correlation_info


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramer's V statistic for categorical variables.

    Parameters:
    -----------
    x : pd.Series
        First categorical variable
    y : pd.Series
        Second categorical variable

    Returns:
    --------
    float
        Cramer's V statistic (0 to 1)
    """
    # Create contingency table
    contingency = pd.crosstab(x, y)

    # If one variable has only one category, correlation is undefined
    if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
        return 0.0

    # Calculate chi-square statistic
    chi2, _, _, _ = chi2_contingency(contingency)

    # Calculate Cramer's V
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    # Avoid division by zero
    if min((kcorr - 1), (rcorr - 1)) == 0:
        return 0.0

    # Calculate Cramer's V
    v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return float(v)


def calculate_correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Calculate the correlation ratio (eta) between a categorical and a numeric variable.

    Parameters:
    -----------
    categories : pd.Series
        Categorical variable
    values : pd.Series
        Numeric variable

    Returns:
    --------
    float
        Correlation ratio (0 to 1)
    """
    # Group numeric values by category
    groups = values.groupby(categories)

    # Calculate total mean and sum of squares
    overall_mean = values.mean()
    overall_ss = ((values - overall_mean) ** 2).sum()

    # If overall sum of squares is zero, return 0
    if overall_ss == 0:
        return 0.0

    # Calculate sum of squares between groups
    between_ss = sum([(group.mean() - overall_mean) ** 2 * len(group) for _, group in groups])

    # Calculate correlation ratio
    correlation_ratio = np.sqrt(between_ss / overall_ss)
    return float(correlation_ratio)


def calculate_point_biserial(binary_var: pd.Series, numeric_var: pd.Series) -> Tuple[float, float]:
    """
    Calculate point-biserial correlation between a binary and a numeric variable.

    Parameters:
    -----------
    binary_var : pd.Series
        Binary variable (will be converted to 0/1)
    numeric_var : pd.Series
        Numeric variable

    Returns:
    --------
    Tuple[float, float]
        Correlation coefficient and p-value
    """
    # Convert binary variable to 0/1
    binary_values = pd.factorize(binary_var)[0]

    # Calculate point-biserial correlation
    coef, p_value = pointbiserialr(binary_values, numeric_var)
    return float(coef), float(p_value)


def interpret_correlation(correlation_value: float, method: str) -> str:
    """
    Interpret the correlation coefficient.

    Parameters:
    -----------
    correlation_value : float
        Correlation coefficient
    method : str
        Correlation method used

    Returns:
    --------
    str
        Interpretation of the correlation
    """
    # Take absolute value for strength interpretation
    abs_corr = abs(correlation_value)

    # Different scales depending on method
    if method in ['cramers_v', 'correlation_ratio']:
        # Scale for non-directional measures
        if abs_corr < 0.1:
            strength = "Negligible association"
        elif abs_corr < 0.2:
            strength = "Weak association"
        elif abs_corr < 0.4:
            strength = "Moderate association"
        elif abs_corr < 0.6:
            strength = "Relatively strong association"
        elif abs_corr < 0.8:
            strength = "Strong association"
        else:
            strength = "Very strong association"

        # These methods don't have direction
        return strength
    else:
        # Scale for directional measures
        if abs_corr < 0.1:
            strength = "Negligible correlation"
        elif abs_corr < 0.3:
            strength = "Weak correlation"
        elif abs_corr < 0.5:
            strength = "Moderate correlation"
        elif abs_corr < 0.7:
            strength = "Strong correlation"
        elif abs_corr < 0.9:
            strength = "Very strong correlation"
        else:
            strength = "Near perfect correlation"

        # Add direction
        direction = "positive" if correlation_value >= 0 else "negative"
        return f"{strength} ({direction})"


def handle_null_values(df: pd.DataFrame, method: str = 'drop') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle null values in DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    method : str
        Method to handle nulls: 'drop', 'fill', or 'pairwise'

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Processed DataFrame and null statistics
    """
    # Calculate null statistics
    null_stats = {
        'total_rows': len(df),
        'null_rows': df.isna().any(axis=1).sum(),
        'null_percentage': round((df.isna().any(axis=1).sum() / len(df)) * 100, 2) if len(df) > 0 else 0,
        'field_nulls': {col: int(df[col].isna().sum()) for col in df.columns}
    }

    # Handle nulls based on method
    if method == 'drop':
        # Drop rows with any null values
        df_clean = df.dropna()
    elif method == 'fill':
        # Fill nulls with appropriate values based on data type
        df_clean = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_clean[col] = df[col].fillna(0)
            else:
                df_clean[col] = df[col].fillna('')
    elif method == 'pairwise':
        # Keep rows with at least one non-null value (for correlation matrix)
        df_clean = df.copy()
    else:
        # Default to drop
        logger.warning(f"Unknown null handling method: {method}. Defaulting to 'drop'.")
        df_clean = df.dropna()

    # Update null statistics
    null_stats['rows_after_handling'] = len(df_clean)
    null_stats['rows_removed'] = len(df) - len(df_clean)

    return df_clean, null_stats


def find_significant_correlations(corr_matrix: pd.DataFrame,
                                  threshold: float = 0.3,
                                  p_values: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Find significant correlations in a correlation matrix.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    threshold : float
        Minimum absolute correlation threshold
    p_values : Dict[str, float], optional
        Dictionary of p-values

    Returns:
    --------
    List[Dict[str, Any]]
        List of significant correlations
    """
    significant = []

    # Iterate through upper triangle of correlation matrix
    for i, field1 in enumerate(corr_matrix.index):
        for j, field2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr_value = corr_matrix.loc[field1, field2]

                # Check if correlation is significant
                if pd.notna(corr_value) and abs(corr_value) >= threshold:
                    result = {
                        'field1': field1,
                        'field2': field2,
                        'correlation': float(corr_value),
                        'absolute_correlation': float(abs(corr_value))
                    }

                    # Add p-value if available
                    if p_values and f"{field1}_{field2}" in p_values:
                        result['p_value'] = p_values[f"{field1}_{field2}"]
                        # Consider statistical significance
                        result['statistically_significant'] = result['p_value'] < 0.05

                    significant.append(result)

    # Sort by absolute correlation value
    significant.sort(key=lambda x: x['absolute_correlation'], reverse=True)
    return significant


def prepare_mvf_fields(df: pd.DataFrame,
                       field1: str,
                       field2: str,
                       mvf_parser: Callable) -> pd.DataFrame:
    """
    Prepare multi-valued fields for correlation analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field
    mvf_parser : Callable
        Function to parse multi-valued fields

    Returns:
    --------
    pd.DataFrame
        DataFrame with processed MVF fields
    """
    df_result = df.copy()

    try:
        # Check if fields might be MVF by checking for string type
        print(f"Preparing MVF fields: {field1}, {field2}")
        print(f"Field types: {df[field1].dtype}, {df[field2].dtype}")
        
        for field in [field1, field2]:
            if (
                (
                    pd.api.types.is_string_dtype(df[field])
                    or pd.api.types.is_object_dtype(df[field])
                )
                and not pd.api.types.is_numeric_dtype(df[field])
            ):
                # Try to parse as MVF and convert to string representation
                try:
                    def mvf_to_string(value):
                        if pd.isna(value):
                            return np.nan
                        parsed_values = mvf_parser(value)
                        return ", ".join(parsed_values) if parsed_values else ""

                    df_result[f"{field}_parsed"] = df[field].apply(mvf_to_string)
                    # Replace original field with parsed version
                    df_result[field] = df_result[f"{field}_parsed"]
                    df_result.drop(columns=[f"{field}_parsed"], inplace=True)
                except Exception as e:
                    logger.warning(f"Failed to parse field {field} as MVF: {e}")
    except Exception as e:
        logger.error(f"Error preparing MVF fields: {e}")

    return df_result


def prepare_plot_data(df: pd.DataFrame,
                      field1: str,
                      field2: str,
                      is_numeric1: bool,
                      is_numeric2: bool) -> Dict[str, Any]:
    """
    Prepare data for correlation plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field
    is_numeric1 : bool
        Whether field1 is numeric
    is_numeric2 : bool
        Whether field2 is numeric

    Returns:
    --------
    Dict[str, Any]
        Plot data configuration
    """
    # Determine plot type based on field types
    if is_numeric1 and is_numeric2:
        plot_type = "scatter"
        plot_data = {
            'type': plot_type,
            'x_values': df[field1].tolist(),
            'y_values': df[field2].tolist(),
            'x_label': field1,
            'y_label': field2
        }
    elif (is_numeric1 and not is_numeric2) or (not is_numeric1 and is_numeric2):
        # For numeric + categorical: boxplot
        plot_type = "boxplot"
        num_field = field1 if is_numeric1 else field2
        cat_field = field2 if is_numeric1 else field1

        # Limit to top categories
        top_categories = df[cat_field].value_counts().head(10).index.tolist()
        df_plot = df[df[cat_field].isin(top_categories)]

        plot_data = {
            'type': plot_type,
            'categories': df_plot[cat_field].tolist(),
            'values': df_plot[num_field].tolist(),
            'x_label': cat_field,
            'y_label': num_field,
            'top_categories': top_categories
        }
    else:
        # For categorical + categorical: heatmap
        plot_type = "heatmap"

        # Limit to top categories
        top_cat1 = df[field1].value_counts().head(10).index.tolist()
        top_cat2 = df[field2].value_counts().head(10).index.tolist()

        df_plot = df[df[field1].isin(top_cat1) & df[field2].isin(top_cat2)]

        # Create contingency table
        contingency = pd.crosstab(df_plot[field1], df_plot[field2], normalize='index')

        plot_data = {
            'type': plot_type,
            'matrix': contingency.to_dict(),
            'x_categories': top_cat2,
            'y_categories': top_cat1,
            'x_label': field2,
            'y_label': field1
        }

    return plot_data


def estimate_resources(df: pd.DataFrame, field1: str, field2: str) -> Dict[str, Any]:
    """
    Estimate resources needed for correlation analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field

    Returns:
    --------
    Dict[str, Any]
        Estimated resource requirements
    """
    # Validate fields
    for field in [field1, field2]:
        if field not in df.columns:
            return {'error': f"Field {field} not found in DataFrame"}

    # Get data types and size
    is_numeric1 = pd.api.types.is_numeric_dtype(df[field1])
    is_numeric2 = pd.api.types.is_numeric_dtype(df[field2])
    row_count = len(df)

    # Count unique values
    unique_count1 = df[field1].nunique()
    unique_count2 = df[field2].nunique()

    # Estimate memory requirements
    # This is a simplified estimation based on field types and data size
    base_memory = 5  # MB
    memory_per_row = 0.0001  # MB per row

    # More unique values in categorical fields require more memory
    cat_factor = 0
    if not is_numeric1:
        cat_factor += unique_count1 * 0.005  # MB per unique value
    if not is_numeric2:
        cat_factor += unique_count2 * 0.005

    # Calculate total estimated memory
    estimated_memory = base_memory + (row_count * memory_per_row) + cat_factor

    # Estimate processing time
    # Processing time increases with row count and complexity of correlation method
    base_time = 0.5  # seconds
    time_per_row = 0.00002  # seconds per row for simple correlations

    # Complex correlations (categorical) take longer
    complexity_factor = 1
    if not is_numeric1 and not is_numeric2:
        # Cramer's V is more expensive
        complexity_factor = 5
        time_per_row = 0.0001
    elif not is_numeric1 or not is_numeric2:
        # Correlation ratio is moderate
        complexity_factor = 2
        time_per_row = 0.00005

    # Increased unique values increase processing time
    if not is_numeric1:
        complexity_factor *= (1 + (unique_count1 / 1000))
    if not is_numeric2:
        complexity_factor *= (1 + (unique_count2 / 1000))

    estimated_time = base_time + (row_count * time_per_row * complexity_factor)

    # Detect potential issues
    issues = []
    if row_count > 1000000:
        issues.append("Large dataset (>1M rows) may cause performance issues")

    if not is_numeric1 and unique_count1 > 1000:
        issues.append(f"Field {field1} has {unique_count1} unique values, which may impact performance")

    if not is_numeric2 and unique_count2 > 1000:
        issues.append(f"Field {field2} has {unique_count2} unique values, which may impact performance")

    if estimated_memory > 1000:
        issues.append(f"High memory usage estimated ({estimated_memory:.1f} MB)")

    # Calculate a recommended correlation method
    recommended_method = detect_correlation_type(df, field1, field2)

    return {
        'estimated_memory_mb': round(estimated_memory, 2),
        'estimated_time_seconds': round(estimated_time, 2),
        'row_count': row_count,
        'unique_values': {
            field1: unique_count1,
            field2: unique_count2
        },
        'potential_issues': issues,
        'recommended_method': recommended_method,
        'field_types': {
            field1: 'numeric' if is_numeric1 else 'categorical',
            field2: 'numeric' if is_numeric2 else 'categorical'
        }
    }