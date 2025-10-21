"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Field Utilities
Package:       pamola_core.utils.ops
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides essential field manipulation utilities for the PAMOLA.CORE
  operations framework. It standardizes field naming, conditional operations,
  field type inference, cross-field operations, and basic field analysis across
  all operation types.

Key Features:
  - Standardized output field name generation
  - Universal conditional operators for filtering
  - Field compatibility validation
  - Basic field statistics
  - Field type inference with pattern matching
  - K-anonymity field name generation
  - Cross-field operations support
  - Composite key generation with optional hashing

Design Principles:
  - Simple and focused functionality
  - Consistent naming conventions
  - Framework-agnostic implementation
  - Minimal dependencies

Changelog:
  1.2.0 - Added cross-field operations and composite key generation
  1.1.0 - Added field type inference and K-anonymity field naming
  1.0.0 - Initial implementation
"""

import base64
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Common patterns for field type detection
# Common patterns for field type detection
FIELD_PATTERNS = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "phone": re.compile(r"^[+]?[(]?[0-9]{3}[)]?[-\s.]?[0-9]{3}[-\s.]?[0-9]{4,6}$"),
    "ipv4": re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    ),
    "url": re.compile(r"^https?://[\w\s.-]+$"),
    "postal_code": re.compile(r"^\d{5}(?:-\d{4})?$"),
}


def generate_output_field_name(
    field_name: str,
    mode: str,
    output_field_name: Optional[str] = None,
    operation_suffix: Optional[str] = None,
    column_prefix: str = "_",
) -> str:
    """
    Generate standardized output field name based on operation mode.

    Parameters:
    -----------
    field_name : str
        Original field name
    mode : str
        Operation mode: "REPLACE" or "ENRICH"
    output_field_name : Optional[str], optional
        Explicitly specified output field name
    operation_suffix : Optional[str], optional
        Suffix to add for ENRICH mode (e.g., "masked", "generalized")
    column_prefix : str, optional
        Prefix for generated field names in ENRICH mode (default: "_")

    Returns:
    --------
    str
        Output field name

    Raises:
    -------
    ValueError
        If mode is not "REPLACE" or "ENRICH"

    Examples:
    ---------
    >>> generate_output_field_name("salary", "REPLACE")
    'salary'
    >>> generate_output_field_name("salary", "ENRICH", operation_suffix="masked")
    '_salary_masked'
    >>> generate_output_field_name("salary", "ENRICH", output_field_name="salary_hidden")
    'salary_hidden'
    """
    if mode not in ["REPLACE", "ENRICH"]:
        raise ValueError(f"Mode must be 'REPLACE' or 'ENRICH', got '{mode}'")

    if mode == "REPLACE":
        return field_name

    # ENRICH mode
    if output_field_name:
        return output_field_name

    if operation_suffix:
        return f"{column_prefix}{field_name}_{operation_suffix}"

    return f"{column_prefix}{field_name}"


def generate_ka_field_name(
    quasi_identifiers: List[str],
    prefix: str = "KA_",
    max_length: int = 3,
    separator: str = "_",
) -> str:
    """
    Generate a compact field name for k-anonymity metrics.

    Parameters:
    -----------
    quasi_identifiers : List[str]
        List of quasi-identifier field names
    prefix : str, optional
        Prefix for the field (default: "KA_")
    max_length : int, optional
        Number of initial characters from each identifier (default: 3)
    separator : str, optional
        Separator between abbreviated names (default: "_")

    Returns:
    --------
    str
        Generated KA field name, e.g., "KA_age_cit_pos" for ["age", "city", "postal_code"]

    Examples:
    ---------
    >>> generate_ka_field_name(["age", "city", "postal_code"])
    'KA_age_cit_pos'
    >>> generate_ka_field_name(["education", "salary"], max_length=2)
    'KA_ed_sa'
    """
    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier is required")

    # Create abbreviations
    abbreviations = []
    for qi in quasi_identifiers:
        # Handle underscored names by taking first part
        parts = qi.split("_")
        abbrev = parts[0][:max_length]
        abbreviations.append(abbrev)

    # Join with separator
    core = separator.join(abbreviations)

    return f"{prefix}{core}"


def infer_field_type(
    series: pd.Series, check_patterns: bool = True, sample_size: int = 100
) -> str:
    """
    Infer the semantic type of a field.

    Parameters:
    -----------
    series : pd.Series
        Field to analyze
    check_patterns : bool, optional
        Whether to check for specific patterns like email, phone (default: True)
    sample_size : int, optional
        Number of non-null values to sample for pattern checking (default: 100)

    Returns:
    --------
    str
        Field type: 'numeric', 'categorical', 'string', 'datetime', 'boolean',
        'email', 'phone', 'ipv4', 'url', 'postal_code', or 'unknown'

    Examples:
    ---------
    >>> s = pd.Series(['user@example.com', 'test@domain.org'])
    >>> infer_field_type(s)
    'email'
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> infer_field_type(s)
    'numeric'
    """
    dtype = series.dtype

    # Basic type checks
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # For string/object types, check patterns if requested
    if (pd.api.types.is_string_dtype(dtype) or dtype == "object") and check_patterns:
        # Sample non-null values for pattern matching
        non_null = series.dropna()
        if len(non_null) > 0:
            sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)

            # Check each pattern
            for pattern_name, pattern in FIELD_PATTERNS.items():
                # Convert to string and check if most values match
                string_values = sample.astype(str)
                matches = string_values.apply(lambda x: bool(pattern.match(x)))

                # If >80% of sampled values match, consider it this type
                if matches.sum() / len(matches) > 0.8:
                    return pattern_name

        return "string"

    return "unknown"


def apply_condition_operator(
    series: pd.Series, condition_values: Optional[List[Any]], operator: str
) -> pd.Series:
    """
    Apply conditional operator to create a boolean mask.

    Parameters:
    -----------
    series : pd.Series
        Series to apply condition to
    condition_values : Optional[List[Any]]
        Values for the condition (can be None for some operators)
    operator : str
        Operator type: "in", "not_in", "gt", "lt", "eq", "ne", "ge", "le", "range", "all"

    Returns:
    --------
    pd.Series
        Boolean mask series

    Raises:
    -------
    ValueError
        If operator is invalid or condition_values are missing when required
    TypeError
        If input types are invalid
    """
    # Validate input
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    # Special case: all
    if operator == "all":
        if condition_values is not None and len(condition_values) > 0:
            raise ValueError("Operator 'all' does not accept condition_values")
        return pd.Series(True, index=series.index)

    # Validate condition_values
    if condition_values is None:
        raise ValueError(f"Operator '{operator}' requires condition_values")
    if not isinstance(condition_values, list):
        raise TypeError("condition_values must be a list")
    if len(condition_values) == 0:
        raise ValueError(f"Operator '{operator}' requires non-empty condition_values")

    # Helper: check comparable types
    def ensure_comparable(val):
        try:
            _ = series.iloc[0] > val  # test comparability
        except Exception:
            raise TypeError(
                f"Operator '{operator}' not supported for type '{type(series.iloc[0])}' "
                f"with value type '{type(val)}'"
            )

    try:
        if operator == "in":
            return series.isin(condition_values)

        elif operator == "not_in":
            return ~series.isin(condition_values)

        elif operator == "eq":
            return series == condition_values[0]

        elif operator == "ne":
            return series != condition_values[0]

        elif operator in ("gt", "lt", "ge", "le"):
            ensure_comparable(condition_values[0])
            if operator == "gt":
                return series > condition_values[0]
            elif operator == "lt":
                return series < condition_values[0]
            elif operator == "ge":
                return series >= condition_values[0]
            elif operator == "le":
                return series <= condition_values[0]

        elif operator == "range":
            if len(condition_values) < 2:
                raise ValueError("Operator 'range' requires at least 2 values [min, max]")
            ensure_comparable(condition_values[0])
            ensure_comparable(condition_values[1])
            return (series >= condition_values[0]) & (series <= condition_values[1])

        else:
            raise ValueError(f"Unknown operator: '{operator}'")

    except Exception as e:
        raise ValueError(f"Error applying operator '{operator}': {str(e)}") from e


def validate_field_compatibility(
    source_field: pd.Series, target_field: pd.Series, operation_type: str
) -> Dict[str, Any]:
    """
    Validate compatibility between fields for operations.

    Parameters:
    -----------
    source_field : pd.Series
        Source field
    target_field : pd.Series
        Target field
    operation_type : str
        Type of operation: "merge", "join", "compare", "replace"

    Returns:
    --------
    Dict[str, Any]
        Validation result with 'compatible' flag and 'issues' list

    Examples:
    ---------
    >>> s1 = pd.Series([1, 2, 3])
    >>> s2 = pd.Series(['a', 'b', 'c'])
    >>> result = validate_field_compatibility(s1, s2, "compare")
    >>> result['compatible']
    False
    """
    issues = []

    # Check length compatibility for certain operations
    if operation_type in ["merge", "replace"]:
        if len(source_field) != len(target_field):
            issues.append(
                f"Length mismatch: {len(source_field)} vs {len(target_field)}"
            )

    # Check data type compatibility
    source_dtype = str(source_field.dtype)
    target_dtype = str(target_field.dtype)

    if operation_type == "compare":
        # For comparison, types should be compatible
        numeric_types = ["int", "float"]
        source_is_numeric = any(t in source_dtype for t in numeric_types)
        target_is_numeric = any(t in target_dtype for t in numeric_types)

        if source_is_numeric != target_is_numeric:
            issues.append(f"Type incompatibility: {source_dtype} vs {target_dtype}")

    elif operation_type == "replace":
        # For replace, exact type match is preferred
        if source_dtype != target_dtype:
            issues.append(
                f"Type mismatch for replace: {source_dtype} vs {target_dtype}"
            )

    # Check index compatibility
    if operation_type in ["merge", "join"]:
        if not source_field.index.equals(target_field.index):
            issues.append("Index mismatch between fields")

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "source_dtype": source_dtype,
        "target_dtype": target_dtype,
        "source_length": len(source_field),
        "target_length": len(target_field),
    }


def get_field_statistics(
    series: pd.Series, include_percentiles: bool = False
) -> Dict[str, Any]:
    """
    Get basic statistics for a field of any type.

    Parameters:
    -----------
    series : pd.Series
        Field to analyze
    include_percentiles : bool, optional
        Whether to include percentile information (default: False)

    Returns:
    --------
    Dict[str, Any]
        Field statistics including type-appropriate metrics

    Examples:
    ---------
    >>> s = pd.Series([1, 2, 3, 4, 5, None])
    >>> stats = get_field_statistics(s)
    >>> stats['count']
    5
    >>> stats['null_count']
    1
    """
    stats = {
        "count": series.count(),
        "null_count": series.isnull().sum(),
        "null_percentage": (
            round(series.isnull().sum() / len(series) * 100, 2)
            if len(series) > 0
            else 0
        ),
        "dtype": str(series.dtype),
        "unique_count": series.nunique(),
        "unique_percentage": (
            round(series.nunique() / series.count() * 100, 2)
            if series.count() > 0
            else 0
        ),
    }

    # Add numeric statistics if applicable
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = series.dropna()
        if len(numeric_series) > 0:
            stats.update(
                {
                    "mean": float(numeric_series.mean()),
                    "std": float(numeric_series.std()),
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                }
            )

            if include_percentiles:
                stats["percentiles"] = {
                    "25%": float(numeric_series.quantile(0.25)),
                    "50%": float(numeric_series.quantile(0.50)),
                    "75%": float(numeric_series.quantile(0.75)),
                }

    # Add categorical statistics if applicable
    elif isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == "object":
        value_counts = series.value_counts()
        if len(value_counts) > 0:
            stats.update(
                {
                    "most_common_value": str(value_counts.index[0]),
                    "most_common_count": int(value_counts.iloc[0]),
                    "top_5_values": value_counts.head(5).to_dict(),
                }
            )

    return stats


def create_field_mask(
    df: pd.DataFrame,
    field_name: str,
    condition_field: Optional[str] = None,
    condition_values: Optional[List[Any]] = None,
    condition_operator: str = "in",
) -> pd.Series:
    """
    Create a boolean mask for field processing based on conditions.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    field_name : str
        Field to create mask for
    condition_field : Optional[str], optional
        Field to apply condition to (if None, applies to field_name)
    condition_values : Optional[List[Any]], optional
        Values for the condition
    condition_operator : str, optional
        Operator for condition (default: "in")

    Returns:
    --------
    pd.Series
        Boolean mask

    Examples:
    ---------
    >>> df = pd.DataFrame({'age': [20, 30, 40], 'city': ['NY', 'LA', 'NY']})
    >>> mask = create_field_mask(df, 'age', 'city', ['NY'], 'in')
    >>> mask
    0     True
    1    False
    2     True
    dtype: bool
    """
    if field_name not in df.columns:
        raise ValueError(f"Field '{field_name}' not found in DataFrame")

    # Determine which field to apply condition to
    target_field = condition_field if condition_field else field_name

    if target_field not in df.columns:
        raise ValueError(f"Condition field '{target_field}' not found in DataFrame")

    # If no condition specified, return all True
    if condition_values is None and condition_operator != "all":
        return pd.Series([True] * len(df), index=df.index)

    # Apply condition operator
    return apply_condition_operator(
        df[target_field], condition_values, condition_operator
    )


def get_field_name_variants(
    field_name: str,
    prefixes: Optional[List[str]] = None,
    suffixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate common field name variants for matching.

    Parameters:
    -----------
    field_name : str
        Base field name
    prefixes : Optional[List[str]], optional
        List of prefixes to try
    suffixes : Optional[List[str]], optional
        List of suffixes to try

    Returns:
    --------
    List[str]
        List of field name variants

    Examples:
    ---------
    >>> get_field_name_variants("age", suffixes=["_years", "_months"])
    ['age', 'age_years', 'age_months']
    """
    variants = [field_name]

    if prefixes:
        for prefix in prefixes:
            variants.append(f"{prefix}{field_name}")

    if suffixes:
        for suffix in suffixes:
            variants.append(f"{field_name}{suffix}")

    return variants


def generate_privacy_metric_field_name(
    field_name: str, metric_type: str, quasi_identifiers: Optional[List[str]] = None
) -> str:
    """
    Generate standardized field names for privacy metrics.

    Parameters:
    -----------
    field_name : str
        Base field name
    metric_type : str
        Type of metric: "k_anonymity", "l_diversity", "t_closeness", "risk_score"
    quasi_identifiers : Optional[List[str]], optional
        List of quasi-identifiers (for k-anonymity metrics)

    Returns:
    --------
    str
        Generated metric field name

    Examples:
    ---------
    >>> generate_privacy_metric_field_name("age", "risk_score")
    'age_risk_score'
    >>> generate_privacy_metric_field_name("", "k_anonymity", ["age", "city"])
    'KA_age_cit'
    """
    if metric_type == "k_anonymity" and quasi_identifiers:
        return generate_ka_field_name(quasi_identifiers)

    # For other metrics, use simple suffix approach
    metric_suffixes = {
        "l_diversity": "_l_div",
        "t_closeness": "_t_close",
        "risk_score": "_risk_score",
        "entropy": "_entropy",
    }

    suffix = metric_suffixes.get(metric_type, f"_{metric_type}")
    return f"{field_name}{suffix}" if field_name else metric_type


def create_composite_key(
    df: pd.DataFrame,
    fields: List[str],
    separator: str = "_",
    null_handling: str = "skip",
    hash_key: bool = False,
    hash_algorithm: str = "sha256",
) -> pd.Series:
    """
    Create a composite key from multiple fields with optional hashing.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    fields : List[str]
        List of field names to combine
    separator : str, optional
        Separator for composite key (default: "_")
    null_handling : str, optional
        How to handle nulls: "skip", "include", "replace" (default: "skip")
    hash_key : bool, optional
        Whether to hash the composite key (default: False)
    hash_algorithm : str, optional
        Hash algorithm: "sha256", "sha1", "md5" (default: "sha256")

    Returns:
    --------
    pd.Series
        Composite keys (plain or hashed)

    Examples:
    ---------
    >>> df = pd.DataFrame({'age': [25, 30], 'city': ['NY', 'LA']})
    >>> create_composite_key(df, ['age', 'city'])
    0    25_NY
    1    30_LA
    dtype: object

    >>> create_composite_key(df, ['age', 'city'], hash_key=True)
    0    a3f5d8e9...  # SHA256 hash
    1    b7c2a1f4...  # SHA256 hash
    dtype: object
    """
    if not fields:
        raise ValueError("At least one field is required")

    # Validate all fields exist
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"Fields not found: {missing}")

    # First create plain composite keys
    if null_handling == "skip":
        null_mask = df[fields].isnull().any(axis=1)
        result = pd.Series(index=df.index, dtype="object")

        non_null_idx = ~null_mask
        if non_null_idx.any():
            combined = (
                df.loc[non_null_idx, fields]
                .astype(str)
                .apply(lambda row: separator.join(row), axis=1)
            )
            result.loc[non_null_idx] = combined

    elif null_handling == "include":
        result = df[fields].astype(str).apply(lambda row: separator.join(row), axis=1)

    elif null_handling == "replace":
        filled_df = df[fields].fillna("NULL")
        result = filled_df.astype(str).apply(lambda row: separator.join(row), axis=1)
    else:
        raise ValueError(f"Unknown null_handling: {null_handling}")

    # Apply hashing if requested
    if hash_key:
        hash_funcs = {
            "sha256": hashlib.sha256,
            "sha1": hashlib.sha1,
            "md5": hashlib.md5,
        }

        if hash_algorithm not in hash_funcs:
            raise ValueError(f"Unknown hash algorithm: {hash_algorithm}")

        hash_func = hash_funcs[hash_algorithm]

        # Hash non-null values
        def hash_value(val):
            if pd.isna(val):
                return val
            return hash_func(str(val).encode("utf-8")).hexdigest()

        result = result.apply(hash_value)

    return result


def create_reversible_composite_key(
    df: pd.DataFrame, fields: List[str], encoding: str = "base64"
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Create a reversible composite key with encoding.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    fields : List[str]
        List of field names to combine
    encoding : str, optional
        Encoding type: "base64", "hex" (default: "base64")

    Returns:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        (encoded_keys, decoding_info)

    Examples:
    ---------
    >>> df = pd.DataFrame({'age': [25, 30], 'city': ['NY', 'LA']})
    >>> keys, info = create_reversible_composite_key(df, ['age', 'city'])
    >>> keys
    0    MjV8Tnk=    # base64 encoded "25|NY"
    1    MzB8TEE=    # base64 encoded "30|LA"
    dtype: object
    """
    # Use pipe as separator to avoid conflicts
    separator = "|"

    # Create composite keys
    composite = (
        df[fields].fillna("").astype(str).apply(lambda row: separator.join(row), axis=1)
    )

    # Encode based on method
    if encoding == "base64":
        encoded = composite.apply(
            lambda x: base64.b64encode(x.encode("utf-8")).decode("utf-8")
        )
    elif encoding == "hex":
        encoded = composite.apply(lambda x: x.encode("utf-8").hex())
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Store decoding information
    decoding_info = {
        "fields": fields,
        "separator": separator,
        "encoding": encoding,
        "field_positions": {field: i for i, field in enumerate(fields)},
    }

    return encoded, decoding_info


def create_multi_field_mask(
    df: pd.DataFrame, conditions: List[Dict[str, Any]], logic: str = "AND"
) -> pd.Series:
    """
    Create a mask based on conditions across multiple fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    conditions : List[Dict[str, Any]]
        List of conditions: [{"field": "age", "operator": "gt", "values": [18]}, ...]
    logic : str, optional
        How to combine conditions: "AND", "OR" (default: "AND")

    Returns:
    --------
    pd.Series
        Boolean mask

    Examples:
    ---------
    >>> df = pd.DataFrame({'age': [25, 35, 45], 'income': [30000, 50000, 70000]})
    >>> conditions = [
    ...     {"field": "age", "operator": "gt", "values": [30]},
    ...     {"field": "income", "operator": "ge", "values": [50000]}
    ... ]
    >>> create_multi_field_mask(df, conditions, logic="AND")
    0    False
    1     True
    2     True
    dtype: bool
    """
    if not conditions:
        return pd.Series([True] * len(df), index=df.index)

    masks = []
    for condition in conditions:
        field = condition.get("field")
        operator = condition.get("operator", "in")
        values = condition.get("values", [])

        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame columns")

        mask = apply_condition_operator(df[field], values, operator)
        masks.append(mask)

    # Combine masks
    if logic == "AND":
        result = masks[0]
        for mask in masks[1:]:
            result = result & mask
        return result
    elif logic == "OR":
        result = masks[0]
        for mask in masks[1:]:
            result = result | mask
        return result
    else:
        raise ValueError(f"Unknown logic: {logic}. Use 'AND' or 'OR'")


def validate_fields_for_operation(
    df: pd.DataFrame, fields: List[str], operation_type: str
) -> Dict[str, Any]:
    """
    Validate multiple fields for cross-field operations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the fields
    fields : List[str]
        List of field names
    operation_type : str
        Type of operation: "microaggregation", "multivariate_generalization", "composite_key"

    Returns:
    --------
    Dict[str, Any]
        Validation results with issues and recommendations
    """
    issues = []
    recommendations = []

    # Check all fields exist
    missing = [f for f in fields if f not in df.columns]
    if missing:
        issues.append(f"Missing fields: {missing}")
        return {"valid": False, "issues": issues, "recommendations": []}

    # Get field types
    field_types = {f: infer_field_type(df[f]) for f in fields}

    if operation_type == "microaggregation":
        # All fields should be numeric for microaggregation
        non_numeric = [f for f, t in field_types.items() if t != "numeric"]
        if non_numeric:
            issues.append(f"Non-numeric fields for microaggregation: {non_numeric}")
            recommendations.append(
                "Convert categorical fields to numeric encoding first"
            )

    elif operation_type == "composite_key":
        # Check for high cardinality fields that might create too many unique keys
        for field in fields:
            unique_ratio = df[field].nunique() / len(df)
            if unique_ratio > 0.9:
                recommendations.append(
                    f"Field '{field}' has very high cardinality ({unique_ratio:.2%})"
                )

    elif operation_type == "multivariate_generalization":
        # Check for compatible types
        type_groups = {}
        for field, ftype in field_types.items():
            type_groups.setdefault(ftype, []).append(field)

        if len(type_groups) > 1:
            recommendations.append(f"Mixed field types: {type_groups}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "field_types": field_types,
    }


# Module metadata
__version__ = "1.2.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    "generate_output_field_name",
    "generate_ka_field_name",
    "generate_privacy_metric_field_name",
    "infer_field_type",
    "apply_condition_operator",
    "validate_field_compatibility",
    "get_field_statistics",
    "create_field_mask",
    "get_field_name_variants",
    "create_composite_key",
    "create_reversible_composite_key",
    "create_multi_field_mask",
    "validate_fields_for_operation",
    "FIELD_PATTERNS",
]
