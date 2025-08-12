"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: validation_utils.py
Description: Validation utilities for transformation operations, ensuring parameter integrity,
             schema correctness, and constraint enforcement.
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import pandas as pd

logger = logging.getLogger(__name__)


def validate_fields_exist(
    df: pd.DataFrame, required_fields: List[str]
) -> Tuple[bool, Optional[List[str]]]:
    """
    Validate that required fields exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    required_fields : List[str]
        List of required field names.

    Returns
    -------
    Tuple[bool, Optional[List[str]]]
        (True, None) if all fields exist, (False, [missing_fields]) otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a pandas DataFrame")
        raise TypeError("Input 'df' must be a pandas DataFrame")

    if not isinstance(required_fields, list):
        logger.error("Input 'required_fields' must be a list")
        raise TypeError("Input 'required_fields' must be a list")

    existing_fields = set(df.columns)
    missing_fields = [
        field for field in required_fields if field not in existing_fields
    ]

    if missing_fields:
        logger.warning(f"Missing fields detected: {missing_fields}")
        return False, missing_fields

    return True, None


def validate_field_types(
    df: pd.DataFrame, field_types: Dict[str, str]
) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Validate that fields have the expected data types.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    field_types : Dict[str, str]
        Mapping of field names to expected type strings.

    Returns
    -------
    Tuple[bool, Optional[Dict[str, str]]]
        (True, None) if all types match,
        (False, {field: "expected vs actual"}) otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a pandas DataFrame")
        raise TypeError("Input 'df' must be a pandas DataFrame")

    if not isinstance(field_types, dict):
        logger.error("Input 'field_types' must be a dictionary")
        raise TypeError("Input 'field_types' must be a dictionary")

    fields_exist, missing_fields = validate_fields_exist(df, list(field_types.keys()))
    if not fields_exist:
        logger.error(f"Cannot validate types for missing fields: {missing_fields}")
        raise ValueError(f"Missing fields: {missing_fields}")

    type_errors: Dict[str, str] = {}

    for field, expected_type in field_types.items():
        actual_type = str(df[field].dtype)

        if expected_type in ("numeric", "number"):
            if not pd.api.types.is_numeric_dtype(df[field]):
                type_errors[field] = f"{expected_type} vs {actual_type}"
        elif expected_type == "datetime":
            if not pd.api.types.is_datetime64_dtype(df[field]):
                type_errors[field] = f"{expected_type} vs {actual_type}"
        elif actual_type != expected_type:
            type_errors[field] = f"{expected_type} vs {actual_type}"

    if type_errors:
        logger.warning(f"Field type mismatches found: {type_errors}")
        return False, type_errors

    return True, None


def validate_parameters(
    parameters: Dict[str, Any], required_params: List[str], param_types: Dict[str, Type]
) -> Tuple[bool, Optional[List[str]]]:
    """
    Validate operation parameters for presence and type.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary of parameters to validate.
    required_params : List[str]
        List of required parameter names.
    param_types : Dict[str, Type]
        Mapping of parameter names to expected types.

    Returns
    -------
    Tuple[bool, Optional[List[str]]]
        (True, None) if all parameters are valid,
        (False, list_of_error_messages) otherwise.
    """
    if not isinstance(parameters, dict):
        logger.error("Input 'parameters' must be a dictionary")
        raise TypeError("Input 'parameters' must be a dictionary")

    if not isinstance(required_params, list):
        logger.error("Input 'required_params' must be a list")
        raise TypeError("Input 'required_params' must be a list")

    if not isinstance(param_types, dict):
        logger.error("Input 'param_types' must be a dictionary")
        raise TypeError("Input 'param_types' must be a dictionary")

    errors: List[str] = []

    # Check for missing required parameters
    for param in required_params:
        if param not in parameters:
            errors.append(f"Missing required parameter: '{param}'")

    # Check type correctness
    for param, expected_type in param_types.items():
        if param in parameters:
            param_value = parameters[param]
            # Handle Union types
            if (
                hasattr(expected_type, "__origin__")
                and expected_type.__origin__ is Union
            ):
                if not any(isinstance(param_value, t) for t in expected_type.__args__):
                    errors.append(
                        f"Parameter '{param}' expected type {expected_type} but got {type(param_value).__name__}"
                    )
            elif not isinstance(param_value, expected_type):
                errors.append(
                    f"Parameter '{param}' expected type {expected_type.__name__} but got {type(param_value).__name__}"
                )

    if errors:
        logger.warning(f"Parameter validation failed: {errors}")
        return False, errors

    return True, None


def validate_constraints(
    df: pd.DataFrame, constraints: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Validate data against a dictionary of field constraints.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of violations: {field: {constraint: {...}}}
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a pandas DataFrame")
        raise TypeError("Input 'df' must be a pandas DataFrame")

    if not isinstance(constraints, dict):
        logger.error("Input 'constraints' must be a dictionary")
        raise TypeError("Input 'constraints' must be a dictionary")

    fields_exist, missing_fields = validate_fields_exist(df, list(constraints.keys()))
    if not fields_exist:
        logger.error(
            f"Cannot validate constraints for missing fields: {missing_fields}"
        )
        raise ValueError(f"Missing fields: {missing_fields}")

    violations: Dict[str, Dict[str, Any]] = {}

    for field, field_constraints in constraints.items():
        field_violations: Dict[str, Any] = {}

        # Not null constraint
        if field_constraints.get("not_null", False):
            null_indices = df.index[df[field].isna()].tolist()
            if null_indices:
                field_violations["not_null"] = {
                    "indices": null_indices,
                    "count": len(null_indices),
                    "example_values": df[field].iloc[null_indices[:5]].tolist(),
                }

        non_null_df = df[~df[field].isna()]

        # Min value constraint
        if "min" in field_constraints:
            min_value = field_constraints["min"]
            indices = non_null_df.index[non_null_df[field] < min_value].tolist()
            if indices:
                field_violations["min"] = {
                    "indices": indices,
                    "count": len(indices),
                    "example_values": non_null_df[field].iloc[:5].tolist(),
                }

        # Max value constraint
        if "max" in field_constraints:
            max_value = field_constraints["max"]
            indices = non_null_df.index[non_null_df[field] > max_value].tolist()
            if indices:
                field_violations["max"] = {
                    "indices": indices,
                    "count": len(indices),
                    "example_values": non_null_df[field].iloc[:5].tolist(),
                }

        # Allowed values constraint
        if "allowed_values" in field_constraints:
            allowed = field_constraints["allowed_values"]
            indices = non_null_df.index[~non_null_df[field].isin(allowed)].tolist()
            if indices:
                field_violations["allowed_values"] = {
                    "indices": indices,
                    "count": len(indices),
                    "example_values": non_null_df[field].iloc[:5].tolist(),
                }

        # Unique constraint
        if field_constraints.get("unique", False):
            duplicated = df.duplicated(subset=[field], keep=False)
            indices = df.index[duplicated].tolist()
            if indices:
                field_violations["unique"] = {
                    "indices": indices,
                    "count": len(indices),
                    "example_values": df[field].iloc[indices[:5]].tolist(),
                }

        # Regex constraint for string fields
        if "regex" in field_constraints and pd.api.types.is_string_dtype(df[field]):
            pattern = field_constraints["regex"]
            matches = non_null_df[field].str.match(pattern, na=False)
            indices = non_null_df.index[~matches].tolist()
            if indices:
                field_violations["regex"] = {
                    "indices": indices,
                    "count": len(indices),
                    "example_values": non_null_df[field].iloc[:5].tolist(),
                }

        if field_violations:
            violations[field] = field_violations

    if violations:
        logger.info(f"Constraint violations detected: {violations}")
    else:
        logger.info("No constraint violations detected.")

    return violations


def validate_dataframe(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Helper function to validate if specified columns exist in the DataFrame.

    Parameters:
        df (pd.DataFrame): The pandas DataFrame to validate.
        columns (List[str]): A list of column names to check for existence in the DataFrame.

    Raises:
        ValueError: If one or more specified columns are missing from the DataFrame.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")


def validate_group_and_aggregation_fields(
    df: pd.DataFrame,
    group_by_fields: List[str],
    aggregations: Optional[Dict[str, List[str]]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
) -> None:
    """
    Validate that all group_by_fields and aggregation fields exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    group_by_fields : List[str]
        Fields to group by.
    aggregations : Optional[Dict[str, List[str]]]
        Aggregation functions per field.
    custom_aggregations : Optional[Dict[str, Callable]]
        Custom aggregation functions per field.

    Raises
    ------
    ValueError
        If any required field is missing in the DataFrame.
    """
    # Validate group by fields
    validate_dataframe(df, group_by_fields)

    # Collect all fields used in aggregations and custom aggregations
    agg_fields = set()
    if aggregations:
        agg_fields.update(aggregations.keys())
    if custom_aggregations:
        agg_fields.update(custom_aggregations.keys())

    if agg_fields:
        validate_dataframe(df, list(agg_fields))


def validate_join_type(join_type: str) -> None:
    """
    Helper function to validate join type.

    Parameters:
        join_type (str): The type of join to validate. Expected values are
                         "left", "right", "inner", or "outer".

    Raises:
        ValueError: If the provided join_type is not one of the valid options.
    """
    valid_join_types = ["left", "right", "inner", "outer"]
    if join_type not in valid_join_types:
        raise ValueError(
            f"Invalid join type: {join_type}. Must be one of: {valid_join_types}"
        )
