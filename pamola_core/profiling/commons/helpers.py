"""
Helper functions for the profiling package.

This module provides utility functions for data preparation, type inference,
and other common operations used across different profiling modules.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from pamola_core.common.constants import Constants
from pamola_core.profiling.commons.data_types import (
    DataType,
    DataTypeDetection,
    ProfilerConfig,
)

# Import our custom dtype helpers instead of using pd.api.types directly
from pamola_core.profiling.commons.dtype_helpers import (
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
    is_datetime64_dtype,
    is_categorical_dtype,
)
from pamola_core.utils import helpers
from pamola_core.utils.io import load_data_operation, write_json
from pamola_core.utils.ops.op_result import OperationResult
from pamola_core.utils.ops.op_data_source import DataSource

# Configure logger
logger = logging.getLogger(__name__)


# Define our own utility functions to avoid io.py dependency
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def ensure_directory(directory_path):
    """Ensure a directory exists, creating it if necessary."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    return directory_path


def get_profiling_directory(profile_type=None):
    """Get the directory path for storing profiling results."""
    base_dir = Path(os.environ.get("PROFILING_OUTPUT_DIR", "profiling_output"))
    if profile_type:
        return base_dir / profile_type
    return base_dir


def save_profiling_result(
    result, field_name, output_name, format="json", include_timestamp=True
):
    """Save profiling results to a file."""
    directory = get_profiling_directory(field_name)
    ensure_directory(directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
    filename = (
        f"{output_name}_{timestamp}.{format}"
        if timestamp
        else f"{output_name}.{format}"
    )
    file_path = directory / filename

    if format.lower() == "json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    elif format.lower() == "csv":
        # Convert result to DataFrame and save as CSV
        pd.DataFrame(result).to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return file_path


def infer_data_type(series: pd.Series) -> DataType:
    """
    Infer the data type of a pandas Series.

    Parameters:
    -----------
    series : pd.Series
        The series to analyze

    Returns:
    -------
    DataType
        The inferred data type
    """
    # Handle empty series
    if series.empty:
        return DataType.UNKNOWN

    # Handle series with all null values
    if series.isna().all():
        return DataType.UNKNOWN

    # Check if numeric
    if is_numeric_dtype(series):
        return DataType.NUMERIC

    # Check if boolean
    if is_bool_dtype(series):
        return DataType.BOOLEAN

    # For object or string types, perform more detailed analysis
    if is_object_dtype(series) or is_string_dtype(series):
        # Sample non-null values for analysis
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return DataType.UNKNOWN

        # Use a sample for faster detection
        sample_size = min(len(non_null_values), 100)
        sample = (
            non_null_values.sample(sample_size)
            if len(non_null_values) > sample_size
            else non_null_values
        )

        # Check if it's email
        if series.name == "email" or "email" in str(series.name).lower():
            # Check if it matches email pattern
            email_pattern = re.compile(DataTypeDetection.EMAIL_REGEX)
            email_matches = sum(
                1
                for val in sample
                if isinstance(val, str) and email_pattern.match(val.strip())
            )
            if email_matches / len(sample) >= 0.8:  # 80% match threshold
                return DataType.EMAIL

        # Check if it's phone (HH specific format)
        if "phone" in str(series.name).lower():
            # Check if it matches phone pattern
            phone_pattern = re.compile(DataTypeDetection.PHONE_BASIC_REGEX)
            phone_matches = sum(
                1
                for val in sample
                if isinstance(val, str) and phone_pattern.match(val.strip())
            )
            if phone_matches / len(sample) >= 0.8:  # 80% match threshold
                return DataType.PHONE

        # Check if it might be a boolean in string form
        lowercase_sample = sample.str.lower() if hasattr(sample, "str") else sample
        boolean_matches = sum(
            1
            for val in lowercase_sample
            if isinstance(val, str)
            and val.lower()
            in DataTypeDetection.BOOLEAN_TRUE_VALUES.union(
                DataTypeDetection.BOOLEAN_FALSE_VALUES
            )
        )
        if boolean_matches / len(sample) >= 0.8:  # 80% match threshold
            return DataType.BOOLEAN

        # Check if it might be a date
        date_matches = 0
        for pattern in DataTypeDetection.DATE_PATTERNS:
            try:
                pd.to_datetime(sample, format=pattern, errors="raise")
                date_matches = len(sample)
                break
            except (ValueError, TypeError):
                continue

        if date_matches / len(sample) >= 0.8:  # 80% match threshold
            return DataType.DATE

        # Check if JSON
        json_matches = 0
        for val in sample:
            if not isinstance(val, str):
                continue
            val_stripped = val.strip()
            try:
                if (val_stripped.startswith("{") and val_stripped.endswith("}")) or (
                    val_stripped.startswith("[") and val_stripped.endswith("]")
                ):
                    json.loads(val_stripped)
                    json_matches += 1
            except (json.JSONDecodeError, TypeError):
                pass

        if json_matches / len(sample) >= 0.8:  # 80% match threshold
            return DataType.JSON

        # Check if multi-valued field
        separator_counts = {
            sep: sum(1 for val in sample if isinstance(val, str) and sep in val)
            for sep in DataTypeDetection.MVF_INDICATORS
        }
        if any(
            count / len(sample) >= DataTypeDetection.MVF_THRESHOLD
            for count in separator_counts.values()
        ):
            return DataType.MULTI_VALUED

        # Check if array (simplified check)
        array_matches = sum(
            1
            for val in sample
            if isinstance(val, str)
            and val.strip().startswith("[")
            and val.strip().endswith("]")
        )
        if array_matches / len(sample) >= 0.8:  # 80% match threshold
            return DataType.ARRAY

        # Check if categorical based on cardinality
        if series.nunique() <= DataTypeDetection.CATEGORICAL_THRESHOLD:
            return DataType.CATEGORICAL

        # Check if long text
        if hasattr(sample, "str") and (
            sample.str.len().mean() > ProfilerConfig.LONGTEXT_MIN_LENGTH
        ):
            return DataType.LONGTEXT

        # If not any of the above, assume text
        return DataType.TEXT

    # For datetime types
    if is_datetime64_dtype(series):
        return DataType.DATETIME

    # For categorical types
    if is_categorical_dtype(series):
        return DataType.CATEGORICAL

    # Default to unknown
    return DataType.UNKNOWN


def prepare_field_for_analysis(
    df: pd.DataFrame, field_name: str
) -> Tuple[pd.Series, DataType]:
    """
    Prepare a field for analysis, handling missing values and type conversion.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to prepare

    Returns:
    --------
    Tuple[pd.Series, DataType]
        The prepared series and its inferred data type
    """
    if field_name not in df.columns:
        raise ValueError(f"Field {field_name} not found in DataFrame")

    series = df[field_name].copy()
    data_type = infer_data_type(series)

    # Handle missing values based on data type
    if data_type == DataType.NUMERIC:
        # Don't replace nulls for numeric
        pass
    elif data_type == DataType.CATEGORICAL:
        # Convert to category if not already
        if not is_categorical_dtype(series):
            series = series.astype("category")
    elif data_type == DataType.DATE or data_type == DataType.DATETIME:
        # Try to convert to datetime
        try:
            series = pd.to_datetime(series, errors="coerce")
        except Exception as e:
            logger.warning(f"Error converting {field_name} to datetime: {e}")
    elif data_type == DataType.BOOLEAN:
        # Try to convert to boolean
        try:
            series = series.map(
                lambda x: (
                    str(x).lower() in DataTypeDetection.BOOLEAN_TRUE_VALUES
                    if pd.notna(x)
                    else np.nan
                )
            )
        except Exception as e:
            logger.warning(f"Error converting {field_name} to boolean: {e}")
    elif data_type == DataType.EMAIL:
        # Clean email addresses
        try:
            series = (
                series.str.strip().str.lower() if hasattr(series, "str") else series
            )
        except Exception as e:
            logger.warning(f"Error cleaning email addresses in {field_name}: {e}")
    elif data_type == DataType.PHONE:
        # Phone values are kept as is for special parsing
        pass
    elif data_type == DataType.JSON:
        # JSON values are kept as strings for special parsing
        pass
    elif data_type == DataType.ARRAY:
        # Array values are kept as strings for special parsing
        pass
    elif data_type == DataType.MULTI_VALUED:
        # MVF values are kept as strings for special parsing
        pass

    return series, data_type


def parse_multi_valued_field(value: Any, separator: str = None) -> List[str]:
    """
    Parse a multi-valued field into a list of values.

    Parameters:
    -----------
    value : Any
        The value to parse
    separator : str, optional
        The separator character. If None, tries to detect automatically.

    Returns:
    --------
    List[str]
        List of individual values
    """
    if pd.isna(value):
        return []

    # Convert to string
    str_value = str(value).strip()
    if not str_value:
        return []

    # If separator is not provided, try to detect
    if separator is None:
        for sep in DataTypeDetection.MVF_INDICATORS:
            if sep in str_value:
                separator = sep
                break

        # Default to comma if no separator found
        if separator is None:
            separator = ProfilerConfig.MVF_SEPARATOR

    # Split the string and clean up individual values
    result = [item.strip() for item in str_value.split(separator)]
    return [item for item in result if item]  # Remove empty items


def detect_json_field(series: pd.Series) -> bool:
    """
    Detect if a field contains JSON data.

    Parameters:
    -----------
    series : pd.Series
        The series to analyze

    Returns:
    --------
    bool
        True if the field appears to contain JSON data
    """
    if len(series) == 0:
        return False

    # Get a sample of non-null values
    sample = series.dropna().sample(min(100, len(series.dropna())))

    valid_count = 0
    for value in sample:
        if not isinstance(value, str):
            continue

        value = value.strip()
        if (value.startswith("{") and value.endswith("}")) or (
            value.startswith("[") and value.endswith("]")
        ):
            try:
                json.loads(value)
                valid_count += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Consider it JSON if at least 80% of the sample is valid JSON
    return valid_count >= 0.8 * len(sample)


def parse_json_field(value: Any) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON field value.

    Parameters:
    -----------
    value : Any
        The value to parse

    Returns:
    --------
    Dict[str, Any] or None
        The parsed JSON, or None if parsing fails
    """
    if pd.isna(value):
        return None

    if not isinstance(value, str):
        return None

    try:
        return json.loads(value.strip())
    except (json.JSONDecodeError, TypeError):
        return None


def detect_array_field(series: pd.Series) -> bool:
    """
    Detect if a field contains array data.

    Parameters:
    -----------
    series : pd.Series
        The series to analyze

    Returns:
    --------
    bool
        True if the field appears to contain array data
    """
    if len(series) == 0:
        return False

    # Get a sample of non-null values
    sample = series.dropna().sample(min(100, len(series.dropna())))

    array_pattern = re.compile(DataTypeDetection.ARRAY_REGEX)
    array_count = sum(
        1 for val in sample if isinstance(val, str) and array_pattern.match(val.strip())
    )

    # Consider it an array if at least 80% of the sample matches the array pattern
    return array_count >= 0.8 * len(sample)


def parse_array_field(value: Any, separator: str = ",") -> List[Any]:
    """
    Parse an array field value.

    Parameters:
    -----------
    value : Any
        The value to parse
    separator : str
        The separator character for array elements

    Returns:
    --------
    List[Any]
        The parsed array elements
    """
    if pd.isna(value):
        return []

    if not isinstance(value, str):
        return []

    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []

    # Remove brackets and split
    content = value[1:-1].strip()
    if not content:
        return []

    # Try to parse as JSON array first
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Fall back to simple splitting
        return [item.strip() for item in content.split(separator) if item.strip()]


def is_valid_email(email: str) -> bool:
    """
    Check if a string is a valid email address.

    Parameters:
    -----------
    email : str
        The email address to check

    Returns:
    --------
    bool
        True if the email is valid
    """
    if pd.isna(email) or not isinstance(email, str):
        return False

    email_pattern = re.compile(DataTypeDetection.EMAIL_REGEX)
    return bool(email_pattern.match(email.strip()))


def extract_email_domain(email: str) -> Optional[str]:
    """
    Extract the domain from an email address.

    Parameters:
    -----------
    email : str
        The email address

    Returns:
    --------
    str or None
        The domain, or None if the email is invalid
    """
    if not is_valid_email(email):
        return None

    return email.split("@")[1].lower()


def is_phone_number_format(phone: str) -> bool:
    """
    Check if a string matches the expected phone number format.

    Parameters:
    -----------
    phone : str
        The phone number to check

    Returns:
    --------
    bool
        True if the phone number format is valid
    """
    if pd.isna(phone) or not isinstance(phone, str):
        return False

    phone_pattern = re.compile(DataTypeDetection.PHONE_BASIC_REGEX)
    return bool(phone_pattern.match(phone.strip()))


def parse_phone_number(phone: str) -> Optional[Dict[str, Any]]:
    """
    Parse a phone number in the format (country_code,operator_code,number,"comment").

    Parameters:
    -----------
    phone : str
        The phone number to parse

    Returns:
    --------
    Dict[str, Any] or None
        The parsed phone components, or None if parsing fails
    """
    if pd.isna(phone) or not isinstance(phone, str):
        return None

    phone = phone.strip()
    pattern = re.compile(ProfilerConfig.PHONE_FORMAT_REGEX)
    match = pattern.match(phone)

    if not match:
        return {"is_valid": False, "original": phone, "error": "Invalid format"}

    country_code = match.group(1)
    operator_code = match.group(2)
    number = match.group(3)
    comment = match.group(5) if len(match.groups()) >= 5 and match.group(5) else ""

    return {
        "is_valid": True,
        "original": phone,
        "country_code": country_code,
        "operator_code": operator_code,
        "number": number,
        "comment": comment,
    }


def save_profiling_results(
    result: Dict[str, Any],
    profile_type: str,
    output_name: str,
    format: str = "json",
    include_timestamp: bool = True,
) -> str:
    """
    Saves profiling results for a specific profile type.

    Parameters:
    -----------
    result : dict
        Profiling results to save
    profile_type : str
        Type of profile (e.g., 'details', 'identification', 'contacts')
    output_name : str
        Base name for the output file
    format : str
        Output format: "json" or "csv" (default: "json")
    include_timestamp : bool
        Whether to include a timestamp in the filename (default: True)

    Returns:
    --------
    str
        Path to the saved file as a string
    """
    # Convert result to standard Python types for JSON serialization
    converted_result = convert_numpy_types(result)

    # Use our own function defined above
    file_path = save_profiling_result(
        result=converted_result,
        field_name=profile_type,
        output_name=output_name,
        format=format,
        include_timestamp=include_timestamp,
    )

    # Return path as string for compatibility
    return str(file_path)


def cleanup_memory(
    df: Optional[pd.DataFrame] = None,
    analyzed_df: Optional[pd.DataFrame] = None,
    values_dict: Optional[pd.DataFrame] = None,
    combinations_dict: Optional[pd.DataFrame] = None,
    analysis_results: Optional[Dict[str, Any]] = None,
    instance: Optional[Any] = None,
    **kwargs,
) -> None:
    """
    Clean up memory after operation completes.

    For large datasets, explicitly free memory by deleting
    references and optionally calling garbage collection.

    Parameters:
    -----------
    df : pd.DataFrame, optional
        Input DataFrame to clear from memory
    analysis_results : Dict[str, Any], optional
        Analysis results dictionary to clear from memory
    instance : Any, optional
        Class instance to clean up attributes from
    force_gc : bool, optional
        Whether to force garbage collection (default is False)
    """
    try:
        # Delete DataFrame references
        if df is not None:
            del df
        if analysis_results is not None:
            del analysis_results
        if analyzed_df is not None:
            del analyzed_df
        if values_dict is not None:
            del values_dict
        if combinations_dict is not None:
            del combinations_dict

        # cleanup memory from instance
        helpers.cleanup_memory(instance=instance)
    except Exception as e:
        logger.warning(f"Error cleanup_memory: {e}")


def validate_and_get_dataframe(
    data_source: DataSource, dataset_name: str, **kwargs: Any
) -> pd.DataFrame:
    """
    Validate data source and retrieve the main dataframe.

    Parameters:
    -----------
    data_source : DataSource
        The data source to validate
    dataset_name : str
        The name of the dataset to retrieve
    **kwargs : Any
        Additional keyword arguments to pass to the data loading function

    Returns:
    --------
    pd.DataFrame
        The validated dataframe

    Raises:
    -------
    ValueError
        If no valid dataframe is found or the field is missing
    """
    # Get DataFrame from the data source
    df = load_data_operation(data_source, dataset_name, **kwargs)
    if df is None:
        error_message = f"Failed to load input data!"
        raise ValueError(error_message)

    # Apply data types from data source
    try:
        df = data_source.apply_data_types(df, dataset_name)
    except ValueError as e:
        error_msg = f"Failed to apply data types for dataset '{dataset_name}': {str(e)}"
        raise ValueError(error_msg) from e

    except TypeError as e:
        error_msg = f"Invalid dataframe type for dataset '{dataset_name}': {str(e)}"
        raise TypeError(error_msg) from e

    return df


def save_dtypes_output(
    df: pd.DataFrame,
    result: OperationResult,
    reporter: Any,
    operation_name: str,
    task_dir: Path,
    output_filename: str,
    encryption_key: str = None,
    encryption_mode: str = None,
    task_logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Saves data types dataframe format to a JSON file.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe whose dtypes are to be saved
    result : OperationResult
        The operation result to add metrics and artifacts to
    reporter : Any
        The reporter to add artifacts to
    operation_name : str
        Name of the operation for description
    task_dir : Path
        Directory to save the output file
    output_filename : str
        Base name for the output file (default: "data_types_output.json")
    encryption_key : str, optional
        Encryption key for the output file, if any
    encryption_mode : str, optional
        Encryption mode for the output file, if any
    task_logger : logging.Logger, optional
        Logger to use for logging messages (default: None)

    Returns
    -------
    Path or None
        Path to saved file if success, otherwise None
    """
    try:
        # Use provided logger or default
        if task_logger is not None:
            logger = task_logger

        # Get the dtypes of the columns as a Series
        dtypes_series = df.dtypes

        # Convert the dtypes Series to a dictionary
        dtypes_dict = dtypes_series.astype(str).to_dict()

        # Generate standardized output filename with timestamp
        output_dir = task_dir / "metrics"
        dtype_filename = f"data_types_{output_filename}"
        dtype_path = output_dir / dtype_filename

        write_json(
            dtypes_dict,
            str(dtype_path),
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
        )

        result.add_metric(dtype_path.stem, dtypes_dict)

        result.add_artifact(
            artifact_type="json",
            path=dtype_path,
            description=f"Data types of output {operation_name}",
            category=Constants.Artifact_Category_Metrics,
        )

        if reporter:
            reporter.add_artifact(
                artifact_type="json",
                path=str(dtype_path),
                description=f"Data types of output {operation_name}",
            )

        logger.info(f"Dtypes output saved to: {dtype_path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to save dtypes format: {str(e)}")
        return False
