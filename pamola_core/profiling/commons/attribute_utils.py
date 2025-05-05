"""
Attribute utilities for data profiling in the HHR project.

This module provides utility functions for analyzing and categorizing attributes
of datasets based on their names, content, and statistical properties.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)

# Default attribute role categories
DEFAULT_ATTRIBUTE_ROLES = {
    "DIRECT_IDENTIFIER": {
        "description": "Explicit identifiers (name, passport, email, UID) - unique and directly identify",
        "keywords": [
            "id", "uuid", "uid", "guid", "identifier", "record_id",
            "email", "e-mail", "mail_address",
            "phone", "telephone", "cell", "mobile", "contact_number",
            "passport", "ssn", "sin", "nino", "driver_license", "id_number", "national_id",
            "first_name", "last_name", "middle_name", "surname", "name_latin", "full_name"
        ]
    },
    "QUASI_IDENTIFIER": {
        "description": "Not unique individually, but allow identification in combination (birth_day, region, etc.)",
        "keywords": [
            "address", "street", "building", "apt", "flat", "postal_code", "zip", "location_detail",
            "region", "province", "state", "district", "oblast", "kra", "municipality",
            "city", "town", "village", "settlement", "locality", "metro_area",
            "country", "nation",
            "birth", "dob", "birth_date", "birth_day",
            "gender", "sex",
            "job", "position", "occupation", "profession", "role", "rank",
            "company", "employer", "organization", "department",
            "experience", "work_history", "job_years",
            "education", "degree", "university", "college", "school", "faculty"
        ]
    },
    "SENSITIVE_ATTRIBUTE": {
        "description": "Confidential or sensitive fields (health, finance, behavior)",
        "keywords": [
            "ethnicity", "race", "nationality",
            "religion", "faith", "belief",
            "salary", "income", "earnings", "revenue", "pay", "compensation", "bonus", "cost",
            "diagnosis", "disease", "health_condition", "illness", "disorder", "symptom",
            "treatment", "medication", "therapy", "mental",
            "bank", "iban", "bic", "account", "credit_card", "debit", "payment_method",
            "transaction", "balance", "loan", "mortgage",
            "vote", "opinion", "choice", "survey_answer", "rating", "reaction"
        ]
    },
    "INDIRECT_IDENTIFIER": {
        "description": "Long texts, behavioral profiles, addresses - potentially identify through content analysis",
        "keywords": [
            "geo", "longitude", "latitude", "location", "coordinates", "gps",
            "ip", "mac", "device_id", "browser_fingerprint", "hardware", "imei", "udid",
            "photo", "image", "avatar", "face_encoding", "biometric", "voice", "fingerprint",
            "resume_text", "description", "comments", "notes", "feedback", "free_text", "bio", "profile_summary",
            "device", "browser", "os", "platform", "screen", "resolution"
        ]
    },
    "NON_SENSITIVE": {
        "description": "Other fields not containing sensitive information",
        "keywords": [
            "timestamp", "date", "datetime", "created_at", "updated_at", "log_time", "event_time",
            "status", "stage", "state", "approved", "rejected", "active", "inactive",
            "event", "log", "action", "activity", "operation",
            "tag", "label", "category", "cluster"
        ]
    }
}

# Statistical thresholds
DEFAULT_THRESHOLDS = {
    "entropy_high": 5.0,
    "entropy_mid": 3.0,
    "uniqueness_high": 0.9,
    "uniqueness_low": 0.2,
    "text_length_long": 100,
    "mvf_threshold": 2
}


def load_attribute_dictionary(file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load attribute dictionary from a user-defined or default JSON file.

    Priority of dictionary loading:
    1. Explicitly provided file_path
    2. Standard project directories
    3. Fallback to default dictionary

    Args:
        file_path (str, Path, optional): Path to custom dictionary JSON file

    Returns:
        Dict[str, Any]: Dictionary with attribute roles, keywords, and thresholds
    """
    # First, check if user provided an explicit path
    if file_path:
        file_path = Path(file_path)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Loading user-defined dictionary from {file_path}")
                    return _validate_dictionary(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading user dictionary {file_path}: {e}")
                # Optionally, you might want to raise an error here if user-provided dict is critical

    # If no explicit path, search in standard project directories
    possible_paths = [
        # Current working directory
        Path.cwd() / 'DATA' / 'external_dictionaries' / 'attribute_roles_dictionary.json',
        Path.cwd() / 'data' / 'external_dictionaries' / 'attribute_roles_dictionary.json',

        # Relative to script location
        Path(__file__).parent.parent.parent / 'DATA' / 'external_dictionaries' / 'attribute_roles_dictionary.json',
        Path(__file__).parent.parent.parent / 'data' / 'external_dictionaries' / 'attribute_roles_dictionary.json',

        # Other potential locations
        Path.home() / 'HHR_PROJECT' / 'DATA' / 'external_dictionaries' / 'attribute_roles_dictionary.json'
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    logger.info(f"Found dictionary at {path}")
                    return _validate_dictionary(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading dictionary from {path}: {e}")

    # Fallback to default dictionary if no custom dictionary found
    logger.warning("No custom attribute dictionary found. Using default dictionary.")
    return {
        "categories": DEFAULT_ATTRIBUTE_ROLES,
        "statistical_thresholds": DEFAULT_THRESHOLDS
    }


def _validate_dictionary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and complete the loaded dictionary.

    Args:
        data (Dict[str, Any]): Loaded dictionary data

    Returns:
        Dict[str, Any]: Validated and completed dictionary
    """
    # Ensure required keys are present
    if "categories" not in data:
        logger.warning("Missing 'categories' in dictionary. Using defaults.")
        data["categories"] = DEFAULT_ATTRIBUTE_ROLES

    if "statistical_thresholds" not in data:
        logger.warning("Missing 'statistical_thresholds' in dictionary. Using defaults.")
        data["statistical_thresholds"] = DEFAULT_THRESHOLDS

    return data


def infer_data_type(series: pd.Series) -> str:
    """
    Infer the logical data type of a series.

    Parameters:
    -----------
    series : pd.Series
        The series to analyze

    Returns:
    --------
    str
        Inferred data type: 'numeric', 'categorical', 'datetime', 'text', 'boolean', 'long_text', 'mvf'
    """
    # Check for empty series
    if series.empty:
        return 'unknown'

    # Get pandas dtype
    dtype = series.dtype

    # Handle boolean type
    if pd.api.types.is_bool_dtype(dtype):
        return 'boolean'

    # Handle numeric types
    if pd.api.types.is_numeric_dtype(dtype):
        # Check if the values are mostly integers or have limited unique values
        if pd.api.types.is_integer_dtype(dtype) or series.nunique() <= min(10, int(len(series) * 0.05)):
            return 'categorical'
        return 'numeric'

    # Handle datetime
    if pd.api.types.is_datetime64_dtype(dtype):
        return 'datetime'

    # For object/string types, we need more analysis
    non_null = series.dropna()

    # If empty after dropping nulls, return unknown
    if len(non_null) == 0:
        return 'unknown'

    # Check if it contains lists (MVF)
    if non_null.apply(lambda x: isinstance(x, (list, tuple))).any():
        return 'mvf'

    # Check for MVF in string format
    mvf_pattern = re.compile(r'\[.*]|\{.*}|.*;.*|.*,.*|.*\|.*')
    sample = non_null.sample(min(100, len(non_null)))
    mvf_count = sample.str.contains(mvf_pattern, regex=True, na=False).sum()

    if mvf_count >= len(sample) * 0.5:  # If at least 50% match MVF pattern
        sample_values = sample.iloc[0] if not sample.empty else ""
        avg_token_length = np.mean([len(str(token).strip()) for token in str(sample_values).split(',')])

        # If average token length is small, likely MVF
        if avg_token_length < 15:
            return 'mvf'

    # Check for text length
    avg_length = non_null.astype(str).str.len().mean()

    if avg_length > DEFAULT_THRESHOLDS['text_length_long']:
        return 'long_text'
    else:
        # If small number of unique values, likely categorical
        if non_null.nunique() <= min(10, int(len(non_null) * 0.1)):
            return 'categorical'
        return 'text'


def calculate_entropy(series: pd.Series) -> float:
    """
    Calculate Shannon entropy of a series.

    Parameters:
    -----------
    series : pd.Series
        The series to calculate entropy for

    Returns:
    --------
    float
        Entropy value (higher means more diverse/random distribution)
    """
    # Drop null values
    values = series.dropna()

    # If empty after dropping nulls, return 0
    if len(values) == 0:
        return 0.0

    # Calculate value frequencies
    value_counts = values.value_counts(normalize=True)

    # Calculate entropy: -sum(p_i * log2(p_i))
    entropy = -np.sum(value_counts * np.log2(value_counts))

    return entropy


def calculate_normalized_entropy(series: pd.Series) -> float:
    """
    Calculate normalized entropy (0-1 scale).

    Parameters:
    -----------
    series : pd.Series
        The series to calculate normalized entropy for

    Returns:
    --------
    float
        Normalized entropy value between 0 and 1
    """
    # Calculate raw entropy
    entropy = calculate_entropy(series)

    # Get number of unique values
    n_unique = series.nunique()

    # If no unique values or just one, entropy is 0
    if n_unique <= 1:
        return 0.0

    # Calculate max possible entropy for this number of unique values
    max_entropy = math.log2(n_unique)

    # Avoid division by zero
    if max_entropy == 0:
        return 0.0

    # Normalize entropy to 0-1 scale
    normalized_entropy = entropy / max_entropy

    return normalized_entropy


def calculate_uniqueness_ratio(series: pd.Series) -> float:
    """
    Calculate uniqueness ratio (unique values / total values).

    Parameters:
    -----------
    series : pd.Series
        The series to calculate uniqueness ratio for

    Returns:
    --------
    float
        Uniqueness ratio between 0 and 1
    """
    # Drop null values
    values = series.dropna()

    # If empty after dropping nulls, return 0
    if len(values) == 0:
        return 0.0

    # Calculate uniqueness ratio
    uniqueness_ratio = values.nunique() / len(values)

    return uniqueness_ratio


def is_mvf_field(series: pd.Series) -> bool:
    """
    Check if a field contains multi-valued data (lists, delimited strings).

    Parameters:
    -----------
    series : pd.Series
        The series to check

    Returns:
    --------
    bool
        True if the field appears to contain multi-valued data
    """
    # Drop null values
    non_null = series.dropna()

    # If empty after dropping nulls, return False
    if len(non_null) == 0:
        return False

    # Check for list/tuple types
    if non_null.apply(lambda x: isinstance(x, (list, tuple))).any():
        return True

    # Check for common MVF string patterns
    mvf_pattern = re.compile(r'\[.*]|\{.*}|.*;.*|.*,.*|.*\|.*')

    # Sample the data to avoid processing very large series
    sample = non_null.sample(min(100, len(non_null)))
    mvf_count = sample.str.contains(mvf_pattern, regex=True, na=False).sum()

    # If at least 50% match MVF pattern
    if mvf_count >= len(sample) * 0.5:
        # Check average token length to distinguish from free text
        sample_values = sample.iloc[0] if not sample.empty else ""
        avg_token_length = np.mean([len(str(token).strip()) for token in str(sample_values).split(',')])

        # If average token length is small, likely MVF
        if avg_token_length < 15:
            return True

    return False


def analyze_column_values(df: pd.DataFrame, column: str, sample_size: int = 10) -> Dict[str, Any]:
    """
    Analyze comprehensive statistics and characteristics of a specific column in a DataFrame.

    This function performs in-depth analysis of a column, extracting statistical metrics,
    identifying data types, handling multi-valued fields, and sampling values.

    Args:
        df (pd.DataFrame): Input DataFrame containing the column to analyze
        column (str): Name of the column to be analyzed
        sample_size (int, optional): Number of sample values to extract. Defaults to 10.

    Returns:
        Dict[str, Any]: Comprehensive dictionary of column statistics and metadata
    """
    try:
        # Extract the specific column series from the DataFrame
        series = df[column]

        # Compute comprehensive column statistics in a single dictionary creation
        stats = {
            "count": len(series),  # Total number of records
            "missing_count": series.isna().sum(),  # Number of missing values
            "missing_rate": series.isna().mean(),  # Percentage of missing values
            "unique_values": series.nunique(),  # Number of unique values
            "data_type": str(series.dtype),  # Original pandas data type
            "inferred_type": infer_data_type(series),  # Logical data type inference
            "is_mvf": is_mvf_field(series),  # Check if multi-valued field
            "entropy": calculate_entropy(series),  # Shannon entropy calculation
            "normalized_entropy": calculate_normalized_entropy(series),  # Normalized entropy
            "uniqueness_ratio": calculate_uniqueness_ratio(series)  # Unique values ratio
        }

        # Sample values extraction with special handling for different data types
        if stats["inferred_type"] == "long_text":
            # Truncate long text samples to improve readability
            samples = series.dropna().sample(min(sample_size, len(series.dropna()))).apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
            ).tolist()
        else:
            # Standard sampling for other data types
            samples = series.dropna().sample(min(sample_size, len(series.dropna()))).tolist()

        # Add sampled values to statistics
        stats["samples"] = samples

        # Additional metrics for text-based columns
        if stats["inferred_type"] in ["text", "long_text"]:
            text_lengths = series.dropna().astype(str).str.len()
            stats["avg_text_length"] = text_lengths.mean()
            stats["max_text_length"] = text_lengths.max()

        # Specialized analysis for multi-valued fields (MVF)
        if stats["is_mvf"]:
            try:
                # Initialize MVF processing variables
                flattened = []
                delimiter = None

                # Handle list/tuple type MVF
                if series.apply(lambda x: isinstance(x, (list, tuple))).any():
                    flattened = [item for sublist in series.dropna() for item in sublist]
                else:
                    # Detect and process string-based MVF
                    non_null_series = series.dropna()
                    if not non_null_series.empty:
                        sample_value = str(non_null_series.iloc[0])
                        # Try common delimiters
                        for delim in [',', ';', '|']:
                            if delim in sample_value:
                                delimiter = delim
                                break

                        if delimiter:
                            for val in non_null_series:
                                if isinstance(val, str):
                                    items = [item.strip() for item in val.split(delimiter)]
                                    flattened.extend(items)

                # Count unique values in MVF
                stats["mvf_unique_values"] = len(set(flattened)) if flattened else None

                # Calculate average number of items per record
                if delimiter:
                    mvf_lengths = []
                    for x in series.dropna():
                        if isinstance(x, (list, tuple)):
                            mvf_lengths.append(len(x))
                        elif isinstance(x, str):
                            mvf_lengths.append(len(x.split(delimiter)))

                    # Safely calculate mean
                    stats["mvf_avg_items_per_record"] = np.mean(mvf_lengths) if mvf_lengths else None
                else:
                    stats["mvf_avg_items_per_record"] = None

            except Exception as e:
                # Log any issues during MVF processing
                logger.debug(f"Error analyzing MVF field {column}: {str(e)}")

        return stats

    except Exception as e:
        # Comprehensive error handling
        logger.warning(f"Error analyzing column {column}: {str(e)}")
        return {
            "error": str(e),
            "count": len(df),
            "missing_count": df[column].isna().sum() if column in df.columns else None,
            "samples": []
        }


def categorize_column_by_name(column_name: str,
                              dictionary: Dict[str, Any],
                              language: str = "en") -> Tuple[str, float]:
    """
    Categorize a column based on its name using the keyword dictionary.

    Parameters:
    -----------
    column_name : str
        The name of the column to categorize
    dictionary : Dict[str, Any]
        Dictionary with attribute roles and keywords
    language : str
        Language code for keyword matching (default: "en")

    Returns:
    --------
    Tuple[str, float]
        Tuple containing (role_category, confidence_score)
    """
    # Normalize column name (lowercase, underscores to spaces)
    normalized_name = column_name.lower().replace('_', ' ')

    # Initialize best match
    best_match = ("NON_SENSITIVE", 0.0)

    # Check each role category
    categories = dictionary.get("categories", {})
    for role, role_info in categories.items():
        # Get keywords for this role and language
        keywords = []

        # Handle both old (flat list) and new (language-specific) keyword formats
        if isinstance(role_info.get("keywords", []), list):
            # Old format - flat list
            keywords = role_info.get("keywords", [])
        else:
            # New format - language-specific lists
            keywords = role_info.get("keywords", {}).get(language, [])

        # Check for exact matches
        for keyword in keywords:
            if keyword.lower() == normalized_name or keyword.lower() in normalized_name.split():
                return (role, 1.0)

        # Check for partial matches
        for keyword in keywords:
            if keyword.lower() in normalized_name:
                confidence = len(keyword) / len(normalized_name)
                if confidence > best_match[1]:
                    best_match = (role, confidence)

    # Check for specific patterns if defined
    for role, role_info in categories.items():
        if "patterns" in role_info:
            for pattern in role_info["patterns"]:
                if re.match(pattern, column_name):
                    return (role, 0.95)  # High confidence for regex matches

    return best_match


def categorize_column_by_statistics(stats: Dict[str, Any],
                                    dictionary: Dict[str, Any]) -> Tuple[str, float]:
    """
    Categorize a column based on its statistical properties.

    Parameters:
    -----------
    stats : Dict[str, Any]
        Dictionary with column statistics
    dictionary : Dict[str, Any]
        Dictionary with attribute roles and thresholds

    Returns:
    --------
    Tuple[str, float]
        Tuple containing (role_category, confidence_score)
    """
    # Get thresholds
    thresholds = dictionary.get("statistical_thresholds", DEFAULT_THRESHOLDS)

    # Extract key statistics
    entropy = stats.get("entropy", 0)
    normalized_entropy = stats.get("normalized_entropy", 0)
    uniqueness_ratio = stats.get("uniqueness_ratio", 0)
    inferred_type = stats.get("inferred_type", "unknown")

    # High entropy + high uniqueness often indicates identifiers
    if uniqueness_ratio > thresholds["uniqueness_high"] and entropy > thresholds["entropy_high"]:
        return ("DIRECT_IDENTIFIER", 0.8)

    # Long text likely contains indirect identifying information
    if inferred_type == "long_text":
        return ("INDIRECT_IDENTIFIER", 0.7)

    # Medium-high entropy with moderate uniqueness often indicates quasi-identifiers
    if uniqueness_ratio > 0.3 and entropy > thresholds["entropy_mid"]:
        return ("QUASI_IDENTIFIER", 0.6)

    # Low entropy fields are often non-sensitive
    if normalized_entropy < 0.3 and uniqueness_ratio < thresholds["uniqueness_low"]:
        return ("NON_SENSITIVE", 0.7)

    # MVF fields often contain quasi-identifying information
    if stats.get("is_mvf", False):
        return ("QUASI_IDENTIFIER", 0.5)

    # Default to NON_SENSITIVE with low confidence
    return ("NON_SENSITIVE", 0.3)


def resolve_category_conflicts(semantic_result: Tuple[str, float],
                               statistical_result: Tuple[str, float]) -> Tuple[str, float, Dict[str, Any]]:
    """
    Resolve conflicts between semantic and statistical categorization.

    Parameters:
    -----------
    semantic_result : Tuple[str, float]
        Result from semantic analysis (role_category, confidence_score)
    statistical_result : Tuple[str, float]
        Result from statistical analysis (role_category, confidence_score)

    Returns:
    --------
    Tuple[str, float, Dict[str, Any]]
        Tuple containing (final_role_category, confidence_score, conflict_details)
    """
    semantic_category, semantic_confidence = semantic_result
    statistical_category, statistical_confidence = statistical_result

    # No conflict case
    if semantic_category == statistical_category:
        return (semantic_category, max(semantic_confidence, statistical_confidence), {})

    # Conflict case - prepare details
    conflict_details = {
        "semantic_category": semantic_category,
        "semantic_confidence": semantic_confidence,
        "statistical_category": statistical_category,
        "statistical_confidence": statistical_confidence
    }

    # Semantic analysis has priority, especially for high-confidence matches
    if semantic_confidence >= 0.7:
        return (semantic_category, semantic_confidence, conflict_details)

    # Prioritize more sensitive categories when confidence is similar
    category_priority = {
        "DIRECT_IDENTIFIER": 5,
        "SENSITIVE_ATTRIBUTE": 4,
        "QUASI_IDENTIFIER": 3,
        "INDIRECT_IDENTIFIER": 2,
        "NON_SENSITIVE": 1
    }

    semantic_priority = category_priority.get(semantic_category, 0)
    statistical_priority = category_priority.get(statistical_category, 0)

    # If statistical result is more sensitive and has decent confidence
    if statistical_priority > semantic_priority and statistical_confidence >= 0.6:
        return (statistical_category, statistical_confidence, conflict_details)

    # Default to semantic result with conflict details
    return (semantic_category, semantic_confidence, conflict_details)


def categorize_column(df: pd.DataFrame,
                      column: str,
                      dictionary: Dict[str, Any],
                      language: str = "en",
                      sample_size: int = 10) -> Dict[str, Any]:
    """
    Analyze and categorize a column based on both name and statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the column
    column : str
        The name of the column to categorize
    dictionary : Dict[str, Any]
        Dictionary with attribute roles and thresholds
    language : str
        Language code for keyword matching
    sample_size : int
        Number of sample values to return

    Returns:
    --------
    Dict[str, Any]
        Dictionary with categorization results and analysis
    """
    # Analyze column values
    stats = analyze_column_values(df, column, sample_size)

    # Categorize by name (semantic)
    semantic_result = categorize_column_by_name(column, dictionary, language)

    # Categorize by statistics
    statistical_result = categorize_column_by_statistics(stats, dictionary)

    # Resolve conflicts
    final_category, confidence, conflict_details = resolve_category_conflicts(
        semantic_result, statistical_result
    )

    # Prepare result
    result = {
        "column_name": column,
        "role": final_category,
        "confidence": confidence,
        "statistics": stats,
        "semantic_analysis": {
            "category": semantic_result[0],
            "confidence": semantic_result[1]
        },
        "statistical_analysis": {
            "category": statistical_result[0],
            "confidence": statistical_result[1]
        }
    }

    # Add conflict details if any
    if conflict_details:
        result["conflict_details"] = conflict_details

    return result


def analyze_dataset_attributes(df: pd.DataFrame,
                               dictionary: Optional[Dict[str, Any]] = None,
                               dictionary_path: Optional[Union[str, Path]] = None,
                               language: str = "en",
                               sample_size: int = 10,
                               max_columns: Optional[int] = None,
                               id_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze and categorize all columns in a dataset.

    Args:
        df (pd.DataFrame): The DataFrame to analyze
        dictionary (Dict[str, Any], optional): Dictionary with attribute roles and thresholds
        dictionary_path (str or Path, optional): Path to the dictionary file
        language (str): Language code for keyword matching
        sample_size (int): Number of sample values to return per column
        max_columns (int, optional): Maximum number of columns to analyze
        id_column (str, optional): Name of ID column for record-level analysis

    Returns:
        Dict[str, Any]: Dictionary with dataset attribute analysis results
    """
    # Load dictionary if not provided
    if dictionary is None:
        dictionary = load_attribute_dictionary(dictionary_path)

    # Initialize results
    results = {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "analyzed_at": pd.Timestamp.now().isoformat()
        },
        "columns": {},
        "summary": {
            "DIRECT_IDENTIFIER": 0,
            "QUASI_IDENTIFIER": 0,
            "SENSITIVE_ATTRIBUTE": 0,
            "INDIRECT_IDENTIFIER": 0,
            "NON_SENSITIVE": 0
        },
        "column_groups": {
            "DIRECT_IDENTIFIER": [],
            "QUASI_IDENTIFIER": [],
            "SENSITIVE_ATTRIBUTE": [],
            "INDIRECT_IDENTIFIER": [],
            "NON_SENSITIVE": []
        }
    }

    # Limit columns if needed
    columns_to_analyze = df.columns[:max_columns] if max_columns else df.columns

    # Analyze each column
    for column in columns_to_analyze:
        try:
            column_result = categorize_column(df, column, dictionary, language, sample_size)
            results["columns"][column] = column_result

            # Update summary counts
            role = column_result["role"]
            results["summary"][role] += 1

            # Add to column groups
            results["column_groups"][role].append(column)

        except Exception as e:
            logger.warning(f"Error analyzing column {column}: {str(e)}")
            results["columns"][column] = {"error": str(e)}

    # Calculate overall entropy and uniqueness metrics
    if results["columns"]:
        # Safely extract entropy and uniqueness values
        entropy_values = [
            col.get("statistics", {}).get("entropy", 0)
            for col in results["columns"].values()
            if "statistics" in col and "entropy" in col["statistics"]
        ]

        uniqueness_values = [
            col.get("statistics", {}).get("uniqueness_ratio", 0)
            for col in results["columns"].values()
            if "statistics" in col and "uniqueness_ratio" in col["statistics"]
        ]

        # Calculate averages only if values exist
        results["dataset_metrics"] = {
            "avg_entropy": np.mean(entropy_values) if entropy_values else 0,
            "avg_uniqueness": np.mean(uniqueness_values) if uniqueness_values else 0
        }

    # Record conflicts
    conflicts = [
        {"column": col_name, "details": col_data.get("conflict_details", {})}
        for col_name, col_data in results["columns"].items()
        if "conflict_details" in col_data
    ]

    if conflicts:
        results["conflicts"] = conflicts

    return results
