"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Risk Scoring Utilities
Package:       pamola_core.metrics.commons.risk_scoring
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides functions for estimating re-identification risk in datasets based on
  direct identifiers, quasi-identifiers, and record uniqueness. Supports both
  provisional risk estimation (sampling-based) and full dataset risk evaluation.

Key Features:
  - Provisional risk scoring on sampled subsets (e.g., 10%)
  - Full risk assessment across entire dataset
  - Weighted scoring using tunable parameters
  - Confidence level estimation based on sample size
  - Breakdown of risk components (direct coverage, quasi coverage, uniqueness)

Design Principles:
  - Simple functional interface for quick integration
  - Tunable scoring model using sigmoid weighting
  - Interpretable outputs with overall risk score and component breakdown
  - Modular design for extension with additional risk measures

Dependencies:
  - pandas - DataFrame operations
  - numpy - mathematical functions and vectorized operations
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pamola_core.analysis.privacy_risk import _calculate_k_anonymity
from pamola_core.profiling.commons.attribute_utils import categorize_column_by_name
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)

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
        "description": "Critical sensitive fields including financial, medical, and biometric data",
        "keywords": [
            # Financial
            "salary", "income", "earnings", "revenue", "pay", "compensation", "bonus", "cost",
            "bank", "iban", "bic", "account", "credit_card", "debit", "payment_method",
            "transaction", "balance", "loan", "mortgage",

            # Medical
            "diagnosis", "disease", "health_condition", "illness", "disorder", "symptom",
            "treatment", "medication", "therapy", "mental",

            # Biometric
            "fingerprint", "face", "facial_recognition", "iris", "retina", "dna",
            "voice", "palmprint", "gait", "hand_geometry"
        ]
    },
}


def calculate_provisional_risk(
    df: pd.DataFrame,
    direct_identifiers: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
    sensitives: Optional[List[str]] = None,
    dictionary_path: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    sigmoid_midpoints: Optional[Dict[str, float]] = None,
    penalty_sensitive: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Quickly estimate the provisional privacy risk score of a dataset based on
    the presence of direct identifiers, quasi-identifiers, and uniqueness patterns.

    This metric provides a **lightweight risk estimate** without performing full
    privacy model calculations (e.g., l-diversity, t-closeness), useful for early-stage
    data profiling or automated pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    direct_identifiers : List[str], optional
        List of direct identifiers (e.g., name, SSN). If None, will attempt automatic detection.
    quasi_identifiers : List[str], optional
        List of quasi-identifiers (e.g., age, ZIP). If None, will attempt automatic detection.
    sensitives : List[str], optional
        List of sensitive attributes (e.g., disease). If None, will attempt automatic detection.
    dictionary_path : str, optional
        Path to a metadata dictionary for field detection (currently unused).
    weights : Dict[str, float], optional
        Custom weights for each risk component. Default:
            - direct_identifier : 0.50
            - quasi_identifier  : 0.30
            - uniqueness_estimate : 0.20
    sigmoid_midpoints : Dict[str, float], optional
        Custom sigmoid midpoints controlling the curve for each component.
        Default:
            - direct_identifier : 0.2
            - quasi_identifier  : 0.4
            - uniqueness_estimate : 0.1
    penalty_sensitive : float, optional
        Penalty added to risk score if sensitive attributes are detected. Default 0.1.

    Returns
    -------
    Dict[str, Any]
        Dictionary of provisional risk score.
    """
    # --- 1. Default Configuration ---
    if weights is None:
        weights = {
            "direct_identifier": 0.50,
            "quasi_identifier": 0.30,
            "uniqueness_estimate": 0.20,
        }

    if sigmoid_midpoints is None:
        sigmoid_midpoints = {
            "direct_identifier": 0.2,
            "quasi_identifier": 0.4,
            "uniqueness_estimate": 0.1,
        }

    if penalty_sensitive is None:
        penalty_sensitive = 0.1

    total_fields = len(df.columns)
    if total_fields == 0:
        return {
            "direct_identifiers_detected": [],
            "quasi_identifiers_detected": [],
            "sensitive_patterns_detected": [],
            "coverage_direct": 0.0,
            "coverage_quasi": 0.0,
            "uniqueness_estimate": 0.0,
            "provisional_score": 0,
            "confidence": 0.0,
            "k_anonymity": 0,
        }

    # --- 2. Detect fields by role if not explicitly provided ---
    if direct_identifiers is None:
        direct_identifiers = _detect_fields_by_role_category(
            df.columns, "DIRECT_IDENTIFIER"
        )

    if quasi_identifiers is None:
        quasi_identifiers = _detect_fields_by_role_category(
            df.columns, "QUASI_IDENTIFIER"
        )

    if sensitives is None:
        sensitives = _detect_fields_by_role_category(df.columns, "SENSITIVE_ATTRIBUTE")

    # --- 3. Direct Identifier Component ---
    coverage_direct = len(direct_identifiers) / total_fields
    sigmoid_direct = _sigmoid(
        coverage_direct, sigmoid_midpoints.get("direct_identifier", 0.2)
    )
    direct_component = weights.get("direct_identifier", 0.5) * sigmoid_direct

    # --- 4. Quasi-Identifier Component ---
    coverage_quasi = len(quasi_identifiers) / total_fields
    sigmoid_quasi = _sigmoid(
        coverage_quasi, sigmoid_midpoints.get("quasi_identifier", 0.4)
    )
    quasi_component = weights.get("quasi_identifier", 0.3) * sigmoid_quasi

    # --- 5. Uniqueness Estimate Component ---
    uniqueness_estimate = 0.0
    uniqueness_component = 0.0
    if quasi_identifiers:
        # Convert each row's QIs to tuple for uniqueness check
        tuple_series = df[quasi_identifiers].apply(tuple, axis=1)
        uniqueness_estimate = tuple_series.nunique() / len(tuple_series)
        sigmoid_uniqueness = _sigmoid(
            uniqueness_estimate, sigmoid_midpoints.get("uniqueness_estimate", 0.1)
        )
        uniqueness_component = (
            weights.get("uniqueness_estimate", 0.2) * sigmoid_uniqueness
        )

    # --- 6. Sensitive Penalty ---
    sensitive_penalty = penalty_sensitive if sensitives else 0.0

    # --- 7. Aggregate Provisional Risk ---
    provisional_score = (
        direct_component + quasi_component + uniqueness_component + sensitive_penalty
    )

    # Clamp between [0, 1] to avoid overflow due to penalty
    provisional_score = min(max(provisional_score, 0.0), 1.0)

    # --- 8. Supplementary metrics ---
    k_result = (
        _calculate_k_anonymity(df, quasi_identifiers) if quasi_identifiers else {"k": 0}
    )
    confidence_level = _calculate_confidence_level(df)

    # --- 9. Return structured result ---
    return {
        "direct_identifiers_detected": direct_identifiers,
        "quasi_identifiers_detected": quasi_identifiers,
        "sensitive_patterns_detected": sensitives,
        "coverage_direct": round(coverage_direct, 2),
        "coverage_quasi": round(coverage_quasi, 2),
        "uniqueness_estimate": round(uniqueness_estimate, 2),
        "provisional_score": int(provisional_score * 100),
        "confidence": confidence_level,
        "k_anonymity": int(k_result.get("k", 0)),
    }


def _sigmoid(x: float, midpoint: float = 0.5) -> float:
    """
    Logistic sigmoid function for smooth scaling of proportions.

    Parameters
    ----------
    x : float
        Input proportion (e.g., coverage value between 0 and 1).
    midpoint : float, default 0.5
        Inflection point where the output is 0.5. Controls horizontal shift.

    Returns
    -------
    float
        Scaled value between 0 and 1.
    """
    # Prevent overflow for large values of (x - midpoint)
    z = np.clip(x - midpoint, -50, 50)
    return 1 / (1 + np.exp(-z))


def _detect_fields_by_role_category(
    field_names: List[str],
    role_category: str,
    language: str = "en",
    dictionary_path: Optional[str] = None,
) -> List[str]:
    """
    Detect fields belonging to a specific role category based on their names
    using a keyword dictionary (either custom or default).

    Parameters
    ----------
    field_names : List[str]
        List of field/column names to categorize.
    role_category : str
        Target role category to detect.
        Supported values:
            - "DIRECT_IDENTIFIER"
            - "QUASI_IDENTIFIER"
            - "SENSITIVE_ATTRIBUTE"
    language : str, default "en"
        Language code for keyword matching. Currently not used for filtering,
        but can be extended for multilingual dictionaries.
    dictionary_path : Optional[str], default None
        Optional path to a custom dictionary file.
        If None, a default built-in dictionary is used.

    Returns
    -------
    List[str]
        List of field names that match the given role category.
    """
    categorized_fields = []

    # Load attribute dictionary (either custom or default)
    # TODO: Extend to support loading custom dictionaries from dictionary_path
    dictionary = {"categories": DEFAULT_ATTRIBUTE_ROLES}

    for field_name in field_names:
        # categorize_column_by_name returns a tuple like (role_category, matched_keyword)
        detected_category, _ = categorize_column_by_name(field_name, dictionary)

        # Match with the target category
        if detected_category == role_category:
            categorized_fields.append(field_name)

    return categorized_fields


def _calculate_confidence_level(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> str:
    """
    Calculate overall confidence level based on:
    - Sample size (number of rows)
    - Coverage consistency (percentage of non-null values)

    Confidence rules:
    -----------------
    Sample size:
        < 100     -> low
        100–600   -> medium
        > 600     -> high

    Coverage consistency (%):
        < 60      -> low
        60–85     -> medium
        > 85      -> high

    Combination:
        both high          -> high
        any low            -> low
        otherwise          -> medium

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.
    columns : List[str], optional
        Specific columns to evaluate. Defaults to all columns.

    Returns
    -------
    str
        One of {"low", "medium", "high"} representing the confidence level.
    """
    if columns is None:
        columns = df.columns

    # --- Sample size confidence ---
    sample_size = len(df)
    if sample_size < 100:
        sample_confidence = "low"
    elif sample_size <= 600:
        sample_confidence = "medium"
    else:
        sample_confidence = "high"

    # --- Coverage confidence ---
    total_cells = len(df) * len(columns)
    if total_cells == 0:
        return "low"  # No data → low confidence

    non_null_count = df[columns].notnull().sum().sum()
    coverage_pct = (non_null_count / total_cells) * 100

    if coverage_pct < 60:
        coverage_confidence = "low"
    elif coverage_pct <= 85:
        coverage_confidence = "medium"
    else:
        coverage_confidence = "high"

    # --- Combine both factors ---
    if sample_confidence == "high" and coverage_confidence == "high":
        return "high"
    if sample_confidence == "low" or coverage_confidence == "low":
        return "low"
    return "medium"
