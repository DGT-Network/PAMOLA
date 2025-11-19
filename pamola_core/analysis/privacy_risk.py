"""
PAMOLA.CORE - Privacy Risk Assessment Module
------------------------------------------------
Module:        Privacy Risk Assessment
Package:       pamola_core.analysis
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Privacy-aware risk scoring utilities for tabular datasets. Provides an
  immediate, automatically computed risk score using classical privacy models
  (k-anonymity, l-diversity, t-closeness) together with simple simulated attack
  metrics (re-identification, attribute disclosure, membership inference).
  Normalizes outputs for easy integration with downstream monitoring and
  governance workflows.

Key Features:
  - Computes k-anonymity, l-diversity and t-closeness per dataset
  - Simulates linkage, attribute inference and membership inference attacks
  - Combines metrics into a configurable weighted risk score
  - Returns structured dictionaries suitable for JSON serialization
  - Includes logging and safe defaults for production use

Dependencies:
  - pandas  - DataFrame operations
  - numpy   - Numeric operations and entropy calculations
  - scipy.stats.wasserstein_distance - Earth Mover's Distance (t-closeness)
  - typing  - Type hints
  - pamola_core.utils.logging - Module logging helper
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


def calculate_full_risk(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Perform a full privacy risk assessment of a dataset using formal privacy models
    and simulated attack risks.

    This function combines multiple privacy metrics (k-anonymity, l-diversity)
    with simulated attack-based risks (re-identification, attribute disclosure,
    membership inference) to compute a weighted overall risk score.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to assess.
    quasi_identifiers : List[str]
        Columns used to define equivalence classes (e.g., age, zip code).
    sensitive_attributes : List[str]
        Columns containing sensitive information (e.g., disease).
    weights : Optional[Dict[str, float]], optional
        Custom weights for each risk component. Keys and default values:
            - "k_anonymity" : 0.40
            - "l_diversity" : 0.10
            - "attribute_disclosure_risk" : 0.30
            - "membership_inference_risk" : 0.20

        All weights should sum to 1.0. If None, defaults are used.

    Returns
    -------
    Dict[str, Any]
        Dictionary of full risk assessment.
    """
    # --- 1. Default weight configuration ---
    if weights is None:
        weights = {
            "k_anonymity": 0.40,
            "l_diversity": 0.10,
            "attribute_disclosure_risk": 0.30,
            "membership_inference_risk": 0.20,
        }

    # Validate weight sum
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0 (got {total_weight}).")

    # --- 2. Formal Privacy Metrics ---
    # k-anonymity: measures minimum group size under quasi-identifiers
    k_anonymity = _calculate_k_anonymity(df, quasi_identifiers)

    # l-diversity: measures sensitive value diversity within equivalence classes
    l_diversity_result, max_entropy = _calculate_l_diversity(
        df, quasi_identifiers, sensitive_attributes
    )

    # --- 3. Simulated Attack Risks ---
    # Re-identification risk (linkage attack): % of individuals uniquely identified
    reidentification_risk = _simulate_linkage_attack(
        df, quasi_identifiers, sensitive_attributes
    )

    # Attribute disclosure: ability to infer sensitive attributes from QIs
    attribute_disclosure_risk = _simulate_attribute_inference(
        df, quasi_identifiers, sensitive_attributes
    )

    # Membership inference: detect whether an individual is in the dataset
    membership_inference_risk = _simulate_membership_inference(
        df, quasi_identifiers, []
    )

    # --- 4. Weighted Risk Aggregation ---
    # Convert k-anonymity to risk: smaller k → higher risk
    k_risk_component = weights["k_anonymity"] * (
        1 / k_anonymity["k"] if k_anonymity["k"] > 0 else 1.0
    )

    # l-diversity risk: lower entropy relative to max → higher risk
    l_risk_component = weights["l_diversity"] * (
        1 - (l_diversity_result["entropy"] / max_entropy if max_entropy > 0 else 0.0)
    )

    attr_disclosure_component = (
        weights["attribute_disclosure_risk"] * attribute_disclosure_risk
    )
    membership_component = (
        weights["membership_inference_risk"] * membership_inference_risk
    )

    # Final risk score (0-1)
    risk_score = (
        k_risk_component
        + l_risk_component
        + attr_disclosure_component
        + membership_component
    )

    # --- 5. Return result object ---
    return {
        "reidentification_risk": reidentification_risk,
        "attribute_disclosure_risk": attribute_disclosure_risk,
        "membership_inference_risk": membership_inference_risk,
        "k_anonymity": k_anonymity,
        "l_diversity": l_diversity_result,
        "risk_assessment": int(min(risk_score, 1.0) * 100),
    }


def _calculate_k_anonymity(
    df: pd.DataFrame, quasi_identifiers: List[str]
) -> Dict[str, Any]:
    """
    Calculate k-anonymity for a given dataset.

    k-anonymity is the minimum number of records in any equivalence class
    formed by grouping on quasi-identifiers. Higher k means better anonymity.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.
    quasi_identifiers : List[str]
        Columns used to form equivalence classes.

    Returns
    -------
    Dict[str, Any]
        Dictionary of k-anonymity.
    """
    # --- 1. Handle empty data ---
    if df is None or df.empty:
        return {
            "k": 0,
            "quasi_identifiers": quasi_identifiers,
            "equivalence_classes": 0,
            "smallest_class": 0,
            "records_in_smallest_classes": 0,
        }

    # --- 2. Validate QI columns ---
    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier must be provided.")
    missing_cols = [c for c in quasi_identifiers if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in DataFrame.")

    # --- 3. Compute equivalence class sizes ---
    equivalence_class_sizes = (
        df.groupby(quasi_identifiers, observed=True).size().rename("group_size")
    )

    # --- 4. Compute k-anonymity and related stats ---
    k_value = int(equivalence_class_sizes.min())
    total_classes = int(equivalence_class_sizes.shape[0])
    smallest_class_size = k_value

    # All groups that are exactly size = k
    records_in_smallest_classes = int(
        equivalence_class_sizes[equivalence_class_sizes == k_value].sum()
    )

    # --- 5. Return results ---
    return {
        "k": k_value,
        "quasi_identifiers": quasi_identifiers,
        "equivalence_classes": total_classes,
        "smallest_class": smallest_class_size,
        "records_in_smallest_classes": records_in_smallest_classes,
    }


def _calculate_l_diversity(
    df: pd.DataFrame, quasi_identifiers: List[str], sensitive_attributes: List[str]
) -> Tuple[Dict[str, Any], float]:
    """
    Calculate l-diversity and entropy-based diversity of a dataset.

    l-diversity measures how well sensitive attributes are protected within
    each equivalence class formed by quasi-identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.
    quasi_identifiers : List[str]
        Columns used to form equivalence classes.
    sensitive_attributes : List[str]
        Columns containing sensitive information.

    Returns
    -------
    Tuple(Dict[str, Any], float)
        Dictionary of l-diversity, max entropy.
    """
    # --- 1. Validate input ---
    if df is None or df.empty:
        return {
            "l": 0,
            "sensitive_attributes": sensitive_attributes,
            "entropy": 0.0,
        }, 0.0

    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier must be provided.")

    if not sensitive_attributes:
        raise ValueError("At least one sensitive attribute must be provided.")

    missing_qi = [c for c in quasi_identifiers if c not in df.columns]
    missing_sa = [c for c in sensitive_attributes if c not in df.columns]
    if missing_qi or missing_sa:
        raise KeyError(f"Missing columns in DataFrame: {missing_qi + missing_sa}")

    # --- 2. Group by quasi-identifiers to form equivalence classes ---
    grouped = df.groupby(quasi_identifiers, observed=True)

    # --- 3. Compute l-diversity for each equivalence class ---
    l_values = [
        group[sensitive_attributes].apply(tuple, axis=1).nunique()
        for _, group in grouped
    ]

    # Handle case where no equivalence class exists
    if not l_values:
        l_diversity = 0
    else:
        l_diversity = int(min(l_values))

    # --- 4. Compute global entropy for each sensitive attribute ---
    entropy_values = []
    for attr in sensitive_attributes:
        freqs = df[attr].value_counts(normalize=True)
        entropy = -np.sum(freqs * np.log2(freqs))
        entropy_values.append(entropy)

    avg_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
    max_entropy = float(np.max(entropy_values)) if entropy_values else 0.0

    # --- 5. Return structured result ---
    return {
        "l": l_diversity,
        "sensitive_attributes": sensitive_attributes,
        "entropy": avg_entropy,
    }, max_entropy


def _calculate_t_closeness(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
    distance_metric: str = "EMD",
) -> Dict[str, Any]:
    """
    Calculate t-closeness of a dataset.

    t-closeness measures the distance between the distribution of sensitive
    attributes within each equivalence class (defined by quasi-identifiers)
    and the overall distribution of sensitive attributes in the dataset.

    A dataset satisfies t-closeness if for each equivalence class, this
    distance does not exceed a threshold t.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.
    quasi_identifiers : List[str]
        Columns used to form equivalence classes.
    sensitive_attributes : List[str]
        Columns containing sensitive information.
    distance_metric : str, optional
        Metric for distribution distance. Currently supports:
        - "EMD" : Earth Mover's Distance (Wasserstein distance)

    Returns
    -------
    Dict[str, Any]
        Dictionary of t-closeness.
    """
    # --- 1. Validate input ---
    if df is None or df.empty:
        return {"t": 0.0, "distance_metric": distance_metric, "num_classes": 0}

    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier must be provided.")

    if not sensitive_attributes:
        raise ValueError("At least one sensitive attribute must be provided.")

    missing_cols = [
        c for c in quasi_identifiers + sensitive_attributes if c not in df.columns
    ]
    if missing_cols:
        raise KeyError(f"Missing columns in DataFrame: {missing_cols}")

    # --- 2. Overall distribution of sensitive attributes ---
    overall_dist = (
        df[sensitive_attributes].value_counts(normalize=True, sort=False).sort_index()
    )

    # --- 3. Compute distance for each equivalence class ---
    t_values = []
    grouped = df.groupby(quasi_identifiers, observed=True)

    for _, group in grouped:
        if group.empty:
            continue

        class_dist = (
            group[sensitive_attributes]
            .value_counts(normalize=True, sort=False)
            .sort_index()
        )

        # Align the two distributions
        all_indices = overall_dist.index.union(class_dist.index)
        overall_aligned = overall_dist.reindex(all_indices, fill_value=0)
        class_aligned = class_dist.reindex(all_indices, fill_value=0)

        if distance_metric.upper() == "EMD":
            distance_value = wasserstein_distance(
                np.arange(len(all_indices)),  # Fake positions
                np.arange(len(all_indices)),
                u_weights=overall_aligned.values,
                v_weights=class_aligned.values,
            )
        else:
            raise NotImplementedError(
                f"Distance metric '{distance_metric}' is not supported."
            )

        t_values.append(distance_value)

    t_closeness = max(t_values) if t_values else 0.0

    return {
        "t": round(float(t_closeness), 6),
        "distance_metric": distance_metric,
        "num_classes": len(t_values),
    }


def _simulate_linkage_attack(
    df: pd.DataFrame, quasi_identifiers: List[str], sensitive_attributes: List[str]
) -> float:
    """
    Simulate Linkage Attack Risk (Re-identification Risk).

    Linkage attack risk estimates the proportion of records that can be
    uniquely re-identified by linking quasi-identifiers (QIs) and possibly
    sensitive attributes. A record is considered at risk if it falls into
    a singleton equivalence class (group size = 1).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    quasi_identifiers : List[str]
        List of quasi-identifier columns used for grouping.
    sensitive_attributes : List[str]
        List of sensitive attribute columns (optional, may be empty).

    Returns
    -------
    float
        Re-identification risk ∈ [0, 1].
        - 0.0 → No record is unique under the given QIs.
        - 1.0 → All records are unique (maximal risk).
    """
    # --- 1. Validate input ---
    if df is None or df.empty:
        return 0.0

    if not quasi_identifiers and not sensitive_attributes:
        raise ValueError(
            "At least one quasi-identifier or sensitive attribute must be provided."
        )

    # --- 2. Determine grouping keys ---
    group_cols = list(quasi_identifiers) + list(sensitive_attributes)
    group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        # None of the provided columns exist in df
        return 0.0

    # --- 3. Compute equivalence class sizes ---
    group_sizes = df.groupby(group_cols, observed=True).size().rename("group_size")

    # --- 4. Identify singleton groups (group_size == 1) ---
    singleton_groups = group_sizes[group_sizes == 1]

    # Number of records in singleton groups == number of such groups (each size=1)
    singleton_count = len(singleton_groups)

    # --- 5. Compute risk ---
    risk = singleton_count / len(df)

    return float(round(risk, 4))


def _simulate_attribute_inference(
    df: pd.DataFrame, quasi_identifiers: List[str], sensitive_attributes: List[str]
) -> float:
    """
    Simulate Attribute Inference Risk.

    Attribute disclosure risk measures how often sensitive attributes
    can be inferred deterministically from quasi-identifiers.

    Specifically, if within an equivalence class defined by QIs, the sensitive
    attribute has only one unique value, then that value can be inferred perfectly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to analyze.
    quasi_identifiers : List[str]
        Columns used to form equivalence classes.
    sensitive_attributes : List[str]
        Sensitive attribute columns to check for determinism within groups.

    Returns
    -------
    float
        Attribute disclosure risk in [0, 1]:
        - 0.0 = No deterministic sensitive values per group.
        - 1.0 = All groups allow deterministic inference of ≥1 sensitive attribute.
    """
    # --- 1. Validate input ---
    if df is None or df.empty:
        return 0.0

    if not quasi_identifiers and not sensitive_attributes:
        raise ValueError(
            "At least one quasi-identifier or sensitive attribute must be provided."
        )

    if not quasi_identifiers or not sensitive_attributes:
        # no QIs → can't form equivalence classes
        # no sensitive attrs → no risk to calculate
        return 0.0

    # --- 2. Group by quasi-identifiers ---
    # For each group, count number of unique sensitive values per attribute
    nunique_per_group = df.groupby(quasi_identifiers, observed=True)[
        sensitive_attributes
    ].nunique(
        dropna=False
    )  # count NaN as unique if present

    # --- 3. Identify deterministic groups ---
    # A group is deterministic if ANY sensitive attribute has exactly one unique value
    deterministic_flags = (nunique_per_group == 1).any(axis=1)

    # --- 4. Compute risk ---
    # Risk = proportion of deterministic groups over total groups
    attribute_disclosure_risk = deterministic_flags.mean()

    return float(round(attribute_disclosure_risk, 4))


def _simulate_membership_inference(
    df: pd.DataFrame, quasi_identifiers: List[str], sensitive_attributes: List[str]
) -> float:
    """
    Simulate Membership Inference Risk.

    Membership inference risk estimates the likelihood that an adversary can infer
    whether a specific individual's data is part of the dataset. A record is considered
    at risk if it forms a **unique equivalence class** (group size = 1) when grouped
    by quasi-identifiers and sensitive attributes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to evaluate.
    quasi_identifiers : List[str]
        Columns that indirectly identify individuals (e.g., age, ZIP code, gender).
    sensitive_attributes : List[str]
        Sensitive columns used together with QIs to form equivalence classes.

    Returns
    -------
    float
        Membership inference risk ∈ [0, 1], representing the proportion of records
        that belong to unique equivalence classes.
        - 0.0 → No unique records (low risk)
        - 1.0 → All records unique (maximum risk)
    """
    # --- 1. Handle empty DataFrame ---
    if df is None or df.empty:
        return 0.0

    # --- 2. Validate and filter grouping columns ---
    grouping_cols = list(quasi_identifiers) + list(sensitive_attributes)
    if not grouping_cols:
        raise ValueError(
            "At least one quasi-identifier or sensitive attribute must be provided."
        )

    missing_cols = [c for c in grouping_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in DataFrame.")

    # --- 3. Compute equivalence class sizes ---
    class_sizes = df.groupby(grouping_cols, observed=True).size().rename("group_size")

    # --- 4. Identify unique (singleton) classes ---
    unique_classes = class_sizes[class_sizes == 1]

    # Number of singleton records == number of singleton classes
    unique_count = len(unique_classes)

    # --- 5. Compute membership risk ---
    risk = unique_count / len(df)

    return float(round(risk, 4))
