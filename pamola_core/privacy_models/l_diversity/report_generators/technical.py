"""
PAMOLA.CORE - L-Diversity Technical Reporting
---------------------------------------------
This module provides specialized functions for generating
technical reports for l-diversity anonymization.

Key Features:
- Detailed diversity metrics for different l-diversity types
- Group-level analysis
- Performance and scalability metrics
- Attribute-specific technical analysis
- Comparison with other privacy models

This module extends the main reporting functionality with
specialized technical reporting capabilities.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Import from pamola_core utilities
from pamola_core.utils__old_15_04.file_io import write_json, write_csv

# Configure logging
logger = logging.getLogger(__name__)


def generate_distinct_diversity_report(
        processor,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate technical report for distinct l-diversity

    Parameters:
    -----------
    processor : object
        L-Diversity processor instance
    data : pd.DataFrame, optional
        Input dataset (if not already processed by processor)
    quasi_identifiers : List[str], optional
        Quasi-identifier columns (required if data provided)
    sensitive_attributes : List[str], optional
        Sensitive attribute columns (required if data provided)
    output_path : str or Path, optional
        Path to save report (if None, report is only returned)

    Returns:
    --------
    Dict[str, Any]
        Distinct l-diversity technical report
    """
    # Import main report class
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport, generate_technical_report

    # Create base technical report
    base_report = generate_technical_report(
        processor,
        data,
        quasi_identifiers,
        sensitive_attributes
    )

    # Add distinct l-diversity specific metrics
    if data is not None and quasi_identifiers and sensitive_attributes:
        distinct_metrics = calculate_distinct_diversity_metrics(
            data, quasi_identifiers, sensitive_attributes
        )

        if distinct_metrics:
            base_report["distinct_l_diversity_analysis"] = distinct_metrics

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(base_report, str(path))
        logger.info(f"Distinct l-diversity technical report saved to {path}")

    return base_report


def generate_entropy_diversity_report(
        processor,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate technical report for entropy l-diversity

    Parameters:
    -----------
    processor : object
        L-Diversity processor instance
    data : pd.DataFrame, optional
        Input dataset (if not already processed by processor)
    quasi_identifiers : List[str], optional
        Quasi-identifier columns (required if data provided)
    sensitive_attributes : List[str], optional
        Sensitive attribute columns (required if data provided)
    output_path : str or Path, optional
        Path to save report (if None, report is only returned)

    Returns:
    --------
    Dict[str, Any]
        Entropy l-diversity technical report
    """
    # Import main report class
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport, generate_technical_report

    # Create base technical report
    base_report = generate_technical_report(
        processor,
        data,
        quasi_identifiers,
        sensitive_attributes
    )

    # Add entropy l-diversity specific metrics
    if data is not None and quasi_identifiers and sensitive_attributes:
        entropy_metrics = calculate_entropy_diversity_metrics(
            data, quasi_identifiers, sensitive_attributes
        )

        if entropy_metrics:
            base_report["entropy_l_diversity_analysis"] = entropy_metrics

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(base_report, str(path))
        logger.info(f"Entropy l-diversity technical report saved to {path}")

    return base_report


def generate_recursive_diversity_report(
        processor,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate technical report for recursive l-diversity

    Parameters:
    -----------
    processor : object
        L-Diversity processor instance
    data : pd.DataFrame, optional
        Input dataset (if not already processed by processor)
    quasi_identifiers : List[str], optional
        Quasi-identifier columns (required if data provided)
    sensitive_attributes : List[str], optional
        Sensitive attribute columns (required if data provided)
    output_path : str or Path, optional
        Path to save report (if None, report is only returned)

    Returns:
    --------
    Dict[str, Any]
        Recursive l-diversity technical report
    """
    # Import main report class
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport, generate_technical_report

    # Create base technical report
    base_report = generate_technical_report(
        processor,
        data,
        quasi_identifiers,
        sensitive_attributes
    )

    # Add recursive l-diversity specific metrics
    if data is not None and quasi_identifiers and sensitive_attributes:
        # Get c-value from processor or default to 1.0
        c_value = getattr(processor, 'c_value', 1.0)

        recursive_metrics = calculate_recursive_diversity_metrics(
            data, quasi_identifiers, sensitive_attributes, c_value
        )

        if recursive_metrics:
            base_report["recursive_l_diversity_analysis"] = recursive_metrics

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(base_report, str(path))
        logger.info(f"Recursive l-diversity technical report saved to {path}")

    return base_report


def generate_attribute_diversity_report(
        processor,
        sensitive_attribute: str,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate technical report for a specific sensitive attribute

    Parameters:
    -----------
    processor : object
        L-Diversity processor instance
    sensitive_attribute : str
        Sensitive attribute to analyze
    data : pd.DataFrame, optional
        Input dataset (if not already processed by processor)
    quasi_identifiers : List[str], optional
        Quasi-identifier columns (required if data provided)
    output_path : str or Path, optional
        Path to save report (if None, report is only returned)

    Returns:
    --------
    Dict[str, Any]
        Attribute-specific diversity report
    """
    # Get diversity type from processor
    diversity_type = getattr(processor, 'diversity_type', 'distinct')

    # Initialize report
    report = {
        "report_metadata": {
            "creation_time": datetime.now().isoformat(),
            "report_type": f"{diversity_type} l-diversity attribute analysis",
            "attribute": sensitive_attribute
        },
        "l_diversity_configuration": {
            "l_value": getattr(processor, 'l', 3),
            "diversity_type": diversity_type,
            "k_value": getattr(processor, 'k', 2)
        }
    }

    # Add c-value if diversity type is recursive
    if diversity_type == "recursive":
        report["l_diversity_configuration"]["c_value"] = getattr(processor, 'c_value', 1.0)

    # Calculate attribute-specific metrics
    if data is not None and quasi_identifiers:
        attribute_metrics = calculate_attribute_diversity_metrics(
            data, quasi_identifiers, sensitive_attribute, diversity_type
        )

        if attribute_metrics:
            report["attribute_analysis"] = attribute_metrics

    # Calculate group-level metrics
    if data is not None and quasi_identifiers:
        group_metrics = calculate_group_level_metrics(
            data, quasi_identifiers, sensitive_attribute, diversity_type
        )

        if group_metrics:
            report["group_analysis"] = group_metrics

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, str(path))
        logger.info(f"Attribute diversity report saved to {path}")

    return report


def calculate_distinct_diversity_metrics(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str]
) -> Dict[str, Any]:
    """
    Calculate detailed distinct l-diversity metrics

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns

    Returns:
    --------
    Dict[str, Any]
        Detailed distinct l-diversity metrics
    """
    # Group data by quasi-identifiers
    groups = data.groupby(quasi_identifiers)

    # Initialize metrics
    metrics = {
        "attribute_metrics": {},
        "group_size_distribution": {},
        "l_value_distribution": {}
    }

    # Calculate metrics for each sensitive attribute
    for sa in sensitive_attributes:
        if sa not in data.columns:
            continue

        # Calculate distinct counts for each group
        distinct_counts = []
        group_sizes = []

        for _, group in groups:
            if sa in group.columns:
                distinct_counts.append(group[sa].nunique())
                group_sizes.append(len(group))

        # Skip if no distinct counts
        if not distinct_counts:
            continue

        # Calculate attribute metrics
        min_distinct = min(distinct_counts)
        max_distinct = max(distinct_counts)
        avg_distinct = sum(distinct_counts) / len(distinct_counts)
        median_distinct = np.median(distinct_counts)

        # Calculate l-value distribution
        l_distribution = {}
        for l_value in sorted(set(distinct_counts)):
            count = distinct_counts.count(l_value)
            percentage = (count / len(distinct_counts)) * 100
            l_distribution[str(l_value)] = {
                "count": count,
                "percentage": percentage
            }

        # Calculate group size distribution
        size_distribution = {}
        unique_sizes = sorted(set(group_sizes))
        for size in unique_sizes:
            count = group_sizes.count(size)
            percentage = (count / len(group_sizes)) * 100
            size_distribution[str(size)] = {
                "count": count,
                "percentage": percentage
            }

        # Store attribute metrics
        metrics["attribute_metrics"][sa] = {
            "min_distinct": min_distinct,
            "max_distinct": max_distinct,
            "avg_distinct": avg_distinct,
            "median_distinct": median_distinct,
            "groups_analyzed": len(distinct_counts)
        }

        metrics["l_value_distribution"][sa] = l_distribution
        metrics["group_size_distribution"][sa] = size_distribution

    return metrics


def calculate_entropy_diversity_metrics(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str]
) -> Dict[str, Any]:
    """
    Calculate detailed entropy l-diversity metrics

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns

    Returns:
    --------
    Dict[str, Any]
        Detailed entropy l-diversity metrics
    """
    # Group data by quasi-identifiers
    groups = data.groupby(quasi_identifiers)

    # Initialize metrics
    metrics = {
        "attribute_metrics": {},
        "entropy_distribution": {},
        "effective_l_distribution": {}
    }

    # Calculate metrics for each sensitive attribute
    for sa in sensitive_attributes:
        if sa not in data.columns:
            continue

        # Calculate entropy for each group
        entropy_values = []
        effective_l_values = []

        for _, group in groups:
            if sa in group.columns:
                # Calculate value frequencies
                value_counts = group[sa].value_counts(normalize=True)

                # Calculate entropy
                entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                entropy_values.append(entropy)

                # Calculate effective l-value
                effective_l = np.exp(entropy) if entropy > 0 else 1
                effective_l_values.append(effective_l)

        # Skip if no entropy values
        if not entropy_values:
            continue

        # Calculate attribute metrics
        min_entropy = min(entropy_values)
        max_entropy = max(entropy_values)
        avg_entropy = sum(entropy_values) / len(entropy_values)
        median_entropy = np.median(entropy_values)

        min_effective_l = min(effective_l_values)
        max_effective_l = max(effective_l_values)
        avg_effective_l = sum(effective_l_values) / len(effective_l_values)
        median_effective_l = np.median(effective_l_values)

        # Calculate entropy distribution
        entropy_distribution = {}
        # Group entropy values into bins
        bins = np.linspace(0, max_entropy, 10)
        hist, bin_edges = np.histogram(entropy_values, bins=bins)

        for i in range(len(hist)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_key = f"{bin_start:.2f}-{bin_end:.2f}"

            entropy_distribution[bin_key] = {
                "count": int(hist[i]),
                "percentage": (hist[i] / len(entropy_values)) * 100
            }

        # Calculate effective l-value distribution
        effective_l_distribution = {}
        # Group effective l-values into bins
        bins = np.linspace(1, max(max_effective_l, 5), 10)
        hist, bin_edges = np.histogram(effective_l_values, bins=bins)

        for i in range(len(hist)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_key = f"{bin_start:.2f}-{bin_end:.2f}"

            effective_l_distribution[bin_key] = {
                "count": int(hist[i]),
                "percentage": (hist[i] / len(effective_l_values)) * 100
            }

        # Store attribute metrics
        metrics["attribute_metrics"][sa] = {
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "avg_entropy": avg_entropy,
            "median_entropy": median_entropy,
            "min_effective_l": min_effective_l,
            "max_effective_l": max_effective_l,
            "avg_effective_l": avg_effective_l,
            "median_effective_l": median_effective_l,
            "groups_analyzed": len(entropy_values)
        }

        metrics["entropy_distribution"][sa] = entropy_distribution
        metrics["effective_l_distribution"][sa] = effective_l_distribution

    return metrics


def calculate_recursive_diversity_metrics(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        c_value: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate detailed recursive l-diversity metrics

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns
    c_value : float, optional
        C-value for recursive l-diversity

    Returns:
    --------
    Dict[str, Any]
        Detailed recursive l-diversity metrics
    """
    # Group data by quasi-identifiers
    groups = data.groupby(quasi_identifiers)

    # Initialize metrics
    metrics = {
        "attribute_metrics": {},
        "c_value": c_value,
        "compliance_distribution": {}
    }

    # Get l-value (default to 3)
    l_value = 3

    # Calculate metrics for each sensitive attribute
    for sa in sensitive_attributes:
        if sa not in data.columns:
            continue

        # Analyze each group for recursive diversity
        compliant_groups = 0
        non_compliant_groups = 0
        c_ratios = []

        for _, group in groups:
            if sa in group.columns:
                # Get value frequencies
                value_counts = group[sa].value_counts()

                # Check if we have enough distinct values
                if len(value_counts) >= l_value:
                    # Sort frequencies in descending order
                    sorted_counts = value_counts.sort_values(ascending=False)

                    # Get most frequent value
                    most_frequent = sorted_counts.iloc[0]

                    # Get sum of l-1 least frequent values
                    least_frequent_sum = sorted_counts.iloc[-l_value + 1:].sum()

                    # Calculate c-ratio
                    if least_frequent_sum > 0:
                        c_ratio = most_frequent / least_frequent_sum
                        c_ratios.append(c_ratio)

                        # Check compliance
                        if c_ratio <= c_value:
                            compliant_groups += 1
                        else:
                            non_compliant_groups += 1

        # Skip if no c-ratios
        if not c_ratios:
            continue

        # Calculate attribute metrics
        min_c_ratio = min(c_ratios)
        max_c_ratio = max(c_ratios)
        avg_c_ratio = sum(c_ratios) / len(c_ratios)
        median_c_ratio = np.median(c_ratios)

        # Calculate compliance distribution
        compliance_distribution = {
            "compliant": {
                "count": compliant_groups,
                "percentage": (compliant_groups / (compliant_groups + non_compliant_groups)) * 100
                if (compliant_groups + non_compliant_groups) > 0 else 0
            },
            "non_compliant": {
                "count": non_compliant_groups,
                "percentage": (non_compliant_groups / (compliant_groups + non_compliant_groups)) * 100
                if (compliant_groups + non_compliant_groups) > 0 else 0
            }
        }

        # Store attribute metrics
        metrics["attribute_metrics"][sa] = {
            "min_c_ratio": min_c_ratio,
            "max_c_ratio": max_c_ratio,
            "avg_c_ratio": avg_c_ratio,
            "median_c_ratio": median_c_ratio,
            "compliant_groups": compliant_groups,
            "non_compliant_groups": non_compliant_groups,
            "total_groups": compliant_groups + non_compliant_groups
        }

        metrics["compliance_distribution"][sa] = compliance_distribution

    return metrics


def calculate_attribute_diversity_metrics(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        diversity_type: str = "distinct"
) -> Dict[str, Any]:
    """
    Calculate diversity metrics for a specific attribute

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attribute : str
        Sensitive attribute to analyze
    diversity_type : str, optional