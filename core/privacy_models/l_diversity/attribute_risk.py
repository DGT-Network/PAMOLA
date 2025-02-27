"""
PAMOLA.CORE - L-Diversity Attribute Disclosure Risk
---------------------------------------------------
This module provides specialized functions for evaluating attribute disclosure
risk in l-diversity anonymized datasets. It focuses on the risk of revealing
sensitive attribute values even when identities are protected by k-anonymity.

Key Features:
- Attribute disclosure risk calculation
- Predictability analysis
- Skewness and homogeneity assessment
- Value distribution analysis
- Detailed attribute risk profiling

This module extends the core privacy risk assessment with attribute-specific
analysis capabilities.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class AttributeDisclosureRiskAnalyzer:
    """
    Specialized analyzer for attribute disclosure risk

    Provides detailed analysis of disclosure risk for specific sensitive
    attributes, focusing on value predictability and distribution.
    """

    def __init__(self,
                 l_threshold: int = 3,
                 diversity_type: str = "distinct",
                 c_value: float = 1.0,
                 high_risk_threshold: float = 50.0):
        """
        Initialize Attribute Disclosure Risk Analyzer

        Parameters:
        -----------
        l_threshold : int, optional
            The minimum l value considered acceptable (default: 3)
        diversity_type : str, optional
            Type of l-diversity to use (default: "distinct")
        c_value : float, optional
            Parameter for recursive (c,l)-diversity (default: 1.0)
        high_risk_threshold : float, optional
            Threshold for defining high risk (default: 50.0)
        """
        self.l_threshold = l_threshold
        self.diversity_type = diversity_type
        self.c_value = c_value
        self.high_risk_threshold = high_risk_threshold
        self.logger = logging.getLogger(__name__)

    def calculate_attribute_disclosure_risk(self,
                                            data: pd.DataFrame,
                                            quasi_identifiers: List[str],
                                            sensitive_attribute: str,
                                            **kwargs) -> Dict[str, Any]:
        """
        Calculate comprehensive attribute disclosure risk

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attribute : str
            Sensitive attribute column to analyze
        **kwargs : dict
            Additional risk assessment parameters

        Returns:
        --------
        Dict[str, Any]
            Comprehensive attribute disclosure risk metrics
        """
        try:
            self.logger.info(f"Calculating attribute disclosure risk for: {sensitive_attribute}")

            # Validate input
            if sensitive_attribute not in data.columns:
                raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in dataset")

            # Extract parameters
            l_threshold = kwargs.get('l_threshold', self.l_threshold)
            diversity_type = kwargs.get('diversity_type', self.diversity_type)
            c_value = kwargs.get('c_value', self.c_value)
            detailed = kwargs.get('detailed', False)

            # Calculate diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(
                data, quasi_identifiers, sensitive_attribute,
                diversity_type, l_threshold, c_value
            )

            # Calculate predictability metrics
            predictability_metrics = self._calculate_predictability_metrics(
                data, quasi_identifiers, sensitive_attribute
            )

            # Calculate value distribution metrics
            distribution_metrics = self._calculate_distribution_metrics(
                data, sensitive_attribute
            )

            # Calculate vulnerability metrics
            vulnerability_metrics = self._calculate_vulnerability_metrics(
                data, quasi_identifiers, sensitive_attribute,
                diversity_metrics, predictability_metrics
            )

            # Combine all metrics
            risk_metrics = {
                'diversity': diversity_metrics,
                'predictability': predictability_metrics,
                'distribution': distribution_metrics,
                'vulnerability': vulnerability_metrics,
                'overall_risk': self._calculate_overall_risk(
                    diversity_metrics, predictability_metrics, distribution_metrics
                )
            }

            # Add detailed group metrics if requested
            if detailed:
                risk_metrics['group_details'] = self._calculate_group_details(
                    data, quasi_identifiers, sensitive_attribute,
                    diversity_type, l_threshold, c_value
                )

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Attribute disclosure risk calculation error: {e}")
            raise

    def _calculate_diversity_metrics(self,
                                     data: pd.DataFrame,
                                     quasi_identifiers: List[str],
                                     sensitive_attribute: str,
                                     diversity_type: str,
                                     l_threshold: int,
                                     c_value: float) -> Dict[str, Any]:
        """
        Calculate diversity metrics for the attribute

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attribute : str
            Sensitive attribute column to analyze
        diversity_type : str
            Type of l-diversity to assess
        l_threshold : int
            Minimum acceptable diversity level
        c_value : float
            Parameter for recursive (c,l)-diversity

        Returns:
        --------
        Dict[str, Any]
            Diversity metrics
        """
        # Group data by quasi-identifiers
        grouped = data.groupby(quasi_identifiers)

        # Initialize metrics
        diversity_values = []
        below_threshold = 0
        total_groups = len(grouped)

        # Process each group
        for _, group in grouped:
            # Calculate diversity for this group
            if diversity_type == "entropy":
                # Entropy l-diversity
                value_counts = group[sensitive_attribute].value_counts(normalize=True)
                entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                diversity_value = entropy
                threshold = np.log(l_threshold)

            elif diversity_type == "recursive":
                # Recursive (c,l)-diversity
                value_counts = group[sensitive_attribute].value_counts()

                if len(value_counts) < l_threshold:
                    diversity_value = len(value_counts)
                    compliant = False
                else:
                    # Get l-1 least frequent values
                    least_frequent = value_counts.nsmallest(l_threshold - 1)
                    least_sum = least_frequent.sum()

                    # Get most frequent value
                    most_frequent = value_counts.nlargest(1).iloc[0]

                    if least_sum > 0:
                        # Check recursive criterion
                        compliant = most_frequent <= c_value * least_sum
                    else:
                        compliant = False

                    diversity_value = l_threshold if compliant else len(value_counts)

                threshold = l_threshold

            else:
                # Distinct l-diversity
                diversity_value = group[sensitive_attribute].nunique()
                threshold = l_threshold

            # Store diversity value
            diversity_values.append(diversity_value)

            # Check if below threshold
            if (diversity_type == "entropy" and diversity_value < np.log(l_threshold)) or \
                    (diversity_type != "entropy" and diversity_value < l_threshold):
                below_threshold += 1

        # Calculate statistics
        if diversity_values:
            min_diversity = min(diversity_values)
            max_diversity = max(diversity_values)
            mean_diversity = np.mean(diversity_values)
            median_diversity = np.median(diversity_values)
        else:
            min_diversity = max_diversity = mean_diversity = median_diversity = 0

        # Calculate compliance
        if diversity_type == "entropy":
            compliant = min_diversity >= np.log(l_threshold)
        else:
            compliant = min_diversity >= l_threshold

        # Calculate risk percentage
        risk_percentage = (below_threshold / total_groups * 100) if total_groups > 0 else 0

        return {
            'diversity_type': diversity_type,
            'l_threshold': l_threshold,
            'c_value': c_value if diversity_type == "recursive" else None,
            'min_diversity': float(min_diversity),
            'max_diversity': float(max_diversity),
            'mean_diversity': float(mean_diversity),
            'median_diversity': float(median_diversity),
            'groups_below_threshold': int(below_threshold),
            'total_groups': int(total_groups),
            'risk_percentage': float(risk_percentage),
            'compliant': bool(compliant)
        }

    def _calculate_predictability_metrics(self,
                                          data: pd.DataFrame,
                                          quasi_identifiers: List[str],
                                          sensitive_attribute: str) -> Dict[str, Any]:
        """
        Calculate predictability metrics for the attribute

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attribute : str
            Sensitive attribute column to analyze

        Returns:
        --------
        Dict[str, Any]
            Predictability metrics
        """
        # Group data by quasi-identifiers
        groups = data.groupby(quasi_identifiers)

        # Calculate total groups
        total_groups = len(groups)

        # Calculate homogeneous groups (groups with a single sensitive value)
        homogeneous_groups = 0
        max_frequency_sum = 0

        # Most common value overall
        overall_mode = data[sensitive_attribute].mode().iloc[0] if len(data[sensitive_attribute].mode()) > 0 else None
        aligned_with_mode = 0

        # Process each group
        for _, group in groups:
            # Calculate number of distinct values in this group
            distinct_values = group[sensitive_attribute].nunique()

            # Check if group is homogeneous (only one value)
            if distinct_values == 1:
                homogeneous_groups += 1

            # Calculate max frequency in group
            value_counts = group[sensitive_attribute].value_counts(normalize=True)
            if not value_counts.empty:
                max_frequency = value_counts.max()
                max_frequency_sum += max_frequency

                # Check if most frequent value matches overall mode
                group_mode = value_counts.idxmax()
                if group_mode == overall_mode:
                    aligned_with_mode += 1

        # Calculate average max frequency
        avg_max_frequency = max_frequency_sum / total_groups if total_groups > 0 else 0

        # Calculate mode alignment percentage
        mode_alignment = aligned_with_mode / total_groups * 100 if total_groups > 0 else 0

        # Calculate homogeneous percentage
        homogeneous_percentage = homogeneous_groups / total_groups * 100 if total_groups > 0 else 0

        # Calculate baseline predictability (if guessing the most common value)
        overall_counts = data[sensitive_attribute].value_counts(normalize=True)
        baseline_predictability = overall_counts.max() * 100 if not overall_counts.empty else 0

        # Calculate disclosure gain (how much QIs improve prediction)
        disclosure_gain = avg_max_frequency * 100 - baseline_predictability

        return {
            'unique_combinations': int(total_groups),
            'homogeneous_groups': int(homogeneous_groups),
            'homogeneous_percentage': float(homogeneous_percentage),
            'avg_max_frequency': float(avg_max_frequency),
            'mode_alignment_percentage': float(mode_alignment),
            'baseline_predictability': float(baseline_predictability),
            'disclosure_gain': float(disclosure_gain)
        }

    def _calculate_distribution_metrics(self,
                                        data: pd.DataFrame,
                                        sensitive_attribute: str) -> Dict[str, Any]:
        """
        Calculate value distribution metrics for the attribute

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        sensitive_attribute : str
            Sensitive attribute column to analyze

        Returns:
        --------
        Dict[str, Any]
            Distribution metrics
        """
        # Calculate value counts
        value_counts = data[sensitive_attribute].value_counts(normalize=True)

        # Calculate number of unique values
        unique_values = len(value_counts)

        # Calculate entropy of distribution
        entropy = -sum(p * np.log(p) for p in value_counts if p > 0)

        # Calculate skewness of distribution
        if len(value_counts) > 1:
            top_freq = value_counts.iloc[0]
            avg_other_freq = value_counts.iloc[1:].mean()
            if avg_other_freq > 0:
                skewness = top_freq / avg_other_freq
            else:
                skewness = float('inf')
        else:
            skewness = float('inf')

        # Calculate most and least frequent values
        most_frequent = {}
        least_frequent = {}

        # Get top 3 values
        for value, count in value_counts.nlargest(3).items():
            most_frequent[str(value)] = float(count)

        # Get bottom 3 values
        for value, count in value_counts.nsmallest(3).items():
            least_frequent[str(value)] = float(count)

        # Calculate effective number of classes
        effective_num_classes = np.exp(entropy) if entropy > 0 else 1

        return {
            'unique_values': int(unique_values),
            'entropy': float(entropy),
            'effective_num_classes': float(effective_num_classes),
            'skewness': float(skewness),
            'most_frequent_values': most_frequent,
            'least_frequent_values': least_frequent
        }

    def _calculate_vulnerability_metrics(self,
                                         data: pd.DataFrame,
                                         quasi_identifiers: List[str],
                                         sensitive_attribute: str,
                                         diversity_metrics: Dict[str, Any],
                                         predictability_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate vulnerability metrics for the attribute

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attribute : str
            Sensitive attribute column to analyze
        diversity_metrics : Dict[str, Any]
            Previously calculated diversity metrics
        predictability_metrics : Dict[str, Any]
            Previously calculated predictability metrics

        Returns:
        --------
        Dict[str, Any]
            Vulnerability metrics
        """
        # Extract key metrics
        min_diversity = diversity_metrics.get('min_diversity', 0)
        homogeneous_percentage = predictability_metrics.get('homogeneous_percentage', 0)
        disclosure_gain = predictability_metrics.get('disclosure_gain', 0)

        # Calculate skewness-based vulnerability
        if 'skewness' in diversity_metrics:
            skewness = diversity_metrics['skewness']
            skewness_vulnerability = min(100, skewness * 20)  # Scale skewness to 0-100
        else:
            skewness_vulnerability = 0

        # Calculate diversity-based vulnerability
        diversity_type = diversity_metrics.get('diversity_type', 'distinct')
        l_threshold = diversity_metrics.get('l_threshold', 3)

        if diversity_type == 'entropy':
            # For entropy, vulnerability is based on exp(entropy) vs l_threshold
            threshold = np.log(l_threshold)
            if min_diversity <= 0:
                diversity_vulnerability = 100
            else:
                diversity_vulnerability = max(0, min(100, (threshold - min_diversity) / threshold * 100))
        else:
            # For distinct and recursive
            if min_diversity <= 0:
                diversity_vulnerability = 100
            else:
                diversity_vulnerability = max(0, min(100, (l_threshold - min_diversity) / l_threshold * 100))

            if diversity_vulnerability < 0:
                diversity_vulnerability = 0

        # Calculate homogeneity-based vulnerability
        homogeneity_vulnerability = homogeneous_percentage

        # Calculate disclosure gain vulnerability
        disclosure_vulnerability = max(0, min(100, disclosure_gain))

        # Combine vulnerabilities with weights
        overall_vulnerability = (
                0.4 * diversity_vulnerability +
                0.3 * homogeneity_vulnerability +
                0.2 * disclosure_vulnerability +
                0.1 * skewness_vulnerability
        )

        return {
            'diversity_vulnerability': float(diversity_vulnerability),
            'homogeneity_vulnerability': float(homogeneity_vulnerability),
            'disclosure_vulnerability': float(disclosure_vulnerability),
            'skewness_vulnerability': float(skewness_vulnerability),
            'overall_vulnerability': float(overall_vulnerability)
        }

    def _calculate_overall_risk(self,
                                diversity_metrics: Dict[str, Any],
                                predictability_metrics: Dict[str, Any],
                                distribution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall attribute disclosure risk

        Parameters:
        -----------
        diversity_metrics : Dict[str, Any]
            Previously calculated diversity metrics
        predictability_metrics : Dict[str, Any]
            Previously calculated predictability metrics
        distribution_metrics : Dict[str, Any]
            Previously calculated distribution metrics

        Returns:
        --------
        Dict[str, Any]
            Overall risk assessment
        """
        # Extract key metrics
        min_diversity = diversity_metrics.get('min_diversity', 0)
        risk_percentage = diversity_metrics.get('risk_percentage', 0)
        homogeneous_percentage = predictability_metrics.get('homogeneous_percentage', 0)
        disclosure_gain = predictability_metrics.get('disclosure_gain', 0)
        entropy = distribution_metrics.get('entropy', 0)

        # Weighted risk calculation
        diversity_risk = min(100, 100 / min_diversity) if min_diversity > 0 else 100

        # Calculate overall risk score
        risk_score = (
                0.5 * diversity_risk +
                0.2 * homogeneous_percentage +
                0.2 * disclosure_gain +
                0.1 * (100 - min(100, entropy * 20))  # Lower entropy = higher risk
        )

        # Determine risk level
        if risk_score < 20:
            risk_level = "Very Low"
        elif risk_score < 40:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Moderate"
        elif risk_score < 80:
            risk_level = "High"
        else:
            risk_level = "Very High"

        # Determine compliance
        diversity_type = diversity_metrics.get('diversity_type', 'distinct')
        l_threshold = diversity_metrics.get('l_threshold', 3)
        compliant = diversity_metrics.get('compliant', False)

        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'compliant': bool(compliant),
            'interpretation': self._interpret_risk(risk_score, diversity_type, l_threshold, compliant)
        }

    def _interpret_risk(self,
                        risk_score: float,
                        diversity_type: str,
                        l_threshold: int,
                        compliant: bool) -> str:
        """
        Interpret attribute disclosure risk

        Parameters:
        -----------
        risk_score : float
            Overall risk score
        diversity_type : str
            Type of l-diversity
        l_threshold : int
            L-threshold value
        compliant : bool
            Whether the attribute is compliant

        Returns:
        --------
        str
            Human-readable risk interpretation
        """
        # Base interpretation on risk score
        if risk_score < 20:
            base = f"Very Low Risk: Excellent protection of sensitive values."
        elif risk_score < 40:
            base = f"Low Risk: Good protection of sensitive values."
        elif risk_score < 60:
            base = f"Moderate Risk: Acceptable for many scenarios."
        elif risk_score < 80:
            base = f"High Risk: Significant chance of attribute disclosure."
        else:
            base = f"Very High Risk: Sensitive values are poorly protected."

        # Add compliance information
        if compliant:
            compliance = f" Meets {diversity_type} {l_threshold}-diversity requirements."
        else:
            compliance = f" Does not meet {diversity_type} {l_threshold}-diversity requirements."

        return base + compliance

    def _calculate_group_details(self,
                                 data: pd.DataFrame,
                                 quasi_identifiers: List[str],
                                 sensitive_attribute: str,
                                 diversity_type: str,
                                 l_threshold: int,
                                 c_value: float) -> Dict[str, Any]:
        """
        Calculate detailed metrics for each quasi-identifier group

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attribute : str
            Sensitive attribute column to analyze
        diversity_type : str
            Type of l-diversity to assess
        l_threshold : int
            Minimum acceptable diversity level
        c_value : float
            Parameter for recursive (c,l)-diversity

        Returns:
        --------
        Dict[str, Any]
            Detailed group metrics
        """
        # Group data by quasi-identifiers
        grouped = data.groupby(quasi_identifiers)

        # Initialize results
        group_details = {}

        # Process each group
        for group_name, group in grouped:
            # Create a key for the group
            if isinstance(group_name, tuple):
                group_key = "_".join(str(x) for x in group_name)
            else:
                group_key = str(group_name)

            # Calculate diversity
            if diversity_type == "entropy":
                # Entropy l-diversity
                value_counts = group[sensitive_attribute].value_counts(normalize=True)
                entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                diversity_value = entropy
                effective_diversity = np.exp(entropy) if entropy > 0 else 1
                compliant = entropy >= np.log(l_threshold)

                # Risk calculation
                risk = min(100, 100 / effective_diversity) if effective_diversity > 0 else 100

            elif diversity_type == "recursive":
                # Recursive (c,l)-diversity
                value_counts = group[sensitive_attribute].value_counts()

                if len(value_counts) < l_threshold:
                    diversity_value = len(value_counts)
                    compliant = False
                    risk = 100
                else:
                    # Get l-1 least frequent values
                    least_frequent = value_counts.nsmallest(l_threshold - 1)
                    least_sum = least_frequent.sum()

                    # Get most frequent value
                    most_frequent = value_counts.nlargest(1).iloc[0]

                    if least_sum > 0:
                        # Check recursive criterion
                        c_actual = most_frequent / least_sum
                        compliant = c_actual <= c_value

                        # Calculate risk
                        if compliant:
                            risk = 100 * (c_actual / c_value)
                        else:
                            risk = 100
                    else:
                        compliant = False
                        risk = 100

                    diversity_value = l_threshold if compliant else len(value_counts)

            else:
                # Distinct l-diversity
                diversity_value = group[sensitive_attribute].nunique()
                compliant = diversity_value >= l_threshold

                # Risk calculation
                risk = min(100, 100 / diversity_value) if diversity_value > 0 else 100

            # Calculate most common value
            top_value = group[sensitive_attribute].mode().iloc[0] if len(
                group[sensitive_attribute].mode()) > 0 else None
            top_frequency = (group[sensitive_attribute] == top_value).mean() if top_value is not None else 0

            # Store group details
            group_details[group_key] = {
                'group_size': int(len(group)),
                'diversity_value': float(diversity_value),
                'compliant': bool(compliant),
                'risk': float(risk),
                'top_value': str(top_value) if top_value is not None else None,
                'top_frequency': float(top_frequency),
                'unique_values': int(group[sensitive_attribute].nunique())
            }

        return group_details


# Utility functions

def calculate_attribute_disclosure_risk(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        diversity_type: str = "distinct",
        l_threshold: int = 3,
        c_value: float = 1.0,
        detailed: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Calculate attribute disclosure risk for a specific attribute

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attribute : str
        Sensitive attribute column to analyze
    diversity_type : str, optional
        Type of l-diversity to assess (default: "distinct")
    l_threshold : int, optional
        Minimum acceptable diversity level (default: 3)
    c_value : float, optional
        Parameter for recursive (c,l)-diversity (default: 1.0)
    detailed : bool, optional
        Whether to include detailed group metrics (default: False)
    **kwargs : dict
        Additional parameters for risk calculation

    Returns:
    --------
    Dict[str, Any]
        Comprehensive attribute disclosure risk metrics
    """
    analyzer = AttributeDisclosureRiskAnalyzer(
        l_threshold=l_threshold,
        diversity_type=diversity_type,
        c_value=c_value
    )

    return analyzer.calculate_attribute_disclosure_risk(
        data, quasi_identifiers, sensitive_attribute,
        detailed=detailed, **kwargs
    )


def identify_vulnerable_attributes(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        diversity_type: str = "distinct",
        l_threshold: int = 3,
        c_value: float = 1.0,
        risk_threshold: float = 50.0
) -> Dict[str, Dict[str, Any]]:
    """
    Identify sensitive attributes with high vulnerability

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attributes : List[str]
        Sensitive attribute columns to analyze
    diversity_type : str, optional
        Type of l-diversity to assess (default: "distinct")
    l_threshold : int, optional
        Minimum acceptable diversity level (default: 3)
    c_value : float, optional
        Parameter for recursive (c,l)-diversity (default: 1.0)
    risk_threshold : float, optional
        Threshold for high risk (default: 50.0)

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Risk assessment for each attribute, sorted by risk level
    """
    # Initialize analyzer
    analyzer = AttributeDisclosureRiskAnalyzer(
        l_threshold=l_threshold,
        diversity_type=diversity_type,
        c_value=c_value
    )

    # Analyze each attribute
    attribute_risks = {}

    for attribute in sensitive_attributes:
        # Skip if attribute not in dataset
        if attribute not in data.columns:
            logger.warning(f"Sensitive attribute '{attribute}' not found in dataset")
            continue

        # Calculate risk
        risk_metrics = analyzer.calculate_attribute_disclosure_risk(
            data, quasi_identifiers, attribute, detailed=False
        )

        # Get overall risk score
        risk_score = risk_metrics.get('overall_risk', {}).get('risk_score', 0)

        # Store risk assessment
        attribute_risks[attribute] = {
            'risk_score': risk_score,
            'risk_level': risk_metrics.get('overall_risk', {}).get('risk_level', 'Unknown'),
            'compliant': risk_metrics.get('overall_risk', {}).get('compliant', False),
            'high_risk': risk_score >= risk_threshold
        }

    # Sort by risk score (highest first)
    return dict(sorted(
        attribute_risks.items(),
        key=lambda x: x[1]['risk_score'],
        reverse=True
    ))


def compare_attribute_risks(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        diversity_configs: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare attribute risks under different diversity configurations

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attributes : List[str]
        Sensitive attribute columns to analyze
    diversity_configs : List[Dict[str, Any]]
        List of diversity configurations to compare
        Each config should have 'diversity_type', 'l_threshold', 'c_value'

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Comparison of risks under different configurations
    """
    comparison = {}

    # Process each attribute
    for attribute in sensitive_attributes:
        # Skip if attribute not in dataset
        if attribute not in data.columns:
            logger.warning(f"Sensitive attribute '{attribute}' not found in dataset")
            continue

        # Initialize attribute comparison
        attribute_comparison = {
            'configs': {}
        }

        # Analyze with each configuration
        for config in diversity_configs:
            # Extract configuration parameters
            diversity_type = config.get('diversity_type', 'distinct')
            l_threshold = config.get('l_threshold', 3)
            c_value = config.get('c_value', 1.0)

            # Create config name
            config_name = f"{diversity_type}-{l_threshold}"
            if diversity_type == 'recursive':
                config_name += f"-{c_value}"

            # Create analyzer
            analyzer = AttributeDisclosureRiskAnalyzer(
                l_threshold=l_threshold,
                diversity_type=diversity_type,
                c_value=c_value
            )

            # Calculate risk
            risk_metrics = analyzer.calculate_attribute_disclosure_risk(
                data, quasi_identifiers, attribute, detailed=False
            )

            # Store config results
            attribute_comparison['configs'][config_name] = {
                'risk_score': risk_metrics.get('overall_risk', {}).get('risk_score', 0),
                'compliant': risk_metrics.get('overall_risk', {}).get('compliant', False),
                'config': config
            }

        # Find best configuration (lowest risk while compliant)
        compliant_configs = {
            name: config for name, config in attribute_comparison['configs'].items()
            if config.get('compliant', False)
        }

        if compliant_configs:
            # Get config with lowest risk among compliant ones
            best_config = min(
                compliant_configs.items(),
                key=lambda x: x[1]['risk_score']
            )[0]
        else:
            # If none are compliant, get the one with lowest risk
            best_config = min(
                attribute_comparison['configs'].items(),
                key=lambda x: x[1]['risk_score']
            )[0]

        # Store best configuration
        attribute_comparison['best_config'] = best_config

        # Store attribute comparison
        comparison[attribute] = attribute_comparison

    return comparison