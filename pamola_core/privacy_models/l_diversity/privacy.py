"""
PAMOLA.CORE - L-Diversity Privacy Risk Assessment
-------------------------------------------------
This module provides comprehensive privacy risk assessment capabilities
for l-diversity anonymization techniques.

Key Features:
- Integrated risk assessment with multiple attack models
- Cache-aware computation for performance optimization
- Support for distinct, entropy, and recursive (c,l)-diversity
- Attribute disclosure risk evaluation
- Human-readable risk interpretations

The module is designed to work seamlessly with the L-Diversity processor's
centralized cache mechanism while providing detailed insight into potential
privacy vulnerabilities.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class LDiversityPrivacyRiskAssessor:
    """
    Evaluates privacy risks for l-diversity datasets

    Provides comprehensive risk assessment methods for identifying potential
    privacy vulnerabilities using various attack models and risk metrics.
    """

    def __init__(self, processor=None, risk_threshold: float = 0.5):
        """
        Initialize Risk Assessor

        Parameters:
        -----------
        processor : object, optional
            L-Diversity processor instance for accessing cached calculations
        risk_threshold : float, optional
            Threshold for determining high-risk groups (default: 0.5)
        """
        self.processor = processor
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger(__name__)

    def assess_privacy_risks(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            diversity_type: str = "distinct",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive privacy risk assessment for l-diversity dataset

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        diversity_type : str, optional
            Type of l-diversity to assess ("distinct", "entropy", "recursive")
        **kwargs : dict
            Additional risk assessment parameters including:
            - l_threshold: int - Minimum acceptable diversity level
            - c_value: float - Parameter for recursive (c,l)-diversity
            - force_recalculate: bool - Whether to force recalculation
            - detailed: bool - Whether to include detailed metrics

        Returns:
        --------
        Dict[str, Any]
            Detailed privacy risk metrics
        """
        try:
            self.logger.info(f"Performing privacy risk assessment for {len(sensitive_attributes)} sensitive attributes")

            # Extract parameters
            l_threshold = kwargs.get('l_threshold', 3)
            c_value = kwargs.get('c_value', 1.0)
            force_recalculate = kwargs.get('force_recalculate', False)
            detailed = kwargs.get('detailed', False)

            # Get group diversity data, using processor's cache if available
            if self.processor:
                group_diversity = self.processor.calculate_group_diversity(
                    data, quasi_identifiers, sensitive_attributes,
                    force_recalculate=force_recalculate
                )
            else:
                # If no processor is available, calculate group diversity directly
                self.logger.warning("No processor provided, calculating group diversity without caching")
                group_diversity = self._calculate_group_diversity_directly(
                    data, quasi_identifiers, sensitive_attributes, diversity_type
                )

            # Calculate risk metrics using the group diversity data
            risk_metrics = self._calculate_risk_metrics(
                group_diversity,
                sensitive_attributes,
                diversity_type=diversity_type,
                l_threshold=l_threshold,
                c_value=c_value,
                detailed=detailed,
                **kwargs
            )

            # Add interpretations for risk values
            risk_metrics['interpretations'] = {
                'prosecutor': self._interpret_risk(risk_metrics['attack_models']['prosecutor_risk']),
                'journalist': self._interpret_risk(risk_metrics['attack_models']['journalist_risk']),
                'marketer': self._interpret_risk(risk_metrics['attack_models']['marketer_risk'])
            }

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Privacy risk assessment error: {e}")
            raise

    def _calculate_group_diversity_directly(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            diversity_type: str
    ) -> pd.DataFrame:
        """
        Calculate group diversity metrics directly when no processor cache is available

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        diversity_type : str
            Type of l-diversity to calculate

        Returns:
        --------
        pd.DataFrame
            Group diversity metrics
        """
        # Group data by quasi-identifiers
        grouped = data.groupby(quasi_identifiers)
        diversity_metrics = []

        # Process each group
        for group_name, group_data in grouped:
            # Initialize group metrics with quasi-identifier values
            group_metrics = {}

            # Add quasi-identifier values to metrics
            if isinstance(group_name, tuple):
                for i, qi in enumerate(quasi_identifiers):
                    group_metrics[qi] = group_name[i]
            else:
                group_metrics[quasi_identifiers[0]] = group_name

            # Calculate diversity metrics for each sensitive attribute
            for sa in sensitive_attributes:
                # Skip if attribute not in dataset
                if sa not in group_data.columns:
                    continue

                # Get unique values
                sa_values = group_data[sa].values

                # Calculate distinct count
                distinct_values = len(np.unique(sa_values))
                group_metrics[f"{sa}_distinct"] = distinct_values

                # Entropy calculation if needed
                if diversity_type == "entropy":
                    # Get value counts and probabilities
                    unique_values, counts = np.unique(sa_values, return_counts=True)
                    probabilities = counts / len(sa_values)
                    # Calculate entropy (add small epsilon to avoid log(0))
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                    group_metrics[f"{sa}_entropy"] = entropy

                # Recursive diversity calculation if needed
                if diversity_type == "recursive":
                    unique_values, counts = np.unique(sa_values, return_counts=True)
                    # Sort in descending order
                    sorted_indices = np.argsort(counts)[::-1]
                    sorted_counts = counts[sorted_indices]

                    # Store for potential recursive check
                    group_metrics[f"{sa}_value_counts"] = sorted_counts

            # Add group size
            group_metrics["group_size"] = len(group_data)
            diversity_metrics.append(group_metrics)

        # Convert to DataFrame
        return pd.DataFrame(diversity_metrics)

    def _calculate_risk_metrics(
            self,
            group_diversity: pd.DataFrame,
            sensitive_attributes: List[str],
            diversity_type: str,
            l_threshold: int = 3,
            c_value: float = 1.0,
            detailed: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate detailed risk metrics from group diversity data

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Grouped diversity information
        sensitive_attributes : List[str]
            Sensitive attribute columns
        diversity_type : str
            Type of l-diversity to assess
        l_threshold : int, optional
            Minimum acceptable diversity level
        c_value : float, optional
            Parameter for recursive (c,l)-diversity
        detailed : bool, optional
            Whether to include detailed metrics

        Returns:
        --------
        Dict[str, Any]
            Comprehensive risk assessment
        """
        # Determine actual metric column based on diversity_type
        if diversity_type == "entropy":
            metric_suffix = "_entropy"
        else:
            metric_suffix = "_distinct"  # Use distinct values for both distinct and recursive

        # Attribute-specific risk calculation
        attribute_risks = {}
        for sa in sensitive_attributes:
            sa_column = f"{sa}{metric_suffix}"

            # Skip if column not found
            if sa_column not in group_diversity.columns:
                self.logger.warning(f"Column {sa_column} not found in group diversity. Skipping.")
                continue

            # Extract diversity metrics for this attribute
            diversity_values = group_diversity[sa_column]

            # Calculate attribute-specific metrics
            attr_min = diversity_values.min()
            attr_max = diversity_values.max()
            attr_mean = diversity_values.mean()
            attr_median = diversity_values.median()

            # Determine risk threshold based on diversity type
            if diversity_type == "entropy":
                # For entropy l-diversity, the threshold is log(l)
                threshold = np.log(l_threshold)
                high_risk_groups = (diversity_values < threshold).sum()
            elif diversity_type == "recursive":
                # For recursive, we need special handling
                high_risk_groups = self._count_recursive_non_compliant(
                    group_diversity, sa, l_threshold, c_value
                )
            else:
                # For distinct l-diversity
                threshold = l_threshold
                high_risk_groups = (diversity_values < threshold).sum()

            # Calculate additional metrics
            groups_count = len(diversity_values)
            risk_percent = (high_risk_groups / groups_count * 100) if groups_count > 0 else 0

            # Store metrics for this attribute
            attribute_risks[sa] = {
                'min_diversity': float(attr_min),
                'max_diversity': float(attr_max),
                'mean_diversity': float(attr_mean),
                'median_diversity': float(attr_median),
                'high_risk_groups': int(high_risk_groups),
                'risk_percent': float(risk_percent),
                'compliant': bool(attr_min >= l_threshold) if diversity_type == "distinct" else
                bool(attr_min >= np.log(l_threshold)) if diversity_type == "entropy" else
                self._check_recursive_compliance(group_diversity, sa, l_threshold, c_value)
            }

            # Add detailed group metrics if requested
            if detailed:
                group_metrics = {}
                for idx, row in group_diversity.iterrows():
                    # Create a key for the group
                    group_key = tuple(row[qi] for qi in group_diversity.columns
                                      if qi in kwargs.get('quasi_identifiers', []))

                    # Add metrics for this group
                    group_metrics[group_key] = {
                        'diversity_value': float(row[sa_column]),
                        'group_size': int(row.get('group_size', 0)),
                        'high_risk': bool(row[sa_column] < threshold) if diversity_type != "recursive" else
                        not self._check_recursive_group(row, sa, l_threshold, c_value)
                    }

                attribute_risks[sa]['group_metrics'] = group_metrics

        # Calculate overall risk based on attribute risks
        overall_min_diversity = min(ar['min_diversity'] for ar in attribute_risks.values()) if attribute_risks else 0
        overall_compliant = all(ar['compliant'] for ar in attribute_risks.values()) if attribute_risks else False
        total_high_risk_groups = max(
            ar['high_risk_groups'] for ar in attribute_risks.values()) if attribute_risks else 0

        # Calculate risk models based on different attack scenarios
        attack_models = self._calculate_attack_models(
            attribute_risks, group_diversity, sensitive_attributes, diversity_type
        )

        # Prepare final risk assessment
        return {
            'overall_risk': {
                'min_diversity': overall_min_diversity,
                'overall_compliant': overall_compliant,
                'high_risk_groups': total_high_risk_groups,
                'diversity_type': diversity_type,
                'l_threshold': l_threshold,
                'c_value': c_value if diversity_type == "recursive" else None
            },
            'attribute_risks': attribute_risks,
            'attack_models': attack_models
        }

    def _calculate_attack_models(
            self,
            attribute_risks: Dict[str, Dict[str, Any]],
            group_diversity: pd.DataFrame,
            sensitive_attributes: List[str],
            diversity_type: str
    ) -> Dict[str, float]:
        """
        Calculate risk under different attack models

        Parameters:
        -----------
        attribute_risks : Dict[str, Dict]
            Pre-calculated risks by attribute
        group_diversity : pd.DataFrame
            Group diversity information
        sensitive_attributes : List[str]
            List of sensitive attributes
        diversity_type : str
            Type of l-diversity being assessed

        Returns:
        --------
        Dict[str, float]
            Risk percentage under different attack models
        """
        # Determine which diversity metric to use
        metric_suffix = "_entropy" if diversity_type == "entropy" else "_distinct"

        # Calculate prosecutor risk (worst-case scenario)
        # This is the risk based on the minimum diversity across all groups
        try:
            min_diversities = [attribute_risks[sa]['min_diversity'] for sa in sensitive_attributes
                               if sa in attribute_risks]

            prosecutor_risk = self._diversity_to_risk(
                min(min_diversities) if min_diversities else 0,
                diversity_type
            )
        except Exception as e:
            self.logger.warning(f"Error calculating prosecutor risk: {e}")
            prosecutor_risk = 100.0  # Default to maximum risk on error

        # Calculate journalist risk (average-case scenario)
        # This is based on the average diversity across all groups
        try:
            # Combine all diversity values from all attributes
            all_diversities = []
            for sa in sensitive_attributes:
                if sa in attribute_risks:
                    sa_column = f"{sa}{metric_suffix}"
                    if sa_column in group_diversity.columns:
                        all_diversities.extend(group_diversity[sa_column].tolist())

            # Calculate mean diversity and convert to risk
            avg_diversity = np.mean(all_diversities) if all_diversities else 0
            journalist_risk = self._diversity_to_risk(avg_diversity, diversity_type)
        except Exception as e:
            self.logger.warning(f"Error calculating journalist risk: {e}")
            journalist_risk = 75.0  # Default to high risk on error

        # Calculate marketer risk (median-case scenario)
        # This is based on the median diversity across all groups
        try:
            # Get median diversity values for each attribute
            median_diversities = [attribute_risks[sa]['median_diversity'] for sa in sensitive_attributes
                                  if sa in attribute_risks]

            # Use median of medians as representative value
            median_diversity = np.median(median_diversities) if median_diversities else 0
            marketer_risk = self._diversity_to_risk(median_diversity, diversity_type)
        except Exception as e:
            self.logger.warning(f"Error calculating marketer risk: {e}")
            marketer_risk = 50.0  # Default to medium risk on error

        return {
            'prosecutor_risk': float(prosecutor_risk),
            'journalist_risk': float(journalist_risk),
            'marketer_risk': float(marketer_risk)
        }

    def _diversity_to_risk(self, diversity_value: float, diversity_type: str) -> float:
        """
        Convert a diversity value to a risk percentage

        Parameters:
        -----------
        diversity_value : float
            L-diversity value to convert
        diversity_type : str
            Type of diversity metric

        Returns:
        --------
        float
            Risk percentage (0-100)
        """
        if diversity_value <= 0:
            return 100.0  # Maximum risk for zero diversity

        if diversity_type == "entropy":
            # For entropy, risk is inversely proportional to exp(entropy)
            effective_diversity = np.exp(diversity_value)
            return min(100.0, 100.0 / effective_diversity)
        else:
            # For distinct and recursive, risk is inversely proportional to diversity
            return min(100.0, 100.0 / diversity_value)

    def _count_recursive_non_compliant(
            self,
            group_diversity: pd.DataFrame,
            sensitive_attribute: str,
            l_threshold: int,
            c_value: float
    ) -> int:
        """
        Count groups that don't satisfy recursive (c,l)-diversity

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Group diversity information
        sensitive_attribute : str
            Sensitive attribute to analyze
        l_threshold : int
            Minimum l value required
        c_value : float
            c parameter for recursive diversity

        Returns:
        --------
        int
            Count of non-compliant groups
        """
        # For recursive diversity, we need value distributions
        value_counts_column = f"{sensitive_attribute}_value_counts"

        if value_counts_column not in group_diversity.columns:
            self.logger.warning(f"Value counts for recursive check not available: {value_counts_column}")
            return 0

        # Count non-compliant groups
        non_compliant = 0

        # Check each group
        for _, row in group_diversity.iterrows():
            if not self._check_recursive_group(row, sensitive_attribute, l_threshold, c_value):
                non_compliant += 1

        return non_compliant

    def _check_recursive_group(
            self,
            group_row: pd.Series,
            sensitive_attribute: str,
            l_threshold: int,
            c_value: float
    ) -> bool:
        """
        Check if a group satisfies recursive (c,l)-diversity

        Parameters:
        -----------
        group_row : pd.Series
            Row from group diversity DataFrame
        sensitive_attribute : str
            Sensitive attribute to analyze
        l_threshold : int
            Minimum l value required
        c_value : float
            c parameter for recursive diversity

        Returns:
        --------
        bool
            Whether the group is compliant
        """
        # Get value counts
        value_counts_column = f"{sensitive_attribute}_value_counts"

        if value_counts_column not in group_row:
            return False

        value_counts = group_row[value_counts_column]

        # Check if we have enough values
        if len(value_counts) < l_threshold:
            return False

        # Get top value and sum of l-1 least frequent values
        most_frequent = value_counts[0]  # Already sorted in descending order
        least_frequent_sum = sum(value_counts[-(l_threshold - 1):])

        # Check recursive criterion
        if least_frequent_sum == 0:
            return False

        return most_frequent <= c_value * least_frequent_sum

    def _check_recursive_compliance(
            self,
            group_diversity: pd.DataFrame,
            sensitive_attribute: str,
            l_threshold: int,
            c_value: float
    ) -> bool:
        """
        Check overall recursive (c,l)-diversity compliance

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Group diversity information
        sensitive_attribute : str
            Sensitive attribute to analyze
        l_threshold : int
            Minimum l value required
        c_value : float
            c parameter for recursive diversity

        Returns:
        --------
        bool
            Whether all groups are compliant
        """
        # Count non-compliant groups
        non_compliant = self._count_recursive_non_compliant(
            group_diversity, sensitive_attribute, l_threshold, c_value
        )

        # If no non-compliant groups, the dataset is compliant
        return non_compliant == 0

    def _interpret_risk(self, risk_value: float) -> str:
        """
        Provide human-readable interpretation of risk values

        Parameters:
        -----------
        risk_value : float
            Risk percentage (0-100)

        Returns:
        --------
        str
            Human-readable risk interpretation
        """
        if risk_value < 5:
            return "Very Low Risk - Excellent privacy protection"
        elif risk_value < 15:
            return "Low Risk - Good privacy protection"
        elif risk_value < 30:
            return "Moderate Risk - Acceptable for many scenarios"
        elif risk_value < 50:
            return "High Risk - Caution recommended when sharing data"
        else:
            return "Very High Risk - Significant privacy concerns"

    def identify_high_risk_groups(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            risk_threshold: float = 0.5,
            diversity_type: str = "distinct",
            l_threshold: int = 3,
            c_value: float = 1.0
    ) -> pd.DataFrame:
        """
        Identify groups with high privacy risks

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        risk_threshold : float, optional
            Threshold for considering a group high-risk
        diversity_type : str, optional
            Type of l-diversity to assess
        l_threshold : int, optional
            Minimum acceptable diversity level
        c_value : float, optional
            Parameter for recursive (c,l)-diversity

        Returns:
        --------
        pd.DataFrame
            High-risk groups with detailed risk information
        """
        try:
            # Use processor's cached diversity calculation if available
            if self.processor:
                group_diversity = self.processor.calculate_group_diversity(
                    data, quasi_identifiers, sensitive_attributes
                )
            else:
                # If no processor is available, calculate group diversity directly
                group_diversity = self._calculate_group_diversity_directly(
                    data, quasi_identifiers, sensitive_attributes, diversity_type
                )

            # Determine how to identify high-risk groups based on diversity type
            high_risk_groups = pd.DataFrame()
            metric_suffix = "_entropy" if diversity_type == "entropy" else "_distinct"

            if diversity_type == "recursive":
                # For recursive, we need to check each group
                risky_indices = []
                for idx, row in group_diversity.iterrows():
                    is_risky = False
                    for sa in sensitive_attributes:
                        if not self._check_recursive_group(row, sa, l_threshold, c_value):
                            is_risky = True
                            break
                    if is_risky:
                        risky_indices.append(idx)

                if risky_indices:
                    high_risk_groups = group_diversity.loc[risky_indices]
            else:
                # For distinct and entropy, use the metric directly
                risk_groups_mask = pd.Series(False, index=group_diversity.index)

                for sa in sensitive_attributes:
                    sa_column = f"{sa}{metric_suffix}"
                    if sa_column in group_diversity.columns:
                        threshold = np.log(l_threshold) if diversity_type == "entropy" else l_threshold
                        risk_groups_mask |= (group_diversity[sa_column] < threshold)

                high_risk_groups = group_diversity[risk_groups_mask]

            # Add risk scores to high-risk groups
            if not high_risk_groups.empty:
                high_risk_groups['risk_score'] = high_risk_groups.apply(
                    lambda row: self._calculate_group_risk_score(
                        row, sensitive_attributes, diversity_type, l_threshold, c_value
                    ),
                    axis=1
                )

                # Sort by risk score (highest risk first)
                high_risk_groups = high_risk_groups.sort_values('risk_score', ascending=False)

            return high_risk_groups

        except Exception as e:
            self.logger.error(f"High-risk group identification error: {e}")
            raise

    def _calculate_group_risk_score(
            self,
            group_row: pd.Series,
            sensitive_attributes: List[str],
            diversity_type: str,
            l_threshold: int,
            c_value: float
    ) -> float:
        """
        Calculate a risk score for a specific group

        Parameters:
        -----------
        group_row : pd.Series
            Row from group diversity DataFrame
        sensitive_attributes : List[str]
            Sensitive attributes to analyze
        diversity_type : str
            Type of l-diversity to assess
        l_threshold : int
            Minimum acceptable diversity level
        c_value : float
            Parameter for recursive (c,l)-diversity

        Returns:
        --------
        float
            Risk score (higher means higher risk)
        """
        # Determine which metrics to use
        metric_suffix = "_entropy" if diversity_type == "entropy" else "_distinct"

        # Calculate risk for each sensitive attribute
        attribute_risks = []

        for sa in sensitive_attributes:
            sa_column = f"{sa}{metric_suffix}"

            if sa_column in group_row:
                # Get diversity value
                diversity_value = group_row[sa_column]

                # Convert to risk
                if diversity_type == "recursive":
                    # For recursive, check compliance
                    if not self._check_recursive_group(group_row, sa, l_threshold, c_value):
                        # Non-compliant groups get high risk scores
                        attribute_risks.append(90.0)
                    else:
                        # Compliant groups get low risk scores
                        attribute_risks.append(10.0)
                else:
                    # For distinct and entropy, convert diversity to risk
                    attribute_risks.append(self._diversity_to_risk(diversity_value, diversity_type))

        # Return maximum risk across attributes (conservative approach)
        return max(attribute_risks) if attribute_risks else 0.0


def evaluate_privacy(data: pd.DataFrame, quasi_identifiers: List[str],
                     sensitive_attributes: List[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Evaluate privacy risks for a dataset using l-diversity principles.

    This function is called by the LDiversityCalculator.evaluate_privacy method
    and serves as a bridge between the calculation module and the privacy module.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attributes : List[str], optional
        Sensitive attribute columns
    **kwargs : dict
        Additional risk assessment parameters

    Returns:
    --------
    Dict[str, Any]
        Comprehensive privacy risk metrics
    """
    # Create a risk assessor (without processor to avoid circular reference)
    risk_assessor = LDiversityPrivacyRiskAssessor()

    # Ensure sensitive_attributes is a list
    if sensitive_attributes is None:
        # Try to find potential sensitive attributes (non-quasi-identifiers)
        sensitive_attributes = [col for col in data.columns if col not in quasi_identifiers]

    # Assess privacy risks
    return risk_assessor.assess_privacy_risks(
        data, quasi_identifiers, sensitive_attributes, **kwargs
    )


def calculate_attribute_disclosure_risk(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        diversity_type: str = "distinct",
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
    **kwargs : dict
        Additional parameters for risk calculation

    Returns:
    --------
    Dict[str, Any]
        Attribute disclosure risk metrics
    """
    # This is a placeholder for the attribute_risk.py implementation
    # The actual implementation will be moved to attribute_risk.py

    # Create a risk assessor
    risk_assessor = LDiversityPrivacyRiskAssessor()

    # Assess privacy risks for a single attribute
    risk_metrics = risk_assessor.assess_privacy_risks(
        data, quasi_identifiers, [sensitive_attribute],
        diversity_type=diversity_type, **kwargs
    )

    # Extract attribute-specific risk
    if sensitive_attribute in risk_metrics.get('attribute_risks', {}):
        attr_risk = risk_metrics['attribute_risks'][sensitive_attribute]

        # Group by quasi-identifiers and analyze sensitive attribute
        grouped = data.groupby(quasi_identifiers)[sensitive_attribute]

        # Calculate predictability metrics
        predictability = {
            'unique_combinations': len(grouped),
            'dominant_value_ratio': grouped.apply(
                lambda x: x.value_counts(normalize=True).max()
            ).mean() if len(grouped) > 0 else 0
        }

        # Add predictability to risk metrics
        attr_risk['predictability'] = predictability

        return attr_risk
    else:
        return {
            'error': f"Could not calculate risk for attribute: {sensitive_attribute}"
        }


