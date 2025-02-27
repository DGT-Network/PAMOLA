"""
PAMOLA.CORE - Disclosure Risk Metrics
-------------------------------------
This module provides metrics for evaluating disclosure risk in anonymized
datasets. It includes various risk models such as prosecutor risk, journalist
risk, and marketer risk, which represent different adversarial models.

Key features:
- Multiple risk models based on different adversarial assumptions
- Support for k-anonymity, l-diversity, and t-closeness based risk assessment
- Record-level and dataset-level risk calculations
- Integration with other privacy metrics

These metrics help data custodians understand the re-identification risks
associated with sharing or publishing anonymized data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any

from core.metrics.base import PrivacyMetric, round_metric_values
from core.utils.group_processing import compute_group_sizes

# Configure logging
logger = logging.getLogger(__name__)


class DisclosureRiskMetric(PrivacyMetric):
    """
    Calculates disclosure risk metrics for a dataset based on quasi-identifiers.

    This class implements various disclosure risk models:
    - Prosecutor model: Assumes the attacker knows the target is in the dataset
    - Journalist model: Assumes the attacker doesn't know if the target is in the dataset
    - Marketer model: Focuses on the proportion of unique records

    The risk values are expressed as percentages, with higher values
    indicating higher risk of re-identification.
    """

    def __init__(self, risk_threshold: float = 5.0):
        """
        Initialize the disclosure risk metric.

        Parameters:
        -----------
        risk_threshold : float, optional
            Threshold percentage above which records are considered at risk (default: 5.0%).
        """
        super().__init__(
            name="Disclosure Risk",
            description="Measures the risk of re-identifying individuals in the dataset"
        )
        self.risk_threshold = risk_threshold

    def calculate(self, data: pd.DataFrame,
                  quasi_identifiers: List[str],
                  k_column: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate disclosure risk metrics for a dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to evaluate.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        k_column : str, optional
            Name of column containing k-values (if already calculated).
        **kwargs : dict
            Additional parameters for risk calculation.

        Returns:
        --------
        dict
            Dictionary with disclosure risk metrics:
            - "prosecutor_risk": Maximum re-identification risk (1/minimum k)
            - "journalist_risk": Average re-identification risk
            - "marketer_risk": Proportion of unique records
            - "records_at_risk": Number of records with risk above threshold
            - "percent_at_risk": Percentage of records at risk
        """
        logger.info(f"Calculating disclosure risk for {len(quasi_identifiers)} quasi-identifiers")

        try:
            # If k-column is provided, use it directly
            if k_column and k_column in data.columns:
                k_values = data[k_column]
            else:
                # Calculate group sizes
                group_sizes = compute_group_sizes(data, quasi_identifiers,
                                                  kwargs.get('use_dask', False))

                # Merge with original data to get k-values for each record
                k_values = data.merge(
                    group_sizes.rename('k_value'),
                    left_on=quasi_identifiers,
                    right_index=True
                )['k_value']

            # Calculate prosecutor risk (maximum re-identification risk)
            # This is 1/k for the smallest group
            min_k = k_values.min()
            prosecutor_risk = 100 / min_k if min_k > 0 else 100

            # Calculate journalist risk (average re-identification risk)
            # This is the average of 1/k across all records
            if (k_values > 0).all():
                journalist_risk = (100 / k_values).mean()
            else:
                journalist_risk = None

            # Calculate marketer risk (proportion of unique records)
            # This is the percentage of records that are unique (k=1)
            marketer_risk = 100 * (k_values == 1).sum() / len(data) if len(data) > 0 else 0

            # Calculate records at risk (using the risk threshold)
            risk_threshold_k = 100 / self.risk_threshold
            records_at_risk = (k_values < risk_threshold_k).sum()
            percent_at_risk = 100 * records_at_risk / len(data) if len(data) > 0 else 0

            # Prepare result
            result = {
                "prosecutor_risk": prosecutor_risk,
                "journalist_risk": journalist_risk,
                "marketer_risk": marketer_risk,
                "records_at_risk": int(records_at_risk),
                "percent_at_risk": percent_at_risk,
                "risk_threshold": self.risk_threshold,
                "min_k": min_k
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Disclosure risk analysis: Prosecutor risk = {prosecutor_risk:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Error during disclosure risk calculation: {e}")
            raise

    def interpret(self, value: float) -> str:
        """
        Interpret a disclosure risk value.

        Parameters:
        -----------
        value : float
            The disclosure risk value (typically prosecutor_risk).

        Returns:
        --------
        str
            Human-readable interpretation of the disclosure risk.
        """
        if value < 1:
            return f"Disclosure Risk: {value:.2f}% - Very low risk, excellent privacy protection"
        elif value < 5:
            return f"Disclosure Risk: {value:.2f}% - Low risk, good privacy protection"
        elif value < 10:
            return f"Disclosure Risk: {value:.2f}% - Moderate risk, acceptable for many scenarios"
        elif value < 20:
            return f"Disclosure Risk: {value:.2f}% - High risk, use caution when sharing data"
        else:
            return f"Disclosure Risk: {value:.2f}% - Very high risk, significant privacy concerns"


class KAnonymityRiskMetric(PrivacyMetric):
    """
    Specialized disclosure risk metric based on k-anonymity properties.

    This metric evaluates privacy protection based on k-anonymity principles,
    focusing on the minimum group size (k value) and the distribution of
    group sizes in the dataset.
    """

    def __init__(self, k_threshold: int = 5):
        """
        Initialize the k-anonymity risk metric.

        Parameters:
        -----------
        k_threshold : int, optional
            The minimum k value considered acceptable (default: 5).
        """
        super().__init__(
            name="k-Anonymity Risk",
            description="Measures re-identification risk based on k-anonymity principles"
        )
        self.k_threshold = k_threshold

    def calculate(self, data: pd.DataFrame,
                  quasi_identifiers: List[str],
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate k-anonymity based risk metrics.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to evaluate.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        **kwargs : dict
            Additional parameters for risk calculation.

        Returns:
        --------
        dict
            Dictionary with k-anonymity risk metrics:
            - "min_k": The minimum k value in the dataset
            - "compliant": Whether the dataset meets the k-threshold
            - "groups_below_threshold": Number of groups with k below threshold
            - "records_in_small_groups": Number of records in groups with k below threshold
            - "k_distribution": Distribution of group sizes (optional)
        """
        logger.info(f"Calculating k-anonymity risk for {len(quasi_identifiers)} quasi-identifiers")

        try:
            # Calculate group sizes
            group_sizes = compute_group_sizes(data, quasi_identifiers,
                                              kwargs.get('use_dask', False))

            # Calculate basic metrics
            min_k = group_sizes.min()
            groups_below_threshold = (group_sizes < self.k_threshold).sum()
            records_in_small_groups = group_sizes[group_sizes < self.k_threshold].sum()
            compliant = min_k >= self.k_threshold

            # Prepare result
            result = {
                "min_k": min_k,
                "compliant": compliant,
                "k_threshold": self.k_threshold,
                "groups_below_threshold": int(groups_below_threshold),
                "records_in_small_groups": int(records_in_small_groups),
                "percent_records_at_risk": round(100 * records_in_small_groups / len(data), 2) if len(data) > 0 else 0
            }

            # Add detailed k-distribution if requested
            if kwargs.get('detailed', False):
                # Get distribution of group sizes
                k_distribution = group_sizes.value_counts().sort_index().to_dict()
                result["k_distribution"] = k_distribution

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"k-Anonymity risk analysis: Minimum k = {min_k}, Compliant = {compliant}")
            return result

        except Exception as e:
            logger.error(f"Error during k-anonymity risk calculation: {e}")
            raise

    def interpret(self, value: int) -> str:
        """
        Interpret a k-anonymity value (min_k).

        Parameters:
        -----------
        value : int
            The minimum k value in the dataset.

        Returns:
        --------
        str
            Human-readable interpretation of the k-anonymity level.
        """
        if value < 2:
            return f"k-Anonymity: k={value} - No anonymity, records can be uniquely identified"
        elif value < self.k_threshold:
            return f"k-Anonymity: k={value} - Insufficient anonymity, below recommended threshold of {self.k_threshold}"
        elif value < 10:
            return f"k-Anonymity: k={value} - Acceptable anonymity for many scenarios"
        elif value < 20:
            return f"k-Anonymity: k={value} - Strong anonymity, suitable for sensitive data"
        else:
            return f"k-Anonymity: k={value} - Very strong anonymity, excellent privacy protection"


class LDiversityRiskMetric(PrivacyMetric):
    """
    Basic l-diversity risk metric implementation.

    This is a simplified version that delegates to the more comprehensive
    implementation in ldiversity_risk.py. It maintains backward compatibility
    while encouraging use of the specialized module for advanced l-diversity
    risk assessment.
    """

    def __init__(self, l_threshold: int = 3, diversity_type: str = "distinct", c_value: float = 1.0):
        """
        Initialize the basic l-diversity risk metric.

        Parameters:
        -----------
        l_threshold : int, optional
            The minimum l value considered acceptable (default: 3).
        diversity_type : str, optional
            Type of l-diversity to use (default: "distinct").
        c_value : float, optional
            Parameter for recursive (c,l)-diversity (default: 1.0).
        """
        super().__init__(
            name="l-Diversity Risk",
            description="Measures attribute disclosure risk based on l-diversity principles"
        )
        self.l_threshold = l_threshold
        self.diversity_type = diversity_type
        self.c_value = c_value

    def calculate(self, data: pd.DataFrame,
                  quasi_identifiers: List[str],
                  sensitive_attributes: List[str],
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate l-diversity based risk metrics.

        This method imports and uses the specialized LDiversityRiskMetric from
        the ldiversity_risk module for complete functionality.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to evaluate.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        sensitive_attributes : list[str]
            List of column names containing sensitive information.
        **kwargs : dict
            Additional parameters for risk calculation.

        Returns:
        --------
        dict
            Dictionary with l-diversity risk metrics.
        """
        try:
            # Import the specialized metric from ldiversity_risk module
            from core.metrics.privacy.ldiversity_risk import LDiversityRiskMetric as SpecializedLDiversityRiskMetric

            # Create an instance of the specialized metric
            specialized_metric = SpecializedLDiversityRiskMetric(
                l_threshold=self.l_threshold,
                diversity_type=self.diversity_type,
                c_value=self.c_value
            )

            # Calculate and return the result
            result = specialized_metric.calculate(
                data=data,
                quasi_identifiers=quasi_identifiers,
                sensitive_attributes=sensitive_attributes,
                **kwargs
            )

            # Store the result
            self.last_result = result

            return result

        except ImportError:
            # Fallback to basic implementation if specialized module is not available
            logger.warning("Specialized ldiversity_risk module not available. Using basic implementation.")
            return self._calculate_basic(data, quasi_identifiers, sensitive_attributes, **kwargs)

    def _calculate_basic(self, data: pd.DataFrame,
                         quasi_identifiers: List[str],
                         sensitive_attributes: List[str],
                         **kwargs) -> Dict[str, Any]:
        """
        Basic implementation of l-diversity risk calculation.

        This is a simplified version that only handles the 'distinct' diversity type.
        For more advanced l-diversity metrics, use the specialized module.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to evaluate.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        sensitive_attributes : list[str]
            List of column names containing sensitive information.
        **kwargs : dict
            Additional parameters for risk calculation.

        Returns:
        --------
        dict
            Dictionary with basic l-diversity risk metrics.
        """
        logger.info(f"Calculating basic l-diversity risk for {len(sensitive_attributes)} sensitive attributes")

        try:
            # Calculate group sizes based on quasi-identifiers
            groups = data.groupby(quasi_identifiers)

            # Initialize result containers
            attribute_diversity = {}
            min_l_values = []
            groups_below_threshold = 0

            # Process each sensitive attribute
            for attribute in sensitive_attributes:
                # Skip if attribute not in data
                if attribute not in data.columns:
                    logger.warning(f"Sensitive attribute '{attribute}' not found in dataset")
                    continue

                # Calculate distinct values per group for this attribute
                l_values = groups[attribute].nunique()
                min_l = l_values.min()
                min_l_values.append(min_l)

                # Count groups below threshold
                below_threshold_count = (l_values < self.l_threshold).sum()
                groups_below_threshold = max(groups_below_threshold, below_threshold_count)

                # Store diversity information for this attribute
                attribute_diversity[attribute] = {
                    "min_l": min_l,
                    "compliant": min_l >= self.l_threshold,
                    "groups_below_threshold": int(below_threshold_count)
                }

            # Overall compliance is the minimum l-value across all attributes
            min_l = min(min_l_values) if min_l_values else 0
            compliant = min_l >= self.l_threshold

            # Prepare result
            result = {
                "min_l": min_l,
                "compliant": compliant,
                "l_threshold": self.l_threshold,
                "groups_below_threshold": int(groups_below_threshold),
                "attribute_diversity": attribute_diversity
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Basic l-Diversity risk analysis: Minimum l = {min_l}, Compliant = {compliant}")
            return result

        except Exception as e:
            logger.error(f"Error during basic l-diversity risk calculation: {e}")
            raise

    def interpret(self, value: int) -> str:
        """
        Interpret an l-diversity value (min_l).

        Parameters:
        -----------
        value : int
            The minimum l value across all sensitive attributes.

        Returns:
        --------
        str
            Human-readable interpretation of the l-diversity level.
        """
        if value < 2:
            return f"l-Diversity: l={value} - No diversity, high attribute disclosure risk"
        elif value < self.l_threshold:
            return f"l-Diversity: l={value} - Insufficient diversity, below recommended threshold of {self.l_threshold}"
        elif value < 5:
            return f"l-Diversity: l={value} - Acceptable diversity for many scenarios"
        elif value < 10:
            return f"l-Diversity: l={value} - Good diversity, low attribute disclosure risk"
        else:
            return f"l-Diversity: l={value} - Excellent diversity, very low attribute disclosure risk"


# Convenience function for calculating all disclosure risk metrics
def calculate_disclosure_risk_metrics(data: pd.DataFrame,
                                      quasi_identifiers: List[str],
                                      sensitive_attributes: Optional[List[str]] = None,
                                      **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Calculate multiple disclosure risk metrics for a dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to evaluate.
    quasi_identifiers : list[str]
        List of column names used as quasi-identifiers.
    sensitive_attributes : list[str], optional
        List of column names containing sensitive information.
        Required for l-diversity metrics.
    **kwargs : dict
        Additional parameters for risk calculation:
        - k_threshold: int - Threshold for k-anonymity (default: 5)
        - l_threshold: int - Threshold for l-diversity (default: 3)
        - risk_threshold: float - Threshold for high risk percentage (default: 5.0)
        - diversity_type: str - Type of l-diversity ('distinct', 'entropy', 'recursive')
        - c_value: float - c parameter for recursive (c,l)-diversity (default: 1.0)
        - detailed: bool - Whether to include detailed metrics (default: False)
        - record_level_risk: bool - Whether to calculate record-level risk (default: False)
        - t_threshold: float - Threshold for t-closeness (if supported)

    Returns:
    --------
    dict
        Dictionary with results from all applicable disclosure risk metrics.
    """
    results = {}

    # Calculate general disclosure risk
    disclosure_risk = DisclosureRiskMetric(
        risk_threshold=kwargs.get('risk_threshold', 5.0)
    )
    results["disclosure_risk"] = disclosure_risk.calculate(
        data, quasi_identifiers, kwargs.get('k_column')
    )

    # Calculate k-anonymity risk
    k_risk = KAnonymityRiskMetric(
        k_threshold=kwargs.get('k_threshold', 5)
    )
    results["k_anonymity_risk"] = k_risk.calculate(
        data, quasi_identifiers, detailed=kwargs.get('detailed', False)
    )

    # Calculate l-diversity risk if sensitive attributes are provided
    if sensitive_attributes:
        # Try to use specialized metrics from ldiversity_risk module
        try:
            from core.metrics.privacy.ldiversity_risk import calculate_ldiversity_risk_metrics

            # Use the specialized function for l-diversity risk metrics
            ldiversity_results = calculate_ldiversity_risk_metrics(
                data=data,
                quasi_identifiers=quasi_identifiers,
                sensitive_attributes=sensitive_attributes,
                **kwargs
            )

            # Merge results
            results.update(ldiversity_results)

        except ImportError:
            # Fallback to basic implementation
            logger.warning("Specialized ldiversity_risk module not available. Using basic implementation.")

            # Extract l-diversity specific parameters
            diversity_type = kwargs.get('diversity_type', 'distinct')
            l_threshold = kwargs.get('l_threshold', 3)
            c_value = kwargs.get('c_value', 1.0)

            l_risk = LDiversityRiskMetric(
                l_threshold=l_threshold,
                diversity_type=diversity_type,
                c_value=c_value
            )

            results["l_diversity_risk"] = l_risk.calculate(
                data,
                quasi_identifiers,
                sensitive_attributes,
                **kwargs
            )

    return results