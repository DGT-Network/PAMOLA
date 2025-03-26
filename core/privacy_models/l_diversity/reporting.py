"""
PAMOLA.CORE - L-Diversity Reporting Module

Provides comprehensive reporting capabilities for l-diversity
anonymization techniques with cache awareness and integration
with base reporting infrastructure.

Key Features:
- Cache-aware report generation
- Multiple report types (general, compliance, technical)
- Support for different l-diversity types (distinct, entropy, recursive)
- Integration with visualization module
- Standardized file I/O through central utilities

This module extends the base reporting functionality in core.utils.base_reporting
to provide specialized reporting for l-diversity anonymization.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Import from core utilities
from core import config
from core.utils.base_reporting import PrivacyReport
from core.utils.file_io import write_json, write_csv

# Configure logging
logger = logging.getLogger(__name__)


class LDiversityReport(PrivacyReport):
    """
    Comprehensive report generation for l-diversity anonymization

    This class extends the base PrivacyReport class to provide specialized
    reporting capabilities for l-diversity anonymization with cache awareness
    and support for different diversity types.
    """

    def __init__(self,
                 processor=None,
                 report_data: Optional[Dict[str, Any]] = None,
                 diversity_type: str = None):
        """
        Initialize reporter with processor for cache access

        Parameters:
        -----------
        processor : object, optional
            L-Diversity processor instance for cached calculations
        report_data : Dict[str, Any], optional
            Explicit report data (if not using processor)
        diversity_type : str, optional
            Type of l-diversity (uses processor's type if None)
        """
        # Initialize report data from processor or provided data
        self.processor = processor
        self.data = None  # Will store dataset if passed in later methods

        # Collect initial report data
        collected_data = self._collect_report_data(processor, report_data)

        # Determine diversity type
        if diversity_type is None and processor:
            self.diversity_type = getattr(processor, 'diversity_type', 'distinct')
        elif diversity_type is None and report_data:
            self.diversity_type = report_data.get(
                'l_diversity_configuration', {}).get('diversity_type', 'distinct')
        else:
            self.diversity_type = diversity_type or 'distinct'

        # Store diversity type in report data
        if 'l_diversity_configuration' in collected_data:
            collected_data['l_diversity_configuration']['diversity_type'] = self.diversity_type

        # Initialize base class
        super().__init__(collected_data, "l-diversity")

        # Store execution times if processor has them
        self.execution_times = {}
        if processor:
            self.execution_times = getattr(processor, 'execution_times', {})

        self.logger = logging.getLogger(__name__)

    def _collect_report_data(self,
                             processor,
                             report_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Collect report data from processor with cache awareness

        Parameters:
        -----------
        processor : object
            L-Diversity processor instance
        report_data : Dict[str, Any], optional
            Explicit report data (if not using processor)

        Returns:
        --------
        Dict[str, Any]
            Collected report data
        """
        # If report_data is provided, use it directly
        if report_data:
            return report_data

        # Initialize empty report data
        collected_data = {
            "report_metadata": {
                "creation_time": datetime.now().isoformat(),
                "pamola_version": getattr(config, "PAMOLA_VERSION", "unknown"),
                "report_type": "l-diversity"
            }
        }

        # If no processor, return minimal data
        if not processor:
            return collected_data

        # Extract configuration from processor
        collected_data["l_diversity_configuration"] = {
            "l_value": getattr(processor, 'l', 3),
            "diversity_type": getattr(processor, 'diversity_type', 'distinct'),
            "k_value": getattr(processor, 'k', 2),
            "suppression": getattr(processor, 'suppression', True)
        }

        # Add c_value if diversity type is recursive
        if collected_data["l_diversity_configuration"]["diversity_type"] == "recursive":
            collected_data["l_diversity_configuration"]["c_value"] = getattr(processor, 'c_value', 1.0)

        # Check for cached risk assessment
        if hasattr(processor, 'risk_assessor'):
            # Store reference to risk assessor for later use
            self.risk_assessor = processor.risk_assessor
        else:
            self.risk_assessor = None

        # Store execution times if available
        if hasattr(processor, 'execution_times') and processor.execution_times:
            collected_data["execution_times"] = processor.execution_times

        return collected_data

    def generate(self,
                 data: Optional[pd.DataFrame] = None,
                 quasi_identifiers: Optional[List[str]] = None,
                 sensitive_attributes: Optional[List[str]] = None,
                 include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive l-diversity report

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset (if not already processed by processor)
        quasi_identifiers : List[str], optional
            Quasi-identifier columns (required if data provided)
        sensitive_attributes : List[str], optional
            Sensitive attribute columns (required if data provided)
        include_visualizations : bool, optional
            Whether to include visualization paths (default: True)

        Returns:
        --------
        Dict[str, Any]
            Comprehensive anonymization report
        """
        # Store data reference if provided
        if data is not None:
            self.data = data

        # Initialize report with metadata
        report = {
            "report_metadata": self.metadata
        }

        # Add l-diversity configuration
        if "l_diversity_configuration" in self.report_data:
            report["l_diversity_configuration"] = self.report_data["l_diversity_configuration"]

        # Add dataset information if data provided
        if data is not None and quasi_identifiers and sensitive_attributes:
            report["dataset_information"] = self._generate_dataset_info(
                data, quasi_identifiers, sensitive_attributes
            )

        # Calculate privacy risks from processor cache or directly
        privacy_risks = self._calculate_privacy_risks(
            data, quasi_identifiers, sensitive_attributes
        )

        if privacy_risks:
            report["privacy_evaluation"] = privacy_risks

        # Add metrics depending on diversity type
        diversity_metrics = self._calculate_diversity_metrics(
            data, quasi_identifiers, sensitive_attributes
        )

        if diversity_metrics:
            diversity_type = self.diversity_type
            report[f"{diversity_type}_diversity_metrics"] = diversity_metrics

        # Add information loss metrics if available
        information_loss = self._calculate_information_loss(
            data, quasi_identifiers, sensitive_attributes
        )

        if information_loss:
            report["information_loss"] = information_loss

        # Add execution times if available
        if self.execution_times:
            report["execution_times"] = self.execution_times

        # Add visualizations if requested and available
        if include_visualizations:
            visualization_paths = self._collect_visualizations()
            if visualization_paths:
                report["visualization_paths"] = visualization_paths

        return report

    def _generate_dataset_info(self,
                               data: pd.DataFrame,
                               quasi_identifiers: List[str],
                               sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Generate dataset information

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
            Dataset information
        """
        dataset_info = {
            "record_count": len(data),
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": sensitive_attributes,
            "column_count": len(data.columns)
        }

        # Add column types
        column_types = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                col_type = "numeric"
            elif pd.api.types.is_categorical_dtype(data[col]):
                col_type = "categorical"
            else:
                col_type = "other"

            column_types[col] = col_type

        dataset_info["column_types"] = column_types

        # Add attribute cardinality (number of unique values)
        cardinality = {}
        for col in quasi_identifiers + sensitive_attributes:
            if col in data.columns:
                cardinality[col] = data[col].nunique()

        dataset_info["attribute_cardinality"] = cardinality

        return dataset_info

    def _calculate_privacy_risks(self,
                                 data: Optional[pd.DataFrame] = None,
                                 quasi_identifiers: Optional[List[str]] = None,
                                 sensitive_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate privacy risks from processor cache or directly

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset (if not already processed by processor)
        quasi_identifiers : List[str], optional
            Quasi-identifier columns (required if data provided)
        sensitive_attributes : List[str], optional
            Sensitive attribute columns (required if data provided)

        Returns:
        --------
        Dict[str, Any]
            Privacy risk metrics
        """
        # Try to get risk metrics from processor or risk assessor
        if self.processor:
            if hasattr(self.processor, 'evaluate_privacy'):
                try:
                    return self.processor.evaluate_privacy(
                        data or self.data,
                        quasi_identifiers or [],
                        sensitive_attributes or []
                    )
                except Exception as e:
                    self.logger.warning(f"Could not use processor's evaluate_privacy: {e}")

            # Alternative: check for risk_assessor attribute
            if hasattr(self.processor, 'risk_assessor'):
                try:
                    risk_assessor = self.processor.risk_assessor
                    return risk_assessor.assess_privacy_risks(
                        data or self.data,
                        quasi_identifiers or [],
                        sensitive_attributes or []
                    )
                except Exception as e:
                    self.logger.warning(f"Could not use processor's risk_assessor: {e}")

        # If data is provided but no processor, calculate basic metrics
        if data is not None and quasi_identifiers and sensitive_attributes:
            return self._calculate_basic_privacy_metrics(
                data, quasi_identifiers, sensitive_attributes
            )

        # If no data and no processor, return empty dict
        return {}

    def _calculate_basic_privacy_metrics(self,
                                         data: pd.DataFrame,
                                         quasi_identifiers: List[str],
                                         sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate basic privacy metrics when processor not available

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
            Basic privacy metrics
        """
        # Calculate group sizes and l-values
        groups = data.groupby(quasi_identifiers)
        l_values = []

        for _, group in groups:
            for sa in sensitive_attributes:
                if sa in group.columns:
                    if self.diversity_type == "entropy":
                        # Calculate entropy l-diversity
                        value_counts = group[sa].value_counts(normalize=True)
                        entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                        l_values.append(entropy)
                    else:
                        # Calculate distinct l-diversity
                        l_values.append(group[sa].nunique())

        # Calculate overall risk metrics
        min_l = min(l_values) if l_values else 0
        l_threshold = self.report_data.get(
            "l_diversity_configuration", {}).get("l_value", 3)

        # Convert min_l for entropy
        if self.diversity_type == "entropy":
            effective_min_l = np.exp(min_l) if min_l > 0 else 1
            compliant = effective_min_l >= l_threshold
        else:
            compliant = min_l >= l_threshold

        # Count records at risk
        records_at_risk = 0
        for _, group in groups:
            at_risk = False
            for sa in sensitive_attributes:
                if sa in group.columns:
                    if self.diversity_type == "entropy":
                        value_counts = group[sa].value_counts(normalize=True)
                        entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                        if np.exp(entropy) < l_threshold:
                            at_risk = True
                    else:
                        if group[sa].nunique() < l_threshold:
                            at_risk = True

            if at_risk:
                records_at_risk += len(group)

        percentage_at_risk = (records_at_risk / len(data) * 100) if len(data) > 0 else 0

        # Return privacy metrics
        privacy_metrics = {
            "min_l": min_l,
            "compliant": compliant,
            "records_at_risk": records_at_risk,
            "percentage_at_risk": percentage_at_risk,
            "diversity_type": self.diversity_type
        }

        # Add diversity type specific metrics
        if self.diversity_type == "entropy":
            privacy_metrics["effective_min_l"] = effective_min_l
        elif self.diversity_type == "recursive":
            # Add placeholder for recursive metrics
            privacy_metrics["recursive_compliant"] = False

        # Add attack models if possible
        try:
            privacy_metrics["attack_models"] = self._calculate_attack_models(
                data, quasi_identifiers, sensitive_attributes, l_values
            )
        except Exception as e:
            self.logger.warning(f"Could not calculate attack models: {e}")

        return privacy_metrics

    def _calculate_attack_models(self,
                                 data: pd.DataFrame,
                                 quasi_identifiers: List[str],
                                 sensitive_attributes: List[str],
                                 l_values: List[float]) -> Dict[str, float]:
        """
        Calculate risk under different attack models

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Quasi-identifier columns
        sensitive_attributes : List[str]
            Sensitive attribute columns
        l_values : List[float]
            List of l-values for each group and sensitive attribute

        Returns:
        --------
        Dict[str, float]
            Risk values for different attack models
        """
        if not l_values:
            return {}

        # For entropy, convert to effective l-values
        if self.diversity_type == "entropy":
            effective_l_values = [np.exp(l) if l > 0 else 1 for l in l_values]
        else:
            effective_l_values = l_values

        # Calculate risks (risk is inversely proportional to l-value)
        risks = [100 / l if l > 0 else 100 for l in effective_l_values]

        # Prosecutor risk (worst case) - based on minimum l-value
        prosecutor_risk = max(risks) if risks else 100

        # Journalist risk (average case) - based on average risk
        journalist_risk = sum(risks) / len(risks) if risks else 100

        # Marketer risk (median case) - based on median risk
        marketer_risk = sorted(risks)[len(risks) // 2] if risks else 100

        return {
            "prosecutor_risk": prosecutor_risk,
            "journalist_risk": journalist_risk,
            "marketer_risk": marketer_risk
        }

    def _calculate_diversity_metrics(self,
                                     data: Optional[pd.DataFrame] = None,
                                     quasi_identifiers: Optional[List[str]] = None,
                                     sensitive_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate diversity metrics based on diversity type

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset
        quasi_identifiers : List[str], optional
            Quasi-identifier columns
        sensitive_attributes : List[str], optional
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Diversity metrics
        """
        # If no data or identifiers, return empty dict
        if (data is None or not quasi_identifiers or not sensitive_attributes) and not self.processor:
            return {}

        # If processor available, try to use its cache
        diversity_metrics = {}

        if self.processor and hasattr(self.processor, '_results_cache'):
            # Try to extract metrics from cache
            try:
                # Create cache key
                data_to_use = data or self.data
                if data_to_use is not None and quasi_identifiers and sensitive_attributes:
                    cache_key = (
                        tuple(quasi_identifiers),
                        tuple(sensitive_attributes),
                        self.diversity_type
                    )

                    if cache_key in self.processor._results_cache:
                        group_diversity = self.processor._results_cache[cache_key]

                        # Extract metrics based on diversity type
                        if self.diversity_type == "entropy":
                            diversity_metrics = self._extract_entropy_metrics(
                                group_diversity, sensitive_attributes
                            )
                        elif self.diversity_type == "recursive":
                            diversity_metrics = self._extract_recursive_metrics(
                                group_diversity, sensitive_attributes
                            )
                        else:
                            diversity_metrics = self._extract_distinct_metrics(
                                group_diversity, sensitive_attributes
                            )
            except Exception as e:
                self.logger.warning(f"Could not extract metrics from cache: {e}")

        # If metrics not available from cache and data provided, calculate directly
        if not diversity_metrics and data is not None and quasi_identifiers and sensitive_attributes:
            if self.diversity_type == "entropy":
                diversity_metrics = self._calculate_entropy_metrics(
                    data, quasi_identifiers, sensitive_attributes
                )
            elif self.diversity_type == "recursive":
                diversity_metrics = self._calculate_recursive_metrics(
                    data, quasi_identifiers, sensitive_attributes
                )
            else:
                diversity_metrics = self._calculate_distinct_metrics(
                    data, quasi_identifiers, sensitive_attributes
                )

        return diversity_metrics

    def _extract_distinct_metrics(self,
                                  group_diversity: pd.DataFrame,
                                  sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Extract distinct diversity metrics from group diversity data

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Group diversity data from cache
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Distinct diversity metrics
        """
        metrics = {
            "attribute_diversity": {}
        }

        for sa in sensitive_attributes:
            sa_column = f"{sa}_distinct"

            if sa_column in group_diversity.columns:
                # Calculate metrics
                min_distinct = group_diversity[sa_column].min()
                max_distinct = group_diversity[sa_column].max()
                avg_distinct = group_diversity[sa_column].mean()

                # Store attribute metrics
                metrics["attribute_diversity"][sa] = {
                    "min_distinct": min_distinct,
                    "max_distinct": max_distinct,
                    "avg_distinct": avg_distinct,
                    "l_values": group_diversity[sa_column].tolist()
                }

        # Calculate overall metrics
        min_l_values = [metrics["attribute_diversity"][sa]["min_distinct"]
                        for sa in metrics["attribute_diversity"]]

        if min_l_values:
            metrics["min_l"] = min(min_l_values)
            metrics["max_l"] = max([metrics["attribute_diversity"][sa]["max_distinct"]
                                    for sa in metrics["attribute_diversity"]])
            metrics["avg_l"] = sum([metrics["attribute_diversity"][sa]["avg_distinct"]
                                    for sa in metrics["attribute_diversity"]]) / len(min_l_values)

            # Calculate compliance
            l_threshold = self.report_data.get(
                "l_diversity_configuration", {}).get("l_value", 3)
            metrics["compliant"] = metrics["effective_min_l"] >= l_threshold

        return metrics

    def _calculate_recursive_metrics(self,
                                     data: pd.DataFrame,
                                     quasi_identifiers: List[str],
                                     sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate recursive diversity metrics directly from data

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
            Recursive diversity metrics
        """
        # Get l and c values from configuration
        l_threshold = self.report_data.get(
            "l_diversity_configuration", {}).get("l_value", 3)
        c_value = self.report_data.get(
            "l_diversity_configuration", {}).get("c_value", 1.0)

        metrics = {
            "attribute_recursive": {},
            "c_value": c_value,
            "l_value": l_threshold
        }

        # Group data by quasi-identifiers
        groups = data.groupby(quasi_identifiers)

        for sa in sensitive_attributes:
            if sa in data.columns:
                # Analyze each group for recursive diversity
                compliant_groups = 0
                non_compliant_groups = 0
                recursive_values = []

                for _, group in groups:
                    if sa in group.columns:
                        # Get value frequencies
                        value_counts = group[sa].value_counts()

                        # Check if we have enough distinct values
                        if len(value_counts) >= l_threshold:
                            # Sort frequencies in descending order
                            sorted_counts = value_counts.sort_values(ascending=False)

                            # Get most frequent value
                            most_frequent = sorted_counts.iloc[0]

                            # Get sum of l-1 least frequent values
                            least_frequent_sum = sorted_counts.iloc[-l_threshold + 1:].sum()

                            # Check recursive criterion
                            if least_frequent_sum > 0:
                                c_actual = most_frequent / least_frequent_sum
                                compliant = c_actual <= c_value

                                if compliant:
                                    compliant_groups += 1
                                    recursive_values.append(l_threshold)  # Store l-value
                                else:
                                    non_compliant_groups += 1
                                    recursive_values.append(len(value_counts))  # Store actual distinct count
                            else:
                                non_compliant_groups += 1
                                recursive_values.append(len(value_counts))
                        else:
                            non_compliant_groups += 1
                            recursive_values.append(len(value_counts))

                if recursive_values:
                    # Calculate metrics
                    min_recursive = min(recursive_values)
                    max_recursive = max(recursive_values)
                    avg_recursive = sum(recursive_values) / len(recursive_values)

                    # Store attribute metrics
                    metrics["attribute_recursive"][sa] = {
                        "min_recursive": min_recursive,
                        "max_recursive": max_recursive,
                        "avg_recursive": avg_recursive,
                        "compliant_groups": compliant_groups,
                        "non_compliant_groups": non_compliant_groups,
                        "recursive_values": recursive_values
                    }

        # Calculate overall metrics
        min_recursive_values = [metrics["attribute_recursive"][sa]["min_recursive"]
                                for sa in metrics["attribute_recursive"]]

        if min_recursive_values:
            metrics["min_recursive"] = min(min_recursive_values)
            metrics["max_recursive"] = max([metrics["attribute_recursive"][sa]["max_recursive"]
                                            for sa in metrics["attribute_recursive"]])
            metrics["avg_recursive"] = sum([metrics["attribute_recursive"][sa]["avg_recursive"]
                                            for sa in metrics["attribute_recursive"]]) / len(min_recursive_values)

            # Calculate compliance
            total_compliant = sum([metrics["attribute_recursive"][sa]["compliant_groups"]
                                   for sa in metrics["attribute_recursive"]])
            total_non_compliant = sum([metrics["attribute_recursive"][sa]["non_compliant_groups"]
                                       for sa in metrics["attribute_recursive"]])

            metrics["compliant_groups"] = total_compliant
            metrics["non_compliant_groups"] = total_non_compliant
            metrics["compliant"] = total_non_compliant == 0

        return metrics

    def _calculate_information_loss(self,
                                    data: Optional[pd.DataFrame] = None,
                                    quasi_identifiers: Optional[List[str]] = None,
                                    sensitive_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate information loss metrics

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset
        quasi_identifiers : List[str], optional
            Quasi-identifier columns
        sensitive_attributes : List[str], optional
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Information loss metrics
        """
        # This is a simplified implementation
        # In a real implementation, you would need the original and anonymized data
        return {}

    def _collect_visualizations(self) -> Dict[str, Any]:
        """
        Collect visualization paths if available

        Returns:
        --------
        Dict[str, Any]
            Visualization paths
        """
        # This would typically be populated by the visualization module
        return {}

    def generate_compliance_report(self,
                                   data: Optional[pd.DataFrame] = None,
                                   quasi_identifiers: Optional[List[str]] = None,
                                   sensitive_attributes: Optional[List[str]] = None,
                                   regulation: str = "GDPR") -> Dict[str, Any]:
        """
        Generate compliance report for specific regulations

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset
        quasi_identifiers : List[str], optional
            Quasi-identifier columns
        sensitive_attributes : List[str], optional
            Sensitive attribute columns
        regulation : str, optional
            Regulatory framework (default: GDPR)

        Returns:
        --------
        Dict[str, Any]
            Compliance assessment report
        """
        # Get basic report
        base_report = self.generate(data, quasi_identifiers, sensitive_attributes, False)

        # Define compliance criteria for different regulations
        compliance_criteria = {
            'GDPR': {
                'l_threshold': 3,
                'k_threshold': 5,
                'diversity_type': 'distinct',
                'description': 'General Data Protection Regulation (EU)'
            },
            'HIPAA': {
                'l_threshold': 4,
                'k_threshold': 10,
                'diversity_type': 'recursive',
                'c_value': 0.5,
                'description': 'Health Insurance Portability and Accountability Act (US)'
            },
            'CCPA': {
                'l_threshold': 3,
                'k_threshold': 4,
                'diversity_type': 'entropy',
                'description': 'California Consumer Privacy Act (US)'
            }
        }

        # Get criteria for specific regulation
        criteria = compliance_criteria.get(regulation.upper(), compliance_criteria['GDPR'])

        # Create compliance report
        compliance_report = {
            "report_metadata": {
                "creation_time": datetime.now().isoformat(),
                "pamola_version": getattr(config, "PAMOLA_VERSION", "unknown"),
                "report_type": f"l-diversity {regulation} compliance"
            },
            "regulation": regulation,
            "regulation_description": criteria['description'],
            "compliance_criteria": criteria,
            "compliance_status": "Unknown"
        }

        # Get privacy evaluation from base report
        if "privacy_evaluation" in base_report:
            privacy = base_report["privacy_evaluation"]

            # Get configuration
            l_value = base_report.get(
                "l_diversity_configuration", {}).get("l_value", 3)
            diversity_type = base_report.get(
                "l_diversity_configuration", {}).get("diversity_type", "distinct")
            k_value = base_report.get(
                "l_diversity_configuration", {}).get("k_value", 2)
            c_value = base_report.get(
                "l_diversity_configuration", {}).get("c_value", 1.0)

            # Check compliance based on regulation
            l_compliant = False
            k_compliant = k_value >= criteria['k_threshold']
            reasons = []

            if diversity_type == criteria['diversity_type']:
                if diversity_type == "entropy":
                    # For entropy l-diversity, check effective l
                    effective_l = privacy.get('effective_min_l', 0)
                    l_compliant = effective_l >= criteria['l_threshold']
                    if not l_compliant:
                        reasons.append(
                            f"Minimum effective l ({effective_l:.2f}) is below required threshold ({criteria['l_threshold']})"
                        )
                elif diversity_type == "recursive":
                    # For recursive l-diversity, check c value
                    c_compliant = c_value <= criteria.get('c_value', 1.0)
                    l_compliant = privacy.get('compliant', False) and c_compliant
                    if not l_compliant:
                        reasons.append(
                            f"Recursive (c,l)-diversity does not meet requirements"
                        )
                else:
                    # For distinct l-diversity, check min_l
                    min_l = privacy.get('min_l', 0)
                    l_compliant = min_l >= criteria['l_threshold']
                    if not l_compliant:
                        reasons.append(
                            f"Minimum l ({min_l}) is below required threshold ({criteria['l_threshold']})"
                        )
            else:
                reasons.append(
                    f"Diversity type ({diversity_type}) does not match required type ({criteria['diversity_type']})"
                )

            if not k_compliant:
                reasons.append(
                    f"Base k-anonymity ({k_value}) is below required threshold ({criteria['k_threshold']})"
                )

            # Set overall compliance status
            compliance_report['compliance_status'] = 'Compliant' if l_compliant and k_compliant else 'Non-compliant'
            compliance_report['compliance_details'] = {
                'l_compliance': l_compliant,
                'k_compliance': k_compliant,
                'current_configuration': {
                    'l_value': l_value,
                    'k_value': k_value,
                    'diversity_type': diversity_type,
                    'c_value': c_value if diversity_type == "recursive" else None
                }
            }

            # Add reasons for non-compliance
            if not l_compliant or not k_compliant:
                compliance_report['compliance_details']['non_compliance_reasons'] = reasons

            # Add privacy metrics
            compliance_report['privacy_metrics'] = {
                'min_l': privacy.get('min_l', 0),
                'compliant': privacy.get('compliant', False),
                'records_at_risk': privacy.get('records_at_risk', 0),
                'percentage_at_risk': privacy.get('percentage_at_risk', 0)
            }

            # Add recommendations
            compliance_report['recommendations'] = self._generate_compliance_recommendations(
                compliance_report['compliance_status'],
                reasons,
                criteria
            )

        return compliance_report

    def _generate_compliance_recommendations(self,
                                             compliance_status: str,
                                             reasons: List[str],
                                             criteria: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for improving compliance

        Parameters:
        -----------
        compliance_status : str
            Current compliance status
        reasons : List[str]
            Reasons for non-compliance
        criteria : Dict[str, Any]
            Compliance criteria

        Returns:
        --------
        List[str]
            Recommendations for improving compliance
        """
        recommendations = []

        if compliance_status == 'Compliant':
            recommendations.append(
                "Current configuration meets regulatory requirements. Maintain current privacy protections."
            )
        else:
            recommendations.append(
                "Current configuration does not meet regulatory requirements. Consider the following improvements:"
            )

            # Generate specific recommendations based on reasons
            for reason in reasons:
                if "minimum l" in reason.lower():
                    recommendations.append(
                        f"Increase l-diversity to at least {criteria['l_threshold']} by generalizing quasi-identifiers "
                        "or applying suppression to records with low diversity."
                    )
                elif "k-anonymity" in reason.lower():
                    recommendations.append(
                        f"Increase base k-anonymity to at least {criteria['k_threshold']} by generalizing "
                        "quasi-identifiers or applying suppression to small equivalence classes."
                    )
                elif "diversity type" in reason.lower():
                    recommendations.append(
                        f"Change diversity type to {criteria['diversity_type']} to meet regulatory requirements."
                    )
                elif "recursive" in reason.lower():
                    recommendations.append(
                        f"Adjust c-value to {criteria.get('c_value', 0.5)} or lower for recursive l-diversity, "
                        "or consider using a different diversity type."
                    )

        return recommendations

    def generate_technical_report(self,
                                  data: Optional[pd.DataFrame] = None,
                                  quasi_identifiers: Optional[List[str]] = None,
                                  sensitive_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate detailed technical report with comprehensive metrics

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset
        quasi_identifiers : List[str], optional
            Quasi-identifier columns
        sensitive_attributes : List[str], optional
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Detailed technical report
        """
        # Get basic report
        base_report = self.generate(data, quasi_identifiers, sensitive_attributes, True)

        # Create technical report structure
        technical_report = {
            "report_metadata": {
                "creation_time": datetime.now().isoformat(),
                "pamola_version": getattr(config, "PAMOLA_VERSION", "unknown"),
                "report_type": "l-diversity technical analysis",
                "diversity_type": self.diversity_type
            }
        }

        # Add configuration details
        if "l_diversity_configuration" in base_report:
            technical_report["configuration"] = base_report["l_diversity_configuration"]

        # Add dataset information
        if "dataset_information" in base_report:
            technical_report["dataset"] = base_report["dataset_information"]

        # Add privacy evaluation
        if "privacy_evaluation" in base_report:
            technical_report["privacy"] = base_report["privacy_evaluation"]

        # Add detailed diversity metrics
        diversity_key = f"{self.diversity_type}_diversity_metrics"
        if diversity_key in base_report:
            technical_report["diversity_metrics"] = base_report[diversity_key]

        # Add execution times
        if "execution_times" in base_report:
            technical_report["performance"] = base_report["execution_times"]

        # Add visualization paths
        if "visualization_paths" in base_report:
            technical_report["visualizations"] = base_report["visualization_paths"]

        # Add technical analysis section
        technical_report["technical_analysis"] = self._generate_technical_analysis(
            data, quasi_identifiers, sensitive_attributes
        )

        return technical_report

    def _generate_technical_analysis(self,
                                     data: Optional[pd.DataFrame] = None,
                                     quasi_identifiers: Optional[List[str]] = None,
                                     sensitive_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate detailed technical analysis

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Input dataset
        quasi_identifiers : List[str], optional
            Quasi-identifier columns
        sensitive_attributes : List[str], optional
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Technical analysis
        """
        # This is a placeholder for detailed technical analysis
        # In a real implementation, you would add more comprehensive metrics
        technical_analysis = {
            "analysis_type": self.diversity_type,
            "timestamp": datetime.now().isoformat()
        }

        return technical_analysis

    def export_report(self,
                      report: Dict[str, Any],
                      path: Union[str, Path],
                      format: str = 'json') -> str:
        """
        Export report to file

        Parameters:
        -----------
        report : Dict[str, Any]
            Report to export
        path : str or Path
            Path to save report
        format : str, optional
            Format to save report (json or csv, default: json)

        Returns:
        --------
        str
            Path to saved report
        """
        try:
            # Convert path to Path object
            path = Path(path)

            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Export based on format
            if format.lower() == 'json':
                return write_json(report, str(path))
            elif format.lower() == 'csv':
                return write_csv(report, str(path))
            else:
                self.logger.warning(f"Unsupported format: {format}. Using JSON.")
                return write_json(report, str(path))

        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise


    # Standalone utility functions

    def generate_report(
            processor,
            data: Optional[pd.DataFrame] = None,
            quasi_identifiers: Optional[List[str]] = None,
            sensitive_attributes: Optional[List[str]] = None,
            output_path: Optional[Union[str, Path]] = None,
            format: str = 'json',
            include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience function for generating and exporting l-diversity report

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
        format : str, optional
            Format to save report (json or csv, default: json)
        include_visualizations : bool, optional
            Whether to include visualization paths (default: True)

        Returns:
        --------
        Dict[str, Any]
            Generated report
        """
        # Create reporter instance
        reporter = LDiversityReport(processor)

        # Generate report
        report = reporter.generate(
            data,
            quasi_identifiers,
            sensitive_attributes,
            include_visualizations
        )

        # Export report if path provided
        if output_path:
            reporter.export_report(report, output_path, format)

        return report


    def generate_compliance_report(
            processor,
            regulation: str = "GDPR",
            data: Optional[pd.DataFrame] = None,
            quasi_identifiers: Optional[List[str]] = None,
            sensitive_attributes: Optional[List[str]] = None,
            output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Convenience function for generating and exporting compliance report

        Parameters:
        -----------
        processor : object
            L-Diversity processor instance
        regulation : str, optional
            Regulatory framework (default: GDPR)
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
            Generated compliance report
        """
        # Create reporter instance
        reporter = LDiversityReport(processor)

        # Generate compliance report
        report = reporter.generate_compliance_report(
            data,
            quasi_identifiers,
            sensitive_attributes,
            regulation
        )

        # Export report if path provided
        if output_path:
            reporter.export_report(report, output_path, 'json')

        return report


    def generate_technical_report(
            processor,
            data: Optional[pd.DataFrame] = None,
            quasi_identifiers: Optional[List[str]] = None,
            sensitive_attributes: Optional[List[str]] = None,
            output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Convenience function for generating and exporting technical report

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
            Generated technical report
        """
        # Create reporter instance
        reporter = LDiversityReport(processor)

        # Generate technical report
        report = reporter.generate_technical_report(
            data,
            quasi_identifiers,
            sensitive_attributes
        )

        # Export report if path provided
        if output_path:
            reporter.export_report(report, output_path, 'json')

        return report

    def _extract_entropy_metrics(self,
                                 group_diversity: pd.DataFrame,
                                 sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Extract entropy diversity metrics from group diversity data

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Group diversity data from cache
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Entropy diversity metrics
        """
        metrics = {
            "attribute_entropy": {}
        }

        for sa in sensitive_attributes:
            sa_column = f"{sa}_entropy"

            if sa_column in group_diversity.columns:
                # Calculate metrics
                min_entropy = group_diversity[sa_column].min()
                max_entropy = group_diversity[sa_column].max()
                avg_entropy = group_diversity[sa_column].mean()

                # Convert to effective number of classes
                effective_min = np.exp(min_entropy) if min_entropy > 0 else 1
                effective_max = np.exp(max_entropy) if max_entropy > 0 else 1
                effective_avg = np.exp(avg_entropy) if avg_entropy > 0 else 1

                # Store attribute metrics
                metrics["attribute_entropy"][sa] = {
                    "min_entropy": min_entropy,
                    "max_entropy": max_entropy,
                    "avg_entropy": avg_entropy,
                    "effective_min_l": effective_min,
                    "effective_max_l": effective_max,
                    "effective_avg_l": effective_avg,
                    "entropy_values": group_diversity[sa_column].tolist()
                }

        # Calculate overall metrics
        min_entropy_values = [metrics["attribute_entropy"][sa]["min_entropy"]
                              for sa in metrics["attribute_entropy"]]

        if min_entropy_values:
            metrics["min_entropy"] = min(min_entropy_values)
            metrics["max_entropy"] = max([metrics["attribute_entropy"][sa]["max_entropy"]
                                          for sa in metrics["attribute_entropy"]])
            metrics["avg_entropy"] = sum([metrics["attribute_entropy"][sa]["avg_entropy"]
                                          for sa in metrics["attribute_entropy"]]) / len(min_entropy_values)

            # Calculate effective values
            metrics["effective_min_l"] = np.exp(metrics["min_entropy"]) if metrics["min_entropy"] > 0 else 1
            metrics["effective_max_l"] = np.exp(metrics["max_entropy"]) if metrics["max_entropy"] > 0 else 1
            metrics["effective_avg_l"] = np.exp(metrics["avg_entropy"]) if metrics["avg_entropy"] > 0 else 1

            # Calculate compliance
            l_threshold = self.report_data.get(
                "l_diversity_configuration", {}).get("l_value", 3)
            metrics["compliant"] = metrics["effective_min_l"] >= l_threshold

        return metrics

    def _extract_recursive_metrics(self,
                                   group_diversity: pd.DataFrame,
                                   sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Extract recursive diversity metrics from group diversity data

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Group diversity data from cache
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Recursive diversity metrics
        """
        # Placeholder for recursive metrics
        metrics = {
            "attribute_recursive": {}
        }

        # Get c-value from configuration
        c_value = self.report_data.get(
            "l_diversity_configuration", {}).get("c_value", 1.0)

        # Add c-value to metrics
        metrics["c_value"] = c_value

        # Extract recursive metrics would require additional data
        # This is a simplified implementation

        return metrics

    def _calculate_distinct_metrics(self,
                                    data: pd.DataFrame,
                                    quasi_identifiers: List[str],
                                    sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate distinct diversity metrics directly from data

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
            Distinct diversity metrics
        """
        metrics = {
            "attribute_diversity": {}
        }

        # Group data by quasi-identifiers
        groups = data.groupby(quasi_identifiers)

        for sa in sensitive_attributes:
            if sa in data.columns:
                # Calculate distinct counts for each group
                distinct_counts = []

                for _, group in groups:
                    if sa in group.columns:
                        distinct_counts.append(group[sa].nunique())

                if distinct_counts:
                    # Calculate metrics
                    min_distinct = min(distinct_counts)
                    max_distinct = max(distinct_counts)
                    avg_distinct = sum(distinct_counts) / len(distinct_counts)

                    # Store attribute metrics
                    metrics["attribute_diversity"][sa] = {
                        "min_distinct": min_distinct,
                        "max_distinct": max_distinct,
                        "avg_distinct": avg_distinct,
                        "l_values": distinct_counts
                    }

        # Calculate overall metrics
        min_l_values = [metrics["attribute_diversity"][sa]["min_distinct"]
                        for sa in metrics["attribute_diversity"]]

        if min_l_values:
            metrics["min_l"] = min(min_l_values)
            metrics["max_l"] = max([metrics["attribute_diversity"][sa]["max_distinct"]
                                    for sa in metrics["attribute_diversity"]])
            metrics["avg_l"] = sum([metrics["attribute_diversity"][sa]["avg_distinct"]
                                    for sa in metrics["attribute_diversity"]]) / len(min_l_values)

            # Calculate compliance
            l_threshold = self.report_data.get(
                "l_diversity_configuration", {}).get("l_value", 3)
            metrics["compliant"] = all(l >= l_threshold for l in min_l_values)

        return metrics

    def _calculate_entropy_metrics(self,
                               data: pd.DataFrame,
                               quasi_identifiers: List[str],
                               sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate entropy diversity metrics directly from data.

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
            Entropy diversity metrics
        """
        metrics = {"attribute_entropy": {}}

        if not quasi_identifiers or not sensitive_attributes:
            return metrics  # Return empty metrics if no attributes are provided

        # Group data by quasi-identifiers
        groups = data.groupby(quasi_identifiers, dropna=False)

        for sa in sensitive_attributes:
            if sa in data.columns:
                entropy_values = []

                for _, group in groups:
                    if sa in group.columns and not group[sa].isna().all():
                        # Calculate value frequencies
                        value_counts = group[sa].value_counts(normalize=True, dropna=True)

                        # Calculate entropy
                        entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                        entropy_values.append(entropy)

                if entropy_values:
                    # Calculate entropy-based metrics
                    min_entropy = min(entropy_values)
                    max_entropy = max(entropy_values)
                    avg_entropy = sum(entropy_values) / len(entropy_values)

                    # Convert to effective number of classes
                    effective_min = np.exp(min_entropy) if min_entropy > 0 else 1
                    effective_max = np.exp(max_entropy) if max_entropy > 0 else 1
                    effective_avg = np.exp(avg_entropy) if avg_entropy > 0 else 1

                    # Store attribute metrics
                    metrics["attribute_entropy"][sa] = {
                        "min_entropy": min_entropy,
                        "max_entropy": max_entropy,
                        "avg_entropy": avg_entropy,
                        "effective_min_l": effective_min,
                        "effective_max_l": effective_max,
                        "effective_avg_l": effective_avg,
                        "entropy_values": entropy_values
                    }

        if metrics["attribute_entropy"]:
            # Calculate overall entropy statistics
            min_entropy_values = [m["min_entropy"] for m in metrics["attribute_entropy"].values()]
            max_entropy_values = [m["max_entropy"] for m in metrics["attribute_entropy"].values()]
            avg_entropy_values = [m["avg_entropy"] for m in metrics["attribute_entropy"].values()]

            metrics["min_entropy"] = min(min_entropy_values)
            metrics["max_entropy"] = max(max_entropy_values)
            metrics["avg_entropy"] = sum(avg_entropy_values) / len(avg_entropy_values)

            # Compute effective l-diversity values
            metrics["effective_min_l"] = np.exp(metrics["min_entropy"]) if metrics["min_entropy"] > 0 else 1
            metrics["effective_max_l"] = np.exp(metrics["max_entropy"]) if metrics["max_entropy"] > 0 else 1
            metrics["effective_avg_l"] = np.exp(metrics["avg_entropy"]) if metrics["avg_entropy"] > 0 else 1

            # Compliance check
            l_threshold = self.report_data.get("l_diversity_configuration", {}).get("l_value", 3)
            metrics["compliance"] = {
                "meets_l_diversity": metrics["effective_min_l"] >= l_threshold
            }

        return metrics