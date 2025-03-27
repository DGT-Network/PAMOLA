"""
PAMOLA.CORE - l-Diversity Reporting Utilities
---------------------------------------------
This module provides specialized reporting functionality for l-diversity
anonymization models. It extends the base reporting infrastructure with
l-diversity-specific metrics, visualizations, and report sections.

Key features:
- l-diversity configuration reporting
- Dataset transformation details
- Risk assessment for attribute disclosure
- Multiple l-diversity type support (distinct, entropy, recursive)
- Information loss metrics specific to l-diversity
- Visualization integration for l-diversity metrics
- Regulatory compliance evaluation for different privacy frameworks

This module works with the l-diversity processor to generate comprehensive
reports that can be used for documentation, compliance, and analysis.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import logging
import math
from datetime import datetime
from typing import Dict, Optional, Any, List

from pamola_core import config
from pamola_core.utils.base_reporting import PrivacyReport
from pamola_core.utils.file_io import write_json, write_csv

# Configure logging
logger = logging.getLogger(__name__)


class LDiversityReport(PrivacyReport):
    """
    Specialized report class for l-diversity anonymization model.

    This class generates comprehensive reports about l-diversity
    transformations, including metrics, visualizations, and analysis.
    It supports different types of l-diversity (distinct, entropy, recursive).
    """

    def __init__(self, report_data: Dict[str, Any]):
        """
        Initialize an l-diversity report.

        Parameters:
        -----------
        report_data : dict
            Dictionary containing l-diversity report data.
        """
        super().__init__(report_data, "l-diversity")

        # Extract l-diversity type for specialized report sections
        self.diversity_type = "distinct"  # Default to distinct
        if "l_diversity_configuration" in report_data:
            self.diversity_type = report_data["l_diversity_configuration"].get("diversity_type", "distinct")

    def generate(self, include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive l-diversity report.

        Parameters:
        -----------
        include_visualizations : bool, optional
            Whether to include visualization paths in the report.

        Returns:
        --------
        dict
            The compiled l-diversity report.
        """
        # Verify that we have necessary data
        if not self.report_data:
            logger.warning("No l-diversity data available for report generation")
            return {"error": "No l-diversity data available"}

        # Compile report with metadata
        report = {
            "report_metadata": self.metadata
        }

        # Add l-diversity configuration if available
        if "l_diversity_configuration" in self.report_data:
            report["l_diversity_configuration"] = self.report_data["l_diversity_configuration"]

        # Add dataset information if available
        if "dataset_information" in self.report_data:
            report["dataset_information"] = self.report_data["dataset_information"]

        # Add privacy evaluation if available
        if "privacy_evaluation" in self.report_data:
            report["privacy_evaluation"] = self.report_data["privacy_evaluation"]

        # Add anonymization result if available
        if "anonymization_result" in self.report_data:
            report["anonymization_result"] = self.report_data["anonymization_result"]

        # Add sensitive attribute metrics if available
        if "sensitive_attribute_metrics" in self.report_data:
            report["sensitive_attribute_metrics"] = self.report_data["sensitive_attribute_metrics"]

        # Add information loss metrics if available
        if "information_loss" in self.report_data:
            report["information_loss"] = self.report_data["information_loss"]

        # Add execution times if available
        if "execution_times" in self.report_data:
            report["execution_times"] = self.report_data["execution_times"]

        # Add visualization paths if included and available
        if include_visualizations and "visualization_paths" in self.report_data:
            report["visualizations"] = self.report_data["visualization_paths"]

        # Add diversity-type specific metrics if available
        diversity_metrics_key = f"{self.diversity_type}_diversity_metrics"
        if diversity_metrics_key in self.report_data:
            report[diversity_metrics_key] = self.report_data[diversity_metrics_key]

        return report

    def get_summary(self) -> str:
        """
        Generate a concise summary of the l-diversity report.

        Returns:
        --------
        str
            A summary of key l-diversity metrics and results.
        """
        summary = [
            f"PAMOLA l-Diversity Summary",
            "=========================",
            "",
            f"Generated on: {self.metadata['creation_time']}"
        ]

        # Add configuration overview
        if "l_diversity_configuration" in self.report_data:
            config = self.report_data["l_diversity_configuration"]
            summary.append(f"\nConfiguration:")
            summary.append(f"- l-value: {config.get('l_value', 'N/A')}")
            summary.append(f"- Diversity type: {config.get('diversity_type', 'distinct')}")
            if config.get('diversity_type') == "recursive":
                summary.append(f"- c-value: {config.get('c_value', 'N/A')}")
            summary.append(f"- k-value (base): {config.get('k_value', 'N/A')}")
            summary.append(f"- Method: {'Suppression' if config.get('suppression') else 'Masking'}")

        # Add dataset overview
        if "dataset_information" in self.report_data:
            dataset = self.report_data["dataset_information"]
            summary.append(f"\nDataset:")
            summary.append(f"- Records: {dataset.get('record_count', 'N/A')}")
            summary.append(f"- Quasi-identifiers: {len(dataset.get('quasi_identifiers', []))}")
            summary.append(f"- Sensitive attributes: {len(dataset.get('sensitive_attributes', []))}")

        # Add key results
        if "anonymization_result" in self.report_data:
            result = self.report_data["anonymization_result"]
            summary.append(f"\nResults:")
            summary.append(f"- Records processed: {result.get('original_records', 'N/A')}")
            summary.append(f"- Records after anonymization: {result.get('anonymized_records', 'N/A')}")
            summary.append(f"- Records removed: {result.get('records_removed', 'N/A')}")
            if not result.get('suppression', True):
                summary.append(f"- Records masked: {result.get('records_masked', 'N/A')}")

        # Add privacy evaluation metrics
        if "privacy_evaluation" in self.report_data:
            privacy = self.report_data["privacy_evaluation"]
            summary.append(f"\nPrivacy:")
            summary.append(f"- Minimum l: {privacy.get('min_l', 'N/A')}")
            summary.append(f"- Records at risk: {privacy.get('at_risk_records', 'N/A')}")
            summary.append(f"- Compliance: {'Yes' if privacy.get('compliant', False) else 'No'}")

            # Add diversity type specific metrics
            if self.diversity_type == "entropy" and "entropy_metrics" in privacy:
                summary.append(f"- Average entropy: {privacy['entropy_metrics'].get('avg_entropy', 'N/A')}")
            elif self.diversity_type == "recursive" and "recursive_metrics" in privacy:
                summary.append(f"- (c,l)-diverse groups: {privacy['recursive_metrics'].get('compliant_groups', 'N/A')}")

        # Add information loss
        if "information_loss" in self.report_data:
            loss = self.report_data["information_loss"]
            summary.append(f"\nInformation Loss:")
            summary.append(f"- Overall: {loss.get('overall_information_loss', 'N/A')}%")

        return "\n".join(summary)

    def generate_detailed_report(self) -> Dict[str, Any]:
        """
        Generate a detailed technical report with comprehensive metrics.

        This extended report includes detailed statistics about each
        sensitive attribute, group-level diversity metrics, and
        technical evaluation data.

        Returns:
        --------
        dict
            Detailed technical report with comprehensive metrics.
        """
        # Start with the basic report
        report = self.generate(include_visualizations=True)

        # Add sensitive attribute details if available
        if "sensitive_attribute_details" in self.report_data:
            report["sensitive_attribute_details"] = self.report_data["sensitive_attribute_details"]

        # Add group-level diversity metrics if available
        if "group_diversity_metrics" in self.report_data:
            report["group_diversity_metrics"] = self.report_data["group_diversity_metrics"]

        # Add technical metrics based on diversity type
        diversity_type = self.report_data.get("l_diversity_configuration", {}).get("diversity_type", "distinct")

        if f"{diversity_type}_technical_metrics" in self.report_data:
            report[f"{diversity_type}_technical_metrics"] = self.report_data[f"{diversity_type}_technical_metrics"]

        # Add comparison to other privacy models if available
        if "privacy_model_comparison" in self.report_data:
            report["privacy_model_comparison"] = self.report_data["privacy_model_comparison"]

        return report


def generate_anonymization_report(report_data: Dict[str, Any],
                                  output_path: Optional[str] = None,
                                  include_visualizations: bool = True,
                                  format: str = "json") -> Dict[str, Any]:
    """
    Generates a comprehensive l-diversity report.

    This function provides a standardized interface for report generation.
    For more customized reports, consider using the LDiversityReport class directly.

    Parameters:
    -----------
    report_data : dict
        Dictionary containing l-diversity report data.
    output_path : str, optional
        Path to save the report. If None, report is only returned but not saved.
    include_visualizations : bool, optional
        Whether to include visualizations in the report (default: True).
    format : str, optional
        Report format: 'json', 'html', or 'text' (default: 'json').

    Returns:
    --------
    dict
        The complete l-diversity report.
    """
    try:
        # Create a report object
        report = LDiversityReport(report_data)

        # Generate the report
        result = report.generate(include_visualizations)

        # Save if output path is provided
        if output_path:
            report.save(output_path, format)

        return result

    except Exception as e:
        logger.error(f"Error generating l-diversity report: {e}")
        raise


def generate_compliance_report(report_data: Dict[str, Any],
                               output_path: Optional[str] = None,
                               regulation: str = "GDPR") -> Dict[str, Any]:
    """
    Generates a regulatory compliance report for l-diversity.

    This specialized report focuses on compliance aspects of
    l-diversity transformations for specific regulations.

    Parameters:
    -----------
    report_data : dict
        Dictionary containing l-diversity report data.
    output_path : str, optional
        Path to save the report.
    regulation : str, optional
        Regulation to check compliance against ('GDPR', 'HIPAA', 'CCPA', etc.)

    Returns:
    --------
    dict
        The compliance report.
    """
    try:
        # Create base report
        report = LDiversityReport(report_data)
        base_report = report.generate(include_visualizations=False)

        # Extract compliance-relevant information
        compliance_report = {
            "report_metadata": {
                "creation_time": datetime.now().isoformat(),
                "pamola_version": getattr(config, "PAMOLA_VERSION", "unknown"),
                "report_type": f"l-diversity {regulation} compliance"
            },
            "regulation": regulation,
            "compliance_status": "Unknown",
            "compliance_details": {}
        }

        # Get diversity type from configuration
        diversity_type = report_data.get("l_diversity_configuration", {}).get("diversity_type", "distinct")

        # Determine compliance based on regulation
        if "privacy_evaluation" in report_data:
            privacy = report_data["privacy_evaluation"]

            # Extract l-diversity configuration
            l_config = report_data.get("l_diversity_configuration", {})
            l_value = l_config.get("l_value", 3)  # Default to 3 if not specified
            diversity_type = l_config.get("diversity_type", "distinct")
            k_value = l_config.get("k_value", 2)  # Base k-anonymity level

            if regulation.upper() == "GDPR":
                # Set compliance thresholds based on GDPR recommendations
                # For GDPR, distinct l-diversity should be at least 3
                # k-value (base anonymity) should be at least 5
                l_threshold = 3
                k_threshold = 5

                # Add compliance details
                compliance_report["compliance_details"] = {
                    "required_l": l_threshold,
                    "required_k": k_threshold,
                    "actual_l": privacy.get("min_l", 0),
                    "actual_k": k_value,
                    "diversity_type": diversity_type,
                    "at_risk_records": privacy.get("at_risk_records", "N/A"),
                    "at_risk_percentage": privacy.get("percentage_at_risk", "N/A")
                }

                # For entropy l-diversity, ensure adequate entropy level
                if diversity_type == "entropy":
                    compliance_report["compliance_details"]["required_entropy"] = math.log(l_threshold)
                    if "entropy_metrics" in privacy:
                        compliance_report["compliance_details"]["actual_entropy"] = privacy["entropy_metrics"].get(
                            "min_entropy", 0)

                # For recursive (c,l)-diversity, ensure c parameter is adequate
                if diversity_type == "recursive":
                    compliance_report["compliance_details"]["required_c"] = 0.5  # Recommended minimum c value
                    compliance_report["compliance_details"]["actual_c"] = l_config.get("c_value", 0)

                # Determine overall compliance
                k_compliant = k_value >= k_threshold
                l_compliant = privacy.get("min_l", 0) >= l_threshold

                if k_compliant and l_compliant:
                    compliance_report["compliance_status"] = "Compliant"
                else:
                    compliance_report["compliance_status"] = "Non-compliant"
                    reasons = []
                    if not k_compliant:
                        reasons.append(f"Base k-anonymity ({k_value}) is below required threshold ({k_threshold})")
                    if not l_compliant:
                        reasons.append(
                            f"Minimum l-diversity ({privacy.get('min_l', 0)}) is below required threshold ({l_threshold})")
                    compliance_report["compliance_details"]["reasons"] = reasons

            elif regulation.upper() == "HIPAA":
                # HIPAA has stricter requirements for healthcare data
                l_threshold = 4  # Recommended minimum l value for HIPAA
                k_threshold = 10  # Recommended minimum k value for HIPAA

                # Add compliance details
                compliance_report["compliance_details"] = {
                    "required_l": l_threshold,
                    "required_k": k_threshold,
                    "actual_l": privacy.get("min_l", 0),
                    "actual_k": k_value,
                    "diversity_type": diversity_type,
                    "at_risk_records": privacy.get("at_risk_records", "N/A")
                }

                # Determine overall compliance
                k_compliant = k_value >= k_threshold
                l_compliant = privacy.get("min_l", 0) >= l_threshold

                if k_compliant and l_compliant:
                    compliance_report["compliance_status"] = "Compliant"
                else:
                    compliance_report["compliance_status"] = "Non-compliant"
                    reasons = []
                    if not k_compliant:
                        reasons.append(f"Base k-anonymity ({k_value}) is below required threshold ({k_threshold})")
                    if not l_compliant:
                        reasons.append(
                            f"Minimum l-diversity ({privacy.get('min_l', 0)}) is below required threshold ({l_threshold})")
                    compliance_report["compliance_details"]["reasons"] = reasons

                # HIPAA requires special handling for sensitive identifiers
                if "sensitive_attribute_metrics" in report_data:
                    sa_metrics = report_data["sensitive_attribute_metrics"]
                    compliance_report["compliance_details"]["sensitive_attribute_compliance"] = sa_metrics

            elif regulation.upper() == "CCPA":
                # California Consumer Privacy Act
                l_threshold = 3  # Recommended minimum l value for CCPA
                k_threshold = 4  # Recommended minimum k value for CCPA

                # Add compliance details
                compliance_report["compliance_details"] = {
                    "required_l": l_threshold,
                    "required_k": k_threshold,
                    "actual_l": privacy.get("min_l", 0),
                    "actual_k": k_value,
                    "diversity_type": diversity_type,
                    "at_risk_records": privacy.get("at_risk_records", "N/A"),
                    "at_risk_percentage": privacy.get("percentage_at_risk", "N/A")
                }

                # Determine overall compliance
                k_compliant = k_value >= k_threshold
                l_compliant = privacy.get("min_l", 0) >= l_threshold

                if k_compliant and l_compliant:
                    compliance_report["compliance_status"] = "Compliant"
                else:
                    compliance_report["compliance_status"] = "Non-compliant"
                    reasons = []
                    if not k_compliant:
                        reasons.append(f"Base k-anonymity ({k_value}) is below required threshold ({k_threshold})")
                    if not l_compliant:
                        reasons.append(
                            f"Minimum l-diversity ({privacy.get('min_l', 0)}) is below required threshold ({l_threshold})")
                    compliance_report["compliance_details"]["reasons"] = reasons

            else:
                # Default case
                compliance_report["compliance_status"] = "Unknown regulation"
                compliance_report["compliance_details"][
                    "note"] = f"No specific compliance criteria defined for {regulation}"

        # Add dataset information
        if "dataset_information" in report_data:
            compliance_report["dataset"] = report_data["dataset_information"]

        # Add anonymization details
        if "anonymization_result" in report_data:
            compliance_report["anonymization_details"] = {
                "method": report_data["anonymization_result"].get("method", "Unknown"),
                "execution_time": report_data["anonymization_result"].get("execution_time", "N/A"),
                "timestamp": report_data["anonymization_result"].get("timestamp", "N/A")
            }

        # Save if output path is provided
        if output_path:
            write_json(compliance_report, output_path)
            logger.info(f"Compliance report saved to {output_path}")

        return compliance_report

    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise


def generate_technical_report(report_data: Dict[str, Any],
                              sensitive_attributes: List[str],
                              output_path: Optional[str] = None,
                              format: str = "json") -> Dict[str, Any]:
    """
    Generates a detailed technical report focused on l-diversity properties
    of sensitive attributes.

    This report provides in-depth analysis of how l-diversity affects each
    sensitive attribute and the distribution of values.

    Parameters:
    -----------
    report_data : dict
        Dictionary containing l-diversity report data.
    sensitive_attributes : list[str]
        List of sensitive attributes to analyze in detail.
    output_path : str, optional
        Path to save the report.
    format : str, optional
        Report format: 'json', 'html', or 'text' (default: 'json').

    Returns:
    --------
    dict
        The technical report with detailed sensitive attribute metrics.
    """
    try:
        # Create a report object
        report = LDiversityReport(report_data)

        # Get basic report
        basic_report = report.generate(include_visualizations=False)

        # Create technical report structure
        technical_report = {
            "report_metadata": basic_report["report_metadata"],
            "technical_analysis": {
                "created": datetime.now().isoformat(),
                "focus": "l-diversity sensitive attribute analysis"
            }
        }

        # Add configuration details
        if "l_diversity_configuration" in report_data:
            technical_report["configuration"] = report_data["l_diversity_configuration"]

        # Add analysis for each sensitive attribute
        technical_report["sensitive_attribute_analysis"] = {}

        for sa in sensitive_attributes:
            # Skip if attribute is not in the data
            if not sa in report_data.get("sensitive_attribute_metrics", {}):
                logger.warning(f"No metrics found for sensitive attribute: {sa}")
                continue

            # Get metrics for this attribute
            sa_metrics = report_data["sensitive_attribute_metrics"].get(sa, {})

            # Create attribute report
            attr_report = {
                "distinct_values": sa_metrics.get("distinct_values", 0),
                "diversity_level": sa_metrics.get("diversity_level", 0),
                "distribution": sa_metrics.get("value_distribution", {}),
                "at_risk_groups": sa_metrics.get("at_risk_groups", 0),
                "risk_metrics": sa_metrics.get("risk_metrics", {})
            }

            # Add diversity type specific metrics
            diversity_type = report_data.get("l_diversity_configuration", {}).get("diversity_type", "distinct")

            if diversity_type == "entropy" and "entropy" in sa_metrics:
                attr_report["entropy_metrics"] = {
                    "min_entropy": sa_metrics["entropy"].get("min", 0),
                    "max_entropy": sa_metrics["entropy"].get("max", 0),
                    "avg_entropy": sa_metrics["entropy"].get("avg", 0),
                    "distribution": sa_metrics["entropy"].get("distribution", {})
                }

            elif diversity_type == "recursive" and "recursive" in sa_metrics:
                attr_report["recursive_metrics"] = {
                    "c_value": report_data.get("l_diversity_configuration", {}).get("c_value", 1.0),
                    "compliant_groups": sa_metrics["recursive"].get("compliant_groups", 0),
                    "non_compliant_groups": sa_metrics["recursive"].get("non_compliant_groups", 0),
                    "skewness": sa_metrics["recursive"].get("skewness", 0)
                }

            # Add attribute report to the technical report
            technical_report["sensitive_attribute_analysis"][sa] = attr_report

        # Add execution details
        if "execution_times" in report_data:
            technical_report["performance_metrics"] = report_data["execution_times"]

        # Save if output path is provided
        if output_path:
            if format.lower() == "json":
                write_json(technical_report, output_path)
            elif format.lower() == "csv":
                write_csv(technical_report, output_path)
            else:
                logger.warning(f"Unsupported format: {format}. Using JSON instead.")
                write_json(technical_report, output_path)

            logger.info(f"Technical report saved to {output_path}")

        return technical_report

    except Exception as e:
        logger.error(f"Error generating technical report: {e}")
        raise


def get_diversity_requirements(regulation: str, diversity_type: str = "distinct") -> Dict[str, Any]:
    """
    Returns standard l-diversity requirements for different regulations.

    This utility function helps in quickly determining the recommended
    diversity levels for different regulatory frameworks.

    Parameters:
    -----------
    regulation : str
        Regulation to get requirements for ('GDPR', 'HIPAA', 'CCPA', etc.)
    diversity_type : str, optional
        Type of l-diversity ('distinct', 'entropy', 'recursive').

    Returns:
    --------
    dict
        Dictionary with recommended l-diversity parameters.
    """
    # Default requirements
    default_requirements = {
        "l_value": 3,
        "k_value": 2,
        "diversity_type": diversity_type,
        "c_value": 1.0 if diversity_type == "recursive" else None
    }

    regulation = regulation.upper()

    if regulation == "GDPR":
        requirements = {
            "l_value": 3,
            "k_value": 5,  # GDPR typically requires k>=5
            "diversity_type": diversity_type,
            "c_value": 0.5 if diversity_type == "recursive" else None,
            "description": "General Data Protection Regulation requirements"
        }

    elif regulation == "HIPAA":
        requirements = {
            "l_value": 4,  # Stricter for healthcare data
            "k_value": 10,  # HIPAA often requires k>=10
            "diversity_type": diversity_type,
            "c_value": 0.33 if diversity_type == "recursive" else None,
            "description": "Health Insurance Portability and Accountability Act requirements"
        }

    elif regulation == "CCPA":
        requirements = {
            "l_value": 3,
            "k_value": 4,
            "diversity_type": diversity_type,
            "c_value": 0.5 if diversity_type == "recursive" else None,
            "description": "California Consumer Privacy Act requirements"
        }

    else:
        logger.warning(f"Unknown regulation: {regulation}. Using default requirements.")
        requirements = default_requirements
        requirements["description"] = "Default l-diversity recommendations"

    return requirements