"""
PAMOLA.CORE - k-Anonymity Reporting Utilities
This module provides specialized reporting functionality for k-anonymity
anonymization models. It extends the base reporting infrastructure with
k-anonymity-specific metrics, visualizations, and report sections.

Key features:
- k-anonymity configuration reporting
- Dataset transformation details
- Risk assessment for re-identification
- Information loss metrics specific to k-anonymity
- Visualization integration for k-anonymity metrics

This module works with the k-anonymity processor to generate comprehensive
reports that can be used for documentation, compliance, and analysis.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any

from pamola_core.utils.base_reporting import PrivacyReport
from pamola_core.utils.io import write_json


# Configure logging
logger = logging.getLogger(__name__)


class KAnonymityReport(PrivacyReport):
    """
    Specialized report class for k-anonymity anonymization model.

    This class generates comprehensive reports about k-anonymity
    transformations, including metrics, visualizations, and analysis.
    """

    def __init__(self, report_data: Dict[str, Any]):
        """
        Initialize a k-anonymity report.

        Parameters
        -----------
        report_data : dict
            Dictionary containing k-anonymity report data.
        """
        super().__init__(report_data, "k-anonymity")

    def generate(self, include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive k-anonymity report.

        Parameters
        -----------
        include_visualizations : bool, optional
            Whether to include visualization paths in the report.

        Returns
        --------
        dict
            The compiled k-anonymity report.
        """
        # Verify that we have necessary data
        if not self.report_data:
            logger.warning("No k-anonymity data available for report generation")
            return {"error": "No k-anonymity data available"}

        # Compile report with metadata
        report = {
            "report_metadata": self.metadata
        }

        # Add k-anonymity configuration if available
        if "k_anonymity_configuration" in self.report_data:
            report["k_anonymity_configuration"] = self.report_data["k_anonymity_configuration"]

        # Add dataset information if available
        if "dataset_information" in self.report_data:
            report["dataset_information"] = self.report_data["dataset_information"]

        # Add anonymization evaluation if available
        if "privacy_evaluation" in self.report_data:
            report["privacy_evaluation"] = self.report_data["privacy_evaluation"]

        # Add anonymization result if available
        if "anonymization_result" in self.report_data:
            report["anonymization_result"] = self.report_data["anonymization_result"]

        # Add information loss metrics if available
        if "information_loss" in self.report_data:
            report["information_loss"] = self.report_data["information_loss"]

        # Add execution times if available
        if "execution_times" in self.report_data:
            report["execution_times"] = self.report_data["execution_times"]

        # Add visualization paths if included and available
        if include_visualizations and "visualization_paths" in self.report_data:
            report["visualizations"] = self.report_data["visualization_paths"]

        return report

    def get_summary(self) -> str:
        """
        Generate a concise summary of the k-anonymity report.

        Returns
        --------
        str
            A summary of key k-anonymity metrics and results.
        """
        summary = [
            "PAMOLA k-Anonymity Summary",
            "=========================",
            "",
            f"Generated on: {self.metadata['creation_time']}"
        ]

        # Add configuration overview
        if "k_anonymity_configuration" in self.report_data:
            config = self.report_data["k_anonymity_configuration"]
            summary.append("\nConfiguration:")
            summary.append(f"- k-value: {config.get('k_value', 'N/A')}")
            summary.append(f"- Method: {'Suppression' if config.get('suppression') else 'Masking'}")

        # Add dataset overview
        if "dataset_information" in self.report_data:
            dataset = self.report_data["dataset_information"]
            summary.append("\nDataset:")
            summary.append(f"- Records: {dataset.get('record_count', 'N/A')}")
            summary.append(f"- Quasi-identifiers: {len(dataset.get('quasi_identifiers', []))}")

        # Add key results
        if "anonymization_result" in self.report_data:
            result = self.report_data["anonymization_result"]
            summary.append("\nResults:")
            summary.append(f"- Records processed: {result.get('original_records', 'N/A')}")
            summary.append(f"- Records after anonymization: {result.get('anonymized_records', 'N/A')}")
            summary.append(f"- Records removed: {result.get('records_removed', 'N/A')}")

        # Add anonymization metrics
        if "privacy_evaluation" in self.report_data:
            privacy = self.report_data["privacy_evaluation"]
            summary.append("\nPrivacy:")
            summary.append(f"- Minimum k: {privacy.get('min_k', 'N/A')}")
            summary.append(f"- Records at risk: {privacy.get('at_risk_records', 'N/A')}")
            summary.append(f"- Threshold met: {'Yes' if privacy.get('threshold_met', False) else 'No'}")

        # Add information loss
        if "information_loss" in self.report_data:
            loss = self.report_data["information_loss"]
            summary.append("\nInformation Loss:")
            summary.append(f"- Overall: {loss.get('overall_information_loss', 'N/A')}%")

        return "\n".join(summary)


def generate_anonymization_report(report_data: Dict[str, Any],
                                  output_path: Optional[str] = None,
                                  include_visualizations: bool = True,
                                  format: str = "json") -> Dict[str, Any]:
    """
    Generates a comprehensive k-anonymity report.

    This function provides backward compatibility with previous implementation.
    For new code, consider using the KAnonymityReport class directly.

    Parameters
    -----------
    report_data : dict
        Dictionary containing k-anonymity report data.
    output_path : str, optional
        Path to save the report. If None, report is only returned but not saved.
    include_visualizations : bool, optional
        Whether to include visualizations in the report (default: True).
    format : str, optional
        Report format: 'json', 'html', or 'text' (default: 'json').

    Returns
    --------
    dict
        The complete k-anonymity report.
    """
    try:
        # Create a report object
        report = KAnonymityReport(report_data)

        # Generate the report
        result = report.generate(include_visualizations)

        # Save if output path is provided
        if output_path:
            report.save(output_path, format)

        return result

    except Exception as e:
        logger.error(f"Error generating k-anonymity report: {e}")
        raise


def generate_threshold_report(report_data: Dict[str, Any],
                              output_path: Optional[str] = None,
                              k_threshold: Optional[int] = None) -> Dict[str, Any]:
    """
    Generates a neutral k-anonymity threshold report.

    States whether the minimum group size reached the threshold and how many
    records fall below it. The threshold is chosen by the caller (default: the
    k the data was processed with). No law or regulation is mapped to a
    threshold and the result is not a statement of legal compliance; it is
    technical evidence for the data owner's own assessment.

    Parameters
    -----------
    report_data : dict
        Dictionary containing k-anonymity report data.
    output_path : str, optional
        Path to save the report.
    k_threshold : int, optional
        Minimum k to test against. Defaults to the configured k value.

    Returns
    --------
    dict
        Report with ``k_threshold``, ``actual_min_k``, ``threshold_met``,
        ``at_risk_records`` and ``at_risk_percentage``.
    """
    try:
        import pamola_core.configs.settings as config_settings

        if k_threshold is None:
            k_threshold = report_data.get("k_anonymity_configuration", {}).get(
                "k_value"
            )

        threshold_report = {
            "report_metadata": {
                "creation_time": datetime.now().isoformat(),
                "pamola_version": getattr(
                    config_settings, "PAMOLA_VERSION", "unknown"
                ),
                "report_type": "k-anonymity threshold check"
            },
            "k_threshold": k_threshold,
            "actual_min_k": None,
            "threshold_met": None,
            "at_risk_records": None,
            "at_risk_percentage": None,
        }

        if "privacy_evaluation" in report_data:
            privacy = report_data["privacy_evaluation"]
            actual_min_k = privacy.get("min_k", 0)
            threshold_report["actual_min_k"] = actual_min_k
            threshold_report["at_risk_records"] = privacy.get("at_risk_records", "N/A")
            threshold_report["at_risk_percentage"] = privacy.get(
                "percentage_at_risk", "N/A"
            )
            if k_threshold is not None:
                threshold_report["threshold_met"] = bool(actual_min_k >= k_threshold)
                if not threshold_report["threshold_met"]:
                    threshold_report["reason"] = (
                        f"Minimum k value ({actual_min_k}) is below the "
                        f"threshold ({k_threshold})"
                    )

        if "dataset_information" in report_data:
            threshold_report["dataset"] = report_data["dataset_information"]

        if "anonymization_result" in report_data:
            threshold_report["anonymization_details"] = {
                "method": report_data["anonymization_result"].get("method", "Unknown"),
                "execution_time": report_data["anonymization_result"].get("execution_time", "N/A"),
                "timestamp": report_data["anonymization_result"].get("timestamp", "N/A")
            }

        if output_path:
            write_json(threshold_report, output_path)
            logger.info(f"Threshold report saved to {output_path}")

        return threshold_report

    except Exception as e:
        logger.error(f"Error generating threshold report: {e}")
        raise
