"""
PAMOLA.CORE - L-Diversity Compliance Reporting
----------------------------------------------
This module provides specialized functions for generating
regulatory compliance reports for l-diversity anonymization.

Key Features:
- GDPR compliance assessment
- HIPAA compliance assessment
- CCPA compliance assessment
- Compliance recommendations
- Regulatory requirement lookup

This module extends the main reporting functionality with
specialized regulatory compliance reporting capabilities.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd

# Import from pamola_core utilities
from pamola_core import configs
from pamola_core.utils.io import write_json

# Configure logging
logger = logging.getLogger(__name__)


def generate_gdpr_compliance_report(
        processor,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate GDPR compliance report for l-diversity anonymization

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
        GDPR compliance report
    """
    # Import main report class
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport

    # Create reporter instance
    reporter = LDiversityReport(processor)

    # Generate GDPR compliance report
    report = reporter.generate_compliance_report(
        data,
        quasi_identifiers,
        sensitive_attributes,
        "GDPR"
    )

    # Add GDPR-specific sections
    report["gdpr_article_5_compliance"] = assess_gdpr_article_5_compliance(report)
    report["gdpr_article_25_compliance"] = assess_gdpr_article_25_compliance(report)
    report["gdpr_article_35_compliance"] = assess_gdpr_article_35_compliance(report)

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, str(path))
        logger.info(f"GDPR compliance report saved to {path}")

    return report


def generate_hipaa_compliance_report(
        processor,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate HIPAA compliance report for l-diversity anonymization

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
        HIPAA compliance report
    """
    # Import main report class
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport

    # Create reporter instance
    reporter = LDiversityReport(processor)

    # Generate HIPAA compliance report
    report = reporter.generate_compliance_report(
        data,
        quasi_identifiers,
        sensitive_attributes,
        "HIPAA"
    )

    # Add HIPAA-specific sections
    report["hipaa_deidentification_standard"] = assess_hipaa_deidentification(report)
    report["phi_protection_assessment"] = assess_phi_protection(
        report, sensitive_attributes
    )

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, str(path))
        logger.info(f"HIPAA compliance report saved to {path}")

    return report


def generate_ccpa_compliance_report(
        processor,
        data: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate CCPA compliance report for l-diversity anonymization

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
        CCPA compliance report
    """
    # Import main report class
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport

    # Create reporter instance
    reporter = LDiversityReport(processor)

    # Generate CCPA compliance report
    report = reporter.generate_compliance_report(
        data,
        quasi_identifiers,
        sensitive_attributes,
        "CCPA"
    )

    # Add CCPA-specific sections
    report["ccpa_deidentification_assessment"] = assess_ccpa_deidentification(report)

    # Export report if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, str(path))
        logger.info(f"CCPA compliance report saved to {path}")

    return report


def assess_gdpr_article_5_compliance(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess compliance with GDPR Article 5 (data minimization)

    Parameters:
    -----------
    report : Dict[str, Any]
        Compliance report

    Returns:
    --------
    Dict[str, Any]
        GDPR Article 5 compliance assessment
    """
    compliance_status = report.get("compliance_status", "Unknown")

    # Simplified Article 5 assessment
    assessment = {
        "article": "Article 5 - Principles relating to processing of personal data",
        "compliant": compliance_status == "Compliant",
        "principles": {
            "data_minimization": {
                "compliant": compliance_status == "Compliant",
                "description": "Personal data shall be adequate, relevant and limited to what is necessary "
                               "in relation to the purposes for which they are processed."
            },
            "storage_limitation": {
                "compliant": compliance_status == "Compliant",
                "description": "Personal data shall be kept in a form which permits identification of data "
                               "subjects for no longer than is necessary for the purposes for which the "
                               "personal data are processed."
            }
        }
    }

    # Add assessment description
    if assessment["compliant"]:
        assessment["description"] = "The anonymization process complies with GDPR Article 5 principles."
    else:
        assessment["description"] = "The anonymization process does not fully comply with GDPR Article 5 principles."

    return assessment