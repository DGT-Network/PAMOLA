"""
PAMOLA.CORE - L-Diversity Compliance Reporting
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
import pandas as pd

# Import from pamola_core utilities
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

    Parameters
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

    Returns
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

    Parameters
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

    Returns
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

    Parameters
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

    Returns
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

    Parameters
    -----------
    report : Dict[str, Any]
        Compliance report

    Returns
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


def assess_gdpr_article_25_compliance(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess compliance with GDPR Article 25 (Data Protection by Design and by Default).

    Evaluates whether the l-diversity configuration demonstrates privacy-by-design
    through adequate diversity thresholds and appropriate diversity type selection.
    """
    compliance_status = report.get("compliance_status", "Unknown")
    l_value = report.get("l_value", 0)
    diversity_type = report.get("diversity_type", "unknown")

    by_design = l_value >= 3 and diversity_type in ("distinct", "entropy", "recursive")
    by_default = compliance_status == "Compliant"

    assessment = {
        "article": "Article 25 - Data protection by design and by default",
        "compliant": by_design and by_default,
        "principles": {
            "by_design": {
                "compliant": by_design,
                "description": (
                    "Appropriate technical measures (l-diversity) implemented "
                    "to protect personal data during processing."
                ),
                "evidence": {
                    "l_value": l_value,
                    "diversity_type": diversity_type,
                    "minimum_recommended_l": 3,
                },
            },
            "by_default": {
                "compliant": by_default,
                "description": (
                    "Only personal data necessary for each specific purpose "
                    "is processed, with adequate diversity guarantees."
                ),
            },
        },
    }

    if assessment["compliant"]:
        assessment["description"] = (
            "The l-diversity configuration satisfies Article 25 requirements "
            "for data protection by design and by default."
        )
    else:
        recommendations = []
        if not by_design:
            recommendations.append(
                f"Increase l-value from {l_value} to at least 3 "
                "and use a recognized diversity type (distinct, entropy, or recursive)."
            )
        if not by_default:
            recommendations.append(
                "Ensure all equivalence classes meet the configured l-diversity threshold."
            )
        assessment["description"] = (
            "The anonymization does not fully satisfy Article 25. "
            "See recommendations."
        )
        assessment["recommendations"] = recommendations

    return assessment


def assess_gdpr_article_35_compliance(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess compliance with GDPR Article 35 (Data Protection Impact Assessment).

    Evaluates whether a DPIA is warranted and whether the l-diversity measures
    adequately mitigate re-identification risk.
    """
    compliance_status = report.get("compliance_status", "Unknown")
    at_risk_pct = report.get("at_risk_percentage", 100.0)
    l_value = report.get("l_value", 0)

    risk_level = "low" if at_risk_pct < 5 else ("medium" if at_risk_pct < 20 else "high")
    dpia_required = risk_level in ("medium", "high")
    risk_mitigated = compliance_status == "Compliant" and at_risk_pct < 10

    assessment = {
        "article": "Article 35 - Data protection impact assessment",
        "dpia_required": dpia_required,
        "risk_level": risk_level,
        "risk_mitigated": risk_mitigated,
        "compliant": risk_mitigated or not dpia_required,
        "evidence": {
            "at_risk_percentage": at_risk_pct,
            "l_value": l_value,
            "compliance_status": compliance_status,
        },
    }

    if not dpia_required:
        assessment["description"] = (
            "Low re-identification risk — a full DPIA is not mandatory, "
            "but periodic review is recommended."
        )
    elif risk_mitigated:
        assessment["description"] = (
            "DPIA recommended due to data sensitivity. Current l-diversity "
            "measures adequately mitigate the identified risks."
        )
    else:
        assessment["description"] = (
            f"DPIA required. {at_risk_pct:.1f}% of records are at risk. "
            "Current l-diversity measures may be insufficient."
        )
        assessment["recommendations"] = [
            f"Increase l-value (currently {l_value}) to reduce at-risk records below 10%.",
            "Consider using entropy or recursive l-diversity for stronger guarantees.",
            "Document residual risks and mitigation plan in the DPIA report.",
        ]

    return assessment


def assess_hipaa_deidentification(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess HIPAA de-identification compliance (Safe Harbor / Expert Determination).

    Under HIPAA Safe Harbor (45 CFR 164.514(b)(2)), 18 identifier categories must be
    removed or generalized. L-diversity provides additional protection for sensitive
    health attributes beyond k-anonymity.
    """
    compliance_status = report.get("compliance_status", "Unknown")
    l_value = report.get("l_value", 0)
    at_risk_pct = report.get("at_risk_percentage", 100.0)

    safe_harbor_met = compliance_status == "Compliant" and l_value >= 2
    expert_determination_met = at_risk_pct < 5 and l_value >= 3

    assessment = {
        "standard": "HIPAA De-identification (45 CFR 164.514)",
        "safe_harbor": {
            "met": safe_harbor_met,
            "description": (
                "Safe Harbor requires removal/generalization of 18 identifier types. "
                "L-diversity adds sensitive attribute protection beyond Safe Harbor."
            ),
            "l_value": l_value,
            "minimum_recommended_l": 2,
        },
        "expert_determination": {
            "met": expert_determination_met,
            "description": (
                "Expert Determination requires statistical/scientific evidence "
                "that re-identification risk is very small."
            ),
            "at_risk_percentage": at_risk_pct,
            "threshold": 5.0,
        },
        "compliant": safe_harbor_met or expert_determination_met,
    }

    if assessment["compliant"]:
        method = "Safe Harbor" if safe_harbor_met else "Expert Determination"
        assessment["description"] = (
            f"HIPAA de-identification requirements met via {method} method."
        )
    else:
        assessment["description"] = (
            "HIPAA de-identification requirements not fully met. "
            "L-diversity alone is insufficient without proper identifier removal."
        )
        assessment["recommendations"] = [
            "Ensure all 18 HIPAA identifier categories are removed or generalized.",
            f"Increase l-value from {l_value} to at least 3 for Expert Determination.",
            "Document the de-identification methodology for compliance audit.",
        ]

    return assessment


def assess_phi_protection(
    report: Dict[str, Any],
    sensitive_attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess Protected Health Information (PHI) protection level.

    Evaluates whether sensitive health attributes are adequately protected
    by the l-diversity configuration against attribute disclosure attacks.
    """
    compliance_status = report.get("compliance_status", "Unknown")
    l_value = report.get("l_value", 0)
    diversity_type = report.get("diversity_type", "unknown")

    phi_categories = [
        "diagnosis", "treatment", "medication", "lab_result",
        "procedure", "condition", "allergy", "vital_sign",
    ]

    protected_attrs = sensitive_attributes or []
    phi_attrs = [a for a in protected_attrs if any(cat in a.lower() for cat in phi_categories)]
    non_phi_attrs = [a for a in protected_attrs if a not in phi_attrs]

    adequate_protection = compliance_status == "Compliant" and l_value >= 3
    strong_protection = adequate_protection and diversity_type in ("entropy", "recursive")

    assessment = {
        "standard": "HIPAA PHI Protection (45 CFR 164.502)",
        "protection_level": "strong" if strong_protection else ("adequate" if adequate_protection else "insufficient"),
        "phi_attributes_detected": phi_attrs,
        "other_sensitive_attributes": non_phi_attrs,
        "compliant": adequate_protection,
        "evidence": {
            "l_value": l_value,
            "diversity_type": diversity_type,
            "total_sensitive_attributes": len(protected_attrs),
        },
    }

    if strong_protection:
        assessment["description"] = (
            f"Strong PHI protection: {diversity_type} l-diversity with l={l_value} "
            f"provides robust attribute disclosure protection for {len(protected_attrs)} attribute(s)."
        )
    elif adequate_protection:
        assessment["description"] = (
            f"Adequate PHI protection with l={l_value}. "
            "Consider entropy or recursive diversity for stronger guarantees."
        )
    else:
        assessment["description"] = (
            "Insufficient PHI protection. L-diversity threshold not met."
        )
        assessment["recommendations"] = [
            f"Increase l-value from {l_value} to at least 3.",
            "Use entropy l-diversity for health data to prevent skewness attacks.",
            "Ensure PHI columns are listed as sensitive attributes in the configuration.",
        ]

    return assessment


def assess_ccpa_deidentification(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess CCPA de-identification compliance.

    Under CCPA (Cal. Civ. Code 1798.140(h)), de-identified information cannot
    reasonably identify a consumer. L-diversity strengthens this guarantee
    by preventing attribute disclosure even when quasi-identifiers are known.
    """
    compliance_status = report.get("compliance_status", "Unknown")
    l_value = report.get("l_value", 0)
    at_risk_pct = report.get("at_risk_percentage", 100.0)

    reasonable_linkability = at_risk_pct < 10 and l_value >= 2
    technical_safeguards = compliance_status == "Compliant"

    assessment = {
        "standard": "CCPA De-identification (Cal. Civ. Code 1798.140(h))",
        "compliant": reasonable_linkability and technical_safeguards,
        "criteria": {
            "reasonable_linkability": {
                "met": reasonable_linkability,
                "description": (
                    "Information cannot reasonably identify, relate to, describe, "
                    "or be linked to a particular consumer or household."
                ),
                "at_risk_percentage": at_risk_pct,
                "threshold": 10.0,
            },
            "technical_safeguards": {
                "met": technical_safeguards,
                "description": (
                    "Business has implemented technical safeguards that prohibit "
                    "re-identification of the consumer."
                ),
                "l_value": l_value,
                "compliance_status": compliance_status,
            },
            "administrative_safeguards": {
                "met": None,
                "description": (
                    "Business processes are in place to prevent re-identification. "
                    "This must be verified through organizational policy review."
                ),
            },
        },
    }

    if assessment["compliant"]:
        assessment["description"] = (
            "Technical de-identification criteria under CCPA are met. "
            "Verify administrative safeguards separately."
        )
    else:
        assessment["description"] = (
            "CCPA de-identification criteria not fully met."
        )
        recommendations = []
        if not reasonable_linkability:
            recommendations.append(
                f"Reduce at-risk records from {at_risk_pct:.1f}% to below 10%."
            )
        if not technical_safeguards:
            recommendations.append(
                f"Ensure all groups meet l-diversity threshold (l={l_value})."
            )
        recommendations.append(
            "Establish organizational policies prohibiting re-identification attempts."
        )
        assessment["recommendations"] = recommendations

    return assessment
