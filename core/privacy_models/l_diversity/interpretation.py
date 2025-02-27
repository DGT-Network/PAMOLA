"""
PAMOLA.CORE - L-Diversity Risk Interpretation
---------------------------------------------
This module provides interpretation functionality for l-diversity risk metrics.
It translates numeric risk values into human-readable assessments and
contextualizes them based on various privacy requirements.

Key Features:
- Risk interpretation for different attack models
- Domain-specific risk categorization
- Regulatory compliance assessment
- Risk interpretation optimization

This module is used by the main privacy.py module but can also be used
independently for custom risk interpretation needs.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class RiskInterpreter:
    """
    Interprets privacy risk metrics for l-diversity

    Provides context-aware interpretation of various risk metrics
    with support for different regulatory frameworks and data domains.
    """

    def __init__(self,
                 domain: str = "general",
                 regulation: str = None,
                 custom_thresholds: Dict[str, List[float]] = None):
        """
        Initialize Risk Interpreter

        Parameters:
        -----------
        domain : str, optional
            Data domain for context-specific interpretation (default: "general")
            Options: "general", "healthcare", "finance", "education", "telecom"
        regulation : str, optional
            Regulatory framework for compliance assessment
            Options: "GDPR", "HIPAA", "CCPA", etc.
        custom_thresholds : Dict[str, List[float]], optional
            Custom risk thresholds for different risk types
        """
        self.domain = domain
        self.regulation = regulation
        self.logger = logging.getLogger(__name__)

        # Default risk thresholds (very low, low, moderate, high, very high)
        self.default_thresholds = {
            'general': [5.0, 15.0, 30.0, 50.0],
            'healthcare': [3.0, 10.0, 20.0, 40.0],
            'finance': [2.0, 8.0, 20.0, 35.0],
            'education': [5.0, 15.0, 30.0, 50.0],
            'telecom': [4.0, 12.0, 25.0, 45.0]
        }

        # Use custom thresholds if provided
        self.custom_thresholds = custom_thresholds or {}

        # Regulatory minimum l-values
        self.regulatory_l_values = {
            'GDPR': 3,
            'HIPAA': 4,
            'CCPA': 3,
            'PIPEDA': 3,
            'APPI': 3
        }

        # Risk terms for different levels
        self.risk_terms = [
            "Very Low Risk",
            "Low Risk",
            "Moderate Risk",
            "High Risk",
            "Very High Risk"
        ]

        # Risk descriptions for different levels
        self.risk_descriptions = [
            "Excellent privacy protection",
            "Good privacy protection",
            "Acceptable for many scenarios",
            "Caution recommended when sharing data",
            "Significant privacy concerns"
        ]

    def interpret_risk(self, risk_value: float, risk_type: str = "general") -> str:
        """
        Provide human-readable interpretation of risk values

        Parameters:
        -----------
        risk_value : float
            Risk percentage (0-100)
        risk_type : str, optional
            Type of risk being interpreted (default: "general")

        Returns:
        --------
        str
            Human-readable risk interpretation
        """
        # Get appropriate thresholds based on domain and risk type
        thresholds = self._get_thresholds(risk_type)

        # Determine risk level based on thresholds
        if risk_value < thresholds[0]:
            level = 0  # Very low
        elif risk_value < thresholds[1]:
            level = 1  # Low
        elif risk_value < thresholds[2]:
            level = 2  # Moderate
        elif risk_value < thresholds[3]:
            level = 3  # High
        else:
            level = 4  # Very high

        # Return formatted interpretation
        return f"{self.risk_terms[level]} - {self.risk_descriptions[level]}"

    def _get_thresholds(self, risk_type: str) -> List[float]:
        """
        Get appropriate thresholds for a risk type and domain

        Parameters:
        -----------
        risk_type : str
            Type of risk being interpreted

        Returns:
        --------
        List[float]
            List of threshold values
        """
        # Check if custom thresholds exist for this risk type
        if risk_type in self.custom_thresholds:
            return self.custom_thresholds[risk_type]

        # Fall back to domain-specific thresholds
        if self.domain in self.default_thresholds:
            return self.default_thresholds[self.domain]

        # Default to general thresholds
        return self.default_thresholds['general']

    def interpret_attack_models(self, risk_metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Interpret risk values for different attack models

        Parameters:
        -----------
        risk_metrics : Dict[str, float]
            Dictionary with risk values for different attack models

        Returns:
        --------
        Dict[str, str]
            Dictionary with human-readable interpretations
        """
        interpretations = {}

        # Interpret each attack model
        for model, risk in risk_metrics.items():
            model_type = model.replace('_risk', '')
            interpretations[model] = self.interpret_risk(risk, model_type)

        return interpretations

    def interpret_compliance(self,
                             min_diversity: float,
                             diversity_type: str = "distinct",
                             regulation: str = None) -> Dict[str, Any]:
        """
        Interpret l-diversity compliance for regulatory frameworks

        Parameters:
        -----------
        min_diversity : float
            Minimum diversity value across the dataset
        diversity_type : str, optional
            Type of diversity being assessed (default: "distinct")
        regulation : str, optional
            Specific regulation to check (overrides instance regulation)

        Returns:
        --------
        Dict[str, Any]
            Compliance assessment
        """
        # Use provided regulation or instance regulation
        reg = regulation or self.regulation

        # If no regulation specified, evaluate general compliance
        if not reg:
            if diversity_type == "entropy":
                # For entropy, convert to effective l
                effective_l = np.exp(min_diversity) if min_diversity > 0 else 1
                compliant = effective_l >= 3  # General minimum requirement
                target = 3
            else:
                # For distinct and recursive
                compliant = min_diversity >= 3  # General minimum requirement
                target = 3

            return {
                'compliant': compliant,
                'target_l': target,
                'achieved_l': min_diversity if diversity_type != "entropy" else effective_l,
                'regulation': "general",
                'recommendation': self._get_compliance_recommendation(
                    compliant, min_diversity, diversity_type, 3
                )
            }

        # Get regulatory l-value requirement
        target_l = self.regulatory_l_values.get(reg, 3)

        # Evaluate compliance based on diversity type
        if diversity_type == "entropy":
            # For entropy, compliance is based on exp(entropy) >= l
            effective_l = np.exp(min_diversity) if min_diversity > 0 else 1
            compliant = effective_l >= target_l
            achieved_l = effective_l
        else:
            # For distinct and recursive
            compliant = min_diversity >= target_l
            achieved_l = min_diversity

        # Return compliance assessment
        return {
            'compliant': compliant,
            'target_l': target_l,
            'achieved_l': achieved_l,
            'regulation': reg,
            'recommendation': self._get_compliance_recommendation(
                compliant, min_diversity, diversity_type, target_l
            )
        }

    def _get_compliance_recommendation(self,
                                       compliant: bool,
                                       min_diversity: float,
                                       diversity_type: str,
                                       target_l: int) -> str:
        """
        Get recommendation for improving compliance

        Parameters:
        -----------
        compliant : bool
            Whether the dataset is compliant
        min_diversity : float
            Minimum diversity value
        diversity_type : str
            Type of diversity being assessed
        target_l : int
            Target l-value for compliance

        Returns:
        --------
        str
            Recommendation for compliance
        """
        if compliant:
            return "Dataset meets compliance requirements"

        # For non-compliant datasets, provide specific recommendations
        if diversity_type == "entropy":
            effective_l = np.exp(min_diversity) if min_diversity > 0 else 1
            gap = target_l - effective_l

            if gap <= 1:
                return "Minor improvement needed: Consider adding more variety to sensitive values in low-diversity groups"
            else:
                return "Significant improvement needed: Consider generalization or suppression of quasi-identifiers"
        else:
            gap = target_l - min_diversity

            if gap <= 1:
                return "Minor improvement needed: Add variety to low-diversity groups"
            else:
                return "Significant restructuring needed: Consider reducing quasi-identifier granularity"

    def interpret_diversity(self,
                            diversity_value: float,
                            diversity_type: str = "distinct",
                            l_threshold: int = 3,
                            c_value: float = 1.0) -> str:
        """
        Interpret a diversity value

        Parameters:
        -----------
        diversity_value : float
            L-diversity value to interpret
        diversity_type : str, optional
            Type of diversity (default: "distinct")
        l_threshold : int, optional
            L-threshold for comparison (default: 3)
        c_value : float, optional
            C parameter for recursive diversity (default: 1.0)

        Returns:
        --------
        str
            Human-readable interpretation
        """
        if diversity_type == "entropy":
            # For entropy l-diversity, interpret as effective number of classes
            effective_l = np.exp(diversity_value) if diversity_value > 0 else 1

            if effective_l < 2:
                return f"Entropy l-Diversity: effective l={effective_l:.2f} - Very low diversity, high attribute disclosure risk"
            elif effective_l < l_threshold:
                return f"Entropy l-Diversity: effective l={effective_l:.2f} - Insufficient diversity, below threshold of {l_threshold}"
            elif effective_l < l_threshold * 1.5:
                return f"Entropy l-Diversity: effective l={effective_l:.2f} - Acceptable diversity, meets minimum requirements"
            elif effective_l < l_threshold * 2:
                return f"Entropy l-Diversity: effective l={effective_l:.2f} - Good diversity, low attribute disclosure risk"
            else:
                return f"Entropy l-Diversity: effective l={effective_l:.2f} - Excellent diversity, very low disclosure risk"

        elif diversity_type == "recursive":
            # For recursive l-diversity, interpretation depends on compliance
            if diversity_value < 2:
                return f"Recursive l-Diversity: l={diversity_value} - No diversity, high attribute disclosure risk"
            elif diversity_value < l_threshold:
                return f"Recursive l-Diversity: l={diversity_value} - Insufficient (c,l)-diversity, below threshold of {l_threshold}"
            else:
                return f"Recursive l-Diversity: l={diversity_value} - Satisfies ({c_value},{l_threshold})-diversity"

        else:  # Default - distinct l-diversity
            if diversity_value < 2:
                return f"Distinct l-Diversity: l={diversity_value} - No diversity, high attribute disclosure risk"
            elif diversity_value < l_threshold:
                return f"Distinct l-Diversity: l={diversity_value} - Insufficient diversity, below threshold of {l_threshold}"
            elif diversity_value < 5:
                return f"Distinct l-Diversity: l={diversity_value} - Acceptable diversity for many scenarios"
            elif diversity_value < 10:
                return f"Distinct l-Diversity: l={diversity_value} - Good diversity, low attribute disclosure risk"
            else:
                return f"Distinct l-Diversity: l={diversity_value} - Excellent diversity, very low attribute disclosure risk"

    def get_domain_specific_assessment(self,
                                       risk_metrics: Dict[str, Any],
                                       domain: str = None) -> Dict[str, Any]:
        """
        Get domain-specific risk assessment

        Parameters:
        -----------
        risk_metrics : Dict[str, Any]
            Dictionary with risk values and metrics
        domain : str, optional
            Data domain (overrides instance domain)

        Returns:
        --------
        Dict[str, Any]
            Domain-specific risk assessment
        """
        # Use provided domain or instance domain
        dom = domain or self.domain

        # Extract key risk metrics
        prosecutor_risk = risk_metrics.get('prosecutor_risk',
                                           risk_metrics.get('attack_models', {}).get('prosecutor_risk', 100.0))
        min_diversity = risk_metrics.get('min_diversity',
                                         risk_metrics.get('overall_risk', {}).get('min_diversity', 0))
        diversity_type = risk_metrics.get('diversity_type',
                                          risk_metrics.get('overall_risk', {}).get('diversity_type', 'distinct'))

        # Default assessment
        assessment = {
            'domain': dom,
            'risk_level': self.interpret_risk(prosecutor_risk),
            'recommendation': "General privacy assessment: "
        }

        # Domain-specific assessments and recommendations
        if dom == "healthcare":
            # Healthcare data has strict privacy requirements
            if prosecutor_risk > 10:
                assessment[
                    'recommendation'] += "High risk for healthcare data. Further anonymization strongly recommended before sharing."
            elif min_diversity < 4:
                assessment[
                    'recommendation'] += "Consider increasing diversity for sensitive medical attributes to meet HIPAA-like standards."
            else:
                assessment['recommendation'] += "Acceptable for internal analytics. Review before external sharing."

        elif dom == "finance":
            # Financial data requires high privacy
            if prosecutor_risk > 8:
                assessment[
                    'recommendation'] += "Risk level too high for financial data. Additional anonymization required."
            elif min_diversity < 3:
                assessment[
                    'recommendation'] += "Diversity too low for financial attributes. Consider increasing diversity."
            else:
                assessment['recommendation'] += "Suitable for controlled sharing with proper agreements."

        elif dom == "education":
            # Education data (often covered by FERPA in US)
            if prosecutor_risk > 15:
                assessment[
                    'recommendation'] += "Risk exceeds typical education data standards. Further anonymization recommended."
            elif min_diversity < 3:
                assessment['recommendation'] += "Consider increasing diversity of sensitive educational attributes."
            else:
                assessment['recommendation'] += "Acceptable for research and internal analytics."

        elif dom == "telecom":
            # Telecom data (location, usage patterns)
            if prosecutor_risk > 12:
                assessment[
                    'recommendation'] += "Risk level concerning for telecom data. Consider reducing quasi-identifier precision."
            elif min_diversity < 3:
                assessment['recommendation'] += "Increase diversity of sensitive telecom attributes."
            else:
                assessment['recommendation'] += "Suitable for internal analytics and aggregated reporting."

        else:
            # General domain
            if prosecutor_risk > 15:
                assessment['recommendation'] += "Risk level is concerning. Consider additional anonymization."
            elif min_diversity < 3:
                assessment['recommendation'] += "Increase diversity of sensitive attributes."
            else:
                assessment['recommendation'] += "Acceptable for many use cases with proper controls."

        return assessment


# Utility functions

def interpret_risk_value(risk_value: float, domain: str = "general") -> str:
    """
    Interpret a risk value with default settings

    Parameters:
    -----------
    risk_value : float
        Risk percentage (0-100)
    domain : str, optional
        Data domain for context-specific interpretation (default: "general")

    Returns:
    --------
    str
        Human-readable risk interpretation
    """
    interpreter = RiskInterpreter(domain=domain)
    return interpreter.interpret_risk(risk_value)


def interpret_risk_metrics(risk_metrics: Dict[str, Any],
                           domain: str = "general",
                           regulation: str = None) -> Dict[str, Any]:
    """
    Comprehensive interpretation of risk metrics

    Parameters:
    -----------
    risk_metrics : Dict[str, Any]
        Dictionary with risk values and metrics
    domain : str, optional
        Data domain for context-specific interpretation (default: "general")
    regulation : str, optional
        Regulatory framework for compliance assessment

    Returns:
    --------
    Dict[str, Any]
        Comprehensive risk interpretation
    """
    interpreter = RiskInterpreter(domain=domain, regulation=regulation)

    # Extract key metrics
    attack_models = risk_metrics.get('attack_models', {})
    overall_risk = risk_metrics.get('overall_risk', {})

    min_diversity = overall_risk.get('min_diversity', 0)
    diversity_type = overall_risk.get('diversity_type', 'distinct')
    l_threshold = overall_risk.get('l_threshold', 3)
    c_value = overall_risk.get('c_value', 1.0)

    # Create interpretations
    interpretations = {
        'attack_models': interpreter.interpret_attack_models(attack_models),
        'diversity_assessment': interpreter.interpret_diversity(
            min_diversity, diversity_type, l_threshold, c_value
        ),
        'compliance': interpreter.interpret_compliance(
            min_diversity, diversity_type, regulation
        ),
        'domain_assessment': interpreter.get_domain_specific_assessment(risk_metrics)
    }

    return interpretations


def get_regulatory_requirements(regulation: str) -> Dict[str, Any]:
    """
    Get l-diversity requirements for different privacy regulations

    Parameters:
    -----------
    regulation : str
        Regulation to get requirements for ('GDPR', 'HIPAA', 'CCPA', etc.)

    Returns:
    --------
    Dict[str, Any]
        Dictionary with recommended l-diversity parameters for the regulation
    """
    regulation = regulation.upper()

    # Default requirements
    default = {
        "l_threshold": 3,
        "diversity_type": "distinct",
        "c_value": 1.0,
        "description": "General recommendation for l-diversity"
    }

    # Regulation-specific requirements
    regulations = {
        "GDPR": {
            "l_threshold": 3,
            "diversity_type": "distinct",
            "c_value": 1.0,
            "description": "General Data Protection Regulation (EU)"
        },
        "HIPAA": {
            "l_threshold": 4,
            "diversity_type": "recursive",
            "c_value": 0.5,
            "description": "Health Insurance Portability and Accountability Act (US)"
        },
        "CCPA": {
            "l_threshold": 3,
            "diversity_type": "entropy",
            "c_value": 1.0,
            "description": "California Consumer Privacy Act (US)"
        },
        "PIPEDA": {
            "l_threshold": 3,
            "diversity_type": "distinct",
            "c_value": 1.0,
            "description": "Personal Information Protection and Electronic Documents Act (Canada)"
        },
        "APPI": {
            "l_threshold": 3,
            "diversity_type": "distinct",
            "c_value": 1.0,
            "description": "Act on the Protection of Personal Information (Japan)"
        }
    }

    return regulations.get(regulation, default)