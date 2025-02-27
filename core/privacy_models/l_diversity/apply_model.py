"""
PAMOLA.CORE - L-Diversity Model Application Module

Comprehensive model application strategies for l-diversity anonymization
with advanced features for flexible data protection.

Key Features:
- Multiple anonymization strategies
- Centralized results management
- Adaptive anonymization techniques
- Detailed provenance tracking

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from core.utils.group_processing import (
    validate_anonymity_inputs,
    optimize_memory_usage
)
from core.config import L_DIVERSITY_DEFAULTS
from core.privacy_models.l_diversity.calculation import LDiversityCalculator


class AnonymizationStrategy:
    """
    Abstract base class for anonymization strategies

    Provides a flexible framework for different anonymization approaches
    """

    def __init__(self, processor: LDiversityCalculator):
        """
        Initialize strategy with l-diversity processor

        Parameters:
        -----------
        processor : LDiversityCalculator
            L-Diversity calculation processor
        """
        self.processor = processor
        self.logger = logging.getLogger(__name__)

    def apply(
            self,
            data: pd.DataFrame,
            non_diverse_groups: List[Tuple],
            quasi_identifiers: List[str],
            sensitive_attributes: List[str]
    ) -> pd.DataFrame:
        """
        Abstract method for applying anonymization strategy

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        non_diverse_groups : List[Tuple]
            Groups identified as non-diverse
        quasi_identifiers : List[str]
            Quasi-identifier columns
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        pd.DataFrame
            Anonymized dataset
        """
        raise NotImplementedError("Subclasses must implement apply method")


class SuppressionStrategy(AnonymizationStrategy):
    """
    Suppression anonymization strategy

    Removes entire groups that do not meet l-diversity requirements
    """

    def apply(
            self,
            data: pd.DataFrame,
            non_diverse_groups: List[Tuple],
            quasi_identifiers: List[str],
            sensitive_attributes: List[str]
    ) -> pd.DataFrame:
        """
        Apply suppression to non-diverse groups

        Removes entire rows belonging to non-diverse groups
        """
        result = data.copy()

        # Create boolean mask for non-diverse groups
        suppression_mask = result.apply(
            lambda row: tuple(row[qi] for qi in quasi_identifiers) in non_diverse_groups,
            axis=1
        )

        # Remove non-diverse groups
        result = result[~suppression_mask]

        return result


class FullMaskingStrategy(AnonymizationStrategy):
    """
    Full masking anonymization strategy

    Replaces entire sensitive attribute values in non-diverse groups
    """

    def apply(
            self,
            data: pd.DataFrame,
            non_diverse_groups: List[Tuple],
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            mask_value: str = "MASKED"
    ) -> pd.DataFrame:
        """
        Apply full masking to non-diverse groups

        Replaces all sensitive attribute values in non-diverse groups
        """
        result = data.copy()

        for group in non_diverse_groups:
            # Create mask for specific group
            group_mask = np.all(
                [result[qi] == group_val for qi, group_val in zip(quasi_identifiers, group)],
                axis=0
            )

            # Mask all sensitive attributes
            for col in sensitive_attributes:
                result.loc[group_mask, col] = mask_value

        return result


class PartialMaskingStrategy(AnonymizationStrategy):
    """
    Partial masking anonymization strategy

    Allows flexible masking of sensitive attributes
    """

    def apply(
            self,
            data: pd.DataFrame,
            non_diverse_groups: List[Tuple],
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            mask_percentage: float = 0.5,
            mask_value: str = "MASKED"
    ) -> pd.DataFrame:
        """
        Apply partial masking to non-diverse groups

        Masks a specified percentage of sensitive attribute values
        """
        result = data.copy()

        for group in non_diverse_groups:
            # Create mask for specific group
            group_mask = np.all(
                [result[qi] == group_val for qi, group_val in zip(quasi_identifiers, group)],
                axis=0
            )

            # Get indices of rows in the group
            group_indices = result.index[group_mask]

            for col in sensitive_attributes:
                # Determine number of values to mask
                num_mask = int(len(group_indices) * mask_percentage)

                # Randomly select indices to mask
                mask_indices = np.random.choice(
                    group_indices,
                    size=num_mask,
                    replace=False
                )

                # Apply masking
                result.loc[mask_indices, col] = mask_value

        return result


class LDiversityModelApplicator:
    """
    Advanced L-Diversity Model Applicator

    Provides comprehensive anonymization with multiple strategies
    and centralized result management
    """

    def __init__(
            self,
            processor: Optional[LDiversityCalculator] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize L-Diversity Model Applicator

        Parameters:
        -----------
        processor : LDiversityCalculator, optional
            Pre-configured l-diversity processor
        config : dict, optional
            Configuration override for anonymization
        """
        # Use or create processor ensuring centralized results
        self.processor = processor or LDiversityCalculator()

        # Configuration setup
        self.config = dict(L_DIVERSITY_DEFAULTS)
        if config:
            self.config.update(config)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize anonymization strategies
        self.strategies = {
            'suppression': SuppressionStrategy(self.processor),
            'full_masking': FullMaskingStrategy(self.processor),
            'partial_masking': PartialMaskingStrategy(self.processor)
        }

    def apply_model(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            strategy: str = 'suppression',
            **kwargs
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Apply l-diversity anonymization with flexible strategies

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset to anonymize
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        strategy : str, optional
            Anonymization strategy to use
        **kwargs : dict
            Additional anonymization parameters

        Returns:
        --------
        Anonymized dataset with optional metadata
        """
        # Validate inputs
        validate_anonymity_inputs(data, quasi_identifiers, self.processor.k)

        # Optimize memory if requested
        if kwargs.get('optimize_memory', self.config.get('optimize_memory', False)):
            data = optimize_memory_usage(data, quasi_identifiers + sensitive_attributes)

        # Retrieve or calculate group diversity from centralized storage
        try:
            group_diversity = self.processor.calculate_group_diversity(
                data,
                quasi_identifiers,
                sensitive_attributes
            )
        except Exception as e:
            self.logger.error(f"Group diversity calculation error: {e}")
            raise

        # Identify non-diverse groups
        non_diverse_groups = self._identify_non_diverse_groups(
            group_diversity,
            sensitive_attributes
        )

        # Select and apply anonymization strategy
        strategy_name = strategy.lower()
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown anonymization strategy: {strategy}")

        # Apply selected strategy
        result = self.strategies[strategy_name].apply(
            data,
            non_diverse_groups,
            quasi_identifiers,
            sensitive_attributes,
            **kwargs
        )

        # Prepare and return results
        return self._prepare_anonymization_result(
            data,
            result,
            non_diverse_groups,
            strategy_name,
            **kwargs
        )

    def _identify_non_diverse_groups(
            self,
            group_diversity: pd.DataFrame,
            sensitive_attributes: List[str]
    ) -> List[Tuple]:
        """
        Identify groups that do not meet l-diversity requirements

        Utilizes processor's adaptive l-levels configuration
        """
        non_diverse_groups = []

        for _, row in group_diversity.iterrows():
            # Reconstruct group key
            group_key = tuple(row[qi] for qi in self.processor.quasi_identifiers)

            # Get adaptive l-level for this group
            adaptive_l = self.processor._get_adaptive_l(group_key)

            for sa in sensitive_attributes:
                if self.processor.diversity_type == "distinct":
                    if row.get(f"{sa}_distinct", 0) < adaptive_l:
                        non_diverse_groups.append(group_key)
                        break
                elif self.processor.diversity_type == "entropy":
                    if row.get(f"{sa}_entropy", 0) < np.log(adaptive_l):
                        non_diverse_groups.append(group_key)
                        break
                elif self.processor.diversity_type == "recursive":
                    if not row.get(f"{sa}_recursive", False):
                        non_diverse_groups.append(group_key)
                        break

        return non_diverse_groups

    def _prepare_anonymization_result(
            self,
            original_data: pd.DataFrame,
            anonymized_data: pd.DataFrame,
            non_diverse_groups: List[Tuple],
            strategy: str,
            **kwargs
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare comprehensive anonymization result

        Includes metadata and optional full result
        """
        # Prepare result metadata
        result_metadata = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "processor_config": {
                    "diversity_type": self.processor.diversity_type,
                    "l_value": self.processor.l,
                    "adaptive_l_levels": bool(self.processor.adaptive_l)
                }
            },
            "dataset_metrics": {
                "original_records": len(original_data),
                "anonymized_records": len(anonymized_data),
                "records_transformed": len(original_data) - len(anonymized_data),
                "non_diverse_groups": len(non_diverse_groups)
            },
            "anonymization_details": {
                "strategy": strategy,
                "parameters": {k: v for k, v in kwargs.items() if k in ['mask_value', 'mask_percentage']}
            }
        }

        # Return full result based on configuration
        return_full_result = kwargs.get('return_full_result', False)

        if return_full_result:
            result_metadata['anonymized_data'] = anonymized_data
            return result_metadata

        return anonymized_data


# Utility function for quick model application
def apply_l_diversity(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        strategy: str = 'suppression',
        **kwargs
) -> pd.DataFrame:
    """
    Convenient utility function for applying l-diversity

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset to anonymize
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attributes : List[str]
        Sensitive attribute columns
    strategy : str, optional
        Anonymization strategy to use
    **kwargs : dict
        Additional anonymization parameters

    Returns:
    --------
    pd.DataFrame
        Anonymized dataset
    """
    applicator = LDiversityModelApplicator()
    return applicator.apply_model(
        data,
        quasi_identifiers,
        sensitive_attributes,
        strategy,
        **kwargs
    )