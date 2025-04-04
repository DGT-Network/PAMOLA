"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Full Masking Processor
------------------------------
This module provides methods for fully masking sensitive attributes
to enhance anonymization. It extends the BaseMaskingProcessor and implements
techniques such as replacing values with standard placeholders,
randomized masks, and format-preserving masking.

Full masking completely removes the original information while
preserving the data structure.

Common approaches include:
- **Fixed Masking**: Replacing values with `"MASKED"` or `"REDACTED"`.
- **Symbol Masking**: Replacing values with predefined symbols (`XXXXXX`, `******`).
- **Format-Preserving Masking**: Retaining the original format while hiding content
  (e.g., `+1-XXX-XXX-XXXX` for phone numbers).

NOTE: This module requires `pandas` and `re` (regular expressions) as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import re
import pandas as pd
from abc import ABC
from typing import List
from pamola_core.anonymization.masking.base import BaseMaskingProcessor

class FullMaskingProcessor(BaseMaskingProcessor, ABC):
    """
    Full Masking Processor for anonymizing sensitive attributes.
    This class extends BaseMaskingProcessor and provides techniques
    for fully masking data while preserving format when required.

    Methods:
    --------
    - apply_fixed_mask(): Replaces values with a predefined mask (`"MASKED"`).
    - apply_symbol_mask(): Replaces values with symbols (`"XXXXXX"`, `"*****"`).
    - apply_format_preserving_mask(): Preserves the format while hiding content.
    """

    def __init__(
            self,
            target_fields: List[str],
            method: str = 'fixed',
            fixed_mask: str = 'MASKED',
            mask_char: str = 'X',
            format_patterns: dict | None = None,
            format_templates: dict | None = None,
            conditional_masking: bool = False,
            masking_conditions: List[dict] | None = None,
            field_specific_masks: dict | None = None,
            apply_to_nulls: bool = False,
            preserve_metadata: bool = False,
            track_masking: bool = False,
            masking_log: str = None
    ):
        """
        Initializes the full masking processor.

        Parameters:
        -----------
        target_fields : list
            Fields to apply masking to.
        method : str, default 'fixed'
            Masking method to use.
        fixed_mask : str, default 'MASKED'
            Replacement text for fixed masking.
        mask_char : str, default 'X'
            Character used for symbol masking.
        format_patterns : dict, optional
            Custom format patterns to detect.
        format_templates : dict, optional
            Output format templates.
        conditional_masking : bool, default False
            Whether to use conditions.
        masking_conditions : list, optional
            Conditions that trigger masking.
        field_specific_masks : dict, optional
            Masks specific to fields.
        apply_to_nulls : bool, default False
            Whether to mask null values.
        preserve_metadata : bool, default False
            Whether to track masked fields.
        track_masking : bool, default False
            Whether to track masking.
        masking_log : str, optional
            Path to save masking log.
        """
        if format_patterns is None:
            format_patterns = {}

        if format_templates is None:
            format_templates = {}

        if masking_conditions is None:
            masking_conditions = []

        if field_specific_masks is None:
            field_specific_masks = {}

        super().__init__()

        self.target_fields = target_fields
        self.method = method
        self.fixed_mask = fixed_mask
        self.mask_char = mask_char
        self.format_patterns = format_patterns
        self.format_templates = format_templates
        self.conditional_masking = conditional_masking
        self.masking_conditions = masking_conditions
        self.field_specific_masks = field_specific_masks
        self.apply_to_nulls = apply_to_nulls
        self.preserve_metadata = preserve_metadata
        self.track_masking = track_masking
        self.masking_log = masking_log

    def mask(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Apply full masking to specified columns.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing sensitive data.
        kwargs : dict, optional
            Additional parameters for masking, such as mask character,
            format preservation settings, or regex patterns.
            - target_fields : list
                Fields to apply masking to.
            - method : str, default 'fixed'
                Masking method to use.
            - fixed_mask : str, default 'MASKED'
                Replacement text for fixed masking.
            - mask_char : str, default 'X'
                Character used for symbol masking.
            - format_patterns : dict, optional
                Custom format patterns to detect.
            - format_templates : dict, optional
                Output format templates.
            - conditional_masking : bool, default False
                Whether to use conditions.
            - masking_conditions : list, optional
                Conditions that trigger masking.
            - field_specific_masks : dict, optional
                Masks specific to fields.
            - apply_to_nulls : bool, default False
                Whether to mask null values.
            - preserve_metadata : bool, default False
                Whether to track masked fields.
            - track_masking : bool, default False
                Whether to track masking.
            - masking_log : str, optional
                Path to save masking log.

        Returns:
        --------
        pd.DataFrame
            The dataset with masked values.
        """
        self.get_custom_parameters(**kwargs)
        self.validate(df)

        if self.method == "fixed":
            return self.apply_fixed_mask(df)
        elif self.method == "symbolic":
            return self.apply_symbol_mask(df)
        elif self.method == "format_preserving":
            return self.apply_format_preserving_mask(df)
        else:
            raise ValueError(f"Invalid masking method: {self.method}")

    def get_custom_parameters(
            self,
            **kwargs
    ) -> None:
        """
        Get custom parameters from kwargs.

        Parameters:
        -----------
        kwargs : dict, optional
            Additional parameters for masking, such as mask character,
            format preservation settings, or regex patterns.
            - target_fields : list
                Fields to apply masking to.
            - method : str, default 'fixed'
                Masking method to use.
            - fixed_mask : str, default 'MASKED'
                Replacement text for fixed masking.
            - mask_char : str, default 'X'
                Character used for symbol masking.
            - format_patterns : dict, optional
                Custom format patterns to detect.
            - format_templates : dict, optional
                Output format templates.
            - conditional_masking : bool, default False
                Whether to use conditions.
            - masking_conditions : list, optional
                Conditions that trigger masking.
            - field_specific_masks : dict, optional
                Masks specific to fields.
            - apply_to_nulls : bool, default False
                Whether to mask null values.
            - preserve_metadata : bool, default False
                Whether to track masked fields.
            - track_masking : bool, default False
                Whether to track masking.
            - masking_log : str, optional
                Path to save masking log.
        """
        self.target_fields = kwargs.get('target_fields', self.target_fields)
        self.method = kwargs.get('method', self.method)
        self.fixed_mask = kwargs.get('fixed_mask', self.fixed_mask)
        self.mask_char = kwargs.get('mask_char', self.mask_char)
        self.format_patterns = kwargs.get('format_patterns', self.format_patterns)
        self.format_templates = kwargs.get('format_templates', self.format_templates)
        self.conditional_masking = kwargs.get('conditional_masking', self.conditional_masking)
        self.masking_conditions = kwargs.get('masking_conditions', self.masking_conditions)
        self.field_specific_masks = kwargs.get('field_specific_masks', self.field_specific_masks)
        self.apply_to_nulls = kwargs.get('apply_to_nulls', self.apply_to_nulls)
        self.preserve_metadata = kwargs.get('preserve_metadata', self.preserve_metadata)
        self.track_masking = kwargs.get('track_masking', self.track_masking)
        self.masking_log = kwargs.get('masking_log', self.masking_log)

    def validate(
            self,
            df: pd.DataFrame
    ) -> None:
        """
        Validate data.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        """
        if not self.target_fields:
            raise ValueError('target_fields must have value!')

        if df.empty:
            raise ValueError('DataFrame is empty!')

    def apply_fixed_mask(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replace all values in specified columns with a fixed mask.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing sensitive data.

        Returns:
        --------
        pd.DataFrame
            The dataset with fully masked values.
        """
        df_not_processed, df_processed = self.split_df_by_condition(df)

        for target_field in self.target_fields:
            if target_field in self.field_specific_masks:
                df_processed[target_field] = self.field_specific_masks[target_field]
            else:
                df_processed[target_field] = self.fixed_mask

        df_anonymization = pd.concat([df_not_processed, df_processed]).sort_index()

        return df_anonymization

    def apply_symbol_mask(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replace all values in specified columns with a symbolic mask.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing sensitive data.

        Returns:
        --------
        pd.DataFrame
            The dataset with symbolically masked values.
        """
        df_not_processed, df_processed = self.split_df_by_condition(df)

        for target_field in self.target_fields:
            if target_field in self.field_specific_masks:
                df_processed[target_field] = self.field_specific_masks[target_field]
            else:
                df_processed[target_field] = (df_processed[target_field].astype(str)
                                              .map(lambda x: self.mask_char * len(x)))

        df_anonymization = pd.concat([df_not_processed, df_processed]).sort_index()

        return df_anonymization

    def apply_format_preserving_mask(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Preserve original format while applying masking.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing sensitive data.

        Returns:
        --------
        pd.DataFrame
            The dataset with format-preserving masked values.
        """
        def format_preserving_mask(value, pattern, template):
            if isinstance(value, str):
                if pattern and not re.match(pattern, value):
                    return value
                else:
                    if template:
                        return template
                    else:
                        return re.sub(r'[a-zA-Z0-9]', self.mask_char, value)
            else:
                return self.fixed_mask

        df_not_processed, df_processed = self.split_df_by_condition(df)

        for target_field in self.target_fields:
            if target_field in self.field_specific_masks:
                df_processed[target_field] = self.field_specific_masks[target_field]
            else:
                format_pattern = self.format_patterns[target_field]\
                    if target_field in self.format_patterns else ''
                format_template = self.format_templates[target_field]\
                    if target_field in self.format_templates else ''

                df_processed[target_field] = (df_processed[target_field].astype(str)
                                              .map(lambda x: format_preserving_mask(
                                                            x, format_pattern, format_template)))

        df_anonymization = pd.concat([df_not_processed, df_processed]).sort_index()

        return df_anonymization

    def split_df_by_condition(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by condition.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing sensitive data.

        Returns:
        --------
        Tuple[pd.DataFrame | None, pd.DataFrame | None]
            The dataset with format-preserving masked values.
        """
        df_processed = pd.DataFrame()
        df_not_processed = pd.DataFrame()

        if self.conditional_masking:
            list_condition = [
                ' '.join(value for (key, value) in condition.items())
                for condition in self.masking_conditions
            ]
            str_query = ' and '.join(list_condition)

            df_processed = df.query(expr=str_query)
        else:
            df_processed = df.copy(deep=True)

        if not self.apply_to_nulls:
            df_processed = df_processed[df_processed[self.target_fields].notnull().any(axis=1)]

        df_not_processed = df.drop(df_processed.index.values.tolist())

        return df_not_processed, df_processed
