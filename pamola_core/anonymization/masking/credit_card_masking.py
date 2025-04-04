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

Module: Credit Card Masking Processor
------------------------------
Operation specifically designed for masking credit card numbers in compliance
with PCI DSS and other payment card security standards.

NOTE: This module requires `pandas` and `re` (regular expressions) as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import re
import math
import pandas as pd
from abc import ABC
from typing import List
from pamola_core.anonymization.masking.base import BaseMaskingProcessor
from pamola_core.common.regex.patterns import Patterns

class CreditCardMaskingProcessor(BaseMaskingProcessor, ABC):
    """
    Credit Card Masking Processor specifically designed for masking credit card
    numbers in compliance with PCI DSS and other payment card security standards.
    """
    CONST_FILTER_REGEX = r'_card_type$|_validation_check$|_luhn_check$'
    CONST_CARD_TYPE = '_card_type'
    CONST_VALIDATION_CHECK = '_validation_check'
    CONST_LUHN_CHECK = '_luhn_check'

    def __init__(
            self,
            target_fields: List[str],
            method: str = 'pci_standard',
            mask_char: str = 'X',
            digits_to_show: int = 4,
            digits_position: str = 'end',
            preserve_issuer: bool = True,
            preserve_check_digit: bool = False,
            preserve_separator: bool = True,
            separator_char: str = '-',
            detect_card_type: bool = True,
            card_type_patterns: dict | None = None,
            validation_check: bool = True,
            luhn_check: bool = False,
            track_masking: bool = False,
            masking_log: str = None
    ):
        """
        Initializes the full masking processor.

        Parameters:
        -----------
        target_fields : list
            Fields containing credit card numbers.
        method : str, default 'pci_standard'
            Masking method to use.
        mask_char : str, default 'X'
            Character used for masking.
        digits_to_show : int, default 4
            Number of digits to preserve.
        digits_position : str, default 'end'
            Position of preserved digits.
        preserve_issuer : bool, default True
            Whether to preserve BIN (first 6).
        preserve_check_digit : bool, default False
            Whether to preserve check digit.
        preserve_separator : bool, default True
            Whether to keep digit separators.
        separator_char : str, default '-'
            Character used as separator.
        detect_card_type : bool, default True
            Whether to detect card type.
        card_type_patterns : dict, optional
            Custom card type detection patterns.
        validation_check : bool, default True
            Whether to validate card format.
        luhn_check : bool, default False
            Whether to validate card format.
        track_masking : bool, default False
            Whether to track masking.
        masking_log : str, optional
            Path to save masking log.
        """
        if card_type_patterns is None:
            card_type_patterns = {}

        super().__init__()

        self.target_fields = target_fields
        self.method = method
        self.mask_char = mask_char
        self.digits_to_show = digits_to_show
        self.digits_position = digits_position
        self.preserve_issuer = preserve_issuer
        self.preserve_check_digit = preserve_check_digit
        self.preserve_separator = preserve_separator
        self.separator_char = separator_char
        self.detect_card_type = detect_card_type
        self.card_type_patterns = card_type_patterns
        self.validation_check = validation_check
        self.luhn_check = luhn_check
        self.track_masking = track_masking
        self.masking_log = masking_log

    def mask(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Masking credit card numbers in compliance with PCI DSS and other
        payment card security standards.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing sensitive data.
        kwargs : dict, optional
            Additional parameters for customization.
            - target_fields : list
                Fields containing credit card numbers.
            - method : str, default 'pci_standard'
                Masking method to use.
            - mask_char : str, default 'X'
                Character used for masking.
            - digits_to_show : int, default 4
                Number of digits to preserve.
            - digits_position : str, default 'end'
                Position of preserved digits.
            - preserve_issuer : bool, default True
                Whether to preserve BIN (first 6).
            - preserve_check_digit : bool, default False
                Whether to preserve check digit.
            - preserve_separator : bool, default True
                Whether to keep digit separators.
            - separator_char : str, default '-'
                Character used as separator.
            - detect_card_type : bool, default True
                Whether to detect card type.
            - card_type_patterns : dict, optional
                Custom card type detection patterns.
            - validation_check : bool, default True
                Whether to validate card format.
            - luhn_check : bool, default False
                Whether to validate card format.
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

        df_processed = self.detect_and_check(df)

        if self.method == 'pci_standard':
            return self.apply_pci_standard(df_processed)
        elif self.method == 'issuer_preserved':
            return self.apply_issuer_preserved(df_processed)
        elif self.method == 'custom':
            return self.apply_custom(df_processed)
        else:
            raise ValueError(f'Invalid masking method: {self.method}')

    def get_custom_parameters(
            self,
            **kwargs
    ) -> None:
        """
        Get custom parameters from kwargs.

        Parameters:
        -----------
        kwargs : dict, optional
            Additional parameters for customization.
            - target_fields : list
                Fields containing credit card numbers.
            - method : str, default 'pci_standard'
                Masking method to use.
            - mask_char : str, default 'X'
                Character used for masking.
            - digits_to_show : int, default 4
                Number of digits to preserve.
            - digits_position : str, default 'end'
                Position of preserved digits.
            - preserve_issuer : bool, default True
                Whether to preserve BIN (first 6).
            - preserve_check_digit : bool, default False
                Whether to preserve check digit.
            - preserve_separator : bool, default True
                Whether to keep digit separators.
            - separator_char : str, default '-'
                Character used as separator.
            - detect_card_type : bool, default True
                Whether to detect card type.
            - card_type_patterns : dict, optional
                Custom card type detection patterns.
            - validation_check : bool, default True
                Whether to validate card format.
            - luhn_check : bool, default False
                Whether to validate card format.
            - track_masking : bool, default False
                Whether to track masking.
            - masking_log : str, optional
                Path to save masking log.
        """
        self.target_fields = kwargs.get('target_fields', self.target_fields)
        self.method = kwargs.get('method', self.method)
        self.mask_char = kwargs.get('mask_char', self.mask_char)
        self.digits_to_show = kwargs.get('digits_to_show', self.digits_to_show)
        self.digits_position = kwargs.get('digits_position', self.digits_position)
        self.preserve_issuer = kwargs.get('preserve_issuer', self.preserve_issuer)
        self.preserve_check_digit = kwargs.get('preserve_check_digit', self.preserve_check_digit)
        self.preserve_separator = kwargs.get('preserve_separator', self.preserve_separator)
        self.separator_char = kwargs.get('separator_char', self.separator_char)
        self.detect_card_type = kwargs.get('detect_card_type', self.detect_card_type)
        self.card_type_patterns = kwargs.get('card_type_patterns', self.card_type_patterns)
        self.validation_check = kwargs.get('validation_check', self.validation_check)
        self.luhn_check = kwargs.get('luhn_check', self.luhn_check)
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

    def apply_pci_standard(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Masking method pci standard.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data to be processed.

        Returns:
        --------
        pd.DataFrame
            Processed data.
        """
        def pci_standard(value):
            mask_value = ''

            is_valid = False
            pattern = Patterns.PCI_PATTERNS
            if re.match(pattern, value):
                is_valid = True

            if is_valid:
                separator_char = self.separator_char if self.preserve_separator else ''
                mask_value = re.sub(r'[^0-9]', separator_char, value)
                idx1 = 7 if len(mask_value) == 19 else 6
                idx2 = 15 if len(mask_value) == 19 else 12

                mask_value = ''.join((
                    mask_value[:idx1],
                    re.sub(r'[0-9]', self.mask_char, mask_value[idx1:idx2]),
                    mask_value[-4:]
                ))
            else:
                mask_value = value

            return mask_value

        for target_field in self.target_fields:
            df[target_field] = (df[target_field].astype(str).map(lambda x: pci_standard(x)))

        drop_columns = list(df.filter(regex=self.CONST_FILTER_REGEX))
        df_anonymization = df.drop(columns=drop_columns)

        return df_anonymization

    def apply_issuer_preserved(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Masking method issuer preserved.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data to be processed.

        Returns:
        --------
        pd.DataFrame
            Processed data.
        """
        def issuer_preserved(value):
            mask_value = ''

            is_valid = False
            pattern = Patterns.CREDIT_PATTERNS
            if re.match(pattern, value):
                is_valid = True

            if is_valid:
                separator_char = self.separator_char if self.preserve_separator else ''
                mask_value = re.sub(r'[^0-9]', separator_char, value)
                idx1 = 6 if len(re.sub(r'[0-9]', '', mask_value)) == 0 else 7

                mask_value = ''.join((
                    mask_value[:idx1],
                    re.sub(r'[0-9]', self.mask_char, mask_value[idx1:])
                ))
            else:
                mask_value = value

            return mask_value

        for target_field in self.target_fields:
            df[target_field] = (df[target_field].astype(str).map(lambda x: issuer_preserved(x)))

        drop_columns = list(df.filter(regex=self.CONST_FILTER_REGEX))
        df_anonymization = df.drop(columns=drop_columns)

        return df_anonymization

    def apply_custom(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Masking method custom.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data to be processed.

        Returns:
        --------
        pd.DataFrame
            Processed data.
        """
        def custom(value):
            mask_value = ''

            is_valid = False
            pattern = Patterns.CREDIT_PATTERNS
            if re.match(pattern, value):
                is_valid = True

            if is_valid:
                separator_char = self.separator_char if self.preserve_separator else ''
                mask_value = re.sub(r'[^0-9]', separator_char, value)

                digits = len(re.sub(r'[^0-9]', '', mask_value))
                digits_to_show = self.digits_to_show if self.digits_to_show > 0 else 0
                digits_to_show = digits_to_show if digits_to_show < digits else digits

                idx1 = 0
                idx2 = 0
                idx3 = 0
                if self.digits_position == 'start':
                    idx1 = digits_to_show
                    if self.preserve_issuer:
                        idx1 = idx1 if idx1 > 6 else 6
                    if self.preserve_check_digit:
                        idx1 = idx1 if idx1 < digits else digits - 1
                        idx3 = 1
                elif self.digits_position == 'end':
                    idx3 = digits_to_show
                    if self.preserve_issuer:
                        idx3 = idx3 if idx3 < digits - 5 else digits - 6
                        idx1 = 6
                    if self.preserve_check_digit:
                        idx3 = idx3 if idx3 > 1 else 1
                else:
                    raise ValueError(f'Invalid digits_position: {self.digits_position}')

                idx1_ext = math.floor((idx1 - 1) / 4) if idx1 > 0 else 0
                idx3_ext = math.floor((digits - idx3 - 1) / 4) if digits - idx3 > 0 else 0

                idx1 = idx1 if len(re.sub(r'[0-9]', '', mask_value)) == 0 else idx1 + idx1_ext
                idx3 = idx3 if len(re.sub(r'[0-9]', '', mask_value)) == 0 else idx3 + 3 - idx3_ext
                idx2 = len(mask_value) - idx3

                mask_value = ''.join((
                    mask_value[:idx1],
                    re.sub(r'[0-9]', self.mask_char, mask_value[idx1:idx2]),
                    mask_value[-idx3:]
                ))
            else:
                mask_value = value

            return mask_value

        for target_field in self.target_fields:
            df[target_field] = (df[target_field].astype(str).map(lambda x: custom(x)))

        drop_columns = list(df.filter(regex=self.CONST_FILTER_REGEX))
        df_anonymization = df.drop(columns=drop_columns)

        return df_anonymization

    def detect_and_check(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect card type and validate.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data to be processed.

        Returns:
        --------
        pd.DataFrame
            Processed data.
        """
        def detect_card_type(value):
            card_type = None

            if isinstance(value, str) and self.detect_card_type:
                card_type = ''
                for type, pattern in self.card_type_patterns.items():
                    if re.match(pattern, value):
                        card_type = type
                        break

            return card_type

        def validation_check(value):
            is_valid = None

            if isinstance(value, str) and self.validation_check:
                is_valid = False
                pattern = Patterns.CREDIT_PATTERNS
                if re.match(pattern, value):
                    is_valid = True

            return is_valid

        def luhn_check(value):
            is_valid = None

            if isinstance(value, str) and self.luhn_check:
                is_valid = False
                final_sum = 0
                list_digit = re.sub(r'[^0-9]', '', value)
                for i, c in enumerate(list_digit):
                    sum_digit = int(c) * (2 if i % 2 == 0 else 1)
                    final_sum = final_sum + sum(divmod(sum_digit, 10))
                if final_sum % 10 == 0:
                    is_valid = True

            return is_valid

        df_processed = df.copy(deep=True)

        for target_field in self.target_fields:
            df_processed[f'{target_field}{self.CONST_CARD_TYPE}'] =\
                df_processed[target_field].astype(str).map(lambda x: detect_card_type(x))
            df_processed[f'{target_field}{self.CONST_VALIDATION_CHECK}'] =\
                df_processed[target_field].astype(str).map(lambda x: validation_check(x))
            df_processed[f'{target_field}{self.CONST_LUHN_CHECK}'] =\
                df_processed[target_field].astype(str).map(lambda x: luhn_check(x))

        return df_processed
