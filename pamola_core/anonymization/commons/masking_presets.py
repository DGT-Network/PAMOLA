"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Masking Presets Library
Package:       pamola_core.anonymization.commons.masking_presets
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
    Provides a comprehensive library of masking presets and configuration utilities
    for common sensitive data types. Supports flexible, configurable, and extensible
    masking strategies for emails, phone numbers, credit cards, SSNs, IP addresses,
    healthcare, and financial identifiers.

Key Features:
    - Centralized preset management for all major identifier types
    - Configurable masking logic with prefix/suffix and format preservation
    - Regex-based validation and type detection utilities
    - Randomized and fixed masking character support
    - Bulk masking and custom configuration creation
    - Extensible architecture for new data types and masking rules

Design Principles:
    - Separation of preset definition, masking logic, and utility functions
    - Configurable, extensible, and testable masking strategies
    - Minimal dependencies, focused on privacy and robustness

Dependencies:
    - re         - Regular expressions for pattern matching
    - random     - Randomized masking character generation
    - dataclasses- Structured configuration objects
    - enum       - Masking type enumeration
    - typing     - Type hints for clarity and safety
    - abc        - Abstract base classes for preset providers
"""

import re
import random
from abc import ABC, abstractmethod
import string
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class MaskingType(Enum):
    """Enumeration of supported masking types."""

    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IP_ADDRESS = "ip_address"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    DATE_ISO = "date_iso"


@dataclass
class MaskingConfig:
    """
    Configuration for data masking operations.

    Attributes:
        pattern_type (str): Type/name of the pattern (e.g., 'email').
        mask_char (str): Character used for masking (default '*').
        preserve_format (bool): Whether to retain special characters/separators.
        preserve_length (bool): Whether to keep the original string length.
        unmasked_prefix (int): Number of unmasked characters at the start of the string.
        unmasked_suffix (int): Number of unmasked characters at the end of the string.
        fixed_length (Optional[int]): Force masked section to this length if set.
        random_mask (bool): Whether to use random characters from `mask_char_pool`.
        mask_char_pool (str): Pool of characters to randomly choose from when masking.
        description (str): Human-readable description of this masking preset.
        example (str): Example input/output demonstrating the preset.
        format_patterns (Dict[str, str]): Regex patterns for input format validation.
        mask_pattern (Optional[str]): Regex pattern indicating what to mask.
        preserve_pattern (Optional[str]): Regex pattern indicating what to preserve.
    """

    pattern_type: str
    mask_char: str = "*"
    preserve_format: bool = True
    preserve_length: bool = True
    unmasked_prefix: int = 0
    unmasked_suffix: int = 0
    fixed_length: Optional[int] = None
    random_mask: bool = False
    mask_char_pool: str = "*"
    description: str = ""
    example: str = ""
    format_patterns: Dict[str, str] = field(default_factory=dict)
    mask_pattern: Optional[str] = None
    preserve_pattern: Optional[str] = None


class BaseMaskingPresets(ABC):
    """
    Abstract base class for masking preset providers.
    Defines the interface for retrieving and applying masking presets.
    """

    @abstractmethod
    def get_presets(self) -> Dict[str, MaskingConfig]:
        """
        Get all masking presets defined in the implementation.

        Returns:
            Dict[str, MaskingConfig]: Mapping of preset names to config objects.
        """
        pass

    @abstractmethod
    def apply_masking(self, data: str, preset_name: str) -> str:
        """
        Apply masking using a given preset.

        Args:
            data (str): The input string to mask.
            preset_name (str): The name of the preset to apply.

        Returns:
            str: Masked string.
        """
        pass

    def list_presets(self) -> List[str]:
        """
        List all available preset names.

        Returns:
            List[str]: List of preset names.
        """
        return list(self.get_presets().keys())

    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """
        Get a summary of the preset configuration for documentation or inspection.

        Args:
            preset_name (str): Name of the preset to describe.

        Returns:
            Dict[str, Any]: Metadata about the preset.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]
        return {
            "name": preset_name,
            "description": config.description,
            "example": config.example,
            "mask_char": config.mask_char,
            "preserve_format": config.preserve_format,
        }


# --------------------------- EMAIL MASKING ------------------------------------
class EmailMaskingPresets(BaseMaskingPresets):
    """
    Email masking presets and implementation logic.
    Provides multiple strategies to anonymize email addresses.
    """

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """
        Define and return email-specific masking presets.

        Returns:
            Dict[str, MaskingConfig]: Preset name mapped to its masking configuration.
        """
        return {
            "FULL_DOMAIN": MaskingConfig(
                pattern_type="email",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=2,
                description="Keep first 1-2 chars and full domain",
                example="user@example.com → us**@example.com",
            ),
            "DOMAIN_ONLY": MaskingConfig(
                pattern_type="email_domain_only",
                mask_char="*",
                preserve_format=True,
                description="Keep domain only, mask local part",
                example="user@example.com → ****@example.com",
            ),
            "PARTIAL_DOMAIN": MaskingConfig(
                pattern_type="email",
                mask_char="*",
                unmasked_prefix=2,
                mask_pattern=r"[^@]+(?=@)",
                preserve_pattern=r"@[^@]+(?=\.)",
                preserve_format=True,
                description="Keep first 2 chars and domain structure",
                example="user@example.com → us**@example.***",
            ),
            "PRIVACY_FOCUSED": MaskingConfig(
                pattern_type="email",
                mask_char="#",
                unmasked_prefix=1,
                fixed_length=8,
                preserve_format=False,
                mask_pattern=r"[^@]+",
                description="High privacy with minimal visibility",
                example="user@example.com → u#######@example.com",
            ),
            "GDPR_COMPLIANT": MaskingConfig(
                pattern_type="email",
                mask_char="X",
                unmasked_prefix=1,
                random_mask=True,
                mask_char_pool="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                preserve_format=True,
                description="GDPR compliant with random masking",
                example="user@example.com → uRTQ@example.com",
            ),
            "UTILITY_BALANCED": MaskingConfig(
                pattern_type="email",
                mask_char="*",
                unmasked_prefix=2,
                preserve_format=True,
                description="Balance between privacy and utility",
                example="user@example.com → us**@example.com",
            ),
            "MINIMAL_EXPOSURE": MaskingConfig(
                pattern_type="email",
                mask_char="*",
                unmasked_prefix=1,
                preserve_format=True,
                description="Minimal exposure with single character",
                example="user@example.com → u***@example.com",
            ),
        }

    def apply_masking(
        self, email: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply masking to a given email using the specified preset.

        Args:
            email (str): Input email address to be masked.
            preset_name (str): Preset name from the configuration.
            random_mask (bool): Whether to use random characters from the pool for masking.

        Returns:
            str: Masked email string.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]
        email_pattern = r"^([^@]+)@(.+)$"
        match = re.match(email_pattern, email)
        if not match:
            return email  # Return original if invalid format

        local_part, domain = match.groups()

        if config.pattern_type == "email_domain_only":
            masked_local = config.mask_char * len(local_part)
            return f"{masked_local}@{domain}"

        masked_local = self._mask_string(local_part, config, random_mask)

        # Apply domain masking logic if applicable (e.g., PARTIAL_DOMAIN)
        if preset_name == "PARTIAL_DOMAIN":
            domain_parts = domain.split(".")
            if len(domain_parts) > 1:
                domain_parts[-1] = config.mask_char * len(domain_parts[-1])
                domain = ".".join(domain_parts)

        return f"{masked_local}@{domain}"

    def _mask_string(
        self, text: str, config: MaskingConfig, random_mask: bool = False
    ) -> str:
        """Perform string-level masking on part of an email."""
        if len(text) <= config.unmasked_prefix + config.unmasked_suffix:
            return text

        prefix = text[: config.unmasked_prefix] if config.unmasked_prefix else ""
        suffix = text[-config.unmasked_suffix :] if config.unmasked_suffix else ""

        middle_length = config.fixed_length or (
            len(text) - config.unmasked_prefix - config.unmasked_suffix
        )

        if random_mask and config.mask_char_pool:
            middle = "".join(random.choices(config.mask_char_pool, k=middle_length))
        else:
            middle = config.mask_char * middle_length

        return prefix + middle + suffix


# --------------------------- PHONE MASKING ------------------------------------
class PhoneMaskingPresets(BaseMaskingPresets):
    """Phone number masking configurations and implementation."""

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """List phone masking presets."""
        return {
            "US_STANDARD": MaskingConfig(
                pattern_type="phone",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=3,
                unmasked_suffix=4,
                description="Keep area code and last 4 digits",
                format_patterns={
                    "us_dash": r"(\d{3})-(\d{3})-(\d{4})",
                    "us_dot": r"(\d{3})\.(\d{3})\.(\d{4})",
                    "us_space": r"(\d{3}) (\d{3}) (\d{4})",
                },
                example="555-123-4567 → 555-***-4567",
            ),
            "US_FORMATTED": MaskingConfig(
                pattern_type="phone_us_formatted",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=3,
                unmasked_suffix=4,
                description="Keep area code and last 4 (formatted)",
                example="(555) 123-4567 → (555) ***-4567",
            ),
            "INTERNATIONAL": MaskingConfig(
                pattern_type="phone_international",
                mask_char="*",
                unmasked_prefix=4,
                unmasked_suffix=4,
                preserve_format=True,
                description="Keep country code and last 4",
                example="+1-555-123-4567 → +1-***-***-4567",
            ),
            "LAST_FOUR_ONLY": MaskingConfig(
                pattern_type="phone",
                unmasked_prefix=0,
                unmasked_suffix=4,
                mask_char="X",
                preserve_format=True,
                description="Keep last 4 digits only",
                example="555-123-4567 → XXX-XXX-4567",
            ),
            "AREA_CODE_ONLY": MaskingConfig(
                pattern_type="phone",
                unmasked_prefix=3,
                unmasked_suffix=0,
                mask_char="*",
                preserve_format=True,
                description="Keep area code only",
                example="555-123-4567 → 555-***-****",
            ),
            "FULL_MASK": MaskingConfig(
                pattern_type="phone",
                unmasked_prefix=0,
                unmasked_suffix=0,
                mask_char="*",
                preserve_format=True,
                description="Complete phone masking",
                example="555-123-4567 → ***-***-****",
            ),
        }

    def apply_masking(
        self, phone: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply phone masking using specified preset.

        Args:
            phone (str): Input phone number (may include separators).
            preset_name (str): Which preset to use.
            random_mask (bool): If True, replace masked characters with random digits.

        Returns:
            str: Masked phone. If invalid input, returns original.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        config = presets[preset_name]
        digits = re.sub(r"\D", "", phone)
        if len(digits) < 10:
            return phone  # Not enough digits

        if preset_name == "US_STANDARD":
            return self._mask_us_standard(phone, digits, config, random_mask)
        elif preset_name == "US_FORMATTED":
            return self._mask_us_formatted(phone, digits, config, random_mask)
        elif preset_name == "INTERNATIONAL":
            return self._mask_international(phone, digits, config, random_mask)
        else:
            return self._mask_generic_phone(phone, digits, config, random_mask)

    def _mask_us_standard(
        self, phone: str, digits: str, config: MaskingConfig, random_mask: bool
    ) -> str:
        """Mask standard US phone format."""
        if len(digits) == 10:
            area = digits[:3]
            middle = self._random_or_fixed_mask(config.mask_char, 3, random_mask)
            last = digits[-4:]
            if "-" in phone:
                return f"{area}-{middle}-{last}"
            elif "." in phone:
                return f"{area}.{middle}.{last}"
            elif " " in phone:
                return f"{area} {middle} {last}"
            else:
                return f"{area}{middle}{last}"
        return phone

    def _mask_us_formatted(
        self, phone: str, digits: str, config: MaskingConfig, random_mask: bool
    ) -> str:
        """Mask US formatted phone (with parentheses)."""
        if len(digits) == 10:
            area = digits[:3]
            middle = self._random_or_fixed_mask(config.mask_char, 3, random_mask)
            last = digits[-4:]
            if "(" in phone and ")" in phone:
                return f"({area}) {middle}-{last}"
            else:
                return f"{area}-{middle}-{last}"
        return phone

    def _mask_international(
        self, phone: str, digits: str, config: MaskingConfig, random_mask: bool
    ) -> str:
        """Mask international phone number."""
        if phone.startswith("+"):
            country_match = re.match(r"(\+\d{1,3})", phone)
            if country_match:
                country_code = country_match.group(1)
                remaining = phone[len(country_code) :]
                remaining_digits = re.sub(r"\D", "", remaining)
                if len(remaining_digits) >= 4:
                    masked_middle_length = len(remaining_digits) - 4
                    last_four = remaining_digits[-4:]
                    result = country_code
                    digit_index = 0
                    for char in remaining:
                        if char.isdigit():
                            if digit_index < masked_middle_length:
                                result += (
                                    str(random.randint(0, 9))
                                    if random_mask
                                    else config.mask_char
                                )
                            else:
                                result += last_four[digit_index - masked_middle_length]
                            digit_index += 1
                        else:
                            result += char
                    return result
        return phone

    def _mask_generic_phone(
        self, phone: str, digits: str, config: MaskingConfig, random_mask: bool
    ) -> str:
        """Apply generic phone masking using configuration."""
        masked_digits = self._mask_string(digits, config, random_mask)
        result = ""
        digit_index = 0
        for char in phone:
            if char.isdigit():
                if digit_index < len(masked_digits):
                    result += masked_digits[digit_index]
                    digit_index += 1
            else:
                result += char
        return result

    def _mask_string(self, text: str, config: MaskingConfig, random_mask: bool) -> str:
        """Generic masking for string with prefix/suffix options."""
        if len(text) <= config.unmasked_prefix + config.unmasked_suffix:
            return text
        prefix = text[: config.unmasked_prefix] if config.unmasked_prefix > 0 else ""
        suffix = text[-config.unmasked_suffix :] if config.unmasked_suffix > 0 else ""
        middle_length = len(text) - config.unmasked_prefix - config.unmasked_suffix
        middle = self._random_or_fixed_mask(
            config.mask_char, middle_length, random_mask
        )
        return prefix + middle + suffix

    def _random_or_fixed_mask(
        self, mask_char: str, length: int, random_mask: bool
    ) -> str:
        """Helper to create mask string."""
        if random_mask:
            return "".join(str(random.randint(0, 9)) for _ in range(length))
        return mask_char * length


# ------------- CREDIT CARD MASKING -----------------------------------
class CreditCardMaskingPresets(BaseMaskingPresets):
    """Credit card masking presets conforming to PCI DSS and other custom rules."""

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """
        Define masking presets for credit card numbers.

        Returns:
            Dict[str, MaskingConfig]: Mapping of preset names to configurations.
        """
        return {
            "PCI_COMPLIANT": MaskingConfig(
                pattern_type="credit_card",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=6,
                unmasked_suffix=4,
                description="PCI compliant: first 6 (BIN) and last 4",
                example="4111-1111-1111-1111 → 4111-11**-****-1111",
            ),
            "STRICT": MaskingConfig(
                pattern_type="credit_card_strict",
                mask_char="X",
                preserve_format=True,
                unmasked_prefix=0,
                unmasked_suffix=4,
                description="Strict: last 4 digits only",
                example="4111-1111-1111-1111 → XXXX-XXXX-XXXX-1111",
            ),
            "FULL_MASK": MaskingConfig(
                pattern_type="credit_card",
                mask_char="*",
                preserve_format=True,
                preserve_length=True,
                unmasked_prefix=0,
                unmasked_suffix=0,
                description="Complete masking with format",
                example="4111-1111-1111-1111 → ****-****-****-****",
            ),
            "NUMERIC_ONLY": MaskingConfig(
                pattern_type="credit_card",
                mask_char="9",
                preserve_format=False,
                preserve_length=True,
                unmasked_prefix=0,
                unmasked_suffix=0,
                description="Numeric masking without separators",
                example="4111111111111111 → 9999999999999999",
            ),
            "FIRST_LAST_FOUR": MaskingConfig(
                pattern_type="credit_card",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=4,
                unmasked_suffix=4,
                description="Keep first 4 and last 4 digits",
                example="4111-1111-1111-1111 → 4111-****-****-1111",
            ),
        }

    def apply_masking(
        self, card_number: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply masking to a credit card number using the specified preset.

        Args:
            card_number (str): Input credit card number (may contain separators).
            preset_name (str): Preset name as defined in get_presets().
            random_mask (bool): If True, use random digits/characters instead of fixed mask_char.

        Returns:
            str: Masked credit card number, preserving formatting if configured.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        config = presets[preset_name]

        digits = re.sub(r"\D", "", card_number)
        if len(digits) < 13 or len(digits) > 19:
            return card_number  # Invalid credit card length

        masked_digits = self._mask_string(digits, config, random_mask=random_mask)
        if not config.preserve_format:
            return masked_digits

        result = ""
        digit_index = 0
        for char in card_number:
            if char.isdigit():
                if digit_index < len(masked_digits):
                    result += masked_digits[digit_index]
                    digit_index += 1
            else:
                result += char
        return result

    def _mask_string(
        self, text: str, config: MaskingConfig, random_mask: bool = False
    ) -> str:
        """Mask the given numeric string using the configuration."""
        if len(text) <= config.unmasked_prefix + config.unmasked_suffix:
            return text

        prefix = text[: config.unmasked_prefix] if config.unmasked_prefix > 0 else ""
        suffix = text[-config.unmasked_suffix :] if config.unmasked_suffix > 0 else ""

        middle_length = len(text) - config.unmasked_prefix - config.unmasked_suffix

        if random_mask:
            if config.mask_char.isdigit():
                middle = "".join(
                    str(random.randint(0, 9)) for _ in range(middle_length)
                )
            elif config.mask_char.isalpha():
                middle = "".join(
                    random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    for _ in range(middle_length)
                )
            else:
                middle = "".join(
                    random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    for _ in range(middle_length)
                )
        else:
            middle = config.mask_char * middle_length

        return prefix + middle + suffix


# ----------------- SSN MASKING -------------------------
class SSNMaskingPresets(BaseMaskingPresets):
    """Social Security Number masking configurations."""

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """
        Return available SSN masking presets.

        Returns:
            Dict[str, MaskingConfig]: Dictionary of masking presets.
        """
        return {
            "LAST_FOUR": MaskingConfig(
                pattern_type="ssn",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=0,
                unmasked_suffix=4,
                description="Keep last 4 digits only (standard)",
                example="123-45-6789 → ***-**-6789",
            ),
            "FIRST_THREE": MaskingConfig(
                pattern_type="ssn_middle",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=3,
                unmasked_suffix=4,
                description="Keep first 3 and last 4 digits",
                example="123-45-6789 → 123-**-6789",
            ),
            "FULL_MASK": MaskingConfig(
                pattern_type="ssn",
                mask_char="*",
                preserve_format=True,
                preserve_length=True,
                unmasked_prefix=0,
                unmasked_suffix=0,
                description="Complete masking with format",
                example="123-45-6789 → ***-**-****",
            ),
            "NUMERIC_MASK": MaskingConfig(
                pattern_type="ssn",
                mask_char="0",
                preserve_format=False,
                preserve_length=True,
                unmasked_prefix=0,
                unmasked_suffix=0,
                description="Numeric masking without separators",
                example="123456789 → 000000000",
            ),
            "AREA_NUMBER_ONLY": MaskingConfig(
                pattern_type="ssn",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=3,
                unmasked_suffix=0,
                description="Keep area number (first 3) only",
                example="123-45-6789 → 123-**-****",
            ),
        }

    def apply_masking(
        self, ssn: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply SSN masking using specified preset.

        Args:
            ssn (str): SSN string (with or without separators).
            preset_name (str): Which preset to use.
            random_mask (bool): Whether to use random digits/characters instead of fixed mask_char.

        Returns:
            str: Masked SSN. If invalid input or preset, returns original string.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]
        digits = re.sub(r"\D", "", ssn)
        if len(digits) != 9:
            return ssn

        masked_digits = self._mask_string(digits, config, random_mask=random_mask)

        if not config.preserve_format:
            return masked_digits
        return f"{masked_digits[:3]}-{masked_digits[3:5]}-{masked_digits[5:]}"

    def _mask_string(
        self, text: str, config: MaskingConfig, random_mask: bool = False
    ) -> str:
        """Mask numeric string with prefix/suffix options."""
        if len(text) <= config.unmasked_prefix + config.unmasked_suffix:
            return text

        prefix = text[: config.unmasked_prefix] if config.unmasked_prefix > 0 else ""
        suffix = text[-config.unmasked_suffix :] if config.unmasked_suffix > 0 else ""
        middle_length = len(text) - config.unmasked_prefix - config.unmasked_suffix

        if random_mask:
            if config.mask_char.isdigit():
                mask_pool = "0123456789"
            else:
                mask_pool = (
                    config.mask_char * 5
                    + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                )
            middle = "".join(random.choices(mask_pool, k=middle_length))
        else:
            middle = config.mask_char * middle_length

        return prefix + middle + suffix


# ----------------- IP ADDRESS MASKING -------------------------
class IPAddressMaskingPresets(BaseMaskingPresets):
    """IP Address masking configurations."""

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """Get IP address masking presets."""
        return {
            "SUBNET_MASK": MaskingConfig(
                pattern_type="ip_address",
                mask_char="*",
                preserve_format=True,
                description="Keep first two octets (subnet)",
                example="192.168.1.100 → 192.168.*.*",
            ),
            "NETWORK_ONLY": MaskingConfig(
                pattern_type="ip_address_last_only",
                mask_char="*",
                preserve_format=True,
                description="Keep first octet only",
                example="192.168.1.100 → 192.*.*.*",
            ),
            "FULL_MASK": MaskingConfig(
                pattern_type="ip_address",
                mask_char="*",
                preserve_format=True,
                preserve_length=True,
                description="Complete IP masking",
                example="192.168.1.100 → ***.***.*.**",
            ),
            "ZERO_MASK": MaskingConfig(
                pattern_type="ip_address",
                mask_char="0",
                preserve_format=True,
                preserve_length=False,
                description="Zero out IP address",
                example="192.168.1.100 → 0.0.0.0",
            ),
            "PRIVATE_NETWORK": MaskingConfig(
                pattern_type="ip_address",
                mask_char="*",
                preserve_format=True,
                description="Keep private network identifier",
                example="10.0.1.100 → 10.*.*.*",
            ),
        }

    def apply_masking(
        self, ip_address: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply IP address masking using the specified preset.

        Parameters
        ----------
        ip_address : str
            Input IP address (IPv4 format).
        preset_name : str
            The name of the masking preset to apply.
        random_mask : bool
            Whether to use random digits instead of fixed mask characters.

        Returns
        -------
        str
            Masked IP address string.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]

        # Validate IP address format
        ip_pattern = r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
        match = re.match(ip_pattern, ip_address)

        if not match:
            return ip_address  # Return original if invalid format

        octets = match.groups()

        if preset_name == "SUBNET_MASK":
            return f"{octets[0]}.{octets[1]}.{self._mask_string(len(octets[2]), config.mask_char, random_mask)}.{self._mask_string(len(octets[3]), config.mask_char, random_mask)}"

        elif preset_name == "NETWORK_ONLY":
            return f"{octets[0]}.{self._mask_string(len(octets[1]), config.mask_char, random_mask)}.{self._mask_string(len(octets[2]), config.mask_char, random_mask)}.{self._mask_string(len(octets[3]), config.mask_char, random_mask)}"

        elif preset_name == "ZERO_MASK":
            return "0.0.0.0"

        elif preset_name == "PRIVATE_NETWORK":
            return f"{octets[0]}.{self._mask_string(len(octets[1]), config.mask_char, random_mask)}.{self._mask_string(len(octets[2]), config.mask_char, random_mask)}.{self._mask_string(len(octets[3]), config.mask_char, random_mask)}"

        elif preset_name == "FULL_MASK":
            if config.preserve_length:
                masked_octets = [
                    self._mask_string(len(o), config.mask_char, random_mask)
                    for o in octets
                ]
            else:
                masked_octets = [
                    self._mask_string(1, config.mask_char, random_mask) for _ in octets
                ]
            return ".".join(masked_octets)

        return ip_address

    def _mask_string(
        length: int, mask_char: str = "*", random_mask: bool = False
    ) -> str:
        """Generate a masked string of given length."""
        if random_mask:
            return "".join(random.choices("0123456789", k=length))
        return mask_char * length


# ----------------- HEALTHCARE MASKING -------------------------
class HealthcareMaskingPresets(BaseMaskingPresets):
    """Healthcare-specific masking configurations."""

    def get_presets(self) -> Dict[str, MaskingConfig]:
        return {
            "MEDICAL_RECORD": MaskingConfig(
                pattern_type="medical_record",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=2,
                unmasked_suffix=2,
                description="Keep facility code and partial ID",
                example="MR12345678 → MR******78",
            ),
            "PATIENT_ID": MaskingConfig(
                pattern_type="patient_id",
                mask_char="*",
                preserve_format=False,
                unmasked_prefix=2,
                unmasked_suffix=2,
                description="Partial patient ID masking",
                example="PAT123456 → PA****56",
            ),
            "NPI_NUMBER": MaskingConfig(
                pattern_type="npi",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=3,
                unmasked_suffix=3,
                description="National Provider Identifier masking",
                example="1234567890 → 123****890",
            ),
            "DEA_NUMBER": MaskingConfig(
                pattern_type="dea",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=2,
                unmasked_suffix=2,
                description="DEA number masking",
                example="AB1234567 → AB***67",
            ),
        }

    def apply_masking(
        self, data: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply masking to the given string based on preset.

        Parameters
        ----------
        data : str
            The original string (e.g., medical record ID).
        preset_name : str
            The name of the preset to use.
        random_mask : bool
            Whether to apply randomized masking characters (e.g., aB@1#).

        Returns
        -------
        str
            Masked string.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]
        return self._mask_string(data, config, random_mask=random_mask)

    def _mask_string(
        self, text: str, config: MaskingConfig, random_mask: bool = False
    ) -> str:
        """Generate a masked string based on the given configuration."""
        total_len = len(text)
        unmasked_prefix = config.unmasked_prefix or 0
        unmasked_suffix = config.unmasked_suffix or 0

        # No masking needed if text too short
        if total_len <= unmasked_prefix + unmasked_suffix:
            return text

        prefix = text[:unmasked_prefix] if unmasked_prefix > 0 else ""
        suffix = text[-unmasked_suffix:] if unmasked_suffix > 0 else ""

        mask_len = total_len - unmasked_prefix - unmasked_suffix
        masked_middle = self._generate_mask_string(
            mask_len, config.mask_char, random_mask
        )

        return prefix + masked_middle + suffix

    def _generate_mask_string(
        self, length: int, mask_char: str, random_mask: bool = False
    ) -> str:
        """Generate a masked string of given length."""
        if not random_mask:
            return mask_char * length

        mask_pool = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%^&*"
        )
        return "".join(random.choice(mask_pool) for _ in range(length))


# ------------------------- FINANCIAL MASKING -------------------------
class FinancialMaskingPresets(BaseMaskingPresets):
    """Financial data masking configurations and application logic."""

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """Return predefined masking configurations for financial data."""
        return {
            "ACCOUNT_NUMBER": MaskingConfig(
                pattern_type="account_number",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=2,
                unmasked_suffix=4,
                description="Keep first 2 and last 4 characters",
                example="1234567890 → 12****7890",
            ),
            "ROUTING_NUMBER": MaskingConfig(
                pattern_type="routing_number",
                unmasked_prefix=2,
                unmasked_suffix=2,
                mask_char="*",
                preserve_format=False,
                description="Partial routing number masking",
                example="123456789 → 12*****89",
            ),
            "BANK_STANDARD": MaskingConfig(
                pattern_type="bank_account",
                unmasked_prefix=4,
                unmasked_suffix=4,
                mask_char="*",
                preserve_format=True,
                description="Keep first 4 and last 4 characters",
                example="1234567890123 → 1234*****0123",
            ),
            "SWIFT_CODE": MaskingConfig(
                pattern_type="swift",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=4,
                unmasked_suffix=2,
                description="Keep bank code and country code",
                example="CHASUS33XXX → CHAS****33",
            ),
            "IBAN": MaskingConfig(
                pattern_type="iban",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=4,
                unmasked_suffix=4,
                description="Keep country code and last 4 chars",
                example="GB29NWBK60161331926819 → GB29****************6819",
            ),
            "CREDIT_LIMIT": MaskingConfig(
                pattern_type="credit_limit",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=1,
                unmasked_suffix=2,
                description="Mask credit limit keeping first and last 2 digits",
                example="50000 → 5***00",
            ),
            "LOAN_NUMBER": MaskingConfig(
                pattern_type="loan_number",
                mask_char="*",
                preserve_format=True,
                unmasked_prefix=3,
                unmasked_suffix=3,
                description="Keep first 3 and last 3 characters",
                example="LN123456789 → LN1****789",
            ),
        }

    def apply_masking(
        self, data: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply financial masking using specified preset.

        Parameters
        ----------
        data : str
            The input financial data to mask.
        preset_name : str
            The preset key defined in get_presets().
        random_mask : bool, default=False
            If True, masking characters are randomly selected (A-Z, 0-9).
            If False, uses fixed mask_char from config.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]

        if preset_name == "IBAN":
            return self._mask_iban(data, config, random_mask)
        elif preset_name == "SWIFT_CODE":
            return self._mask_swift(data, config, random_mask)
        elif preset_name == "ROUTING_NUMBER":
            return self._mask_routing_number(data, config, random_mask)
        else:
            return self._mask_string(data, config, random_mask)

    def _mask_string(self, text: str, config: MaskingConfig, random_mask: bool) -> str:
        """Apply general masking to string based on config."""
        if len(text) <= config.unmasked_prefix + config.unmasked_suffix:
            return text

        prefix = text[: config.unmasked_prefix] if config.unmasked_prefix else ""
        suffix = text[-config.unmasked_suffix :] if config.unmasked_suffix else ""

        middle_length = len(text) - len(prefix) - len(suffix)
        if random_mask:
            middle = self._random_mask(middle_length)
        else:
            middle = config.mask_char * middle_length

        return prefix + middle + suffix

    def _mask_iban(self, iban: str, config: MaskingConfig, random_mask: bool) -> str:
        """Apply IBAN-specific masking."""
        clean_iban = iban.replace(" ", "").upper()
        if len(clean_iban) < 15:
            return iban

        country_check = clean_iban[:4]
        account_part = clean_iban[4:]
        account_len = len(account_part)

        if account_len > 4:
            masked_len = account_len - 4
            masked_account = (
                self._random_mask(masked_len)
                if random_mask
                else config.mask_char * masked_len
            ) + account_part[-4:]
        else:
            masked_account = account_part

        masked_iban = country_check + masked_account

        return (
            " ".join([masked_iban[i : i + 4] for i in range(0, len(masked_iban), 4)])
            if " " in iban
            else masked_iban
        )

    def _mask_swift(self, swift: str, config: MaskingConfig, random_mask: bool) -> str:
        """Apply SWIFT code specific masking."""
        clean_swift = swift.replace(" ", "").upper()
        if len(clean_swift) < 8:
            return swift

        bank_code = clean_swift[:4]
        country_code = clean_swift[4:6]
        location = clean_swift[6:8]

        masked_location = (
            self._random_mask(len(location))
            if random_mask
            else config.mask_char * len(location)
        )

        if len(clean_swift) == 8:
            return f"{bank_code}{country_code}{masked_location}"
        elif len(clean_swift) == 11:
            branch = clean_swift[8:11]
            return f"{bank_code}{country_code}{masked_location}{branch}"
        return swift

    def _mask_routing_number(
        self, routing: str, config: MaskingConfig, random_mask: bool
    ) -> str:
        """Apply routing number specific masking."""
        digits = re.sub(r"\D", "", routing)
        if len(digits) != 9:
            return routing

        masked_middle = self._random_mask(5) if random_mask else config.mask_char * 5
        return digits[:2] + masked_middle + digits[-2:]

    def _random_mask(self, length: int) -> str:
        """Generate a random alphanumeric mask of specified length."""
        charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(random.choices(charset, k=length))


# ------------------------- DATE MASKING -------------------------
class DateMaskingPresets(BaseMaskingPresets):
    """
    Date masking presets and implementation logic.
    Provides strategies to anonymize date fields by masking specific components.
    Supports ISO date format.
    """

    def get_presets(self) -> Dict[str, MaskingConfig]:
        """
        Define and return date-specific masking presets.

        Returns:
            Dict[str, MaskingConfig]: Preset name mapped to its masking configuration.
        """
        return {
            # Mask only the day component (e.g., 2024-07-15 → 2024-07-XX)
            "MASK_DAY": MaskingConfig(
                pattern_type="date_iso",
                mask_char="X",
                preserve_format=True,
                description="Mask only the day component of the date",
                example="2024-07-15 → 2024-07-XX",
                mask_pattern=r"(?<=\d{4}-\d{2}-)\d{2}",
            ),
            # Mask month only (e.g., 2024-07-15 → 2024-XX-15)
            "MASK_MONTH": MaskingConfig(
                pattern_type="date_iso",
                mask_char="X",
                preserve_format=True,
                description="Mask only the month component of the date",
                example="2024-07-15 → 2024-XX-15",
                mask_pattern=r"(?<=\d{4}-)\d{2}(?=-\d{2})",
            ),
            # Mask both month and day (e.g., 2024-07-23 → 2024-XX-XX)
            "MASK_MONTH_DAY": MaskingConfig(
                pattern_type="date_iso",
                mask_char="X",
                preserve_format=True,
                description="Mask both month and day components of the date",
                example="2024-07-23 → 2024-XX-XX",
                mask_pattern=r"(?<=\d{4}-)\d{2}-\d{2}",
            ),
            # Mask year only (e.g., 2024-07-23 → XXXX-07-23)
            "MASK_YEAR": MaskingConfig(
                pattern_type="date_iso",
                mask_char="X",
                preserve_format=True,
                description="Mask only the year component of the date",
                example="2024-07-23 → XXXX-07-23",
                mask_pattern=r"^\d{4}(?=-)",
            ),
            # Mask entire date (e.g., 2024-07-23 → XXXX-XX-XX)
            "MASK_FULL": MaskingConfig(
                pattern_type="date_iso",
                mask_char="X",
                preserve_format=True,
                description="Mask entire date (year, month, day)",
                example="2024-07-23 → XXXX-XX-XX",
                mask_pattern=r"\d{4}-\d{2}-\d{2}",
            ),
        }

    def apply_masking(
        self,
        date_str: str, preset_name: str, random_mask: bool = False
    ) -> str:
        """
        Apply masking to a given date string using the specified preset.

        Args:
            date_str (str): Input date string (e.g., "2024-07-23").
            preset_name (str): Preset name from the configuration.
            random_mask (bool): Whether to use random characters for masking.

        Returns:
            str: Masked date string.
        """
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        config = presets[preset_name]
        pattern = config.mask_pattern
        mask_char = config.mask_char

        def mask_match(m):
            length = len(m.group())
            return (
                "".join(random.choices(string.ascii_uppercase, k=length))
                if random_mask
                else mask_char * length
            )

        return re.sub(pattern, mask_match, date_str)


class MaskingPresetManager:
    """Central manager for all masking presets."""

    def __init__(self):
        # Initialize mapping of masking types to their preset managers
        self.presets = {
            MaskingType.EMAIL: EmailMaskingPresets(),
            MaskingType.PHONE: PhoneMaskingPresets(),
            MaskingType.CREDIT_CARD: CreditCardMaskingPresets(),
            MaskingType.SSN: SSNMaskingPresets(),
            MaskingType.IP_ADDRESS: IPAddressMaskingPresets(),
            MaskingType.HEALTHCARE: HealthcareMaskingPresets(),
            MaskingType.FINANCIAL: FinancialMaskingPresets(),
            MaskingType.DATE_ISO: DateMaskingPresets(),
        }

    def get_preset_manager(self, masking_type: MaskingType) -> BaseMaskingPresets:
        """
        Get preset manager for a specific masking type.

        Parameters:
        - masking_type (MaskingType): The type of data to be masked.

        Returns:
        - BaseMaskingPresets: Corresponding preset manager instance.
        """
        if masking_type not in self.presets:
            raise ValueError(f"Masking type '{masking_type}' not supported")
        return self.presets[masking_type]

    def list_all_presets(self) -> Dict[str, List[str]]:
        """
        List all available presets for each masking type.

        Returns:
        - Dict[str, List[str]]: Dictionary of preset names grouped by type.
        """
        return {
            masking_type.value: manager.list_presets()
            for masking_type, manager in self.presets.items()
        }

    def apply_masking(
        self,
        data: str,
        masking_type: MaskingType,
        preset_name: str,
        random_mask: bool = False,
    ) -> str:
        """
        Apply a masking preset to the input data.

        Parameters:
        - data (str): Input string to mask.
        - masking_type (MaskingType): Type of data (e.g., PHONE, EMAIL).
        - preset_name (str): Preset name to use.
        - random_mask (bool): Whether to randomize mask characters (default False).

        Returns:
        - str: Masked data.
        """
        if masking_type not in self.presets:
            raise ValueError(f"Masking type '{masking_type}' not supported")

        manager = self.presets[masking_type]
        return manager.apply_masking(data, preset_name, random_mask=random_mask)

    def get_preset_info(
        self,
        masking_type: MaskingType,
        preset_name: str,
    ) -> Dict[str, Any]:
        """
        Retrieve metadata or description of a preset.

        Parameters:
        - masking_type (MaskingType): Data type category.
        - preset_name (str): Preset identifier.

        Returns:
        - Dict[str, Any]: Details of the masking preset.
        """
        manager = self.get_preset_manager(masking_type)
        return manager.get_preset_info(preset_name)

    def validate_data(self, data: str, masking_type: MaskingType) -> bool:
        """
        Validate input data format based on type.

        Parameters:
        - data (str): Input string to validate.
        - masking_type (MaskingType): Expected data type.

        Returns:
        - bool: True if valid, else False.
        """
        patterns = {
            MaskingType.EMAIL: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            MaskingType.PHONE: r"^[\+]?[\d\s\-\(\)\.]{10,}$",
            MaskingType.CREDIT_CARD: r"^[\d\s\-]{13,19}$",
            MaskingType.SSN: r"^\d{3}-?\d{2}-?\d{4}$",
            MaskingType.IP_ADDRESS: r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            MaskingType.HEALTHCARE: r"^[A-Z]{2,3}\d+$",
            MaskingType.FINANCIAL: r"^[A-Z0-9]{8,34}$",
            MaskingType.DATE_ISO: r"^\d{4}-\d{2}-\d{2}$",
        }

        return bool(re.match(patterns.get(masking_type, ""), data))


# Additional utility functions
class MaskingUtils:
    """Utility functions for masking operations."""

    @staticmethod
    def detect_data_type(data: str) -> Optional[MaskingType]:
        """
        Attempt to detect the data type from the input string.

        Parameters:
        - data (str): The input string to analyze.

        Returns:
        - Optional[MaskingType]: Detected type or None if not matched.
        """
        patterns = {
            MaskingType.EMAIL: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            MaskingType.SSN: r"^\d{3}-?\d{2}-?\d{4}$",
            MaskingType.CREDIT_CARD: r"^[\d\s\-]{13,19}$",
            MaskingType.IP_ADDRESS: r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            MaskingType.DATE_ISO: r"^\d{4}-\d{2}-\d{2}$",
            MaskingType.PHONE: r"^[\+]?[\d\s\-\(\)\.]{10,}$",
            MaskingType.HEALTHCARE: r"^[A-Z]{2,3}\d+$",
            MaskingType.FINANCIAL: r"^[A-Z0-9]{8,34}$",
        }

        for data_type, pattern in patterns.items():
            if re.match(pattern, data.strip()):
                return data_type

        return None

    @staticmethod
    def bulk_mask(
        data_list: List[str],
        masking_type: MaskingType,
        preset_name: str,
        random_mask: bool = False,
    ) -> List[str]:
        """
        Apply masking to a list of data items using a specified type and preset.

        Parameters:
        - data_list (List[str]): List of data strings.
        - masking_type (MaskingType): Type of data (e.g. PHONE, EMAIL).
        - preset_name (str): The preset name to apply.
        - random_mask (bool): Whether to use randomized masking characters.

        Returns:
        - List[str]: Masked data list.
        """
        manager = MaskingPresetManager()
        result = []

        for data in data_list:
            try:
                masked = manager.apply_masking(
                    data, masking_type, preset_name, random_mask=random_mask
                )
                result.append(masked)
            except Exception:
                result.append(data)  # Fallback: return original on failure

        return result

    @staticmethod
    def create_custom_config(
        mask_char: str = "*",
        unmasked_prefix: int = 0,
        unmasked_suffix: int = 0,
        preserve_format: bool = True,
        description: str = "Custom masking configuration",
    ) -> MaskingConfig:
        """
        Create a custom masking configuration.

        Parameters:
        - mask_char (str): Character to use for masking.
        - unmasked_prefix (int): Characters to keep at the start.
        - unmasked_suffix (int): Characters to keep at the end.
        - preserve_format (bool): Whether to retain original format.
        - description (str): Optional description.

        Returns:
        - MaskingConfig: Config object for masking.
        """
        return MaskingConfig(
            pattern_type="custom",
            mask_char=mask_char,
            unmasked_prefix=unmasked_prefix,
            unmasked_suffix=unmasked_suffix,
            preserve_format=preserve_format,
            description=description,
        )
