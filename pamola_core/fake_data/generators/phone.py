"""
Phone number generation for fake data system.

This module provides the PhoneGenerator class for generating synthetic phone numbers
while preserving statistical properties of the original data and supporting
consistent mapping.
"""

import random
import re
import string
from typing import Dict, Any, List, Optional, Tuple, Union

from pamola_core.fake_data.commons import dict_helpers
from pamola_core.fake_data.commons.prgn import PRNGenerator
from pamola_core.fake_data.dictionaries import phones
from pamola_core.fake_data.generators.base_generator import BaseGenerator


class PhoneGenerator(BaseGenerator):
    """
    Generator for synthetic phone numbers.

    Generates phone numbers in various formats, optionally preserving
    country and operator codes from original numbers and supporting
    consistent mapping with regional specificity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize phone generator with configuration.

        Args:
            config: Configuration parameters including:
                - country_codes: Used country codes (list or dict with weights)
                - operator_codes_dict: Path to dictionary of operator codes
                - format: Output format template (e.g., "+CC (AAA) XXX-XX-XX")
                - validate_source: Whether to validate source phone numbers
                - handle_invalid_phone: How to handle invalid numbers
                - default_country: Default country code for generation
                - preserve_country_code: Whether to preserve country code from original
                - preserve_operator_code: Whether to preserve operator code from original
                - region: Region/country for formatting
                - key: Key for PRGN
                - context_salt: Salt for PRGN
        """
        super().__init__(config)

        # Store config in attributes for easy access
        self.country_codes = self._process_country_codes(self.config.get('country_codes', {}))
        self.operator_codes_dict = self.config.get('operator_codes_dict')
        self.format = self.config.get('format')
        self.validate_source = self.config.get('validate_source', True)
        self.handle_invalid_phone = self.config.get('handle_invalid_phone', 'generate_new')
        self.default_country = self.config.get('default_country', 'us')
        self.preserve_country_code = self.config.get('preserve_country_code', True)
        self.preserve_operator_code = self.config.get('preserve_operator_code', False)
        self.region = self.config.get('region', self.default_country)

        # Load formats
        self._phone_formats = phones.get_phone_formats()

        # Load operator codes from dictionary or embedded
        self._operator_codes = self._load_operator_codes()

        # Set up PRGN generator if needed
        self.prgn_generator = None
        key = self.config.get('key')
        if key:
            self.prgn_generator = PRNGenerator(global_seed=key)

        # Phone validation patterns
        self.phone_pattern = re.compile(r'^\+?[0-9\s()-.]+$')
        self.digits_pattern = re.compile(r'[0-9]+')

    def _process_country_codes(self, country_codes: Union[Dict, List, None]) -> Dict[str, float]:
        """
        Process and normalize country codes configuration.

        Args:
            country_codes: Country codes as list, dict, or None

        Returns:
            Dict[str, float]: Normalized dictionary of country codes with weights
        """
        # Get all available country codes
        all_country_codes = phones.get_country_codes()

        # If no country codes provided, use default weights
        if not country_codes:
            # Set default weights - common countries get higher weights
            result = {
                "1": 0.4,  # US/Canada
                "44": 0.2,  # UK
                "7": 0.1,  # Russia
                "49": 0.05,  # Germany
                "33": 0.05,  # France
                "86": 0.05,  # China
                "81": 0.05,  # Japan
                "61": 0.05,  # Australia
                "91": 0.05,  # India
            }
            return result

        # If provided as list, convert to dict with equal weights
        if isinstance(country_codes, list):
            total = len(country_codes)
            if total == 0:
                return {"1": 1.0}  # Default to US if empty list

            weight = 1.0 / total
            result = {}

            # Convert country codes to actual dialing codes if needed
            for code in country_codes:
                # If it's a country name (e.g., "us"), get the actual dial code
                if isinstance(code, str) and code.lower() in all_country_codes:
                    dial_code = all_country_codes[code.lower()]
                    result[dial_code] = weight
                else:
                    # Assume it's already a dial code
                    result[str(code)] = weight

            return result

        # If provided as dict, normalize weights
        if isinstance(country_codes, dict):
            result = {}
            total_weight = sum(country_codes.values())

            # Convert country codes and normalize weights
            for code, weight in country_codes.items():
                # If it's a country name, get the actual dial code
                if isinstance(code, str) and code.lower() in all_country_codes:
                    dial_code = all_country_codes[code.lower()]
                    result[dial_code] = weight / total_weight if total_weight > 0 else 0
                else:
                    # Assume it's already a dial code
                    result[str(code)] = weight / total_weight if total_weight > 0 else 0

            return result

        # Fallback to default
        return {"1": 1.0}  # Default to US

    def _load_operator_codes(self) -> Dict[str, List[str]]:
        """
        Load operator codes from dictionary or embedded source.

        Returns:
            Dict[str, List[str]]: Dictionary of operator codes by country code
        """
        # Initialize empty dictionary for operator codes
        result_operator_codes: Dict[str, List[str]] = {}

        # First try to load from provided dictionary path
        if self.operator_codes_dict:
            try:
                lines = dict_helpers.load_dictionary_from_text(self.operator_codes_dict)

                # Parse the dictionary format (e.g., "+1,201,917,646")
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        country_code = parts[0].strip().lstrip('+')
                        codes = [code.strip() for code in parts[1:] if code.strip()]

                        if country_code not in result_operator_codes:
                            result_operator_codes[country_code] = []

                        result_operator_codes[country_code].extend(codes)
            except Exception as e:
                print(f"Error loading operator codes from {self.operator_codes_dict}: {e}")

        # If no codes loaded or provided, use embedded defaults
        if not result_operator_codes:
            # Explicitly specify the return type
            country_codes_dict: Dict[str, str] = phones.get_country_codes()

            # Map country names to dial codes
            for country_name in country_codes_dict:
                # Use safe dictionary access
                dial_code = country_codes_dict.get(country_name, "")
                # Get operator codes for this country
                country_operators = phones.get_operator_codes(country_name)

                if country_operators:
                    if dial_code not in result_operator_codes:
                        result_operator_codes[dial_code] = []
                    result_operator_codes[dial_code].extend(country_operators)

        return result_operator_codes

    def validate_phone(self, phone: str) -> bool:
        """
        Validate phone number format.

        Args:
            phone: Phone number to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not phone or not isinstance(phone, str):
            return False

        # Check basic format
        if not self.phone_pattern.match(phone):
            return False

        # Extract digits
        digits = ''.join(self.digits_pattern.findall(phone))

        # Check minimum length
        if len(digits) < 7:  # Minimum reasonable phone number length
            return False

        # Check maximum length
        if len(digits) > 15:  # Maximum length according to E.164
            return False

        return True

    def extract_country_code(self, phone: str) -> Optional[str]:
        """
        Extract country code from phone number.

        Args:
            phone: Phone number

        Returns:
            Optional[str]: Country code or None if not found
        """
        if not phone or not isinstance(phone, str):
            return None

        # Clean phone number to digits only
        digits = ''.join(self.digits_pattern.findall(phone))

        # Try to identify country code
        if digits.startswith('1') and len(digits) >= 11:  # US/Canada
            return '1'
        elif digits.startswith('44') and len(digits) >= 12:  # UK
            return '44'
        elif digits.startswith('7') and len(digits) >= 11:  # Russia
            return '7'

        # Try common prefixes
        possible_country_codes = sorted(
            self._operator_codes.keys(),
            key=len,
            reverse=True  # Try longest codes first
        )

        for code in possible_country_codes:
            if digits.startswith(code):
                return code

        # Try with '+' prefix
        if phone.startswith('+'):
            digits_after_plus = ''.join(self.digits_pattern.findall(phone[1:]))

            for code in possible_country_codes:
                if digits_after_plus.startswith(code):
                    return code

        return None

    def extract_operator_code(self, phone: str, country_code: Optional[str] = None) -> Optional[str]:
        """
        Extract operator/area code from phone number.

        Args:
            phone: Phone number
            country_code: Country code (detected if not provided)

        Returns:
            Optional[str]: Operator code or None if not found
        """
        if not phone or not isinstance(phone, str):
            return None

        # Clean phone number to digits only
        digits = ''.join(self.digits_pattern.findall(phone))

        # Get country code if not provided
        if not country_code:
            country_code = self.extract_country_code(phone)

        if not country_code:
            return None

        # Remove country code from the beginning
        if digits.startswith(country_code):
            digits = digits[len(country_code):]

        # Get operator codes for this country
        if country_code not in self._operator_codes:
            return None

        operator_codes = self._operator_codes[country_code]

        # Try to match operator codes
        for code in sorted(operator_codes, key=len, reverse=True):  # Try longest codes first
            if digits.startswith(code):
                return code

        return None

    def generate_country_code(self, original_country_code: Optional[str] = None) -> str:
        """
        Generate country code based on configuration.

        Args:
            original_country_code: Original country code to possibly preserve

        Returns:
            str: Generated country code
        """
        # If original provided and preservation enabled, use it
        if self.preserve_country_code and original_country_code:
            return original_country_code

        # Otherwise select from available country codes
        country_codes = list(self.country_codes.keys())
        weights = list(self.country_codes.values())

        # Nothing to choose from
        if not country_codes:
            return self._get_country_code_for_region(self.default_country)

        # If using PRGN, ensure deterministic selection
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                original_country_code or "country_code",
                salt="country-code-selection"
            )

            # Select based on cumulative weights
            value = rng.random()
            cumulative = 0

            for i, weight in enumerate(weights):
                cumulative += weight
                if value <= cumulative:
                    return country_codes[i]

            # Fallback to last in list
            return country_codes[-1]

        # Use random weighted choice
        return random.choices(country_codes, weights=weights, k=1)[0]

    def _get_country_code_for_region(self, region: str) -> str:
        """
        Get country code for a region name.

        Args:
            region: Region name (e.g., "us", "ru")

        Returns:
            str: Country code
        """
        country_codes = phones.get_country_codes()
        region = region.lower()

        if region in country_codes:
            return country_codes[region]

        # Default to US
        return "1"

    def generate_operator_code(self, country_code: str, original_operator_code: Optional[str] = None) -> Optional[str]:
        """
        Generate operator code for a country.

        Args:
            country_code: Country code
            original_operator_code: Original operator code to possibly preserve

        Returns:
            Optional[str]: Generated operator code or None if not available
        """
        # If original provided and preservation enabled, use it if valid
        if self.preserve_operator_code and original_operator_code:
            # Check if this operator code is valid for the country
            if country_code in self._operator_codes:
                if original_operator_code in self._operator_codes[country_code]:
                    return original_operator_code

        # Get operator codes for this country
        if country_code not in self._operator_codes or not self._operator_codes[country_code]:
            return None

        operator_codes = self._operator_codes[country_code]

        # If using PRGN, ensure deterministic selection
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                original_operator_code or country_code,
                salt="operator-code-selection"
            )

            index = rng.randint(0, len(operator_codes) - 1)
            return operator_codes[index]

        # Use random selection
        return random.choice(operator_codes)

    def _generate_random_digits(self, length: int, original_value: Optional[str] = None) -> str:
        """
        Generate random digits.

        Args:
            length: Number of digits to generate
            original_value: Original value for deterministic generation

        Returns:
            str: Generated digits
        """
        if length <= 0:
            return ""

        # If using PRGN, ensure deterministic generation
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                original_value or f"random_digits_{length}",
                salt=f"digits-generation-{length}"
            )

            # Generate digits
            return ''.join(str(rng.randint(0, 9)) for _ in range(length))

        # Use random generation
        return ''.join(str(random.randint(0, 9)) for _ in range(length))

    def format_phone(self, country_code: str, operator_code: Optional[str],
                     number: str, format_template: Optional[str] = None) -> str:
        """
        Format phone number components according to template.

        Args:
            country_code: Country code
            operator_code: Operator code
            number: Remaining digits
            format_template: Format template to use

        Returns:
            str: Formatted phone number
        """
        # Get the format to use
        if not format_template:
            if self.format:
                format_template = self.format
            else:
                # Get format for country
                country_name = self._get_country_name_for_code(country_code)
                format_template = phones.get_phone_format_for_country(country_name)

        # Default to international format with spaces if no format found
        if not format_template:
            format_template = "+CC AAA XXX XXX"

        # Replace CC placeholder with country code
        result = format_template.replace("CC", country_code)

        # Replace operator code placeholder if available
        if operator_code:
            # Handle different area code placeholders
            if "AAA" in result:
                result = result.replace("AAA", operator_code)
            elif "AAAA" in result:
                # Pad or truncate to 4 digits if needed
                op_code = operator_code.ljust(4, '0') if len(operator_code) < 4 else operator_code[:4]
                result = result.replace("AAAA", op_code)
            elif "AA" in result:
                # Pad or truncate to 2 digits if needed
                op_code = operator_code.ljust(2, '0') if len(operator_code) < 2 else operator_code[:2]
                result = result.replace("AA", op_code)
            elif "A" in result:
                # Use first digit
                result = result.replace("A", operator_code[0])
        else:
            # If no operator code, replace placeholders with appropriate number of digits from number
            if "AAA" in result:
                result = result.replace("AAA", number[:3])
                number = number[3:]
            elif "AAAA" in result:
                result = result.replace("AAAA", number[:4])
                number = number[4:]
            elif "AA" in result:
                result = result.replace("AA", number[:2])
                number = number[2:]
            elif "A" in result:
                result = result.replace("A", number[:1])
                number = number[1:]

        # Replace number placeholders with remaining digits
        remaining_digits = number
        index = 0

        # Replace XXX placeholders
        while "XXX" in result and index + 3 <= len(remaining_digits):
            result = result.replace("XXX", remaining_digits[index:index + 3], 1)
            index += 3

        # Replace XX placeholders
        while "XX" in result and index + 2 <= len(remaining_digits):
            result = result.replace("XX", remaining_digits[index:index + 2], 1)
            index += 2

        # Replace X placeholders
        while "X" in result and index < len(remaining_digits):
            result = result.replace("X", remaining_digits[index], 1)
            index += 1

        # If there are still X placeholders, fill with random digits
        if "X" in result:
            result = re.sub(r'X+', lambda m: self._generate_random_digits(len(m.group())), result)

        return result

    def _get_country_name_for_code(self, country_code: str) -> str:
        """
        Get country name for a country code.

        Args:
            country_code: Country code (e.g., "1", "7")

        Returns:
            str: Country name (e.g., "us", "ru")
        """
        country_codes = phones.get_country_codes()

        # Find country with this code
        for country, code in country_codes.items():
            if code == country_code:
                return country

        # Defaults
        if country_code == "1":
            return "us"
        elif country_code == "44":
            return "uk"
        elif country_code == "7":
            return "ru"

        # Unknown, use default region
        return self.default_country

    def _determine_phone_length(self, country_code: str) -> int:
        """
        Determine appropriate phone number length for a country.

        Args:
            country_code: Country code

        Returns:
            int: Appropriate length for phone number digits
        """
        country_name = self._get_country_name_for_code(country_code)
        length_ranges = phones.get_phone_length_ranges()

        # Get range for this country or default
        range_tuple = length_ranges.get(country_name, length_ranges.get("default", (10, 10)))
        min_len, max_len = range_tuple

        # If using PRGN, deterministically pick a length
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                country_code,
                salt="phone-length-determination"
            )

            return rng.randint(min_len, max_len)

        # Randomly pick a length in the range
        return random.randint(min_len, max_len)

    def generate_phone_number(self, country_code: Optional[str] = None,
                              operator_code: Optional[str] = None,
                              original_number: Optional[str] = None) -> str:
        """
        Generate a complete phone number.

        Args:
            country_code: Country code (generated if None)
            operator_code: Operator code (generated if None)
            original_number: Original number for deterministic generation

        Returns:
            str: Generated phone number
        """
        # Generate country code if not provided
        if not country_code:
            original_country = None
            if original_number:
                original_country = self.extract_country_code(original_number)

            country_code = self.generate_country_code(original_country)

        # Generate operator code if not provided
        if not operator_code:
            original_operator = None
            if original_number:
                original_operator = self.extract_operator_code(original_number, country_code)

            operator_code = self.generate_operator_code(country_code, original_operator)

        # Determine how many more digits we need based on country
        total_length = self._determine_phone_length(country_code)

        # Subtract operator code length
        op_len = len(operator_code) if operator_code else 0
        remaining_length = max(0, total_length - op_len)

        # Generate remaining digits
        digits = self._generate_random_digits(remaining_length, original_number)

        # Format the phone number
        return self.format_phone(country_code, operator_code, digits)

    def generate(self, count: int, **params) -> List[str]:
        """
        Generate specified number of synthetic phone numbers.

        Args:
            count: Number of values to generate
            **params: Additional parameters including:
                - country_code: Specific country code to use
                - operator_code: Specific operator code to use
                - format: Override configured format
                - region: Region for formatting

        Returns:
            List[str]: Generated phone numbers
        """
        result = []

        for _ in range(count):
            phone = self._generate_phone(**params)
            result.append(phone)

        return result

    def _generate_phone(self, **params) -> str:
        """
        Generate a single phone number based on parameters.

        Args:
            **params: Parameters including:
                - country_code: Specific country code to use
                - operator_code: Specific operator code to use
                - format: Override configured format
                - region: Region for formatting

        Returns:
            str: Generated phone number
        """
        # Extract parameters
        country_code = params.get("country_code")
        operator_code = params.get("operator_code")
        format_template = params.get("format")
        region = params.get("region")

        # If region provided, get corresponding country code
        if region and not country_code:
            country_code = self._get_country_code_for_region(region)

        # Generate the phone number
        return self.generate_phone_number(country_code, operator_code)

    def generate_like(self, original_value: str, **params) -> str:
        """
        Generate a synthetic phone number similar to the original one.

        Args:
            original_value: Original phone number
            **params: Additional parameters

        Returns:
            str: Generated phone number
        """
        # Check if the original value is empty or None
        if original_value is None or original_value == "":
            return ""

        # Validate the original phone if validation is enabled
        is_valid = False
        if self.validate_source:
            is_valid = self.validate_phone(original_value)
        else:
            # If validation is disabled, treat as valid if it has digits
            is_valid = bool(self.digits_pattern.search(original_value))

        # For valid phones, try to preserve characteristics
        if is_valid:
            # Extract country and operator codes
            country_code = self.extract_country_code(original_value)
            operator_code = self.extract_operator_code(original_value, country_code)

            # Update parameters for generation
            gen_params = params.copy()

            # Only use extracted values if we should preserve them
            if self.preserve_country_code and country_code:
                gen_params["country_code"] = country_code

            if self.preserve_operator_code and operator_code:
                gen_params["operator_code"] = operator_code

            # Generate new phone with preserved characteristics
            return self.generate_phone_number(
                gen_params.get("country_code"),
                gen_params.get("operator_code"),
                original_value
            )

        # Handle invalid phone according to configuration
        if not is_valid:
            if self.handle_invalid_phone == "keep_empty":
                return ""
            elif self.handle_invalid_phone == "generate_with_default_country":
                # Generate with default country code
                return self.generate_phone_number(
                    self._get_country_code_for_region(self.default_country)
                )
            else:  # generate_new (default)
                # Generate completely new phone
                return self.generate_phone_number()

        # Fallback to default generation
        return self.generate_phone_number()

    def transform(self, values: List[str], **params) -> List[str]:
        """
        Transform a list of original values into synthetic ones.

        Args:
            values: List of original values
            **params: Additional parameters for generation

        Returns:
            List[str]: List of transformed values
        """
        return [self.generate_like(value, **params) for value in values]

    def validate(self, value: str) -> bool:
        """
        Check if a value is a valid phone number.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return self.validate_phone(value)