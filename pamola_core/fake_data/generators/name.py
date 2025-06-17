"""
Name generator for synthetic personal names.

This module provides functionality to generate synthetic person names
with support for different languages, gender awareness, and formatting options.
It uses dictionaries of names and supports consistent generation.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple

from faker import Faker

from pamola_core.fake_data.commons import dict_helpers
from pamola_core.fake_data.commons import utils
from pamola_core.fake_data.commons.prgn import PRNGenerator
from pamola_core.fake_data.generators.base_generator import BaseGenerator

# Set up logger
logger = logging.getLogger(__name__)

# Try to import Faker with graceful degradation
try:
    import faker

    FAKER_AVAILABLE = True
except ImportError:
    logger.warning("Faker library not available, falling back to dictionary-based generation")
    FAKER_AVAILABLE = False


class NameGenerator(BaseGenerator):
    """
    Generator for personal names with support for different languages,
    gender awareness, and formatting options.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize name generator with configuration.

        Args:
            config: Dictionary with configuration parameters
                - language: Language code (e.g., "ru", "en", "vi")
                - gender: Gender code ("M", "F", None for neutral)
                - format: Format for full names (e.g., "FML", "FL", "LF")
                - use_faker: Whether to use Faker library if available
                - case: Case formatting (upper, lower, title)
                - gender_from_name: Whether to infer gender from name
                - f_m_ratio: Ratio of female/male names for random generation
                - dictionaries: Paths to custom dictionaries
        """
        super().__init__(config)

        # Set default values
        self.language = self._normalize_language(self.config.get("language", "en"))
        self.gender = self.config.get("gender")
        self.format = self.config.get("format", "FL")
        self.use_faker = self.config.get("use_faker", False) and FAKER_AVAILABLE
        self.case = self.config.get("case", "title")
        self.gender_from_name = self.config.get("gender_from_name", False)
        self.f_m_ratio = self.config.get("f_m_ratio", 0.5)

        self.dictionaries = self.config.get("dictionaries", {})

        # Initialize PRN generator if provided in config
        self.prgn_generator = None
        if self.config.get("prgn_generator"):
            self.prgn_generator = self.config.get("prgn_generator")
        elif self.config.get("key"):
            seed = utils.hash_value(self.config.get("key"), self.config.get("context_salt", ""))
            self.prgn_generator = PRNGenerator(global_seed=seed)

        # Initialize mapping store if provided
        self.mapping_store = self.config.get("mapping_store")
        self.use_mapping = self.config.get("use_mapping", False)

        # Initialize Faker if requested
        self.faker = None
        if self.use_faker and FAKER_AVAILABLE:
            faker_locale = self._get_faker_locale(self.language)
            self.faker = Faker(faker_locale)

        # Load dictionaries or use default ones
        self._load_dictionaries()

        # Dictionary for cached values
        self._cache = {}

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language code to standard format.

        Args:
            language: Language code to normalize

        Returns:
            Normalized language code
        """
        # Create mapping of common language codes to our internal format
        language_mapping = {
            # Russian
            'ru_ru': 'ru', 'ru_RU': 'ru', 'rus': 'ru', 'russian': 'ru',
            # English
            'en_us': 'en', 'en_US': 'en', 'en_uk': 'en', 'en_UK': 'en',
            'en_gb': 'en', 'en_GB': 'en', 'eng': 'en', 'english': 'en',
            # Vietnamese
            'vn': 'vi', 'vi_VN': 'vi', 'vie': 'vi', 'vietnamese': 'vi'
        }

        # Normalize to lowercase
        lang_lower = language.lower() if language else 'en'

        # Return mapped or original value
        return language_mapping.get(lang_lower, lang_lower)

    def _get_faker_locale(self, language: str) -> str:
        """
        Convert internal language code to Faker locale.

        Args:
            language: Internal language code

        Returns:
            Faker locale string
        """
        faker_mapping = {
            'ru': 'ru_RU',
            'en': 'en_US',
            'vn': 'vi_VN'
        }
        return faker_mapping.get(language, 'en_US')

    def _load_dictionaries(self) -> None:
        """
        Load name dictionaries based on configuration.
        Attempts to load from specified paths first, then falls back to built-in dictionaries.
        """
        # Initialize dictionaries
        self._first_names_male = {}
        self._first_names_female = {}
        self._last_names = {}
        self._middle_names_male = {}
        self._middle_names_female = {}

        # Load dictionaries for supported languages
        for lang in ["en", "ru", "vi"]:
            # Try to load from explicit paths first
            dict_paths = self.dictionaries.get(lang, {})

            # Male first names
            self._first_names_male[lang] = dict_helpers.load_multi_dictionary(
                "name",
                {
                    "language": lang,
                    "gender": "M",
                    "name_type": "first_name",
                    "path": dict_paths.get("male_first_names")
                },
                fallback_to_embedded=True
            )

            # Female first names
            self._first_names_female[lang] = dict_helpers.load_multi_dictionary(
                "name",
                {
                    "language": lang,
                    "gender": "F",
                    "name_type": "first_name",
                    "path": dict_paths.get("female_first_names")
                },
                fallback_to_embedded=True
            )

            # Last names
            self._last_names[lang] = dict_helpers.load_multi_dictionary(
                "name",
                {
                    "language": lang,
                    "name_type": "last_name",
                    "path": dict_paths.get("last_names")
                },
                fallback_to_embedded=True
            )

            # Middle names (for languages that use them)
            if lang in ["ru"]:
                self._middle_names_male[lang] = dict_helpers.load_multi_dictionary(
                    "name",
                    {
                        "language": lang,
                        "gender": "M",
                        "name_type": "middle_name",
                        "path": dict_paths.get("male_middle_names")
                    },
                    fallback_to_embedded=True
                )

                self._middle_names_female[lang] = dict_helpers.load_multi_dictionary(
                    "name",
                    {
                        "language": lang,
                        "gender": "F",
                        "name_type": "middle_name",
                        "path": dict_paths.get("female_middle_names")
                    },
                    fallback_to_embedded=True
                )
            else:
                # Empty lists for languages without middle names
                self._middle_names_male[lang] = []
                self._middle_names_female[lang] = []

        # Log dictionary sizes for debugging
        for lang in ["en", "ru", "vi"]:
            logger.debug(f"Loaded dictionaries for {lang}:")
            logger.debug(f"  - Male first names: {len(self._first_names_male.get(lang, []))} items")
            logger.debug(f"  - Female first names: {len(self._first_names_female.get(lang, []))} items")
            logger.debug(f"  - Last names: {len(self._last_names.get(lang, []))} items")
            logger.debug(f"  - Male middle names: {len(self._middle_names_male.get(lang, []))} items")
            logger.debug(f"  - Female middle names: {len(self._middle_names_female.get(lang, []))} items")

    def generate(self, count: int, **params) -> List[str]:
        """
        Generate specified number of name values.

        Args:
            count: Number of names to generate
            **params: Additional parameters
                - gender: Override default gender
                - language: Override default language
                - format: Override default format
                - seed: Random seed for reproducibility

        Returns:
            List of generated names
        """
        gender = params.get("gender", self.gender)
        language = self._normalize_language(params.get("language", self.language))
        format_str = params.get("format", self.format)
        seed = params.get("seed")

        # Set random seed if provided
        if seed is not None and not self.prgn_generator:
            random.seed(seed)

        # Use Faker if available and requested
        if self.use_faker and self.faker:
            return self._generate_with_faker(count, gender, language, format_str)

        result = []
        for _ in range(count):
            # Determine gender for this name if not specified
            if gender is None:
                if random.random() < self.f_m_ratio:
                    curr_gender = "F"
                else:
                    curr_gender = "M"
            else:
                curr_gender = gender

            name = self.generate_full_name(curr_gender, language, format_str)
            result.append(name)

        return result

    def _generate_with_faker(self, count: int, gender: Optional[str],
                             language: str, format_str: str) -> List[str]:
        """
        Generate names using Faker library.

        Args:
            count: Number of names to generate
            gender: Gender for generation
            language: Language for generation
            format_str: Format string

        Returns:
            List of generated names
        """
        result = []

        # Set locale based on language
        locale = self._get_faker_locale(language)
        if locale != self.faker.locale:
            self.faker = Faker(locale)

        for _ in range(count):
            # Determine gender if not specified
            if gender is None:
                curr_gender = "F" if random.random() < self.f_m_ratio else "M"
            else:
                curr_gender = gender

            # Generate appropriate components
            if curr_gender == "M":
                first_name = self.faker.first_name_male()
                middle_name = ""
                if language == "ru":
                    middle_name = self.faker.middle_name_male()
                last_name = self.faker.last_name_male()
            else:
                first_name = self.faker.first_name_female()
                middle_name = ""
                if language == "ru":
                    middle_name = self.faker.middle_name_female()
                last_name = self.faker.last_name_female()

            # Format name according to format string
            name = self._format_name(first_name, middle_name, last_name, format_str)
            result.append(name)

        return result

    def generate_like(self, original_value: str, **params) -> str:
        """
        Generate a name similar to the original one.

        Args:
            original_value: Original name
            **params: Additional parameters
                - gender: Gender for generation
                - language: Language for generation
                - format: Format for generation
                - context_salt: Salt for PRGN generation

        Returns:
            Generated name
        """
        # Handle empty or None values
        if not original_value:
            return ""

        gender = params.get("gender", self.gender)
        language = self._normalize_language(params.get("language", self.language))
        format_str = params.get("format", self.format)
        context_salt = params.get("context_salt", "name-generation")

        # Try to detect gender from name if gender_from_name is True and gender not specified
        if gender is None and self.gender_from_name:
            detected_gender = self.detect_gender(original_value, language)
            if detected_gender:
                gender = detected_gender

        # Parse original value to determine format if not specified
        if format_str is None:
            parsed_name = self.parse_full_name(original_value, language)
            if parsed_name.get("middle_name"):
                format_str = "FML"
            else:
                format_str = "FL"

        # Check if we're using mapping store
        if self.use_mapping and self.mapping_store:
            field_name = params.get("field_name", "name")
            # Check if mapping already exists
            if hasattr(self.mapping_store, "get_mapping"):
                synthetic = self.mapping_store.get_mapping(field_name, original_value)
                if synthetic:
                    return synthetic

        # Check if using PRGN
        if self.prgn_generator:
            # Generate deterministically based on original value
            if gender == "M":
                dict_to_use = self._first_names_male.get(language, [])
            else:
                dict_to_use = self._first_names_female.get(language, [])

            # Fall back to English if dictionary is empty
            if not dict_to_use:
                if gender == "M":
                    dict_to_use = self._first_names_male.get("en", [])
                else:
                    dict_to_use = self._first_names_female.get("en", [])

            # Generate first name
            first_name = self.prgn_generator.select_from_list(
                dict_to_use,
                original_value,
                salt=f"{context_salt}-first-{gender}-{language}"
            )

            # Generate last name
            last_name_dict = self._last_names.get(language, [])
            if not last_name_dict:
                last_name_dict = self._last_names.get("en", [])

            last_name = self.prgn_generator.select_from_list(
                last_name_dict,
                original_value,
                salt=f"{context_salt}-last-{language}"
            )

            # Generate middle name if needed
            middle_name = ""
            if "M" in format_str and language in ["ru"]:
                if gender == "M":
                    middle_dict = self._middle_names_male.get(language, [])
                else:
                    middle_dict = self._middle_names_female.get(language, [])

                if middle_dict:
                    middle_name = self.prgn_generator.select_from_list(
                        middle_dict,
                        original_value,
                        salt=f"{context_salt}-middle-{gender}-{language}"
                    )

            # Format full name
            result = self._format_name(first_name, middle_name, last_name, format_str)

            # Store in mapping if using mapping store
            if self.use_mapping and self.mapping_store:
                field_name = params.get("field_name", "name")
                if hasattr(self.mapping_store, "add_mapping"):
                    self.mapping_store.add_mapping(field_name, original_value, result)

            return result

        # Otherwise, generate using standard method
        return self.generate_full_name(gender, language, format_str)

    def generate_first_name(self, gender: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Generate a first name.

        Args:
            gender: Gender for name generation (M/F)
            language: Language for name generation

        Returns:
            Generated first name
        """
        language = self._normalize_language(language or self.language)

        # Use Faker if available and requested
        if self.use_faker and self.faker:
            locale = self._get_faker_locale(language)
            if locale != self.faker.locale:
                self.faker = Faker(locale)

            if gender == "M":
                return self._apply_case(self.faker.first_name_male())
            elif gender == "F":
                return self._apply_case(self.faker.first_name_female())
            else:
                return self._apply_case(self.faker.first_name())

        # Determine gender if not specified
        if gender is None:
            if random.random() < self.f_m_ratio:
                gender = "F"
            else:
                gender = "M"

        # Select appropriate dictionary
        if gender == "M":
            names_dict = self._first_names_male.get(language, [])
        else:
            names_dict = self._first_names_female.get(language, [])

        # Fallback to English if dictionary is empty
        if not names_dict and language != "en":
            if gender == "M":
                names_dict = self._first_names_male.get("en", [])
            else:
                names_dict = self._first_names_female.get("en", [])

        # Generate name
        if names_dict:
            name = random.choice(names_dict)
            return self._apply_case(name)
        else:
            # Fallback to simple placeholder if no dictionary available
            return self._apply_case("John" if gender == "M" else "Jane")

    def generate_last_name(self, gender: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Generate a last name.

        Args:
            gender: Gender for name generation (might affect last name in some languages)
            language: Language for name generation

        Returns:
            Generated last name
        """
        language = self._normalize_language(language or self.language)

        # Use Faker if available and requested
        if self.use_faker and self.faker:
            locale = self._get_faker_locale(language)
            if locale != self.faker.locale:
                self.faker = Faker(locale)

            if gender == "M":
                return self._apply_case(self.faker.last_name_male())
            elif gender == "F":
                return self._apply_case(self.faker.last_name_female())
            else:
                return self._apply_case(self.faker.last_name())

        # Select appropriate dictionary
        names_dict = self._last_names.get(language, [])

        # Fallback to English if dictionary is empty
        if not names_dict and language != "en":
            names_dict = self._last_names.get("en", [])

        # Generate name
        if names_dict:
            name = random.choice(names_dict)
            return self._apply_case(name)
        else:
            # Fallback to simple placeholder if no dictionary available
            return self._apply_case("Smith")

    def generate_middle_name(self, gender: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Generate a middle name (or patronymic for languages that use them).

        Args:
            gender: Gender for name generation
            language: Language for name generation

        Returns:
            Generated middle name or empty string if not applicable
        """
        language = self._normalize_language(language or self.language)

        # Use Faker if available and requested
        if self.use_faker and self.faker and language == "ru":
            locale = self._get_faker_locale(language)
            if locale != self.faker.locale:
                self.faker = Faker(locale)

            if gender == "M":
                return self._apply_case(self.faker.middle_name_male())
            elif gender == "F":
                return self._apply_case(self.faker.middle_name_female())

        # Determine gender if not specified
        if gender is None:
            if random.random() < self.f_m_ratio:
                gender = "F"
            else:
                gender = "M"

        # Select appropriate dictionary
        if gender == "M":
            names_dict = self._middle_names_male.get(language, [])
        else:
            names_dict = self._middle_names_female.get(language, [])

        # Generate name if dictionary exists
        if names_dict:
            name = random.choice(names_dict)
            return self._apply_case(name)
        else:
            # Return empty string for languages without middle names
            return ""

    def generate_full_name(self, gender: Optional[str] = None,
                           language: Optional[str] = None,
                           format_str: Optional[str] = None) -> str:
        """
        Generate a full name according to specified format.

        Args:
            gender: Gender for name generation
            language: Language for name generation
            format_str: Format of the name:
                - FML: FirstName MiddleName LastName
                - FL: FirstName LastName
                - LF: LastName FirstName
                - LFM: LastName FirstName MiddleName
                - F_L: FirstName_LastName

                Uppercase/lowercase variants affect casing:
                - FML: Title Case
                - fml: lowercase
                - FML_: UPPERCASE

        Returns:
            Generated full name
        """
        language = self._normalize_language(language or self.language)
        format_str = format_str or self.format

        # Extract case information from format string
        format_str, case = self._parse_format_case(format_str)

        # Use Faker if available and requested
        if self.use_faker and self.faker:
            locale = self._get_faker_locale(language)
            if locale != self.faker.locale:
                self.faker = Faker(locale)

            # Generate components
            if gender == "M":
                first_name = self.faker.first_name_male()
                last_name = self.faker.last_name_male()
                middle_name = ""
                if language == "ru" and "M" in format_str:
                    middle_name = self.faker.middle_name_male()
            elif gender == "F":
                first_name = self.faker.first_name_female()
                last_name = self.faker.last_name_female()
                middle_name = ""
                if language == "ru" and "M" in format_str:
                    middle_name = self.faker.middle_name_female()
            else:
                first_name = self.faker.first_name()
                last_name = self.faker.last_name()
                middle_name = ""
                if language == "ru" and "M" in format_str:
                    # Default to male for non-specified gender
                    middle_name = self.faker.middle_name_male()

            return self._format_name(first_name, middle_name, last_name, format_str, case)

        # Generate name components
        first_name = self.generate_first_name(gender, language)
        last_name = self.generate_last_name(gender, language)

        # Generate middle name only if needed
        if "M" in format_str:
            middle_name = self.generate_middle_name(gender, language)
        else:
            middle_name = ""

        return self._format_name(first_name, middle_name, last_name, format_str, case)

    def _format_name(self, first_name: str, middle_name: str,
                     last_name: str, format_str: str, case: Optional[str] = None) -> str:
        """
        Format name components according to format string.

        Args:
            first_name: First name
            middle_name: Middle name
            last_name: Last name
            format_str: Format string
            case: Case format (upper, lower, title)

        Returns:
            Formatted full name
        """
        # Format name according to pattern
        if format_str == "FML":
            if middle_name:
                full_name = f"{first_name} {middle_name} {last_name}"
            else:
                full_name = f"{first_name} {last_name}"
        elif format_str == "FL":
            full_name = f"{first_name} {last_name}"
        elif format_str == "LF":
            full_name = f"{last_name} {first_name}"
        elif format_str == "LFM":
            if middle_name:
                full_name = f"{last_name} {first_name} {middle_name}"
            else:
                full_name = f"{last_name} {first_name}"
        elif format_str == "F_L":
            full_name = f"{first_name}_{last_name}"
        elif format_str == "L_F":
            full_name = f"{last_name}_{first_name}"
        else:
            # Default format
            full_name = f"{first_name} {last_name}"

        # Apply case
        return self._apply_case(full_name, case)

    def detect_gender(self, name: str, language: Optional[str] = None) -> Optional[str]:
        """
        Detect gender from a name.

        Args:
            name: Name to analyze
            language: Language of the name

        Returns:
            "M" for male, "F" for female, or None if undetermined
        """
        if not name:
            return None

        language = self._normalize_language(language or self.language)

        # Parse the name to extract first name
        parsed_name = self.parse_full_name(name, language)
        first_name = parsed_name.get("first_name", "")

        if not first_name:
            return None

        # Check against dictionaries
        first_name_lower = first_name.lower()

        male_names = [n.lower() for n in self._first_names_male.get(language, [])]
        if first_name_lower in male_names:
            return "M"

        female_names = [n.lower() for n in self._first_names_female.get(language, [])]
        if first_name_lower in female_names:
            return "F"

        # If language is not the default, try the default language
        if language != "en":
            male_names = [n.lower() for n in self._first_names_male.get("en", [])]
            if first_name_lower in male_names:
                return "M"

            female_names = [n.lower() for n in self._first_names_female.get("en", [])]
            if first_name_lower in female_names:
                return "F"

        # No match found
        return None

    def parse_full_name(self, full_name: str, language: Optional[str] = None) -> Dict[str, str]:
        """
        Parse a full name into components.

        Args:
            full_name: Full name to parse
            language: Language of the name

        Returns:
            Dictionary with components: first_name, middle_name, last_name
        """
        # Handle empty or None values
        if not full_name:
            return {
                "first_name": "",
                "middle_name": "",
                "last_name": ""
            }

        language = self._normalize_language(language or self.language)

        # Use the dictionary helper's parse_full_name if available
        try:
            parsed = dict_helpers.parse_full_name(full_name, language)
            return parsed
        except (AttributeError, ImportError):
            # Fallback to our own implementation
            parts = full_name.split()

            result = {
                "first_name": "",
                "middle_name": "",
                "last_name": ""
            }

            if not parts:
                return result

            # Simple heuristic parsing based on number of parts
            if len(parts) == 1:
                # Just one name, assume it's a first name
                result["first_name"] = parts[0]
            elif len(parts) == 2:
                # Two parts, assume first+last
                if language in ["ru", "vi"]:
                    # In Russian and Vietnamese, last name comes first
                    result["last_name"] = parts[0]
                    result["first_name"] = parts[1]
                else:
                    # In most Western languages, first name comes first
                    result["first_name"] = parts[0]
                    result["last_name"] = parts[1]
            elif len(parts) >= 3:
                # Three or more parts, assume first+middle+last
                if language == "ru":
                    # In Russian: last_name first_name middle_name
                    result["last_name"] = parts[0]
                    result["first_name"] = parts[1]
                    result["middle_name"] = " ".join(parts[2:])
                elif language in ["vn", "vi"]:
                    # In Vietnamese: last_name middle_name(s) first_name
                    result["last_name"] = parts[0]
                    result["first_name"] = parts[-1]
                    result["middle_name"] = " ".join(parts[1:-1])
                else:
                    # In most Western languages: first_name middle_name(s) last_name
                    result["first_name"] = parts[0]
                    result["last_name"] = parts[-1]
                    result["middle_name"] = " ".join(parts[1:-1])

            return result

    def _parse_format_case(self, format_str: str) -> Tuple[str, str]:
        """
        Parse format string to extract case information.

        Args:
            format_str: Format string with case information

        Returns:
            Tuple of (format, case)
        """
        # Default
        case = self.case

        # Check format modifiers
        if format_str.endswith("_"):
            # Uppercase
            case = "upper"
            format_str = format_str[:-1]
        elif format_str.islower():
            # Lowercase
            case = "lower"

        return format_str, case

    def _apply_case(self, text: str, case: Optional[str] = None) -> str:
        """
        Apply case formatting to text.

        Args:
            text: Text to format
            case: Case format (upper, lower, title)

        Returns:
            Formatted text
        """
        if not text:
            return ""

        case = case or self.case

        if case == "upper":
            return text.upper()
        elif case == "lower":
            return text.lower()
        elif case == "title":
            return " ".join(word.capitalize() for word in text.split())
        else:
            return text