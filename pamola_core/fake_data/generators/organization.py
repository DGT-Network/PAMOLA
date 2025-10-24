"""
Organization name generation for fake data system.

This module provides the OrganizationGenerator class for generating synthetic
organization names while preserving statistical properties of the original data
and supporting consistent mapping with regional specificity.
"""

import random
import re
from typing import Dict, Any, List, Optional

from pamola_core.fake_data.commons import dict_helpers
from pamola_core.fake_data.commons.prgn import PRNGenerator
from pamola_core.fake_data.dictionaries import organizations
from pamola_core.fake_data.generators.base_generator import BaseGenerator


class OrganizationGenerator(BaseGenerator):
    """
    Generator for synthetic organization names.

    Generates organization names of various types (general, educational,
    government, etc.) with optional prefixes and suffixes, supporting
    regional specificity and consistent mapping.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize organization generator with configuration.

        Args:
            config: Configuration parameters including:
                - organization_type: Type of organization to generate
                - dictionaries: Paths to dictionaries with organization names
                - prefixes: Paths to dictionaries with prefixes
                - suffixes: Paths to dictionaries with suffixes
                - add_prefix_probability: Probability of adding a prefix
                - add_suffix_probability: Probability of adding a suffix
                - key: Key for PRGN
                - context_salt: Salt for PRGN
                - region: Region code for organization naming conventions
                - preserve_type: Whether to preserve organization type
                - industry: Specific industry for 'industry' type
        """
        super().__init__(config)

        # Store config in attributes for easy access
        self.organization_type = self.config.get('organization_type', 'general')
        self.dictionaries = self.config.get('dictionaries', {})
        self.prefixes_dict = self.config.get('prefixes', {})
        self.suffixes_dict = self.config.get('suffixes', {})
        self.add_prefix_probability = self.config.get('add_prefix_probability', 0.3)
        self.add_suffix_probability = self.config.get('add_suffix_probability', 0.5)
        self.region = self.config.get('region', 'en')
        self.preserve_type = self.config.get('preserve_type', True)
        self.industry = self.config.get('industry')

        # Load organization names for each type
        self._org_names = self._load_organization_names()

        # Load prefixes and suffixes
        self._prefixes = self._load_prefixes()
        self._suffixes = self._load_suffixes()

        # Set up PRGN generator if needed
        self.prgn_generator = None
        key = self.config.get('key')
        if key:
            self.prgn_generator = PRNGenerator(global_seed=key)

        # Precompile patterns for type detection
        self._type_patterns = self._compile_type_patterns()

        # Dictionary for storing dictionary info
        self._dictionary_info = None

    def _load_organization_names(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load organization names from dictionaries.

        Returns:
            Dict with organization names by type and region
        """
        result = {}

        # Define types to load
        org_types = ['general', 'educational', 'manufacturing', 'government', 'industry']
        if self.industry:
            # Add specific industry to the list if provided
            org_types.append(self.industry)

        # Load each organization type
        for org_type in org_types:
            result[org_type] = {}

            # Try to load from provided dictionaries
            dict_path = None
            if org_type in self.dictionaries:
                dict_path = self.dictionaries[org_type]

            # Load from provided dictionary or fallback to embedded
            if dict_path:
                try:
                    names = dict_helpers.load_dictionary_from_text(dict_path)
                    # Process names by region based on naming conventions
                    for name in names:
                        region_code = self._determine_region_from_name(name, org_type)
                        if region_code not in result[org_type]:
                            result[org_type][region_code] = []
                        result[org_type][region_code].append(name)
                except Exception as e:
                    print(f"Error loading organization dictionary from {dict_path}: {e}")

            # If no names loaded or dictionary path not provided, use embedded
            if not result[org_type]:
                # Map org_type to the appropriate function in organizations module
                if org_type == 'educational':
                    for country in ['US', 'RU', 'GB']:
                        names = organizations.get_educational_institutions(country)
                        region = country.lower() if country != 'US' else 'en'
                        if region not in result[org_type]:
                            result[org_type][region] = []
                        result[org_type][region].extend(names)
                elif org_type == 'government':
                    for country in ['US', 'RU']:
                        names = organizations.get_government_organizations(country)
                        region = country.lower() if country != 'US' else 'en'
                        if region not in result[org_type]:
                            result[org_type][region] = []
                        result[org_type][region].extend(names)
                elif org_type == 'industry' and self.industry:
                    for country in ['US', 'RU']:
                        names = organizations.get_business_organizations(country, self.industry)
                        region = country.lower() if country != 'US' else 'en'
                        if region not in result[org_type]:
                            result[org_type][region] = []
                        result[org_type][region].extend(names)
                else:
                    # General business organizations
                    for country in ['US', 'RU']:
                        names = organizations.get_business_organizations(country)
                        region = country.lower() if country != 'US' else 'en'
                        if region not in result[org_type]:
                            result[org_type][region] = []
                        result[org_type][region].extend(names)

        # Store dictionary info for metrics
        self._update_dictionary_info(result)

        return result

    def _update_dictionary_info(self, loaded_dict: Dict[str, Dict[str, List[str]]]):
        """
        Update dictionary information for metrics.

        Args:
            loaded_dict: Loaded dictionary data
        """
        self._dictionary_info = {
            "types": {},
            "regions": {},
            "total_names": 0
        }

        for org_type, regions in loaded_dict.items():
            type_count = sum(len(names) for names in regions.values())
            self._dictionary_info["types"][org_type] = type_count

            for region, names in regions.items():
                if region not in self._dictionary_info["regions"]:
                    self._dictionary_info["regions"][region] = 0
                self._dictionary_info["regions"][region] += len(names)
                self._dictionary_info["total_names"] += len(names)

    def _determine_region_from_name(self, name: str, org_type: str) -> str:
        """
        Determine region code from organization name.

        Args:
            name: Organization name
            org_type: Organization type

        Returns:
            Region code (e.g., 'en', 'ru')
        """
        # Simple heuristic for determining region from name
        # This could be improved with more sophisticated language detection

        # Check for Cyrillic characters (Russian)
        if re.search(r'[А-Яа-я]', name):
            return 'ru'

        # Check for typical Vietnamese characters
        if re.search(r'[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỂưạảấầẩẫậắằẳẵặẹẻẽềể]', name):
            return 'vn'

        # Default to English
        return 'en'

    def _load_prefixes(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load organization prefixes.

        Returns:
            Dict with prefixes by type and region
        """
        result = {}

        # Define organization types to load prefixes for
        org_types = ['general', 'educational', 'manufacturing', 'government', 'industry']
        if self.industry:
            org_types.append(self.industry)

        # Load prefixes for each type
        for org_type in org_types:
            result[org_type] = {}

            # Try to load from provided dictionary
            dict_path = None
            if org_type in self.prefixes_dict:
                dict_path = self.prefixes_dict[org_type]

            # Load from provided dictionary if available
            if dict_path:
                try:
                    prefixes = dict_helpers.load_dictionary_from_text(dict_path)
                    # Process prefixes by region
                    for prefix in prefixes:
                        region_code = self._determine_region_from_name(prefix, org_type)
                        if region_code not in result[org_type]:
                            result[org_type][region_code] = []
                        result[org_type][region_code].append(prefix)
                except Exception as e:
                    print(f"Error loading prefixes dictionary from {dict_path}: {e}")

            # If no prefixes loaded or dictionary path not provided, use default prefixes
            if not result[org_type]:
                # Define default prefixes by type and region
                default_prefixes = {
                    'general': {
                        'en': ['Global', 'United', 'International', 'American', 'Advanced', 'Professional',
                               'Strategic'],
                        'ru': ['Глобал', 'Объединенный', 'Международный', 'Российский', 'Передовой', 'Профессиональный']
                    },
                    'educational': {
                        'en': ['National', 'State', 'Community', 'Public', 'Private', 'Regional'],
                        'ru': ['Национальный', 'Государственный', 'Общественный', 'Публичный', 'Частный',
                               'Региональный']
                    },
                    'manufacturing': {
                        'en': ['Industrial', 'Manufacturing', 'Production', 'Engineering', 'Technical'],
                        'ru': ['Промышленный', 'Производственный', 'Инженерный', 'Технический']
                    },
                    'government': {
                        'en': ['Federal', 'State', 'National', 'Regional', 'Municipal'],
                        'ru': ['Федеральный', 'Национальный', 'Региональный', 'Муниципальный', 'Городской']
                    },
                    'industry': {
                        'en': ['Leading', 'Premier', 'Elite', 'Superior', 'Innovative'],
                        'ru': ['Ведущий', 'Премиум', 'Элитный', 'Передовой', 'Инновационный']
                    }
                }

                # Add specific industry prefixes if needed
                if self.industry and self.industry not in default_prefixes:
                    default_prefixes[self.industry] = default_prefixes['industry']

                # Copy default prefixes to result
                if org_type in default_prefixes:
                    for region, prefixes in default_prefixes[org_type].items():
                        if region not in result[org_type]:
                            result[org_type][region] = []
                        result[org_type][region].extend(prefixes)

        return result

    def _load_suffixes(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load organization suffixes.

        Returns:
            Dict with suffixes by type and region
        """
        result = {}

        # Define organization types to load suffixes for
        org_types = ['general', 'educational', 'manufacturing', 'government', 'industry']
        if self.industry:
            org_types.append(self.industry)

        # Load suffixes for each type
        for org_type in org_types:
            result[org_type] = {}

            # Try to load from provided dictionary
            dict_path = None
            if org_type in self.suffixes_dict:
                dict_path = self.suffixes_dict[org_type]

            # Load from provided dictionary if available
            if dict_path:
                try:
                    suffixes = dict_helpers.load_dictionary_from_text(dict_path)
                    # Process suffixes by region
                    for suffix in suffixes:
                        region_code = self._determine_region_from_name(suffix, org_type)
                        if region_code not in result[org_type]:
                            result[org_type][region_code] = []
                        result[org_type][region_code].append(suffix)
                except Exception as e:
                    print(f"Error loading suffixes dictionary from {dict_path}: {e}")

            # If no suffixes loaded or dictionary path not provided, use default suffixes
            if not result[org_type]:
                # Define default suffixes by type and region
                default_suffixes = {
                    'general': {
                        'en': ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Limited', 'Group', 'Holdings', 'International',
                               'Worldwide'],
                        'ru': ['ООО', 'ЗАО', 'ОАО', 'НКО', 'АО', 'Групп', 'Холдинг', 'Интернэшнл']
                    },
                    'educational': {
                        'en': ['University', 'College', 'School', 'Institute', 'Academy'],
                        'ru': ['Университет', 'Институт', 'Школа', 'Академия', 'Колледж']
                    },
                    'manufacturing': {
                        'en': ['Manufacturing', 'Industries', 'Production', 'Factory', 'Works'],
                        'ru': ['Производство', 'Завод', 'Фабрика', 'Промышленность']
                    },
                    'government': {
                        'en': ['Agency', 'Department', 'Authority', 'Commission', 'Office'],
                        'ru': ['Министерство', 'Департамент', 'Агентство', 'Комиссия', 'Служба']
                    },
                    'industry': {
                        'en': ['Solutions', 'Systems', 'Services', 'Consulting', 'Technologies'],
                        'ru': ['Решения', 'Системы', 'Сервис', 'Консалтинг', 'Технологии']
                    }
                }

                # Add specific industry suffixes if needed
                if self.industry and self.industry not in default_suffixes:
                    default_suffixes[self.industry] = default_suffixes['industry']

                # Copy default suffixes to result
                if org_type in default_suffixes:
                    for region, suffixes in default_suffixes[org_type].items():
                        if region not in result[org_type]:
                            result[org_type][region] = []
                        result[org_type][region].extend(suffixes)

        return result

    def _compile_type_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for detecting organization types.

        Returns:
            Dict of compiled regex patterns
        """
        patterns = {
            'educational': re.compile(r'(?i)(university|college|school|academy|institute|образователь|университет|'
                                      r'школа|академия|институт|учебный|училище|гимназия)'),
            'manufacturing': re.compile(r'(?i)(factory|manufacturing|industries|mill|завод|фабрика|'
                                        r'производств|промышлен|индустри)'),
            'government': re.compile(r'(?i)(ministry|department|agency|bureau|committee|commission|'
                                     r'министерство|департамент|агентство|комитет|комиссия|государствен)'),
            'industry': re.compile(r'(?i)(technologies|solutions|systems|consulting|group|holding|'
                                   r'технологии|решения|системы|консалтинг|групп|холдинг)')
        }

        # Add specific industry pattern if provided
        if self.industry:
            industry_terms = self.industry.split('_')
            pattern_str = '|'.join(industry_terms)
            patterns[self.industry] = re.compile(f'(?i)({pattern_str})')

        return patterns

    def detect_organization_type(self, org_name: str) -> str:
        """
        Detect organization type from its name.

        Args:
            org_name: Organization name

        Returns:
            Detected organization type
        """
        if not org_name:
            return 'general'

        # Check each pattern
        for org_type, pattern in self._type_patterns.items():
            if pattern.search(org_name):
                return org_type

        # Default to general if no pattern matches
        return 'general'

    def generate_organization_name(self, org_type: Optional[str] = None,
                                   region: Optional[str] = None) -> str:
        """
        Generate a random organization name.

        Args:
            org_type: Type of organization to generate
            region: Region code for naming conventions

        Returns:
            Generated organization name
        """
        # Use default type if not provided
        if not org_type:
            org_type = self.organization_type

        # For 'industry' type with specific industry
        if org_type == 'industry' and self.industry:
            org_type = self.industry

        # Use default region if not provided
        if not region:
            region = self.region
        region = region.lower()

        # Fall back to 'general' if no organizations of the requested type/region
        if (org_type not in self._org_names or
                region not in self._org_names[org_type] or
                not self._org_names[org_type][region]):

            # Try general type with the same region
            if 'general' in self._org_names and region in self._org_names['general']:
                org_type = 'general'
            # Try the requested type with 'en' region
            elif org_type in self._org_names and 'en' in self._org_names[org_type]:
                region = 'en'
            # Fall back to general/en as last resort
            elif 'general' in self._org_names and 'en' in self._org_names['general']:
                org_type = 'general'
                region = 'en'
            else:
                # If all else fails, return a generic name
                return f"Organization {random.randint(1000, 9999)}"

        # Select a base name
        names = self._org_names[org_type][region]

        # If using PRGN, ensure deterministic selection
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(self.original_value, salt=self.context_salt)
            index = rng.randint(0, len(names) - 1)
            name = names[index]
        else:
            # Random selection
            name = random.choice(names)

        return name

    def add_prefix(self, name: str, org_type: Optional[str] = None,
                   region: Optional[str] = None) -> str:
        """
        Add a prefix to an organization name.

        Args:
            name: Base organization name
            org_type: Organization type
            region: Region code

        Returns:
            Organization name with prefix
        """
        # Use default type if not provided
        if not org_type:
            org_type = self.organization_type

        # For 'industry' type with specific industry
        if org_type == 'industry' and self.industry:
            org_type = self.industry

        # Use default region if not provided
        if not region:
            region = self.region

        # Get prefixes for this type/region, falling back as needed
        prefixes = []

        # Try exact type/region match
        if org_type in self._prefixes and region in self._prefixes[org_type]:
            prefixes = self._prefixes[org_type][region]
        # Try general type with same region
        elif 'general' in self._prefixes and region in self._prefixes['general']:
            prefixes = self._prefixes['general'][region]
        # Try requested type with 'en' region
        elif org_type in self._prefixes and 'en' in self._prefixes[org_type]:
            prefixes = self._prefixes[org_type]['en']
        # Fall back to general/en
        elif 'general' in self._prefixes and 'en' in self._prefixes['general']:
            prefixes = self._prefixes['general']['en']

        # If no prefixes available, return original name
        if not prefixes:
            return name

        # Select a prefix
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                name,
                salt="org-prefix-selection"
            )
            index = rng.randint(0, len(prefixes) - 1)
            prefix = prefixes[index]
        else:
            # Random selection
            prefix = random.choice(prefixes)

        # Add space if needed
        if not prefix.endswith(' '):
            prefix += ' '

        return f"{prefix}{name}"

    def add_suffix(self, name: str, org_type: Optional[str] = None,
                   region: Optional[str] = None) -> str:
        """
        Add a suffix to an organization name.

        Args:
            name: Base organization name
            org_type: Organization type
            region: Region code

        Returns:
            Organization name with suffix
        """
        # Use default type if not provided
        if not org_type:
            org_type = self.organization_type

        # For 'industry' type with specific industry
        if org_type == 'industry' and self.industry:
            org_type = self.industry

        # Use default region if not provided
        if not region:
            region = self.region

        # Get suffixes for this type/region, falling back as needed
        suffixes = []

        # Try exact type/region match
        if org_type in self._suffixes and region in self._suffixes[org_type]:
            suffixes = self._suffixes[org_type][region]
        # Try general type with same region
        elif 'general' in self._suffixes and region in self._suffixes['general']:
            suffixes = self._suffixes['general'][region]
        # Try requested type with 'en' region
        elif org_type in self._suffixes and 'en' in self._suffixes[org_type]:
            suffixes = self._suffixes[org_type]['en']
        # Fall back to general/en
        elif 'general' in self._suffixes and 'en' in self._suffixes['general']:
            suffixes = self._suffixes['general']['en']

        # If no suffixes available, return original name
        if not suffixes:
            return name

        # Select a suffix
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                name,
                salt="org-suffix-selection"
            )
            index = rng.randint(0, len(suffixes) - 1)
            suffix = suffixes[index]
        else:
            # Random selection
            suffix = random.choice(suffixes)

        # Add space if needed
        if not suffix.startswith(' '):
            suffix = f" {suffix}"

        return f"{name}{suffix}"

    def get_dictionary_info(self) -> Dict[str, Any]:
        """
        Get information about the dictionaries used by the generator.

        Returns:
            Dictionary information for metrics
        """
        return self._dictionary_info

    def validate_organization_name(self, name: str) -> bool:
        """
        Validate an organization name format.

        Args:
            name: Organization name to validate

        Returns:
            True if valid, False otherwise
        """
        if not name or not isinstance(name, str):
            return False

        # Basic validation - check minimum length
        if len(name) < 2:
            return False

        # Check for at least one alphabetic character
        if not any(c.isalpha() for c in name):
            return False

        return True

    def generate(self, count: int, **params) -> List[str]:
        """
        Generate specified number of synthetic organization names.

        Args:
            count: Number of names to generate
            **params: Additional parameters including:
                - organization_type: Type of organization
                - region: Region code
                - add_prefix: Whether to add prefix
                - add_suffix: Whether to add suffix

        Returns:
            List of generated organization names
        """
        result = []

        # Extract parameters
        org_type = params.get('organization_type', self.organization_type)
        region = params.get('region', self.region)
        add_prefix = params.get('add_prefix')
        add_suffix = params.get('add_suffix')

        # Default to probabilities if not explicitly specified
        if add_prefix is None:
            add_prefix = self.add_prefix_probability > 0
        if add_suffix is None:
            add_suffix = self.add_suffix_probability > 0

        for _ in range(count):
            # Generate base name
            name = self.generate_organization_name(org_type, region)

            # Add prefix with probability
            if add_prefix and (isinstance(add_prefix, bool) or
                               random.random() < self.add_prefix_probability):
                name = self.add_prefix(name, org_type, region)

            # Add suffix with probability
            if add_suffix and (isinstance(add_suffix, bool) or
                               random.random() < self.add_suffix_probability):
                name = self.add_suffix(name, org_type, region)

            result.append(name)

        return result

    def generate_like(self, original_value: str, **params) -> str:
        """
        Generate a synthetic organization name similar to the original one.

        Args:
            original_value: Original organization name
            **params: Additional parameters

        Returns:
            Generated organization name
        """
        # Check if the original value is empty or None
        if original_value is None or original_value == "":
            return ""

        # Determine organization type from original name if preservation enabled
        org_type = self.organization_type
        if self.preserve_type and original_value:
            detected_type = self.detect_organization_type(original_value)
            if detected_type:
                org_type = detected_type

        # Allow override via params
        org_type = params.get('organization_type', org_type)

        # Determine region from original name
        region = self.region
        if original_value:
            detected_region = self._determine_region_from_name(original_value, org_type)
            if detected_region:
                region = detected_region

        # Allow override via params
        region = params.get('region', region)

        self.original_value = original_value
        self.context_salt = params.get("context_salt", None)

        # Generate base name with detected properties
        new_name = self.generate_organization_name(org_type, region)

        # Get deterministic prefixes/suffixes based on original value
        add_prefix = False
        add_suffix = False

        # Decide on prefix based on configured probability or explicit parameter
        if 'add_prefix' in params:
            add_prefix = params['add_prefix']
        elif self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                original_value,
                salt="org-prefix-decision"
            )
            add_prefix = rng.random() < self.add_prefix_probability
        else:
            add_prefix = random.random() < self.add_prefix_probability

        # Decide on suffix based on configured probability or explicit parameter
        if 'add_suffix' in params:
            add_suffix = params['add_suffix']
        elif self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(
                original_value,
                salt="org-suffix-decision"
            )
            add_suffix = rng.random() < self.add_suffix_probability
        else:
            add_suffix = random.random() < self.add_suffix_probability

        # Add prefix if decided
        if add_prefix:
            new_name = self.add_prefix(new_name, org_type, region)

        # Add suffix if decided
        if add_suffix:
            new_name = self.add_suffix(new_name, org_type, region)

        return new_name

    def transform(self, values: List[str], **params) -> List[str]:
        """
        Transform a list of original values into synthetic ones.

        Args:
            values: List of original values
            **params: Additional parameters for generation

        Returns:
            List of transformed values
        """
        return [self.generate_like(value, **params) for value in values]

    def validate(self, value: str) -> bool:
        """
        Check if a value is a valid organization name.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        return self.validate_organization_name(value)