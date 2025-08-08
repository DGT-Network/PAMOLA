"""
Email address generation for fake data system.

This module provides the EmailGenerator class for generating synthetic email addresses
while preserving statistical properties of the original data and supporting
consistent mapping.
"""

import random
import re
import string
from typing import Dict, Any, List, Optional

from pamola_core.fake_data.commons import dict_helpers
from pamola_core.fake_data.commons.prgn import PRNGenerator
from pamola_core.fake_data.generators.base_generator import BaseGenerator


class EmailGenerator(BaseGenerator):
    """
    Generator for synthetic email addresses.

    Generates email addresses in various formats, optionally preserving
    domain information from original addresses and supporting consistent mapping.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize email generator with configuration.

        Args:
            config: Configuration parameters including:
                - domains: List of domains or path to domain dictionary
                - format: Format for email generation (name_surname, surname_name, nickname, existing_domain)
                - format_ratio: Distribution of format usage
                - validate_source: Whether to validate source email addresses
                - handle_invalid_email: How to handle invalid emails
                - nicknames_dict: Path to nickname dictionary
                - max_length: Maximum length for email address (default: 254)
                - max_local_part_length: Maximum length for local part (default: 64)
                - max_domain_length: Maximum length for domain part (default: 255)
                - separator_options: List of separators to use (default: [".", "_", "-", ""])
                - number_suffix_probability: Probability of adding number suffix (default: 0.4)
                - preserve_domain_ratio: Probability of preserving original domain (default: 0.5)
                - business_domain_ratio: Probability of using business domains (default: 0.2)
                - key: Key for PRGN
                - context_salt: Salt for PRGN
        """
        super().__init__(config)

        # Store config in attributes for easy access
        self.domains = self.config.get('domains', [])
        self.format = self.config.get('format')
        self.format_ratio = self.config.get('format_ratio', {})
        self.validate_source = self.config.get('validate_source', True)
        self.handle_invalid_email = self.config.get('handle_invalid_email', 'generate_new')
        self.nicknames_dict = self.config.get('nicknames_dict')
        self.max_length = self.config.get('max_length', 254)  # RFC 5321 SMTP limit
        self.max_local_part_length = self.config.get('max_local_part_length', 64)  # RFC 5321 limit
        self.max_domain_length = self.config.get('max_domain_length', 255)  # RFC 1035 limit

        # New configuration parameters
        self.separator_options = self.config.get('separator_options') or [".", "_", "-", ""]
        self.number_suffix_probability = self.config.get('number_suffix_probability', 0.4)
        self.preserve_domain_ratio = self.config.get('preserve_domain_ratio', 0.5)
        self.business_domain_ratio = self.config.get('business_domain_ratio', 0.2)

        # Load domains
        self._domain_list = self._load_domains()

        # Categorize domains into different types for more flexible generation
        self._common_domains = []
        self._business_domains = []
        self._educational_domains = []
        self._categorize_domains()

        # Load nicknames
        self._nicknames = self._load_nicknames()

        # Set up PRGN generator if needed
        self.prgn_generator = None
        key = self.config.get('key')
        if key:
            self.prgn_generator = PRNGenerator(global_seed=key)

        # Email validation regex pattern
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        # Allowed characters for local part
        self.allowed_local_chars = string.ascii_letters + string.digits + '.-_+'

    def _categorize_domains(self):
        """
        Categorize domains into common, business, and educational domains.
        """
        from pamola_core.fake_data.dictionaries import domains as domain_dicts

        # Try to load lists from domains module
        try:
            self._common_domains = domain_dicts.get_common_email_domains()
            self._business_domains = domain_dicts.get_business_email_domains()
            self._educational_domains = domain_dicts.get_educational_email_domains()
        except Exception:
            # In case of error, use heuristic categorization
            # Categorize domains by their names
            for domain in self._domain_list:
                if any(business_term in domain for business_term in
                       ['company', 'corp', 'enterprise', 'business', 'inc', 'llc', 'agency', 'consulting']):
                    self._business_domains.append(domain)
                elif any(edu_term in domain for edu_term in
                         ['edu', 'ac.', 'university', 'school', 'college']):
                    self._educational_domains.append(domain)
                else:
                    self._common_domains.append(domain)

    def _load_domains(self) -> List[str]:
        """
        Load domain list from configuration or default sources.

        Returns:
            List[str]: List of domain names
        """
        domains = []

        # If domains is a list, use it directly
        if isinstance(self.domains, list) and self.domains:
            domains = self.domains
        # If domains is a string, treat it as a file path
        elif isinstance(self.domains, str) and self.domains:
            try:
                domains = dict_helpers.load_dictionary_from_text(self.domains)
            except Exception as e:
                print(f"Error loading domains from {self.domains}: {e}")

        # If no domains loaded, try to load from embedded dictionary
        if not domains:
            try:
                domains = dict_helpers.load_multi_dictionary(
                    "domain",
                    {"language": "en"},
                    fallback_to_embedded=True
                )
            except Exception as e:
                print(f"Error loading embedded domains: {e}")

        # If still no domains, use a minimal default list
        if not domains:
            domains = [
                "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                "mail.com", "example.com", "company.com", "organization.org"
            ]

        return domains

    def _load_nicknames(self) -> List[str]:
        """
        Load nicknames from dictionary or generate default ones.

        Returns:
            List[str]: List of nicknames
        """
        nicknames = []

        # If nicknames_dict is provided, try to load from it
        if self.nicknames_dict:
            try:
                nicknames = dict_helpers.load_dictionary_from_text(self.nicknames_dict)
            except Exception as e:
                print(f"Error loading nicknames from {self.nicknames_dict}: {e}")

        # If no nicknames loaded, generate some default ones
        if not nicknames:
            # Common prefixes for nicknames
            prefixes = ['cool', 'super', 'cyber', 'digital', 'tech', 'web', 'net', 'data',
                        'pro', 'smart', 'bright', 'swift', 'quick', 'fast', 'clever',
                        'happy', 'jolly', 'sunny', 'star', 'moon', 'sky', 'ocean', 'mountain']

            # Common stems for nicknames
            stems = ['user', 'person', 'friend', 'buddy', 'pal', 'mate', 'coder', 'dev',
                     'guru', 'ninja', 'master', 'expert', 'geek', 'nerd', 'fan', 'lover',
                     'mind', 'thinker', 'brain', 'wizard', 'sage', 'explorer', 'traveler']

            # Generate combinations
            for prefix in prefixes:
                for stem in stems:
                    nicknames.append(f"{prefix}{stem}")

            # Add some simple ones
            nicknames.extend(['user', 'admin', 'info', 'contact', 'support', 'help',
                              'sales', 'marketing', 'service', 'customer', 'client',
                              'mailbox', 'inbox', 'mail', 'email', 'account', 'profile'])

        return nicknames

    def _select_random_domain(self, original_domain: Optional[str] = None) -> str:
        """
        Select a random domain from the available list.

        Args:
            original_domain: Original domain to avoid (if any)

        Returns:
            str: Selected domain
        """
        if not self._domain_list:
            raise ValueError("No domain list available")

        # Determine which type of domain to use
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(original_domain or "domain", salt="domain-type-selection")
            domain_type_value = rng.random()
        else:
            domain_type_value = random.random()

        # Decide whether to preserve the original domain
        if original_domain and domain_type_value < self.preserve_domain_ratio:
            return original_domain

        # Calculate which type of domain to use
        if domain_type_value < self.business_domain_ratio:
            # Use business domain
            domain_list = self._business_domains if self._business_domains else self._domain_list
        elif domain_type_value < self.business_domain_ratio + 0.1:  # 10% for educational domains
            # Use educational domain
            domain_list = self._educational_domains if self._educational_domains else self._domain_list
        else:
            # Use common domain
            domain_list = self._common_domains if self._common_domains else self._domain_list

        # If the list is empty, use the general list
        if not domain_list:
            domain_list = self._domain_list

        # Remove original domain from candidates if it exists
        available_domains = [d for d in domain_list if d != original_domain]
        if not available_domains:
            available_domains = domain_list

        # If using PRGN generator for deterministic selection
        if self.prgn_generator:
            # Use deterministic selection but avoid the original domain
            rng = self.prgn_generator.get_random_by_value(original_domain or "domain", salt="domain-selection")
            index = rng.randint(0, len(available_domains) - 1)
            return available_domains[index]

        # Otherwise use random selection
        return random.choice(available_domains)

    def _generate_random_nickname(self) -> str:
        """
        Generate a random nickname.

        Returns:
            str: Generated nickname
        """
        if not self._nicknames:
            # Fallback to simple generation if no nicknames available
            letters = string.ascii_lowercase
            length = random.randint(5, 10)
            return ''.join(random.choice(letters) for _ in range(length))

        # If using PRGN generator for deterministic selection
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value(self.original_value, salt=self.context_salt)
            index = rng.randint(0, len(self._nicknames) - 1)
            return self._nicknames[index]

        # Otherwise use random selection
        return random.choice(self._nicknames)

    def _generate_random_separator(self) -> str:
        """
        Generate a random separator for email parts.

        Returns:
            str: Generated separator from configured options
        """
        separators = self.separator_options

        # If using PRGN generator for deterministic selection
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value("separator", salt="separator-selection")
            index = rng.randint(0, len(separators) - 1)
            return separators[index]

        # Otherwise use random selection
        return random.choice(separators)

    def _generate_random_number(self) -> str:
        """
        Generate a random number suffix for email.

        Returns:
            str: Generated number as string (or empty string)
        """
        # Check probability to have no number
        probability_threshold = 1.0 - self.number_suffix_probability

        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value("number", salt="number-generation")
            value = rng.random()
            if value < probability_threshold:
                return ""

            # Generate 1-4 digit number
            digits = rng.randint(1, 4)
            return str(rng.randint(1, 10 ** digits - 1))
        else:
            if random.random() < probability_threshold:
                return ""

            # Generate 1-4 digit number
            digits = random.randint(1, 4)
            return str(random.randint(1, 10 ** digits - 1))

    def _sanitize_local_part(self, text: str) -> str:
        """
        Sanitize the local part of an email address to ensure it's valid.

        Args:
            text: Text to sanitize

        Returns:
            str: Sanitized text
        """
        # Replace non-ASCII characters
        sanitized = ''
        for char in text:
            if char in self.allowed_local_chars:
                sanitized += char
            elif char.isalnum():  # Allow other alphanumeric chars but convert to ASCII range
                sanitized += 'x'
            else:
                sanitized += ''  # Remove other characters

        # Ensure it's not empty and not too long
        if not sanitized:
            sanitized = "user"

        if len(sanitized) > self.max_local_part_length:
            sanitized = sanitized[:self.max_local_part_length]

        # Ensure it doesn't start or end with a dot
        sanitized = sanitized.strip('.')

        # Replace consecutive dots
        while '..' in sanitized:
            sanitized = sanitized.replace('..', '.')

        return sanitized

    def validate_email(self, email: str) -> bool:
        """
        Validate an email address.

        Args:
            email: Email address to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not email or not isinstance(email, str):
            return False

        # Check basic format
        if not self.email_pattern.match(email):
            return False

        # Check length constraints
        if len(email) > self.max_length:
            return False

        # Check if it has exactly one @ symbol
        parts = email.split('@')
        if len(parts) != 2:
            return False

        local_part, domain = parts

        # Check local part length
        if len(local_part) == 0 or len(local_part) > 64:
            return False

        # Check domain length
        if len(domain) > self.max_domain_length:
            return False

        # Check domain format (must have at least one dot)
        if '.' not in domain:
            return False

        # Check each domain label doesn't start or end with a dash
        for label in domain.split('.'):
            if label.startswith('-') or label.endswith('-'):
                return False

        return True

    def extract_domain(self, email: str) -> Optional[str]:
        """
        Extract domain from an email address.

        Args:
            email: Email address

        Returns:
            Optional[str]: Domain or None if invalid
        """
        if not email or not isinstance(email, str):
            return None

        parts = email.split('@')
        if len(parts) != 2:
            return None

        return parts[1]

    def parse_email_format(self, email: str) -> str:
        """
        Determines the format of an email address.

        Attempts to identify which format was used to generate the email.

        Args:
            email: Email address to analyze

        Returns:
            str: Determined format (name_surname, surname_name, nickname, existing_domain or unknown)
        """
        if not email or not isinstance(email, str) or '@' not in email:
            return "unknown"

        # Split email into local part and domain
        local_part, domain = email.split('@', 1)

        # Check if local part contains a separator
        has_separator = any(sep in local_part for sep in ['.', '_', '-'])

        # If local part has at least one separator, it could be name_surname or surname_name
        if has_separator:
            # Split local part into components
            separator = None
            for sep in ['.', '_', '-']:
                if sep in local_part:
                    separator = sep
                    break

            if separator:
                parts = local_part.split(separator)
                # Remove potential numeric suffix
                if len(parts) > 2 and parts[-1].isdigit():
                    parts = parts[:-1]

                if len(parts) == 2:
                    # Check if this is name_surname or surname_name format
                    # This is a heuristic as we can't determine with certainty without additional info
                    first_part, second_part = parts

                    # If first part is shorter, it's likely a first name
                    if len(first_part) < len(second_part):
                        return "name_surname"
                    else:
                        return "surname_name"

        # If local part contains digits or is short, it's likely a nickname
        if any(c.isdigit() for c in local_part) or len(local_part) < 6:
            return "nickname"

        # Try to determine if this is existing_domain
        # This is difficult without knowing the original domain
        # We can use domain popularity as an indicator
        if domain in self._domain_list[:5]:  # If domain is in top 5 popular
            return "existing_domain"

        # If all checks fail, return "unknown"
        return "unknown"

    def _generate_from_name_components(self, first_name: Optional[str], last_name: Optional[str],
                                       format_type: str) -> str:
        """
        Generate email local part from name components.

        Args:
            first_name: First name
            last_name: Last name
            format_type: Format type (name_surname or surname_name)

        Returns:
            str: Generated local part
        """
        # Handle missing components
        first_name = self._sanitize_local_part(first_name) if first_name else ""
        last_name = self._sanitize_local_part(last_name) if last_name else ""

        # If either component is missing, use the other or fallback to nickname
        if not first_name and not last_name:
            return self._generate_random_nickname()
        elif not first_name:
            return last_name.lower()
        elif not last_name:
            return first_name.lower()

        separator = self._generate_random_separator()

        # Format based on type
        if format_type == "surname_name":
            local_part = f"{last_name}{separator}{first_name}"
        else:  # name_surname format
            local_part = f"{first_name}{separator}{last_name}"

        # Add random number suffix based on probability
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value("number_suffix", salt="number-suffix-decision")
            add_number = rng.random() < self.number_suffix_probability
        else:
            add_number = random.random() < self.number_suffix_probability

        if add_number:
            number = self._generate_random_number()
            if number:
                local_part = f"{local_part}{separator}{number}"

        return local_part.lower()

    def _generate_nickname_format(self) -> str:
        """
        Generate local part using nickname format.

        Returns:
            str: Generated local part
        """
        nickname = self._generate_random_nickname()
        separator = self._generate_random_separator()

        # Add number suffix based on probability
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value("number_suffix", salt="nickname-number-suffix")
            add_number = rng.random() < self.number_suffix_probability
        else:
            add_number = random.random() < self.number_suffix_probability

        if add_number:
            number = self._generate_random_number()
            if number:
                return f"{nickname}{separator}{number}".lower()

        return nickname.lower()

    def _select_format(self) -> str:
        """
        Select a format based on configured ratios.

        Returns:
            str: Selected format
        """
        # If specific format is set, use it
        if self.format:
            return self.format

        # If format ratio is provided, use weighted selection
        if self.format_ratio:
            formats = list(self.format_ratio.keys())
            weights = list(self.format_ratio.values())

            # Normalize weights if they don't sum to 1
            total = sum(weights)
            if total != 1.0:
                weights = [w / total for w in weights]

            # If using PRGN generator for deterministic selection
            if self.prgn_generator:
                # Simple weighted selection using PRGN
                # Fix: use get_random_by_value instead of generate_float
                rng = self.prgn_generator.get_random_by_value("format_selection", salt="email-format-selection")
                value = rng.random()  # Get random number between 0 and 1

                cumulative = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if value <= cumulative:
                        return formats[i]
                return formats[-1]

            # Use random weighted choice
            return random.choices(formats, weights=weights, k=1)[0]

        # Default formats with equal probabilities
        default_formats = ["name_surname", "surname_name", "nickname"]

        # If using PRGN generator for deterministic selection
        if self.prgn_generator:
            rng = self.prgn_generator.get_random_by_value("format_default", salt="format-default-selection")
            index = rng.randint(0, len(default_formats) - 1)
            return default_formats[index]

        # Otherwise use random selection
        return random.choice(default_formats)

    def generate(self, count: int, **params) -> List[str]:
        """
        Generate specified number of synthetic email addresses.

        Args:
            count: Number of values to generate
            **params: Additional parameters including:
                - first_name: First name (for name_surname/surname_name formats)
                - last_name: Last name (for name_surname/surname_name formats)
                - format: Override configured format
                - domain: Specific domain to use
                - original_email: Original email to extract domain from

        Returns:
            List[str]: Generated email addresses
        """
        result = []

        for _ in range(count):
            email = self._generate_email(**params)
            result.append(email)

        return result

    def _generate_email(self, **params) -> str:
        """
        Generate a single email address based on parameters.

        Args:
            **params: Parameters including:
                - first_name: First name component
                - last_name: Last name component
                - format: Format to use
                - domain: Specific domain to use
                - original_email: Original email to extract domain from

        Returns:
            str: Generated email address
        """
        # Extract parameters
        first_name = params.get("first_name")
        last_name = params.get("last_name")
        format_type = params.get("format", self._select_format())
        domain = params.get("domain")
        original_email = params.get("original_email")

        # Generate local part (before @) based on format
        local_part = ""

        if format_type == "name_surname" or format_type == "surname_name":
            local_part = self._generate_from_name_components(first_name, last_name, format_type)
        elif format_type == "nickname":
            local_part = self._generate_nickname_format()
        else:
            # Default to nickname if format not recognized
            local_part = self._generate_nickname_format()

        # Ensure local part doesn't exceed limit
        if len(local_part) > self.max_local_part_length:
            local_part = local_part[:self.max_local_part_length]

        # Determine domain
        email_domain = domain

        # If no domain provided but format is existing_domain and original_email is valid
        if not email_domain and format_type == "existing_domain" and original_email:
            if self.validate_email(original_email):
                email_domain = self.extract_domain(original_email)

        # If still no domain, select a random one
        if not email_domain:
            # Get original domain to avoid (for variety)
            original_domain = None
            if original_email and self.validate_email(original_email):
                original_domain = self.extract_domain(original_email)

            email_domain = self._select_random_domain(original_domain)

        # Combine to create email
        email = f"{local_part}@{email_domain}"

        # Ensure it doesn't exceed max length
        if len(email) > self.max_length:
            # Trim local part to fit
            max_local_length = self.max_length - len(email_domain) - 1  # -1 for @
            local_part = local_part[:max_local_length]
            email = f"{local_part}@{email_domain}"

        return email

    def generate_like(self, original_value: str, **params) -> str:
        """
        Generate a synthetic email address similar to the original one.

        Args:
            original_value: Original email address
            **params: Additional parameters including:
                - first_name: First name for name-based formats
                - last_name: Last name for name-based formats
                - format: Override configured format
                - full_name: Full name string to extract components from
                - name_format: Format of the full_name (FL, FML, LF, etc.)

        Returns:
            str: Generated email address
        """
        # Extract parameters
        format_override = params.get("format")

        # Parse full name if provided
        first_name = params.get("first_name")
        last_name = params.get("last_name")
        full_name = params.get("full_name")
        name_format = params.get("name_format")

        if full_name and not (first_name and last_name):
            # Extract name components from full name
            name_components = self._parse_full_name(full_name, name_format)
            if "first_name" in name_components and not first_name:
                first_name = name_components["first_name"]
            if "last_name" in name_components and not last_name:
                last_name = name_components["last_name"]

        # Validate the original email if validation is enabled
        is_valid = False
        if self.validate_source:
            is_valid = self.validate_email(original_value)
        else:
            # If validation is disabled, treat as valid if it has basic structure
            is_valid = isinstance(original_value, str) and '@' in original_value

        self.original_value = original_value
        self.context_salt = params.get("context_salt", None)

        # Select format to use
        format_to_use = format_override or self._select_format()

        # For existing_domain format, use original domain if valid
        if format_to_use == "existing_domain" and is_valid:
            # Create email with original domain but new local part
            domain = self.extract_domain(original_value)

            # Generate appropriate local part
            if first_name and last_name:
                # If name components available, use them
                local_format = params.get("local_format", "name_surname")
                local_part = self._generate_from_name_components(first_name, last_name, local_format)
            else:
                # Otherwise use nickname format
                local_part = self._generate_nickname_format()

            # Return combined email
            return f"{local_part}@{domain}"

        params["first_name"] = first_name
        params["last_name"] = last_name
        params["format"] = format_to_use
        params["original_email"] = original_value

        # For other cases or invalid emails, generate new email
        if not is_valid:
            # Handle invalid email according to configuration
            if self.handle_invalid_email == "keep_empty":
                return ""
            elif self.handle_invalid_email == "generate_with_default_domain":
                # Generate with default domain
                params["domain"] = random.choice(self._domain_list) if self._domain_list else "example.com"
                return self._generate_email(**params)
            else:  # generate_new (default)
                return self._generate_email(**params)

        # For valid emails with other formats, generate as normal
        return self._generate_email(**params)

    def _parse_full_name(self, full_name: str, name_format: Optional[str] = None) -> Dict[str, str]:
        """
        Parse a full name into components based on format.

        Args:
            full_name: Full name string
            name_format: Format code (FL, FML, LF, LFM, etc.)

        Returns:
            Dict[str, str]: Name components
        """
        if not full_name:
            return {}

        # Split the full name into parts
        parts = full_name.split()

        # Default result with empty components
        result = {
            "first_name": "",
            "middle_name": "",
            "last_name": ""
        }

        # Return empty if no parts
        if not parts:
            return result

        # For single part, assume it's either first or last name
        if len(parts) == 1:
            # Default to first name
            result["first_name"] = parts[0]
            return result

        # Handle different formats based on name_format parameter
        if name_format:
            if name_format.upper() == "FL" and len(parts) >= 2:
                result["first_name"] = parts[0]
                result["last_name"] = " ".join(parts[1:])
            elif name_format.upper() == "LF" and len(parts) >= 2:
                result["last_name"] = parts[0]
                result["first_name"] = " ".join(parts[1:])
            elif name_format.upper() == "FML" and len(parts) >= 3:
                result["first_name"] = parts[0]
                result["middle_name"] = parts[1]
                result["last_name"] = " ".join(parts[2:])
            elif name_format.upper() == "LFM" and len(parts) >= 3:
                result["last_name"] = parts[0]
                result["first_name"] = parts[1]
                result["middle_name"] = " ".join(parts[2:])
            elif name_format.upper() == "F" and len(parts) >= 1:
                result["first_name"] = " ".join(parts)
            elif name_format.upper() == "L" and len(parts) >= 1:
                result["last_name"] = " ".join(parts)
            else:
                # Default to guessing based on number of parts
                return self._parse_full_name(full_name, None)
        else:
            # Guess based on number of parts
            if len(parts) == 2:
                # Assume First Last
                result["first_name"] = parts[0]
                result["last_name"] = parts[1]
            elif len(parts) >= 3:
                # Assume First Middle Last
                result["first_name"] = parts[0]
                result["middle_name"] = parts[1]
                result["last_name"] = " ".join(parts[2:])

        return result

    def transform(self, values: List[str], **params) -> List[str]:
        """
        Transform a list of original values into synthetic ones.

        Default implementation calls generate_like for each value.

        Args:
            values: List of original values
            **params: Additional parameters for generation

        Returns:
            List of transformed values
        """
        return [self.generate_like(value, **params) for value in values]

    def validate(self, value: str) -> bool:
        """
        Check if a value is a valid email address.

        Args:
            value: Value to validate

        Returns:
            bool: True if value is a valid email, False otherwise
        """
        return self.validate_email(value)