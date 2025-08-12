"""
PAMOLA.CORE - Regex Patterns Utilities
---------------------------------------------------
This module contains common regular expressions (regex) for general usage.

Features:
 - Common patterns for email, phone, credit card, domains, etc.
 - Language-specific phone extension formats.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
See: https://opensource.org/licenses/BSD-3-Clause
"""


class CommonPatterns:
    EMAIL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    PHONE = r"^\+?(?:\d[\d\-. ]+)?(?:\([\d\-. ]+\))?[\d\-. ]+\d$"
    DOMAIN = r"@([\w.-]+)"

    DATE_REGEX_FORMATS = {
        # ISO format with optional timezone
        r"\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?": "ISO8601",
        # Date with month name (English)
        r"\d{1,2}-[A-Za-z]{3}-\d{4}": "%d-%b-%Y",
        r"\d{1,2}-[A-Za-z]+-\d{4}": "%d-%B-%Y",
        r"\d{1,2} [A-Za-z]{3} \d{4}": "%d %b %Y",
        r"\d{1,2} [A-Za-z]+ \d{4}": "%d %B %Y",
        r"[A-Za-z]+ \d{1,2}, \d{4}": "%B %d, %Y",
        r"[A-Za-z]{3} \d{1,2}, \d{4}": "%b %d, %Y",
        r"\d{1,2} [A-Za-z]+ \d{4} \d{2}:\d{2}": "%d %B %Y %H:%M",
        r"\d{1,2} [A-Za-z]+ \d{4} \d{1,2}:\d{2} [APMapm]{2}": "%d %B %Y %I:%M %p",
        r"[A-Za-z]+ \d{1,2}, \d{4} \d{2}:\d{2}": "%B %d, %Y %H:%M",
        r"[A-Za-z]+ \d{1,2}, \d{4} \d{1,2}:\d{2} [APMapm]{2}": "%B %d, %Y %I:%M %p",
        # DMY and MDY with time
        r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}": "%d.%m.%Y %H:%M",
        r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}": "%d.%m.%Y %H:%M:%S",
        r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}": "%d/%m/%Y %H:%M",
        r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}": "%d/%m/%Y %H:%M:%S",
        r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}": "%d-%m-%Y %H:%M",
        r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}": "%d-%m-%Y %H:%M:%S",
        r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}": "%m/%d/%Y %H:%M",
        r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}": "%m/%d/%Y %H:%M:%S",
        r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}": "%m-%d-%Y %H:%M",
        r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}": "%m-%d-%Y %H:%M:%S",
        # DMY & MDY no time
        r"\d{2}\.\d{2}\.\d{4}": "%d.%m.%Y",
        r"\d{2}/\d{2}/\d{4}": "%d/%m/%Y",
        r"\d{2}-\d{2}-\d{4}": "%d-%m-%Y",
        r"\d{2}/\d{2}/\d{4}": "%m/%d/%Y",
        r"\d{2}-\d{2}-\d{4}": "%m-%d-%Y",
        # YMD formats
        r"\d{4}-\d{2}-\d{2}": "%Y-%m-%d",
        r"\d{4}/\d{2}/\d{2}": "%Y/%m/%d",
        r"\d{4}\.\d{2}\.\d{2}": "%Y.%m.%d",
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}": "%Y-%m-%d %H:%M",
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}": "%Y-%m-%d %H:%M:%S",
        r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}": "%Y/%m/%d %H:%M",
        r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}": "%Y/%m/%d %H:%M:%S",
        # Non-separated
        r"\d{8}": "%Y%m%d",
        r"\d{2}\d{2}\d{4}": "%d%m%Y",
        # With AM/PM
        r"\d{2}\.\d{2}\.\d{4} \d{1,2}:\d{2} [APMapm]{2}": "%d.%m.%Y %I:%M %p",
        r"\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} [APMapm]{2}": "%m/%d/%Y %I:%M %p",
        r"\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}:\d{2} [APMapm]{2}": "%m/%d/%Y %I:%M:%S %p",
        r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2} [APMapm]{2}": "%Y-%m-%d %I:%M %p",
    }

    EMAIL_NAME_STYLE_PATTERNS = {
        "name_dot_surname": r"^[a-zA-Z]+\.[a-zA-Z]+@",
        "name_underscore_surname": r"^[a-zA-Z]+_[a-zA-Z]+@",
        "surname_dot_name": r"^[a-zA-Z]+\.[a-zA-Z]+@",
        "surname_underscore_name": r"^[a-zA-Z]+_[a-zA-Z]+@",
        "name_surname": r"^[a-zA-Z]+[a-zA-Z]+@",
        "surname_name": r"^[a-zA-Z]+[a-zA-Z]+@",
    }


class FinancialPatterns:
    CREDIT_CARD_STRICT = r"^\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}$"
    CREDIT_CARD_FLEXIBLE = (
        r"^\d{13,16}$|^(\d{4}[ ]{1}){3}\d{1,4}$|^(\d{4}[-]{1}){3}\d{1,4}$"
    )
    PCI = r"^\d{16}$|^(\d{4}[ ]{1}){3}\d{4}$|^(\d{4}[-]{1}){3}\d{4}$"


class PhonePatterns:
    EXTENSION_PATTERNS = [
        r"(?:ext(?:ension)?[\s:.-]*)(\d{1,6})",  # English
        r"(?:внутр|доб)[\s:.-]*(\d{1,6})",  # Russian
        r"(?:poste|int)[\s:.-]*(\d{1,6})",  # French
        r"(?:nebenstelle|durchwahl)[\s:.-]*(\d{1,6})",  # German
        r"(?:x|#)\s?(\d{1,6})",  # Abbreviated
    ]

    PHONE_REGEX_PATTERNS = {
        "TEN_DIGIT": r"^(\(\d{3}\)|\d{3})[-\s]?(\d{3})-(\d{4})$",
        "INTERNATIONAL": r"^\+?(\d{1,3})[-\s]+(\d{2,4})[-\s]?(\d{4,})$",
        "PARENTHESES": r"^\((\d{1,3})\)\s*(\d{2,4})[-\s]*(\d{4,})$",
    }
