"""
Embedded dictionaries for email domains and usernames.

Provides common email domains and username patterns for email generation.
"""

from typing import Dict, List

# Dictionary cache
_dictionary_cache = {}


def get_common_email_domains() -> List[str]:
    """
    Returns list of common email domains.

    Returns:
        List[str]: List of common email domains
    """
    if 'email_domains' not in _dictionary_cache:
        _dictionary_cache['email_domains'] = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "mail.ru", "yandex.ru", "protonmail.com", "icloud.com", "zoho.com",
            "gmx.com", "tutanota.com", "fastmail.com", "mailchimp.com", "inbox.com"
        ]
    return _dictionary_cache['email_domains']


def get_business_email_domains() -> List[str]:
    """
    Returns list of business email domains.

    Returns:
        List[str]: List of business email domains
    """
    if 'business_domains' not in _dictionary_cache:
        _dictionary_cache['business_domains'] = [
            "company.com", "enterprise.com", "corp.com", "business.com", "firm.com",
            "inc.com", "llc.com", "agency.com", "consulting.com", "solutions.com",
            "international.com", "global.com", "group.com", "partners.com", "associates.com"
        ]
    return _dictionary_cache['business_domains']


def get_educational_email_domains() -> List[str]:
    """
    Returns list of educational institution email domains.

    Returns:
        List[str]: List of educational email domains
    """
    if 'educational_domains' not in _dictionary_cache:
        _dictionary_cache['educational_domains'] = [
            "edu", "ac.uk", "edu.au", "edu.cn", "ac.jp",
            "harvard.edu", "stanford.edu", "mit.edu", "oxford.ac.uk", "cambridge.ac.uk",
            "berkeley.edu", "columbia.edu", "princeton.edu", "yale.edu", "caltech.edu"
        ]
    return _dictionary_cache['educational_domains']


def get_username_prefixes() -> List[str]:
    """
    Returns list of common username prefixes.

    Returns:
        List[str]: List of common username prefixes
    """
    if 'username_prefixes' not in _dictionary_cache:
        _dictionary_cache['username_prefixes'] = [
            "user", "member", "client", "customer", "student",
            "professor", "doctor", "employee", "staff", "admin",
            "support", "help", "info", "contact", "service"
        ]
    return _dictionary_cache['username_prefixes']


def get_tlds_by_country() -> Dict[str, str]:
    """
    Returns mapping of country codes to their top-level domains.

    Returns:
        Dict[str, str]: Dictionary mapping country codes to TLDs
    """
    if 'country_tlds' not in _dictionary_cache:
        _dictionary_cache['country_tlds'] = {
            "ru": "ru",
            "us": "com",
            "uk": "co.uk",
            "ca": "ca",
            "au": "com.au",
            "de": "de",
            "fr": "fr",
            "jp": "jp",
            "cn": "cn",
            "br": "com.br",
            "in": "in",
            "it": "it",
            "es": "es",
            "nl": "nl",
            "se": "se"
        }
    return _dictionary_cache['country_tlds']


def get_domain_by_country(country_code: str) -> List[str]:
    """
    Returns domain extensions for a specific country.

    Args:
        country_code: ISO country code (e.g., "ru", "us")

    Returns:
        List[str]: List of domain extensions for the country
    """
    country_code = country_code.lower()
    tlds = get_tlds_by_country()

    if country_code in tlds:
        tld = tlds[country_code]
        return [f"mail.{tld}", f"email.{tld}", f"inbox.{tld}", f"webmail.{tld}"]

    return []


def clear_cache():
    """
    Clears the dictionary cache to free memory.
    """
    global _dictionary_cache
    _dictionary_cache = {}