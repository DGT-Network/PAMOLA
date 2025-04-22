"""
Embedded dictionaries for phone number generation.

Provides country codes, operator codes, and formatting templates for
phone number generation with regional specificity.
"""

from typing import Dict, List, Tuple

# Dictionary cache
_dictionary_cache = {}


def get_country_codes() -> Dict[str, str]:
    """
    Returns a dictionary of country codes.

    Returns:
        Dict[str, str]: Dictionary mapping country codes to dial codes
    """
    if 'country_codes' not in _dictionary_cache:
        _dictionary_cache['country_codes'] = {
            "us": "1",
            "ca": "1",
            "uk": "44",
            "ru": "7",
            "kz": "7",
            "cn": "86",
            "jp": "81",
            "de": "49",
            "fr": "33",
            "it": "39",
            "es": "34",
            "br": "55",
            "au": "61",
            "in": "91",
            "mx": "52",
            "sg": "65",
            "ae": "971",
            "sa": "966",
            "vn": "84"
        }
    return _dictionary_cache['country_codes']


def get_operator_codes(country_code: str) -> List[str]:
    """
    Returns operator codes for a specific country.

    Args:
        country_code: Country code (e.g., "us", "ru")

    Returns:
        List[str]: List of operator codes for the country
    """
    cache_key = f'operator_codes_{country_code}'

    if cache_key not in _dictionary_cache:
        # Default operator codes for common countries
        if country_code.lower() == "us":
            _dictionary_cache[cache_key] = [
                "201", "202", "203", "205", "206", "207", "208", "209",
                "210", "212", "213", "214", "215", "216", "217", "218", "219",
                "301", "302", "303", "304", "305", "307", "308", "309",
                "401", "402", "403", "404", "405", "406", "407", "408", "409",
                "501", "502", "503", "504", "505", "507", "508", "509",
                "601", "602", "603", "604", "605", "606", "607", "608", "609",
                "701", "702", "703", "704", "705", "706", "707", "708", "709",
                "801", "802", "803", "804", "805", "806", "807", "808", "809",
                "901", "902", "903", "904", "905", "906", "907", "908", "909"
            ]
        elif country_code.lower() == "ru":
            _dictionary_cache[cache_key] = [
                "900", "901", "902", "903", "904", "905", "906", "908", "909",
                "910", "911", "912", "913", "914", "915", "916", "917", "918", "919",
                "920", "921", "922", "923", "924", "925", "926", "927", "928", "929",
                "930", "931", "932", "933", "934", "935", "936", "937", "938", "939",
                "950", "951", "952", "953", "954", "955", "956", "958", "959",
                "960", "961", "962", "963", "964", "965", "966", "967", "968", "969",
                "977", "978", "980", "981", "982", "983", "984", "985", "986", "987", "988", "989",
                "495", "499"  # Moscow landline
            ]
        elif country_code.lower() == "uk":
            _dictionary_cache[cache_key] = [
                "7400", "7401", "7402", "7403", "7500", "7501", "7502", "7503",
                "7505", "7506", "7507", "7508", "7509", "7510", "7511", "7512",
                "7513", "7514", "7515", "7516", "7517", "7518", "7519", "7520",
                "7521", "7522", "7523", "7525", "7526", "7527", "7528", "7529",
                "7530", "7531", "7532", "7533", "7534", "7536", "7537", "7538",
                "7539", "7540", "7541", "7542", "7543", "7544", "7545", "7546",
                "7547", "7548", "7549", "7550", "7551", "7552", "7553", "7554",
                "7555", "7556", "7557", "7559", "7570", "7572", "7573", "7574",
                "7575", "7576", "7577", "7578", "7579", "7580", "7581", "7582",
                "7583", "7584", "7585", "7586", "7587", "7588", "7589", "7590",
                "7591", "7592", "7593", "7594", "7595", "7596", "7597", "7598", "7599",
                "20"  # London landline
            ]
        elif country_code.lower() == "ca":
            _dictionary_cache[cache_key] = [
                "204", "226", "236", "249", "250", "289", "306", "343", "365",
                "387", "403", "416", "418", "431", "437", "438", "450", "506",
                "514", "519", "548", "579", "581", "587", "604", "613", "639",
                "647", "705", "709", "778", "780", "782", "807", "819", "825",
                "867", "873", "902", "905"
            ]
        elif country_code.lower() == "de":
            _dictionary_cache[cache_key] = [
                "151", "152", "157", "159", "160", "162", "163", "170", "171",
                "172", "173", "174", "175", "176", "177", "178", "179"
            ]
        elif country_code.lower() == "fr":
            _dictionary_cache[cache_key] = [
                "6", "7"  # Mobile prefixes in France
            ]
        else:
            # Default empty list for unknown countries
            _dictionary_cache[cache_key] = []

    return _dictionary_cache[cache_key]


def get_phone_formats() -> Dict[str, str]:
    """
    Returns a dictionary of phone number formatting templates.

    The format strings use placeholders:
    - CC: Country code
    - AAA/AAAA: Operator/area code
    - XXXX/XXX/XX: Random digits

    Returns:
        Dict[str, str]: Dictionary of format templates
    """
    if 'phone_formats' not in _dictionary_cache:
        _dictionary_cache['phone_formats'] = {
            # International formats
            "e164": "+CC{operator_code}{number}",  # E.164 format (no spaces or separators)

            # Country-specific formats
            "us": "+CC (AAA) XXX-XXXX",  # US format: +1 (555) 123-4567
            "uk": "+CC AAAA XXXXXX",  # UK format: +44 7911 123456
            "ru": "+CC (AAA) XXX-XX-XX",  # Russia format: +7 (903) 123-45-67
            "ca": "+CC (AAA) XXX-XXXX",  # Canada format: +1 (416) 123-4567
            "de": "+CC AAA XXXXXXX",  # Germany format: +49 170 1234567
            "fr": "+CC A XX XX XX XX",  # France format: +33 6 12 34 56 78
            "cn": "+CC AAA XXXX XXXX",  # China format: +86 133 1234 5678
            "jp": "+CC AA XXXX XXXX",  # Japan format: +81 90 1234 5678
            "au": "+CC AAA XXX XXX",  # Australia format: +61 412 345 678

            # Generic formats
            "intl_spaces": "+CC AAA XXX XXX",  # Generic international with spaces
            "intl_dashes": "+CC-AAA-XXX-XXXX",  # Generic international with dashes
            "local": "(AAA) XXX-XXXX",  # Local format without country code
            "numeric": "CCAAAXXXXXXX"  # Just numbers, no formatting
        }
    return _dictionary_cache['phone_formats']


def get_phone_format_for_country(country_code: str) -> str:
    """
    Returns the appropriate phone format for a specific country.

    Args:
        country_code: Country code (e.g., "us", "ru")

    Returns:
        str: Format template for the country
    """
    formats = get_phone_formats()
    country_code = country_code.lower()

    # Return country-specific format if available
    if country_code in formats:
        return formats[country_code]

    # Default to international format with spaces
    return formats["intl_spaces"]


def get_phone_length_ranges() -> Dict[str, Tuple[int, int]]:
    """
    Returns typical ranges for phone number lengths (excluding country code).

    Returns:
        Dict[str, Tuple[int, int]]: Dictionary mapping country codes to (min, max) length ranges
    """
    if 'phone_lengths' not in _dictionary_cache:
        _dictionary_cache['phone_lengths'] = {
            "us": (10, 10),  # US: 10 digits
            "ca": (10, 10),  # Canada: 10 digits
            "uk": (10, 10),  # UK: usually 10 digits
            "ru": (10, 10),  # Russia: 10 digits
            "de": (10, 11),  # Germany: 10-11 digits
            "fr": (9, 9),  # France: 9 digits
            "cn": (11, 11),  # China: 11 digits
            "jp": (10, 10),  # Japan: 10 digits
            "au": (9, 9),  # Australia: 9 digits
            "default": (9, 12)  # Default range for other countries
        }
    return _dictionary_cache['phone_lengths']


def clear_cache():
    """
    Clears the dictionary cache to free memory.
    """
    global _dictionary_cache
    _dictionary_cache = {}