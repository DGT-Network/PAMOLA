"""
Embedded dictionaries for address components.

Provides street names, city names, postal codes, and other address components
for different countries.
"""

from typing import Dict, List

# Dictionary cache
_dictionary_cache = {}


# ----------------------- RUSSIAN ADDRESSES (RU) -----------------------

def get_ru_street_names() -> List[str]:
    """
    Returns Russian street names.

    Returns:
        List[str]: List of Russian street names
    """
    if 'ru_streets' not in _dictionary_cache:
        _dictionary_cache['ru_streets'] = [
            "Ленина", "Пушкина", "Гагарина", "Мира", "Советская",
            "Московская", "Центральная", "Школьная", "Молодежная", "Лесная",
            "Садовая", "Набережная", "Заводская", "Октябрьская", "Комсомольская",
            "Первомайская", "Железнодорожная", "Пролетарская", "Гоголя", "Кирова"
        ]
    return _dictionary_cache['ru_streets']


def get_ru_cities() -> List[str]:
    """
    Returns Russian city names.

    Returns:
        List[str]: List of Russian city names
    """
    if 'ru_cities' not in _dictionary_cache:
        _dictionary_cache['ru_cities'] = [
            "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
            "Нижний Новгород", "Челябинск", "Самара", "Омск", "Ростов-на-Дону",
            "Уфа", "Красноярск", "Воронеж", "Пермь", "Волгоград",
            "Краснодар", "Саратов", "Тюмень", "Тольятти", "Ижевск"
        ]
    return _dictionary_cache['ru_cities']


def get_ru_regions() -> List[str]:
    """
    Returns Russian region names.

    Returns:
        List[str]: List of Russian region names
    """
    if 'ru_regions' not in _dictionary_cache:
        _dictionary_cache['ru_regions'] = [
            "Московская область", "Ленинградская область", "Новосибирская область",
            "Свердловская область", "Республика Татарстан", "Нижегородская область",
            "Челябинская область", "Самарская область", "Омская область",
            "Ростовская область", "Республика Башкортостан", "Красноярский край",
            "Воронежская область", "Пермский край", "Волгоградская область"
        ]
    return _dictionary_cache['ru_regions']


def get_ru_postal_codes() -> Dict[str, List[str]]:
    """
    Returns Russian postal codes by city.

    Returns:
        Dict[str, List[str]]: Dictionary mapping cities to postal codes
    """
    if 'ru_postcodes' not in _dictionary_cache:
        _dictionary_cache['ru_postcodes'] = {
            "Москва": ["101000", "105062", "107031", "109012", "115172"],
            "Санкт-Петербург": ["190000", "191186", "195197", "197198", "199034"],
            "Новосибирск": ["630000", "630099", "630111", "630132", "630087"],
            "Екатеринбург": ["620000", "620014", "620034", "620075", "620085"],
            "Казань": ["420000", "420015", "420066", "420094", "420097"]
        }
    return _dictionary_cache['ru_postcodes']


# ----------------------- ENGLISH ADDRESSES (US) -----------------------

def get_us_street_names() -> List[str]:
    """
    Returns:
        List[str]: List of US street names
    """
    if 'us_streets' not in _dictionary_cache:
        _dictionary_cache['us_streets'] = [
            "Main Street", "Oak Street", "Maple Avenue", "Washington Street", "Park Avenue",
            "Elm Street", "Broadway", "Highland Avenue", "Cedar Street", "Lake Street",
            "Lincoln Avenue", "Church Street", "First Street", "Second Avenue", "Pine Street",
            "Chestnut Street", "Jefferson Avenue", "River Road", "Market Street", "Forest Avenue"
        ]
    return _dictionary_cache['us_streets']

def get_us_cities() -> List[str]:
    """
    Returns US city names.

    Returns:
        List[str]: List of US city names
    """
    if 'us_cities' not in _dictionary_cache:
        _dictionary_cache['us_cities'] = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "San Francisco", "Columbus", "Indianapolis",
            "Seattle", "Denver", "Boston", "Portland", "Atlanta"
        ]
    return _dictionary_cache['us_cities']

def get_us_states() -> Dict[str, str]:
    """
    Returns US states with their abbreviations.

    Returns:
        Dict[str, str]: Dictionary mapping state names to abbreviations
    """
    if 'us_states' not in _dictionary_cache:
        _dictionary_cache['us_states'] = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
            "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
            "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
            "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
            "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
            "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
            "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
        }
    return _dictionary_cache['us_states']

def get_us_zip_codes() -> Dict[str, List[str]]:
    """
    Returns US ZIP codes by city.

    Returns:
        Dict[str, List[str]]: Dictionary mapping cities to ZIP codes
    """
    if 'us_zipcodes' not in _dictionary_cache:
        _dictionary_cache['us_zipcodes'] = {
            "New York": ["10001", "10002", "10003", "10004", "10005"],
            "Los Angeles": ["90001", "90007", "90012", "90017", "90024"],
            "Chicago": ["60601", "60602", "60603", "60604", "60605"],
            "Houston": ["77001", "77002", "77003", "77004", "77005"],
            "Phoenix": ["85001", "85003", "85004", "85006", "85007"]
        }
    return _dictionary_cache['us_zipcodes']

# ----------------------- VIETNAMESE ADDRESSES (VN) -----------------------

def get_vn_street_names() -> List[str]:
    """
    Returns Vietnamese street names.

    Returns:
        List[str]: List of Vietnamese street names
    """
    if 'vn_streets' not in _dictionary_cache:
        _dictionary_cache['vn_streets'] = [
            "Đường Lê Lợi", "Đường Nguyễn Huệ", "Đường Trần Hưng Đạo", "Đường Lý Thường Kiệt",
            "Đường Phan Chu Trinh", "Đường Hai Bà Trưng", "Đường Đinh Tiên Hoàng",
            "Đường Phan Đình Phùng", "Đường Nguyễn Trãi", "Đường Hàm Nghi"
        ]
    return _dictionary_cache['vn_streets']

def get_vn_cities() -> List[str]:
    """
    Returns Vietnamese city names.

    Returns:
        List[str]: List of Vietnamese city names
    """
    if 'vn_cities' not in _dictionary_cache:
        _dictionary_cache['vn_cities'] = [
            "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
            "Biên Hòa", "Nha Trang", "Vũng Tàu", "Huế", "Hạ Long"
        ]
    return _dictionary_cache['vn_cities']

def get_vn_districts() -> Dict[str, List[str]]:
    """
    Returns Vietnamese districts by city.

    Returns:
        Dict[str, List[str]]: Dictionary mapping cities to districts
    """
    if 'vn_districts' not in _dictionary_cache:
        _dictionary_cache['vn_districts'] = {
            "Hà Nội": ["Ba Đình", "Hoàn Kiếm", "Tây Hồ", "Long Biên", "Cầu Giấy"],
            "Hồ Chí Minh": ["Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5"],
            "Đà Nẵng": ["Hải Châu", "Thanh Khê", "Sơn Trà", "Ngũ Hành Sơn", "Liên Chiểu"]
        }
    return _dictionary_cache['vn_districts']

def get_vn_postal_codes() -> Dict[str, str]:
    """
    Returns Vietnamese postal codes by city.

    Returns:
        Dict[str, str]: Dictionary mapping cities to postal codes
    """
    if 'vn_postcodes' not in _dictionary_cache:
        _dictionary_cache['vn_postcodes'] = {
            "Hà Nội": "100000",
            "Hồ Chí Minh": "700000",
            "Đà Nẵng": "550000",
            "Hải Phòng": "180000",
            "Cần Thơ": "900000"
        }
    return _dictionary_cache['vn_postcodes']

# ----------------------- PUBLIC API -----------------------

def get_address_component(country_code: str, component_type: str) -> List[str]:
    """
    Returns address components for a specific country.

    Args:
        country_code: ISO country code (e.g., "RU", "US", "VN")
        component_type: Type of component ("street", "city", "region", "postal_code")

    Returns:
        List[str]: List of address components
    """
    country_code = country_code.upper()

    # Streets
    if component_type == "street":
        if country_code == "RU":
            return get_ru_street_names()
        elif country_code in ["US", "EN"]:
            return get_us_street_names()
        elif country_code == "VN":
            return get_vn_street_names()

    # Cities
    elif component_type == "city":
        if country_code == "RU":
            return get_ru_cities()
        elif country_code in ["US", "EN"]:
            return get_us_cities()
        elif country_code == "VN":
            return get_vn_cities()

    # Regions
    elif component_type == "region":
        if country_code == "RU":
            return get_ru_regions()
        elif country_code in ["US", "EN"]:
            return list(get_us_states().keys())
        elif country_code == "VN":
            return []  # Not implemented for VN

    # Return empty list if no matching component found
    return []

def get_postal_code_for_city(country_code: str, city: str) -> str:
    """
    Returns a postal code for a specific city.

    Args:
        country_code: ISO country code
        city: City name

    Returns:
        str: A postal code for the specified city
    """
    country_code = country_code.upper()

    if country_code == "RU":
        postcodes = get_ru_postal_codes()
        if city in postcodes:
            return postcodes[city][0]  # Return first postal code for the city

    elif country_code in ["US", "EN"]:
        zipcodes = get_us_zip_codes()
        if city in zipcodes:
            return zipcodes[city][0]  # Return first ZIP code for the city

    elif country_code == "VN":
        postcodes = get_vn_postal_codes()
        if city in postcodes:
            return postcodes[city]

    # Return empty string if no matching postal code found
    return ""

def clear_cache():
    """
    Clears the dictionary cache to free memory.
    """
    global _dictionary_cache
    _dictionary_cache = {}