"""
Embedded dictionaries for organization names.

Provides educational institution names, business names, and other organization types
for different countries.
"""

from typing import List

# Dictionary cache
_dictionary_cache = {}


# ----------------------- EDUCATIONAL INSTITUTIONS -----------------------

def get_educational_institutions(country_code: str = "US") -> List[str]:
    """
    Returns educational institution names for a specific country.

    Args:
        country_code: ISO country code

    Returns:
        List[str]: List of educational institution names
    """
    country_code = country_code.upper()
    cache_key = f"{country_code}_edu"

    if cache_key not in _dictionary_cache:
        if country_code == "US":
            _dictionary_cache[cache_key] = [
                "Harvard University", "Stanford University", "MIT", "Yale University",
                "Princeton University", "Columbia University", "University of Chicago",
                "University of Pennsylvania", "California Institute of Technology",
                "Duke University", "Northwestern University", "Johns Hopkins University",
                "University of California, Berkeley", "Cornell University", "Brown University"
            ]
        elif country_code == "RU":
            _dictionary_cache[cache_key] = [
                "Московский государственный университет", "Санкт-Петербургский государственный университет",
                "МГТУ им. Баумана", "Московский физико-технический институт",
                "Высшая школа экономики", "Российский университет дружбы народов",
                "Санкт-Петербургский политехнический университет", "Томский государственный университет",
                "Новосибирский государственный университет", "Казанский федеральный университет"
            ]
        elif country_code == "GB":
            _dictionary_cache[cache_key] = [
                "University of Oxford", "University of Cambridge", "Imperial College London",
                "University College London", "University of Edinburgh", "King's College London",
                "London School of Economics", "University of Manchester", "University of Bristol",
                "University of Warwick"
            ]
        else:
            _dictionary_cache[cache_key] = []

    return _dictionary_cache[cache_key]


# ----------------------- BUSINESS ORGANIZATIONS -----------------------

def get_business_organizations(country_code: str = "US", industry: str = None) -> List[str]:
    """
    Returns business organization names for a specific country and industry.

    Args:
        country_code: ISO country code
        industry: Industry type (optional)

    Returns:
        List[str]: List of business organization names
    """
    country_code = country_code.upper()
    cache_key = f"{country_code}_business"

    if cache_key not in _dictionary_cache:
        if country_code == "US":
            _dictionary_cache[cache_key] = {
                "tech": [
                    "Acme Technologies", "Pinnacle Software", "Horizon Digital", "Apex Computing",
                    "Stellar Tech Solutions", "Infinity Systems", "Quantum Software", "Spark Technologies",
                    "Summit Digital", "Nexus Technologies"
                ],
                "finance": [
                    "Heritage Financial", "Prosperity Bank", "Cornerstone Investments", "Liberty Financial",
                    "Premier Trust", "Sentinel Banking", "Alliance Capital", "Guardian Finance",
                    "Legacy Investments", "Meridian Banking"
                ],
                "retail": [
                    "Evergreen Stores", "Urban Essentials", "Central Market", "Harmony Retail",
                    "Citywide Shops", "Quality Merchandise", "The Trading Post", "Marketplace Goods",
                    "Retail Solutions", "Consumer Essentials"
                ],
                "healthcare": [
                    "Wellness Medical Center", "Integrated Health", "Compassionate Care", "Lifeline Medical",
                    "Healing Hands Clinic", "Premier Healthcare", "Advanced Medical Solutions", "Vitality Health",
                    "Complete Care", "Optimal Wellness"
                ]
            }
        elif country_code == "RU":
            _dictionary_cache[cache_key] = {
                "tech": [
                    "Технологии Будущего", "Цифровые Решения", "ИнфоТех", "Системные Инновации",
                    "Технологический Альянс", "Цифровой Горизонт", "Квантовые Технологии", "Техносфера",
                    "Информационные Системы", "Технологический Центр"
                ],
                "finance": [
                    "Финансовый Альянс", "Инвест-Капитал", "Финансовые Решения", "Банк Развития",
                    "Капитал Групп", "Инвестиционная Компания", "Финансовая Корпорация", "Коммерческий Банк",
                    "Инвестиционный Фонд", "Финансовая Стратегия"
                ],
                "retail": [
                    "Торговый Дом", "Универмаг", "Торговая Сеть", "Розничная Компания",
                    "Торговый Центр", "Ритейл Групп", "Торговая Платформа", "Магазины Товаров",
                    "Сеть Магазинов", "Торговая Корпорация"
                ]
            }
        else:
            _dictionary_cache[cache_key] = {}

    # Return all industries if industry is None
    if industry is None:
        all_businesses = []
        for ind_list in _dictionary_cache[cache_key].values():
            all_businesses.extend(ind_list)
        return all_businesses

    # Return businesses for specific industry
    return _dictionary_cache[cache_key].get(industry, [])


# ----------------------- GOVERNMENT ORGANIZATIONS -----------------------

def get_government_organizations(country_code: str = "US") -> List[str]:
    """
    Returns government organization names for a specific country.

    Args:
        country_code: ISO country code

    Returns:
        List[str]: List of government organization names
    """
    country_code = country_code.upper()
    cache_key = f"{country_code}_gov"

    if cache_key not in _dictionary_cache:
        if country_code == "US":
            _dictionary_cache[cache_key] = [
                "Department of State", "Department of Defense", "Department of Justice",
                "Department of the Interior", "Department of Agriculture", "Department of Commerce",
                "Department of Labor", "Department of Health and Human Services", "Department of Education",
                "Department of Veterans Affairs", "Department of Homeland Security", "Environmental Protection Agency"
            ]
        elif country_code == "RU":
            _dictionary_cache[cache_key] = [
                "Министерство иностранных дел", "Министерство обороны", "Министерство юстиции",
                "Министерство внутренних дел", "Министерство сельского хозяйства",
                "Министерство промышленности и торговли",
                "Министерство труда", "Министерство здравоохранения", "Министерство образования и науки",
                "Федеральная служба безопасности", "Федеральная налоговая служба"
            ]
        else:
            _dictionary_cache[cache_key] = []

    return _dictionary_cache[cache_key]


# ----------------------- PUBLIC API -----------------------

def get_organization_names(country_code: str = "US", org_type: str = "business", industry: str = None) -> List[str]:
    """
    Returns organization names for a specific country, type, and industry.

    Args:
        country_code: ISO country code
        org_type: Organization type ("business", "education", "government")
        industry: Industry type for businesses (optional)

    Returns:
        List[str]: List of organization names
    """
    country_code = country_code.upper()

    if org_type == "education":
        return get_educational_institutions(country_code)
    elif org_type == "business":
        return get_business_organizations(country_code, industry)
    elif org_type == "government":
        return get_government_organizations(country_code)

    # Return empty list if no matching organizations found
    return []


def clear_cache():
    """
    Clears the dictionary cache to free memory.
    """
    global _dictionary_cache
    _dictionary_cache = {}