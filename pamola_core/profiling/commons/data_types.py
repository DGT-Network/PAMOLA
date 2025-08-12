"""
Type definitions and enumerations for the profiling package.

This module defines various types, enumerations, and constants used
throughout the profiling system to ensure consistency and type safety.
"""

from enum import Enum, auto
from typing import Dict, List, Any, Set, Union, Optional


class DataType(Enum):
    """Enumeration of data types for profiling."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    LONGTEXT = "longtext"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    BOOLEAN = "boolean"
    MULTI_VALUED = "multi_valued"
    JSON = "json"
    ARRAY = "array"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    CORRELATION = "correlation"
    CORRELATION_MATRIX = "correlation_matrix"


class AnalysisType(Enum):
    """Enumeration of analysis types."""
    BASIC = "basic"
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    TEXT_ANALYSIS = "text_analysis"
    GROUP_VARIATION = "group_variation"
    DUPLICATES = "duplicates"
    EMAIL_ANALYSIS = "email_analysis"
    PHONE_ANALYSIS = "phone_analysis"
    NAME_ANALYSIS = "name_analysis"
    NER_ANALYSIS = "ner_analysis"


class ResultType(Enum):
    """Enumeration of result types."""
    STATS = "statistics"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    DICTIONARY = "dictionary"
    VISUALIZATION = "visualization"
    ERROR = "error"


class ArtifactType(Enum):
    """Enumeration of artifact types."""
    JSON = "json"
    CSV = "csv"
    PNG = "png"
    HTML = "html"
    TEXT = "text"


class OperationStatus(Enum):
    """Enumeration of operation statuses."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class PrivacyLevel(Enum):
    """Enumeration of privacy levels for fields."""
    PUBLIC = "public"  # No privacy concerns
    RESTRICTED = "restricted"  # Limited disclosure
    SENSITIVE = "sensitive"  # Strong protection required
    IDENTIFIER = "identifier"  # PII or direct identifier


class ProfilerConfig:
    """Configuration settings for profilers."""

    # Default values for profiling operations
    DEFAULT_TOP_N = 20
    DEFAULT_MIN_GROUP_SIZE = 2
    DEFAULT_CORRELATION_THRESHOLD = 0.1

    # Path configuration
    PROFILING_DIR_NAME = "profiling"
    DICTIONARIES_DIR_NAME = "dictionaries"
    VISUALIZATION_DIR_NAME = "visualizations"

    # Visualization settings
    DEFAULT_FIGURE_WIDTH = 12
    DEFAULT_FIGURE_HEIGHT = 8
    DEFAULT_DPI = 300

    # Sampling settings
    DEFAULT_SAMPLE_SIZE = 10000
    SAMPLING_ENABLED = True

    # MVF settings
    MVF_SEPARATOR = ","
    MVF_QUOTE_CHAR = '"'

    # Email settings
    EMAIL_DOMAINS_OF_INTEREST = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'mail.ru', 'yandex.ru'
    }

    # Phone settings
    DEFAULT_COUNTRY_CODE = "7"  # Russia
    PHONE_FORMAT_REGEX = r'\((\d+),(\d+),(\d+)(,"([^"]*)"|)\)'

    # Text analysis settings
    LONGTEXT_MIN_LENGTH = 200  # Characters
    DEFAULT_TEXT_SAMPLE_SIZE = 100  # Number of samples for text analysis

    # NER settings
    NER_ENABLED = False  # Enable/disable NER analysis

    # JSON settings
    JSON_MAX_DEPTH = 5  # Maximum depth for JSON analysis

    # Array settings
    ARRAY_MAX_ELEMENTS = 100  # Maximum number of array elements to analyze


# Type aliases for improved type hints
FieldName = str
FieldMapping = Dict[FieldName, Any]
AnalysisResults = Dict[FieldName, Dict[str, Any]]
FieldDefinition = Dict[str, Any]
FieldList = List[FieldName]
FieldSet = Set[FieldName]


# Data type detection patterns and thresholds
class DataTypeDetection:
    """Constants and thresholds for data type detection."""

    # Thresholds for categorical vs text fields
    CATEGORICAL_THRESHOLD = 100  # If nunique <= this, consider categorical

    # Date format patterns for detection
    DATE_PATTERNS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y"
    ]

    # Boolean value mappings
    BOOLEAN_TRUE_VALUES = {"true", "yes", "1", "y", "t"}
    BOOLEAN_FALSE_VALUES = {"false", "no", "0", "n", "f"}

    # Multi-value field detection
    MVF_INDICATORS = {",", ";", "|", "+"}
    MVF_THRESHOLD = 0.1  # If >= this proportion of values contain separators, consider MVF

    # Email detection
    EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    # Phone detection (relaxed pattern for basic detection)
    PHONE_BASIC_REGEX = r'^\(\d+,\d+,\d+.*\)$'

    # JSON detection
    JSON_START_CHARS = {'{', '['}
    JSON_END_CHARS = {'}', ']'}

    # Array detection
    ARRAY_REGEX = r'^\[.*\]$'


# Field categories for specialized profiling
class FieldCategory:
    """Categories of fields for specialized profiling."""

    PERSONAL_IDENTIFIERS = {"name", "first_name", "last_name", "middle_name", "full_name", "file_as", "UID"}
    CONTACT_INFORMATION = {"email", "home_phone", "work_phone", "cell_phone", "address", "city", "country", "zipcode",
                           "metro_station_name"}
    TEMPORAL = {"date", "time", "datetime", "timestamp", "birth_day", "birthdate", "birthday"}
    GEOGRAPHIC = {"country", "region", "area_name", "city", "address", "location", "coordinates", "zipcode",
                  "metro_station_name"}
    EDUCATIONAL = {"education", "education_level", "degree", "university", "school", "college", "qualification"}
    OCCUPATIONAL = {"job", "post", "position", "title", "role", "department", "company", "employer", "employments",
                    "work_schedules"}
    FINANCIAL = {"salary", "salary_currency", "income", "revenue", "cost", "price"}
    PREFERENCE = {"relocation", "road_time_type", "business_trip_readiness", "driver_license_types", "has_vehicle"}

    # Mapping from field name patterns to categories
    FIELD_CATEGORY_PATTERNS = {
        "name": PERSONAL_IDENTIFIERS,
        "phone": CONTACT_INFORMATION,
        "email": CONTACT_INFORMATION,
        "address": CONTACT_INFORMATION,
        "birth": TEMPORAL,
        "date": TEMPORAL,
        "time": TEMPORAL,
        "city": GEOGRAPHIC,
        "country": GEOGRAPHIC,
        "region": GEOGRAPHIC,
        "area": GEOGRAPHIC,
        "education": EDUCATIONAL,
        "school": EDUCATIONAL,
        "university": EDUCATIONAL,
        "job": OCCUPATIONAL,
        "work": OCCUPATIONAL,
        "position": OCCUPATIONAL,
        "post": OCCUPATIONAL,
        "company": OCCUPATIONAL,
        "salary": FINANCIAL,
        "currency": FINANCIAL,
        "income": FINANCIAL,
        "relocation": PREFERENCE,
        "vehicle": PREFERENCE,
        "driver": PREFERENCE
    }

    @classmethod
    def get_category_for_field(cls, field_name: str) -> str:
        """
        Determine the category for a field based on its name.

        Parameters:
        -----------
        field_name : str
            The name of the field

        Returns:
        --------
        str
            The category name or "UNKNOWN"
        """
        field_lower = field_name.lower()

        # Direct match in a category
        for category_name, fields in {
            "PERSONAL_IDENTIFIERS": cls.PERSONAL_IDENTIFIERS,
            "CONTACT_INFORMATION": cls.CONTACT_INFORMATION,
            "TEMPORAL": cls.TEMPORAL,
            "GEOGRAPHIC": cls.GEOGRAPHIC,
            "EDUCATIONAL": cls.EDUCATIONAL,
            "OCCUPATIONAL": cls.OCCUPATIONAL,
            "FINANCIAL": cls.FINANCIAL,
            "PREFERENCE": cls.PREFERENCE
        }.items():
            if field_lower in fields:
                return category_name

        # Pattern match
        for pattern, category in cls.FIELD_CATEGORY_PATTERNS.items():
            if pattern in field_lower:
                for category_name, fields in {
                    "PERSONAL_IDENTIFIERS": cls.PERSONAL_IDENTIFIERS,
                    "CONTACT_INFORMATION": cls.CONTACT_INFORMATION,
                    "TEMPORAL": cls.TEMPORAL,
                    "GEOGRAPHIC": cls.GEOGRAPHIC,
                    "EDUCATIONAL": cls.EDUCATIONAL,
                    "OCCUPATIONAL": cls.OCCUPATIONAL,
                    "FINANCIAL": cls.FINANCIAL,
                    "PREFERENCE": cls.PREFERENCE
                }.items():
                    if category == fields:
                        return category_name

        return "UNKNOWN"