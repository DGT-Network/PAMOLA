"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Enumerations
Package:       pamola_core.utils.nlp.llm.enums
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides canonical enumeration definitions for the LLM subsystem.
All enum types used across preprocessing, processing, and postprocessing
stages are defined here to avoid duplication and circular imports.

Key Features:
- Single source of truth for all LLM-related enumerations
- Prevents enum duplication across modules
- Avoids circular import dependencies
- Consistent enum values across the entire LLM pipeline
- Easy to extend with new enum types

Design Principles:
- All enums inherit from standard Enum (not str, Enum) for type safety
- Values use lowercase with underscores for consistency
- Clear, descriptive names that match their usage context
- Comprehensive docstrings for each enum and value

Framework:
Part of PAMOLA.CORE LLM utilities, providing foundational enumerations
for type-safe operations across the LLM processing pipeline.

Usage:
```python
from pamola_core.utils.nlp.llm.enums import ResponseType, ProcessingStage
from pamola_core.utils.nlp.llm.enums import QualityLevel, TokenEstimationMethod
```

Changelog:
1.0.0 - Initial implementation
     - ResponseType enum (migrated from processing.py)
     - ProcessingStage enum (migrated from data_contracts.py)
     - QualityLevel enum (migrated from data_contracts.py)
     - TokenEstimationMethod enum (from config.py)
     - TruncationStrategy enum (from config.py)
     - CacheType enum (from config.py)
     - Provider enum (from config.py)

Dependencies:
- enum - Standard library enumeration support

TODO:
- Add EntityType enum when NER functionality is implemented
- Add ValidationResult enum when model validation is enhanced
- Consider adding custom enum base class for common functionality
"""

from enum import Enum


# ------------------------------------------------------------------------------
# Response and Processing Enumerations
# ------------------------------------------------------------------------------

class ResponseType(Enum):
    """
    Types of LLM responses.

    Used throughout the pipeline to classify and handle different
    types of responses from language models.
    """
    VALID = "valid"  # Valid processed response ready for use
    SERVICE = "service"  # Service/meta response (asking for input, etc.)
    ERROR = "error"  # Error response from LLM or processing
    EMPTY = "empty"  # Empty or too short response
    INVALID = "invalid"  # Invalid format or unparseable content
    PII_DETECTED = "pii_detected"  # Personal information detected in output (future)


class ProcessingStage(Enum):
    """
    Enumeration of processing pipeline stages.

    Used for tracking where in the pipeline processing occurs
    and for stage-specific logging and metrics.
    """
    PREPROCESSING = "preprocessing"  # Text preprocessing stage
    LLM_PROCESSING = "llm_processing"  # LLM inference stage
    POSTPROCESSING = "postprocessing"  # Response postprocessing stage
    COMPLETE = "complete"  # Processing fully complete


class QualityLevel(Enum):
    """
    Quality assessment levels for processed text.

    Used to classify the quality of anonymization and text processing
    results for monitoring and quality control.
    """
    EXCELLENT = "excellent"  # High confidence, no issues detected
    GOOD = "good"  # Good quality with minor issues
    ACCEPTABLE = "acceptable"  # Acceptable but with notable issues
    POOR = "poor"  # Poor quality, multiple issues
    FAILED = "failed"  # Processing failed completely


# ------------------------------------------------------------------------------
# Configuration Enumerations
# ------------------------------------------------------------------------------

class Provider(Enum):
    """
    Supported LLM providers.

    Defines the available LLM service providers and their
    corresponding API implementations.
    """
    LMSTUDIO = "lmstudio"  # Local LM Studio instance
    OPENAI = "openai"  # OpenAI API service
    ANTHROPIC = "anthropic"  # Anthropic Claude API
    HUGGINGFACE = "huggingface"  # Hugging Face models
    CUSTOM = "custom"  # Custom provider implementation


class CacheType(Enum):
    """
    Cache backend types.

    Defines the available caching mechanisms for LLM responses
    and intermediate processing results.
    """
    MEMORY = "memory"  # In-memory cache (fast, non-persistent)
    FILE = "file"  # File-based cache (persistent, slower)
    REDIS = "redis"  # Redis cache (fast, shared, persistent)
    NONE = "none"  # No caching (always process fresh)


class TokenEstimationMethod(Enum):
    """
    Token estimation methods.

    Defines the available methods for estimating token counts
    in text before sending to LLM APIs.
    """
    SIMPLE = "simple"  # Character-based estimation (fast, approximate)
    TIKTOKEN = "tiktoken"  # OpenAI tiktoken library (accurate, slower)
    CUSTOM = "custom"  # Custom tokenizer implementation


class TruncationStrategy(Enum):
    """
    Text truncation strategies.

    Defines how text should be truncated when it exceeds
    the maximum token limit for LLM processing.
    """
    END = "end"  # Keep beginning, truncate end
    MIDDLE = "middle"  # Keep beginning and end, truncate middle
    SMART = "smart"  # Try to preserve sentence/paragraph boundaries


# ------------------------------------------------------------------------------
# Validation and Status Enumerations
# ------------------------------------------------------------------------------

class ValidationResult(Enum):
    """
    Model validation result types.

    Used when validating model availability and configuration
    before processing starts.
    """
    VALID_ALIAS = "valid_alias"  # Valid model alias resolved successfully
    VALID_PRESET = "valid_preset"  # Model has configuration preset
    VALID_NAME = "valid_name"  # Valid model name (no preset)
    NOT_AVAILABLE = "not_available"  # Model not available in provider
    INVALID = "invalid"  # Invalid model name or alias


class ServiceCategory(Enum):
    """
    Categories of service responses from LLMs.

    Used to classify different types of service/meta responses
    for better handling and debugging.
    """
    REQUEST_FOR_INPUT = "request_for_input"  # Asking for more input
    ACKNOWLEDGMENT = "acknowledgment"  # Simple acknowledgment response
    ERROR_RESPONSE = "error_response"  # Error message from model
    META_COMMENTARY = "meta_commentary"  # Commentary about the task
    CLARIFICATION = "clarification"  # Asking for clarification
    REFUSAL = "refusal"  # Refusing to complete task
    UNKNOWN = "unknown"  # Unclassified service response


# ------------------------------------------------------------------------------
# Future Extension Enumerations (Placeholders)
# ------------------------------------------------------------------------------

class EntityType(Enum):
    """
    Types of entities that can be detected.

    Placeholder for future NER (Named Entity Recognition) functionality.
    Will be populated when entity detection is implemented.
    """
    PERSON = "person"  # Personal names
    ORGANIZATION = "organization"  # Company/organization names
    LOCATION = "location"  # Geographic locations
    EMAIL = "email"  # Email addresses
    PHONE = "phone"  # Phone numbers
    DATE = "date"  # Date references
    MONEY = "money"  # Monetary amounts
    TECHNOLOGY = "technology"  # Technology/software names
    PROJECT = "project"  # Project names
    OTHER = "other"  # Other entity types


class PIIRiskLevel(Enum):
    """
    Risk levels for personally identifiable information.

    Placeholder for future PII detection functionality.
    Will be used to classify the risk level of detected PII.
    """
    NONE = "none"  # No PII detected
    LOW = "low"  # Low-risk PII (generic terms)
    MEDIUM = "medium"  # Medium-risk PII (specific but not unique)
    HIGH = "high"  # High-risk PII (uniquely identifying)
    CRITICAL = "critical"  # Critical PII (highly sensitive)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def get_all_enum_types() -> dict:
    """
    Get all enum types defined in this module.

    Returns
    -------
    dict
        Dictionary mapping enum class names to enum classes
    """
    import inspect

    enums = {}
    for name, obj in inspect.getmembers(inspect.getmodule(inspect.currentframe())):
        if inspect.isclass(obj) and issubclass(obj, Enum) and obj != Enum:
            enums[name] = obj

    return enums


def validate_enum_values(enum_class: type, expected_values: set) -> bool:
    """
    Validate that an enum contains exactly the expected values.

    Parameters
    ----------
    enum_class : type
        Enum class to validate
    expected_values : set
        Set of expected string values

    Returns
    -------
    bool
        True if enum values match exactly
    """
    if not issubclass(enum_class, Enum):
        return False

    actual_values = {item.value for item in enum_class}
    return actual_values == expected_values


def get_enum_by_value(enum_class: type, value: str):
    """
    Get enum member by its string value.

    Parameters
    ----------
    enum_class : type
        Enum class to search
    value : str
        String value to find

    Returns
    -------
    Enum member or None
        Matching enum member or None if not found
    """
    if not issubclass(enum_class, Enum):
        return None

    for item in enum_class:
        if item.value == value:
            return item

    return None


def list_enum_values(enum_class: type) -> list:
    """
    Get list of all string values in an enum.

    Parameters
    ----------
    enum_class : type
        Enum class to list

    Returns
    -------
    list
        List of string values
    """
    if not issubclass(enum_class, Enum):
        return []

    return [item.value for item in enum_class]


# ------------------------------------------------------------------------------
# Validation on Import
# ------------------------------------------------------------------------------

def _validate_critical_enums():
    """
    Validate critical enums have expected values on module import.

    This helps catch enum value changes that might break compatibility
    with existing code or data.
    """
    # Validate ResponseType has core required values
    required_response_types = {"valid", "service", "error", "empty", "invalid", "pii_detected"}
    if not validate_enum_values(ResponseType, required_response_types):
        import warnings
        warnings.warn(
            f"ResponseType enum values don't match expected: {required_response_types}. "
            f"Actual: {list_enum_values(ResponseType)}"
        )

    # Validate ProcessingStage has required stages
    required_stages = {"preprocessing", "llm_processing", "postprocessing", "complete"}
    if not validate_enum_values(ProcessingStage, required_stages):
        import warnings
        warnings.warn(
            f"ProcessingStage enum values don't match expected: {required_stages}. "
            f"Actual: {list_enum_values(ProcessingStage)}"
        )


# Run validation on import
try:
    _validate_critical_enums()
except Exception as e:
    import warnings

    warnings.warn(f"Enum validation failed during import: {e}")