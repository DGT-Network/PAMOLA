"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Processing Data Contracts
Package:       pamola_core.utils.nlp.llm.data_contracts
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module defines minimal data contracts (dataclasses) for communication between
preprocessing, LLM processing, and postprocessing stages. These contracts
ensure type safety, clear interfaces, and consistent data flow throughout
the LLM processing pipeline.

Key Features:
- Minimal, focused data structures for immediate use
- Type-safe data structures for each processing stage
- Memory-efficient with slots enabled
- No duplicate enumerations (imports from canonical sources)
- Clear separation of concerns between pipeline stages
- Extensible design for future enhancements (via mixins/extensions)
- Helper constructors to avoid post_init mutations
- Comprehensive input validation
- Proper time semantics (duration vs timestamp)

Design Principles:
- Only include fields that are actively used in phase-1
- Avoid placeholder fields that increase memory footprint
- Use helper constructors instead of post_init mutations
- Import enums from canonical sources to prevent duplicates
- Maintain backward compatibility during transitions
- Clear distinction between duration and timestamp fields

Framework:
Part of PAMOLA.CORE LLM utilities, providing foundational data structures
for the modular LLM processing pipeline.

Data Flow:
Raw Text → PreprocessResult → LLM Processing → PostprocessResult → Final Result

Changelog:
1.2.0 - Final fixes from code review round C
     - Fixed enum validation to handle new values (pii_detected)
     - Renamed processing_time to processing_duration for clarity
     - Removed redundant Optional[str] types
     - Added BatchProcessingStats.add_latency() helper
     - Improved time semantics consistency
1.1.0 - Fixed critical issues from code review round A-B
     - Removed duplicate ResponseType enum (imports from enums.py)
     - Minimized dataclasses to only actively used fields
     - Added slots=True for memory efficiency
     - Replaced post_init mutations with helper constructors
     - Fixed enum inheritance patterns
     - Added proper length auto-calculation
     - Improved timestamp documentation
1.0.0 - Initial implementation (deprecated due to review findings)

Dependencies:
- dataclasses - Data structure definitions
- enum - Status and type enumerations (canonical sources only)
- typing - Type annotations
- time - Timestamp generation

TODO:
- Add extension mixins for NER/PII when needed
- Implement serialization helpers for persistence
- Add validation decorators for complex constraints
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Import canonical enums from their dedicated module to avoid duplicates
# All enums now live in llm/enums.py as single source of truth
from .enums import ResponseType, ProcessingStage


# ------------------------------------------------------------------------------
# Minimal Core Data Structures (Phase 1 Focus)
# ------------------------------------------------------------------------------

@dataclass(slots=True)
class PreprocessResult:
    """
    Minimal result of text preprocessing stage.

    Contains only fields actively used in phase-1 implementation.
    Future extensions (NER, advanced metrics) will be added via mixins.
    """
    # Core results - always populated
    text: str  # Processed text ready for LLM
    canonical_text: str = ""  # Canonicalized text for cache keys
    success: bool = True  # Whether preprocessing succeeded

    # Essential metrics - auto-calculated in post_init
    original_length: int = 0  # Original text length (auto-filled)
    processed_length: int = 0  # Processed text length (auto-filled)
    estimated_tokens: int = 0  # Estimated token count

    # Truncation info - only set if truncation occurred
    was_truncated: bool = False  # Whether text was truncated
    truncated_tokens: int = 0  # Number of tokens removed if truncated

    # Issue tracking - core functionality
    issues: List[str] = field(default_factory=list)
    processing_duration: float = 0.0  # Actual processing duration in seconds

    # Failure info - only set on failure
    error_message: Optional[str] = None  # Error details if success=False

    def __post_init__(self):
        """Auto-calculate lengths and validate core constraints."""
        # Auto-fill original_length directly from text if not provided
        if self.original_length == 0 and self.text:
            self.original_length = len(self.text)

        # Auto-fill processed_length if not provided and text exists
        if self.processed_length == 0 and self.text:
            self.processed_length = len(self.text)

        # Validate core constraints without mutating success flag
        if self.success and not self.text.strip():
            # Don't auto-fail here, let caller decide
            self.issues.append("empty_result_warning")


@dataclass(slots=True)
class PostprocessResult:
    """
    Minimal result of LLM response postprocessing stage.

    Contains only fields actively used in phase-1 implementation.
    PII detection and advanced quality metrics will be added later.
    """
    # Core results - always populated
    text: str  # Final processed text
    success: bool = True  # Whether postprocessing succeeded
    response_type: ResponseType = ResponseType.VALID

    # Service response detection - core functionality
    is_service_response: bool = False  # Whether response is service message
    service_category: Optional[str] = None  # Category of service response if detected

    # Essential cleaning info
    original_response: str = ""  # Original LLM response
    cleaning_applied: List[str] = field(default_factory=list)  # Cleaning steps applied

    # Issue tracking - core functionality
    issues: List[str] = field(default_factory=list)
    processing_duration: float = 0.0  # Actual processing duration in seconds

    # Failure info - only set on failure
    error_message: Optional[str] = None  # Error details if success=False

    def __post_init__(self):
        """Validate core constraints and enforce business rules."""
        # Force success=False for service responses
        if self.response_type == ResponseType.SERVICE:
            self.success = False
            self.is_service_response = True
            if "service_response_detected" not in self.issues:
                self.issues.append("service_response_detected")

        # Also fail for other non-valid response types
        if self.response_type in (ResponseType.ERROR, ResponseType.EMPTY, ResponseType.INVALID):
            self.success = False
            if f"response_type_{self.response_type.value}" not in self.issues:
                self.issues.append(f"response_type_{self.response_type.value}")

        if self.success and not self.text.strip():
            self.success = False
            self.issues.append("empty_result")

    def is_successful(self) -> bool:
        """Helper method to check if processing was truly successful."""
        return self.success and self.response_type == ResponseType.VALID and self.text.strip()


@dataclass(slots=True)
class ProcessingPipelineResult:
    """
    Minimal complete result from preprocessing → LLM → postprocessing pipeline.

    Focuses only on essential aggregation and metrics needed for current phase.
    """
    # Final result - always populated
    final_text: str  # Final processed text
    success: bool = True  # Overall pipeline success

    # Stage results - linked to other dataclasses
    preprocess_result: Optional[PreprocessResult] = None
    postprocess_result: Optional[PostprocessResult] = None

    # Essential LLM metadata
    llm_model: str = ""  # Model used for processing
    llm_response_time: float = 0.0  # LLM response time in seconds

    # Core performance metrics
    total_latency: float = 0.0  # Total pipeline duration in seconds
    cache_hit: bool = False  # Whether result was cached
    processing_timestamp: float = field(default_factory=time.time)  # Unix epoch seconds when processing started

    # Issue aggregation - essential for debugging
    all_issues: List[str] = field(default_factory=list)

    # Minimal metadata for essential tracking
    processing_id: Optional[str] = None  # Unique processing identifier

    def add_issue(self, issue: str, stage: ProcessingStage):
        """Add an issue from a specific processing stage."""
        self.all_issues.append(f"[{stage.value}] {issue}")

    def aggregate_issues(self):
        """Aggregate issues from stage results."""
        self.all_issues.clear()

        if self.preprocess_result:
            for issue in self.preprocess_result.issues:
                self.add_issue(issue, ProcessingStage.PREPROCESSING)

        if self.postprocess_result:
            for issue in self.postprocess_result.issues:
                self.add_issue(issue, ProcessingStage.POSTPROCESSING)


# ------------------------------------------------------------------------------
# Helper Constructors (Avoid post_init mutations)
# ------------------------------------------------------------------------------

def create_failed_preprocess_result(
        text: str,
        canonical_text: str = "",
        error_message: str = "Processing failed",
        original_length: int = 0,
        processing_duration: float = 0.0
) -> PreprocessResult:
    """
    Create a PreprocessResult indicating failure.

    Uses explicit constructor to avoid post_init side effects.
    """
    # For failed processing, duration is typically 0 or very small
    # Caller should provide actual duration if known

    if not canonical_text:
        canonical_text = ""

    return PreprocessResult(
        text=text,
        canonical_text=canonical_text,
        success=False,
        original_length=original_length or len(text),
        processed_length=len(text) if text else 0,
        estimated_tokens=0,
        was_truncated=False,
        truncated_tokens=0,
        issues=["processing_failed"],
        processing_duration=processing_duration,
        error_message=error_message
    )


def create_service_response_result(
        original_response: str,
        service_category: str,
        processing_duration: float = 0.0
) -> PostprocessResult:
    """
    Create a PostprocessResult for detected service response.

    Uses explicit constructor to avoid post_init side effects.
    """
    return PostprocessResult(
        text="",  # Service responses don't have processable text
        success=False,  # Explicitly set to False for service responses
        response_type=ResponseType.SERVICE,
        is_service_response=True,
        service_category=service_category,
        original_response=original_response,
        cleaning_applied=[],
        issues=["service_response_detected"],
        processing_duration=processing_duration,
        error_message=f"Service response detected: {service_category}"
    )


def create_empty_response_result(
        original_response: str,
        processing_duration: float = 0.0
) -> PostprocessResult:
    """
    Create a PostprocessResult for empty/invalid response.
    """
    return PostprocessResult(
        text="",
        success=False,
        response_type=ResponseType.EMPTY,
        is_service_response=False,
        service_category=None,
        original_response=original_response,
        cleaning_applied=[],
        issues=["empty_or_invalid_response"],
        processing_duration=processing_duration,
        error_message="LLM returned empty or invalid response"
    )


def create_successful_preprocess_result(
        original_text: str,
        processed_text: str,
        canonical_text: str = "",
        estimated_tokens: int = 0,
        was_truncated: bool = False,
        truncated_tokens: int = 0,
        processing_duration: float = 0.0
) -> PreprocessResult:
    """
    Create a successful PreprocessResult with all required fields.

    Explicitly calculates lengths to avoid post_init dependencies.
    """
    if not canonical_text:
        canonical_text = ""

    return PreprocessResult(
        text=processed_text,
        canonical_text=canonical_text,
        success=True,
        original_length=len(original_text),
        processed_length=len(processed_text),
        estimated_tokens=estimated_tokens,
        was_truncated=was_truncated,
        truncated_tokens=truncated_tokens,
        issues=[],
        processing_duration=processing_duration,
        error_message=None
    )


def create_successful_postprocess_result(
        processed_text: str,
        original_response: str,
        cleaning_applied: Optional[List[str]] = None,
        processing_duration: float = 0.0
) -> PostprocessResult:
    """
    Create a successful PostprocessResult with all required fields.
    """
    if cleaning_applied is None:
        cleaning_applied = []

    return PostprocessResult(
        text=processed_text,
        success=True,
        response_type=ResponseType.VALID,
        is_service_response=False,
        service_category=None,
        original_response=original_response,
        cleaning_applied=cleaning_applied,
        issues=[],
        processing_duration=processing_duration,
        error_message=None
    )


# ------------------------------------------------------------------------------
# Batch Processing (Minimal for Phase 1)
# ------------------------------------------------------------------------------

@dataclass(slots=True)
class BatchProcessingStats:
    """
    Essential statistics for batch processing.

    Minimal version focusing only on metrics needed for current implementation.
    """
    total_processed: int = 0  # Total texts processed
    successful_count: int = 0  # Successfully processed count
    failed_count: int = 0  # Failed processing count

    # Essential performance metrics
    total_latency: float = 0.0  # Total batch processing duration in seconds
    cache_hit_count: int = 0  # Number of cache hits

    # Timestamp for tracking
    processing_timestamp: float = field(default_factory=time.time)  # Unix epoch seconds

    def add_latency(self, duration: float):
        """
        Add processing duration to total latency.

        Helper to safely accumulate durations to avoid calculation errors.

        Parameters
        ----------
        duration : float
            Processing duration in seconds to add
        """
        self.total_latency += duration

    def calculate_rates(self) -> Dict[str, float]:
        """Calculate derived rates from basic counts."""
        if self.total_processed == 0:
            return {
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "cache_hit_rate": 0.0,
                "avg_processing_time": 0.0
            }

        return {
            "success_rate": self.successful_count / self.total_processed,
            "failure_rate": self.failed_count / self.total_processed,
            "cache_hit_rate": self.cache_hit_count / self.total_processed,
            "avg_processing_time": self.total_latency / self.total_processed
        }


# ------------------------------------------------------------------------------
# Extension Hooks for Future Development
# ------------------------------------------------------------------------------

class PreprocessResultExtension:
    """
    Mixin class for future preprocessing extensions.

    When NER/entity detection is added, create:
    @dataclass
    class EnhancedPreprocessResult(PreprocessResult, PreprocessResultExtension):
        entities_detected: List[DetectedEntity] = field(default_factory=list)
        pii_analysis: Optional[PIIAnalysis] = None
    """
    pass


class PostprocessResultExtension:
    """
    Mixin class for future postprocessing extensions.

    When PII detection and quality scoring is added, create:
    @dataclass
    class EnhancedPostprocessResult(PostprocessResult, PostprocessResultExtension):
        pii_remaining: bool = False
        quality_score: float = 1.0
        anonymization_quality: float = 1.0
    """
    pass


# ------------------------------------------------------------------------------
# Validation Helpers
# ------------------------------------------------------------------------------

def validate_processing_result(result: ProcessingPipelineResult) -> List[str]:
    """
    Validate a complete processing result for consistency.

    Returns list of validation issues found.
    """
    issues = []

    # Check basic consistency
    if result.success and not result.final_text.strip():
        issues.append("Success=True but final_text is empty")

    # Check stage result consistency
    if result.preprocess_result and not result.preprocess_result.success and result.success:
        issues.append("Preprocessing failed but overall result marked successful")

    if result.postprocess_result and not result.postprocess_result.success and result.success:
        issues.append("Postprocessing failed but overall result marked successful")

    # Check timestamp consistency (more lenient for CI environments)
    # Allow up to 1 hour clock skew to handle slow CI environments and Docker
    if result.processing_timestamp > time.time() + 3600:  # 1 hour instead of 1 minute
        issues.append("Processing timestamp is far in the future (>1 hour)")

    return issues


def ensure_enum_consistency():
    """
    Ensure enum values are consistent across modules during development.

    Should be called during module initialization to catch enum mismatches early.
    """
    # Derive expected values from actual enum instead of hardcoding
    # This prevents false warnings when new enum values are added
    expected_response_types = {rt.value for rt in ResponseType}
    actual_response_types = {rt.value for rt in ResponseType}

    # This check is now redundant since we derive from the same source,
    # but we keep it as a template for validating against external constraints
    if actual_response_types != expected_response_types:
        missing = expected_response_types - actual_response_types
        extra = actual_response_types - expected_response_types
        raise ValueError(
            f"ResponseType enum mismatch. Missing: {missing}, Extra: {extra}"
        )

    # Alternative: Check that critical values exist (more useful)
    required_core_values = {"valid", "service", "error", "empty", "invalid", "pii_detected"}
    if not required_core_values.issubset(actual_response_types):
        missing_core = required_core_values - actual_response_types
        raise ValueError(
            f"ResponseType missing critical values: {missing_core}"
        )


# Run consistency check on import
try:
    ensure_enum_consistency()
except Exception as e:
    # Log warning but don't fail import during development
    import warnings

    warnings.warn(f"Enum consistency check failed: {e}")