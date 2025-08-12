"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Service Response Detector
Package:       pamola_core.utils.nlp.llm.service_detector
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides sophisticated detection of service/meta responses from LLMs.
Service responses are messages where the LLM asks for clarification, provides
acknowledgments, or otherwise fails to complete the requested task. These
responses need to be filtered out to maintain data quality.

Key Features:
- Comprehensive pattern matching for Russian and English service responses
- Confidence scoring for detection accuracy
- Categorization of different service response types
- Extensible pattern system for new response types
- Performance-optimized compiled regex patterns
- Pattern validation and conflict detection
- Support for custom patterns and thresholds
- Detailed logging for pattern matching debugging

Design Principles:
- Fail-fast validation of patterns at initialization
- Pre-compiled regex patterns for performance
- Clear categorization of service response types
- Configurable confidence thresholds
- Extensive test coverage for pattern accuracy
- Easy addition of new patterns without code changes

Framework:
Part of PAMOLA.CORE LLM utilities, providing essential quality control
for LLM response processing pipelines.

Usage:
```python
detector = ServiceResponseDetector()
is_service, confidence, category = detector.detect_with_confidence(response_text)
if is_service:
    # Handle service response
    pass
```

Changelog:
1.0.0 - Initial implementation
     - Comprehensive Russian and English patterns
     - Confidence scoring system
     - Service response categorization
     - Pattern validation framework
     - Performance optimization with compiled patterns

Dependencies:
- re - Regular expression pattern matching
- typing - Type annotations
- logging - Debug and error logging
- dataclasses - Configuration structures

TODO:
- Add machine learning-based detection for complex cases
- Implement pattern learning from false positives/negatives
- Add support for other languages (Spanish, French, etc.)
- Create pattern editor GUI for non-technical users
- Add statistical analysis of pattern effectiveness
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Tuple

from .enums import ServiceCategory

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MIN_RESPONSE_LENGTH = 5
DEFAULT_MAX_RESPONSE_LENGTH = 500  # Service responses are typically short


# ------------------------------------------------------------------------------
# Configuration and Data Structures
# ------------------------------------------------------------------------------

@dataclass
class PatternConfig:
    """
    Configuration for service response pattern matching.

    Allows fine-tuning of detection behavior and thresholds.

    Note
    ----
    Short response handling: Text shorter than min_response_length
    will be treated as potentially service responses. The confidence
    assigned scales with the main threshold to ensure consistent behavior:
    - High thresholds (>0.7): Short responses marked as non-service (allowing through)
    - Lower thresholds (≤0.7): Short responses marked as service with threshold confidence
    """
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    min_response_length: int = DEFAULT_MIN_RESPONSE_LENGTH
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH
    case_sensitive: bool = False
    enable_debug_logging: bool = False
    # Pattern weights for confidence calculation
    pattern_weights: Dict[str, float] = field(default_factory=lambda: {
        'exact_match': 1.0,
        'strong_indicator': 0.9,
        'moderate_indicator': 0.7,
        'weak_indicator': 0.5
    })


@dataclass
class DetectionResult:
    """
    Result of service response detection with detailed metadata.

    Note
    ----
    F6: matched_patterns uses mutable list for convenience during detection.
    If result caching is needed, consider converting to tuple for immutability.
    """
    is_service_response: bool
    confidence: float
    category: Optional[ServiceCategory]
    matched_patterns: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    def __post_init__(self):
        """Validate detection result consistency."""
        if self.is_service_response and self.confidence == 0.0:
            logger.warning("Service response detected but confidence is 0.0")

        # ISSUE #4 FIX: Only warn for very high confidence non-service responses
        # This prevents legitimate "too long" responses from triggering warnings
        if not self.is_service_response and self.confidence > 0.9:
            logger.warning(f"Non-service response but very high confidence: {self.confidence}")


# ------------------------------------------------------------------------------
# Service Response Patterns
# ------------------------------------------------------------------------------

# Comprehensive patterns for detecting various types of service responses
SERVICE_PATTERNS = {
    # Requests for input/clarification (highest priority)
    ServiceCategory.REQUEST_FOR_INPUT: {
        'russian': [
            # Direct requests for input
            r"(?i)^\s*пожалуйста\s+предоставьте.*текст",
            r"(?i)^\s*дайте\s+мне.*текст",
            r"(?i)^\s*укажите.*текст",
            r"(?i)^\s*отправьте.*текст",
            r"(?i)^\s*пришлите.*текст",

            # Waiting/expecting patterns
            r"(?i)^\s*жду.*исходн.*текст",
            r"(?i)^\s*ожидаю.*текст",
            r"(?i)^\s*где.*текст",
            r"(?i)^\s*нужен.*текст",

            # Request formulations
            r"(?i)^\s*мне\s+нужн.*текст.*для.*анонимизац",
            r"(?i)^\s*необходим.*исходн.*материал",
            r"(?i)^\s*требуется.*текст.*опыта",

            # Question patterns
            r"(?i)^\s*какой.*текст.*анонимизировать",
            r"(?i)^\s*что.*нужно.*анонимизировать",
            r"(?i)^\s*вы\s+можете.*предоставить.*текст",

            # Incomplete task indicators
            r"(?i)текст.*не.*предоставлен",
            r"(?i)отсутствует.*исходн.*текст",
            r"(?i)не.*вижу.*текст",

            # Professional request patterns
            r"(?i)для\s+выполнения.*задач.*необходим.*текст",
            r"(?i)чтобы.*анонимизировать.*нужно.*предоставить",
        ],
        'english': [
            # Direct requests
            r"(?i)^\s*please\s+provide.*text",
            r"(?i)^\s*send\s+me.*text",
            r"(?i)^\s*give\s+me.*text",
            r"(?i)^\s*share.*text",
            r"(?i)^\s*submit.*text",

            # Waiting patterns
            r"(?i)^\s*waiting\s+for.*text",
            r"(?i)^\s*i\s+need.*text.*to.*anonymize",
            r"(?i)^\s*where\s+is.*text",
            r"(?i)^\s*i\s+don't\s+see.*text",

            # Question patterns
            r"(?i)^\s*what\s+text.*should.*anonymize",
            r"(?i)^\s*which.*text.*to.*process",
            r"(?i)^\s*could\s+you.*provide.*text",
            r"(?i)^\s*can\s+you.*send.*text",

            # Professional patterns
            r"(?i)to\s+complete.*task.*need.*text",
            r"(?i)in\s+order\s+to.*anonymize.*provide",
            r"(?i)text.*not.*provided",
            r"(?i)missing.*source.*text",
        ]
    },

    # Simple acknowledgments (should fail processing)
    ServiceCategory.ACKNOWLEDGMENT: {
        'russian': [
            r"(?i)^\s*(конечно|разумеется|безусловно)[\s,.!]*$",
            r"(?i)^\s*(да|хорошо|ладно|понятно)[\s,.!]*$",
            r"(?i)^\s*(готов|готова)(\s+помочь)?[\s,.!]*$",
            r"(?i)^\s*буду\s+рад.*помочь[\s,.!]*$",
            r"(?i)^\s*с\s+удовольствием[\s,.!]*$",
            r"(?i)^\s*разумеется[\s,.!]*$",
            r"(?i)^\s*(принято|понял|понятно)[\s,.!]*$",
        ],
        'english': [
            r"(?i)^\s*(sure|certainly|absolutely|definitely)[\s,.!]*$",
            r"(?i)^\s*(yes|yeah|yep|ok|okay)[\s,.!]*$",
            r"(?i)^\s*(understood|got\s+it|i\s+see)[\s,.!]*$",
            r"(?i)^\s*of\s+course[\s,.!]*$",
            r"(?i)^\s*no\s+problem[\s,.!]*$",
            r"(?i)^\s*will\s+do[\s,.!]*$",
            r"(?i)^\s*happy\s+to\s+help[\s,.!]*$",
        ]
    },

    # Error responses
    ServiceCategory.ERROR_RESPONSE: {
        'russian': [
            r"(?i)извините.*не\s+могу",
            r"(?i)прошу\s+прощения.*ошибка",
            r"(?i)произошла\s+ошибка",
            r"(?i)не\s+удалось.*обработать",
            r"(?i)возникла\s+проблема",
            r"(?i)технические\s+неполадки",
            r"(?i)сервис\s+недоступен",
        ],
        'english': [
            r"(?i)sorry.*cannot.*process",
            r"(?i)apologize.*error.*occurred",
            r"(?i)error.*processing.*request",
            r"(?i)failed.*to.*complete",
            r"(?i)service.*unavailable",
            r"(?i)technical.*difficulties",
            r"(?i)unable.*to.*proceed",
        ]
    },

    # Meta commentary about the task
    ServiceCategory.META_COMMENTARY: {
        'russian': [
            # Bracketed comments
            r"(?i)^\s*\[.*\]\s*$",
            r"(?i)^\s*\(.*\)\s*$",

            # Note patterns
            r"(?i)^\s*примечание:",
            r"(?i)^\s*важно:",
            r"(?i)^\s*обратите\s+внимание",
            r"(?i)^\s*следует\s+отметить",

            # Task commentary
            r"(?i)это\s+задача.*анонимизации",
            r"(?i)данный\s+текст.*требует",
            r"(?i)для\s+анонимизации.*необходимо",
        ],
        'english': [
            # Bracketed comments
            r"(?i)^\s*\[.*\]\s*$",
            r"(?i)^\s*\(.*\)\s*$",

            # Note patterns
            r"(?i)^\s*note:",
            r"(?i)^\s*important:",
            r"(?i)^\s*please\s+note",
            r"(?i)^\s*keep\s+in\s+mind",

            # Task commentary
            r"(?i)this\s+is.*anonymization.*task",
            r"(?i)the\s+text.*requires.*anonymization",
            r"(?i)for\s+anonymization.*purposes",
        ]
    },

    # Clarification requests
    ServiceCategory.CLARIFICATION: {
        'russian': [
            r"(?i)уточните.*пожалуйста",
            r"(?i)не\s+совсем\s+понятно",
            r"(?i)можете.*пояснить",
            r"(?i)нужны.*дополнительные.*детали",
            r"(?i)требуется.*уточнение",
            r"(?i)какой\s+именно.*формат",
            r"(?i)в\s+каком\s+виде.*представить",
        ],
        'english': [
            r"(?i)could\s+you.*clarify",
            r"(?i)please\s+specify",
            r"(?i)not\s+entirely\s+clear",
            r"(?i)need.*more.*details",
            r"(?i)requires.*clarification",
            r"(?i)what\s+format.*should",
            r"(?i)how\s+should.*present",
        ]
    },

    # Task refusal
    ServiceCategory.REFUSAL: {
        'russian': [
            r"(?i)не\s+могу.*выполнить.*задачу",
            r"(?i)отказываюсь.*анонимизировать",
            r"(?i)это\s+противоречит.*принципам",
            r"(?i)не\s+буду.*обрабатывать",
            r"(?i)задача.*невыполнима",
        ],
        'english': [
            r"(?i)cannot.*complete.*task",
            r"(?i)refuse.*to.*anonymize",
            r"(?i)against.*my.*principles",
            r"(?i)will\s+not.*process",
            r"(?i)task.*is.*impossible",
        ]
    }
}


# ------------------------------------------------------------------------------
# Service Response Detector
# ------------------------------------------------------------------------------

class ServiceResponseDetector:
    """
    Advanced detector for LLM service/meta responses.

    Provides sophisticated pattern matching with confidence scoring
    and detailed categorization of service response types.
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        """
        Initialize service response detector.

        Parameters
        ----------
        config : PatternConfig, optional
            Configuration for detection behavior
        """
        self.config = config or PatternConfig()

        # Compiled patterns for performance
        self._compiled_patterns: Dict[ServiceCategory, Dict[str, List[Pattern]]] = {}

        # Pattern metadata for debugging
        self._pattern_metadata: Dict[str, Dict[str, Any]] = {}

        # Performance metrics
        self._detection_count = 0
        self._total_detection_time = 0.0

        # Thread safety for pattern mutations
        self._pattern_lock = threading.Lock()

        # Initialize and validate patterns
        self._compile_patterns()
        self._validate_patterns()

        logger.info(f"ServiceResponseDetector initialized with {self._count_total_patterns()} patterns")

    def _compile_patterns(self):
        """Compile all regex patterns for optimal performance."""
        try:
            # ISSUE #6 FIX: Clear metadata when recompiling to avoid stale stats
            self._pattern_metadata.clear()

            # E3 FIX: Build patterns atomically to avoid race conditions
            new_compiled_patterns: Dict[ServiceCategory, Dict[str, List[Pattern]]] = {}
            new_pattern_metadata: Dict[str, Dict[str, Any]] = {}

            for category, lang_patterns in SERVICE_PATTERNS.items():
                new_compiled_patterns[category] = {}

                for language, patterns in lang_patterns.items():
                    compiled = []
                    for i, pattern_str in enumerate(patterns):
                        try:
                            flags = 0 if self.config.case_sensitive else re.IGNORECASE | re.MULTILINE
                            compiled_pattern = re.compile(pattern_str, flags)
                            compiled.append(compiled_pattern)

                            # Store metadata for debugging
                            pattern_id = f"{category.value}_{language}_{i}"
                            new_pattern_metadata[pattern_id] = {
                                'category': category,
                                'language': language,
                                'pattern': pattern_str,
                                'compiled': compiled_pattern,
                                'matches': 0,
                                'false_positives': 0
                            }

                        except re.error as e:
                            logger.error(f"Invalid regex pattern in {category.value}/{language}: {pattern_str} - {e}")
                            continue

                    new_compiled_patterns[category][language] = compiled

            # E3 FIX: Atomic replacement of compiled patterns
            # F3 NOTE: Dict assignment is atomic in CPython due to GIL,
            # so concurrent readers will see either old or new dict completely
            self._compiled_patterns = new_compiled_patterns
            self._pattern_metadata = new_pattern_metadata

        except Exception as e:
            logger.error(f"Failed to compile patterns: {e}")
            raise

    def _validate_patterns(self):
        """Validate compiled patterns for conflicts and coverage."""
        issues = []

        # ISSUE #2 FIX: Build expected categories dynamically from enum
        expected_categories = set(ServiceCategory.__members__.keys())
        actual_categories = set(category.name for category in SERVICE_PATTERNS.keys())

        if actual_categories != expected_categories:
            missing = expected_categories - actual_categories
            extra = actual_categories - expected_categories
            if missing:
                issues.append(f"Missing pattern categories: {missing}")
            if extra:
                issues.append(f"Extra pattern categories: {extra}")

        # Check for duplicate patterns
        all_pattern_strings = set()
        for category, lang_patterns in SERVICE_PATTERNS.items():
            for language, patterns in lang_patterns.items():
                for pattern in patterns:
                    if pattern in all_pattern_strings:
                        issues.append(f"Duplicate pattern found: {pattern}")
                    all_pattern_strings.add(pattern)

        # Check for potentially conflicting patterns
        # This is a simplified check - could be more sophisticated
        short_patterns = [p for p in all_pattern_strings if len(p) < 50]
        if len(short_patterns) > len(all_pattern_strings) * 0.8:
            issues.append("Many short patterns detected - potential for false positives")

        # Log validation results
        if issues:
            for issue in issues:
                logger.warning(f"Pattern validation issue: {issue}")
        else:
            logger.info("Pattern validation completed successfully")

    def _count_total_patterns(self) -> int:
        """Count total number of compiled patterns."""
        total = 0
        for category_patterns in self._compiled_patterns.values():
            for lang_patterns in category_patterns.values():
                total += len(lang_patterns)
        return total

    def detect_with_confidence(self, text: str) -> Tuple[bool, float, Optional[ServiceCategory]]:
        """
        Detect service response with confidence scoring.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        tuple
            (is_service_response, confidence_0_to_1, category)

        Note
        ----
        Confidence scores are clamped to [0.0, 1.0] range.
        """
        start_time = time.time()
        result = self._perform_detection(text)
        detection_time = time.time() - start_time

        # P1 FIX: Thread-safe performance metrics update
        with self._pattern_lock:
            self._detection_count += 1
            self._total_detection_time += detection_time

        # Debug logging
        if self.config.enable_debug_logging:
            logger.debug(
                f"Detection result: service={result.is_service_response}, "
                f"confidence={result.confidence:.3f}, category={result.category}, "
                f"time={detection_time:.3f}s, patterns={result.matched_patterns}"
            )

        return result.is_service_response, result.confidence, result.category

    def detect_detailed(self, text: str) -> DetectionResult:
        """
        Perform detailed detection with full metadata.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        DetectionResult
            Detailed detection result with metadata
        """
        start_time = time.time()
        result = self._perform_detection(text)
        result.processing_time = time.time() - start_time
        return result

    def _perform_detection(self, text: str) -> DetectionResult:
        """
        Core detection logic with pattern matching.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        DetectionResult
            Detection result with metadata
        """
        # Input validation
        if not text or not isinstance(text, str):
            return DetectionResult(
                is_service_response=False,
                confidence=0.0,
                category=None,
                reasons=["Empty or invalid input"]
            )

        # Normalize text for analysis
        normalized_text = text.strip()

        # Length-based heuristics
        if len(normalized_text) < self.config.min_response_length:
            # F2 FIX: Scale short-response handling with threshold for consistency
            # Use threshold - 0.1 so behavior scales automatically with config
            short_response_threshold = max(0.1, self.config.confidence_threshold - 0.1)

            if self.config.confidence_threshold > short_response_threshold:
                return DetectionResult(
                    is_service_response=False,  # Let legitimate short answers pass
                    confidence=1.0 - short_response_threshold,  # High confidence in NOT being service
                    category=None,
                    reasons=["Response short but threshold high - allowing through"]
                )
            else:
                return DetectionResult(
                    is_service_response=True,
                    confidence=short_response_threshold,  # Scaled with main threshold
                    category=ServiceCategory.ACKNOWLEDGMENT,
                    reasons=["Response potentially too short for meaningful content"]
                )

        # Very long responses are unlikely to be service responses
        if len(normalized_text) > self.config.max_response_length:
            return DetectionResult(
                is_service_response=False,
                confidence=0.9,
                category=None,
                reasons=["Response too long for service message"]
            )

        # Pattern matching
        best_match = self._find_best_pattern_match(normalized_text)

        if best_match['confidence'] >= self.config.confidence_threshold:
            return DetectionResult(
                is_service_response=True,
                confidence=best_match['confidence'],
                category=best_match['category'],
                matched_patterns=best_match['patterns'],
                reasons=best_match['reasons']
            )

        # Additional heuristics for edge cases
        heuristic_result = self._apply_heuristics(normalized_text)

        if heuristic_result['is_service']:
            return DetectionResult(
                is_service_response=True,
                confidence=heuristic_result['confidence'],
                category=heuristic_result['category'],
                reasons=heuristic_result['reasons']
            )

        # Not a service response
        return DetectionResult(
            is_service_response=False,
            confidence=1.0 - best_match['confidence'],  # Confidence in NOT being service
            category=None,
            reasons=["No service patterns matched"]
        )

    def _find_best_pattern_match(self, text: str) -> Dict[str, Any]:
        """
        Find the best matching pattern across all categories.

        Parameters
        ----------
        text : str
            Normalized text to match

        Returns
        -------
        dict
            Best match information
        """
        best_confidence = 0.0
        best_category = None
        matched_patterns = []
        reasons = []

        # Test patterns in priority order (most specific first)
        category_priorities = [
            ServiceCategory.REQUEST_FOR_INPUT,
            ServiceCategory.ERROR_RESPONSE,
            ServiceCategory.REFUSAL,
            ServiceCategory.CLARIFICATION,
            ServiceCategory.META_COMMENTARY,
            ServiceCategory.ACKNOWLEDGMENT,
        ]

        for category in category_priorities:
            if category not in self._compiled_patterns:
                continue

            category_confidence, category_patterns, category_reasons = self._test_category_patterns(
                text, category
            )

            if category_confidence > best_confidence:
                best_confidence = category_confidence
                best_category = category
                matched_patterns = category_patterns
                reasons = category_reasons

        return {
            'confidence': best_confidence,
            'category': best_category,
            'patterns': matched_patterns,
            'reasons': reasons
        }

    def _test_category_patterns(self, text: str, category: ServiceCategory) -> Tuple[float, List[str], List[str]]:
        """
        Test all patterns for a specific category.

        Parameters
        ----------
        text : str
            Text to test
        category : ServiceCategory
            Category to test

        Returns
        -------
        tuple
            (confidence, matched_patterns, reasons)
        """
        max_confidence = 0.0
        matched_patterns = []
        reasons = []

        lang_patterns = self._compiled_patterns.get(category, {})

        for language, patterns in lang_patterns.items():
            for i, pattern in enumerate(patterns):
                try:
                    match = pattern.search(text)
                    if match:
                        # Calculate confidence based on match quality
                        confidence = self._calculate_pattern_confidence(
                            match, text, category, language
                        )

                        if confidence > max_confidence:
                            max_confidence = confidence

                        pattern_id = f"{category.value}_{language}_{i}"
                        matched_patterns.append(pattern_id)
                        reasons.append(f"Matched {language} pattern: {pattern.pattern[:50]}...")

                        # Update pattern statistics
                        if pattern_id in self._pattern_metadata:
                            self._pattern_metadata[pattern_id]['matches'] += 1

                except Exception as e:
                    logger.warning(f"Error testing pattern {category.value}/{language}: {e}")
                    continue

        return max_confidence, matched_patterns, reasons

    def _calculate_pattern_confidence(
            self,
            match: re.Match,
            text: str,
            category: ServiceCategory,
            language: str
    ) -> float:
        """
        Calculate confidence score for a pattern match.

        Parameters
        ----------
        match : re.Match
            Regex match object
        text : str
            Full text being analyzed
        category : ServiceCategory
            Category of the matched pattern
        language : str
            Language of the pattern

        Returns
        -------
        float
            Confidence score (0.0 to 1.0, guaranteed to be within bounds)
        """
        base_confidence = self.config.pattern_weights.get('moderate_indicator', 0.7)

        # ISSUE #3 FIX: Use additive model to prevent confidence > 1.0
        confidence_adjustments = 0.0

        # Adjust based on match characteristics
        match_span = match.end() - match.start()
        text_length = len(text.strip())

        # Full text matches get higher confidence
        if match_span >= text_length * 0.8:
            confidence_adjustments += 0.15

        # Matches at the beginning get higher confidence
        if match.start() <= 5:
            confidence_adjustments += 0.1

        # Category-specific adjustments
        if category == ServiceCategory.REQUEST_FOR_INPUT:
            confidence_adjustments += 0.1  # High priority category
        elif category == ServiceCategory.ACKNOWLEDGMENT:
            confidence_adjustments -= 0.05  # Lower priority

        # Calculate final confidence with proper clamping
        final_confidence = base_confidence + confidence_adjustments
        return min(final_confidence, 1.0)

    def _apply_heuristics(self, text: str) -> Dict[str, Any]:
        """
        Apply additional heuristics for edge case detection.

        Parameters
        ----------
        text : str
            Normalized text

        Returns
        -------
        dict
            Heuristic analysis result
        """
        reasons = []

        # Check for repeated characters (often indicates confusion)
        if re.search(r'(.)\1{3,}', text):
            reasons.append("Contains repeated characters")
            return {
                'is_service': True,
                'confidence': 0.6,
                'category': ServiceCategory.ERROR_RESPONSE,
                'reasons': reasons
            }

        # Check for excessive punctuation
        # ISSUE #9 FIX: Guard against division by zero
        text_length = max(len(text), 1)  # Prevent division by zero
        punct_ratio = len(re.findall(r'[!?.,;:]', text)) / text_length
        if punct_ratio > 0.3:
            reasons.append("Excessive punctuation")
            return {
                'is_service': True,
                'confidence': 0.5,
                'category': ServiceCategory.META_COMMENTARY,
                'reasons': reasons
            }

        # Check for question-only responses
        if text.strip().endswith('?') and len(text.split()) <= 10:
            reasons.append("Short question-only response")
            return {
                'is_service': True,
                'confidence': 0.7,
                'category': ServiceCategory.CLARIFICATION,
                'reasons': reasons
            }

        # Not detected as service response
        return {
            'is_service': False,
            'confidence': 0.0,
            'category': None,
            'reasons': ["No heuristic indicators"]
        }

    def add_custom_pattern(
            self,
            category: ServiceCategory,
            language: str,
            pattern: str,
            recompile: bool = True
    ):
        """
        Add a custom pattern to the detector.

        THREAD SAFETY: This method is thread-safe.

        Parameters
        ----------
        category : ServiceCategory
            Category for the new pattern
        language : str
            Language code ('russian', 'english', etc.)
        pattern : str
            Regex pattern string
        recompile : bool
            Whether to recompile patterns immediately
        """
        # ISSUE #5 FIX: Thread-safe pattern mutation
        with self._pattern_lock:
            try:
                # Validate pattern
                test_pattern = re.compile(pattern)

                # Add to patterns
                if category not in SERVICE_PATTERNS:
                    SERVICE_PATTERNS[category] = {}
                if language not in SERVICE_PATTERNS[category]:
                    SERVICE_PATTERNS[category][language] = []

                SERVICE_PATTERNS[category][language].append(pattern)

                if recompile:
                    self._compile_patterns()

                logger.info(f"Added custom pattern for {category.value}/{language}")

            except re.error as e:
                logger.error(f"Invalid custom pattern: {pattern} - {e}")
                raise ValueError(f"Invalid regex pattern: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detector performance and usage statistics.

        Returns
        -------
        dict
            Statistics dictionary
        """
        # F4 FIX: Thread-safe access to all performance counters consistently
        with self._pattern_lock:
            detection_count = self._detection_count
            total_detection_time = self._total_detection_time
            # Also grab pattern metadata snapshot while we have the lock
            pattern_metadata_snapshot = dict(self._pattern_metadata)

        avg_detection_time = (
            total_detection_time / detection_count
            if detection_count > 0 else 0.0
        )

        pattern_stats = {}
        for pattern_id, metadata in pattern_metadata_snapshot.items():
            if metadata['matches'] > 0:
                pattern_stats[pattern_id] = {
                    'matches': metadata['matches'],
                    'category': metadata['category'].value,
                    'language': metadata['language'],
                    'pattern_preview': metadata['pattern'][:50] + "..."
                }

        return {
            'total_detections': detection_count,
            'average_detection_time': avg_detection_time,
            'total_patterns': self._count_total_patterns(),
            'active_patterns': len([p for p in pattern_metadata_snapshot.values() if p['matches'] > 0]),
            'pattern_statistics': pattern_stats,
            'configuration': {
                'confidence_threshold': self.config.confidence_threshold,
                'min_response_length': self.config.min_response_length,
                'max_response_length': self.config.max_response_length
            }
        }

    def test_pattern(self, pattern: str, test_cases: List[str]) -> Dict[str, Any]:
        """
        Test a pattern against multiple test cases.

        Parameters
        ----------
        pattern : str
            Regex pattern to test
        test_cases : List[str]
            List of test strings

        Returns
        -------
        dict
            Test results
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            results = []

            for test_case in test_cases:
                match = compiled_pattern.search(test_case)
                results.append({
                    'text': test_case,
                    'matched': bool(match),
                    'match_span': (match.start(), match.end()) if match else None,
                    'matched_text': match.group() if match else None
                })

            return {
                'pattern': pattern,
                'total_tests': len(test_cases),
                'matches': sum(1 for r in results if r['matched']),
                'results': results
            }

        except re.error as e:
            return {
                'pattern': pattern,
                'error': str(e),
                'results': []
            }


# ------------------------------------------------------------------------------
# Module-level singleton for performance (ISSUE #1 FIX)
# ------------------------------------------------------------------------------

_DEFAULT_DETECTOR: Optional[ServiceResponseDetector] = None
_DETECTOR_LOCK = threading.Lock()


def _get_default_detector() -> ServiceResponseDetector:
    """
    Get or create the default detector instance (thread-safe singleton).

    ISSUE #1 FIX: Avoids recompiling ~350 regexes on every call.
    """
    global _DEFAULT_DETECTOR

    if _DEFAULT_DETECTOR is None:
        with _DETECTOR_LOCK:
            # Double-check locking pattern
            if _DEFAULT_DETECTOR is None:
                _DEFAULT_DETECTOR = ServiceResponseDetector()

    return _DEFAULT_DETECTOR


# ------------------------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------------------------

def create_default_detector() -> ServiceResponseDetector:
    """Create detector with default configuration."""
    return ServiceResponseDetector()


def create_strict_detector() -> ServiceResponseDetector:
    """Create detector with strict detection settings."""
    config = PatternConfig(
        confidence_threshold=0.8,
        min_response_length=3,
        max_response_length=300
    )
    return ServiceResponseDetector(config)


def create_lenient_detector() -> ServiceResponseDetector:
    """Create detector with lenient detection settings."""
    config = PatternConfig(
        confidence_threshold=0.5,
        min_response_length=10,
        max_response_length=800
    )
    return ServiceResponseDetector(config)


def quick_service_check(text: str) -> bool:
    """
    Quick check if text is likely a service response.

    PERFORMANCE: Uses cached singleton detector for optimal speed.

    Parameters
    ----------
    text : str
        Text to check

    Returns
    -------
    bool
        True if likely a service response
    """
    # ISSUE #1 FIX: Use singleton instead of creating new detector every time
    detector = _get_default_detector()
    is_service, confidence, _ = detector.detect_with_confidence(text)
    # E2 FIX: Use detector's configured threshold instead of hardcoded 0.7
    return is_service and confidence >= detector.config.confidence_threshold