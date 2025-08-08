"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Processing Utilities
Package:       pamola_core.utils.nlp.llm.processing
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides utilities for text processing in LLM operations,
including text normalization, truncation, response extraction, marker
management, and batch processing coordination. It handles the complexities
of preparing text for LLM input and processing LLM output.

Key Features:
- Text normalization and cleaning
- Smart text truncation with multiple strategies
- Response extraction from various LLM formats
- Processing marker management for resumable operations
- Service response detection and cleaning
- Batch text processing utilities
- Token counting and estimation
- Text chunking for long documents
- Response validation and quality checks
- Enhanced anonymization response cleaning
- Text canonicalization for consistent cache keys

Framework:
Part of PAMOLA.CORE LLM utilities, providing text processing functions
used throughout the LLM subsystem.

Changelog:
1.2.0 - Added text canonicalization support for cache consistency
     - Enhanced MarkerManager with canonicalization integration
     - Added cache-aware text processing functions
     - Improved marker handling to prevent double-marking
1.1.0 - Enhanced response cleaning for anonymization tasks
     - Added removal of intro phrases and bracketed comments
     - Improved whitespace normalization after cleaning
     - Optimized regex compilation for better performance
1.0.0 - Initial implementation

Dependencies:
- Standard library for core functionality
- Optional: tiktoken for accurate token counting
- Optional: pandas for DataFrame operations

TODO:
- Add support for structured output parsing (JSON, XML)
- Implement language-specific processing rules
- Add text quality scoring metrics
- Support for multi-modal content handling
- Add text compression techniques for context optimization
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

import pandas as pd

from pamola_core.utils.nlp.base import DependencyManager

# Configure logger
logger = logging.getLogger(__name__)

# Constants
TOKEN_CHAR_RATIO = 4  # Average characters per token
DEFAULT_MARKER = "~"
MAX_RESPONSE_ATTEMPTS = 3
MIN_VALID_RESPONSE_LENGTH = 10

# Common service response patterns
SERVICE_RESPONSE_PATTERNS = {
    'request_for_input': [
        r"(?i)^\s*(please|пожалуйста)\s+(provide|предоставьте|дайте|укажите).*",
        r"(?i)^\s*(i need|мне нужн[оы]|необходим[оы]?).*исходн.*текст.*",
        r"(?i)^\s*(give me|дайте мне|предоставьте мне).*текст.*",
        r"(?i)^\s*нужно\s+(еще\s+)?указать.*",
        r"(?i)^\s*(what|какой|что за)\s+(text|текст).*\?",
        r"(?i)^\s*я\s+готов.*помочь.*предоставьте.*",
    ],
    'acknowledgment': [
        r"(?i)^(конечно|sure|да|yes|готов|ready)[\s,.!]*",
        r"(?i)^(я могу|i can|i will|буду рад).*помочь.*",
        r"(?i)^(понял|understood|got it|принято)[\s,.!]*",
    ],
    'error_response': [
        r"(?i)^(извините|sorry|прошу прощения).*не могу.*",
        r"(?i)^(ошибка|error|проблема|problem).*",
        r"(?i)^не удалось.*",
    ],
    'meta_commentary': [
        r"(?i)^\s*\[.*\]\s*$",  # Text in brackets
        r"(?i)^\s*\(.*\)\s*$",  # Text in parentheses
        r"(?i)^примечание:.*",
        r"(?i)^note:.*",
    ]
}

# Compiled patterns for efficiency
COMPILED_SERVICE_PATTERNS: Dict[str, List[Pattern]] = {
    category: [re.compile(pattern) for pattern in patterns]
    for category, patterns in SERVICE_RESPONSE_PATTERNS.items()
}

# Anonymization-specific intro patterns to remove
INTRO_CLEAN_PATTERNS = [
    # Basic acknowledgments and lead-ins
    re.compile(r"^(Вот|Готово|Конечно|Да|Sure|Yes|Here is|Here's)[\s,.:!]*", re.IGNORECASE | re.MULTILINE),
    # Anonymization-specific intros
    re.compile(r"^.*анонимизированн(ая|ый|ое|ые)?\s*(версия|текст|вариант|результат)[\s:]*",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*обезличенн(ая|ый|ое|ые)?\s*(версия|текст|вариант|результат)[\s:]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Результат[\s:]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Анонимизированный текст[\s:]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Обезличенный текст[\s:]*", re.IGNORECASE | re.MULTILINE),
    # English variants
    re.compile(r"^Anonymized\s*(version|text|result)[\s:]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Here\s*is\s*the\s*anonymized.*[\s:]*", re.IGNORECASE | re.MULTILINE),
]

# Patterns for removing bracketed comments
BRACKET_COMMENT_PATTERN = re.compile(r"\([^)]*\)")  # Matches (...)
SQUARE_COMMENT_PATTERN = re.compile(r"\[[^]]*]")  # Matches [...]

# Trailing meta-comments to remove
TRAILING_CLEAN_PATTERNS = [
    re.compile(r"\s*(Надеюсь, это поможет|Hope this helps)[^.!?]*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"\s*(Если нужно|If you need|Let me know)[^.!?]*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"\s*(Обратите внимание|Please note)[^.!?]*[.!?]?\s*$", re.IGNORECASE),
]


def canonicalize_text(text: str, processing_marker: str = DEFAULT_MARKER) -> str:
    """
    Canonicalize text for consistent cache key generation.

    This function ensures that text with different representations (with/without
    markers, different line endings, extra whitespace) will generate the same
    cache key by normalizing the text to a canonical form.

    Parameters
    ----------
    text : str
        Input text to canonicalize
    processing_marker : str
        Processing marker to remove from the beginning (default: "~")

    Returns
    -------
    str
        Canonicalized text suitable for cache key generation

    Notes
    -----
    The canonicalization process:
    1. Handles None/NA values by returning empty string
    2. Converts to string representation
    3. Removes processing marker from the beginning
    4. Normalizes line endings (CRLF/CR -> LF)
    5. Strips leading/trailing whitespace

    This ensures cache consistency regardless of how the text was processed
    or marked previously.
    """
    # Handle null-like values (pandas NA, None, etc.)
    if text is None or (hasattr(text, 'isna') and text.isna()):
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove processing marker if present at the beginning
    # This prevents cache misses for already processed text
    if text.startswith(processing_marker):
        text = text[len(processing_marker):]

    # Normalize line endings for consistent representation
    # Windows (CRLF) and old Mac (CR) -> Unix (LF)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Strip leading/trailing whitespace
    # This handles cases where text has extra spaces/newlines
    text = text.strip()

    return text


class ResponseType(Enum):
    """Types of LLM responses."""
    VALID = "valid"  # Valid processed response
    SERVICE = "service"  # Service/meta response
    ERROR = "error"  # Error response
    EMPTY = "empty"  # Empty or too short response
    INVALID = "invalid"  # Invalid format or content


@dataclass
class ProcessedText:
    """Container for processed text with metadata."""
    text: str
    original_length: int
    processed_length: int
    was_truncated: bool = False
    truncated_tokens: int = 0
    estimated_tokens: int = 0
    processing_notes: List[str] = None

    def __post_init__(self):
        """Initialize processing notes if not provided."""
        if self.processing_notes is None:
            self.processing_notes = []


@dataclass
class ResponseAnalysis:
    """Analysis result for LLM response."""
    response_type: ResponseType
    cleaned_text: str
    original_text: str
    issues: List[str]
    confidence: float = 1.0
    service_category: Optional[str] = None


class TextNormalizer:
    """Utility class for text normalization."""

    @staticmethod
    def normalize(text: Any) -> str:
        """
        Normalize any input to a clean string.

        Parameters
        ----------
        text : Any
            Input text (can be str, None, NaN, etc.)

        Returns
        -------
        str
            Normalized string
        """
        # Handle null-like values
        if text is None or pd.isna(text):
            return ""

        # Handle pandas NA
        if pd.api.types.is_scalar(text) and pd.isna(text):
            return ""

        # Convert to string
        try:
            result = str(text).strip()

            # Check for string representations of null
            if result.lower() in ('nan', 'none', 'null', '<na>', 'n/a', 'na'):
                return ""

            return result
        except Exception:
            return ""

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        Clean up whitespace in text.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        str
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove trailing whitespace from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        return text.strip()

    @staticmethod
    def remove_special_characters(text: str, keep_chars: str = "") -> str:
        """
        Remove special characters from text.

        Parameters
        ----------
        text : str
            Input text
        keep_chars : str
            Additional characters to keep

        Returns
        -------
        str
            Text with special characters removed
        """
        # Define allowed characters
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-\n")
        allowed.update(keep_chars)

        # Add Cyrillic characters
        allowed.update("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

        # Filter characters
        filtered = ''.join(c for c in text if c in allowed)

        return filtered

    @staticmethod
    def normalize_encoding(text: str) -> str:
        """
        Fix common encoding issues.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        str
            Text with encoding issues fixed
        """
        # Common replacements
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
            '\u00a0': ' ',  # Non-breaking space
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text


class TokenEstimator:
    """Utility class for token estimation."""

    def __init__(self, method: str = "simple"):
        """
        Initialize token estimator.

        Parameters
        ----------
        method : str
            Estimation method ('simple', 'tiktoken', 'chars')
        """
        self.method = method
        self._tiktoken_encoding = None

        # Try to load tiktoken if requested
        if method == "tiktoken":
            tiktoken = DependencyManager.get_module('tiktoken')
            if tiktoken:
                try:
                    self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    logger.warning(f"Failed to load tiktoken encoding: {e}")
                    self.method = "simple"

    def estimate(self, text: str) -> int:
        """
        Estimate token count for text.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        int
            Estimated token count
        """
        if not text:
            return 0

        if self.method == "tiktoken" and self._tiktoken_encoding:
            try:
                return len(self._tiktoken_encoding.encode(text))
            except Exception:
                # Fallback to simple method
                pass

        if self.method == "chars":
            # Even simpler character-based estimation
            return len(text)

        # Simple word-based estimation
        # More accurate than pure character division
        words = text.split()
        # Approximate 1.3 tokens per word for English, 2.0 for other languages
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in text)
        tokens_per_word = 2.0 if has_cyrillic else 1.3

        return int(len(words) * tokens_per_word)

    def estimate_with_buffer(self, text: str, buffer_percentage: float = 0.1) -> int:
        """
        Estimate tokens with safety buffer.

        Parameters
        ----------
        text : str
            Input text
        buffer_percentage : float
            Buffer to add (0.1 = 10%)

        Returns
        -------
        int
            Estimated tokens with buffer
        """
        base_estimate = self.estimate(text)
        buffer = int(base_estimate * buffer_percentage)
        return base_estimate + buffer


class TextTruncator:
    """Advanced text truncation utilities."""

    def __init__(self, estimator: Optional[TokenEstimator] = None):
        """
        Initialize truncator.

        Parameters
        ----------
        estimator : TokenEstimator, optional
            Token estimator to use
        """
        self.estimator = estimator or TokenEstimator()

    def truncate(
            self,
            text: str,
            max_tokens: int,
            strategy: str = "smart",
            add_ellipsis: bool = True
    ) -> ProcessedText:
        """
        Truncate text to fit within token limit.

        Parameters
        ----------
        text : str
            Text to truncate
        max_tokens : int
            Maximum token count
        strategy : str
            Truncation strategy ('end', 'middle', 'smart')
        add_ellipsis : bool
            Whether to add ellipsis

        Returns
        -------
        ProcessedText
            Processed text with metadata
        """
        original_length = len(text)
        current_tokens = self.estimator.estimate(text)

        # No truncation needed
        if current_tokens <= max_tokens:
            return ProcessedText(
                text=text,
                original_length=original_length,
                processed_length=original_length,
                was_truncated=False,
                estimated_tokens=current_tokens
            )

        # Calculate target character count
        char_ratio = len(text) / current_tokens if current_tokens > 0 else TOKEN_CHAR_RATIO
        target_chars = int(max_tokens * char_ratio * 0.9)  # 90% to be safe

        # Apply truncation strategy
        if strategy == "end":
            truncated = self._truncate_end(text, target_chars)
        elif strategy == "middle":
            truncated = self._truncate_middle(text, target_chars)
        else:  # smart
            truncated = self._truncate_smart(text, target_chars)

        # Add ellipsis if requested
        if add_ellipsis and not truncated.endswith('...'):
            truncated += '...'

        # Calculate final metrics
        final_tokens = self.estimator.estimate(truncated)
        truncated_tokens = current_tokens - final_tokens

        return ProcessedText(
            text=truncated,
            original_length=original_length,
            processed_length=len(truncated),
            was_truncated=True,
            truncated_tokens=truncated_tokens,
            estimated_tokens=final_tokens,
            processing_notes=[f"Truncated using {strategy} strategy"]
        )

    def _truncate_end(self, text: str, max_chars: int) -> str:
        """Truncate at the end preserving word boundaries."""
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        # Try to truncate at sentence boundary
        last_sentence = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )
        if last_sentence > max_chars * 0.8:
            return truncated[:last_sentence + 1].strip()

        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.9:
            return truncated[:last_space].strip()

        return truncated.strip()

    def _truncate_middle(self, text: str, max_chars: int) -> str:
        """Remove middle portion preserving start and end."""
        if len(text) <= max_chars:
            return text

        # Reserve space for separator
        available_chars = max_chars - 5  # " ... "
        half_chars = available_chars // 2

        start = text[:half_chars]
        end = text[-half_chars:]

        # Try to align with word boundaries
        if start and not start[-1].isspace():
            last_space = start.rfind(' ')
            if last_space > half_chars * 0.8:
                start = start[:last_space]

        if end and not end[0].isspace():
            first_space = end.find(' ')
            if first_space < half_chars * 0.2:
                end = end[first_space + 1:]

        return f"{start.strip()} ... {end.strip()}"

    def _truncate_smart(self, text: str, max_chars: int) -> str:
        """Smart truncation preserving complete units (sentences/paragraphs)."""
        if len(text) <= max_chars:
            return text

        # Try paragraph-based truncation first
        paragraphs = text.split('\n\n')
        result = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para) + (2 if result else 0)  # Account for \n\n
            if current_length + para_length > max_chars:
                break
            result.append(para)
            current_length += para_length

        if result:
            return '\n\n'.join(result)

        # Fall back to sentence-based truncation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        current_length = 0

        for sent in sentences:
            sent_length = len(sent) + (1 if result else 0)  # Account for space
            if current_length + sent_length > max_chars:
                break
            result.append(sent)
            current_length += sent_length

        if result:
            return ' '.join(result)

        # Last resort: truncate at word boundary
        return self._truncate_end(text, max_chars)


class ResponseProcessor:
    """Utility class for processing LLM responses."""

    def __init__(self):
        """Initialize response processor."""
        self.normalizer = TextNormalizer()

    def extract_text(self, response: Any) -> str:
        """
        Extract text from various response formats.

        Parameters
        ----------
        response : Any
            LLM response in any format

        Returns
        -------
        str
            Extracted text
        """
        # Handle None
        if response is None:
            return ""

        # Already a string
        if isinstance(response, str):
            return self.normalizer.normalize(response)

        # Try common attributes
        for attr in ['text', 'content', 'result', 'message', 'output', 'response', 'completion']:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if value is not None:
                    return self.normalizer.normalize(value)

        # Try dictionary access
        if isinstance(response, dict):
            # Try nested structures
            if 'choices' in response and isinstance(response['choices'], list):
                if response['choices']:
                    choice = response['choices'][0]
                    if isinstance(choice, dict):
                        for key in ['text', 'message', 'content']:
                            if key in choice:
                                return self.normalizer.normalize(choice[key])

            # Try direct keys
            for key in ['text', 'content', 'result', 'message', 'output', 'response']:
                if key in response:
                    return self.normalizer.normalize(response[key])

        # Try list access (some APIs return lists)
        if isinstance(response, list) and response:
            return self.extract_text(response[0])

        # Last resort: string conversion
        try:
            text = str(response)
            # Avoid object representations
            if not text.startswith('<') and 'object at 0x' not in text:
                return self.normalizer.normalize(text)
        except Exception:
            pass

        return ""

    def analyze_response(self, response_text: str,
                         expected_min_length: int = MIN_VALID_RESPONSE_LENGTH) -> ResponseAnalysis:
        """
        Analyze LLM response for validity and type.

        Parameters
        ----------
        response_text : str
            Response text to analyze
        expected_min_length : int
            Minimum expected length for valid response

        Returns
        -------
        ResponseAnalysis
            Analysis results
        """
        original = response_text
        cleaned = self.normalizer.clean_whitespace(response_text)
        issues = []

        # Check for empty response
        if not cleaned or len(cleaned.strip()) == 0:
            return ResponseAnalysis(
                response_type=ResponseType.EMPTY,
                cleaned_text="",
                original_text=original,
                issues=["Empty response"]
            )

        # Check for service responses
        service_check = self._check_service_response(cleaned)
        if service_check[0]:
            return ResponseAnalysis(
                response_type=ResponseType.SERVICE,
                cleaned_text=cleaned,
                original_text=original,
                issues=[f"Service response: {service_check[1]}"],
                service_category=service_check[1]
            )

        # Check for error responses
        if self._is_error_response(cleaned):
            return ResponseAnalysis(
                response_type=ResponseType.ERROR,
                cleaned_text=cleaned,
                original_text=original,
                issues=["Error response detected"]
            )

        # Check minimum length
        if len(cleaned) < expected_min_length:
            issues.append(f"Response too short ({len(cleaned)} < {expected_min_length})")

        # Clean the response - enhanced cleaning for anonymization
        cleaned = self._clean_response(cleaned)

        # Determine final type
        if issues:
            response_type = ResponseType.INVALID
        else:
            response_type = ResponseType.VALID

        return ResponseAnalysis(
            response_type=response_type,
            cleaned_text=cleaned,
            original_text=original,
            issues=issues
        )

    def _check_service_response(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if response is a service message."""
        text_lower = text.lower()

        for category, patterns in COMPILED_SERVICE_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(text):
                    return True, category

        # Additional heuristics
        if len(text) < 200:  # Short responses more likely to be service messages
            # Check for question marks at the end
            if text.strip().endswith('?'):
                return True, "question"

            # Check for instruction keywords
            instruction_words = ['предоставьте', 'provide', 'укажите', 'specify', 'дайте', 'give']
            if any(word in text_lower for word in instruction_words):
                return True, "instruction"

        return False, None

    def _is_error_response(self, text: str) -> bool:
        """Check if response indicates an error."""
        error_indicators = [
            'error', 'ошибка', 'failed', 'не удалось',
            'exception', 'исключение', 'unable', 'не могу',
            'cannot', 'не может', 'invalid', 'недопустим'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in error_indicators)

    def _clean_response(self, text: str) -> str:
        """
        Clean response from common artifacts.

        Parameters
        ----------
        text : str
            Response text

        Returns
        -------
        str
            Cleaned text
        """
        # Remove anonymization-specific intro patterns
        for pattern in INTRO_CLEAN_PATTERNS:
            text = pattern.sub("", text)

        # Remove bracketed comments
        text = BRACKET_COMMENT_PATTERN.sub("", text)
        text = SQUARE_COMMENT_PATTERN.sub("", text)

        # Remove trailing meta-comments
        for pattern in TRAILING_CLEAN_PATTERNS:
            text = pattern.sub("", text)

        # Remove markdown artifacts if present
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text)
        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove bold/italic markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic

        # Final whitespace normalization
        text = self.normalizer.clean_whitespace(text)

        return text.strip()


class MarkerManager:
    """
    Enhanced utility class for managing processing markers.

    This class handles the complexities of marker management including:
    - Preventing double-marking of already processed text
    - Integration with text canonicalization for cache consistency
    - Safe marker addition/removal operations
    """

    def __init__(self, marker: str = DEFAULT_MARKER):
        """
        Initialize marker manager.

        Parameters
        ----------
        marker : str
            Processing marker string
        """
        self.marker = marker

    def has_marker(self, text: str) -> bool:
        """
        Check if text has processing marker.

        Parameters
        ----------
        text : str
            Text to check

        Returns
        -------
        bool
            True if text starts with the processing marker
        """
        if not text:
            return False
        normalized = TextNormalizer.normalize(text)
        return normalized.startswith(self.marker)

    def add_marker(self, text: str) -> str:
        """
        Add processing marker to text.

        This method is idempotent - adding a marker to already marked text
        will not result in double-marking.

        Parameters
        ----------
        text : str
            Text to mark

        Returns
        -------
        str
            Marked text
        """
        if not text:
            return f"{self.marker}"

        # Check if already has marker to prevent double-marking
        if self.has_marker(text):
            return text

        return f"{self.marker}{text}"

    def remove_marker(self, text: str) -> str:
        """
        Remove processing marker from text.

        Parameters
        ----------
        text : str
            Text to clean

        Returns
        -------
        str
            Text without marker
        """
        if not text:
            return ""

        if self.has_marker(text):
            return text[len(self.marker):].strip()

        return text

    def extract_marked_and_clean(self, text: str) -> Tuple[bool, str]:
        """
        Extract marker status and clean text.

        This is a convenience method that both checks for marker presence
        and returns the cleaned text in a single operation.

        Returns
        -------
        tuple
            (has_marker, clean_text)
        """
        has_mark = self.has_marker(text)
        clean = self.remove_marker(text) if has_mark else text
        return has_mark, clean

    def get_canonical_text(self, text: str) -> str:
        """
        Get canonical form of text for cache key generation.

        This method integrates with the canonicalize_text function to ensure
        consistent cache keys regardless of marker presence.

        Parameters
        ----------
        text : str
            Text to canonicalize

        Returns
        -------
        str
            Canonical text suitable for cache key generation
        """
        return canonicalize_text(text, self.marker)


class BatchProcessor:
    """Utility class for batch text processing."""

    @staticmethod
    def prepare_batch(
            texts: List[str],
            max_batch_tokens: int,
            estimator: Optional[TokenEstimator] = None
    ) -> List[List[int]]:
        """
        Prepare optimal batches based on token limits.

        Parameters
        ----------
        texts : List[str]
            Texts to batch
        max_batch_tokens : int
            Maximum tokens per batch
        estimator : TokenEstimator, optional
            Token estimator

        Returns
        -------
        List[List[int]]
            List of batch indices
        """
        if not estimator:
            estimator = TokenEstimator()

        batches = []
        current_batch = []
        current_tokens = 0

        for i, text in enumerate(texts):
            text_tokens = estimator.estimate(text)

            # Check if adding this text exceeds limit
            if current_batch and current_tokens + text_tokens > max_batch_tokens:
                batches.append(current_batch)
                current_batch = [i]
                current_tokens = text_tokens
            else:
                current_batch.append(i)
                current_tokens += text_tokens

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    @staticmethod
    def merge_results(
            results: List[Any],
            indices: List[int],
            total_size: int,
            default_value: Any = None
    ) -> List[Any]:
        """
        Merge batch results back to original order.

        Parameters
        ----------
        results : List[Any]
            Batch results
        indices : List[int]
            Original indices
        total_size : int
            Total expected size
        default_value : Any
            Default value for missing indices

        Returns
        -------
        List[Any]
            Merged results in original order
        """
        merged = [default_value] * total_size

        for idx, result in zip(indices, results):
            merged[idx] = result

        return merged


class TextChunker:
    """Utility for chunking long texts."""

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 100,
            estimator: Optional[TokenEstimator] = None
    ):
        """
        Initialize text chunker.

        Parameters
        ----------
        chunk_size : int
            Target chunk size in tokens
        chunk_overlap : int
            Overlap between chunks in tokens
        estimator : TokenEstimator, optional
            Token estimator
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.estimator = estimator or TokenEstimator()

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Parameters
        ----------
        text : str
            Text to chunk

        Returns
        -------
        List[str]
            List of text chunks
        """
        # Estimate total tokens
        total_tokens = self.estimator.estimate(text)

        if total_tokens <= self.chunk_size:
            return [text]

        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.estimator.estimate(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Add current chunk if exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph
                para_chunks = self._split_paragraph(para)
                chunks.extend(para_chunks)

            # Check if adding paragraph exceeds limit
            elif current_tokens + para_tokens > self.chunk_size:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last few paragraphs for overlap
                    overlap_chunks = []
                    overlap_tokens = 0

                    for prev_para in reversed(current_chunk):
                        prev_tokens = self.estimator.estimate(prev_para)
                        if overlap_tokens + prev_tokens <= self.chunk_overlap:
                            overlap_chunks.insert(0, prev_para)
                            overlap_tokens += prev_tokens
                        else:
                            break

                    current_chunk = overlap_chunks + [para]
                    current_tokens = overlap_tokens + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _split_paragraph(self, paragraph: str) -> List[str]:
        """Split large paragraph into chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.estimator.estimate(sent)

            if current_tokens + sent_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


# Convenience functions
def normalize_text(text: Any) -> str:
    """Normalize any input to clean string."""
    return TextNormalizer.normalize(text)


def clean_whitespace(text: str) -> str:
    """Clean up whitespace in text."""
    return TextNormalizer.clean_whitespace(text)


def estimate_tokens(text: str, method: str = "simple") -> int:
    """Estimate token count for text."""
    estimator = TokenEstimator(method)
    return estimator.estimate(text)


def truncate_text(
        text: str,
        max_tokens: int,
        strategy: str = "smart",
        add_ellipsis: bool = True
) -> ProcessedText:
    """Truncate text to fit within token limit."""
    truncator = TextTruncator()
    return truncator.truncate(text, max_tokens, strategy, add_ellipsis)


def extract_response_text(response: Any) -> str:
    """Extract text from LLM response."""
    processor = ResponseProcessor()
    return processor.extract_text(response)


def analyze_response(response_text: str) -> ResponseAnalysis:
    """Analyze LLM response for validity."""
    processor = ResponseProcessor()
    return processor.analyze_response(response_text)


def has_processing_marker(text: str, marker: str = DEFAULT_MARKER) -> bool:
    """Check if text has processing marker."""
    manager = MarkerManager(marker)
    return manager.has_marker(text)


def add_processing_marker(text: str, marker: str = DEFAULT_MARKER) -> str:
    """Add processing marker to text."""
    manager = MarkerManager(marker)
    return manager.add_marker(text)


def remove_processing_marker(text: str, marker: str = DEFAULT_MARKER) -> str:
    """Remove processing marker from text."""
    manager = MarkerManager(marker)
    return manager.remove_marker(text)


def chunk_long_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
) -> List[str]:
    """Split long text into chunks."""
    chunker = TextChunker(chunk_size, chunk_overlap)
    return chunker.chunk_text(text)