"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Text Preprocessing
Package:       pamola_core.utils.nlp.llm.preprocessing
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides comprehensive text preprocessing functionality for LLM
operations. It handles text validation, normalization, canonicalization,
token estimation, truncation, and marker management. Designed as part of
the modular LLM processing pipeline, it prepares text for optimal LLM
processing while maintaining compatibility with caching and resumable
operations.

Key Features:
- Comprehensive text validation and early filtering
- Advanced text normalization (encoding, whitespace, special chars)
- Text canonicalization for consistent cache keys
- Token estimation and intelligent truncation
- Processing marker management for resumable operations
- Extensible architecture with NER/entity detection hooks
- Memory-efficient processing with minimal overhead
- Thread-safe operations for concurrent processing

Design Principles:
- Fail-fast validation to avoid wasting LLM resources
- Preserve original text information for debugging
- Consistent interfaces using data contracts
- Extensible architecture for future enhancements
- Backward compatibility with existing processing.py

Framework:
Part of PAMOLA.CORE LLM subsystem, providing the preprocessing stage
of the text transformation pipeline.

Changelog:
1.0.0 - Initial implementation
    - Migrated functionality from processing.py
    - Added TextPreprocessor main class
    - Integrated with data contracts
    - Added NER hooks for future expansion

Dependencies:
- re - Regular expression operations
- logging - Debug and error logging
- dataclasses - Data structures
- typing - Type annotations
- pandas - DataFrame operations
- numpy - Numerical operations

TODO:
- Implement EntityDetector for NER integration
- Add language detection for multilingual support
- Implement custom tokenizers for domain-specific text
- Add text quality scoring metrics
- Support for structured text preprocessing
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union

import numpy as np
import pandas as pd

# Import from existing modules for compatibility
from ..base import DependencyManager
from .data_contracts import (
    PreprocessResult,
    create_failed_preprocess_result,
    create_successful_preprocess_result
)
from .enums import ProcessingStage, TruncationStrategy, TokenEstimationMethod

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MARKER = "~"
TOKEN_CHAR_RATIO = 4  # Average characters per token
MIN_VALID_TEXT_LENGTH = 3   # Minimum meaningful text length
MAX_TEXT_LENGTH = 50000   # Maximum text length to prevent DOS

# Encoding fix patterns
ENCODING_REPLACEMENTS = {
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

# Special character sets for different languages
ALLOWED_CHARS = {
    'latin': set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    'cyrillic': set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"),
    'digits': set("0123456789"),
    # ИСПРАВЛЕНИЕ ЗДЕСЬ: Используем тройные двойные кавычки для удобства включения одинарных и двойных кавычек
    'punctuation': set(""" .,!?-–—:;'"()[]{}«»„‚'"""),
    'special': set("\n\r\t@#$%^&*+=_/\\|~`")
}


# ------------------------------------------------------------------------------
# Text Normalization Components
# ------------------------------------------------------------------------------

class TextNormalizer:
    """
    Advanced text normalization utilities.

    Handles various text cleaning operations including encoding fixes,
    whitespace normalization, and special character handling.
    """

    def __init__(self, preserve_special_chars: bool = False):
        """
        Initialize text normalizer.

        Parameters
        ----------
        preserve_special_chars : bool
            Whether to preserve special characters during normalization
        """
        self.preserve_special_chars = preserve_special_chars
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for performance."""
        return {
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n\s*\n\s*\n+'),
            'trailing_whitespace': re.compile(r'[ \t]+$', re.MULTILINE),
            'control_chars': re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]'),
            'url_pattern': re.compile(r'https?://[^\s]+'),
            'email_pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        }

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

    def clean_whitespace(self, text: str) -> str:
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
        if not text:
            return ""

        # Replace multiple spaces with single space
        text = self._compiled_patterns['multiple_spaces'].sub(' ', text)

        # Normalize newlines
        text = self._compiled_patterns['multiple_newlines'].sub('\n\n', text)

        # Remove trailing whitespace from lines
        text = self._compiled_patterns['trailing_whitespace'].sub('', text)

        return text.strip()

    def fix_encoding(self, text: str) -> str:
        """
        Fix common encoding issues.

        Parameters
        ----------
        text : str
            Input text with potential encoding issues

        Returns
        -------
        str
            Text with encoding issues fixed
        """
        for old, new in ENCODING_REPLACEMENTS.items():
            text = text.replace(old, new)

        # Remove control characters
        text = self._compiled_patterns['control_chars'].sub('', text)

        return text

    def remove_special_characters(
            self,
            text: str,
            keep_chars: Optional[str] = None,
            preserve_structure: bool = True
    ) -> str:
        """
        Remove special characters from text.

        Parameters
        ----------
        text : str
            Input text
        keep_chars : str, optional
            Additional characters to keep
        preserve_structure : bool
            Whether to preserve text structure (newlines, punctuation)

        Returns
        -------
        str
            Text with special characters removed
        """
        if self.preserve_special_chars:
            return text

        # Build allowed character set
        allowed = set()
        allowed.update(ALLOWED_CHARS['latin'])
        allowed.update(ALLOWED_CHARS['cyrillic'])
        allowed.update(ALLOWED_CHARS['digits'])

        if preserve_structure:
            allowed.update(ALLOWED_CHARS['punctuation'])
            allowed.add('\n')
        else:
            allowed.add(' ')  # Only basic space

        if keep_chars:
            allowed.update(keep_chars)

        # Filter characters
        filtered = ''.join(c for c in text if c in allowed)

        return filtered

    def normalize_for_llm(self, text: str) -> str:
        """
        Complete normalization pipeline for LLM input.

        Parameters
        ----------
        text : str
            Raw input text

        Returns
        -------
        str
            Fully normalized text ready for LLM
        """
        # Basic normalization
        text = self.normalize(text)

        if not text:
            return ""

        # Fix encoding
        text = self.fix_encoding(text)

        # Clean whitespace
        text = self.clean_whitespace(text)

        # Optional: Remove URLs and emails for privacy
        # text = self._compiled_patterns['url_pattern'].sub('[URL]', text)
        # text = self._compiled_patterns['email_pattern'].sub('[EMAIL]', text)

        return text


# ------------------------------------------------------------------------------
# Token Estimation and Management
# ------------------------------------------------------------------------------

class TokenEstimator:
    """
    Advanced token estimation with multiple methods.

    Supports simple character-based estimation, tiktoken for OpenAI models,
    and custom tokenizers.
    """

    def __init__(self, method: Union[str, TokenEstimationMethod] = TokenEstimationMethod.SIMPLE):
        """
        Initialize token estimator.

        Parameters
        ----------
        method : str or TokenEstimationMethod
            Estimation method to use
        """
        if isinstance(method, str):
            method = TokenEstimationMethod(method)

        self.method = method
        self._tiktoken_encoding = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer based on method."""
        if self.method == TokenEstimationMethod.TIKTOKEN:
            tiktoken = DependencyManager.get_module('tiktoken')
            if tiktoken:
                try:
                    self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    logger.warning(f"Failed to load tiktoken encoding: {e}")
                    self.method = TokenEstimationMethod.SIMPLE

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

        if self.method == TokenEstimationMethod.TIKTOKEN and self._tiktoken_encoding:
            try:
                return len(self._tiktoken_encoding.encode(text))
            except Exception:
                # Fallback to simple method
                pass

        # Simple word-based estimation
        words = text.split()

        # Check for Cyrillic content
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in text)

        # Adjust tokens per word based on language
        # Russian/Cyrillic tends to have more tokens per word
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


# ------------------------------------------------------------------------------
# Text Truncation
# ------------------------------------------------------------------------------

class TextTruncator:
    """
    Advanced text truncation with multiple strategies.

    Handles intelligent truncation while preserving text coherence
    and important information.
    """

    def __init__(self, estimator: Optional[TokenEstimator] = None):
        """
        Initialize text truncator.

        Parameters
        ----------
        estimator : TokenEstimator, optional
            Token estimator to use
        """
        self.estimator = estimator or TokenEstimator()
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def truncate(
            self,
            text: str,
            max_tokens: int,
            strategy: Union[str, TruncationStrategy] = TruncationStrategy.SMART,
            add_ellipsis: bool = True
    ) -> Tuple[str, bool, int]:
        """
        Truncate text to fit within token limit.

        Parameters
        ----------
        text : str
            Text to truncate
        max_tokens : int
            Maximum token count
        strategy : str or TruncationStrategy
            Truncation strategy
        add_ellipsis : bool
            Whether to add ellipsis

        Returns
        -------
        tuple
            (truncated_text, was_truncated, tokens_removed)
        """
        if isinstance(strategy, str):
            strategy = TruncationStrategy(strategy)

        current_tokens = self.estimator.estimate(text)

        # No truncation needed
        if current_tokens <= max_tokens:
            return text, False, 0

        # Calculate target character count
        char_ratio = len(text) / current_tokens if current_tokens > 0 else TOKEN_CHAR_RATIO
        target_chars = int(max_tokens * char_ratio * 0.9)  # 90% to be safe

        # Apply truncation strategy
        if strategy == TruncationStrategy.END:
            truncated = self._truncate_end(text, target_chars)
        elif strategy == TruncationStrategy.MIDDLE:
            truncated = self._truncate_middle(text, target_chars)
        else:  # SMART
            truncated = self._truncate_smart(text, target_chars)

        # Add ellipsis if requested
        if add_ellipsis and not truncated.endswith('...'):
            truncated += '...'

        # Calculate final metrics
        final_tokens = self.estimator.estimate(truncated)
        tokens_removed = current_tokens - final_tokens

        return truncated, True, tokens_removed

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

        # Align with word boundaries
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
        """Smart truncation preserving semantic units."""
        if len(text) <= max_chars:
            return text

        # Try paragraph-based truncation first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
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
        sentences = self._sentence_pattern.split(text)
        result = []
        current_length = 0

        for sent in sentences:
            sent_length = len(sent) + (1 if result else 0)  # Account for space
            if current_length + sent_length > max_chars:
                break
            result.append(sent)
            current_length += sent_length

        if result:
            # Add period if last sentence doesn't end with punctuation
            final_text = ' '.join(result)
            if final_text and final_text[-1] not in '.!?':
                final_text += '.'
            return final_text

        # Last resort: truncate at word boundary
        return self._truncate_end(text, max_chars)


# ------------------------------------------------------------------------------
# Marker Management
# ------------------------------------------------------------------------------

class MarkerManager:
    """
    Enhanced marker management for resumable processing.

    Handles processing markers with consistency checks and
    integration with canonicalization.
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
        self._marker_pattern = re.compile(f'^{re.escape(marker)}')

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
            True if text starts with marker
        """
        if not text:
            return False
        return self._marker_pattern.match(text) is not None

    def add_marker(self, text: str) -> str:
        """
        Add processing marker to text (idempotent).

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

        # Prevent double-marking
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
            return self._marker_pattern.sub('', text).strip()

        return text

    def extract_marked_and_clean(self, text: str) -> Tuple[bool, str]:
        """
        Extract marker status and clean text.

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        tuple
            (has_marker, clean_text)
        """
        has_mark = self.has_marker(text)
        clean = self.remove_marker(text) if has_mark else text
        return has_mark, clean


# ------------------------------------------------------------------------------
# Text Canonicalization
# ------------------------------------------------------------------------------

def canonicalize_text(text: str, processing_marker: str = DEFAULT_MARKER) -> str:
    """
    Canonicalize text for consistent cache key generation.

    This function ensures that text with different representations
    (with/without markers, different line endings, extra whitespace)
    will generate the same cache key.

    Parameters
    ----------
    text : str
        Input text to canonicalize
    processing_marker : str
        Processing marker to remove

    Returns
    -------
    str
        Canonicalized text
    """
    # Handle null-like values
    if text is None or (hasattr(text, 'isna') and text.isna()):
        return ""

    # Convert to string
    text = str(text)

    # Remove processing marker if present
    marker_manager = MarkerManager(processing_marker)
    _, text = marker_manager.extract_marked_and_clean(text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ------------------------------------------------------------------------------
# Main Preprocessing Class
# ------------------------------------------------------------------------------

class TextPreprocessor:
    """
    Main text preprocessing orchestrator.

    Coordinates all preprocessing operations and provides a unified
    interface for the LLM processing pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text preprocessor.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with preprocessing settings
        """
        self.config = config or {}

        # Initialize components
        self.normalizer = TextNormalizer(
            preserve_special_chars=self.config.get('preserve_special_chars', False)
        )

        # Get token estimation method from config
        token_method = self.config.get('token_estimation_method', 'simple')
        if isinstance(token_method, str):
            token_method = TokenEstimationMethod(token_method)
        self.estimator = TokenEstimator(token_method)

        self.truncator = TextTruncator(self.estimator)
        self.marker_manager = MarkerManager(
            self.config.get('processing_marker', DEFAULT_MARKER)
        )

        # Validation settings
        self.min_text_length = self.config.get('min_text_length', MIN_VALID_TEXT_LENGTH)
        self.max_text_length = self.config.get('max_text_length', MAX_TEXT_LENGTH)
        self.max_input_tokens = self.config.get('max_input_tokens', 1000)

        # Get truncation strategy from config
        truncation_strategy = self.config.get('truncation_strategy', 'smart')
        if isinstance(truncation_strategy, str):
            truncation_strategy = TruncationStrategy(truncation_strategy)
        self.truncation_strategy = truncation_strategy

        # Future: Entity detector hook
        self.entity_detector = None  # Will be EntityDetector() later

        logger.info(
            f"TextPreprocessor initialized with: "
            f"token_method={token_method.value}, "
            f"truncation={truncation_strategy.value}, "
            f"max_tokens={self.max_input_tokens}"
        )

    def preprocess(
            self,
            text: str,
            field_name: Optional[str] = None,
            skip_marker_check: bool = False
    ) -> PreprocessResult:
        """
        Main preprocessing pipeline.

        Parameters
        ----------
        text : str
            Text to preprocess
        field_name : str, optional
            Field name for context
        skip_marker_check : bool
            Whether to skip marker checking

        Returns
        -------
        PreprocessResult
            Preprocessing result with all metadata
        """
        import time
        start_time = time.time()

        # Store original for metrics
        original_text = text

        # 1. Basic validation
        validation_result = self._validate_input(text)
        if not validation_result['valid']:
            return create_failed_preprocess_result(
                text=text,
                error_message=validation_result['reason'],
                original_length=len(str(text)) if text else 0,
                processing_duration=time.time() - start_time
            )

        # 2. Check for existing marker
        if not skip_marker_check and self.config.get('use_processing_marker', True):
            has_marker, clean_text = self.marker_manager.extract_marked_and_clean(text)
            if has_marker:
                # Already processed - return as-is
                return create_successful_preprocess_result(
                    original_text=original_text,
                    processed_text=text,  # Keep marker
                    canonical_text=canonicalize_text(clean_text, self.marker_manager.marker),
                    estimated_tokens=self.estimator.estimate(clean_text),
                    was_truncated=False,
                    truncated_tokens=0,
                    processing_duration=time.time() - start_time
                )
        else:
            clean_text = text

        # 3. Normalize text
        normalized = self.normalizer.normalize_for_llm(clean_text)

        if not normalized:
            return create_failed_preprocess_result(
                text=text,
                error_message="Text normalized to empty string",
                original_length=len(original_text),
                processing_duration=time.time() - start_time
            )

        # 4. Canonicalize for cache
        canonical = canonicalize_text(normalized, self.marker_manager.marker)

        # 5. Check token count and truncate if needed
        current_tokens = self.estimator.estimate(normalized)

        if current_tokens > self.max_input_tokens:
            truncated_text, was_truncated, tokens_removed = self.truncator.truncate(
                normalized,
                self.max_input_tokens,
                self.truncation_strategy,
                add_ellipsis=True
            )
            final_tokens = self.estimator.estimate(truncated_text)
        else:
            truncated_text = normalized
            was_truncated = False
            tokens_removed = 0
            final_tokens = current_tokens

        # 6. Future: Entity detection
        entities = self._detect_entities_placeholder(truncated_text)

        # 7. Build successful result
        result = create_successful_preprocess_result(
            original_text=original_text,
            processed_text=truncated_text,
            canonical_text=canonical,
            estimated_tokens=final_tokens,
            was_truncated=was_truncated,
            truncated_tokens=tokens_removed,
            processing_duration=time.time() - start_time
        )

        # Add any detected issues
        if was_truncated:
            result.issues.append(f"Text truncated: removed {tokens_removed} tokens")

        if entities:
            result.issues.append(f"Detected {len(entities)} entities")

        return result

    def _validate_input(self, text: Any) -> Dict[str, Any]:
        """
        Validate input text.

        Parameters
        ----------
        text : Any
            Input to validate

        Returns
        -------
        dict
            Validation result with 'valid' and 'reason'
        """
        # Check for None/NaN
        if text is None or pd.isna(text):
            return {'valid': False, 'reason': 'Text is None or NaN'}

        # Convert to string
        text_str = str(text).strip()

        # Check for empty
        if not text_str:
            return {'valid': False, 'reason': 'Text is empty'}

        # Check for null representations
        if text_str.lower() in ('nan', 'none', 'null', '<na>', 'n/a', 'na'):
            return {'valid': False, 'reason': f'Text is null representation: {text_str}'}

        # Check minimum length
        if len(text_str) < self.min_text_length:
            return {
                'valid': False,
                'reason': f'Text too short: {len(text_str)} < {self.min_text_length}'
            }

        # Check maximum length
        if len(text_str) > self.max_text_length:
            return {
                'valid': False,
                'reason': f'Text too long: {len(text_str)} > {self.max_text_length}'
            }

        # Check for meaningful content (not just punctuation/spaces)
        clean_text = re.sub(r'[^\w\s]', '', text_str)
        if len(clean_text.strip()) < 2:
            return {'valid': False, 'reason': 'Text contains no meaningful content'}

        return {'valid': True, 'reason': 'Valid text'}

    def _detect_entities_placeholder(self, text: str) -> List[str]:
        """
        Placeholder for future NER integration.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        list
            Detected entities (empty for now)
        """
        # TODO: Integrate EntityDetector when implemented
        # if self.entity_detector:
        #     return self.entity_detector.detect_entities(text)
        return []

    def batch_preprocess(
            self,
            texts: List[str],
            field_name: Optional[str] = None
    ) -> List[PreprocessResult]:
        """
        Preprocess multiple texts.

        Parameters
        ----------
        texts : list
            List of texts to preprocess
        field_name : str, optional
            Field name for context

        Returns
        -------
        list
            List of preprocessing results
        """
        results = []

        for text in texts:
            result = self.preprocess(text, field_name)
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics.

        Returns
        -------
        dict
            Statistics dictionary
        """
        return {
            'config': {
                'min_text_length': self.min_text_length,
                'max_text_length': self.max_text_length,
                'max_input_tokens': self.max_input_tokens,
                'token_method': self.estimator.method.value,
                'truncation_strategy': self.truncation_strategy.value,
                'processing_marker': self.marker_manager.marker
            },
            'components': {
                'normalizer': 'TextNormalizer',
                'estimator': 'TokenEstimator',
                'truncator': 'TextTruncator',
                'marker_manager': 'MarkerManager',
                'entity_detector': 'Not implemented'
            }
        }


# ------------------------------------------------------------------------------
# Convenience Functions (for backward compatibility)
# ------------------------------------------------------------------------------

def preprocess_text(
        text: str,
        max_tokens: int = 1000,
        truncation_strategy: str = 'smart',
        processing_marker: str = DEFAULT_MARKER
) -> PreprocessResult:
    """
    Simple preprocessing function for quick use.

    Parameters
    ----------
    text : str
        Text to preprocess
    max_tokens : int
        Maximum tokens allowed
    truncation_strategy : str
        How to truncate ('end', 'middle', 'smart')
    processing_marker : str
        Marker for processed text

    Returns
    -------
    PreprocessResult
        Preprocessing result
    """
    config = {
        'max_input_tokens': max_tokens,
        'truncation_strategy': truncation_strategy,
        'processing_marker': processing_marker,
        'use_processing_marker': True
    }

    preprocessor = TextPreprocessor(config)
    return preprocessor.preprocess(text)


def normalize_text(text: Any) -> str:
    """Normalize any input to clean string."""
    return TextNormalizer.normalize(text)


def estimate_tokens(text: str, method: str = "simple") -> int:
    """Estimate token count for text."""
    estimator = TokenEstimator(method)
    return estimator.estimate(text)


def truncate_text(
        text: str,
        max_tokens: int,
        strategy: str = "smart"
) -> Tuple[str, bool, int]:
    """Truncate text to fit token limit."""
    truncator = TextTruncator()
    return truncator.truncate(text, max_tokens, strategy)


# ------------------------------------------------------------------------------
# Future Extension Hooks
# ------------------------------------------------------------------------------

class EntityDetector:
    """
    Placeholder for future NER functionality.

    Will integrate with spaCy, Natasha, or custom entity detection.
    """

    def __init__(self, method: str = "regex"):
        """
        Initialize entity detector.

        Parameters
        ----------
        method : str
            Detection method ('regex', 'spacy', 'natasha')
        """
        self.method = method

    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect named entities in text.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        list
            List of detected entities
        """
        # TODO: Implement actual detection
        return []