"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Configuration
Package:       pamola_core.utils.nlp.llm.config
Version:       2.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides configuration dataclasses and utilities for the LLM
subsystem. It defines configurations for LLM connections, processing
parameters, generation settings, caching, and monitoring. Also includes
model presets and provider-specific configurations.

Key Features:
- Comprehensive configuration dataclasses
- Model-specific presets for optimal parameters
- Provider-specific API parameter mapping
- Configuration validation and defaults
- Model aliases for convenient access
- Support for multiple LLM providers
- Automatic alias resolution in configurations
- Model availability checking with per-model caching
- Debug utilities for configuration troubleshooting
- Path to string conversion for JSON compatibility

Framework:
Part of PAMOLA.CORE LLM subsystem, providing centralized configuration
management for all LLM operations.

Changelog:
2.2.0 - Fixed None values overwriting presets in merge_with_model_defaults
     - Fixed global cache TTL issue with per-key timestamps
     - Added Path to string conversion for JSON serialization
     - Added generation parameter validation
     - Reorganized alias definitions for better readability
     - Added provider-specific configuration validation
2.1.0 - Fixed LM Studio parameter handling (temperature in 'args' field)
     - Reorganized aliases to eliminate duplicates
     - Added MODEL_ALIAS_SYNONYMS for legacy support
     - Fixed merge_with_model_defaults to respect user overrides
     - Added strict validation mode for development
     - Added debug utilities for configuration inspection
     - Improved OpenAI parameter filtering
     - Added environment variable support for defaults
2.0.1 - Fixed Python 3.8 compatibility
     - Added automatic alias resolution in LLMConfig
     - Enhanced model availability checking with caching
     - Improved validation with detailed reasons
     - Added duplicate alias detection
2.0.0 - Added model aliases support
     - Updated model presets based on testing
     - Added resolve_model_name function
     - Enhanced documentation
1.0.0 - Initial implementation
     - Basic configuration dataclasses
     - Model presets for common models

Dependencies:
- dataclasses - Configuration structures
- enum - Configuration enumerations
- typing - Type annotations
- time - For cache TTL
- logging - Debug logging
- os - Environment variable support
- warnings - Configuration warnings
- pathlib - Path handling

TODO:
- Add configuration profiles for different use cases
- Implement configuration inheritance
- Add automatic model detection from LM Studio
- Support for configuration overlays
- Add configuration migration utilities
- Implement provider-specific validation rules
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, cast

# Configure logger
logger = logging.getLogger(__name__)

# Model availability cache with per-key timestamps
_model_availability_cache: Dict[str, Tuple[bool, float]] = {}


# ------------------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------------------

class Provider(str, Enum):
    """Supported LLM providers."""
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class CacheType(str, Enum):
    """Cache backend types."""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    NONE = "none"


class TokenEstimationMethod(str, Enum):
    """Token estimation methods."""
    SIMPLE = "simple"  # Character-based estimation
    TIKTOKEN = "tiktoken"  # OpenAI tiktoken library
    CUSTOM = "custom"  # Custom tokenizer


class TruncationStrategy(str, Enum):
    """Text truncation strategies."""
    END = "end"  # Keep beginning, truncate end
    MIDDLE = "middle"  # Keep beginning and end, truncate middle
    SMART = "smart"  # Try to preserve sentence boundaries


class ValidationResult(str, Enum):
    """Model validation result types."""
    VALID_ALIAS = "valid_alias"
    VALID_PRESET = "valid_preset"
    VALID_NAME = "valid_name"
    NOT_AVAILABLE = "not_available"
    INVALID = "invalid"


# ------------------------------------------------------------------------------
# Model Presets
# ------------------------------------------------------------------------------

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # Quality model - conservative settings for consistency
    "gemma-2-9b-it-russian-function-calling": {
        "temperature": 0.3,  # Low for consistent output
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 512,
        "repeat_penalty": 1.1,
        "stop_sequences": ["</text>", "\n\n", "<end_of_turn>"],
    },

    # Balanced model - moderate settings
    "google/gemma-3-4b": {
        "temperature": 0.4,  # Slightly higher for variety
        "top_p": 0.90,
        "top_k": 40,
        "max_tokens": 256,  # Limited for faster processing
        "repeat_penalty": 1.05,
        "stop_sequences": ["</text>", "\n\n", "<end_of_turn>"],
    },

    # Alternative uncensored model
    "gemma-3-it-4b-uncensored-db1-x": {
        "temperature": 0.5,
        "top_p": 0.90,
        "top_k": 40,
        "max_tokens": 256,
        "repeat_penalty": 1.05,
        "stop_sequences": ["</text>", "\n\n"],
    },

    # Ultra-fast model - minimal settings
    "gemma-3-1b-it-qat": {
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": 30,
        "max_tokens": 128,  # Very limited for speed
        "repeat_penalty": 1.0,
        "stop_sequences": ["\n\n"],
    },

    # Llama models
    "llama-3.1-8b-lexi-uncensored": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 512,
        "repeat_penalty": 1.1,
        "stop_sequences": ["</text>", "\n\n", "Human:", "Assistant:"],
    },

    # Specialized anonymization model
    "deid-anonymization-llama3": {
        "temperature": 0.2,  # Very low for deterministic anonymization
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 512,
        "repeat_penalty": 1.0,
        "stop_sequences": ["</text>", "\n\n"],
    },

    # Fast Phi model - optimized for speed
    "phi-3-mini-128k-it-russian-q4-k-m": {
        "temperature": 0.2,  # Low for consistency
        "top_p": 0.85,
        "top_k": 30,
        "max_tokens": 200,  # Limited for speed
        "repeat_penalty": 1.0,
        "stop_sequences": ["\n\n", "###"],
    },

    # DeepSeek models
    "deepseek/deepseek-r1-0528-qwen3-8b": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 512,
        "repeat_penalty": 1.1,
        "stop_sequences": ["</think>", "\n\n", "<|im_end|>"],
    },
}

# ------------------------------------------------------------------------------
# Model aliases
# ------------------------------------------------------------------------------

# Primary descriptive aliases
MODEL_ALIASES = {
    "QUALITY": "gemma-2-9b-it-russian-function-calling",
    "BALANCED": "google/gemma-3-4b",
    "FAST": "phi-3-mini-128k-it-russian-q4-k-m",
    "ULTRA_FAST": "gemma-3-1b-it-qat",
    "ALT": "llama-3.1-8b-lexi-uncensored",
    "ANON": "deid-anonymization-llama3",
}

# Legacy aliases mapped to primary aliases for backward compatibility
MODEL_ALIAS_SYNONYMS = {
    "LLM1": "QUALITY",
    "LLM2": "BALANCED",
    "LLM3": "FAST",
}


# ------------------------------------------------------------------------------
# Configuration Dataclasses
# ------------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    LLM connection and API configuration.

    Parameters
    ----------
    provider : str or Provider
        LLM provider (lmstudio, openai, etc.)
    api_url : str
        API endpoint URL
    model_name : str
        Model identifier or alias
    api_key : str, optional
        API key for authentication
    ttl : int
        Connection time-to-live in seconds
    timeout : int
        Request timeout in seconds
    max_retries : int
        Maximum retry attempts
    retry_delay : float
        Delay between retries in seconds
    max_workers : int
        Maximum concurrent workers
    thread_safe_model : bool
        Whether model supports concurrent requests
    """
    provider: Union[str, Provider] = Provider.LMSTUDIO
    api_url: str = "http://localhost:1234/v1"
    model_name: str = "gemma-2-9b-it-russian-function-calling"
    api_key: Optional[str] = None
    ttl: int = 3600
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_workers: int = 1
    thread_safe_model: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Resolve model alias FIRST
        self.model_name = resolve_model_name(self.model_name)

        # Ensure provider is enum
        if isinstance(self.provider, str):
            self.provider = Provider(self.provider.lower())

        # Set default API URLs based on provider
        if self.api_url == "http://localhost:1234/v1" and self.provider != Provider.LMSTUDIO:
            if self.provider == Provider.OPENAI:
                self.api_url = "https://api.openai.com/v1"
            elif self.provider == Provider.ANTHROPIC:
                self.api_url = "https://api.anthropic.com/v1"


@dataclass
class ProcessingConfig:
    """
    Text processing configuration.

    Parameters
    ----------
    batch_size : int
        Number of texts to process in batch
    max_input_tokens : int
        Maximum tokens per input
    token_estimation_method : TokenEstimationMethod
        Method for estimating tokens
    truncation_strategy : TruncationStrategy
        How to truncate long texts
    use_processing_marker : bool
        Whether to mark processed texts
    processing_marker : str
        Marker character for processed texts
    skip_processed : bool
        Skip already processed texts
    max_records : int, optional
        Maximum records to process
    adaptive_batch_size : bool
        Adjust batch size based on performance
    min_batch_size : int
        Minimum batch size
    max_batch_size : int
        Maximum batch size
    memory_cleanup_interval : int
        Records between memory cleanup
    """
    batch_size: int = 1
    max_input_tokens: int = 1000
    token_estimation_method: Union[str, TokenEstimationMethod] = TokenEstimationMethod.SIMPLE
    truncation_strategy: Union[str, TruncationStrategy] = TruncationStrategy.SMART
    use_processing_marker: bool = True
    processing_marker: str = "~"
    skip_processed: bool = True
    max_records: Optional[int] = None
    adaptive_batch_size: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 32
    memory_cleanup_interval: int = 1000

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure enums
        if isinstance(self.token_estimation_method, str):
            self.token_estimation_method = TokenEstimationMethod(self.token_estimation_method)
        if isinstance(self.truncation_strategy, str):
            self.truncation_strategy = TruncationStrategy(self.truncation_strategy)

        # Validate batch sizes
        if self.min_batch_size > self.max_batch_size:
            self.min_batch_size, self.max_batch_size = self.max_batch_size, self.min_batch_size

        self.batch_size = max(self.min_batch_size, min(self.batch_size, self.max_batch_size))


@dataclass
class GenerationConfig:
    """
    Text generation parameters.

    Parameters
    ----------
    temperature : float
        Sampling temperature (0.0-2.0)
    top_p : float
        Nucleus sampling threshold
    top_k : int
        Top-k sampling
    max_tokens : int
        Maximum generation length
    stop_sequences : List[str]
        Stop generation sequences
    repeat_penalty : float, optional
        Repetition penalty
    presence_penalty : float, optional
        Presence penalty (OpenAI)
    frequency_penalty : float, optional
        Frequency penalty (OpenAI)
    stream : bool
        Stream responses
    seed : int, optional
        Random seed for reproducibility
    """
    temperature: float = field(
        default_factory=lambda: float(os.getenv('PAMOLA_DEFAULT_TEMPERATURE', '0.7'))
    )
    top_p: float = field(
        default_factory=lambda: float(os.getenv('PAMOLA_DEFAULT_TOP_P', '0.95'))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv('PAMOLA_DEFAULT_TOP_K', '40'))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv('PAMOLA_DEFAULT_MAX_TOKENS', '512'))
    )
    stop_sequences: List[str] = field(default_factory=lambda: ["</text>", "\n\n", "<end_of_turn>"])
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stream: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate generation parameters."""
        # Validate temperature range
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        # Validate top_p range
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

        # Validate top_k
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

        # Validate max_tokens
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    def to_api_params(self, provider: Union[str, Provider]) -> Dict[str, Any]:
        """
        Convert to provider-specific API parameters.

        Parameters
        ----------
        provider : str or Provider
            Target provider

        Returns
        -------
        Dict[str, Any]
            API-specific parameters
        """
        if isinstance(provider, str):
            provider = Provider(provider.lower())

        # LM Studio specific format
        if provider == Provider.LMSTUDIO:
            # LM Studio REST API expects parameters under "args" key
            # Explicit type annotation to prevent type narrowing
            params: Dict[str, Any] = {
                "args": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "max_tokens": self.max_tokens,
                },
                "stream": self.stream,
            }

            # Add optional parameters to args
            if self.repeat_penalty is not None and self.repeat_penalty > 0:
                params["args"]["repeat_penalty"] = self.repeat_penalty
            if self.seed is not None:
                params["args"]["seed"] = self.seed

            # Stop sequences go at top level
            if self.stop_sequences:
                params["stop"] = self.stop_sequences

            return params

        elif provider == Provider.OPENAI:
            # Build params with only non-None values
            # Explicit type annotation to allow mixed types
            params: Dict[str, Any] = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'top_p': self.top_p,
                'stream': self.stream,
            }

            # Only include optional parameters if set and not default
            if self.presence_penalty is not None and self.presence_penalty != 0.0:
                params['presence_penalty'] = self.presence_penalty
            if self.frequency_penalty is not None and self.frequency_penalty != 0.0:
                params['frequency_penalty'] = self.frequency_penalty
            if self.seed is not None:
                params['seed'] = self.seed
            if self.stop_sequences:
                params['stop'] = self.stop_sequences

            return params

        elif provider == Provider.ANTHROPIC:
            # Explicit type annotation for mixed value types
            params: Dict[str, Any] = {
                'top_p': self.top_p,
                'top_k': self.top_k,
                'stream': self.stream,
                'max_tokens_to_sample': self.max_tokens,
                'temperature': self.temperature,
            }

            if self.stop_sequences:
                params['stop_sequences'] = self.stop_sequences

            return params

        else:
            # Generic format for other providers
            # Explicit annotation to prevent type inference issues
            params: Dict[str, Any] = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'stop': self.stop_sequences,
                'stream': self.stream,
            }

            # Add all available parameters
            if self.top_p is not None:
                params['top_p'] = self.top_p
            if self.top_k is not None:
                params['top_k'] = self.top_k
            if self.repeat_penalty is not None and self.repeat_penalty > 0:
                params['repeat_penalty'] = self.repeat_penalty
            if self.seed is not None:
                params['seed'] = self.seed

            # Remove None values
            return {k: v for k, v in params.items() if v is not None}

    def merge_with_model_defaults(self, model_name: str) -> 'GenerationConfig':
        """
        Merge with model-specific defaults.

        User values take precedence over presets, but None values don't override.

        Parameters
        ----------
        model_name : str
            Model identifier

        Returns
        -------
        GenerationConfig
            Merged configuration with user values taking precedence
        """
        # Resolve any aliases first
        resolved_model = resolve_model_name(model_name)

        if resolved_model in MODEL_PRESETS:
            preset = MODEL_PRESETS[resolved_model]

            # Start with preset values as base
            merged_params = preset.copy()

            # Override with user values ONLY if not None
            current_dict = asdict(self)

            for key, value in current_dict.items():
                # Only override if user explicitly set a value (not None)
                if value is not None:
                    merged_params[key] = value

            return GenerationConfig(**merged_params)

        return self


@dataclass
class CacheConfig:
    """
    Cache configuration.

    Parameters
    ----------
    enabled : bool
        Whether caching is enabled
    cache_type : CacheType
        Cache backend type
    ttl : int
        Cache time-to-live in seconds
    max_size : int, optional
        Maximum cache size (entries or MB)
    eviction_policy : str
        Cache eviction policy
    """
    enabled: bool = True
    cache_type: Union[str, CacheType] = CacheType.MEMORY
    ttl: int = 86400  # 24 hours
    max_size: Optional[int] = None
    eviction_policy: str = "lru"

    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.cache_type, str):
            # Support "off" as alias for disabled
            if self.cache_type.lower() == "off":
                self.enabled = False
                self.cache_type = CacheType.NONE
            else:
                self.cache_type = CacheType(self.cache_type)


@dataclass
class MonitoringConfig:
    """
    Monitoring and debugging configuration.

    Parameters
    ----------
    debug_mode : bool
        Enable debug logging
    debug_log_file : Path, optional
        Debug log file path
    log_requests : bool
        Log all LLM requests
    log_responses : bool
        Log all LLM responses
    monitor_performance : bool
        Track performance metrics
    profile_memory : bool
        Profile memory usage
    slow_request_threshold : float
        Threshold for slow request warnings (seconds)
    """
    debug_mode: bool = False
    debug_log_file: Optional[Path] = None
    log_requests: bool = False
    log_responses: bool = False
    monitor_performance: bool = True
    profile_memory: bool = False
    slow_request_threshold: float = 10.0

    def __post_init__(self):
        """Validate and normalize configuration."""
        if self.debug_log_file and not isinstance(self.debug_log_file, Path):
            self.debug_log_file = Path(cast(str, self.debug_log_file))


# ------------------------------------------------------------------------------
# Model Management Functions
# ------------------------------------------------------------------------------

def resolve_model_name(model_input: str) -> str:
    """
    Resolve model alias to actual model name.

    First checks synonyms (e.g., LLM1 -> QUALITY), then primary aliases,
    then returns input unchanged if not found.

    Parameters
    ----------
    model_input : str
        Model alias or actual model name

    Returns
    -------
    str
        Actual model name

    Examples
    --------
    >>> resolve_model_name("LLM1")
    'gemma-2-9b-it-russian-function-calling'
    >>> resolve_model_name("QUALITY")
    'gemma-2-9b-it-russian-function-calling'
    >>> resolve_model_name("some-actual-model-name")
    'some-actual-model-name'
    """
    # Normalize to uppercase
    model_upper = model_input.upper()

    # First check synonyms
    if model_upper in MODEL_ALIAS_SYNONYMS:
        model_upper = MODEL_ALIAS_SYNONYMS[model_upper]

    # Then check primary aliases
    return MODEL_ALIASES.get(model_upper, model_input)


def get_available_aliases() -> Dict[str, str]:
    """
    Get all available model aliases (primary only).

    Returns
    -------
    Dict[str, str]
        Dictionary mapping primary aliases to model names
    """
    return MODEL_ALIASES.copy()


def get_all_aliases() -> Dict[str, str]:
    """
    Get all aliases including synonyms.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping all aliases (primary and synonyms) to model names
    """
    all_aliases = MODEL_ALIASES.copy()

    # Add resolved synonyms
    for synonym, primary in MODEL_ALIAS_SYNONYMS.items():
        if primary in MODEL_ALIASES:
            all_aliases[synonym] = MODEL_ALIASES[primary]

    return all_aliases


def validate_model_name(
        model_name: str,
        available_models: Optional[List[str]] = None
) -> Tuple[bool, str, ValidationResult]:
    """
    Validate if model name or alias is valid.

    Parameters
    ----------
    model_name : str
        Model name or alias to validate
    available_models : List[str], optional
        List of available models from LM Studio

    Returns
    -------
    Tuple[bool, str, ValidationResult]
        (is_valid, resolved_name, reason)

    Examples
    --------
    >>> validate_model_name("LLM1")
    (True, 'gemma-2-9b-it-russian-function-calling', <ValidationResult.VALID_ALIAS>)
    >>> validate_model_name("unknown-model")
    (False, 'unknown-model', <ValidationResult.INVALID>)
    """
    resolved = resolve_model_name(model_name)
    model_upper = model_name.upper()

    # Check if it's a synonym
    if model_upper in MODEL_ALIAS_SYNONYMS:
        # It's a valid synonym that resolved to actual model
        if available_models and resolved not in available_models:
            return False, resolved, ValidationResult.NOT_AVAILABLE
        return True, resolved, ValidationResult.VALID_ALIAS

    # Check if it's a primary alias
    if model_upper in MODEL_ALIASES:
        if available_models and resolved not in available_models:
            return False, resolved, ValidationResult.NOT_AVAILABLE
        return True, resolved, ValidationResult.VALID_ALIAS

    # Check if in presets
    if resolved in MODEL_PRESETS:
        if available_models and resolved not in available_models:
            return False, resolved, ValidationResult.NOT_AVAILABLE
        return True, resolved, ValidationResult.VALID_PRESET

    # Check if available (if list provided)
    if available_models:
        if resolved in available_models:
            return True, resolved, ValidationResult.VALID_NAME
        return False, resolved, ValidationResult.NOT_AVAILABLE

    # Heuristic check
    looks_like_model = len(resolved) > 5 and ('-' in resolved or '_' in resolved or '/' in resolved)
    if looks_like_model:
        return True, resolved, ValidationResult.VALID_NAME

    return False, resolved, ValidationResult.INVALID


def check_model_availability(
        model_name: str,
        lm_studio_url: str = "http://localhost:1234/v1"
) -> bool:
    """
    Check if model is available in LM Studio with per-model caching.

    Parameters
    ----------
    model_name : str
        Resolved model name
    lm_studio_url : str
        LM Studio API URL

    Returns
    -------
    bool
        True if model is available or check failed

    Notes
    -----
    Results are cached for 60 seconds per model to avoid excessive API calls.
    Returns True on connection errors to avoid blocking execution.
    """
    global _model_availability_cache

    cache_key = f"{lm_studio_url}:{model_name}"
    current_time = time.time()

    # Check cache with per-key TTL
    if cache_key in _model_availability_cache:
        is_available, cache_time = _model_availability_cache[cache_key]
        if current_time - cache_time < 60:  # 60 seconds TTL
            return is_available

    try:
        import requests
        response = requests.get(f"{lm_studio_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('data', [])
            # Check both 'id' and 'name' fields
            for model in models:
                if (model.get('id') == model_name or
                        model.get('name') == model_name):
                    _model_availability_cache[cache_key] = (True, current_time)
                    return True

            _model_availability_cache[cache_key] = (False, current_time)
            return False
    except Exception as e:
        logger.debug(f"Could not check model availability: {e}")

    return True  # Assume available if can't check


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get model information including presets and aliases.

    Parameters
    ----------
    model_name : str
        Model name or alias

    Returns
    -------
    Dict[str, Any]
        Model information including resolved name, presets, and aliases
    """
    resolved = resolve_model_name(model_name)

    info = {
        'input': model_name,
        'resolved': resolved,
        'is_alias': model_name.upper() in get_all_aliases(),
        'is_primary_alias': model_name.upper() in MODEL_ALIASES,
        'is_synonym': model_name.upper() in MODEL_ALIAS_SYNONYMS,
        'has_preset': resolved in MODEL_PRESETS,
        'preset': MODEL_PRESETS.get(resolved, {}),
        'aliases': [alias for alias, name in MODEL_ALIASES.items() if name == resolved],
        'synonyms': [syn for syn, primary in MODEL_ALIAS_SYNONYMS.items()
                     if MODEL_ALIASES.get(primary) == resolved],
    }

    return info


def get_model_debug_info(model_name: str, provider: Provider = Provider.LMSTUDIO) -> Dict[str, Any]:
    """
    Get complete debug information for a model.

    Parameters
    ----------
    model_name : str
        Model name or alias
    provider : Provider
        Target provider for API params

    Returns
    -------
    Dict[str, Any]
        Complete debug information including generation configs
    """
    resolved = resolve_model_name(model_name)

    info = get_model_info(model_name)

    # Add generation parameters info
    if resolved in MODEL_PRESETS:
        preset = MODEL_PRESETS[resolved]
        default_gen = GenerationConfig()
        merged_gen = default_gen.merge_with_model_defaults(resolved)

        info['generation'] = {
            'defaults': asdict(default_gen),
            'preset': preset,
            'merged': asdict(merged_gen),
            'api_params': {
                'lmstudio': merged_gen.to_api_params(Provider.LMSTUDIO),
                'openai': merged_gen.to_api_params(Provider.OPENAI),
                'anthropic': merged_gen.to_api_params(Provider.ANTHROPIC),
            }
        }

    return info


def validate_generation_config(config: GenerationConfig, provider: Provider) -> List[str]:
    """
    Validate generation config for specific provider.

    Parameters
    ----------
    config : GenerationConfig
        Configuration to validate
    provider : Provider
        Target provider

    Returns
    -------
    List[str]
        List of warnings/issues
    """
    issues = []

    if provider == Provider.LMSTUDIO:
        # LM Studio specific validations
        if config.repeat_penalty and config.repeat_penalty < 0.1:
            issues.append("repeat_penalty < 0.1 may cause excessive repetition")
        if config.temperature > 1.5:
            issues.append("temperature > 1.5 may produce incoherent output")

    elif provider == Provider.OPENAI:
        # OpenAI specific validations
        if config.top_k is not None:
            issues.append("OpenAI does not support top_k parameter")
        if config.repeat_penalty is not None:
            issues.append("OpenAI uses frequency_penalty instead of repeat_penalty")

    elif provider == Provider.ANTHROPIC:
        # Anthropic specific validations
        if config.presence_penalty is not None or config.frequency_penalty is not None:
            issues.append("Anthropic does not support presence/frequency penalties")

    # General validations
    if config.max_tokens > 4096:
        issues.append(f"max_tokens={config.max_tokens} may exceed model's context window")

    if config.temperature == 0.0 and config.top_p < 1.0:
        issues.append("temperature=0 with top_p<1.0 may produce unexpected results")

    return issues


# ------------------------------------------------------------------------------
# Configuration Utilities
# ------------------------------------------------------------------------------

def _convert_paths_to_strings(obj: Any) -> Any:
    """
    Recursively convert Path objects to strings for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    Any
        Object with Path instances converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_strings(item) for item in obj)
    return obj


def create_default_config(
        provider: Union[str, Provider] = Provider.LMSTUDIO,
        model_name: str = "QUALITY"
) -> Dict[str, Any]:
    """
    Create default configuration for a provider and model.

    Parameters
    ----------
    provider : str or Provider
        LLM provider
    model_name : str
        Model name or alias

    Returns
    -------
    Dict[str, Any]
        Complete configuration dictionary with Path objects converted to strings
    """
    resolved_model = resolve_model_name(model_name)

    llm_config = LLMConfig(provider=provider, model_name=resolved_model)
    processing_config = ProcessingConfig()
    generation_config = GenerationConfig().merge_with_model_defaults(resolved_model)
    cache_config = CacheConfig()
    monitoring_config = MonitoringConfig()

    config_dict = {
        'llm': asdict(llm_config),
        'processing': asdict(processing_config),
        'generation': asdict(generation_config),
        'cache': asdict(cache_config),
        'monitoring': asdict(monitoring_config),
    }

    # Convert any Path objects to strings for JSON serialization
    return _convert_paths_to_strings(config_dict)


# ------------------------------------------------------------------------------
# Validation on module import
# ------------------------------------------------------------------------------

def _validate_aliases(strict: bool = False):
    """
    Validate aliases don't have conflicts.

    Parameters
    ----------
    strict : bool
        If True, raise exception on duplicates (for development)
        Can also be controlled by PAMOLA_STRICT_VALIDATION env var
    """
    # Check environment variable
    if os.getenv('PAMOLA_STRICT_VALIDATION', '').lower() == 'true':
        strict = True

    # Check primary aliases for duplicate model mappings
    seen_models = {}
    duplicates = []

    for alias, model in MODEL_ALIASES.items():
        if model in seen_models:
            duplicates.append(
                f"Model '{model}' has duplicate primary aliases: "
                f"{seen_models[model]} and {alias}"
            )
        seen_models[model] = alias

    # Check that synonyms point to valid primary aliases
    for synonym, primary in MODEL_ALIAS_SYNONYMS.items():
        if primary not in MODEL_ALIASES:
            duplicates.append(
                f"Synonym '{synonym}' points to non-existent primary alias '{primary}'"
            )

    # Check for synonym/primary conflicts
    for synonym in MODEL_ALIAS_SYNONYMS:
        if synonym in MODEL_ALIASES:
            duplicates.append(
                f"'{synonym}' exists as both primary alias and synonym"
            )

    if duplicates:
        message = "Alias configuration errors:\n" + "\n".join(duplicates)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)


# Run validation on import
_validate_aliases()