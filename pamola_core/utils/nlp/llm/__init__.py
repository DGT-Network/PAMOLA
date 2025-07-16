"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Subsystem Package
Package:       pamola_core.utils.nlp.llm
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This package provides a comprehensive subsystem for Large Language Model
operations within PAMOLA.CORE. It includes client interfaces, configuration
management, prompt engineering, metrics collection, and text processing
utilities specifically designed for LLM interactions.

The package is organized into the following modules:
- client: LLM client implementations (LM Studio, OpenAI, etc.)
- config: Configuration dataclasses and validation
- metrics: Performance and quality metrics collection
- processing: Text processing utilities (normalization, truncation, etc.)
- prompt: Prompt template management and formatting

Key Features:
- Unified interface for multiple LLM providers
- Comprehensive configuration management
- Advanced prompt engineering capabilities
- Real-time metrics collection and analysis
- Robust error handling and retry logic
- Memory-efficient batch processing
- Extensible architecture for new providers

Framework:
Part of PAMOLA.CORE NLP utilities, providing modular LLM capabilities
that can be used independently or integrated with higher-level operations.

Usage:
    from pamola_core.utils.nlp.llm import (
        create_llm_client,
        LLMConfig,
        GenerationConfig,
        PromptTemplate,
        MetricsCollector
    )

    # Create client with configuration
    config = LLMConfig(
        provider='your_provider',
        api_url='your_api_url',
        model_name='your_model'
    )
    client = create_llm_client(**config.to_dict())

    # Generate text with custom parameters
    gen_config = GenerationConfig(
        temperature=0.7,
        max_tokens=512
    )
    response = client.generate("Your prompt", gen_config)

Dependencies:
- Standard library for core functionality
- Optional: lmstudio for LM Studio support
- Optional: tiktoken for token counting
- Optional: numpy for advanced metrics
"""

__version__ = "1.0.0"

# Define what should be available when using "from pamola_core.utils.nlp.llm import *"
__all__ = [
    # Client interfaces
    'BaseLLMClient',
    'LMStudioClient',
    'create_llm_client',
    'LLMResponse',
    'LLMGenerationParams',

    # Configuration classes
    'LLMConfig',
    'GenerationConfig',
    'ProcessingConfig',
    'CacheConfig',
    'MonitoringConfig',
    'ProcessingMode',
    'TokenEstimationMethod',
    'TruncationStrategy',
    'CacheType',
    'get_model_preset',
    'create_config_from_dict',

    # Prompt management
    'PromptTemplate',
    'PromptLibrary',
    'PromptFormatter',
    'PromptChainBuilder',
    'PromptStrategy',
    'PromptValidationError',
    'PromptConfig',
    'create_prompt_formatter',
    'load_prompt_library',

    # Metrics and results
    'ProcessingResult',
    'BatchResult',
    'MetricsCollector',
    'MetricsAggregator',
    'ResultStatus',
    'MetricType',
    'LatencyMetrics',
    'ThroughputMetrics',
    'TokenMetrics',
    'CacheMetrics',
    'QualityMetrics',
    'ResourceMetrics',
    'AggregatedMetrics',
    'create_metrics_summary',
    'calculate_percentiles',
    'format_latency_ms',
    'format_throughput',

    # Processing utilities
    'TextNormalizer',
    'TokenEstimator',
    'TextTruncator',
    'ResponseProcessor',
    'MarkerManager',
    'BatchProcessor',
    'TextChunker',
    'ProcessedText',
    'ResponseAnalysis',
    'ResponseType',

    # Convenience functions
    'normalize_text',
    'clean_whitespace',
    'estimate_tokens',
    'truncate_text',
    'extract_response_text',
    'analyze_response',
    'has_processing_marker',
    'add_processing_marker',
    'remove_processing_marker',
    'chunk_long_text',

    # Error classes (imported from base)
    'LLMError',
    'LLMConnectionError',
    'LLMGenerationError',
    'LLMResponseError',
]

# Lazy imports to prevent circular dependencies and improve startup time
_module_cache = {}


def _lazy_import(module_name, items):
    """
    Lazy import helper to avoid loading all modules at package import time.

    Parameters
    ----------
    module_name : str
        Name of the module to import from
    items : list
        List of items to import from the module

    Returns
    -------
    dict
        Dictionary of imported items
    """
    if module_name not in _module_cache:
        if module_name == 'client':
            from . import client
            _module_cache[module_name] = client
        elif module_name == 'config':
            from . import config
            _module_cache[module_name] = config
        elif module_name == 'prompt':
            from . import prompt
            _module_cache[module_name] = prompt
        elif module_name == 'metrics':
            from . import metrics
            _module_cache[module_name] = metrics
        elif module_name == 'processing':
            from . import processing_utils as processing
            _module_cache[module_name] = processing
        elif module_name == 'base':
            from .. import base
            _module_cache[module_name] = base

    module = _module_cache[module_name]
    return {item: getattr(module, item) for item in items if hasattr(module, item)}


def __getattr__(name):
    """
    Lazy loading of module attributes.

    This function is called when an attribute is accessed but not found
    in the module's namespace. It performs lazy imports to load the
    requested attribute on demand.

    Parameters
    ----------
    name : str
        Name of the attribute to load

    Returns
    -------
    Any
        The requested attribute

    Raises
    ------
    AttributeError
        If the attribute is not found in any module
    """
    # Client module exports
    client_exports = [
        'BaseLLMClient', 'LMStudioClient', 'create_llm_client',
        'LLMResponse', 'LLMGenerationParams'
    ]
    if name in client_exports:
        items = _lazy_import('client', [name])
        if name in items:
            return items[name]

    # Config module exports
    config_exports = [
        'LLMConfig', 'GenerationConfig', 'ProcessingConfig',
        'CacheConfig', 'MonitoringConfig', 'ProcessingMode',
        'TokenEstimationMethod', 'TruncationStrategy', 'CacheType',
        'get_model_preset', 'create_config_from_dict'
    ]
    if name in config_exports:
        items = _lazy_import('config', [name])
        if name in items:
            return items[name]

    # Prompt module exports
    prompt_exports = [
        'PromptTemplate', 'PromptLibrary', 'PromptFormatter',
        'PromptChainBuilder', 'PromptStrategy', 'PromptValidationError',
        'PromptConfig', 'create_prompt_formatter', 'load_prompt_library'
    ]
    if name in prompt_exports:
        items = _lazy_import('prompt', [name])
        if name in items:
            return items[name]

    # Metrics module exports
    metrics_exports = [
        'ProcessingResult', 'BatchResult', 'MetricsCollector',
        'MetricsAggregator', 'ResultStatus', 'MetricType',
        'LatencyMetrics', 'ThroughputMetrics', 'TokenMetrics',
        'CacheMetrics', 'QualityMetrics', 'ResourceMetrics',
        'AggregatedMetrics', 'create_metrics_summary',
        'calculate_percentiles', 'format_latency_ms', 'format_throughput'
    ]
    if name in metrics_exports:
        items = _lazy_import('metrics', [name])
        if name in items:
            return items[name]

    # Processing module exports
    processing_exports = [
        'TextNormalizer', 'TokenEstimator', 'TextTruncator',
        'ResponseProcessor', 'MarkerManager', 'BatchProcessor',
        'TextChunker', 'ProcessedText', 'ResponseAnalysis',
        'ResponseType', 'normalize_text', 'clean_whitespace',
        'estimate_tokens', 'truncate_text', 'extract_response_text',
        'analyze_response', 'has_processing_marker',
        'add_processing_marker', 'remove_processing_marker',
        'chunk_long_text'
    ]
    if name in processing_exports:
        items = _lazy_import('processing', [name])
        if name in items:
            return items[name]

    # Error classes from base module
    error_exports = [
        'LLMError', 'LLMConnectionError', 'LLMGenerationError',
        'LLMResponseError'
    ]
    if name in error_exports:
        items = _lazy_import('base', [name])
        if name in items:
            return items[name]

    # If we get here, the attribute wasn't found
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Return list of available attributes.

    This is used by dir() and autocomplete in interactive environments.

    Returns
    -------
    list
        List of available attribute names
    """
    return __all__


# Package-level initialization
def initialize():
    """
    Initialize the LLM subsystem.

    This function can be called to perform any necessary initialization
    for the LLM subsystem, such as checking dependencies or setting up
    default configurations.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Check optional dependencies
    try:
        import lmstudio
        logger.debug("lmstudio package available")
    except ImportError:
        logger.debug("lmstudio package not available - LM Studio support disabled")

    try:
        import tiktoken
        logger.debug("tiktoken package available - accurate token counting enabled")
    except ImportError:
        logger.debug("tiktoken package not available - using approximate token counting")

    try:
        import numpy
        logger.debug("numpy package available - advanced metrics enabled")
    except ImportError:
        logger.debug("numpy package not available - basic metrics only")


# Convenience function for configuration loading
def load_config_from_file(filepath):
    """
    Load LLM configuration from a JSON or YAML file.

    Parameters
    ----------
    filepath : str or Path
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary ready for use

    Examples
    --------
    >>> # Load from project config
    >>> config = load_config_from_file('configs/llm_config.json')
    >>> client = create_llm_client(**config['llm'])
    """
    import json
    from pathlib import Path

    path = Path(filepath)

    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif path.suffix in ['.yml', '.yaml']:
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


# Module metadata
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"
__maintainer__ = "PAMOLA Core Team"
__email__ = "team@pamola_core"
__status__ = "stable"