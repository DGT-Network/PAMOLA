"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Transformer
Package:       pamola_core.utils.nlp.text_transformer
Version:       2.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides a high-level interface for text transformation using
Large Language Models. It orchestrates LLM operations, batch processing,
caching, and progress tracking while leveraging the modular LLM subsystem
components for specific functionality.

Key Features:
- High-level text transformation interface
- Batch processing with DataFrame support
- Integrated caching with multiple backends
- Progress tracking and checkpointing
- Comprehensive metrics collection
- Error handling and retry logic
- Memory-efficient processing
- Support for multiple LLM providers
- Configurable processing strategies
- Text canonicalization for cache consistency

Framework:
Part of PAMOLA.CORE NLP utilities, serving as the main interface for
LLM-based text processing operations. Integrates with the core
infrastructure for caching, progress tracking, and error handling.

Changelog:
2.2.0 - Fixed cache key generation with text canonicalization
    - Integrated with new TextCache for consistent caching
    - Improved marker handling to prevent cache collisions
    - Added proper handling of already processed text
    - Enhanced cache key generation for reliability
2.1.0 - Fixed integration with modular LLM subsystem
    - Added proper GenerationConfig to LLMGenerationParams conversion
    - Fixed client method calls (disconnect vs close)
    - Improved cache key generation
    - Added model preset merging
    - Enhanced error handling
    - Fixed variable name conflicts
2.0.0 - Complete refactoring using modular LLM subsystem
    - Separated concerns into specialized modules
    - Improved architecture and maintainability
    - Enhanced performance and monitoring
1.8.0 - Last monolithic version before refactoring

Dependencies:
- pamola_core.utils.nlp.llm.* - LLM subsystem modules
- pamola_core.utils.nlp.cache - Caching infrastructure
- pamola_core.utils.progress - Progress tracking
- pandas - DataFrame operations

TODO:
- Add support for streaming transformations
- Implement multi-model ensemble processing
- Add automatic prompt optimization
- Support for few-shot learning workflows
- Implement adaptive batching based on model load
"""

import gc
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

# PAMOLA Core imports
from pamola_core.utils.nlp.cache import get_cache, CacheBase
from pamola_core.utils.nlp.llm.client import (
    create_llm_client,
    BaseLLMClient,
    LLMConnectionError,
    LLMGenerationParams,
)
from pamola_core.utils.nlp.llm.config import (
    LLMConfig,
    ProcessingConfig,
    GenerationConfig,
    CacheConfig,
    MonitoringConfig,
)
from pamola_core.utils.nlp.llm.metrics import (
    ProcessingResult,
    BatchResult,
    MetricsCollector,
    ResultStatus,
    create_metrics_summary,
)
from pamola_core.utils.nlp.llm.processing import (
    TextNormalizer,
    TokenEstimator,
    TextTruncator,
    ResponseProcessor,
    MarkerManager,
    BatchProcessor,
    has_processing_marker,
    ResponseType,
    canonicalize_text,
)
from pamola_core.utils.nlp.llm.prompt import (
    PromptTemplate,
    PromptFormatter,
    PromptLibrary,
)
from pamola_core.utils.progress import track_operation_safely

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHECKPOINT_PREFIX = "text_transformer"
CHECKPOINT_VERSION = "2.0"


def _load_checkpoint(checkpoint_path: Path) -> int:
    """Load checkpoint if exists."""
    if not checkpoint_path.exists():
        return 0

    try:
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        # Validate version
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            logger.warning("Checkpoint version mismatch, ignoring")
            return 0

        return checkpoint.get("last_index", 0)

    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return 0


class TextTransformer:
    """
    High-level interface for LLM-based text transformation.

    This class provides a unified interface for transforming text using
    Large Language Models, with support for batch processing, caching,
    and comprehensive metrics collection.

    Parameters
    ----------
    llm_config : LLMConfig or dict
        LLM connection configuration
    processing_config : ProcessingConfig or dict
        Processing behavior configuration
    generation_config : GenerationConfig or dict
        Text generation parameters
    prompt_template : PromptTemplate or str or dict
        Prompt template or template name
    task_dir : Path or str
        Directory for task artifacts
    cache_config : CacheConfig or dict, optional
        Cache configuration
    monitoring_config : MonitoringConfig or dict, optional
        Monitoring configuration

    Examples
    --------
    >>> # Basic usage with built-in template
    >>> df = pd.DataFrame({"text_column": ["A"]})
    >>> transformer = TextTransformer(
    ...     llm_config={'provider': 'lmstudio', 'model_name': 'gemma-2'},
    ...     processing_config={'batch_size': 8},
    ...     generation_config={'temperature': 0.3},
    ...     prompt_template='anonymize_experience_ru',
    ...     task_dir='./output'
    ... )
    >>>
    >>> # Process DataFrame
    >>> result_df = transformer.process_dataframe(
    ...     df, 'text_column', 'processed_column'
    ... )
    """

    def __init__(
        self,
        llm_config: Union[Dict[str, Any], LLMConfig],
        processing_config: Union[Dict[str, Any], ProcessingConfig],
        generation_config: Union[Dict[str, Any], GenerationConfig],
        prompt_template: Union[str, Dict[str, Any], PromptTemplate],
        task_dir: Union[str, Path],
        cache_config: Optional[Union[Dict[str, Any], CacheConfig]] = None,
        monitoring_config: Optional[Union[Dict[str, Any], MonitoringConfig]] = None,
    ):
        """Initialize the text transformer."""
        # Convert configs to dataclasses if needed
        self.llm_config = (
            llm_config if isinstance(llm_config, LLMConfig) else LLMConfig(**llm_config)
        )
        self.processing_config = (
            processing_config
            if isinstance(processing_config, ProcessingConfig)
            else ProcessingConfig(**processing_config)
        )
        self.generation_config = (
            generation_config
            if isinstance(generation_config, GenerationConfig)
            else GenerationConfig(**generation_config)
        )

        # Apply model presets to generation config
        self.generation_config = self.generation_config.merge_with_model_defaults(
            self.llm_config.model_name
        )

        self.cache_config = (
            cache_config
            if isinstance(cache_config, CacheConfig)
            else CacheConfig(**cache_config) if cache_config else CacheConfig()
        )
        self.monitoring_config = (
            monitoring_config
            if isinstance(monitoring_config, MonitoringConfig)
            else (
                MonitoringConfig(**monitoring_config)
                if monitoring_config
                else MonitoringConfig()
            )
        )

        # Set up directories
        self.task_dir = Path(task_dir)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.task_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize prompt template
        self._setup_prompt_template(prompt_template)

        # Initialize components
        self.llm_client: Optional[BaseLLMClient] = None
        self.cache: Optional[CacheBase] = None
        self.metrics_collector = MetricsCollector()

        # Processing utilities
        self.text_normalizer = TextNormalizer()
        self.token_estimator = TokenEstimator(
            self.processing_config.token_estimation_method.value
        )
        self.text_truncator = TextTruncator(self.token_estimator)
        self.response_processor = ResponseProcessor()
        self.marker_manager = MarkerManager(self.processing_config.processing_marker)
        self.batch_processor = BatchProcessor()
        self.prompt_formatter = PromptFormatter()

        # Thread safety
        self._client_lock = threading.RLock()
        self._cache_lock = threading.RLock()

        # Initialize cache if enabled
        if self.cache_config.enabled:
            self._initialize_cache()

        # Initialize LLM client
        self._initialize_llm_client()

        # Processing state
        self._stop_flag_path = self.task_dir / "stop.flag"
        self._processed_count = 0
        self._error_count = 0

        logger.info(
            f"TextTransformer initialized with model: {self.llm_config.model_name}, "
            f"task_dir: {self.task_dir}"
        )

    def _setup_prompt_template(
        self, template_input: Union[str, Dict[str, Any], PromptTemplate]
    ) -> None:
        """Set up prompt template from various input types."""
        if isinstance(template_input, PromptTemplate):
            self.prompt_template = template_input
        elif isinstance(template_input, str):
            # Load from library
            library = PromptLibrary()
            self.prompt_template = library.get(template_input)
        elif isinstance(template_input, dict):
            # Create from dict
            self.prompt_template = PromptTemplate(**template_input)
        else:
            raise ValueError(f"Invalid prompt template type: {type(template_input)}")

        logger.info(f"Using prompt template: {self.prompt_template.name}")

    def _initialize_cache(self) -> None:
        """Initialize cache backend."""
        with self._cache_lock:
            # Use text cache type for better text handling
            cache_type = (
                "text"
                if self.cache_config.cache_type.value == "memory"
                else self.cache_config.cache_type.value
            )
            self.cache = get_cache(cache_type)
            logger.info(f"Initialized {cache_type} cache")

    def _initialize_llm_client(self) -> None:
        """Initialize LLM client."""
        with self._client_lock:
            try:
                # Create client with proper parameters
                self.llm_client = create_llm_client(
                    provider=self.llm_config.provider,
                    websocket_url=self.llm_config.api_url,  # For lmstudio
                    model_name=self.llm_config.model_name,
                    api_key=self.llm_config.api_key,
                    ttl=self.llm_config.ttl,
                    timeout=self.llm_config.timeout,
                    max_retries=self.llm_config.max_retries,
                    retry_delay=self.llm_config.retry_delay,
                    debug_logging=self.monitoring_config.debug_mode,
                    debug_log_file=self.monitoring_config.debug_log_file,
                )

                # Connect to the service
                self.llm_client.connect()

                # Test connection
                self._test_connection()

                logger.info(f"Successfully connected to {self.llm_config.provider}")

            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise

    def _test_connection(self) -> None:
        """Test LLM connection with a simple request."""
        test_prompt = self.prompt_template.format(text="Test connection")

        try:
            params = self._build_generation_params()
            response = self.llm_client.generate(test_prompt, params)

            # Check response validity
            if not response.text:
                raise LLMConnectionError("Empty response from LLM")

            logger.info(
                f"Connection test successful, response length: {len(response.text)}"
            )
        except Exception as e:
            raise LLMConnectionError(f"Connection test failed: {e}")

    def _build_generation_params(self) -> LLMGenerationParams:
        """Convert GenerationConfig to LLMGenerationParams."""
        # Build params dict first to handle conditional inclusion
        params = {
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
            "max_tokens": self.generation_config.max_tokens,
            "stop_sequences": self.generation_config.stop_sequences,
            "stream": self.generation_config.stream,
        }

        # Only include repeat_penalty if it's not None and > 0
        if (
            self.generation_config.repeat_penalty
            and self.generation_config.repeat_penalty > 0
        ):
            params["repeat_penalty"] = self.generation_config.repeat_penalty

        return LLMGenerationParams(**params)

    def _generate_cache_key(self, text: str, field_name: Optional[str] = None) -> str:
        """
        Generate cache key for text using canonicalization.

        This method ensures consistent cache keys by:
        1. Canonicalizing the text (removing markers, normalizing whitespace)
        2. Including all relevant generation parameters
        3. Creating a deterministic hash

        Parameters
        ----------
        text : str
            Input text
        field_name : str, optional
            Field name for additional context

        Returns
        -------
        str
            Cache key hash
        """
        import hashlib

        # Canonicalize text to ensure consistency
        # This removes any processing markers and normalizes the text
        canonical_text = canonicalize_text(
            text, self.processing_config.processing_marker
        )

        # Get API parameters for provider
        api_params_str = str(
            self.generation_config.to_api_params(self.llm_config.provider)
        )

        # Include all relevant parameters in cache key
        key_parts = [
            canonical_text,
            self.prompt_template.template,
            self.llm_config.model_name,
            api_params_str,
            field_name or "",
        ]

        # Create deterministic key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _process_single_text(
        self, text: str, field_name: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single text through LLM.

        Parameters
        ----------
        text : str
            Text to process
        field_name : str, optional
            Field name for context

        Returns
        -------
        ProcessingResult
            Processing result with metrics
        """
        start_time = time.time()

        # Normalize input
        normalized_text = self.text_normalizer.normalize(text)
        original_text = normalized_text

        # Check if already processed
        if self.processing_config.use_processing_marker:
            has_marker, clean_text = self.marker_manager.extract_marked_and_clean(
                normalized_text
            )
            if has_marker:
                # Already processed - return as-is
                return ProcessingResult(
                    text=normalized_text,
                    original_text=original_text,
                    success=True,
                    status=ResultStatus.SUCCESS,
                    from_cache=False,
                    processing_time=time.time() - start_time,
                    metadata={"already_processed": True},
                )
        else:
            clean_text = normalized_text

        # Generate cache key using clean text (without marker)
        # This ensures consistent cache keys regardless of processing state
        cache_key = self._generate_cache_key(clean_text, field_name)

        # Check cache
        if self.cache_config.enabled and self.cache:
            # For text cache, we can pass the original text directly
            # The TextCache will handle canonicalization internally
            cached_result = self.cache.get(clean_text)

            if cached_result is not None:
                # Add marker if needed
                final_text = (
                    self.marker_manager.add_marker(cached_result)
                    if self.processing_config.use_processing_marker
                    else cached_result
                )

                return ProcessingResult(
                    text=final_text,
                    original_text=original_text,
                    success=True,
                    status=ResultStatus.SUCCESS,
                    from_cache=True,
                    processing_time=time.time() - start_time,
                    tokens_input=self.token_estimator.estimate(clean_text),
                )

        # Truncate if needed
        truncated_result = self.text_truncator.truncate(
            clean_text,
            self.processing_config.max_input_tokens,
            self.processing_config.truncation_strategy.value,
        )

        # Format prompt with sensible defaults
        prompt_vars = {
            "text": truncated_result.text,
            "field": field_name or "experience",
        }
        formatted_prompt = self.prompt_formatter.format_prompt(
            self.prompt_template, prompt_vars
        )

        # Call LLM
        try:
            llm_start = time.time()
            params = self._build_generation_params()
            response = self.llm_client.generate(formatted_prompt, params)
            model_time = time.time() - llm_start

            # Process response
            analysis = self.response_processor.analyze_response(response.text)

            if analysis.response_type == ResponseType.VALID:
                # Success - cache the result
                if self.cache_config.enabled and self.cache:
                    # Cache the clean result without marker
                    # Using clean_text as key for TextCache
                    self.cache.set(
                        clean_text, analysis.cleaned_text, ttl=self.cache_config.ttl
                    )

                # Add marker if needed
                final_text = (
                    self.marker_manager.add_marker(analysis.cleaned_text)
                    if self.processing_config.use_processing_marker
                    else analysis.cleaned_text
                )

                return ProcessingResult(
                    text=final_text,
                    original_text=original_text,
                    success=True,
                    status=ResultStatus.SUCCESS,
                    processing_time=time.time() - start_time,
                    model_time=model_time,
                    tokens_input=truncated_result.estimated_tokens,
                    tokens_output=self.token_estimator.estimate(analysis.cleaned_text),
                    tokens_truncated=truncated_result.truncated_tokens,
                    confidence_score=analysis.confidence,
                    metadata={
                        "response_type": analysis.response_type.value,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                        "model_name": response.model_name,
                        "finish_reason": response.finish_reason,
                    },
                )
            else:
                # Invalid response
                return ProcessingResult(
                    text=original_text,
                    original_text=original_text,
                    success=False,
                    status=ResultStatus.FAILURE,
                    error=f"Invalid response: {analysis.response_type.value}",
                    processing_time=time.time() - start_time,
                    model_time=model_time,
                    tokens_input=truncated_result.estimated_tokens,
                    metadata={
                        "response_type": analysis.response_type.value,
                        "issues": analysis.issues,
                    },
                )

        except LLMConnectionError as e:
            # Specific handling for connection errors
            logger.error(f"LLM connection error: {e}")
            return ProcessingResult(
                text=original_text,
                original_text=original_text,
                success=False,
                status=ResultStatus.FAILURE,
                error=f"Connection error: {str(e)}",
                processing_time=time.time() - start_time,
                tokens_input=self.token_estimator.estimate(clean_text),
            )
        except Exception as e:
            # General error handling
            logger.error(f"LLM processing failed: {e}")
            return ProcessingResult(
                text=original_text,
                original_text=original_text,
                success=False,
                status=ResultStatus.FAILURE,
                error=str(e),
                processing_time=time.time() - start_time,
                tokens_input=self.token_estimator.estimate(clean_text),
            )

    def _process_batch(
        self,
        texts: List[str],
        field_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Process a batch of texts."""
        batch_begin = time.time()  # Renamed from batch_start to avoid conflict
        results = []

        for i, text in enumerate(texts):
            # Check stop flag
            if self._check_stop_flag():
                logger.info("Stop flag detected, interrupting batch processing")
                break

            # Process single text
            result = self._process_single_text(text, field_name)
            results.append(result)

            # Update progress
            if progress_callback:
                progress_callback(i + 1, len(texts))

        # Create batch result
        batch_result = BatchResult(
            results=results,
            batch_id=f"batch_{int(time.time())}",
            total_time=time.time() - batch_begin,
        )

        # Update metrics
        self.metrics_collector.add_batch_result(batch_result)

        return batch_result

    def process_dataframe(
        self,
        df: pd.DataFrame,
        source_column: str,
        target_column: str,
        error_column: Optional[str] = None,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
        id_column: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        skip_processed: Optional[bool] = None,
        max_records: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process DataFrame texts through LLM.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str
            Column containing source text
        target_column : str
            Column to store processed text
        error_column : str, optional
            Column to store error messages
        start_id : int, optional
            Starting record ID
        end_id : int, optional
            Ending record ID
        id_column : str, optional
            Column for ID-based filtering
        progress_callback : callable, optional
            Progress callback function
        skip_processed : bool, optional
            Skip already processed records
        max_records : int, optional
            Maximum records to process

        Returns
        -------
        pd.DataFrame
            DataFrame with processed texts
        """
        # Validate inputs
        if source_column not in df.columns:
            raise ValueError(f"Source column '{source_column}' not found")

        # Use config defaults if not specified
        if skip_processed is None:
            skip_processed = self.processing_config.skip_processed
        if max_records is None:
            max_records = self.processing_config.max_records

        # Work with copy to avoid modifying original
        result_df = df.copy()

        # Initialize columns
        if target_column not in result_df.columns:
            result_df[target_column] = ""
        if error_column and error_column not in result_df.columns:
            result_df[error_column] = ""

        # Filter records to process
        indices_to_process = self._get_indices_to_process(
            result_df,
            source_column,
            target_column,
            start_id,
            end_id,
            id_column,
            skip_processed,
            max_records,
        )

        total_records = len(indices_to_process)
        logger.info(f"Processing {total_records} records")

        # Load checkpoint if exists
        checkpoint_path = self._get_checkpoint_path(
            source_column, id_column, start_id, end_id
        )
        last_processed = _load_checkpoint(checkpoint_path)

        # Process records
        processed_count = 0

        with track_operation_safely(
            description=f"Processing {source_column}",
            total=total_records,
            unit="records",
        ) as pbar:
            # Process in batches
            batch_size = self.processing_config.batch_size

            for batch_start in range(last_processed, total_records, batch_size):
                # Check stop flag
                if self._check_stop_flag():
                    logger.info("Stop flag detected, saving checkpoint")
                    self._save_checkpoint(checkpoint_path, batch_start)
                    break

                # Get batch indices
                batch_end = min(batch_start + batch_size, total_records)
                batch_indices = indices_to_process[batch_start:batch_end]

                # Extract batch texts
                batch_texts = [
                    self.text_normalizer.normalize(result_df.at[idx, source_column])
                    for idx in batch_indices
                ]

                # Process batch
                batch_result = self._process_batch(
                    batch_texts, source_column, progress_callback
                )

                # Update DataFrame
                for idx, result in zip(batch_indices, batch_result.results):
                    result_df.at[idx, target_column] = result.text
                    if error_column and not result.success:
                        result_df.at[idx, error_column] = (
                            result.error or "Processing failed"
                        )

                    if result.success:
                        processed_count += 1
                    else:
                        self._error_count += 1

                # Update progress
                pbar.update(
                    len(batch_indices),
                    {
                        "processed": processed_count,
                        "errors": self._error_count,
                        "cache_hits": sum(
                            1 for r in batch_result.results if r.from_cache
                        ),
                    },
                )

                # Save checkpoint periodically
                if batch_end % (batch_size * 10) == 0:
                    self._save_checkpoint(checkpoint_path, batch_end)

                # Clean memory periodically
                if batch_end % self.processing_config.memory_cleanup_interval == 0:
                    self._cleanup_memory()

        # Clean up checkpoint
        if processed_count >= total_records and checkpoint_path.exists():
            checkpoint_path.unlink()

        # Log summary
        metrics = self.metrics_collector.get_current_metrics()
        logger.info(
            f"Processing completed: {processed_count}/{total_records} successful"
        )
        logger.info(f"Errors: {self._error_count}")
        logger.info(f"Cache hit rate: {metrics.cache.hit_rate:.1%}")

        return result_df

    def _get_indices_to_process(
        self,
        df: pd.DataFrame,
        source_column: str,
        target_column: str,
        start_id: Optional[int],
        end_id: Optional[int],
        id_column: Optional[str],
        skip_processed: bool,
        max_records: Optional[int],
    ) -> List[int]:
        """Get indices of records to process."""
        # Apply ID range filtering
        if id_column and id_column in df.columns:
            mask = pd.Series([True] * len(df), index=df.index)
            if start_id is not None:
                mask &= df[id_column] >= start_id
            if end_id is not None:
                mask &= df[id_column] <= end_id
            indices = df[mask].index.tolist()
        else:
            # Use positional indexing
            start_idx = start_id if start_id is not None else 0
            end_idx = min(end_id, len(df)) if end_id is not None else len(df)
            indices = df.index[start_idx:end_idx].tolist()

        # Filter already processed if needed
        if skip_processed and self.processing_config.use_processing_marker:
            unprocessed_indices = []
            for idx in indices:
                target_value = df.at[idx, target_column]
                if not has_processing_marker(
                    target_value, self.processing_config.processing_marker
                ):
                    unprocessed_indices.append(idx)

            skipped = len(indices) - len(unprocessed_indices)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already processed records")

            indices = unprocessed_indices

        # Apply max_records limit
        if max_records and len(indices) > max_records:
            logger.info(f"Limiting to {max_records} records")
            indices = indices[:max_records]

        return indices

    def _get_checkpoint_path(
        self,
        source_column: str,
        id_column: Optional[str],
        start_id: Optional[int],
        end_id: Optional[int],
    ) -> Path:
        """Generate checkpoint file path using deterministic hash."""
        import hashlib

        # Create deterministic components
        components = [
            DEFAULT_CHECKPOINT_PREFIX,
            source_column,
            id_column or "idx",
            str(start_id or 0),
            str(end_id or "end"),
            self.llm_config.model_name,
            self.prompt_template.name,
        ]

        # Generate hash from components
        hash_input = "|".join(components)
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        filename = f"checkpoint_{file_hash}.json"
        return self.checkpoint_dir / filename

    def _save_checkpoint(self, checkpoint_path: Path, last_index: int) -> None:
        """Save checkpoint."""
        try:
            checkpoint = {
                "version": CHECKPOINT_VERSION,
                "last_index": last_index,
                "timestamp": datetime.now().isoformat(),
                "model": self.llm_config.model_name,
                "processed_count": self._processed_count,
                "error_count": self._error_count,
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _check_stop_flag(self) -> bool:
        """Check if stop flag exists."""
        return self._stop_flag_path.exists()

    def _cleanup_memory(self) -> None:
        """Clean up memory."""
        # Always collect garbage to free memory in production
        gc.collect()

        # Log memory usage only if profiling is enabled
        if self.monitoring_config.profile_memory:
            try:
                import psutil

                process = psutil.Process()
                memory_info = process.memory_info()
                logger.debug(f"Memory usage: RSS={memory_info.rss / 1024 / 1024:.1f}MB")
            except ImportError:
                pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns
        -------
        dict
            Comprehensive metrics dictionary
        """
        metrics = self.metrics_collector.get_current_metrics()

        # Add client statistics
        if self.llm_client:
            client_stats = self.llm_client.get_stats()
            metrics_dict = metrics.to_dict()
            metrics_dict["llm_client"] = client_stats
        else:
            metrics_dict = metrics.to_dict()

        # Add cache statistics
        if self.cache:
            cache_stats = self.cache.get_stats()
            metrics_dict["cache_backend"] = cache_stats

        return metrics_dict

    def print_metrics_summary(self) -> None:
        """Print human-readable metrics summary."""
        metrics = self.metrics_collector.get_current_metrics()
        summary = create_metrics_summary(metrics)
        print(summary)

    def export_metrics(self, filepath: Union[str, Path]) -> None:
        """Export metrics to file."""
        self.metrics_collector.export_metrics(filepath)
        logger.info(f"Metrics exported to {filepath}")

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics_collector.reset()
        self._processed_count = 0
        self._error_count = 0
        logger.info("Metrics reset")

    def close(self) -> None:
        """Clean up resources."""
        # Disconnect LLM client (use correct method)
        if self.llm_client:
            try:
                self.llm_client.disconnect()
                logger.info("LLM client disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting LLM client: {e}")

        # Log final metrics
        self.print_metrics_summary()

        # Export metrics
        metrics_path = (
            self.task_dir
            / f"final_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.export_metrics(metrics_path)

        # Clear cache reference
        self.cache = None

        # Final memory cleanup
        self._cleanup_memory()

        logger.info("TextTransformer closed successfully")
