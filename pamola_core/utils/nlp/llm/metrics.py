"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Metrics and Results
Package:       pamola_core.utils.nlp.llm.metrics
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides comprehensive metrics collection, aggregation, and
analysis for LLM operations. It defines dataclasses for various metric
types, result containers, and utilities for tracking performance, quality,
and resource usage across LLM interactions.

Key Features:
- Structured result containers for LLM operations
- Performance metrics tracking (latency, throughput, etc.)
- Quality metrics for text generation
- Cache performance analytics
- Token usage tracking and analysis
- Batch processing metrics
- Statistical aggregation utilities
- Metric visualization helpers
- Time-series metric tracking
- Alert thresholds and monitoring

Framework:
Part of PAMOLA.CORE LLM utilities, providing standardized metrics
and monitoring capabilities for all LLM operations.

Dependencies:
- Standard library for core functionality
- Optional: numpy for statistical calculations
- Optional: matplotlib for visualization

TODO:
- Add streaming metrics support
- Implement metric persistence layer
- Add comparative metrics between models
- Create metric dashboards
- Add anomaly detection for metrics
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_METRIC_WINDOW = 1000  # Keep last N metrics
DEFAULT_PERCENTILES = [50, 75, 90, 95, 99]
METRIC_DECIMAL_PLACES = 3


class MetricType(Enum):
    """Types of metrics collected."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKEN_USAGE = "token_usage"
    CACHE_PERFORMANCE = "cache_performance"
    ERROR_RATE = "error_rate"
    QUALITY = "quality"
    RESOURCE = "resource"


class ResultStatus(Enum):
    """Status of operation results."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ProcessingResult:
    """
    Result of single text processing operation.

    Attributes
    ----------
    text : str
        Processed text output
    original_text : str
        Original input text
    success : bool
        Whether processing succeeded
    status : ResultStatus
        Detailed status
    from_cache : bool
        Whether result came from cache
    error : Optional[str]
        Error message if failed
    retry_count : int
        Number of retries attempted
    processing_time : float
        Total processing time in seconds
    model_time : float
        Time spent in model inference
    tokens_input : int
        Input token count
    tokens_output : int
        Output token count
    tokens_truncated : int
        Tokens removed by truncation
    confidence_score : float
        Confidence in result quality (0-1)
    metadata : Dict[str, Any]
        Additional metadata
    timestamp : datetime
        When processing occurred
    """
    text: str
    original_text: str = ""
    success: bool = True
    status: ResultStatus = ResultStatus.SUCCESS
    from_cache: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0
    model_time: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_truncated: int = 0
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self.tokens_input + self.tokens_output

    @property
    def overhead_time(self) -> float:
        """Time spent outside model inference."""
        return max(0, self.processing_time - self.model_time)


@dataclass
class BatchResult:
    """
    Result of batch processing operation.

    Attributes
    ----------
    results : List[ProcessingResult]
        Individual results
    batch_id : str
        Unique batch identifier
    total_items : int
        Total items in batch
    successful_items : int
        Successfully processed items
    failed_items : int
        Failed items
    cached_items : int
        Items served from cache
    total_time : float
        Total batch processing time
    average_time_per_item : float
        Average processing time
    metadata : Dict[str, Any]
        Batch metadata
    """
    results: List[ProcessingResult]
    batch_id: str = ""
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    cached_items: int = 0
    total_time: float = 0.0
    average_time_per_item: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields."""
        if self.results and not self.total_items:
            self.total_items = len(self.results)
            self.successful_items = sum(1 for r in self.results if r.success)
            self.failed_items = self.total_items - self.successful_items
            self.cached_items = sum(1 for r in self.results if r.from_cache)

            if self.total_items > 0:
                self.average_time_per_item = self.total_time / self.total_items

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.cached_items / self.total_items if self.total_items > 0 else 0.0


@dataclass
class LatencyMetrics:
    """
    Latency metrics for LLM operations.

    Attributes
    ----------
    mean : float
        Mean latency in seconds
    median : float
        Median latency
    std : float
        Standard deviation
    min : float
        Minimum latency
    max : float
        Maximum latency
    percentiles : Dict[int, float]
        Latency percentiles (50, 75, 90, 95, 99)
    samples : int
        Number of samples
    """
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)
    samples: int = 0

    @classmethod
    def from_values(cls, values: List[float], percentiles: List[int] = None) -> 'LatencyMetrics':
        """
        Create from list of latency values.

        Parameters
        ----------
        values : List[float]
            Latency measurements
        percentiles : List[int], optional
            Percentiles to calculate

        Returns
        -------
        LatencyMetrics
            Calculated metrics
        """
        if not values:
            return cls()

        if percentiles is None:
            percentiles = DEFAULT_PERCENTILES

        arr = np.array(values)
        percentile_values = {
            p: float(np.percentile(arr, p))
            for p in percentiles
        }

        return cls(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            percentiles=percentile_values,
            samples=len(values)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with formatted values."""
        return {
            'mean_ms': round(self.mean * 1000, METRIC_DECIMAL_PLACES),
            'median_ms': round(self.median * 1000, METRIC_DECIMAL_PLACES),
            'std_ms': round(self.std * 1000, METRIC_DECIMAL_PLACES),
            'min_ms': round(self.min * 1000, METRIC_DECIMAL_PLACES),
            'max_ms': round(self.max * 1000, METRIC_DECIMAL_PLACES),
            'percentiles': {
                f'p{k}': round(v * 1000, METRIC_DECIMAL_PLACES)
                for k, v in self.percentiles.items()
            },
            'samples': self.samples
        }


@dataclass
class ThroughputMetrics:
    """
    Throughput metrics for LLM operations.

    Attributes
    ----------
    requests_per_second : float
        Average requests processed per second
    tokens_per_second : float
        Average tokens processed per second
    bytes_per_second : float
        Average bytes processed per second
    peak_rps : float
        Peak requests per second
    measurement_duration : float
        Duration of measurement in seconds
    total_requests : int
        Total requests processed
    """
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    bytes_per_second: float = 0.0
    peak_rps: float = 0.0
    measurement_duration: float = 0.0
    total_requests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with formatted values."""
        return {
            'requests_per_second': round(self.requests_per_second, METRIC_DECIMAL_PLACES),
            'tokens_per_second': round(self.tokens_per_second, METRIC_DECIMAL_PLACES),
            'bytes_per_second': round(self.bytes_per_second, METRIC_DECIMAL_PLACES),
            'peak_rps': round(self.peak_rps, METRIC_DECIMAL_PLACES),
            'measurement_duration_seconds': round(self.measurement_duration, METRIC_DECIMAL_PLACES),
            'total_requests': self.total_requests
        }


@dataclass
class TokenMetrics:
    """
    Token usage metrics.

    Attributes
    ----------
    total_input_tokens : int
        Total input tokens processed
    total_output_tokens : int
        Total output tokens generated
    total_truncated_tokens : int
        Total tokens truncated
    average_input_tokens : float
        Average input tokens per request
    average_output_tokens : float
        Average output tokens per request
    max_input_tokens : int
        Maximum input tokens in single request
    max_output_tokens : int
        Maximum output tokens in single request
    token_efficiency : float
        Ratio of output to input tokens
    """
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_truncated_tokens: int = 0
    average_input_tokens: float = 0.0
    average_output_tokens: float = 0.0
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    token_efficiency: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def truncation_rate(self) -> float:
        """Rate of token truncation."""
        total = self.total_input_tokens + self.total_truncated_tokens
        return self.total_truncated_tokens / total if total > 0 else 0.0

    def update(self, result: ProcessingResult) -> None:
        """Update metrics with new result."""
        self.total_input_tokens += result.tokens_input
        self.total_output_tokens += result.tokens_output
        self.total_truncated_tokens += result.tokens_truncated
        self.max_input_tokens = max(self.max_input_tokens, result.tokens_input)
        self.max_output_tokens = max(self.max_output_tokens, result.tokens_output)


@dataclass
class CacheMetrics:
    """
    Cache performance metrics.

    Attributes
    ----------
    total_requests : int
        Total cache requests
    cache_hits : int
        Number of cache hits
    cache_misses : int
        Number of cache misses
    hit_rate : float
        Cache hit rate (0-1)
    average_hit_time : float
        Average time for cache hit
    average_miss_time : float
        Average time for cache miss
    memory_usage_bytes : int
        Cache memory usage
    evictions : int
        Number of cache evictions
    """
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    average_hit_time: float = 0.0
    average_miss_time: float = 0.0
    memory_usage_bytes: int = 0
    evictions: int = 0

    def update_hit(self, response_time: float) -> None:
        """Update metrics for cache hit."""
        self.total_requests += 1
        self.cache_hits += 1

        # Update average hit time
        if self.cache_hits == 1:
            self.average_hit_time = response_time
        else:
            self.average_hit_time = (
                    (self.average_hit_time * (self.cache_hits - 1) + response_time) /
                    self.cache_hits
            )

        self._update_hit_rate()

    def update_miss(self, response_time: float) -> None:
        """Update metrics for cache miss."""
        self.total_requests += 1
        self.cache_misses += 1

        # Update average miss time
        if self.cache_misses == 1:
            self.average_miss_time = response_time
        else:
            self.average_miss_time = (
                    (self.average_miss_time * (self.cache_misses - 1) + response_time) /
                    self.cache_misses
            )

        self._update_hit_rate()

    def _update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests


@dataclass
class QualityMetrics:
    """
    Text generation quality metrics.

    Attributes
    ----------
    average_confidence : float
        Average confidence score
    min_confidence : float
        Minimum confidence score
    max_confidence : float
        Maximum confidence score
    invalid_responses : int
        Number of invalid responses
    service_responses : int
        Number of service responses
    error_responses : int
        Number of error responses
    empty_responses : int
        Number of empty responses
    response_diversity : float
        Diversity score of responses
    """
    average_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    invalid_responses: int = 0
    service_responses: int = 0
    error_responses: int = 0
    empty_responses: int = 0
    response_diversity: float = 0.0

    def update(self, confidence: float, response_type: str = "valid") -> None:
        """Update quality metrics."""
        # Update confidence stats
        self.min_confidence = min(self.min_confidence, confidence)
        self.max_confidence = max(self.max_confidence, confidence)

        # Track response types
        if response_type == "invalid":
            self.invalid_responses += 1
        elif response_type == "service":
            self.service_responses += 1
        elif response_type == "error":
            self.error_responses += 1
        elif response_type == "empty":
            self.empty_responses += 1


@dataclass
class ResourceMetrics:
    """
    Resource usage metrics.

    Attributes
    ----------
    peak_memory_mb : float
        Peak memory usage in MB
    average_memory_mb : float
        Average memory usage in MB
    cpu_usage_percent : float
        Average CPU usage percentage
    active_threads : int
        Number of active threads
    open_connections : int
        Number of open connections
    """
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_threads: int = 0
    open_connections: int = 0


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across all operations.

    Attributes
    ----------
    latency : LatencyMetrics
        Latency statistics
    throughput : ThroughputMetrics
        Throughput statistics
    tokens : TokenMetrics
        Token usage statistics
    cache : CacheMetrics
        Cache performance
    quality : QualityMetrics
        Quality metrics
    resource : ResourceMetrics
        Resource usage
    error_rate : float
        Overall error rate
    total_operations : int
        Total operations performed
    start_time : datetime
        When metrics collection started
    last_update : datetime
        Last metric update
    """
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    cache: CacheMetrics = field(default_factory=CacheMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    resource: ResourceMetrics = field(default_factory=ResourceMetrics)
    error_rate: float = 0.0
    total_operations: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def uptime(self) -> timedelta:
        """Calculate uptime."""
        return self.last_update - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        return 1.0 - self.error_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary."""
        return {
            'summary': {
                'total_operations': self.total_operations,
                'success_rate': round(self.success_rate * 100, 2),
                'error_rate': round(self.error_rate * 100, 2),
                'uptime_seconds': self.uptime.total_seconds(),
                'last_update': self.last_update.isoformat()
            },
            'latency': self.latency.to_dict(),
            'throughput': self.throughput.to_dict(),
            'tokens': asdict(self.tokens),
            'cache': asdict(self.cache),
            'quality': asdict(self.quality),
            'resource': asdict(self.resource)
        }


class MetricsCollector:
    """
    Collects and tracks metrics over time.

    This class maintains a sliding window of metrics for real-time
    monitoring and analysis.
    """

    def __init__(self, window_size: int = DEFAULT_METRIC_WINDOW):
        """
        Initialize metrics collector.

        Parameters
        ----------
        window_size : int
            Size of sliding window for metrics
        """
        self.window_size = window_size
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._throughput_samples: Deque[Tuple[datetime, int]] = deque(maxlen=window_size)
        self._results: Deque[ProcessingResult] = deque(maxlen=window_size)
        self._start_time = datetime.now()
        self._total_processed = 0
        self._total_errors = 0

    def add_result(self, result: ProcessingResult) -> None:
        """Add processing result to metrics."""
        self._results.append(result)
        self._latencies.append(result.processing_time)
        self._throughput_samples.append((result.timestamp, 1))
        self._total_processed += 1

        if not result.success:
            self._total_errors += 1

    def add_batch_result(self, batch_result: BatchResult) -> None:
        """Add batch result to metrics."""
        for result in batch_result.results:
            self.add_result(result)

    def get_current_metrics(self) -> AggregatedMetrics:
        """Get current aggregated metrics."""
        metrics = AggregatedMetrics()
        metrics.total_operations = self._total_processed
        metrics.start_time = self._start_time
        metrics.last_update = datetime.now()

        # Calculate error rate
        if self._total_processed > 0:
            metrics.error_rate = self._total_errors / self._total_processed

        # Calculate latency metrics
        if self._latencies:
            metrics.latency = LatencyMetrics.from_values(list(self._latencies))

        # Calculate throughput
        metrics.throughput = self._calculate_throughput()

        # Calculate token metrics
        for result in self._results:
            metrics.tokens.update(result)

        # Calculate averages
        if len(self._results) > 0:
            metrics.tokens.average_input_tokens = (
                    metrics.tokens.total_input_tokens / len(self._results)
            )
            metrics.tokens.average_output_tokens = (
                    metrics.tokens.total_output_tokens / len(self._results)
            )

            if metrics.tokens.total_input_tokens > 0:
                metrics.tokens.token_efficiency = (
                        metrics.tokens.total_output_tokens /
                        metrics.tokens.total_input_tokens
                )

        # Calculate cache metrics
        cache_hits = sum(1 for r in self._results if r.from_cache)
        cache_misses = len(self._results) - cache_hits

        if cache_hits > 0:
            hit_times = [r.processing_time for r in self._results if r.from_cache]
            metrics.cache.average_hit_time = np.mean(hit_times)

        if cache_misses > 0:
            miss_times = [r.processing_time for r in self._results if not r.from_cache]
            metrics.cache.average_miss_time = np.mean(miss_times)

        metrics.cache.cache_hits = cache_hits
        metrics.cache.cache_misses = cache_misses
        metrics.cache.total_requests = len(self._results)
        if metrics.cache.total_requests > 0:
            metrics.cache.hit_rate = cache_hits / metrics.cache.total_requests

        # Calculate quality metrics
        confidence_scores = [r.confidence_score for r in self._results]
        if confidence_scores:
            metrics.quality.average_confidence = np.mean(confidence_scores)
            metrics.quality.min_confidence = min(confidence_scores)
            metrics.quality.max_confidence = max(confidence_scores)

        return metrics

    def _calculate_throughput(self) -> ThroughputMetrics:
        """Calculate throughput metrics."""
        if not self._throughput_samples:
            return ThroughputMetrics()

        # Get time window
        now = datetime.now()
        window_start = now - timedelta(seconds=60)  # Last minute

        # Count requests in window
        recent_requests = sum(
            count for timestamp, count in self._throughput_samples
            if timestamp > window_start
        )

        # Calculate duration
        if len(self._throughput_samples) > 1:
            duration = (
                    self._throughput_samples[-1][0] -
                    self._throughput_samples[0][0]
            ).total_seconds()
        else:
            duration = 1.0

        # Calculate metrics
        rps = recent_requests / max(duration, 1.0)

        # Calculate token throughput
        recent_tokens = sum(
            r.total_tokens for r in self._results
            if r.timestamp > window_start
        )
        tps = recent_tokens / max(duration, 1.0)

        return ThroughputMetrics(
            requests_per_second=rps,
            tokens_per_second=tps,
            measurement_duration=duration,
            total_requests=self._total_processed
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._latencies.clear()
        self._throughput_samples.clear()
        self._results.clear()
        self._start_time = datetime.now()
        self._total_processed = 0
        self._total_errors = 0

    def export_metrics(self, filepath: Union[str, Path]) -> None:
        """Export metrics to JSON file."""
        metrics = self.get_current_metrics()
        path = Path(filepath)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2)


class MetricsAggregator:
    """
    Aggregates metrics from multiple collectors.

    Useful for combining metrics from parallel workers or
    multiple processing pipelines.
    """

    def __init__(self):
        """Initialize aggregator."""
        self._collectors: Dict[str, MetricsCollector] = {}

    def add_collector(self, name: str, collector: MetricsCollector) -> None:
        """Add a collector to aggregate."""
        self._collectors[name] = collector

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all collectors."""
        all_metrics = {}

        for name, collector in self._collectors.items():
            all_metrics[name] = collector.get_current_metrics().to_dict()

        # Calculate combined metrics
        combined = self._combine_metrics(all_metrics)

        return {
            'individual': all_metrics,
            'combined': combined
        }

    def _combine_metrics(self, all_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine metrics from multiple collectors."""
        if not all_metrics:
            return {}

        # Aggregate basic counts
        total_operations = sum(
            m['summary']['total_operations']
            for m in all_metrics.values()
        )

        # Aggregate latencies
        all_latencies = []
        for metrics in all_metrics.values():
            if 'latency' in metrics and 'samples' in metrics['latency']:
                # This is simplified - in real implementation would need actual values
                samples = metrics['latency']['samples']
                mean = metrics['latency']['mean_ms'] / 1000
                all_latencies.extend([mean] * samples)

        # Calculate combined latency
        combined_latency = {}
        if all_latencies:
            arr = np.array(all_latencies)
            for p in DEFAULT_PERCENTILES:
                combined_latency[f'p{p}'] = float(np.percentile(arr, p)) * 1000
            combined_latency['mean_ms'] = float(np.mean(arr)) * 1000
            combined_latency['median_ms'] = float(np.median(arr)) * 1000

        # Aggregate throughput
        total_rps = sum(
            m.get('throughput', {}).get('requests_per_second', 0)
            for m in all_metrics.values()
        )

        return {
            'total_operations': total_operations,
            'total_throughput_rps': total_rps,
            'combined_latency': combined_latency,
            'collector_count': len(all_metrics)
        }


# Utility functions for metrics analysis
def calculate_percentiles(values: List[float], percentiles: List[int] = None) -> Dict[int, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {}

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    arr = np.array(values)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def format_latency_ms(seconds: float) -> str:
    """Format latency in milliseconds."""
    return f"{seconds * 1000:.1f}ms"


def format_throughput(value: float, unit: str = "rps") -> str:
    """Format throughput value."""
    return f"{value:.2f} {unit}"


def create_metrics_summary(metrics: AggregatedMetrics) -> str:
    """Create human-readable metrics summary."""
    lines = [
        "=== LLM Performance Metrics ===",
        f"Total Operations: {metrics.total_operations:,}",
        f"Success Rate: {metrics.success_rate * 100:.1f}%",
        f"Uptime: {metrics.uptime}",
        "",
        "Latency:",
        f"  Mean: {format_latency_ms(metrics.latency.mean)}",
        f"  P50: {format_latency_ms(metrics.latency.percentiles.get(50, 0))}",
        f"  P95: {format_latency_ms(metrics.latency.percentiles.get(95, 0))}",
        f"  P99: {format_latency_ms(metrics.latency.percentiles.get(99, 0))}",
        "",
        "Throughput:",
        f"  Requests: {format_throughput(metrics.throughput.requests_per_second)}",
        f"  Tokens: {format_throughput(metrics.throughput.tokens_per_second, 'tps')}",
        "",
        "Cache Performance:",
        f"  Hit Rate: {metrics.cache.hit_rate * 100:.1f}%",
        f"  Avg Hit Time: {format_latency_ms(metrics.cache.average_hit_time)}",
        f"  Avg Miss Time: {format_latency_ms(metrics.cache.average_miss_time)}",
    ]

    return "\n".join(lines)