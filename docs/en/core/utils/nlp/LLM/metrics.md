# PAMOLA.CORE LLM Metrics Module Documentation

**Module:** `pamola_core.utils.nlp.llm.metrics`  
**Version:** 1.0.0  
**Status:** Stable  
**Last Updated:** January 2025

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Core Classes](#core-classes)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Performance Considerations](#performance-considerations)

## Overview

The `metrics.py` module provides comprehensive metrics collection, aggregation, and analysis capabilities for LLM operations within the PAMOLA.CORE framework. It defines structured result containers, performance metrics, quality indicators, and utilities for tracking resource usage across all LLM interactions.

### Purpose

This module serves as the central metrics and monitoring system for LLM operations:
- Standardizes result representation across all LLM operations
- Tracks performance metrics (latency, throughput, token usage)
- Monitors cache effectiveness and resource utilization
- Provides quality metrics for text generation
- Enables time-series metric tracking and analysis
- Supports metric aggregation from multiple sources

## Key Features

### 1. **Structured Result Containers**
- `ProcessingResult` for individual operations
- `BatchResult` for batch processing
- Automatic calculation of derived metrics
- Rich metadata support

### 2. **Performance Metrics**
- Latency tracking with percentile calculations
- Throughput monitoring (requests/tokens/bytes per second)
- Token usage analytics
- Cache performance statistics

### 3. **Quality Metrics**
- Confidence score tracking
- Response type classification
- Error rate monitoring
- Response diversity analysis

### 4. **Resource Monitoring**
- Memory usage tracking
- CPU utilization metrics
- Connection and thread monitoring
- Resource peak detection

### 5. **Metric Collection & Aggregation**
- Sliding window metric collection
- Multi-source aggregation
- Time-series analysis
- Export capabilities

## Architecture

### Module Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Package                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   client.py  │  │processing.py │  │postprocessing.py   │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘   │
│         │                 │                    │                │
│         └─────────────────┴────────────────────┘                │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    metrics.py                            │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │Result Types │  │Metric Types  │  │  Collectors   │  │   │
│  │  └─────────────┘  └──────────────┘  └───────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Structure

```python
metrics.py
├── Enums
│   ├── MetricType          # Types of metrics collected
│   └── ResultStatus        # Operation status codes
├── Result Classes
│   ├── ProcessingResult    # Single operation result
│   └── BatchResult         # Batch operation result
├── Metric Classes
│   ├── LatencyMetrics      # Latency statistics
│   ├── ThroughputMetrics   # Throughput measurements
│   ├── TokenMetrics        # Token usage tracking
│   ├── CacheMetrics        # Cache performance
│   ├── QualityMetrics      # Generation quality
│   └── ResourceMetrics     # Resource utilization
├── Aggregation Classes
│   ├── AggregatedMetrics   # Combined metrics
│   ├── MetricsCollector    # Time-series collector
│   └── MetricsAggregator   # Multi-source aggregator
└── Utilities
    ├── calculate_percentiles()
    ├── format_latency_ms()
    └── create_metrics_summary()
```

## Core Classes

### Result Classes

#### ProcessingResult

```python
@dataclass
class ProcessingResult:
    """Result of single text processing operation."""
    
    text: str                           # Processed text output
    original_text: str = ""             # Original input text
    success: bool = True                # Whether processing succeeded
    status: ResultStatus = ResultStatus.SUCCESS
    from_cache: bool = False            # Whether result from cache
    error: Optional[str] = None         # Error message if failed
    retry_count: int = 0                # Number of retries
    processing_time: float = 0.0        # Total processing time
    model_time: float = 0.0             # Model inference time
    tokens_input: int = 0               # Input token count
    tokens_output: int = 0              # Output token count
    tokens_truncated: int = 0           # Truncated tokens
    confidence_score: float = 1.0       # Result confidence (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

#### BatchResult

```python
@dataclass
class BatchResult:
    """Result of batch processing operation."""
    
    results: List[ProcessingResult]     # Individual results
    batch_id: str = ""                  # Unique batch identifier
    total_items: int = 0                # Total items in batch
    successful_items: int = 0           # Successfully processed
    failed_items: int = 0               # Failed items
    cached_items: int = 0               # Items from cache
    total_time: float = 0.0             # Total batch time
    average_time_per_item: float = 0.0  # Average processing time
```

### Metric Classes

#### LatencyMetrics

```python
@dataclass
class LatencyMetrics:
    """Latency metrics for LLM operations."""
    
    mean: float = 0.0                   # Mean latency
    median: float = 0.0                 # Median latency
    std: float = 0.0                    # Standard deviation
    min: float = float('inf')           # Minimum latency
    max: float = 0.0                    # Maximum latency
    percentiles: Dict[int, float]       # Percentiles (50,75,90,95,99)
    samples: int = 0                    # Number of samples
```

#### ThroughputMetrics

```python
@dataclass
class ThroughputMetrics:
    """Throughput metrics for LLM operations."""
    
    requests_per_second: float = 0.0    # Average RPS
    tokens_per_second: float = 0.0      # Average TPS
    bytes_per_second: float = 0.0       # Average BPS
    peak_rps: float = 0.0               # Peak RPS
    measurement_duration: float = 0.0   # Measurement period
    total_requests: int = 0             # Total requests
```

### Collection Classes

#### MetricsCollector

```python
class MetricsCollector:
    """Collects and tracks metrics over time."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize with sliding window size."""
        
    def add_result(self, result: ProcessingResult) -> None:
        """Add processing result to metrics."""
        
    def add_batch_result(self, batch_result: BatchResult) -> None:
        """Add batch result to metrics."""
        
    def get_current_metrics(self) -> AggregatedMetrics:
        """Get current aggregated metrics."""
        
    def reset(self) -> None:
        """Reset all metrics."""
        
    def export_metrics(self, filepath: Union[str, Path]) -> None:
        """Export metrics to JSON file."""
```

## API Reference

### Processing Results

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `ProcessingResult.to_dict()` | Convert to dictionary | None | `Dict[str, Any]` |
| `ProcessingResult.total_tokens` | Get total tokens | None | `int` |
| `ProcessingResult.overhead_time` | Get non-model time | None | `float` |
| `BatchResult.success_rate` | Calculate success rate | None | `float` |
| `BatchResult.cache_hit_rate` | Calculate cache hit rate | None | `float` |

### Metric Calculations

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `LatencyMetrics.from_values()` | Create from latency list | `values: List[float]`<br>`percentiles: List[int]` | `LatencyMetrics` |
| `TokenMetrics.update()` | Update with new result | `result: ProcessingResult` | None |
| `CacheMetrics.update_hit()` | Update for cache hit | `response_time: float` | None |
| `CacheMetrics.update_miss()` | Update for cache miss | `response_time: float` | None |

### Metric Collection

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `MetricsCollector.add_result()` | Add single result | `result: ProcessingResult` | None |
| `MetricsCollector.add_batch_result()` | Add batch result | `batch_result: BatchResult` | None |
| `MetricsCollector.get_current_metrics()` | Get aggregated metrics | None | `AggregatedMetrics` |
| `MetricsCollector.export_metrics()` | Export to JSON | `filepath: Union[str, Path]` | None |

### Utility Functions

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `calculate_percentiles()` | Calculate percentiles | `values: List[float]`<br>`percentiles: List[int]` | `Dict[int, float]` |
| `format_latency_ms()` | Format latency | `seconds: float` | `str` |
| `create_metrics_summary()` | Create summary | `metrics: AggregatedMetrics` | `str` |

## Usage Examples

### Basic Result Tracking

```python
from pamola_core.utils.nlp.llm.metrics import (
    ProcessingResult, ResultStatus, MetricsCollector
)
import time

# Create a processing result
start_time = time.time()
result = ProcessingResult(
    text="Generated response text",
    original_text="Input prompt",
    success=True,
    status=ResultStatus.SUCCESS,
    from_cache=False,
    processing_time=time.time() - start_time,
    model_time=0.8,
    tokens_input=50,
    tokens_output=100,
    confidence_score=0.95
)

# Add to metrics collector
collector = MetricsCollector()
collector.add_result(result)

# Get current metrics
metrics = collector.get_current_metrics()
print(f"Average latency: {metrics.latency.mean * 1000:.1f}ms")
print(f"Token efficiency: {metrics.tokens.token_efficiency:.2f}")
```

### Batch Processing Metrics

```python
from pamola_core.utils.nlp.llm.metrics import BatchResult, ProcessingResult
import uuid

# Process batch of texts
results = []
for i, text in enumerate(texts):
    result = ProcessingResult(
        text=f"Processed: {text}",
        original_text=text,
        success=i % 10 != 0,  # 90% success rate
        from_cache=i % 5 == 0,  # 20% cache hits
        processing_time=0.1 + (i % 3) * 0.05,
        tokens_input=len(text.split()),
        tokens_output=len(text.split()) * 2
    )
    results.append(result)

# Create batch result
batch_result = BatchResult(
    results=results,
    batch_id=str(uuid.uuid4()),
    total_time=sum(r.processing_time for r in results)
)

# Add to collector
collector.add_batch_result(batch_result)

# Check batch metrics
print(f"Batch success rate: {batch_result.success_rate * 100:.1f}%")
print(f"Cache hit rate: {batch_result.cache_hit_rate * 100:.1f}%")
print(f"Average time per item: {batch_result.average_time_per_item:.3f}s")
```

### Latency Analysis

```python
from pamola_core.utils.nlp.llm.metrics import LatencyMetrics

# Collect latency measurements
latencies = [0.1, 0.15, 0.12, 0.18, 0.11, 0.25, 0.13, 0.14, 0.16, 0.19]

# Calculate latency metrics
latency_metrics = LatencyMetrics.from_values(
    latencies,
    percentiles=[50, 75, 90, 95, 99]
)

# Display metrics
print(f"Mean latency: {latency_metrics.mean * 1000:.1f}ms")
print(f"P50 latency: {latency_metrics.percentiles[50] * 1000:.1f}ms")
print(f"P95 latency: {latency_metrics.percentiles[95] * 1000:.1f}ms")
print(f"Max latency: {latency_metrics.max * 1000:.1f}ms")
```

### Cache Performance Tracking

```python
from pamola_core.utils.nlp.llm.metrics import CacheMetrics

# Initialize cache metrics
cache_metrics = CacheMetrics()

# Simulate cache operations
for i in range(100):
    if i % 3 == 0:  # 33% cache hits
        cache_metrics.update_hit(response_time=0.001)
    else:
        cache_metrics.update_miss(response_time=0.1)

# Check performance
print(f"Cache hit rate: {cache_metrics.hit_rate * 100:.1f}%")
print(f"Average hit time: {cache_metrics.average_hit_time * 1000:.1f}ms")
print(f"Average miss time: {cache_metrics.average_miss_time * 1000:.1f}ms")
print(f"Performance gain: {cache_metrics.average_miss_time / cache_metrics.average_hit_time:.1f}x")
```

### Multi-Source Aggregation

```python
from pamola_core.utils.nlp.llm.metrics import MetricsCollector, MetricsAggregator

# Create multiple collectors (e.g., for parallel workers)
collectors = {}
for worker_id in range(4):
    collector = MetricsCollector()
    # Simulate processing
    for _ in range(100):
        result = ProcessingResult(
            text=f"Worker {worker_id} output",
            processing_time=0.1 + worker_id * 0.01,
            tokens_input=50,
            tokens_output=100
        )
        collector.add_result(result)
    collectors[f"worker_{worker_id}"] = collector

# Aggregate metrics
aggregator = MetricsAggregator()
for name, collector in collectors.items():
    aggregator.add_collector(name, collector)

# Get combined metrics
combined = aggregator.get_aggregated_metrics()
print(f"Total operations: {combined['combined']['total_operations']}")
print(f"Combined throughput: {combined['combined']['total_throughput_rps']:.2f} RPS")
```

### Metric Export and Reporting

```python
from pamola_core.utils.nlp.llm.metrics import MetricsCollector, create_metrics_summary
from pathlib import Path

# Collect metrics over time
collector = MetricsCollector(window_size=1000)

# Process requests...
for i in range(500):
    result = ProcessingResult(
        text="Response",
        processing_time=0.1 + (i % 10) * 0.01,
        from_cache=i % 5 == 0,
        tokens_input=50,
        tokens_output=100,
        confidence_score=0.8 + (i % 20) * 0.01
    )
    collector.add_result(result)

# Get current metrics
metrics = collector.get_current_metrics()

# Create human-readable summary
summary = create_metrics_summary(metrics)
print(summary)

# Export to JSON
collector.export_metrics(Path("llm_metrics.json"))
```

## Best Practices

### 1. Result Creation

```python
# Always include timing information
start_time = time.time()
# ... processing ...
result = ProcessingResult(
    text=output_text,
    processing_time=time.time() - start_time,
    model_time=model_inference_time,  # Track separately
    # Include all relevant metrics
    tokens_input=input_tokens,
    tokens_output=output_tokens,
    confidence_score=confidence
)
```

### 2. Error Handling

```python
# Properly track failures
try:
    output = process_text(input_text)
    result = ProcessingResult(
        text=output,
        success=True,
        status=ResultStatus.SUCCESS
    )
except TimeoutError:
    result = ProcessingResult(
        text="",
        success=False,
        status=ResultStatus.TIMEOUT,
        error="Processing timeout"
    )
except Exception as e:
    result = ProcessingResult(
        text="",
        success=False,
        status=ResultStatus.FAILURE,
        error=str(e)
    )
```

### 3. Metric Window Management

```python
# Choose appropriate window size
# Small window for real-time monitoring
realtime_collector = MetricsCollector(window_size=100)

# Large window for historical analysis
historical_collector = MetricsCollector(window_size=10000)

# Periodic metric export
import threading

def export_metrics_periodically(collector, interval=300):
    """Export metrics every 5 minutes."""
    def export():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collector.export_metrics(f"metrics_{timestamp}.json")
    
    timer = threading.Timer(interval, export)
    timer.daemon = True
    timer.start()
```

### 4. Performance Monitoring

```python
# Set up alerts for performance degradation
def check_performance_alerts(metrics: AggregatedMetrics):
    alerts = []
    
    # Latency alerts
    if metrics.latency.p95 > 1.0:  # 1 second P95
        alerts.append("High P95 latency detected")
    
    # Error rate alerts
    if metrics.error_rate > 0.05:  # 5% errors
        alerts.append("High error rate detected")
    
    # Cache performance
    if metrics.cache.hit_rate < 0.2:  # Low cache hits
        alerts.append("Low cache hit rate")
    
    return alerts
```

## Performance Considerations

### Memory Management

1. **Window Size**: Choose window size based on monitoring needs
   - Real-time: 100-1000 samples
   - Historical: 1000-10000 samples
   - Memory usage: ~1KB per result

2. **Result Storage**: Only store essential data in results
   ```python
   # Good: Store only necessary metrics
   result = ProcessingResult(
       text=output[:1000],  # Truncate if needed
       metadata={"model": model_name}  # Minimal metadata
   )
   
   # Avoid: Storing large objects
   result.metadata["full_response"] = large_response_object
   ```

### Performance Optimization

1. **Batch Operations**: Use BatchResult for multiple items
   ```python
   # Efficient: Single batch result
   batch_result = BatchResult(results=all_results)
   collector.add_batch_result(batch_result)
   
   # Inefficient: Individual additions
   for result in all_results:
       collector.add_result(result)
   ```

2. **Metric Calculation**: Defer expensive calculations
   ```python
   # Calculate percentiles only when needed
   if detailed_report_requested:
       latency_metrics = LatencyMetrics.from_values(latencies)
   ```

### Scalability

1. **Multi-Worker Patterns**:
   ```python
   # Use separate collectors per worker
   worker_collectors = {}
   
   # Periodic aggregation
   def aggregate_worker_metrics():
       aggregator = MetricsAggregator()
       for name, collector in worker_collectors.items():
           aggregator.add_collector(name, collector)
       return aggregator.get_aggregated_metrics()
   ```

2. **Metric Persistence**: Export and rotate metrics
   ```python
   # Rotate metric files
   def rotate_metrics(collector, max_files=10):
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       collector.export_metrics(f"metrics_{timestamp}.json")
       
       # Clean old files
       metric_files = sorted(Path(".").glob("metrics_*.json"))
       for old_file in metric_files[:-max_files]:
           old_file.unlink()
   ```

## Summary

The metrics module provides a comprehensive system for tracking and analyzing LLM operation performance. Key capabilities include:

- Structured result representation with ProcessingResult and BatchResult
- Detailed performance metrics (latency, throughput, tokens)
- Cache and quality tracking
- Time-series collection with MetricsCollector
- Multi-source aggregation with MetricsAggregator
- Export and reporting utilities

By following the usage patterns and best practices outlined in this documentation, you can effectively monitor and optimize LLM operations within the PAMOLA.CORE framework.