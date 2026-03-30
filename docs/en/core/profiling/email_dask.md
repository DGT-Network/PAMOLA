# Email Dask Analyzer Module Documentation

## Overview

The Email Dask Analyzer module is a specialized, Dask-enabled variant of the PAMOLA.CORE email profiling system, designed for distributed processing of email fields in very large datasets. While the standard `email.py` module handles datasets up to several million rows, the `email_dask.py` variant leverages Apache Dask for true parallel processing across multiple compute nodes or CPU cores.

This module provides the same email validation, domain extraction, and pattern detection capabilities as `email.py`, but optimized for distributed computing environments where data partitioning and parallel computation are critical for performance.

**Key Difference from email.py**: The `email_dask.py` module imports from `email_utils_dask.py` instead of `email_utils.py`, providing Dask-native implementations of all analysis functions.

## Architecture

The module follows the same separation of concerns as the standard email analyzer:

```
┌──────────────────────┐     ┌──────────────────────┐
│  email_dask.py       │     │  email_utils_dask.py │
│                      │     │                      │
│ ┌──────────────────┐ │     │ ┌─────────────────┐ │
│ │EmailAnalyzer     │─┼─────┼─► analyze_email_  │ │
│ │(Dask-aware)      │ │     │   field(Dask)    │ │
│ └──────────────────┘ │     │ └─────────────────┘ │
│                      │     │                     │
│ ┌──────────────────┐ │     │ ┌─────────────────┐ │
│ │EmailOperation    │ │     │ │create_domain_   │ │
│ │(with Dask mode)  │ │     │ │dictionary(Dask) │ │
│ └──────────────────┘ │     │ └─────────────────┘ │
└──────────────────────┘     └──────────────────────┘
        │     │     │
        ▼     ▼     ▼
┌────────┐ ┌────────┐ ┌─────────────┐ ┌──────────────┐
│ io.py  │ │progress│ │visualization│ │dask_utils.py │
└────────┘ └────────┘ └─────────────┘ └──────────────┘
```

The key difference from standard architecture is the `dask_utils.py` integration which handles:
- DataFrame partitioning
- Distributed computation
- Result aggregation across partitions
- Memory-efficient processing

## Key Components

### EmailAnalyzer (Dask-Aware)

Static methods for analyzing email fields with support for both pandas and Dask DataFrames.

```python
from pamola_core.profiling.analyzers.email_dask import EmailAnalyzer
from typing import Union
import pandas as pd
import dask.dataframe as dd

# Analyze with pandas DataFrame
pd_result = EmailAnalyzer.analyze(
    df=pandas_dataframe,
    field_name="email",
    top_n=20
)

# Analyze with Dask DataFrame (distributed)
dask_df = dd.from_pandas(pandas_dataframe, npartitions=8)
dask_result = EmailAnalyzer.analyze(
    df=dask_df,
    field_name="email",
    top_n=20,
    use_dask=True
)

# Create domain dictionary (works with both)
domain_dict = EmailAnalyzer.create_domain_dictionary(
    df=dask_df,
    field_name="email",
    min_count=5
)
```

### Method Signatures

#### analyze()

```python
@staticmethod
def analyze(
    df: Union[pd.DataFrame, dd.DataFrame],
    field_name: str,
    top_n: int = 20,
    use_dask: bool = False,
    use_vectorization: bool = False,
    chunk_size: int = 1000,
    parallel_processes: Optional[int] = 1,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, Any]
```

**Parameters**:
- `df`: pandas DataFrame or Dask DataFrame
- `field_name`: Email column name
- `top_n`: Number of top domains to include
- `use_dask`: Force Dask-based processing (auto-detected for dd.DataFrame)
- `use_vectorization`: Enable vectorized operations within partitions
- `chunk_size`: Rows per chunk when processing pandas DataFrames
- `parallel_processes`: CPU cores to use (when use_vectorization=True)
- `progress_tracker`: Optional progress tracking object
- `task_logger`: Optional logger instance
- `**kwargs`: Additional analysis parameters

**Returns**: Dictionary with analysis results including domain frequencies, validation metrics, and pattern statistics

#### create_domain_dictionary()

```python
@staticmethod
def create_domain_dictionary(
    df: Union[pd.DataFrame, dd.DataFrame],
    field_name: str,
    min_count: int = 1
) -> Dict[str, Any]
```

**Parameters**:
- `df`: Input DataFrame (pandas or Dask)
- `field_name`: Email column name
- `min_count`: Minimum frequency threshold for inclusion

**Returns**: Dictionary mapping domain names to frequency information

### EmailOperation (Dask-Enabled)

Operation class implementing email analysis with Dask support.

```python
from pamola_core.profiling.analyzers.email_dask import EmailOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create operation
operation = EmailOperation(
    field_name="email",
    top_n=30,
    use_dask=True,
    npartitions=16  # Dask partitions
)

# Execute with large dataset
result = operation.execute(
    data_source=data_source,
    task_dir=Path("output"),
    reporter=reporter,
    track_progress=True
)
```

## Usage Examples

### Example 1: Large Dataset with Dask

```python
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from pamola_core.profiling.analyzers.email_dask import EmailOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Load large CSV file directly into Dask
dask_df = dd.read_csv("large_email_dataset.csv")

# Create data source
data_source = DataSource(dataframes={"main": dask_df})

# Create operation with Dask settings
operation = EmailOperation(
    field_name="email",
    top_n=50,
    use_dask=True,
    npartitions=32  # 32 parallel partitions
)

# Execute analysis
result = operation.execute(
    data_source=data_source,
    task_dir=Path("output/email_analysis"),
    reporter=reporter,
    track_progress=True
)

print(f"Status: {result.status}")
print(f"Artifacts: {len(result.artifacts)}")
```

### Example 2: Processing Parquet Files

```python
import dask.dataframe as dd
from pathlib import Path
from pamola_core.profiling.analyzers.email_dask import EmailOperation

# Read from distributed Parquet files
dask_df = dd.read_parquet("s3://bucket/data/emails/*.parquet")

data_source = DataSource(dataframes={"main": dask_df})

operation = EmailOperation(
    field_name="contact_email",
    top_n=100,
    use_dask=True,
    npartitions=64  # Scale to data size
)

result = operation.execute(
    data_source=data_source,
    task_dir=Path("output"),
    reporter=reporter
)
```

### Example 3: Hybrid Processing (Pandas + Dask)

```python
import pandas as pd
import dask.dataframe as dd
from pamola_core.profiling.analyzers.email_dask import EmailAnalyzer

# Load small dataset as pandas
small_df = pd.read_csv("small_emails.csv")

# Analyze with standard processing
small_result = EmailAnalyzer.analyze(
    df=small_df,
    field_name="email",
    use_dask=False
)

# Load large dataset with Dask
large_dask_df = dd.read_csv("large_emails.csv")

# Analyze with Dask processing
large_result = EmailAnalyzer.analyze(
    df=large_dask_df,
    field_name="email",
    use_dask=True,
    parallel_processes=4  # Can also use vectorization within partitions
)

# Results have same structure for easy comparison
print(f"Small dataset: {small_result['valid_count']} valid emails")
print(f"Large dataset: {large_result['valid_count']} valid emails")
```

### Example 4: Memory-Efficient Batch Processing

```python
import dask.dataframe as dd
from pamola_core.profiling.analyzers.email_dask import EmailAnalyzer

# Load with specific chunk size
dask_df = dd.read_csv(
    "huge_dataset.csv",
    blocksize="256MB"  # 256MB per partition
)

# Analyze
result = EmailAnalyzer.analyze(
    df=dask_df,
    field_name="email",
    use_dask=True,
    chunk_size=50000  # Process 50K rows per task
)

# Access results
print(f"Top domains: {result['top_n_domains']}")
```

## Distributed Processing Modes

### Mode 1: Dask DataFrame Partitions

Data is automatically partitioned by Dask across available cores/nodes:

```python
# Dask distributes computation automatically
dask_df = dd.read_csv("data.csv", blocksize="256MB")
result = EmailAnalyzer.analyze(
    df=dask_df,
    field_name="email",
    use_dask=True
)
```

**Advantages**:
- True parallel processing across CPU cores/nodes
- Automatic memory management
- Fault tolerance (in distributed settings)
- Scales to terabytes of data

**Best for**: Datasets > 1GB, distributed clusters, limited RAM

### Mode 2: Vectorized Processing within Partitions

Process each partition using vectorized operations:

```python
result = EmailAnalyzer.analyze(
    df=dask_df,
    field_name="email",
    use_dask=True,
    use_vectorization=True,
    parallel_processes=4
)
```

**Advantages**:
- Combines Dask distribution with vectorized computation
- Better CPU cache utilization
- Faster processing per partition

**Best for**: Medium-to-large datasets (1-10GB), multi-core machines

## Performance Considerations

### Partition Count

**Too few partitions** (e.g., 4):
- Underutilizes CPU cores
- Higher memory per partition
- Slower overall processing

**Too many partitions** (e.g., 1000):
- Task scheduling overhead
- Small per-partition computations
- Network communication overhead (distributed)

**Recommended**: Number of partitions = 2-4x number of CPU cores

### Chunk Size

```python
# Small chunks: More granular, slower aggregation
# 1000-5000: Good for exploratory analysis
# 10000-50000: Balanced for production
# 100000+: Fastest but higher memory per partition

analyze(df=dask_df, chunk_size=25000)
```

### Memory Management

```python
# Read from file with automatic partitioning
dask_df = dd.read_csv("huge.csv", blocksize="512MB")

# Results are computed lazily; trigger computation explicitly
result = EmailAnalyzer.analyze(df=dask_df, ...)
# Under the hood: result computation triggers dask.compute()
```

## Generated Artifacts

Same as standard email analyzer:

### Data Files
- `{field_name}_email_analysis.json`: Comprehensive analysis results
- `{field_name}_domains.json`: Domain frequency distribution
- `{field_name}_validation_summary.json`: Email validation metrics

### Visualizations
- `{field_name}_email_distribution.png`: Valid/invalid email counts
- `{field_name}_domains_top_n.png`: Top N domain visualization

## Integration with PAMOLA.CORE

The module integrates seamlessly with PAMOLA.CORE:

- **DataSource**: Supports both pandas and Dask DataFrames
- **OperationResult**: Returns standard result format
- **ProgressTracker**: Works with distributed computation
- **Caching**: Compatible with operation cache system
- **Error Handling**: Comprehensive error messages for Dask issues

## Best Practices

1. **Start Small, Scale Up**: Test with pandas first, then migrate to Dask
2. **Monitor Task Graphs**: Use Dask dashboard to visualize computation
3. **Persist Intermediate Results**: Cache large DataFrames if reusing
4. **Set Appropriate Partitions**: Match your data size and hardware
5. **Use Blocksize Smartly**: 256MB-512MB per partition is usually optimal
6. **Handle Persistence**: Save Dask results to Parquet for reuse
7. **Log Comprehensively**: Enable progress tracking for long-running jobs

## Differences from email.py

| Aspect | email.py | email_dask.py |
|--------|----------|---------------|
| **Input** | pandas only | pandas + Dask |
| **Parallelization** | Chunk-based, joblib | Dask partitions |
| **Max Dataset Size** | < 10GB | > 100GB (scales) |
| **Computation Model** | Eager | Lazy (with dask.compute) |
| **Distribution** | Single machine | Multi-machine ready |
| **Memory Model** | In-core | Out-of-core capable |
| **Fault Tolerance** | Manual checkpointing | Built-in (distributed) |

## Troubleshooting

### OOM (Out of Memory) Errors

```python
# Reduce blocksize or increase partitions
dask_df = dd.read_csv("data.csv", blocksize="128MB")  # Smaller blocks
# or
dask_df = dd.read_csv("data.csv")
dask_df = dask_df.repartition(npartitions=128)  # More partitions
```

### Slow Performance

```python
# Check partition distribution
print(dask_df.npartitions)  # Should be 2-4x CPU cores
print(dask_df.compute())  # To see actual data size

# Tune parallelization
result = EmailAnalyzer.analyze(
    df=dask_df,
    use_vectorization=True,
    parallel_processes=8
)
```

### Dask Scheduler Issues

```python
# Use threaded scheduler for I/O bound
import dask
dask.config.set(scheduler='threads')

# Use processes for CPU bound
dask.config.set(scheduler='processes')
```

## Migration from email.py

If migrating from the standard email analyzer:

```python
# Old code (email.py)
from pamola_core.profiling.analyzers.email import EmailOperation

# New code (email_dask.py) - same API!
from pamola_core.profiling.analyzers.email_dask import EmailOperation

# Code works identically, but scales better
result = operation.execute(...)
```

The API is intentionally identical, allowing seamless switching between implementations.
