# GeneratorOperation Documentation

**Module:** `pamola_core.fake_data.base_generator_op`
**Version:** 3.0.0
**Last Updated:** 2026-03-23

## Overview

The `GeneratorOperation` class serves as the base class for all generator-based synthetic data operations in PAMOLA CORE. It provides a consistent interface for operations that use generator-based approaches (e.g., fake names, fake emails, synthetic IDs), handling consistency mechanisms, mapping stores, metrics collection, and metadata propagation. This class standardizes the lifecycle, execution model, and artifact management across all fake-data generators.

## Key Features

- Standardized operation lifecycle with validation, execution, and result handling
- Support for REPLACE and ENRICH modes for synthetic value injection
- Configurable null handling strategies and conditional processing
- Chunked processing for memory efficiency with Dask and joblib support
- Comprehensive metrics collection and visualization generation
- Deterministic mapping store support for repeatable generation
- Result caching and artifact restoration for faster re-runs
- Consistency mechanism support (PRGN, mapping, etc.)
- Integration with PAMOLA operation framework and reporter interfaces

## Architecture

### Module Position

The `GeneratorOperation` is part of the `pamola_core.fake_data` package and serves as the parent class for all concrete generator operations:

```
pamola_core/fake_data/
├── base_generator_op.py        # GeneratorOperation implementation
├── operations/
│   ├── name_op.py              # FakeNameOperation (subclass)
│   ├── email_op.py             # EmailOperation (subclass)
│   ├── phone_op.py             # FakePhoneOperation (subclass)
│   └── organization_op.py       # FakeOrganizationOperation (subclass)
├── generators/
│   ├── name.py                 # NameGenerator
│   ├── email.py                # EmailGenerator
│   ├── phone.py                # PhoneGenerator
│   └── organization.py         # OrganizationGenerator
└── commons/
    ├── base.py                 # BaseGenerator, BaseOperation interfaces
    └── mapping_store.py        # MappingStore for consistency
```

### Design Principles

- **Type Safety**: Explicit validation for inputs and parameters
- **Extensibility**: Subclasses override batch/value processing and cache parameters
- **Compatibility**: Works seamlessly with PAMOLA operation framework and reporter interfaces
- **Performance**: Efficient batching, optional parallelism, and cached validators
- **Robustness**: Best-effort artifact generation and graceful degradation

## Dependencies

| Dependency | Purpose |
|-----------|---------|
| `pamola_core.fake_data.commons.base` | BaseGenerator interface |
| `pamola_core.fake_data.commons.mapping_store` | Deterministic mapping storage |
| `pamola_core.fake_data.commons.metrics` | Metrics collection and reporting |
| `pamola_core.fake_data.commons.processing_utils` | Dask/joblib processing utilities |
| `pamola_core.utils.ops.op_base` | FieldOperation parent class |
| `pamola_core.utils.ops.op_result` | OperationResult and status |
| `pandas, numpy, dask, joblib` | Data processing and parallelization |

## Core Classes and Methods

### GeneratorOperation Constructor

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `field_name` | str | Target field for generation | Required |
| `generator` | BaseGenerator | Generator instance used to create synthetic data | Required |
| `generator_params` | Dict | Additional generator configuration parameters | {} |
| `consistency_mechanism` | str | Consistency preservation mechanism (prgn, mapping, etc.) | "prgn" |
| `id_field` | str | Unique identifier field for mapping consistency | None |
| `mapping_store_path` | str | Path to persistent mapping store | None |
| `mapping_store` | MappingStore | Pre-loaded mapping store instance | None |
| `save_mapping` | bool | Whether to persist mapping results after processing | False |
| `**kwargs` | dict | Additional arguments for FieldOperation/BaseOperation | |

### Key Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `execute` | Execute the generator operation with timing and error handling | `data_source`, `task_dir`, `reporter`, `**kwargs` | OperationResult |
| `process_batch` | Process a batch of data rows | `batch: pd.DataFrame` | pd.DataFrame |
| `process_value` | Process a single value with retry logic | `value`, `**params` | Generated value |
| `_initialize_generator` | Initialize the generator with configuration | None | None |
| `_initialize_mapping_store` | Initialize mapping store if needed | None | None |
| `_collect_metrics` | Collect metrics for the operation | `df: pd.DataFrame` | Dict |
| `_save_metrics` | Save metrics to file and generate visualizations | `metrics_data: Dict`, `task_dir: Path` | Path |

## Usage Examples

### Basic Usage: Name Generation Operation

```python
from pamola_core.fake_data.generators.name import NameGenerator
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pathlib import Path
from pamola_core.utils.task_reporting import Reporter

# Create a name generator
name_gen = NameGenerator(config={
    "language": "en",
    "gender": None,
    "format": "FL"
})

# Create a generator operation
name_op = GeneratorOperation(
    field_name="person_name",
    generator=name_gen,
    generator_params={"language": "en", "format": "FL"},
    consistency_mechanism="prgn"
)

# Execute the operation
reporter = Reporter()
result = name_op.execute(
    data_source="path/to/data.csv",
    task_dir=Path("./task_directory"),
    reporter=reporter
)

if result.success:
    print(f"Processing completed. Results: {result.data.head()}")
else:
    print(f"Error: {result.error_message}")
```

### Advanced Configuration with Mapping Store

```python
from pamola_core.fake_data.generators.email import EmailGenerator
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pamola_core.fake_data.commons.mapping_store import MappingStore
import pandas as pd
from pathlib import Path

# Create email generator with domain preferences
email_gen = EmailGenerator(config={
    "domains": ["newemail.com", "corporate.org"],
    "format": "name_surname",
    "preserve_domain_ratio": 0.3
})

# Initialize mapping store for consistency
mapping_store = MappingStore()

# Create operation with mapping
email_op = GeneratorOperation(
    field_name="email",
    generator=email_gen,
    generator_params={
        "domains": ["newemail.com", "corporate.org"],
        "format": "name_surname"
    },
    consistency_mechanism="mapping",
    mapping_store=mapping_store,
    save_mapping=True,
    mapping_store_path="./mappings/email.json"
)

# Load data and execute
df = pd.DataFrame({
    'email': ['john.doe@example.com', 'jane.smith@company.org'],
    'first_name': ['John', 'Jane']
})

reporter = Reporter()
result = email_op.execute(
    data_source=df,
    task_dir=Path("./output"),
    reporter=reporter
)

# Access generated data
print(result.data)
```

### Parallel Processing with Dask

```python
from pamola_core.fake_data.generators.phone import PhoneGenerator
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pathlib import Path
import pandas as pd

# Create phone generator
phone_gen = PhoneGenerator(config={
    "country_codes": ["us", "ca", "uk"],
    "region": "us"
})

# Create operation configured for Dask processing
phone_op = GeneratorOperation(
    field_name="phone_number",
    generator=phone_gen,
    generator_params={"country_codes": ["us", "ca", "uk"]},
    processing_method="dask",  # Use Dask for parallel processing
    dask_npartitions=4
)

# Execute with large dataset
result = phone_op.execute(
    data_source="path/to/large_dataset.csv",
    task_dir=Path("./output"),
    reporter=Reporter(),
    npartitions=4,
    dask_partition_size="100MB"
)
```

### Batch Processing with Progress Tracking

```python
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pamola_core.fake_data.generators.organization import OrganizationGenerator
from pamola_core.utils.progress import HierarchicalProgressTracker
from pathlib import Path

# Create organization generator
org_gen = OrganizationGenerator(config={
    "organization_type": "general",
    "region": "en"
})

# Create operation with batch processing
org_op = GeneratorOperation(
    field_name="organization_name",
    generator=org_gen,
    generator_params={"organization_type": "general"},
    processing_method="chunk",
    batch_size=5000
)

# Execute with progress tracking
progress_tracker = HierarchicalProgressTracker()
result = org_op.execute(
    data_source="path/to/data.csv",
    task_dir=Path("./output"),
    reporter=Reporter(),
    progress_tracker=progress_tracker,
    chunk_size=5000
)
```

## Return Values

The `execute` method returns an `OperationResult` object containing:

- **data** - Processed DataFrame with synthetic values
- **success** - Boolean indicating successful execution
- **error_message** - Error description (if any)
- **execution_time** - Time taken for execution in seconds
- **metrics_path** - Path to saved metrics JSON file
- **visualization_paths** - List of paths to generated visualization files
- **status** - Detailed status information

## Processing Methods

GeneratorOperation supports three processing methods for handling large datasets:

| Method | Description | Best For | Configuration |
|--------|-------------|----------|---------------|
| `chunk` | Sequential chunk-based processing | Medium datasets, memory constraints | `chunk_size` parameter |
| `joblib` | Parallel processing using joblib | Multi-core processing | `parallel_processes`, `chunk_size` |
| `dask` | Distributed processing using Dask | Large datasets, cluster processing | `npartitions`, `dask_partition_size` |

## Consistency Mechanisms

| Mechanism | Description | Use Case |
|-----------|-------------|----------|
| `prgn` | Pseudo-Random Number Generator for deterministic generation | Default, consistent across runs with same seed |
| `mapping` | Store-based mapping for exact value reuse | When same original values must map to same synthetic values |
| `cache` | In-memory caching of generated values | Performance optimization for repeated values |

## Limitations

1. **Memory Usage**: Large datasets with detailed metrics enabled may consume significant memory
2. **Generator Dependencies**: Quality depends on the underlying generator implementation
3. **Mapping Storage**: Mapping store can grow large with extensive datasets
4. **Retry Overhead**: Retry mechanisms improve robustness but may increase processing time
5. **Performance Trade-offs**: More detailed metrics collection impacts performance
6. **Visualization Limits**: Generated visualizations are limited to supported chart types
7. **Error Recovery**: Some error conditions may require manual intervention

## Best Practices

1. **Choose Appropriate Processing Method**: Use `chunk` for small/medium data, `joblib` for multi-core systems, `dask` for distributed processing
2. **Configure Batch Sizes**: Adjust batch size based on available memory and data characteristics
3. **Use Mapping Stores for Consistency**: Save and reuse mappings when reproducibility is important
4. **Monitor Progress**: Use progress trackers for long-running operations
5. **Handle Null Values Appropriately**: Configure null strategy based on data semantics
6. **Test with Sample Data**: Verify configuration with small dataset before processing large volumes
7. **Archive Mappings**: Save mapping store files for future reference and auditing

## Related Components

- **BaseGenerator** - Abstract interface for all data generators
- **FieldOperation** - Parent class providing field-specific functionality
- **MappingStore** - Persistent storage for value mappings
- **ProcessingUtils** - Dask/joblib integration utilities
- **OperationResult** - Standardized result container

## Summary Analysis

The `GeneratorOperation` class provides a robust, extensible framework for implementing synthetic data generation operations. By inheriting from `FieldOperation`, it gains standardized field-specific processing capabilities while adding generator-specific functionality like mapping store support, metrics collection, and multiple processing strategies. The class is designed for flexibility, allowing subclasses to customize batch processing, value generation, and cache parameters while maintaining consistency with the PAMOLA operation framework. Its integration with multiple processing backends (chunk-based, joblib, Dask) makes it suitable for datasets ranging from small to distributed workloads.
