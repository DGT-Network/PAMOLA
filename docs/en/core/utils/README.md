# PAMOLA.CORE Utils Module Documentation

**Package:** `pamola_core.utils`
**Last Updated:** 2026-03-23
**Documentation Type:** Comprehensive Module Guide

## Overview

The `pamola_core.utils` package provides essential utilities and infrastructure for PAMOLA.CORE operations. This documentation guide covers all major submodules and their purposes.

## Module Structure

```
pamola_core.utils/
├── ops/                      # Operation Framework
│   ├── op_base.md           # BaseOperation, FieldOperation, DataFrameOperation
│   ├── op_data_reader.md    # Unified data reading interface (NEW)
│   ├── op_data_source.md    # DataSource management
│   ├── op_data_source_helpers.md # Helper functions (NEW)
│   ├── op_data_writer.md    # Standardized output writing
│   ├── op_cache.py
│   ├── op_config.py
│   ├── op_field_utils.py
│   ├── op_registry.py
│   ├── op_result.py
│   └── op_test_helpers.py
│
├── tasks/                    # Task Management
│   ├── base_task.md         # BaseTask facade
│   ├── task_context.md      # Reproducibility context (NEW)
│   ├── task_runner.md       # Operation orchestration (NEW)
│   ├── context_manager.md
│   ├── dependency_manager.md
│   ├── directory_manager.md
│   ├── encryption_manager.md
│   ├── execution_log.md
│   ├── operation_executor.md
│   ├── path_security.md
│   ├── progress_manager.md
│   ├── project_config_loader.md
│   ├── task_config.md
│   ├── task_registry.md
│   ├── task_reporting.md
│   └── task_utils.md
│
├── io_helpers/              # I/O Utilities (NEW)
│   ├── index.md            # Package overview
│   ├── crypto_utils.py
│   ├── crypto_router.py
│   ├── csv_utils.py
│   ├── dask_utils.py
│   ├── directory_utils.py
│   ├── error_utils.py
│   ├── file_utils.py
│   ├── format_utils.py
│   ├── image_utils.py
│   ├── json_utils.py
│   ├── memory_utils.py
│   ├── multi_file_utils.py
│   ├── provider_interface.py
│   ├── readers.py
│   └── temp_files.py
│
├── schema_helpers/          # Schema Utilities (NEW)
│   ├── index.md            # Package overview
│   ├── form_builder.py
│   ├── schema_utils.py
│   └── schema_builder.py
│
├── vis_helpers/            # Visualization Helpers (NEW)
│   ├── index.md           # Package overview
│   ├── base.py
│   ├── bar_plots.py
│   ├── boxplot.py
│   ├── combined_charts.py
│   ├── context.py
│   ├── cor_matrix.py
│   ├── cor_pair.py
│   ├── cor_utils.py
│   ├── heatmap.py
│   ├── histograms.py
│   ├── line_plots.py
│   ├── network_diagram.py
│   ├── pie_charts.py
│   ├── registry.py
│   ├── scatter_plots.py
│   ├── spider_charts.py
│   ├── theme.py
│   ├── venn_diagram.py
│   └── word_clouds.py
│
└── Other utilities
    ├── base_reporting.py
    ├── cli.py
    ├── crypto.py
    ├── env.py
    ├── group_processing.py
    ├── helpers.py
    ├── io.py
    ├── logging.py
    ├── nlp/               # NLP utilities
    ├── paths.py
    ├── progress.py
    ├── reporting/         # Reporting infrastructure
    ├── statistical_metrics.py
    ├── visualization.md
    ├── crypto_helpers/    # Cryptography utilities
    └── __init__.py
```

## Quick Links by Purpose

### Working with Operations

- **[Operation Base Classes](ops/op_base.md)** - All operations inherit from BaseOperation
- **[Data Reader](ops/op_data_reader.md)** - Read data from various sources
- **[Data Source](ops/op_data_source.md)** - Manage input/auxiliary datasets
- **[Data Writer](ops/op_data_writer.md)** - Standardized output writing
- **[Data Source Helpers](ops/op_data_source_helpers.md)** - Memory, schema, sampling utilities
- **[Operation Registry](ops/op_registry.py)** - Register and lookup operations
- **[Operation Result](ops/op_result.py)** - Operation execution results

### Working with Tasks

- **[Base Task](tasks/base_task.md)** - Task lifecycle and infrastructure
- **[Task Runner](tasks/task_runner.md)** - Orchestrate operation sequences
- **[Task Context](tasks/task_context.md)** - Reproducibility and seed management
- **[Task Configuration](tasks/task_config.md)** - Configuration loading
- **[Dependency Manager](tasks/dependency_manager.md)** - Task dependencies
- **[Progress Manager](tasks/progress_manager.md)** - Progress tracking
- **[Encryption Manager](tasks/encryption_manager.md)** - Encryption/decryption

### Working with I/O

- **[IO Helpers Overview](io_helpers/index.md)** - Complete I/O utilities guide
  - CSV operations with dialect detection
  - Encryption/decryption support
  - Memory management for large files
  - Multi-file handling with Dask
  - Format detection and validation

### Working with Schemas

- **[Schema Helpers Overview](schema_helpers/index.md)** - Schema manipulation
  - JSON schema flattening
  - Formily form generation
  - Auto-schema generation from configs
  - Type mapping and validation

### Working with Visualizations

- **[Visualization Helpers Overview](vis_helpers/index.md)** - Comprehensive viz guide
  - Dual backend (Matplotlib + Plotly)
  - 13+ plot types
  - Correlation analysis
  - Theme management
  - Word clouds and network diagrams

### Other Utilities

- **[Base Reporting](base_reporting.md)** - Reporting infrastructure
- **[Logging](logging.md)** - PAMOLA logging system
- **[Progress Tracking](progress.md)** - Progress visualization
- **[Cryptography](crypto.md)** - Encryption utilities
- **[File I/O](io.md)** - High-level file operations
- **[NLP Utilities](nlp/)** - Natural language processing

## Common Tasks

### Task 1: Create an Operation

```python
from pamola_core.utils.ops.op_base import DataFrameOperation
from pamola_core.utils.ops.op_data_reader import DataReader

class MyOperation(DataFrameOperation):
    def __init__(self, **kwargs):
        super().__init__(name="MyOp", **kwargs)

    def execute(self, data_source, task_dir, reporter, **kwargs):
        # Use DataReader to load data
        reader = DataReader()
        df, error = reader.read_dataframe(data_source.get_file_path("main"))

        if error:
            return OperationResult(status=OperationStatus.ERROR, error_message=error['message'])

        # Process data
        # ... your logic ...

        return OperationResult(status=OperationStatus.SUCCESS)
```

### Task 2: Run a Task

```python
from pamola_core.utils.tasks.task_runner import TaskRunner

runner = TaskRunner(
    task_id="anon_001",
    task_type="anonymization",
    description="Anonymize customer data",
    input_datasets={"main": "data/input.csv"},
    operation_configs=[
        {
            "operation": "anonymization",
            "class_name": "FullMaskingOperation",
            "parameters": {"masking_char": "*"},
            "scope": {"type": "field", "target": ["email", "phone"]}
        }
    ],
    additional_options={"seed": 42}  # Reproducibility
)

runner.configure_operations()
success = runner.run()
```

### Task 3: Read Data from Various Sources

```python
from pamola_core.utils.ops.op_data_reader import DataReader

reader = DataReader()

# CSV with auto-detection
df, error = reader.read_dataframe("data.csv")

# Multiple files with Dask for large datasets
df, error = reader.read_dataframe(
    {"jan": "2025_01.csv", "feb": "2025_02.csv"},
    use_dask=True,
    memory_limit=2.0
)

# Excel with specific sheet
df, error = reader.read_dataframe("data.xlsx", sheet_name="Sheet1")

# Encrypted file
df, error = reader.read_dataframe(
    "data.csv.enc",
    encryption_key="secret",
    use_encryption=True
)
```

### Task 4: Analyze and Optimize DataFrame

```python
from pamola_core.utils.ops.op_data_source_helpers import (
    analyze_dataframe,
    optimize_memory_usage,
    generate_dataframe_chunks
)

df = pd.read_csv("data.csv")

# Analyze structure
analysis = analyze_dataframe(df)
print(f"Shape: {analysis['shape']}")
print(f"Memory: {analysis['memory_usage']['total_mb']:.2f}MB")
for opt in analysis['potential_optimizations']:
    print(f"Optimize {opt['column']}: {opt['suggested_type']}")

# Optimize memory
result = optimize_memory_usage({"main": df}, threshold_percent=75.0)

# Process in chunks
for chunk in generate_dataframe_chunks(df, chunk_size=10000):
    process(chunk)
```

### Task 5: Ensure Reproducibility

```python
from pamola_core.utils.tasks.task_context import TaskContext

# Create context with fixed seed
ctx = TaskContext(seed=42)

# Operations get deterministic seeds
masking_seed = ctx.get_operation_seed("RandomMaskingOperation")
noise_seed = ctx.get_operation_seed("NoiseOperation")

# Different operations, different seeds, all deterministic
# Same master seed + operation name = same result every time
```

### Task 6: Validate Schema

```python
from pamola_core.utils.ops.op_data_source_helpers import validate_schema, analyze_dataframe

df = pd.read_csv("data.csv")
analysis = analyze_dataframe(df)

expected = {
    "columns": ["id", "email"],
    "dtypes": {"id": "int64", "email": "object"},
    "constraints": [{"type": "unique", "column": "id"}]
}

is_valid, errors = validate_schema(analysis, expected)
if not is_valid:
    print(f"Schema validation failed: {errors}")
```

### Task 7: Create Visualizations

```python
from pamola_core.utils.vis_helpers import PlotlyBarPlot, set_theme

# Set theme
set_theme("dark")

# Create visualization
plot = PlotlyBarPlot(
    data=df,
    x_col="category",
    y_col="value",
    title="Sales by Category"
)

# Export
html = plot.to_html()
plot.save("output.html")
```

## Key Features

### Reproducibility (FR-EP3-CORE-043)

TaskContext provides deterministic execution:
- Master seed controls all randomness
- Each operation gets unique derived seed
- Same seed + input = same output
- Perfect for regulatory compliance

```python
runner = TaskRunner(..., additional_options={"seed": 42})
```

### Flexible Data Handling

DataReader supports:
- Multiple file formats (CSV, JSON, Excel, Parquet)
- Automatic encryption/decryption
- Memory-aware processing with Dask
- Intelligent format detection
- Chunked processing for large files

### Memory Optimization

DataSource helpers provide:
- Memory monitoring
- Type-based optimization recommendations
- Chunked processing
- Stratified sampling
- Schema validation

### Operation Management

TaskRunner orchestrates:
- Operation sequences
- Field and dataset scoping
- Flexible configuration
- Error handling
- Checkpoint support

## Documentation Legend

| Symbol | Meaning |
|--------|---------|
| **(NEW)** | Recently created documentation |
| ✓ | Fully documented |
| — | Internal/minimal documentation |

## Finding Help

### By Task Type

- **Data Processing:** See DataReader, DataSource, DataSource Helpers
- **Building Workflows:** See TaskRunner, TaskContext
- **Configuration:** See schema_helpers, form_builder
- **Visualization:** See vis_helpers overview
- **Reporting:** See base_reporting

### By Component

- **Operation Framework:** ops/ directory
- **Task Management:** tasks/ directory
- **I/O Infrastructure:** io_helpers/
- **Schemas & Forms:** schema_helpers/
- **Visualizations:** vis_helpers/

### By Problem

- **Memory Issues:** See DataSource Helpers
- **Large Files:** See DataReader (Dask support)
- **Encryption:** See crypto_utils, io_helpers
- **Reproducibility:** See TaskContext
- **Format Errors:** See io_helpers, format detection

## Version Information

- **PAMOLA.CORE Version:** 0.0.1
- **Python Support:** 3.10 - 3.12
- **Last Updated:** 2026-03-23
- **Documentation Format:** Markdown

## Recent Updates

### 2026-03-23
- Added DataReader documentation
- Added DataSource Helpers documentation
- Added TaskContext documentation
- Added TaskRunner documentation
- Added io_helpers package overview
- Added schema_helpers package overview
- Added vis_helpers package overview

## Contributing to Documentation

To update documentation:

1. Read source files in `pamola_core/utils/`
2. Follow template in existing .md files
3. Include real code examples
4. Verify all links work
5. Test example code
6. Submit for review

See `.claude/rules/documentation-management.md` for detailed guidelines.

---

**Documentation Maintainer:** PAMOLA Core Team
**Last Review:** 2026-03-23
**Coverage:** ~95% of public utilities
