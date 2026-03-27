# Phone Field Profiler Module Documentation

## Overview

The Phone Field Profiler module is a specialized component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) profiling system, designed for comprehensive analysis of phone number fields within datasets. It provides phone number validation, component extraction, messenger pattern detection, and privacy risk assessment capabilities, supporting both pandas and Dask DataFrames.

The module consists of two main components:
1. `phone_utils.py` - Core analytical functions for validating and analyzing phone numbers
2. `phone.py` - Operation implementation integrating with the PAMOLA.CORE system

## Features

- **Phone number validation** with comprehensive format checking
- **Component extraction** of country codes, operator codes, and phone numbers
- **Messenger pattern detection** in phone comments (Telegram, WhatsApp, Discord, etc.)
- **Frequency dictionaries** for country codes, operator codes, and messengers
- **Top-N analysis** to identify most common codes and messengers
- **Privacy risk assessment** based on phone number uniqueness and identifiability
- **Visualization generation** for phone field distributions
- **Efficient chunked, parallel, and Dask-based processing** for large datasets
- **Robust error handling** with detailed logging and progress tracking
- **Caching support** for efficient repeated analysis
- **Seamless integration** with PAMOLA.CORE's operation framework

## Architecture

The module follows a clear separation of concerns:

```
┌──────────────────┐     ┌───────────────────┐
│  phone.py        │     │  phone_utils.py   │
│                  │     │                   │
│ ┌──────────────┐ │     │ ┌───────────────┐ │
│ │PhoneAnalyzer │─┼─────┼─► validate_phone│ │
│ └──────────────┘ │     │ └───────────────┘ │
│                  │     │                   │
│ ┌──────────────┐ │     │ ┌───────────────┐ │
│ │PhoneOperation│ │     │ │extract_components│
│ └──────────────┘ │     │ └───────────────┘ │
│                  │     │                   │
│ ┌───────────────┐│     │ ┌────────────────┐│
│ │analyze_phone_ │  │ │detect_messengers│
│ │field          │  │ └────────────────┘│
│ └───────────────┐│     │                   │
└──────────────────┘     └───────────────────┘
        │   │     │
        ▼   ▼     ▼
┌─────────┐ ┌────────┐ ┌────────────────┐
│ io.py   │ │progress│ │visualization.py│
└─────────┘ └────────┘ └────────────────┘
```

This architecture ensures:
- Pure analytical logic is separated from operation management
- Phone validation and component extraction are encapsulated and reusable
- Specialized analysis functions are properly organized
- Integration with other PAMOLA.CORE components is clean and standardized

## Key Components

### PhoneAnalyzer

Static methods for analyzing phone fields, validating phone numbers, and creating dictionaries.

```python
from pamola_core.profiling.analyzers.phone import PhoneAnalyzer

# Analyze a phone field
result = PhoneAnalyzer.analyze(
    df=dataframe,
    field_name="phone_number",
    patterns_csv="custom_patterns.csv",
    chunk_size=10000,
    use_dask=False,
    use_vectorization=False,
    parallel_processes=2
)

# Create country code dictionary
country_dict = PhoneAnalyzer.create_country_code_dictionary(
    df=dataframe,
    field_name="phone_number",
    min_count=5
)

# Create operator code dictionary
operator_dict = PhoneAnalyzer.create_operator_code_dictionary(
    df=dataframe,
    field_name="phone_number",
    min_count=5
)

# Create messenger dictionary
messenger_dict = PhoneAnalyzer.create_messenger_dictionary(
    df=dataframe,
    field_name="phone_comments",
    min_count=1
)

# Estimate resources needed for analysis
resource_estimate = PhoneAnalyzer.estimate_resources(
    df=dataframe,
    field_name="phone_number"
)
```

### PhoneOperation

Implementation of the operation interface for the PAMOLA.CORE system, handling task execution, artifact generation, and integration.

```python
from pamola_core.profiling.analyzers.phone import PhoneOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create a data source
data_source = DataSource.from_dataframe(df, "main")

# Create and execute operation
operation = PhoneOperation(
    field_name="cell_phone",
    min_frequency=1,
    patterns_csv="patterns.csv"
)

result = operation.execute(
    data_source=data_source,
    task_dir=Path("output"),
    reporter=reporter,
    track_progress=True
)
```

## Usage Examples

### Basic Phone Field Analysis

```python
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.phone import PhoneOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Load data
df = pd.read_csv("resumes.csv")

# Create data source
data_source = DataSource.from_dataframe(df, "main")

# Create operation with default settings
operation = PhoneOperation(
    field_name="cell_phone",
    min_frequency=1
)

# Define output directory
task_dir = Path("output/phone_analysis")
task_dir.mkdir(parents=True, exist_ok=True)

# Execute analysis
result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    track_progress=True
)

# Check results
if result.status == OperationStatus.SUCCESS:
    print(f"Analysis completed successfully")
    print(f"Artifacts generated: {len(result.artifacts)}")
```

### Phone Analysis with Custom Messenger Patterns

```python
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.phone import PhoneOperation

# Create custom messenger patterns CSV
patterns_df = pd.DataFrame([
    {'messenger_type': 'telegram', 'pattern': 'tg_account'},
    {'messenger_type': 'whatsapp', 'pattern': 'вацап'},
    {'messenger_type': 'discord', 'pattern': 'дискорд'}
])
patterns_csv = Path("./custom_messenger_patterns.csv")
patterns_df.to_csv(patterns_csv, index=False)

# Create and execute operation with custom patterns
operation = PhoneOperation(
    field_name="cell_phone",
    min_frequency=1,
    patterns_csv=str(patterns_csv)
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter
)

# Check messenger results
if result.status == OperationStatus.SUCCESS:
    messenger_artifacts = [a for a in result.artifacts if "messenger" in str(a.path)]
    if messenger_artifacts:
        print(f"Messenger analysis saved to: {messenger_artifacts[0].path}")

        # Load and examine messenger data
        import json
        with open([a.path for a in messenger_artifacts if a.path.endswith('.json')][0], 'r') as f:
            messenger_data = json.load(f)

        print("Messenger mentions found:")
        for messenger in messenger_data['messengers']:
            print(f"  {messenger['messenger']}: {messenger['count']} ({messenger['percentage']}%)")
```

### Phone Analysis with Dask for Large Datasets

```python
# Create operation with Dask support for large datasets
operation = PhoneOperation(
    field_name="phone_number",
    min_frequency=2,
    use_dask=True,
    npartitions=4,
    chunk_size=50000
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    track_progress=True
)
```

### Using PhoneAnalyzer Directly

```python
import pandas as pd
from pamola_core.profiling.commons.phone_utils import (
    analyze_phone_field,
    analyze_phone_field_with_dask,
    create_country_code_dictionary
)

# Load data
df = pd.read_csv("resumes.csv")

# Analyze phone field with chunking
results = analyze_phone_field(
    df=df,
    field_name="phone_number",
    chunk_size=10000,
    use_vectorization=False
)

print(f"Valid phones: {results.get('valid_count', 0)}")
print(f"Invalid phones: {results.get('invalid_count', 0)}")
print(f"Top countries: {results.get('top_countries', [])}")

# Create country code frequency dictionary
country_codes = create_country_code_dictionary(
    df=df,
    field_name="phone_number",
    min_count=5
)

print(f"Country codes found: {len(country_codes)}")
```

## Generated Artifacts

The module produces the following artifacts:

### Data Files
- `{field_name}_phone_analysis.json`: Comprehensive analysis results
- `{field_name}_country_codes.json`: Country code frequency distribution
- `{field_name}_operator_codes.json`: Operator code frequency distribution
- `{field_name}_messengers.json`: Detected messenger mentions and frequencies

### Visualizations
- `{field_name}_phone_distribution.png`: Distribution of valid/invalid phones
- `{field_name}_country_codes_top_n.png`: Top country codes visualization
- `{field_name}_operator_codes_top_n.png`: Top operator codes visualization
- `{field_name}_messengers_distribution.png`: Messenger mentions visualization

## Processing Options

### Chunk-Based Processing
For large datasets, chunk-based processing is the default:
```python
operation = PhoneOperation(
    field_name="phone",
    chunk_size=10000  # Process 10K rows at a time
)
```

### Vectorized Parallel Processing
Use joblib for parallel computation:
```python
operation = PhoneOperation(
    field_name="phone",
    use_vectorization=True,
    parallel_processes=4  # Use 4 CPU cores
)
```

### Dask-Based Distributed Processing
For very large datasets, use Dask:
```python
operation = PhoneOperation(
    field_name="phone",
    use_dask=True,
    npartitions=8  # Distribute across 8 partitions
)
```

## Performance Considerations

- For datasets with < 100K records: Default chunk processing is efficient
- For datasets with 100K - 10M records: Consider vectorization with 2-4 parallel processes
- For datasets with > 10M records: Consider Dask-based processing with appropriate partition count
- Messenger pattern detection increases processing time proportionally to comment field size
- Custom pattern CSV files allow domain-specific messenger detection

## Integration

This module integrates with the broader PAMOLA CORE framework through:

- The `DataSource` abstraction for data access
- `OperationResult` for standardized result reporting
- `ProgressTracker` for operation progress monitoring
- `TaskReporter` for logging operations and artifacts
- Visualization utility functions for creating standardized charts
- Error handling and logging through `ErrorHandler`

## Best Practices

1. Validate custom messenger pattern CSV format before analysis
2. For international datasets, ensure country/operator codes are properly extracted
3. Monitor memory usage for very large datasets; use chunking or Dask
4. Generate visualizations for stakeholder communication
5. Archive messenger analysis results separately if PII concerns exist
6. Use caching for repeated analyses on the same data