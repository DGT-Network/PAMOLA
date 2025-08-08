# PAMOLA.CORE Email Analysis Module Documentation

## Overview

The Email Analysis module is a specialized component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) profiling system, designed for comprehensive analysis of email address fields within datasets. It provides email validation, domain extraction, pattern recognition, and privacy risk assessment capabilities. This module is particularly valuable for understanding the quality and characteristics of contact information in resume databases, as well as assessing potential privacy risks associated with email addresses.

The module consists of two main components:
1. `email_utils.py` - Pamola Core analytical functions for validating and analyzing email addresses
2. `email.py` - Operation implementation integrating with the PAMOLA.CORE system

## Features

- **Email validation** with comprehensive format checking
- **Domain extraction** and analysis of domain frequency distribution
- **Pattern detection** in email addresses (name.surname, name_surname, etc.)
- **Top domains analysis** to understand most common email providers
- **Privacy risk assessment** based on email uniqueness and identifiability
- **Visualization generation** for domain distributions
- **Dictionary creation** for domain frequency analysis
- **Robust error handling** with detailed logging
- **Performance estimation** for large datasets
- **Progress tracking** for long-running operations
- **Seamless integration** with PAMOLA.CORE's IO, visualization, and logging systems

## Architecture

The module follows a clear separation of concerns:

```
┌──────────────────┐     ┌───────────────────┐
│  email.py        │     │  email_utils.py   │
│                  │     │                   │
│ ┌──────────────┐ │     │ ┌───────────────┐ │
│ │ EmailAnalyzer│─┼─────┼─► is_valid_email│ │
│ └──────────────┘ │     │ └───────────────┘ │
│                  │     │                   │
│ ┌──────────────┐ │     │ ┌───────────────┐ │
│ │EmailOperation│ │     │ │extract_email_domain│
│ └──────────────┘ │     │ └───────────────┘ │
│                  │     │                   │
│ ┌───────────────┐│     │ ┌────────────────┐│
│ │analyze_email_fields│  │ │detect_personal_patterns│
│ └───────────────┐│     │ └────────────────┘│
└──────────────────┘     └───────────────────┘
        │   │     │
        ▼   ▼     ▼
┌─────────┐ ┌────────┐ ┌────────────────┐
│ io.py   │ │progress│ │visualization.py│
└─────────┘ └────────┘ └────────────────┘
```

This architecture ensures:
- Pure analytical logic is separated from operation management
- Email validation and domain extraction are encapsulated and reusable
- Specialized analysis functions are properly organized
- Integration with other PAMOLA.CORE components is clean and standardized

## Key Components

### EmailAnalyzer

Static methods for analyzing email fields, validating emails, and creating domain dictionaries.

```python
from pamola_core.profiling.analyzers.email import EmailAnalyzer

# Analyze an email field
result = EmailAnalyzer.analyze(
    df=dataframe,
    field_name="email",
    top_n=20
)

# Create a domain dictionary
domain_dict = EmailAnalyzer.create_domain_dictionary(
    df=dataframe,
    field_name="email",
    min_count=5
)

# Estimate resources needed for analysis
resource_estimate = EmailAnalyzer.estimate_resources(
    df=dataframe,
    field_name="email"
)
```

### EmailOperation

Implementation of the operation interface for the PAMOLA.CORE system, handling task execution, artifact generation, and integration.

```python
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create a data source
data_source = DataSource(dataframes={"main": dataframe})

# Create and execute operation
operation = EmailOperation(
    field_name="email",
    top_n=20,
    min_frequency=2
)
result = operation.execute(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter,
    analyze_privacy_risk=True
)
```

### analyze_email_fields

Helper function for analyzing multiple email fields in one operation.

```python
from pamola_core.profiling.analyzers.email import analyze_email_fields

# Analyze multiple email fields
results = analyze_email_fields(
    data_source=data_source,
    task_dir=Path("/path/to/task"),
    reporter=reporter,
    email_fields=["email", "alternate_email"]
)
```

## Function Reference

### EmailAnalyzer.analyze

```python
@staticmethod
def analyze(df: pd.DataFrame,
            field_name: str,
            top_n: int = 20,
            **kwargs) -> Dict[str, Any]:
```

**Parameters:**
- `df` (required): DataFrame containing the data to analyze
- `field_name` (required): Name of the field to analyze
- `top_n` (default: 20): Number of top domains to include in the results
- `**kwargs`: Additional parameters for the analysis

**Returns:**
- Dictionary with analysis results including:
  - Basic statistics (total rows, null count, null percentage)
  - Valid email counts and percentages
  - Invalid email counts and percentages
  - Unique domains count
  - Top domains dictionary (domain -> count)
  - Personal patterns analysis (name.surname, name_underscore_surname, etc.)

### EmailOperation.execute

```python
def execute(self,
            data_source: DataSource,
            task_dir: Path,
            reporter: Any,
            progress_tracker: Optional[ProgressTracker] = None,
            **kwargs) -> OperationResult:
```

**Parameters:**
- `data_source` (required): Source of data for the operation
- `task_dir` (required): Directory where task artifacts should be saved
- `reporter` (required): Reporter object for tracking progress and artifacts
- `progress_tracker` (optional): Progress tracker for the operation
- `**kwargs`: Additional parameters for the operation:
  - `generate_visualization` (default: True): Whether to generate visualizations
  - `include_timestamp` (default: True): Whether to include timestamps in filenames
  - `profile_type` (default: 'email'): Type of profiling for organizing artifacts
  - `analyze_privacy_risk` (default: True): Whether to analyze privacy risks

**Returns:**
- `OperationResult` object containing:
  - Status of operation
  - List of generated artifacts
  - Metrics and statistics
  - Error information (if any)

### analyze_email_fields

```python
def analyze_email_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        email_fields: List[str] = None,
        **kwargs) -> Dict[str, OperationResult]:
```

**Parameters:**
- `data_source` (required): Source of data for the operations
- `task_dir` (required): Directory where task artifacts should be saved
- `reporter` (required): Reporter object for tracking progress and artifacts
- `email_fields` (optional): List of email fields to analyze. If None, tries to detect fields with "email" in their name
- `**kwargs`: Additional parameters for the operations:
  - `top_n` (default: 20): Number of top domains to include in results
  - `min_frequency` (default: 1): Minimum frequency for inclusion in domain dictionary
  - `generate_visualization` (default: True): Whether to generate visualization
  - `include_timestamp` (default: True): Whether to include timestamps in filenames
  - `profile_type` (default: 'email'): Type of profiling for organizing artifacts
  - `analyze_privacy_risk` (default: True): Whether to analyze privacy risks

**Returns:**
- Dictionary mapping field names to their operation results

## Privacy Risk Assessment

One of the key features of the Email Analysis module is privacy risk assessment. The `_assess_privacy_risk` method evaluates how identifiable individuals might be based on their email addresses:

```python
def _assess_privacy_risk(self, df: pd.DataFrame, field_name: str) -> Optional[Dict[str, Any]]:
```

This assessment includes:
- Calculation of the uniqueness ratio (unique emails / total valid emails)
- Identification of risk level based on uniqueness (Very High, High, Medium, Low)
- Analysis of singleton emails (appearing only once in the dataset)
- Identification of most frequent email patterns for potential exclusion

An email field with high uniqueness can pose a re-identification risk if not properly anonymized, as email addresses are often directly linked to individuals' identities.

## Generated Artifacts

The module generates the following artifacts:

1. **JSON Analysis Results** (output directory)
   - `{field_name}_stats.json`: Statistical analysis of the email field
   - Contains counts, valid/invalid percentages, domain statistics, etc.
   - `{field_name}_domains_dictionary.json`: Detailed domain frequency data
   - `{field_name}_privacy_risk.json`: Privacy risk assessment (when enabled)

2. **CSV Dictionaries** (dictionaries directory)
   - `{field_name}_domains_dictionary.csv`: Frequency dictionary of email domains
   - Includes domain, count, and percentage columns

3. **Visualizations** (visualizations directory)
   - `{field_name}_domains_distribution.png`: Bar chart showing distribution of top email domains

## Usage Examples

### Basic Email Field Analysis

```python
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Load data
df = pd.read_csv("contact_data.csv")

# Create data source
data_source = DataSource(dataframes={"main": df})

# Set up task directory
task_dir = Path("./analysis_results")


# Create a simple reporter
class SimpleReporter:
    def add_operation(self, *args, **kwargs):
        print(f"Operation: {args}")

    def add_artifact(self, *args, **kwargs):
        print(f"Artifact: {args}")


reporter = SimpleReporter()

# Execute analysis on email field
operation = EmailOperation(
    field_name="email",
    top_n=15,
    min_frequency=2
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    generate_visualization=True,
    analyze_privacy_risk=True
)

# Check results
if result.status == OperationStatus.SUCCESS:
    print("Success! Artifacts:")
    for artifact in result.artifacts:
        print(f"- {artifact.artifact_type}: {artifact.path}")

    # Print some key metrics
    print(f"\nTotal emails: {result.metrics['total_records']}")
    print(f"Valid emails: {result.metrics['valid_count']} ({result.metrics['valid_percentage']}%)")
    print(f"Unique domains: {result.metrics['unique_domains']}")
else:
    print(f"Error: {result.error_message}")
```

### Analysis with Privacy Risk Assessment

```python
from pathlib import Path
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create and execute operation with focus on privacy
operation = EmailOperation(
    field_name="email",
    top_n=20
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    analyze_privacy_risk=True
)

# Check privacy risk metrics
privacy_artifacts = [a for a in result.artifacts if "privacy_risk" in str(a.path)]
if privacy_artifacts:
    print(f"Privacy risk assessment saved to: {privacy_artifacts[0].path}")
    # You could load and examine the JSON data if needed

    # Example of using the privacy metrics directly from the operation
    if "uniqueness_ratio" in result.metrics:
        uniqueness = result.metrics["uniqueness_ratio"]
        print(f"Email uniqueness ratio: {uniqueness:.2f}")

        if uniqueness > 0.9:
            print("HIGH PRIVACY RISK: Emails are highly unique and could pose re-identification risk")
        elif uniqueness > 0.7:
            print("MODERATE PRIVACY RISK: Significant number of unique emails")
        else:
            print("LOWER PRIVACY RISK: Many emails are shared across records")
```

### Multiple Email Fields Analysis

```python
from pamola_core.profiling.analyzers.email import analyze_email_fields

# Define email fields to analyze
email_fields = ["primary_email", "secondary_email", "work_email"]

# Analyze multiple fields
results = analyze_email_fields(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    email_fields=email_fields,
    top_n=10,
    analyze_privacy_risk=True
)

# Process results
for field_name, result in results.items():
    if result.status == OperationStatus.SUCCESS:
        print(f"{field_name}: Analysis successful")
        print(f"  Valid emails: {result.metrics.get('valid_count', 0)} ({result.metrics.get('valid_percentage', 0)}%)")
        print(f"  Unique domains: {result.metrics.get('unique_domains', 0)}")
    else:
        print(f"{field_name}: Analysis failed - {result.error_message}")
```

### Direct Use of EmailAnalyzer for Quick Insights

```python
import pandas as pd
from pamola_core.profiling.analyzers.email import EmailAnalyzer

# Load data
df = pd.read_csv("contacts.csv")

# Get quick insights for an email field
insights = EmailAnalyzer.analyze(
    df=df,
    field_name="email",
    top_n=5
)

# Print key metrics
print(f"Total records: {insights['total_rows']}")
print(f"Null emails: {insights['null_count']} ({insights['null_percentage']}%)")
print(f"Valid emails: {insights['valid_count']} ({insights['valid_percentage']}%)")
print(f"Invalid emails: {insights['invalid_count']} ({insights['invalid_percentage']}%)")
print(f"Unique domains: {insights['unique_domains']}")

# Display top domains
print("\nTop domains:")
for domain, count in insights['top_domains'].items():
    print(f"  {domain}: {count}")

# Show personal patterns
patterns = insights['personal_patterns']
print("\nEmail patterns:")
for pattern, percentage in patterns['pattern_percentages'].items():
    print(f"  {pattern}: {percentage}%")
```

## Pattern Recognition

The Email Analysis module can detect common patterns in email addresses, which is valuable for understanding how users structure their email identities. The detected patterns include:

1. **name_dot_surname**: Pattern like "john.doe@example.com"
2. **name_underscore_surname**: Pattern like "john_doe@example.com"
3. **surname_dot_name**: Pattern like "doe.john@example.com"
4. **surname_underscore_name**: Pattern like "doe_john@example.com"
5. **name_surname**: Pattern like "johndoe@example.com"
6. **surname_name**: Pattern like "doejohn@example.com"

For each pattern, the module calculates both the raw count and the percentage of valid emails that follow this pattern.

## Integration with Other PAMOLA.CORE Components

The Email Analysis module integrates with:

1. **IO System** (`pamola_core.utils.io`)
   - Uses `write_json` for saving analysis results
   - Uses `ensure_directory` for directory management
   - Uses `get_timestamped_filename` for consistent file naming

2. **Visualization System** (`pamola_core.utils.visualization`)
   - Uses `plot_email_domains` for visualizing domain distributions

3. **Progress Tracking** (`pamola_core.utils.progress`)
   - Uses `ProgressTracker` for monitoring operation progress

4. **Task System**
   - Implements the operation interface (`FieldOperation`)
   - Supports task-level reporting and artifact management
   - Returns `OperationResult` objects for consistent handling

## Performance Considerations

The module includes resource estimation to predict memory usage and processing time:

```python
resources = EmailAnalyzer.estimate_resources(df, "email")
print(f"Estimated memory: {resources['estimated_memory_mb']} MB")
print(f"Estimated processing time: {resources['estimated_processing_time_sec']} seconds")
```

Key performance techniques:
1. **Efficient validation**: Email validation is optimized for speed using regex patterns
2. **Domain extraction caching**: Common domains are cached for faster processing
3. **Progress tracking**: Operations provide real-time feedback on processing status
4. **Batch processing capability**: Can handle large datasets by processing in chunks

## Best Practices

1. **Field Selection**
   - Choose fields that actually contain email addresses
   - For tables with multiple email fields, analyze each field separately

2. **Privacy Considerations**
   - Always enable privacy risk assessment when working with real personal data
   - Consider the uniqueness ratio when planning anonymization strategies
   - Be cautious with email domains that might reveal sensitive information (e.g., company domains)

3. **Domain Analysis**
   - Use domain dictionaries to understand the composition of your email dataset
   - Watch for unexpected domains that might indicate data quality issues
   - Consider the distribution of free email providers vs. organizational domains

4. **Integration Recommendations**
   - Use `DataSource` to provide flexible data access
   - Always provide a reporter for tracking operation progress
   - Handle the returned `OperationResult` objects appropriately

## Conclusion

The PAMOLA.CORE Email Analysis module provides comprehensive capabilities for analyzing email address fields within datasets. It offers valuable insights into email validity, domain distributions, and personal patterns, while also addressing privacy concerns through risk assessment.

By leveraging this module, data professionals can better understand the quality and characteristics of email data, identify potential privacy risks, and make informed decisions about data anonymization strategies for contact information.