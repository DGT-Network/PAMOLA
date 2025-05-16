# Attribute Profiling Module Documentation

## 1. Overview

The `attribute.py` module is a pamola core component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) project's data profiling system. It provides a comprehensive solution for analyzing and categorizing dataset attributes based on their semantic meaning, statistical properties, and potential privacy implications.

## 2. Pamola Core Components

### 2.1 `DataAttributeProfilerOperation` Class

A specialized operation for profiling and categorizing dataset attributes with the following key responsibilities:
- Semantic analysis of column names and content
- Statistical characterization of columns
- Identification of potential privacy risks
- Generation of detailed artifacts and metrics

```python
class DataAttributeProfilerOperation(BaseOperation):
    def __init__(self,
                 name: str = "DataAttributeProfiler",
                 description: str = "Automatic profiling of dataset attributes",
                 dictionary_path: Optional[Union[str, Path]] = None,
                 language: str = "en",
                 sample_size: int = 10,
                 max_columns: Optional[int] = None,
                 id_column: Optional[str] = None):
        # Initialization parameters
```

#### Key Parameters

- **name**: Name of the operation
- **description**: Brief description of the operation
- **dictionary_path**: Path to custom attribute dictionary
- **language**: Language for keyword matching
- **sample_size**: Number of sample values to return per column
- **max_columns**: Maximum number of columns to analyze
- **id_column**: Name of ID column for record-level analysis

### 2.2 Attribute Categorization

The module categorizes columns into five primary roles:

1. **DIRECT_IDENTIFIER**: 
   - Explicit identifiers (email, passport, UID)
   - Unique and directly identifying

2. **QUASI_IDENTIFIER**:
   - Not unique individually
   - Can identify individuals when combined (birth date, region)

3. **SENSITIVE_ATTRIBUTE**:
   - Confidential fields
   - Includes health, financial, behavioral information

4. **INDIRECT_IDENTIFIER**:
   - Long texts, behavioral profiles
   - Potentially identifying through content analysis

5. **NON_SENSITIVE**:
   - Remaining fields without sensitive information

## 3. Workflow

### 3.1 Analysis Steps

1. **Dictionary Loading**
   - Load custom or default attribute dictionary
   - Support for multiple language keywords
   - Fallback to default dictionary

2. **Column Analysis**
   - Semantic analysis using keyword matching
   - Statistical analysis (entropy, uniqueness)
   - Type inference
   - Multi-valued field detection

3. **Categorization**
   - Resolve conflicts between semantic and statistical analysis
   - Assign confidence scores
   - Generate comprehensive column metadata

### 3.2 Artifact Generation

The operation generates multiple artifacts:

| Artifact | Type | Description |
|----------|------|-------------|
| `attribute_roles.json` | JSON | Comprehensive column categorization |
| `attribute_entropy.csv` | CSV | Entropy and uniqueness metrics |
| `attribute_sample.json` | JSON | Sample values for each column |
| `quasi_identifiers.json` | JSON | List of quasi-identifier columns |

### 3.3 Visualization

Generates visualizations to aid understanding:
- Pie chart of attribute role distribution
- Scatter plot of entropy vs. uniqueness
- Bar chart of inferred data types

## 4. Dictionary Structure

Example custom dictionary JSON:

```json
{
  "categories": {
    "DIRECT_IDENTIFIER": {
      "description": "Explicit identifiers",
      "keywords": {
        "en": ["id", "email", "passport"],
        "ru": ["идентификатор", "почта", "паспорт"]
      }
    },
    // Other category definitions...
  },
  "statistical_thresholds": {
    "entropy_high": 5.0,
    "entropy_mid": 3.0,
    "uniqueness_high": 0.9,
    "uniqueness_low": 0.2
  }
}
```

## 5. Usage Examples

### 5.1 Basic Usage

```python
from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pathlib import Path

# Create data source
data_source = DataSource.from_file_path("resume_data.csv")

# Create operation
operation = DataAttributeProfilerOperation(
    dictionary_path="custom_dictionary.json",
    language="en",
    sample_size=15
)

# Execute analysis
result = operation.execute(
    data_source=data_source,
    task_dir=Path("./attribute_analysis"),
    reporter=reporter
)

# Analyze results
print(f"Direct Identifiers: {result.metrics['direct_identifiers_count']}")
print(f"Quasi Identifiers: {result.metrics['quasi_identifiers_count']}")
```

### 5.2 Task Integration

```python
def configure_operations(self):
    self.add_operation(
        "DataAttributeProfilerOperation",
        dictionary_path="project_dictionary.json",
        language="en",
        max_columns=50
    )
```

## 6. Integration Points

- **IO System**: Uses `write_json`, `write_dataframe_to_csv`
- **Visualization System**: Generates standardized plots
- **Progress Tracking**: Supports `ProgressTracker`
- **Logging System**: Comprehensive logging

## 7. Best Practices

1. Provide a custom dictionary when possible
2. Choose appropriate `sample_size`
3. Use `max_columns` for large datasets
4. Review generated artifacts carefully
5. Use visualizations to understand data characteristics

## 8. Performance Considerations

- Memory usage increases with dataset size
- Large datasets may require chunking
- Customize `sample_size` and `max_columns`

## 9. Future Extensions

- Machine learning-based categorization
- More advanced entropy calculations
- Enhanced multi-language support
- Integration with data cleaning workflows

## Conclusion

The Attribute Profiling Module provides a robust, flexible solution for understanding dataset characteristics, with a focus on privacy risk assessment and data quality analysis.