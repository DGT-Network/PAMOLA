# PAMOLA.CORE Data Utilities Documentation

## Module: `pamola_core/anonymization/commons/data_utils.py`

### Overview

The `data_utils.py` module provides privacy-specific data processing utilities for anonymization operations within the PAMOLA.CORE framework. It serves as a specialized layer that extends the general-purpose utilities from `pamola_core.utils.ops` modules with privacy-aware functionality.

**Version:** 1.1.0  
**Status:** Stable  
**Package:** `pamola_core.anonymization.commons`  
**License:** BSD 3-Clause

### Key Features

- **Privacy-aware null value handling** with multiple strategies (PRESERVE, EXCLUDE, ANONYMIZE, ERROR)
- **Risk-based record filtering** using k-anonymity scores and conditional criteria
- **Vulnerable record handling** with various mitigation strategies
- **Integration with profiling results** for adaptive anonymization
- **Factory functions** for creating risk-based processors
- **Privacy level processors** for consistent privacy protection
- **Statistical analysis** of risk distributions

### Architecture Position

```
pamola_core/
├── utils/
│   └── ops/
│       ├── op_data_processing.py    # General data utilities
│       └── op_field_utils.py        # Field manipulation
│
└── anonymization/
    └── commons/
        ├── data_utils.py            # THIS MODULE (Privacy layer)
        ├── validation_utils.py
        ├── metric_utils.py
        └── visualization_utils.py
```

The module acts as a thin privacy-specific layer over the framework utilities:
- Uses `op_data_processing.py` for general data operations
- Uses `op_field_utils.py` for field manipulations
- Adds privacy-specific logic for anonymization operations

### Core Functions

#### 1. `process_nulls`

Process null values with privacy-aware strategies.

```python
def process_nulls(
    series: pd.Series,
    strategy: str = "PRESERVE",
    anonymize_value: str = "SUPPRESSED"
) -> pd.Series
```

**Parameters:**
- `series`: The series containing null values to process
- `strategy`: Null handling strategy
  - `"PRESERVE"`: Keep null values as is
  - `"EXCLUDE"`: Remove records with null values
  - `"ANONYMIZE"`: Replace nulls with anonymize_value
- `anonymize_value`: Value to use when strategy is "ANONYMIZE" (default: "SUPPRESSED")

**Returns:**
- `pd.Series`: Series with processed null values

**Example:**
```python
import pandas as pd
from pamola_core.anonymization.commons.data_utils import process_nulls

# Sample data with nulls
s = pd.Series([1, 2, None, 4, None])

# Anonymize null values
result = process_nulls(s, strategy="ANONYMIZE")
# Output: [1, 2, 'SUPPRESSED', 4, 'SUPPRESSED']

# Preserve nulls
result = process_nulls(s, strategy="PRESERVE")
# Output: [1.0, 2.0, NaN, 4.0, NaN]
```

#### 2. `filter_records_conditionally`

Filter DataFrame records based on risk scores and optional conditions.

```python
def filter_records_conditionally(
    df: pd.DataFrame,
    risk_field: Optional[str] = None,
    risk_threshold: float = 5.0,
    operator: str = "ge",
    condition_field: Optional[str] = None,
    condition_values: Optional[List] = None,
    condition_operator: str = "in"
) -> Tuple[pd.DataFrame, pd.Series]
```

**Parameters:**
- `df`: DataFrame to filter
- `risk_field`: Name of the field containing risk scores (e.g., k-anonymity values)
- `risk_threshold`: Threshold value for risk filtering (default: 5.0)
- `operator`: Operator for risk comparison ("ge", "lt", "gt", "le")
- `condition_field`: Additional field for conditional filtering
- `condition_values`: Values for additional condition
- `condition_operator`: Operator for additional condition ("in", "not_in", "gt", "lt", "eq", "range")

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: (Filtered DataFrame, Boolean mask)

**Example:**
```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'k_score': [2, 10, 3, 15],
    'dept': ['IT', 'HR', 'IT', 'HR']
})

# Filter high-risk records (k < 5)
filtered_df, mask = filter_records_conditionally(
    df, 
    risk_field='k_score', 
    risk_threshold=5, 
    operator='lt'
)
# Returns: DataFrame with Alice (k=2) and Charlie (k=3)
```

#### 3. `handle_vulnerable_records`

Handle vulnerable records identified by risk assessment.

```python
def handle_vulnerable_records(
    df: pd.DataFrame,
    field_name: str,
    vulnerability_mask: pd.Series,
    strategy: str = "suppress",
    replacement_value: Optional[Any] = None
) -> pd.DataFrame
```

**Parameters:**
- `df`: DataFrame containing the data
- `field_name`: Name of the field to process
- `vulnerability_mask`: Boolean mask indicating vulnerable records
- `strategy`: Strategy for handling vulnerable records
  - `"suppress"`: Replace with "SUPPRESSED" or null
  - `"remove"`: Remove the vulnerable records entirely
  - `"mean"`: Replace with mean value (numeric fields only)
  - `"mode"`: Replace with mode value (categorical fields)
  - `"custom"`: Replace with replacement_value
- `replacement_value`: Custom value when strategy is "custom"

**Returns:**
- `pd.DataFrame`: DataFrame with handled vulnerable records

**Example:**
```python
df = pd.DataFrame({
    'salary': [50000, 60000, 55000, 65000],
    'risk': [10, 2, 3, 1]
})

# Identify vulnerable records (risk < 5)
mask = df['risk'] < 5

# Replace vulnerable salaries with mean
result = handle_vulnerable_records(
    df, 'salary', mask, strategy='mean'
)
# Vulnerable records' salaries replaced with mean of safe records
```

#### 4. `create_risk_based_processor`

Factory for creating risk-based processing functions.

```python
def create_risk_based_processor(
    strategy: str = "adaptive",
    risk_threshold: float = 5.0
) -> Callable
```

**Parameters:**
- `strategy`: Risk handling strategy
  - `"conservative"`: Suppress all vulnerable records
  - `"adaptive"`: Replace with statistical values (mean/mode)
  - `"aggressive"`: Minimal changes, use custom markers
  - `"remove"`: Remove vulnerable records entirely
- `risk_threshold`: K-anonymity threshold for identifying vulnerable records

**Returns:**
- `Callable`: Function that processes vulnerable records

**Example:**
```python
# Create an adaptive processor
processor = create_risk_based_processor("adaptive")

# Use it to process vulnerable records
vulnerability_mask = df['k_score'] < 5
processed_df = processor(df, "salary", vulnerability_mask)
```

#### 5. `create_privacy_level_processor`

Create a configuration for processing based on privacy level.

```python
def create_privacy_level_processor(
    privacy_level: str = "MEDIUM"
) -> Dict[str, Any]
```

**Parameters:**
- `privacy_level`: Target privacy level
  - `"LOW"`: Minimal privacy, maximum utility
  - `"MEDIUM"`: Balanced privacy and utility
  - `"HIGH"`: Strong privacy, reduced utility
  - `"VERY_HIGH"`: Maximum privacy, minimal utility

**Returns:**
- `Dict[str, Any]`: Configuration parameters including:
  - `k_threshold`: Minimum k-anonymity value
  - `suppression_limit`: Maximum suppression rate
  - `generalization_level`: Target generalization level
  - `risk_processor`: Callable for handling vulnerable records
  - `null_strategy`: Strategy for null handling

**Example:**
```python
# Get configuration for high privacy
privacy_cfg = create_privacy_level_processor("HIGH")

# Use the configuration
k_threshold = privacy_cfg["k_threshold"]  # 10
processor = privacy_cfg["risk_processor"]
null_strategy = privacy_cfg["null_strategy"]  # "ANONYMIZE"
```

#### 6. `apply_adaptive_anonymization`

Apply adaptive anonymization based on risk scores and privacy level.

```python
def apply_adaptive_anonymization(
    df: pd.DataFrame,
    field_name: str,
    risk_scores: pd.Series,
    privacy_level: str = "MEDIUM"
) -> pd.DataFrame
```

**Parameters:**
- `df`: DataFrame to anonymize
- `field_name`: Column name to anonymize
- `risk_scores`: Risk scores (e.g., k-anonymity) for each record
- `privacy_level`: Overall privacy level

**Returns:**
- `pd.DataFrame`: DataFrame with adaptive anonymization applied

**Example:**
```python
import pandas as pd

test_df = pd.DataFrame({"salary": [100, 200, 300, 400, 500]})
test_risk_scores = pd.Series([1, 3, 10, 50, 100])

# Apply adaptive anonymization with high privacy
anonymized = apply_adaptive_anonymization(
    test_df, "salary", test_risk_scores, "HIGH"
)
# Very high risk records (k=1) will be suppressed
# High risk records (k=3) will be replaced based on privacy config
```

#### 7. `get_risk_statistics`

Calculate statistics for risk values in a DataFrame.

```python
def get_risk_statistics(
    df: pd.DataFrame,
    risk_field: str,
    thresholds: Optional[List[float]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `df`: DataFrame containing risk values
- `risk_field`: Name of the field containing risk values
- `thresholds`: List of threshold values for distribution analysis (default: [2, 5, 10])

**Returns:**
- `Dict[str, Any]`: Statistics including min, max, mean, median, distribution across thresholds

**Example:**
```python
df = pd.DataFrame({
    'id': range(100),
    'k_anonymity': np.random.randint(1, 20, 100)
})

stats = get_risk_statistics(df, 'k_anonymity', thresholds=[2, 5, 10])
print(f"Mean k-anonymity: {stats['mean_risk']}")
print(f"Distribution: {stats['distribution']}")
print(f"Risk levels: {stats['risk_level_distribution']}")
```

#### 8. `get_privacy_recommendations`

Generate privacy recommendations based on risk statistics.

```python
def get_privacy_recommendations(
    risk_stats: Dict[str, Any]
) -> Dict[str, Any]
```

**Parameters:**
- `risk_stats`: Risk statistics from `get_risk_statistics()`

**Returns:**
- `Dict[str, Any]`: Recommendations including:
  - `suggested_privacy_level`: Recommended privacy level
  - `suggested_strategies`: List of recommended strategies
  - `reasoning`: Explanation for recommendations

**Example:**
```python
# Get risk statistics
stats = get_risk_statistics(df, 'k_anonymity')

# Get recommendations
recommendations = get_privacy_recommendations(stats)
print(f"Suggested privacy level: {recommendations['suggested_privacy_level']}")
print(f"Strategies: {recommendations['suggested_strategies']}")
print(f"Reasoning: {recommendations['reasoning']}")
```

### Constants and Configuration

The module provides several predefined constants:

```python
# Default thresholds
DEFAULT_K_THRESHOLD = 5
DEFAULT_SUPPRESSION_WARNING = 0.2
DEFAULT_COVERAGE_TARGET = 0.95

# Risk levels for k-anonymity
RISK_LEVELS = {
    "VERY_HIGH": (0, 2),
    "HIGH": (2, 5),
    "MEDIUM": (5, 10),
    "LOW": (10, float('inf'))
}

# Privacy levels configuration
PRIVACY_LEVELS = {
    "LOW": {
        "k_threshold": 2,
        "suppression_limit": 0.3,
        "generalization_level": 0.3
    },
    "MEDIUM": {
        "k_threshold": 5,
        "suppression_limit": 0.2,
        "generalization_level": 0.5
    },
    "HIGH": {
        "k_threshold": 10,
        "suppression_limit": 0.1,
        "generalization_level": 0.7
    },
    "VERY_HIGH": {
        "k_threshold": 20,
        "suppression_limit": 0.05,
        "generalization_level": 0.9
    }
}
```

### Integration with Anonymization Operations

This module is designed to be used within anonymization operations:

```python
from pamola_core.anonymization.commons.data_utils import (
    filter_records_conditionally,
    handle_vulnerable_records,
    create_privacy_level_processor
)

class MyAnonymizationOperation(AnonymizationOperation):
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Get privacy configuration
        privacy_cfg = create_privacy_level_processor(self.privacy_level)
        
        # Filter records based on risk
        if self.risk_field:
            filtered_df, mask = filter_records_conditionally(
                batch,
                risk_field=self.risk_field,
                risk_threshold=privacy_cfg["k_threshold"]
            )
            
            # Handle vulnerable records
            vulnerable_mask = batch[self.risk_field] < privacy_cfg["k_threshold"]
            batch = handle_vulnerable_records(
                batch,
                self.field_name,
                vulnerable_mask,
                strategy=privacy_cfg["vulnerable_strategy"]
            )
        
        return batch
```

### Error Handling

The module includes comprehensive error handling:

- `TypeError`: Raised when incorrect types are passed (e.g., non-Series to `process_nulls`)
- `ValueError`: Raised for invalid parameters (e.g., unknown strategies)
- Logging: Warnings and errors are logged for debugging

### Performance Considerations

- Functions are optimized for pandas operations
- Sampling is used for large datasets in risk statistics
- Copy operations are used to avoid modifying original data
- Type conversions are handled efficiently for mixed data types

### Dependencies

- `pandas`: DataFrame operations
- `numpy`: Numeric computations
- `logging`: Error and progress reporting
- `typing`: Type hints
- `pamola_core.utils.ops.op_field_utils`: Field manipulation utilities