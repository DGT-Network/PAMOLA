Looking at the current documentation, I'll update it to reflect the new version 2.1.0 with the added categorical visualization functions.

```markdown
# visualization_utils.py Documentation

## Module Overview

**Module:** `pamola_core.anonymization.commons.visualization_utils`  
**Version:** 2.1.0  
**Status:** Stable  
**Created:** Mar 2025  
**Modified:** Jun 2025  
**License:** BSD 3-Clause

## Description

The `visualization_utils.py` module provides helper utilities for creating visualizations of anonymization metrics calculated by `metric_utils.py` and `privacy_metric_utils.py`. It serves as a thin wrapper around `pamola_core.utils.visualization` with privacy-specific context, ensuring consistent visualization generation for anonymization operations.

## Key Features

- **Standardized Filename Generation**: Consistent naming for visualization files with timestamps
- **Artifact Registration**: Automated registration of visualizations with operation results and reporters
- **Large Dataset Sampling**: Intelligent sampling for visualization of large datasets
- **Comparison Data Preparation**: Utilities for before/after anonymization comparisons
- **Optimal Binning Calculation**: Automatic determination of histogram bins using statistical methods
- **Metric-Specific Visualizations**: Tailored visualization creation based on metric types
- **Categorical Distribution Comparison**: Specialized visualizations for categorical anonymization
- **Hierarchy Visualization**: Sunburst charts for generalization hierarchy visualization
- **Framework Integration**: Seamless integration with core visualization utilities

## Architecture Position

```
pamola_core/
├── utils/
│   └── visualization.py          # Core visualization functions
│
└── anonymization/
    └── commons/
        ├── metric_utils.py       # Metrics calculation
        ├── privacy_metric_utils.py   # Privacy metrics
        └── visualization_utils.py    # This module (wrapper)
            ↓
        Uses pamola_core.utils.visualization functions
        Provides privacy-specific context
```

The module acts as a bridge between anonymization operations and the core visualization framework, adding privacy-specific context and utilities while delegating actual plotting to `pamola_core.utils.visualization`.

```
┌─────────────────────┐
│ anonymization_op.py   │
└──────────┬──────────┘
           │ calls
           ▼
┌──────────────────────────┐     uses         ┌──────────────────┐
│ visualization_utils.py              ├──────────►│ visualization.py       │
└──────────────────────────┘                     └──────────────────┘
```

## Function Reference

### Core Utility Functions

#### `generate_visualization_filename`

Generates a standardized filename for visualization files.

```python
def generate_visualization_filename(
    field_name: str,
    operation_name: str,
    visualization_type: str,
    timestamp: Optional[str] = None,
    extension: str = "png"
) -> str
```

**Parameters:**
- `field_name`: Name of the field being visualized
- `operation_name`: Name of the operation creating the visualization
- `visualization_type`: Type of visualization (e.g., "histogram", "distribution", "comparison")
- `timestamp`: Optional timestamp string (uses current time if None)
- `extension`: File extension (default: "png")

**Returns:**
- Standardized filename following pattern: `{field}_{operation}_{visType}_{timestamp}.{ext}`

**Example:**
```python
filename = generate_visualization_filename(
    field_name="salary",
    operation_name="generalization",
    visualization_type="histogram"
)
# Result: "salary_generalization_histogram_20250115_143052.png"
```

#### `register_visualization_artifact`

Registers a visualization artifact with the operation result and reporter.

```python
def register_visualization_artifact(
    result: Any,
    reporter: Any,
    path: Path,
    field_name: str,
    visualization_type: str,
    description: Optional[str] = None
) -> None
```

**Parameters:**
- `result`: OperationResult object to add the artifact to
- `reporter`: Reporter object (can be None)
- `path`: Path to the visualization file
- `field_name`: Name of the field being visualized
- `visualization_type`: Type of visualization
- `description`: Optional custom description

**Example:**
```python
register_visualization_artifact(
    result=operation_result,
    reporter=reporter,
    path=Path("/task_dir/salary_histogram.png"),
    field_name="salary",
    visualization_type="histogram",
    description="Salary distribution after anonymization"
)
```

#### `sample_large_dataset`

Samples large datasets to a manageable size for visualization.

```python
def sample_large_dataset(
    data: pd.Series,
    max_samples: int = 10000,
    random_state: int = 42
) -> pd.Series
```

**Parameters:**
- `data`: Original dataset
- `max_samples`: Maximum number of samples to return (default: 10000)
- `random_state`: Random seed for reproducibility

**Returns:**
- Sampled dataset if original exceeds max_samples, otherwise original

#### `prepare_comparison_data`

Prepares data for before/after comparison visualizations.

```python
def prepare_comparison_data(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    max_categories: int = 20
) -> Tuple[Dict[str, Any], str]
```

**Parameters:**
- `original_data`: Original data before anonymization
- `anonymized_data`: Data after anonymization
- `max_categories`: Maximum categories to show for categorical data

**Returns:**
- Tuple of (prepared_data, data_type) where data_type is 'numeric' or 'categorical'

#### `calculate_optimal_bins`

Calculates optimal number of bins for histograms using Sturges' rule.

```python
def calculate_optimal_bins(
    data: pd.Series,
    min_bins: int = 10,
    max_bins: int = 30
) -> int
```

**Parameters:**
- `data`: Data to calculate bins for
- `min_bins`: Minimum number of bins (default: 10)
- `max_bins`: Maximum number of bins (default: 30)

**Returns:**
- Optimal number of bins based on Sturges' rule and square root rule

### Visualization Creation Functions

#### `create_metric_visualization`

Creates a visualization for a specific metric using appropriate chart type.

```python
def create_metric_visualization(
    metric_name: str,
    metric_data: Union[Dict[str, Any], pd.Series, List],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    timestamp: Optional[str] = None
) -> Optional[Path]
```

**Parameters:**
- `metric_name`: Name of the metric (e.g., 'k_anonymity_distribution')
- `metric_data`: The metric data to visualize
- `task_dir`: Task directory where visualization will be saved
- `field_name`: Field name for the visualization
- `operation_name`: Operation name
- `timestamp`: Optional timestamp for consistency

**Returns:**
- Path to created visualization or None if failed

#### `create_comparison_visualization`

Creates a before/after comparison visualization.

```python
def create_comparison_visualization(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    task_dir: Path,
    field_name: str,
    operation_name: str,
    timestamp: Optional[str] = None
) -> Optional[Path]
```

**Parameters:**
- `original_data`: Original data
- `anonymized_data`: Anonymized data
- `task_dir`: Task directory
- `field_name`: Field name
- `operation_name`: Operation name
- `timestamp`: Optional timestamp for consistency

**Returns:**
- Path to visualization or None if failed

#### `create_distribution_visualization`

Creates a distribution visualization for metrics like k-anonymity or risk scores.

```python
def create_distribution_visualization(
    data: Union[pd.Series, Dict[str, int]],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    metric_name: str,
    timestamp: Optional[str] = None
) -> Optional[Path]
```

**Parameters:**
- `data`: Distribution data (e.g., {k_value: count} or Series of values)
- `task_dir`: Task directory
- `field_name`: Field name
- `operation_name`: Operation name
- `metric_name`: Name of the metric being visualized
- `timestamp`: Optional timestamp for consistency

**Returns:**
- Path to visualization or None if failed

### Categorical Visualization Functions (New in 2.1.0)

#### `create_category_distribution_comparison`

Creates a specialized comparison visualization for categorical distributions, showing top categories before and after anonymization with counts and optionally percentages.

```python
def create_category_distribution_comparison(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    task_dir: Path,
    field_name: str,
    operation_name: str,
    max_categories: int = 15,
    show_percentages: bool = True,
    timestamp: Optional[str] = None
) -> Optional[Path]
```

**Parameters:**
- `original_data`: Original categorical data
- `anonymized_data`: Anonymized categorical data
- `task_dir`: Task directory
- `field_name`: Field name
- `operation_name`: Operation name
- `max_categories`: Maximum number of categories to show (default: 15)
- `show_percentages`: Whether to show percentage distribution (default: True)
- `timestamp`: Optional timestamp for consistency

**Returns:**
- Path to visualization or None if failed

**Example:**
```python
# Original data has many categories
original = pd.Series(['A']*100 + ['B']*50 + ['C']*30 + ['D']*10 + ['E']*5)
# After merge_low_freq
anonymized = pd.Series(['A']*100 + ['B']*50 + ['C']*30 + ['OTHER']*15)

viz_path = create_category_distribution_comparison(
    original_data=original,
    anonymized_data=anonymized,
    task_dir=Path("/path/to/task"),
    field_name="department",
    operation_name="categorical_generalization",
    show_percentages=True
)
```

#### `create_hierarchy_sunburst`

Creates a sunburst visualization for hierarchical category structure, useful for visualizing generalization hierarchies and understanding the impact of hierarchy-based anonymization.

```python
def create_hierarchy_sunburst(
    hierarchy_data: Dict[str, Any],
    task_dir: Path,
    field_name: str,
    operation_name: str,
    max_depth: int = 3,
    max_categories: int = 50,
    timestamp: Optional[str] = None
) -> Optional[Path]
```

**Parameters:**
- `hierarchy_data`: Hierarchical structure data. Can be:
  - Nested dict: `{"parent": {"child1": count, "child2": count}}`
  - Flat dict with parent info: `{"child": {"parent": "parent_name", "count": 10}}`
- `task_dir`: Task directory
- `field_name`: Field name
- `operation_name`: Operation name
- `max_depth`: Maximum hierarchy depth to visualize (default: 3)
- `max_categories`: Maximum leaf categories to include (default: 50)
- `timestamp`: Optional timestamp for consistency

**Returns:**
- Path to visualization or None if failed

**Example:**
```python
# Nested hierarchy format
hierarchy = {
    "Location": {
        "North America": {
            "USA": {"New York": 120, "California": 150},
            "Canada": {"Ontario": 80, "Quebec": 70}
        },
        "Europe": {
            "UK": 110,
            "Germany": 130
        }
    }
}

viz_path = create_hierarchy_sunburst(
    hierarchy_data=hierarchy,
    task_dir=Path("/path/to/task"),
    field_name="location",
    operation_name="hierarchy_generalization",
    max_depth=3
)
```

### Helper Functions (Internal)

#### `_is_nested_hierarchy`

Checks if data is in nested hierarchy format.

```python
def _is_nested_hierarchy(data: Dict[str, Any]) -> bool
```

#### `_limit_hierarchy_size`

Limits the size and depth of a hierarchy for visualization.

```python
def _limit_hierarchy_size(
    hierarchy: Dict[str, Any],
    max_categories: int,
    max_depth: int,
    current_depth: int = 0
) -> Dict[str, Any]
```

#### `_count_leaves`

Counts leaf nodes in a hierarchy.

```python
def _count_leaves(hierarchy: Dict[str, Any]) -> int
```

#### `_convert_flat_to_nested_hierarchy`

Converts flat hierarchy format to nested format for sunburst.

```python
def _convert_flat_to_nested_hierarchy(
    flat_data: Dict[str, Dict[str, Any]],
    max_categories: int,
    max_depth: int
) -> Dict[str, Any]
```

## Usage Examples

### Complete Categorical Anonymization Visualization

```python
from pathlib import Path
import pandas as pd
from pamola_core.anonymization.commons import visualization_utils as viz_utils
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Setup
task_dir = Path("/path/to/task_dir")
field_name = "job_title"
operation_name = "categorical_generalization"
timestamp = "20250115_143052"

# Sample data - many specific job titles
original_data = pd.Series([
    "Software Engineer I", "Software Engineer II", "Senior Software Engineer",
    "Data Scientist", "Senior Data Scientist", "ML Engineer",
    "Product Manager", "Senior Product Manager", "VP Product",
    "Accountant", "Senior Accountant", "Accounting Manager"
] * 10)

# After hierarchy-based generalization
anonymized_data = pd.Series([
    "Engineering", "Engineering", "Engineering",
    "Data Science", "Data Science", "Data Science",
    "Product", "Product", "Product",
    "Finance", "Finance", "Finance"
] * 10)

# Create operation result
result = OperationResult(status=OperationStatus.SUCCESS)

# 1. Create category distribution comparison
dist_path = viz_utils.create_category_distribution_comparison(
    original_data=original_data,
    anonymized_data=anonymized_data,
    task_dir=task_dir,
    field_name=field_name,
    operation_name=operation_name,
    max_categories=10,
    show_percentages=True,
    timestamp=timestamp
)

# 2. Create hierarchy sunburst
hierarchy_data = {
    "Company": {
        "Technical": {
            "Engineering": {"Software Engineer I": 10, "Software Engineer II": 10},
            "Data Science": {"Data Scientist": 10, "ML Engineer": 10}
        },
        "Business": {
            "Product": {"Product Manager": 10, "VP Product": 10},
            "Finance": {"Accountant": 10, "Accounting Manager": 10}
        }
    }
}

sunburst_path = viz_utils.create_hierarchy_sunburst(
    hierarchy_data=hierarchy_data,
    task_dir=task_dir,
    field_name=field_name,
    operation_name=operation_name,
    max_depth=4,
    timestamp=timestamp
)

# 3. Register visualizations
for path, viz_type in [(dist_path, "category_distribution"), 
                       (sunburst_path, "hierarchy_sunburst")]:
    if path:
        viz_utils.register_visualization_artifact(
            result=result,
            reporter=None,
            path=path,
            field_name=field_name,
            visualization_type=viz_type
        )
```

### Integration with Categorical Generalization Operation

```python
class CategoricalGeneralizationOperation(AnonymizationOperation):
    def _generate_visualizations(self, original_data, anonymized_data, task_dir):
        """Generate visualizations specific to categorical generalization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_paths = []
        
        # Standard comparison
        comparison_path = viz_utils.create_comparison_visualization(
            original_data=original_data,
            anonymized_data=anonymized_data,
            task_dir=task_dir,
            field_name=self.field_name,
            operation_name=self.name,
            timestamp=timestamp
        )
        if comparison_path:
            viz_paths.append(("comparison", comparison_path))
        
        # Category distribution comparison
        if self.strategy in ["merge_low_freq", "frequency_based"]:
            dist_path = viz_utils.create_category_distribution_comparison(
                original_data=original_data,
                anonymized_data=anonymized_data,
                task_dir=task_dir,
                field_name=self.field_name,
                operation_name=self.name,
                max_categories=20,
                show_percentages=True,
                timestamp=timestamp
            )
            if dist_path:
                viz_paths.append(("category_distribution", dist_path))
        
        # Hierarchy sunburst
        if self.strategy == "hierarchy" and self.hierarchy_dict:
            hierarchy_data = self._build_hierarchy_structure()
            sunburst_path = viz_utils.create_hierarchy_sunburst(
                hierarchy_data=hierarchy_data,
                task_dir=task_dir,
                field_name=self.field_name,
                operation_name=self.name,
                timestamp=timestamp
            )
            if sunburst_path:
                viz_paths.append(("hierarchy", sunburst_path))
        
        return viz_paths
```

## Best Practices

1. **Consistent Timestamps**: Use the same timestamp for all visualizations in an operation to maintain consistency
2. **Sample Large Data**: Always use `sample_large_dataset()` for datasets larger than 10,000 points
3. **Error Handling**: Check return values from visualization functions as they return None on failure
4. **Artifact Registration**: Always register visualizations with both result and reporter objects
5. **Filename Standards**: Use `generate_visualization_filename()` for consistent naming
6. **Category Limits**: For categorical visualizations, limit displayed categories to maintain readability
7. **Hierarchy Depth**: Keep sunburst visualizations to 3-4 levels maximum for clarity

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `pathlib`: Path handling
- `pamola_core.utils.visualization`: Core visualization functions:
  - `create_bar_plot`
  - `create_histogram`
  - `create_sunburst_chart`
  - `create_pie_chart`
- Standard logging module

## Constants

- `DEFAULT_MAX_SAMPLES`: 10000 - Maximum samples for visualization
- `DEFAULT_MAX_CATEGORIES`: 20 - Default maximum categories to display
- `DEFAULT_HISTOGRAM_BINS`: 30 - Maximum histogram bins
- `DEFAULT_TOP_CATEGORIES_FOR_SUNBURST`: 50 - Maximum categories in sunburst

## Version History

- **2.1.0** (2025-01): Added categorical anonymization visualizations
  - Added `create_category_distribution_comparison()`
  - Added `create_hierarchy_sunburst()`
  - Enhanced support for categorical metrics
- **2.0.0** (2024-12): Refactored for integration with ops framework
  - Removed path management (using task_dir directly)
  - Focused on metric visualization only
  - Added wrappers for core visualization functions
- **1.0.0** (2024-01): Initial implementation with basic utilities
```