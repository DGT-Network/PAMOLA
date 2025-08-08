# PAMOLA.CORE Metrics Package - Implementation Plan

## Phase 1: Foundation and Common Infrastructure

### 1.1 Base Module (`base.py`)
- Implement `MetricOperation` base class inheriting from `BaseOperation`
- Define abstract methods: `calculate_metric()`, `_validate_inputs()`, `_normalize_metric()`, `_get_metric_metadata()`
- Implement dual-dataset handling logic in base `execute()` method
- Add support for sampling strategies and batch processing
- **Integration Notes**: Must properly override `execute()` while maintaining all framework phases (configuration saving, progress tracking, DataWriter usage)

### 1.2 Commons Utilities
**Order of implementation is critical as all metrics depend on these:**

#### 1.2.1 `commons/validation.py`
- Implement `validate_dataset_compatibility()` for comparing datasets
- Create validators for numeric, categorical, and datetime columns
- Build on top of existing framework validation patterns
- **Key Dependency**: Used by ALL metric operations

#### 1.2.2 `commons/aggregation.py`
- Implement `create_value_dictionary()` - Python equivalent of VBA function
- Add `aggregate_column_metrics()` and `create_composite_score()`
- Implement `calculate_distribution_metrics()` for KS/KL/JS from dictionaries
- **Critical**: This enables the extended KS/KL/R² implementations

#### 1.2.3 `commons/normalize.py`
- Implement metric normalization to [0,1] range
- Add distribution normalization for comparisons
- **Note**: Essential for composite metrics and standardized reporting

## Phase 2: Calculator Layer
**These provide core computation logic, independent of operation framework:**

### 2.1 `calculators/vector_calc.py`
- Implement efficient distance calculations (Euclidean, Manhattan, Cosine, Mahalanobis)
- Add optimized pairwise distance computation
- Support for both dense and sparse matrices
- **Performance Critical**: Consider numba/cython for hot paths

### 2.2 `calculators/fidelity_calc.py`
- Implement statistical test calculations (KS, KL, JS, Chi-squared)
- Add correlation matrix comparison functions
- **Note**: These are pure computation functions, no I/O or framework dependencies

### 2.3 `calculators/privacy_calc.py`
- Implement DCR calculation with multiple distance metrics
- Add NNDR computation
- Include k-anonymity analysis functions
- **Integration**: May need to handle large distance matrices - consider chunking

### 2.4 `calculators/utility_calc.py`
- Implement model training and comparison logic
- Add cross-validation utilities
- **External Dependency**: scikit-learn integration

## Phase 3: Metric Operations Implementation

### 3.1 Fidelity Metrics (`fidelity/`)
**Implementation Order (based on complexity and dependencies):**

1. **`distribution.py`** - KS Test (Extended)
   - Implement value dictionary creation with aggregation
   - Add cumulative distribution calculation
   - **Output**: Use DataWriter for distribution comparison tables
   - **Visualization**: Distribution plots via `write_visualization()`

2. **`distribution.py`** - KL Divergence (Extended)
   - Implement epsilon smoothing
   - Support grouped aggregation
   - **Note**: Reuse value dictionary logic from KS

3. **`distribution.py`** - JS Divergence
   - Build on KL implementation
   - Add symmetric calculation

4. **`distance.py`** - Hellinger Distance
   - Implement multivariate support with KDE/histogram/GMM
   - **Visualization**: Distance heatmaps for multivariate case

5. **`statistical.py`** - Chi-squared Test
   - Handle multiple categorical columns
   - **Output**: Per-column and overall statistics

6. **`correlation.py`** - Correlation & Mahalanobis Analysis
   - Implement Frobenius norm for correlation differences
   - Add Mahalanobis distance distribution analysis
   - **Visualization**: Correlation difference matrices

### 3.2 Utility Metrics (`utility/`)

1. **`regression.py`** - R² and Regression Metrics (Extended)
   - Implement grouped R² calculation (VBA-style)
   - Add standard model-based metrics
   - **Output**: Model performance comparisons
   - **Visualization**: Scatter plots with regression lines

2. **`classification.py`** - Classification Metrics
   - Implement model training and comparison
   - **Visualization**: ROC curves, confusion matrices

### 3.3 Privacy Metrics (`privacy/`)

1. **`distance.py`** - DCR (Extended)
   - Implement with FAISS support for large-scale
   - Add risk assessment based on percentiles
   - **Output**: Risk categorization tables
   - **Visualization**: Distance distribution histograms

2. **`neighbor.py`** - NNDR
   - Implement nearest neighbor ratio calculation
   - **Visualization**: NNDR distribution plots

3. **`identity.py`** - k-anonymity and Uniqueness
   - Implement efficient groupby analysis
   - **Output**: Risk assessment reports

## Phase 4: Combined Operations

### 4.1 `operations/combined_ops.py`
- Implement `ComprehensiveQualityOperation` combining all metrics
- Add `SyntheticDataQualityIndex` (SDQI)
- Create `PrivacyUtilityTradeoffIndex` (PUTI)
- **Integration**: Must coordinate multiple metric operations
- **Output**: Comprehensive reports with multiple visualizations
- **Challenge**: Memory management when running many metrics

## Phase 5: Visualization and Reporting

### 5.1 Visualization Strategy
- All visualizations use `DataWriter.write_visualization()`
- Standard visualization types:
  - Distribution comparisons (histograms, KDE plots)
  - Correlation matrices (heatmaps)
  - Distance distributions (histograms, box plots)
  - Model performance (ROC curves, regression plots)
  - Risk assessments (bar charts, risk matrices)

### 5.2 Output Structure
```
task_dir/
├── output/
│   ├── metric_results.json      # All metric values
│   └── distribution_comparison.csv  # For KS/KL/JS
├── visualizations/
│   ├── ks_distribution_*.png
│   ├── correlation_diff_*.png
│   ├── dcr_histogram_*.png
│   └── risk_assessment_*.png
├── reports/
│   └── comprehensive_report.html
└── metrics/
    └── execution_metrics.json
```

## Implementation Considerations

### Framework Integration Points

1. **Progress Tracking**
   - Use `HierarchicalProgressTracker` for multi-metric operations
   - Create subtasks for each metric calculation
   - Update progress during expensive computations (DCR, model training)

2. **Caching Strategy**
   - Cache expensive calculations (distance matrices, model results)
   - Use operation parameters + data hash for cache keys
   - Consider caching intermediate results (e.g., normalized distributions)

3. **Memory Management**
   - Process large datasets in chunks for distance calculations
   - Use sampling for expensive metrics (DCR, MMD)
   - Release intermediate DataFrames explicitly

4. **Error Handling**
   - Validate dataset compatibility early
   - Handle edge cases (single value, empty columns)
   - Return partial results when some metrics fail

### Testing Strategy

1. **Unit Tests**
   - Test each calculator function with known inputs/outputs
   - Verify metric properties (bounds, symmetry)
   - Test edge cases extensively

2. **Integration Tests**
   - Use `MockDataSource` and `StubDataWriter`
   - Verify artifact creation and structure
   - Test metric aggregation and normalization

3. **Performance Tests**
   - Benchmark against reference implementations
   - Test scalability with increasing data sizes
   - Verify memory usage stays within bounds

### Special Considerations

1. **Extended Implementations (KS, KL, R²)**
   - These require careful handling of grouped data
   - Value dictionary creation is performance-critical
   - Consider parallel processing for group operations

2. **Visualization Quality**
   - Use consistent color schemes across all metrics
   - Ensure plots are readable and informative
   - Save high-resolution images for reports

3. **Composite Metrics**
   - Define clear weighting schemes
   - Document interpretation guidelines
   - Provide sensitivity analysis options

This implementation plan ensures proper integration with the PAMOLA.CORE Operations Framework while maintaining the flexibility and extended functionality required by the metrics package.