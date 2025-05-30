## Introduction

The `pamola_core.metrics` module in PAMOLA Core provides comprehensive measurement tools for evaluating anonymized and synthetic data quality across three key dimensions: fidelity (statistical similarity), utility (usefulness for downstream tasks), and privacy (protection against re-identification). These metrics are essential for balancing the trade-offs between data usefulness and privacy protection.

This specification outlines the architectural structure, implementation requirements, and integration points for the metrics package. The design emphasizes modularity, performance, and integration with the broader PAMOLA Core operation framework.

## Package Structure Overview

```
pamola_core/metrics/
├── __init__.py                # Public API exports
├── base.py                    # Base metric classes and interfaces
├── fidelity/                  # Statistical similarity metrics
│   ├── __init__.py
│   ├── distance.py            # Distance-based metrics (Fréchet, etc.)
│   ├── distribution.py        # Distribution comparison metrics (KS, KL)
│   ├── correlation.py         # Correlation metrics
│   └── structural.py          # Structural similarity metrics
├── utility/                   # Task usefulness metrics
│   ├── __init__.py
│   ├── regression.py          # Regression model metrics (R², MSE)
│   ├── classification.py      # Classification metrics (F1, AUROC)
│   ├── clustering.py          # Clustering quality metrics
│   └── feature.py             # Feature importance metrics
├── privacy/                   # Privacy risk metrics
│   ├── __init__.py
│   ├── distance.py            # Distance-based privacy metrics
│   ├── neighbor.py            # Nearest neighbor metrics
│   ├── identity.py            # Identity disclosure metrics
│   └── information.py         # Information loss metrics
├── operations/                # Operation implementations
│   ├── __init__.py
│   ├── fidelity_ops.py        # Fidelity metric operations
│   ├── utility_ops.py         # Utility metric operations
│   ├── privacy_ops.py         # Privacy metric operations
│   └── combined_ops.py        # Combined metric operations
├── calculators/               # Pamola Pamola Core calculation implementations
│   ├── __init__.py
│   ├── fidelity_calc.py       # Fidelity metric calculations
│   ├── utility_calc.py        # Utility metric calculations
│   ├── privacy_calc.py        # Privacy metric calculations
│   └── vector_calc.py         # Vector-based calculations
└── commons/                   # Common utilities
    ├── __init__.py
    ├── validation.py          # Input validation
    ├── aggregation.py         # Metric aggregation
    └── normalize.py           # Normalization utilities
```

## Pamola Core Metrics Module Implementation Table

Below is a comprehensive table for top-level modules in the `pamola_core.metrics` package, including descriptions and specific metrics to be implemented in each module.

| Module                           | Description                                                       | Implemented Metrics                                                                                                                                                                      |
| -------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **base.py**                      | Core interfaces and abstract classes for metrics implementation   | - `BaseMetric` (abstract class)<br>- `DataFrameMetric` (abstract class)<br>- `ColumnMetric` (abstract class)<br>- `PairwiseMetric` (abstract class)<br>- `MetricResult` (data structure) |
| **fidelity/distance.py**         | Distance-based metrics measuring similarity between distributions | - Fréchet Distance (FD)<br>- Perceptual Loss/Similarity<br>- Earth Mover's Distance (EMD)<br>- Maximum Mean Discrepancy (MMD)                                                            |
| **fidelity/distribution.py**     | Metrics comparing statistical distributions                       | - Kolmogorov-Smirnov Test (KS)<br>- Kullback-Leibler Divergence (KL)<br>- Jensen-Shannon Divergence (JS)<br>- Chi-Square Test<br>- Wasserstein Distance                                  |
| **fidelity/correlation.py**      | Metrics for correlation structure preservation                    | - Correlation Matrix Difference<br>- Correlation Matrix Analysis<br>- Feature Correlation Preservation<br>- Pairwise Correlation Metrics                                                 |
| **fidelity/structural.py**       | Metrics for structural similarity                                 | - Structural Similarity Index (SSIM)<br>- Consistency Metric<br>- Cluster Structure Preservation<br>- Principal Component Structure                                                      |
| **utility/regression.py**        | Metrics for regression task performance                           | - Coefficient of Determination (R²)<br>- Mean Squared Error (MSE)<br>- Mean Absolute Error (MAE)<br>- Private Mean Squared Error (PMSE)<br>- Root Mean Squared Error (RMSE)              |
| **utility/classification.py**    | Metrics for classification task performance                       | - Accuracy Comparison<br>- Precision Score<br>- Recall Score<br>- F1 Score<br>- AUROC/AUC (Area Under ROC Curve)<br>- Precision-Recall AUC                                               |
| **utility/clustering.py**        | Metrics for clustering quality                                    | - Silhouette Score<br>- Inverted Silhouette Score (ISS)<br>- Cluster Consistency Metric<br>- Davies-Bouldin Index<br>- Calinski-Harabasz Index                                           |
| **utility/feature.py**           | Metrics for feature importance preservation                       | - Feature Importance Correlation<br>- Feature Rank Preservation<br>- Feature Importance Analysis<br>- Feature Selection Stability                                                        |
| **privacy/distance.py**          | Distance-based privacy risk metrics                               | - Distance to Closest Record (DCR)<br>- Hausdorff Distance<br>- Mahalanobis Distance Analysis<br>- Distribution of Minimum Distances                                                     |
| **privacy/neighbor.py**          | Nearest neighbor-based privacy metrics                            | - Nearest Neighbor Accuracy<br>- Nearest Neighbor Distance Ratio (NNDR)<br>- k-Nearest Neighbor Disclosure Risk<br>- Neighbor Histogram Analysis                                         |
| **privacy/identity.py**          | Identity disclosure risk metrics                                  | - Identical Match Share<br>- Identity Disclosure Risk<br>- Attribute Disclosure Risk<br>- Uniqueness Analysis                                                                            |
| **privacy/information.py**       | Information loss and entropy metrics                              | - Information Loss<br>- Conditional Entropy<br>- Normalized Entropy Loss<br>- Entropy-based Disclosure Risk                                                                              |
| **operations/fidelity_ops.py**   | Operations for fidelity metric calculation                        | - `CalculateFidelityMetricsOperation`<br>- `DistributionComparisonOperation`<br>- `CorrelationAnalysisOperation`<br>- `StructuralSimilarityOperation`                                    |
| **operations/utility_ops.py**    | Operations for utility metric calculation                         | - `CalculateUtilityMetricsOperation`<br>- `ModelPerformanceOperation`<br>- `ClusteringQualityOperation`<br>- `FeatureImportanceOperation`                                                |
| **operations/privacy_ops.py**    | Operations for privacy risk assessment                            | - `CalculatePrivacyMetricsOperation`<br>- `ReidentificationRiskOperation`<br>- `DistancePrivacyOperation`<br>- `InformationLossOperation`                                                |
| **operations/combined_ops.py**   | Operations for comprehensive metric evaluation                    | - `DataQualityAssessmentOperation`<br>- `PrivacyUtilityTradeoffOperation`<br>- `ComprehensiveMetricOperation`<br>- `MetricDashboardOperation`                                            |
| **calculators/fidelity_calc.py** | Pamola Core calculation functions for fidelity metrics                   | - `calculate_frechet_distance`<br>- `calculate_ks_test`<br>- `calculate_kl_divergence`<br>- `calculate_correlation_similarity`<br>- `calculate_structural_similarity`                    |
| **calculators/utility_calc.py**  | Pamola Core calculation functions for utility metrics                    | - `calculate_r_squared`<br>- `calculate_classification_metrics`<br>- `calculate_clustering_metrics`<br>- `calculate_feature_importance`<br>- `evaluate_model_performance`                |
| **calculators/privacy_calc.py**  | Pamola Core calculation functions for privacy metrics                    | - `calculate_dcr`<br>- `calculate_neighbor_metrics`<br>- `calculate_identity_disclosure`<br>- `calculate_information_loss`<br>- `calculate_disclosure_risk`                              |
| **calculators/vector_calc.py**   | Vector-based calculation utilities                                | - `calculate_vector_distance`<br>- `calculate_pairwise_distances`<br>- `find_nearest_neighbors`<br>- `calculate_vector_similarity`<br>- `compute_distance_matrix`                        |
| **commons/validation.py**        | Input validation utilities                                        | - `validate_dataframes`<br>- `validate_columns`<br>- `validate_metric_parameters`<br>- `check_compatibility`<br>- `validate_operation_inputs`                                            |
| **commons/aggregation.py**       | Metric aggregation utilities                                      | - `aggregate_column_metrics`<br>- `aggregate_record_metrics`<br>- `combine_metrics`<br>- `normalize_metrics`<br>- `create_composite_score`                                               |
| **commons/normalize.py**         | Data normalization utilities                                      | - `normalize_numeric`<br>- `normalize_categorical`<br>- `normalize_distributions`<br>- `apply_scaling`<br>- `standardize_vectors`                                                        |


This table provides a clear overview of what each module in the `pamola_core.metrics` package is responsible for and what specific metrics or functions it will implement. This can serve as a reference for implementation and development tracking.

## Module Responsibilities

### 1. Base Module (`base.py`)

**Purpose**: Define pamola core interfaces and abstract classes for metrics.

**Key Components**:

- `BaseMetric` abstract class
- `DataFrameMetric` for DataFrame-wide metrics
- `ColumnMetric` for column-specific metrics
- `PairwiseMetric` for comparing two datasets
- `MetricResult` class for standardized metric results

**Implementation Requirements**:

- Define clear interfaces with type hints
- Support for metric metadata (name, range, interpretation)
- Include validation methods for inputs
- Support for both single and batch calculations

### 2. Registry Module (`registry.py`)

**Purpose**: Provide metric registration and discovery.

**Key Components**:

- `register_metric` decorator for metric registration
- `get_metric` function for metric retrieval
- `list_metrics` function for metric discovery
- Category-based organization (fidelity, utility, privacy)

**Implementation Requirements**:

- Support for metric versioning
- Metadata storage for each metric
- Categorization by metric type and application
- Parameter validation during registration

### 3. Fidelity Metrics (`fidelity/`)

**Purpose**: Measure statistical similarity between original and transformed datasets.

#### 3.1 Distance Metrics (`fidelity/distance.py`)

**Key Metrics**:

- Fréchet Distance (FD): Measure of similarity between distributions
- Perceptual Loss/Perceptual Similarity: Visual similarity based on deep features

**Implementation Requirements**:

- Support for numerical and categorical data
- Vectorized implementation for performance
- Dimensionality reduction for high-dimensional data
- Visualization options for distance metrics

#### 3.2 Distribution Metrics (`fidelity/distribution.py`)

**Key Metrics**:

- Kolmogorov-Smirnov Test (KS): Distribution similarity test
- Kullback-Leibler Divergence (KL): Information loss between distributions
- Jensen-Shannon Divergence (JS): Symmetric measure of similarity

**Implementation Requirements**:

- Support for different data types (continuous, categorical)
- Optimized implementation for large datasets
- Options for handling null values
- Visual comparison of distributions

#### 3.3 Correlation Metrics (`fidelity/correlation.py`)

**Key Metrics**:

- Correlation Analysis: Comparison of correlation matrices
- Correlation Matrix Analysis: Evaluation of correlation preservation

**Implementation Requirements**:

- Support for different correlation methods (Pearson, Spearman)
- Metrics for correlation matrix similarity
- Visualization of correlation differences
- Performance optimization for large matrices

#### 3.4 Structural Metrics (`fidelity/structural.py`)

**Key Metrics**:

- Structural Similarity Index (SSIM): Measures structural similarity
- Consistency: Evaluates consistency of structural relationships
- Clustering Structure: Compares clustering patterns

**Implementation Requirements**:

- Adaptable for different data structures
- Support for multivariate structural analysis
- Visualization of structural differences
- Configurable parameters for different applications

### 4. Utility Metrics (`utility/`)

**Purpose**: Measure usefulness of transformed data for specific tasks.

#### 4.1 Regression Metrics (`utility/regression.py`)

**Key Metrics**:

- Coefficient of Determination (R²): Explained variance
- Mean Squared Error (MSE): Average squared difference
- Mean Absolute Error (MAE): Average absolute difference
- Propensity Mean Squared Error (PMSE): Privacy-aware MSE

**Implementation Requirements**:

- Support for different regression models
- Cross-validation options
- Comparison between original and synthetic performance
- Visualization of regression performance

#### 4.2 Classification Metrics (`utility/classification.py`)

**Key Metrics**:

- Accuracy: Classification accuracy comparison
- Precision: Precision score comparison
- F1 Score: Harmonic mean of precision and recall
- AUROC: Area under ROC curve comparison

**Implementation Requirements**:

- Support for binary and multiclass problems
- Multiple classifier options
- Cross-validation strategies
- Visualization of classification performance

#### 4.3 Clustering Metrics (`utility/clustering.py`)

**Key Metrics**:

- Silhouette Score: Cluster quality comparison
- Inverted Silhouette Score (ISS): Alternative cluster quality measure
- Cluster Preservation: Similarity of cluster structures

**Implementation Requirements**:

- Support for different clustering algorithms
- Visualization of cluster comparisons
- Metrics for cluster stability
- Configurable clustering parameters

#### 4.4 Feature Metrics (`utility/feature.py`)

**Key Metrics**:

- Feature Correlation: Correlation between feature importance
- Feature Importance Analysis: Comparison of feature rankings

**Implementation Requirements**:

- Support for different feature importance methods
- Visualization of feature importance comparison
- Integration with common ML libraries
- Support for complex feature interactions

### 5. Privacy Metrics (`privacy/`)

**Purpose**: Measure risk of re-identification or information disclosure.

#### 5.1 Distance Metrics (`privacy/distance.py`)

**Key Metrics**:

- Distance to Closest Record (DCR): Minimum distance to original records
- Hausdorff Distance: Maximum distance between datasets
- Mahalanobis Distance Analysis: Distance in feature space

**Implementation Requirements**:

- Efficient distance calculation algorithms
- Support for various distance measures
- Optimizations for large datasets
- Visualization of distance distributions

#### 5.2 Neighbor Metrics (`privacy/neighbor.py`)

**Key Metrics**:

- Nearest Neighbor Accuracy: Success rate of re-identification
- Nearest Neighbor Distance Ratio (NNDR): Ratio of closest neighbor distances

**Implementation Requirements**:

- Efficient nearest neighbor algorithms
- Support for various distance metrics
- Vectorized implementation for performance
- Visualization of neighbor relationships

#### 5.3 Identity Metrics (`privacy/identity.py`)

**Key Metrics**:

- Identical Match Share: Proportion of exact matches
- Identity Disclosure Risk: Probability of identity disclosure

**Implementation Requirements**:

- Configurable matching criteria
- Support for partial matching
- Efficient implementation for large datasets
- Risk scoring and visualization

#### 5.4 Information Metrics (`privacy/information.py`)

**Key Metrics**:

- Information Loss: Measure of information lost in anonymization
- Conditional Entropy: Information uncertainty after anonymization

**Implementation Requirements**:

- Support for different information theory metrics
- Handling of categorical and numerical data
- Efficient implementation for large datasets
- Visualization of information loss

### 6. Operations (`operations/`)

**Purpose**: Implement operation classes for metric calculation following the PAMOLA operation framework.

#### 6.1 Fidelity Operations (`operations/fidelity_ops.py`)

**Key Operations**:

- `CalculateFidelityMetricsOperation`: Calculates multiple fidelity metrics
- `DistributionComparisonOperation`: Compares distributions between datasets
- `CorrelationAnalysisOperation`: Analyzes correlation preservation

**Implementation Requirements**:

- Inherit from appropriate operation base classes
- Use DataSource for input/output
- Generate visualizations as artifacts
- Return structured metrics as OperationResult

#### 6.2 Utility Operations (`operations/utility_ops.py`)

**Key Operations**:

- `CalculateUtilityMetricsOperation`: Calculates multiple utility metrics
- `ModelComparisonOperation`: Compares model performance
- `FeatureImportanceOperation`: Analyzes feature importance preservation

**Implementation Requirements**:

- Integrate with ML libraries for model training
- Support for cross-validation
- Generate model comparison visualizations
- Return standardized metrics and artifacts

#### 6.3 Privacy Operations (`operations/privacy_ops.py`)

**Key Operations**:

- `CalculatePrivacyMetricsOperation`: Calculates multiple privacy metrics
- `ReidentificationRiskOperation`: Assesses re-identification risk
- `InformationLossOperation`: Measures information loss

**Implementation Requirements**:

- Configure privacy thresholds based on requirements
- Generate risk visualizations
- Support for different privacy scenarios
- Return detailed privacy metrics

#### 6.4 Combined Operations (`operations/combined_ops.py`)

**Key Operations**:

- `DataQualityAssessmentOperation`: Comprehensive quality assessment
- `QualityTradeoffOperation`: Analyzes privacy-utility tradeoffs

**Implementation Requirements**:

- Combine metrics from multiple categories
- Generate comprehensive reports
- Visualize metric tradeoffs
- Support for metric weighting and prioritization

### 7. Calculators (`calculators/`)

**Purpose**: Implement pamola core calculation logic for metrics.

#### 7.1 Fidelity Calculators (`calculators/fidelity_calc.py`)

**Key Functions**:

- `calculate_frechet_distance`: Calculates Fréchet distance
- `calculate_ks_test`: Performs Kolmogorov-Smirnov test
- `calculate_kl_divergence`: Calculates KL divergence
- `calculate_correlation_similarity`: Measures correlation matrix similarity

**Implementation Requirements**:

- Optimized implementations for performance
- Support for different data types
- Vectorized operations where possible
- Proper handling of edge cases

#### 7.2 Utility Calculators (`calculators/utility_calc.py`)

**Key Functions**:

- `calculate_r_squared`: Calculates R² coefficient
- `calculate_classification_metrics`: Computes classification metrics
- `calculate_clustering_metrics`: Calculates clustering quality metrics
- `calculate_feature_importance_similarity`: Compares feature importance

**Implementation Requirements**:

- Integration with scikit-learn
- Support for different model types
- Efficient cross-validation implementation
- Proper handling of training/testing splits

#### 7.3 Privacy Calculators (`calculators/privacy_calc.py`)

**Key Functions**:

- `calculate_dcr`: Computes Distance to Closest Record
- `calculate_neighbor_metrics`: Calculates nearest neighbor metrics
- `calculate_identity_disclosure`: Estimates identity disclosure risk
- `calculate_information_loss`: Measures information loss

**Implementation Requirements**:

- Efficient distance calculation algorithms
- Optimized implementations for large datasets
- Configurable privacy thresholds
- Support for different distance metrics

#### 7.4 Vector Calculators (`calculators/vector_calc.py`)

**Key Functions**:

- `calculate_vector_distance`: Computes distance between vectors
- `calculate_pairwise_distances`: Calculates all pairwise distances
- `find_nearest_neighbors`: Identifies nearest neighbors
- `calculate_vector_similarity`: Measures similarity between vectors

**Implementation Requirements**:

- Support for sparse and dense vectors
- Optimized distance calculation
- Efficient nearest neighbor search
- Parallelization for large vectors

### 8. Commons (`commons/`)

**Purpose**: Provide common utilities for metric calculations.

#### 8.1 Validation (`commons/validation.py`)

**Key Functions**:

- `validate_dataframes`: Validates input DataFrames
- `validate_columns`: Ensures columns exist and have correct types
- `validate_metric_parameters`: Validates metric parameters
- `check_compatibility`: Checks compatibility between datasets

**Implementation Requirements**:

- Clear error messages
- Type checking
- Schema validation
- Performance optimization

#### 8.2 Aggregation (`commons/aggregation.py`)

**Key Functions**:

- `aggregate_column_metrics`: Aggregates metrics across columns
- `aggregate_record_metrics`: Aggregates metrics across records
- `combine_metrics`: Combines multiple metrics into composite scores
- `normalize_metrics`: Normalizes metrics to standard ranges

**Implementation Requirements**:

- Support for different aggregation methods
- Weighted aggregation options
- Handling of missing values
- Proper scaling and normalization

#### 8.3 Normalize (`commons/normalize.py`)

**Key Functions**:

- `normalize_numeric`: Normalizes numeric data
- `normalize_categorical`: Normalizes categorical data
- `normalize_distributions`: Normalizes distributions for comparison
- `apply_scaling`: Applies various scaling methods

**Implementation Requirements**:

- Support for different normalization techniques
- Handling of outliers
- Preservation of data characteristics
- Reversible normalization options

#### 8.4 Sampling (`commons/sampling.py`)

**Key Functions**:

- `stratified_sample`: Creates stratified samples
- `bootstrap_sample`: Generates bootstrap samples
- `random_sample`: Creates random samples
- `representative_sample`: Creates representative samples

**Implementation Requirements**:

- Reproducible sampling with seed control
- Support for different sampling strategies
- Efficiency for large datasets
- Preservation of data distributions

## Integration Requirements

### 1. Operation Framework Integration

- All metric operations must inherit from appropriate base operation classes
- Operations must implement the `execute()` method as per framework requirements
- Operations must generate and return `OperationResult` with proper status, metrics, and artifacts
- Operations must use `DataSource` for input/output

### 2. I/O Integration

- Use `io.py` for reading/writing data and results
- Support for CSV, JSON, and Parquet formats
- Generate metric results in standardized JSON format
- Save visualization artifacts in appropriate formats (PNG, SVG)

### 3. Visualization Integration

- Use `visualization.py` for generating charts and visualizations
- Create standardized visualizations for different metric types
- Support for single-metric and multi-metric visualizations
- Generate comparison visualizations for original vs. transformed data

### 4. Logging and Progress Tracking

- Use the logging framework for operation logging
- Implement progress tracking for long-running calculations
- Report intermediate progress for large dataset processing
- Log detailed metric results at appropriate levels

## Implementation Considerations

### 1. Performance Optimization

- Vectorized operations using NumPy/Pandas
- Efficient implementations for large datasets
- Chunked processing for memory efficiency
- Optional parallel processing for computationally intensive metrics

### 2. Scalability

- Support for datasets of varying sizes
- Sampling strategies for very large datasets
- Memory-efficient implementations
- Progress tracking for long-running calculations

### 3. Configurability

- Configurable parameters for all metrics
- Default parameters based on best practices
- Support for domain-specific configurations
- Parameter validation and suggestions

### 4. Extensibility

- Clear extension points for new metrics
- Support for custom metric implementations
- Integration with external metric libraries
- Registration mechanism for custom metrics

## Example Implementation

Here's a simplified example of how the `CalculateFidelityMetricsOperation` might be implemented:

python

```python
# operations/fidelity_ops.py
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.metrics.calculators.fidelity_calc import (
    calculate_ks_test,
    calculate_kl_divergence,
    calculate_correlation_similarity
)
from pamola_core.io import save_json, save_figure
from pamola_core.visualization import create_distribution_comparison, create_correlation_heatmap


class CalculateFidelityMetricsOperation(BaseOperation):
    """
    Operation for calculating statistical fidelity metrics between original and 
    transformed datasets.
    """
    
    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        visualize: bool = True
    ):
        """
        Initialize the fidelity metrics calculation operation.
        
        Parameters:
        -----------
        numeric_columns : List[str], optional
            List of numeric columns to analyze. If None, all numeric columns are used.
        categorical_columns : List[str], optional
            List of categorical columns to analyze. If None, all categorical columns are used.
        metrics : List[str], optional
            List of metrics to calculate. If None, all available metrics are calculated.
            Options include: "ks_test", "kl_divergence", "correlation_similarity"
        visualize : bool, default=True
            Whether to generate visualizations for the metrics.
        """
        super().__init__()
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.metrics = metrics or ["ks_test", "kl_divergence", "correlation_similarity"]
        self.visualize = visualize
    
    def execute(
        self,
        data_source: DataSource,
        task_dir: str,
        reporter: Optional[object] = None,
        **kwargs
    ) -> OperationResult:
        """
        Execute the fidelity metrics calculation.
        
        Parameters:
        -----------
        data_source : DataSource
            Data source containing original and transformed datasets.
        task_dir : str
            Directory for storing task outputs.
        reporter : object, optional
            Reporter for progress tracking.
        **kwargs : dict
            Additional parameters.
            
        Returns:
        --------
        OperationResult
            Operation result with metrics and artifacts.
        """
        # Get datasets
        original_df = data_source.get_dataframe("original")
        transformed_df = data_source.get_dataframe("transformed")
        
        if reporter:
            reporter.update_progress(0.1, "Validating input data")
        
        # Validate datasets
        if original_df is None or transformed_df is None:
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message="Both original and transformed datasets are required"
            )
        
        # Determine columns to analyze
        numeric_cols = self.numeric_columns or original_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.categorical_columns or original_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Initialize results
        metrics_results = {}
        artifacts = []
        
        # Calculate metrics
        if reporter:
            reporter.update_progress(0.2, "Calculating distribution metrics")
        
        # KS-Test for numeric columns
        if "ks_test" in self.metrics and numeric_cols:
            ks_results = {}
            for col in numeric_cols:
                if col in transformed_df.columns:
                    ks_stat, p_value = calculate_ks_test(
                        original_df[col].values,
                        transformed_df[col].values
                    )
                    ks_results[col] = {"ks_statistic": ks_stat, "p_value": p_value}
            
            metrics_results["ks_test"] = ks_results
            
            # Generate visualizations
            if self.visualize:
                for col in numeric_cols:
                    if col in transformed_df.columns:
                        fig = create_distribution_comparison(
                            original_df[col],
                            transformed_df[col],
                            title=f"Distribution Comparison - {col}"
                        )
                        fig_path = f"{task_dir}/distribution_{col}.png"
                        save_figure(fig, fig_path)
                        artifacts.append({
                            "path": fig_path,
                            "description": f"Distribution comparison for {col}",
                            "type": "image/png"
                        })
        
        if reporter:
            reporter.update_progress(0.5, "Calculating KL divergence")
            
        # KL Divergence for numeric columns
        if "kl_divergence" in self.metrics and numeric_cols:
            kl_results = {}
            for col in numeric_cols:
                if col in transformed_df.columns:
                    kl_value = calculate_kl_divergence(
                        original_df[col].values,
                        transformed_df[col].values
                    )
                    kl_results[col] = kl_value
            
            metrics_results["kl_divergence"] = kl_results
        
        if reporter:
            reporter.update_progress(0.8, "Calculating correlation similarity")
            
        # Correlation similarity
        if "correlation_similarity" in self.metrics and len(numeric_cols) > 1:
            corr_sim = calculate_correlation_similarity(
                original_df[numeric_cols],
                transformed_df[numeric_cols]
            )
            metrics_results["correlation_similarity"] = corr_sim
            
            # Generate visualization
            if self.visualize:
                fig = create_correlation_heatmap(
                    original_df[numeric_cols].corr() - transformed_df[numeric_cols].corr(),
                    title="Correlation Difference Matrix"
                )
                fig_path = f"{task_dir}/correlation_diff.png"
                save_figure(fig, fig_path)
                artifacts.append({
                    "path": fig_path,
                    "description": "Correlation difference matrix",
                    "type": "image/png"
                })
        
        # Save metrics results
        metrics_path = f"{task_dir}/fidelity_metrics.json"
        save_json(metrics_results, metrics_path)
        artifacts.append({
            "path": metrics_path,
            "description": "Fidelity metrics results",
            "type": "application/json"
        })
        
        if reporter:
            reporter.update_progress(1.0, "Completed fidelity metrics calculation")
        
        return OperationResult(
            status=OperationStatus.SUCCESS,
            metrics=metrics_results,
            artifacts=artifacts
        )
```

