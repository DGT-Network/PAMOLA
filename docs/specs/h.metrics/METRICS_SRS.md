# PAMOLA.CORE Metrics Package Software Requirements Specification

**Document Version:** 1.0.0  
**Last Updated:** 2025-06-15  
**Status:** Initial Draft

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document defines the requirements for the PAMOLA.CORE metrics package. The package implements comprehensive evaluation operations that measure the quality and characteristics of anonymized, synthesized, or otherwise transformed data across three fundamental dimensions: fidelity (statistical similarity), utility (usefulness for downstream tasks), and privacy (protection against re-identification). These metrics are essential for quantifying the effectiveness of privacy-enhancing transformations and enabling informed decisions about privacy-utility tradeoffs.

### 1.2 Scope

The metrics package provides **read-only evaluation operations** organized into three core categories:
- **Fidelity Metrics**: Measure statistical and structural similarity between original and transformed datasets
- **Utility Metrics**: Evaluate the usefulness of transformed data for machine learning and analytical tasks
- **Privacy Metrics**: Assess disclosure risks and information loss without performing active attacks

Each metric is implemented as an independent, atomic process following the PAMOLA.CORE Operations Framework. The package focuses exclusively on measurement and evaluation—it does not modify data, generate synthetic records, or simulate attacks.

### 1.3 Distinction from Process Metrics

This package implements **outcome evaluation metrics** that measure the final results of privacy-enhancing transformations. This is fundamentally different from **process metrics** collected during operation execution:

- **Process Metrics** (collected by operations): Execution time, memory usage, records processed, error counts
- **Outcome Metrics** (this package): Statistical fidelity, model performance, privacy risks, information loss

While anonymization operations may collect basic effectiveness indicators during execution, comprehensive quality assessment requires the dedicated metric operations defined in this specification.

### 1.4 Document Conventions

- **REQ-MET-XXX**: General metrics package requirements
- **REQ-FID-XXX**: Fidelity metric requirements
- **REQ-UTL-XXX**: Utility metric requirements
- **REQ-PRV-XXX**: Privacy metric requirements
- **REQ-OP-XXX**: Operation-specific requirements
- **REQ-COM-XXX**: Commons sub-framework requirements

Priority levels:
- **MUST**: Essential requirement (MVP)
- **SHOULD**: Important but not essential
- **MAY**: Optional enhancement

## 2. Core Architecture Principles

### 2.1 Operation Contract

**REQ-MET-001 [MUST]** Every metric operation is an atomic, stateless process that:
- Inherits from `MetricOperation` (which inherits from `BaseOperation`)
- Computes only metric values without modifying input data
- Uses framework-provided services for I/O, progress tracking, and result reporting
- Returns evaluation metrics via `OperationResult` using `add_metric()` or `add_nested_metric()`
- Supports both pandas and Dask for large-scale processing
- Maintains complete isolation from data transformation operations

**REQ-MET-002 [MUST]** Metric operations SHALL NOT:
- Modify or transform input data
- Generate synthetic records or samples
- Perform attack simulations or adversarial testing
- Implement quality improvement or optimization
- Store state between executions
- Depend on specific anonymization operations

### 2.2 Framework Integration Requirements

**REQ-MET-003 [MUST]** All operations must use these framework components:

| Component | Purpose | Required Methods/Usage |
|-----------|---------|----------------------|
| `DataWriter` | All file output | `write_metrics()`, `write_json()`, `write_figure()` |
| `DataSource` | All data input | `get_dataframe()`, `get_metadata()` |
| `OperationResult` | Result reporting | `add_metric()`, `add_nested_metric()`, `add_artifact()` |
| `ProgressTracker` | Progress updates | `update()`, `create_subtask()` |
| `OperationConfig` | Configuration | Schema validation, `to_dict()`, `save()` |

**REQ-MET-004 [MUST]** All operations must use utilities from:
- `pamola_core.utils.ops.op_field_utils` for field selection and validation
- `pamola_core.utils.ops.op_data_processing` for memory optimization
- `pamola_core.metrics.commons.validation` for input validation
- `pamola_core.metrics.commons.aggregation` for metric aggregation
- `pamola_core.metrics.commons.normalize` for value normalization
- `pamola_core.utils.logging` for logging (module-specific loggers)

### 2.3 Package Structure

**REQ-MET-005 [MUST]** The package structure SHALL be:

```
pamola_core/metrics/
├── __init__.py
├── base.py                    # Base class for all metric operations
├── fidelity/                  # Statistical similarity metrics
│   ├── __init__.py
│   ├── distribution.py        # KS, KL, JS, Wasserstein
│   ├── correlation.py         # Correlation preservation
│   ├── distance.py           # Fréchet, EMD, MMD
│   └── structural.py         # SSIM, PCA structure
├── utility/                  # Task usefulness metrics
│   ├── __init__.py
│   ├── classification.py     # Accuracy, F1, AUROC
│   ├── regression.py         # R², MSE, MAE
│   ├── clustering.py         # Silhouette, Davies-Bouldin
│   └── feature.py           # Feature importance
├── privacy/                  # Privacy risk metrics
│   ├── __init__.py
│   ├── distance.py          # DCR, Hausdorff
│   ├── neighbor.py          # Nearest neighbor metrics
│   ├── identity.py          # Uniqueness, k-anonymity
│   └── information.py       # Information loss, entropy
├── operations/              # Operation implementations
│   ├── __init__.py
│   ├── fidelity_ops.py     # Fidelity metric operations
│   ├── utility_ops.py      # Utility metric operations
│   ├── privacy_ops.py      # Privacy metric operations
│   └── combined_ops.py     # Multi-metric operations
├── calculators/            # Core calculation logic
│   ├── __init__.py
│   ├── fidelity_calc.py   # Fidelity calculations
│   ├── utility_calc.py    # Utility calculations
│   ├── privacy_calc.py    # Privacy calculations
│   └── vector_calc.py     # Vector operations
└── commons/               # Shared utilities
    ├── __init__.py
    ├── validation.py      # Input validation
    ├── aggregation.py     # Metric aggregation
    └── normalize.py       # Value normalization
```

## 3. Base Operation Requirements

### 3.1 MetricOperation Base Class

**REQ-MET-006 [MUST]** All operations SHALL inherit from `MetricOperation` which provides:
- Dual dataset handling (original and transformed/synthetic)
- Automatic column alignment and validation
- Memory-efficient comparison strategies
- Visualization generation for metric results
- Standardized metric normalization
- Batch processing for large datasets

### 3.2 Constructor Interface

**REQ-MET-007 [MUST]** All metric operations SHALL have this constructor signature:

```python
def __init__(self,
             metric_name: str,                    # Metric identifier
             columns: Optional[List[str]] = None, # Columns to evaluate
             mode: str = "comparison",            # comparison, standalone
             normalize: bool = True,              # Normalize to [0,1]
             batch_size: int = 10000,
             sample_size: Optional[int] = None,   # For expensive metrics
             confidence_level: float = 0.95,      # For statistical tests
             visualization: bool = True,
             visualization_format: str = "png",
             cache_results: bool = True,
             parallel_processing: bool = False,
             n_jobs: int = -1,
             optimize_memory: bool = True,
             engine: str = "auto",               # pandas, dask, or auto
             max_rows_in_memory: int = 1000000,
             **metric_specific_params):          # Metric-specific parameters
```

### 3.3 Required Method Implementations

**REQ-MET-008 [MUST]** All operations must implement:

```python
def calculate_metric(self, 
                    original_data: pd.DataFrame,
                    transformed_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate the metric values - core evaluation logic."""
    
def _validate_inputs(self, 
                    original_data: pd.DataFrame,
                    transformed_data: pd.DataFrame) -> None:
    """Validate input datasets for metric calculation."""
    
def _normalize_metric(self, value: float) -> float:
    """Normalize metric value to [0,1] range if required."""
    
def _get_metric_metadata(self) -> Dict[str, Any]:
    """Return metric metadata (range, interpretation, etc.)."""
```

**REQ-MET-009 [SHOULD]** Operations supporting visualization should implement:

```python
def _create_visualization(self, 
                         metric_results: Dict[str, Any],
                         task_dir: Path) -> List[Path]:
    """Create visualization artifacts for metric results."""
```

### 3.4 Execution Lifecycle

**REQ-MET-010 [MUST]** The base class `execute()` method provides these phases:

1. **Configuration Saving**: Save operation config to task_dir
2. **Data Loading**: Load original and transformed datasets
3. **Input Validation**: Validate datasets and column alignment
4. **Column Selection**: Determine columns to evaluate
5. **Sampling**: Apply sampling if specified for expensive metrics
6. **Metric Calculation**: Call `calculate_metric()` with appropriate data
7. **Normalization**: Normalize results if requested
8. **Statistical Testing**: Perform significance tests if applicable
9. **Visualization Generation**: Create charts and plots
10. **Result Aggregation**: Aggregate column-level metrics
11. **Output Writing**: Save results via DataWriter

### 3.5 Metric Result Structure

**REQ-MET-011 [MUST]** All metrics must return standardized results:

```python
{
    "metric_name": str,              # Metric identifier
    "value": float,                  # Primary metric value
    "normalized_value": float,       # Normalized to [0,1]
    "components": Dict[str, float],  # Component values
    "column_results": Dict[str, Dict], # Per-column results
    "statistical_tests": {
        "test_statistic": float,
        "p_value": float,
        "confidence_interval": Tuple[float, float]
    },
    "interpretation": str,           # Human-readable interpretation
    "metadata": {
        "range": Tuple[float, float],
        "higher_is_better": bool,
        "units": str
    }
}
```

## 4. Fidelity Metrics

### 4.1 Purpose and Scope

**REQ-FID-001 [MUST]** Fidelity metrics measure statistical and structural similarity between original and transformed datasets without considering downstream task performance.

### 4.2 Distribution Metrics

**REQ-FID-002 [MUST]** Implement these distribution comparison metrics:

#### 4.2.1 Kolmogorov-Smirnov Test (Extended Implementation)

```python
class KSTestOperation(MetricOperation):
    """
    Extended KS test with aggregation support for grouped data.
    
    Implementation: Custom with aggregation + scipy.stats.ks_2samp fallback
    
    Parameters:
    - key_fields: List[str] (fields for grouping, e.g., ['age', 'income'])
    - value_field: Optional[str] (field to aggregate if using aggregation)
    - aggregation: str ('sum', 'mean', 'min', 'max', 'count', 'first', 'last')
    - alternative: str ('two-sided', 'less', 'greater')
    - exact: bool (exact vs asymptotic p-value)
    - save_distribution: bool (save intermediate distribution)
    - distribution_output_path: Optional[str]
    """
    
    def __init__(self, 
                 key_fields: Optional[List[str]] = None,
                 value_field: Optional[str] = None,
                 aggregation: str = 'sum',
                 **kwargs):
        super().__init__(**kwargs)
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation
    
    def calculate_metric(self, 
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate KS statistic with optional aggregation.
        
        Process:
        1. If key_fields provided, create value dictionaries with aggregation
        2. Calculate cumulative distributions
        3. Find maximum difference (D-statistic)
        4. Calculate p-value
        """
        if self.key_fields and self.value_field:
            # Create aggregated distributions
            orig_dist = self._create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            trans_dist = self._create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )
            
            # Calculate KS from aggregated distributions
            ks_stat, p_value = self._calculate_ks_from_dicts(orig_dist, trans_dist)
        else:
            # Standard KS test for single column
            ks_stat, p_value = stats.ks_2samp(
                original_df[self.columns[0]], 
                transformed_df[self.columns[0]]
            )
        
        return {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "max_difference": ks_stat,
            "interpretation": self._interpret_ks(ks_stat, p_value)
        }
    
    def _create_value_dictionary(self, df: pd.DataFrame, 
                                key_fields: List[str], 
                                value_field: str,
                                aggregation: str) -> Dict[str, float]:
        """Create aggregated value dictionary similar to VBA implementation."""
        # Group by key fields
        grouped = df.groupby(key_fields)[value_field]
        
        # Apply aggregation
        if aggregation == 'sum':
            result = grouped.sum()
        elif aggregation == 'mean':
            result = grouped.mean()
        elif aggregation == 'min':
            result = grouped.min()
        elif aggregation == 'max':
            result = grouped.max()
        elif aggregation == 'count':
            result = grouped.count()
        elif aggregation == 'first':
            result = grouped.first()
        elif aggregation == 'last':
            result = grouped.last()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Convert to dictionary with composite keys
        return {
            "_".join(str(k) for k in key): v 
            for key, v in result.items()
        }
    
    def _calculate_ks_from_dicts(self, 
                                dict1: Dict[str, float], 
                                dict2: Dict[str, float]) -> Tuple[float, float]:
        """Calculate KS statistic from value dictionaries."""
        # Get all keys
        all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))
        
        # Calculate totals
        total1 = sum(dict1.values())
        total2 = sum(dict2.values())
        
        # Calculate cumulative distributions
        cumulative1 = 0.0
        cumulative2 = 0.0
        max_diff = 0.0
        
        for key in all_keys:
            val1 = dict1.get(key, 0.0)
            val2 = dict2.get(key, 0.0)
            
            cumulative1 += val1 / total1
            cumulative2 += val2 / total2
            
            diff = abs(cumulative1 - cumulative2)
            max_diff = max(max_diff, diff)
        
        # Calculate p-value (approximate)
        n1 = len(dict1)
        n2 = len(dict2)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.exp(-2 * en**2 * max_diff**2)
        
        return max_diff, p_value
```

#### 4.2.2 Kullback-Leibler Divergence (Extended Implementation)

```python
class KLDivergenceOperation(MetricOperation):
    """
    Extended KL divergence with aggregation support.
    
    Implementation: Custom with smoothing + scipy.stats.entropy
    
    Parameters:
    - key_fields: List[str] (fields for grouping)
    - value_field: Optional[str] (field to aggregate)
    - aggregation: str ('sum', 'mean', 'count', etc.)
    - base: float (logarithm base, default: e)
    - bins: int (for continuous variables)
    - epsilon: float (smoothing parameter, default: 0.01)
    - handle_zeros: str ('epsilon', 'skip', 'error')
    """
    
    def __init__(self,
                 key_fields: Optional[List[str]] = None,
                 value_field: Optional[str] = None,
                 aggregation: str = 'count',
                 epsilon: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation
        self.epsilon = epsilon
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate KL divergence with smoothing.
        
        Process:
        1. Create value dictionaries if key_fields provided
        2. Apply smoothing for zero values
        3. Normalize to probability distributions
        4. Calculate KL divergence: Σ P(x) * log(P(x)/Q(x))
        """
        if self.key_fields:
            # Create aggregated distributions
            p_dict = self._create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            q_dict = self._create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )
            
            # Calculate KL from dictionaries
            kl_value = self._calculate_kl_from_dicts(p_dict, q_dict)
        else:
            # Standard KL for single column
            p_vals, q_vals = self._prepare_distributions(
                original_df[self.columns[0]],
                transformed_df[self.columns[0]]
            )
            kl_value = stats.entropy(p_vals, q_vals)
        
        return {
            "kl_divergence": kl_value,
            "kl_divergence_bits": kl_value / np.log(2),  # Convert to bits
            "interpretation": self._interpret_kl(kl_value),
            "smoothing_applied": self.epsilon > 0
        }
    
    def _calculate_kl_from_dicts(self,
                                p_dict: Dict[str, float],
                                q_dict: Dict[str, float]) -> float:
        """Calculate KL divergence from value dictionaries with smoothing."""
        # Get all keys from both distributions
        all_keys = set(p_dict.keys()) | set(q_dict.keys())
        
        # Apply smoothing and calculate totals
        smoothed_p = {}
        smoothed_q = {}
        
        for key in all_keys:
            p_val = p_dict.get(key, 0.0)
            q_val = q_dict.get(key, 0.0)
            
            # Apply epsilon smoothing
            if p_val <= 0:
                p_val = self.epsilon
            if q_val <= 0:
                q_val = self.epsilon
            
            smoothed_p[key] = p_val
            smoothed_q[key] = q_val
        
        # Normalize to probabilities
        total_p = sum(smoothed_p.values())
        total_q = sum(smoothed_q.values())
        
        # Calculate KL divergence
        kl = 0.0
        for key in all_keys:
            p = smoothed_p[key] / total_p
            q = smoothed_q[key] / total_q
            
            if p > 0:  # Only add if p > 0 (KL undefined otherwise)
                kl += p * np.log(p / q)
        
        return kl
```

#### 4.2.3 Jensen-Shannon Divergence (Extended Implementation)

```python
class JSDivergenceOperation(MetricOperation):
    """
    Symmetric measure of similarity between distributions.
    
    Implementation: Custom based on KL divergence
    
    Parameters:
    - key_fields: List[str] (fields for grouping)
    - value_field: Optional[str] (field to aggregate)
    - aggregation: str ('sum', 'mean', 'count', etc.)
    - base: float (logarithm base, default: 2)
    - bins: int (for continuous variables)
    - epsilon: float (smoothing parameter)
    
    Formula:
    JSD(P||Q) = 1/2 * KL(P||M) + 1/2 * KL(Q||M)
    where M = 1/2 * (P + Q) is the average distribution
    """
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Jensen-Shannon divergence."""
        if self.key_fields:
            # Create aggregated distributions
            p_dict = self._create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            q_dict = self._create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )
            
            jsd_value = self._calculate_jsd_from_dicts(p_dict, q_dict)
        else:
            # Standard JSD for single column
            p_vals, q_vals = self._prepare_distributions(
                original_df[self.columns[0]],
                transformed_df[self.columns[0]]
            )
            # Calculate average distribution
            m_vals = 0.5 * (p_vals + q_vals)
            
            # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_pm = stats.entropy(p_vals, m_vals, base=self.base)
            kl_qm = stats.entropy(q_vals, m_vals, base=self.base)
            jsd_value = 0.5 * kl_pm + 0.5 * kl_qm
        
        return {
            "jsd": jsd_value,
            "jsd_sqrt": np.sqrt(jsd_value),  # JSD distance metric
            "range": "[0, ln(2)]" if self.base == np.e else "[0, 1]",
            "interpretation": self._interpret_jsd(jsd_value)
        }
```

#### 4.2.4 Wasserstein Distance

```python
class WassersteinDistanceOperation(MetricOperation):
    """
    Implementation: scipy.stats.wasserstein_distance
    Parameters:
    - p: float (norm parameter, default 1)
    """
```

### 4.3 Correlation Metrics

**REQ-FID-003 [MUST]** Implement correlation preservation metrics:

```python
class CorrelationDifferenceOperation(MetricOperation):
    """
    Measures: Frobenius norm of correlation matrix difference
    Implementation: numpy.linalg.norm
    Parameters:
    - method: str ('pearson', 'spearman', 'kendall')
    - threshold: float (minimum correlation to consider)
    """
```

### 4.4 Distance Metrics

**REQ-FID-004 [MUST]** Implement distance-based fidelity metrics:

```python
class FrechetDistanceOperation(MetricOperation):
    """
    Implementation: Custom using mean/covariance comparison
    Parameters:
    - assume_gaussian: bool
    - robust_estimation: bool
    """

class MaximumMeanDiscrepancyOperation(MetricOperation):
    """
    Implementation: Custom kernel-based
    Parameters:
    - kernel: str ('rbf', 'linear', 'polynomial')
    - kernel_params: Dict
    """

class MultivariateHellingerDistanceOperation(MetricOperation):
    """
    Multivariate extension of Hellinger distance.
    
    Implementation: Custom based on probability density estimation
    
    Parameters:
    - estimation_method: str ('kde', 'histogram', 'gmm')
    - bandwidth: Optional[float] (for KDE)
    - n_bins: int (for histogram method)
    - n_components: int (for GMM)
    
    Formula:
    H(P,Q) = 1/√2 * √(Σ(√pi - √qi)²)
    
    For multivariate case, uses density estimation or binning
    """
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate multivariate Hellinger distance."""
        if len(self.columns) == 1:
            # Univariate case
            h_dist = self._calculate_univariate_hellinger(
                original_df[self.columns[0]],
                transformed_df[self.columns[0]]
            )
        else:
            # Multivariate case
            if self.estimation_method == 'kde':
                h_dist = self._calculate_hellinger_kde(
                    original_df[self.columns],
                    transformed_df[self.columns]
                )
            elif self.estimation_method == 'histogram':
                h_dist = self._calculate_hellinger_histogram(
                    original_df[self.columns],
                    transformed_df[self.columns]
                )
            else:  # gmm
                h_dist = self._calculate_hellinger_gmm(
                    original_df[self.columns],
                    transformed_df[self.columns]
                )
        
        return {
            "hellinger_distance": h_dist,
            "range": "[0, 1]",
            "estimation_method": self.estimation_method,
            "interpretation": "0=identical, 1=completely different"
        }
```

### 4.5 Statistical Test Metrics

**REQ-FID-006 [MUST]** Implement statistical test metrics:

```python
class ChiSquaredTestOperation(MetricOperation):
    """
    Chi-squared test for categorical data similarity.
    
    Implementation: scipy.stats.chi2_contingency or custom
    
    Parameters:
    - categorical_columns: List[str] (columns to test)
    - min_expected_frequency: float (minimum expected frequency)
    - correction: bool (Yates' correction for 2x2 tables)
    - handle_zeros: str ('add_epsilon', 'skip', 'error')
    
    Formula:
    χ² = Σ((Oi - Ei)²/Ei)
    where Oi = observed frequency, Ei = expected frequency
    """
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate chi-squared test statistic."""
        results = {}
        
        for col in self.categorical_columns:
            # Get value counts
            orig_counts = original_df[col].value_counts()
            trans_counts = transformed_df[col].value_counts()
            
            # Align categories
            all_categories = sorted(set(orig_counts.index) | set(trans_counts.index))
            
            # Create contingency table
            observed = np.array([
                [orig_counts.get(cat, 0) for cat in all_categories],
                [trans_counts.get(cat, 0) for cat in all_categories]
            ])
            
            # Apply chi-squared test
            chi2, p_value, dof, expected = stats.chi2_contingency(
                observed, 
                correction=self.correction
            )
            
            results[col] = {
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "n_categories": len(all_categories)
            }
        
        # Overall result
        overall_chi2 = sum(r["chi2_statistic"] for r in results.values())
        overall_dof = sum(r["degrees_of_freedom"] for r in results.values())
        
        return {
            "overall_chi2": overall_chi2,
            "overall_p_value": 1 - stats.chi2.cdf(overall_chi2, overall_dof),
            "column_results": results,
            "interpretation": self._interpret_chi2(overall_chi2, overall_dof)
        }
```

## 5. Utility Metrics

### 5.1 Purpose and Scope

**REQ-UTL-001 [MUST]** Utility metrics evaluate the usefulness of transformed data for specific machine learning and analytical tasks.

### 5.2 Classification Metrics

**REQ-UTL-002 [MUST]** Implement model performance comparison for classification:

```python
class ClassificationUtilityOperation(MetricOperation):
    """
    Implementation: scikit-learn classifiers
    Parameters:
    - models: List[str] (default: ['logistic', 'rf', 'svm'])
    - metrics: List[str] (default: ['accuracy', 'f1', 'auroc'])
    - cv_folds: int (cross-validation folds)
    - stratified: bool
    - test_size: float
    """
```

Required metrics:
- Accuracy difference
- F1 score comparison
- AUROC comparison
- Precision/Recall tradeoff

### 5.3 Regression Metrics (Extended Implementation)

**REQ-UTL-003 [MUST]** Implement model performance comparison for regression:

```python
class RegressionUtilityOperation(MetricOperation):
    """
    Extended regression utility with R² calculation similar to VBA.
    
    Implementation: scikit-learn regressors + custom R² calculation
    
    Parameters:
    - models: List[str] (default: ['linear', 'rf', 'svr'])
    - metrics: List[str] (default: ['r2', 'mse', 'mae'])
    - key_fields: Optional[List[str]] (for grouped R² calculation)
    - value_field: Optional[str] (target variable for grouped analysis)
    - aggregation: str (for grouped data)
    - cv_folds: int
    - test_size: float
    """
    
    def __init__(self,
                 key_fields: Optional[List[str]] = None,
                 value_field: Optional[str] = None,
                 aggregation: str = 'sum',
                 **kwargs):
        super().__init__(**kwargs)
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate regression metrics with optional grouped R².
        """
        results = {}
        
        if self.key_fields and self.value_field:
            # Calculate grouped R² similar to VBA implementation
            r2_grouped = self._calculate_grouped_r2(
                original_df, transformed_df, 
                self.key_fields, self.value_field, 
                self.aggregation
            )
            results["grouped_r2"] = r2_grouped
        
        # Standard model-based metrics
        if 'r2' in self.metrics or not self.key_fields:
            model_results = self._calculate_model_metrics(
                original_df, transformed_df
            )
            results.update(model_results)
        
        return results
    
    def _calculate_grouped_r2(self,
                             df1: pd.DataFrame,
                             df2: pd.DataFrame,
                             key_fields: List[str],
                             value_field: str,
                             aggregation: str) -> Dict[str, float]:
        """
        Calculate R² for grouped data similar to VBA implementation.
        
        Process:
        1. Create value dictionaries with aggregation
        2. Calculate sums: Σx, Σy, Σxy, Σx², Σy²
        3. Calculate means and regression coefficients
        4. Calculate SStot and SSres
        5. Calculate R² = 1 - (SSres/SStot)
        """
        # Create aggregated dictionaries
        x_dict = self._create_value_dictionary(df1, key_fields, value_field, aggregation)
        y_dict = self._create_value_dictionary(df2, key_fields, value_field, aggregation)
        
        # Get common keys
        common_keys = set(x_dict.keys()) & set(y_dict.keys())
        n = len(common_keys)
        
        if n == 0:
            return {"r_squared": 0.0, "n_points": 0}
        
        # Calculate sums
        sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
        
        x_values = []
        y_values = []
        
        for key in common_keys:
            x = x_dict[key]
            y = y_dict[key]
            
            x_values.append(x)
            y_values.append(y)
            
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y
        
        # Calculate means
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        # Calculate regression coefficient (slope)
        if (sum_x2 - n * mean_x * mean_x) != 0:
            slope = (sum_xy - n * mean_x * mean_y) / (sum_x2 - n * mean_x * mean_x)
            intercept = mean_y - slope * mean_x
        else:
            return {"r_squared": 0.0, "n_points": n, "error": "Zero variance in x"}
        
        # Calculate SStot and SSres
        ss_tot = ss_res = 0.0
        
        for i, key in enumerate(common_keys):
            x = x_values[i]
            y = y_values[i]
            y_pred = slope * x + intercept
            
            ss_tot += (y - mean_y) ** 2
            ss_res += (y - y_pred) ** 2
        
        # Calculate R²
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0.0
        
        return {
            "r_squared": round(r_squared, 4),
            "n_points": n,
            "slope": slope,
            "intercept": intercept,
            "mean_x": mean_x,
            "mean_y": mean_y
        }
```

Required metrics:
- R² score comparison (standard and grouped)
- MSE/RMSE comparison
- MAE comparison
- Propensity MSE (pMSE)

### 5.4 Clustering Metrics

**REQ-UTL-004 [SHOULD]** Implement clustering quality comparison:

```python
class ClusteringUtilityOperation(MetricOperation):
    """
    Implementation: scikit-learn clustering
    Parameters:
    - algorithms: List[str] (default: ['kmeans', 'dbscan'])
    - n_clusters: Union[int, str] (number or 'auto')
    - metrics: List[str] (default: ['silhouette', 'davies_bouldin'])
    """
```

### 5.5 Feature Importance Metrics

**REQ-UTL-005 [SHOULD]** Implement feature importance preservation:

```python
class FeatureImportanceOperation(MetricOperation):
    """
    Implementation: Model-based feature importance
    Parameters:
    - model: str (model type for importance)
    - importance_type: str ('gain', 'split', 'permutation')
    - correlation_method: str ('spearman', 'pearson')
    """
```

## 6. Privacy Metrics

### 6.1 Purpose and Scope

**REQ-PRV-001 [MUST]** Privacy metrics assess disclosure risks and information loss without performing active attacks or simulations.

### 6.2 Distance-Based Privacy Metrics (Extended)

**REQ-PRV-002 [MUST]** Implement distance to closest record:

```python
class DistanceToClosestRecordOperation(MetricOperation):
    """
    DCR measures minimum distance from synthetic to real records.
    
    Implementation: Custom using scipy.spatial with optimization
    
    Parameters:
    - distance_metric: str ('euclidean', 'manhattan', 'cosine', 'mahalanobis')
    - normalize_features: bool
    - percentiles: List[int] (default: [5, 25, 50, 75, 95])
    - sample_size: Optional[int] (for large datasets)
    - use_faiss: bool (use FAISS for large-scale search)
    - aggregation: str ('min', 'mean_k', 'percentile')
    - k_neighbors: int (for mean_k aggregation)
    """
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate DCR with various aggregation options.
        
        Formula:
        DCR_i = min_j(dist(x_i^s, x_j^r))
        where x_i^s is synthetic record, x_j^r is real record
        """
        # Prepare data
        numeric_cols = self.columns or original_df.select_dtypes(include=np.number).columns
        
        if self.normalize_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(original_df[numeric_cols])
            orig_scaled = scaler.transform(original_df[numeric_cols])
            trans_scaled = scaler.transform(transformed_df[numeric_cols])
        else:
            orig_scaled = original_df[numeric_cols].values
            trans_scaled = transformed_df[numeric_cols].values
        
        # Calculate distances
        if self.use_faiss and orig_scaled.shape[0] > 10000:
            dcr_values = self._calculate_dcr_faiss(orig_scaled, trans_scaled)
        else:
            dcr_values = self._calculate_dcr_scipy(orig_scaled, trans_scaled)
        
        # Calculate statistics
        dcr_stats = {
            "min": np.min(dcr_values),
            "max": np.max(dcr_values),
            "mean": np.mean(dcr_values),
            "std": np.std(dcr_values)
        }
        
        # Percentiles
        for p in self.percentiles:
            dcr_stats[f"p{p}"] = np.percentile(dcr_values, p)
        
        # Risk assessment
        # Records with very small DCR are at higher risk
        risk_thresholds = {
            "high_risk": np.sum(dcr_values < np.percentile(dcr_values, 5)),
            "medium_risk": np.sum(dcr_values < np.percentile(dcr_values, 25)),
            "low_risk": np.sum(dcr_values >= np.percentile(dcr_values, 25))
        }
        
        return {
            "dcr_statistics": dcr_stats,
            "risk_assessment": risk_thresholds,
            "proportion_at_risk": risk_thresholds["high_risk"] / len(dcr_values),
            "privacy_score": self._calculate_privacy_score(dcr_values),
            "interpretation": self._interpret_dcr(dcr_stats)
        }
    
    def _calculate_privacy_score(self, dcr_values):
        """Convert DCR distribution to privacy score [0,1]."""
        # Higher DCR = better privacy
        # Normalize by data dimensionality
        normalized_dcr = dcr_values / np.sqrt(len(self.columns))
        # Use sigmoid-like transformation
        scores = 1 - np.exp(-normalized_dcr)
        return np.mean(scores)
```

### 6.3 Neighbor-Based Privacy Metrics (Extended)

**REQ-PRV-005 [MUST]** Implement enhanced neighbor-based privacy metrics:

```python
class NearestNeighborDistanceRatioOperation(MetricOperation):
    """
    NNDR measures ratio of distances to nearest and second-nearest neighbors.
    
    Implementation: Custom using sklearn.neighbors
    
    Parameters:
    - distance_metric: str ('euclidean', 'manhattan', 'cosine')
    - n_neighbors: int (default: 2, for NNDR)
    - normalize_features: bool
    - threshold: float (NNDR threshold for privacy risk)
    
    Formula:
    NNDR_i = dist(x_i^s, x_1^r) / dist(x_i^s, x_2^r)
    where x_1^r and x_2^r are first and second nearest neighbors
    """
    
    def calculate_metric(self,
                        original_df: pd.DataFrame,
                        transformed_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate NNDR for privacy assessment."""
        from sklearn.neighbors import NearestNeighbors
        
        # Prepare data
        numeric_cols = self.columns or original_df.select_dtypes(include=np.number).columns
        
        if self.normalize_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(original_df[numeric_cols])
            orig_data = scaler.transform(original_df[numeric_cols])
            trans_data = scaler.transform(transformed_df[numeric_cols])
        else:
            orig_data = original_df[numeric_cols].values
            trans_data = transformed_df[numeric_cols].values
        
        # Fit nearest neighbors model
        nn_model = NearestNeighbors(
            n_neighbors=2,
            metric=self.distance_metric,
            algorithm='auto'
        ).fit(orig_data)
        
        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(trans_data)
        
        # Calculate NNDR
        nndr_values = distances[:, 0] / (distances[:, 1] + 1e-10)  # Avoid division by zero
        
        # Statistics
        nndr_stats = {
            "mean": np.mean(nndr_values),
            "std": np.std(nndr_values),
            "min": np.min(nndr_values),
            "max": np.max(nndr_values)
        }
        
        # Risk assessment based on NNDR
        # Lower NNDR indicates higher privacy risk (record is much closer to one real record)
        high_risk_count = np.sum(nndr_values < self.threshold)
        
        results = {
            "nndr_statistics": nndr_stats,
            "high_risk_records": high_risk_count,
            "high_risk_proportion": high_risk_count / len(nndr_values),
            "privacy_classification": {
                "realistic": np.sum(nndr_values > 0.8),  # Close to 1
                "at_risk": np.sum(nndr_values < 0.5)    # Close to 0
            },
            "interpretation": self._interpret_nndr(nndr_stats, high_risk_count)
        }
        
        return results
```

### 6.3 Identity Disclosure Metrics

**REQ-PRV-003 [MUST]** Implement uniqueness and k-anonymity metrics:

```python
class UniquenessOperation(MetricOperation):
    """
    Implementation: Custom groupby analysis
    Parameters:
    - quasi_identifiers: List[str]
    - k_values: List[int] (default: [2, 3, 5, 10])
    - l_diversity: bool (calculate l-diversity)
    - t_closeness: bool (calculate t-closeness)
    """
```

### 6.4 Information Theory Metrics

**REQ-PRV-004 [MUST]** Implement information loss metrics:

```python
class InformationLossOperation(MetricOperation):
    """
    Implementation: Custom entropy calculations
    Parameters:
    - method: str ('entropy', 'mutual_information', 'precision')
    - granularity: str ('cell', 'column', 'dataset')
    - weight_by_importance: bool
    """
```

### 6.5 Nearest Neighbor Metrics

**REQ-PRV-005 [SHOULD]** Implement neighbor-based privacy metrics:

```python
class NearestNeighborPrivacyOperation(MetricOperation):
    """
    Implementation: sklearn.neighbors
    Parameters:
    - n_neighbors: int
    - distance_metric: str
    - accuracy_threshold: float
    """
```

## 7. Combined Metrics Operations

### 7.1 Comprehensive Assessment

**REQ-MET-012 [MUST]** Implement operations that combine multiple metrics:

```python
class ComprehensiveQualityOperation(MetricOperation):
    """
    Calculates all applicable metrics across fidelity, utility, and privacy.
    Parameters:
    - fidelity_metrics: List[str]
    - utility_metrics: List[str]
    - privacy_metrics: List[str]
    - weights: Dict[str, float] (for composite scores)
    - generate_report: bool
    """
```

### 7.2 Composite Indices

**REQ-MET-013 [SHOULD]** Implement standardized composite indices:

```python
class SyntheticDataQualityIndex(MetricOperation):
    """
    Calculates SDQI (Synthetic Data Quality Index).
    Formula: weighted average of normalized metrics
    Parameters:
    - component_weights: Dict[str, float]
    - normalization_method: str
    """

class PrivacyUtilityTradeoffIndex(MetricOperation):
    """
    Calculates PUTI (Privacy-Utility Tradeoff Index).
    Parameters:
    - privacy_weight: float (0-1)
    - utility_weight: float (0-1)
    """
```

## 8. Commons Module Requirements

### 8.1 Validation Module

**REQ-COM-001 [MUST]** Provide comprehensive input validation:

```python
def validate_dataset_compatibility(df1: pd.DataFrame, df2: pd.DataFrame,
                                 require_same_columns: bool = True,
                                 require_same_types: bool = True) -> ValidationResult:
    """Validate that two datasets can be compared."""

def validate_metric_inputs(original: pd.DataFrame, transformed: pd.DataFrame,
                         columns: List[str], metric_type: str) -> None:
    """Validate inputs for specific metric type."""
```

### 8.2 Aggregation Module

**REQ-COM-002 [MUST]** Provide metric aggregation utilities:

```python
def aggregate_column_metrics(column_results: Dict[str, Dict[str, float]],
                           method: str = "mean",
                           weights: Optional[Dict[str, float]] = None) -> float:
    """Aggregate column-level metrics to dataset level."""

def create_composite_score(metrics: Dict[str, float],
                         weights: Dict[str, float],
                         normalization: str = "minmax") -> float:
    """Create weighted composite score from multiple metrics."""
```

### 8.3 Normalization Module

**REQ-COM-003 [MUST]** Provide value normalization:

```python
def normalize_metric_value(value: float, 
                         metric_range: Tuple[float, float],
                         target_range: Tuple[float, float] = (0, 1),
                         higher_is_better: bool = True) -> float:
    """Normalize metric value to target range."""

def normalize_distribution(values: np.ndarray,
                         method: str = "minmax") -> np.ndarray:
    """Normalize distribution for comparison."""
```

## 9. Integration Requirements

### 9.1 DataWriter Integration

**REQ-MET-014 [MUST]** All outputs use DataWriter:

```python
# Write metric results
metrics_result = writer.write_metrics(
    metrics=evaluation_results,
    name=f"{metric_name}_results",
    timestamp_in_name=True
)

# Write visualizations
for viz_path in visualizations:
    viz_result = writer.write_figure(
        figure=fig,
        name=f"{metric_name}_visualization",
        format=self.visualization_format
    )
```

### 9.2 Progress Tracking

**REQ-MET-015 [MUST]** Use ProgressTracker for all operations:

```python
# For multi-metric operations
metric_progress = progress_tracker.create_subtask(
    total=len(metrics_to_calculate),
    description="Calculating metrics",
    unit="metrics"
)

# Update for each metric
metric_progress.update(1, {
    "current_metric": metric_name,
    "completed": completed_count,
    "percentage": (completed_count / total_metrics) * 100
})
```

## 10. Performance and Scalability

### 10.1 Computational Efficiency

**REQ-MET-016 [MUST]** Operations must support:
- Sampling strategies for expensive metrics (DCR, MMD)
- Vectorized operations using NumPy/Pandas
- Optional GPU acceleration for distance calculations
- Caching of intermediate results

### 10.2 Memory Management

**REQ-MET-017 [MUST]** Implement memory-efficient strategies:
- Chunked processing for pairwise comparisons
- Approximate algorithms for large datasets
- Configurable precision/performance tradeoffs
- Automatic downsampling when needed

### 10.3 Parallel Processing

**REQ-MET-018 [SHOULD]** Support parallel computation:
- Column-level parallelization
- Model training parallelization
- Distance calculation parallelization
- Thread-safe metric aggregation

## 11. Testing Requirements

**REQ-MET-019 [MUST]** Each metric operation must have:

1. **Unit tests** covering:
   - Correctness against known results
   - Edge cases (empty data, single value, etc.)
   - Different data types and distributions
   - Normalization accuracy

2. **Integration tests** verifying:
   - End-to-end operation execution
   - Visualization generation
   - Result serialization
   - Framework integration

3. **Performance tests** checking:
   - Scalability with data size
   - Memory usage patterns
   - Execution time benchmarks
   - Sampling effectiveness

4. **Statistical tests** validating:
   - Metric properties (bounds, monotonicity)
   - Statistical test validity
   - Confidence interval accuracy
   - Distribution assumptions

## 12. Implementation Guidelines

### 12.1 External Dependencies

**REQ-MET-020 [MUST]** Use established libraries where appropriate:

| Metric Category | Recommended Libraries |
|----------------|---------------------|
| Statistical Tests | scipy.stats |
| Distance Calculations | scipy.spatial, sklearn.metrics |
| ML Models | scikit-learn |
| Information Theory | scipy.stats, custom implementations |
| Visualization | matplotlib, seaborn (via framework) |

### 12.2 Custom Implementations

**REQ-MET-021 [MUST]** Implement custom calculations for:
- Privacy-specific metrics (DCR, k-anonymity analysis)
- Composite indices (SDQI, PUTI)
- Optimized aggregations
- Domain-specific normalizations

### 12.3 Metric Interpretation

**REQ-MET-022 [SHOULD]** Each metric should provide:
- Clear interpretation guidelines
- Threshold recommendations
- Domain-specific considerations
- Visualization best practices

## 13. Configuration Schema

**REQ-MET-023 [MUST]** Each operation must define configuration schema:

```python
class MetricOperationConfig(OperationConfig):
    """Base configuration for metric operations."""
    
    schema = {
        "type": "object",
        "properties": {
            "metric_name": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "normalize": {"type": "boolean"},
            "visualization": {"type": "boolean"},
            # Metric-specific parameters
        },
        "required": ["metric_name"]
    }
```

## 14. Error Handling

**REQ-MET-024 [MUST]** Standardized error handling:

```python
# In calculate_metric():
try:
    # Validate inputs
    self._validate_inputs(original_data, transformed_data)
    
    # Calculate metric
    result = self._compute_metric(original_data, transformed_data)
    
except IncompatibleDataError as e:
    self.logger.error(f"Data compatibility error: {e}")
    raise
except InsufficientDataError as e:
    self.logger.warning(f"Insufficient data: {e}")
    return self._handle_insufficient_data()
except Exception as e:
    self.logger.error(f"Unexpected error in {self.metric_name}: {e}")
    raise MetricCalculationError(f"Failed to calculate {self.metric_name}: {e}")
```

## 15. Implementation Checklist

### 15.1 New Metric Operation Checklist

**REQ-MET-025 [MUST]** Every new metric operation must:

- [ ] Inherit from `MetricOperation`
- [ ] Implement `calculate_metric()` method
- [ ] Implement `_validate_inputs()` method
- [ ] Implement `_normalize_metric()` if applicable
- [ ] Implement `_get_metric_metadata()` method
- [ ] Use DataWriter for ALL outputs
- [ ] Use ProgressTracker for progress updates
- [ ] Support both pandas and Dask (if applicable)
- [ ] Include comprehensive docstrings
- [ ] Define configuration schema
- [ ] Provide interpretation guidelines
- [ ] Add unit tests with >90% coverage
- [ ] Document external dependencies
- [ ] Register with metric registry

### 15.2 Anti-Patterns to Avoid

**REQ-MET-026 [MUST NOT]** Metric operations must NOT:

- [ ] Modify input data
- [ ] Generate synthetic records
- [ ] Perform attack simulations
- [ ] Open files directly
- [ ] Print to console
- [ ] Store state between calls
- [ ] Implement data transformations
- [ ] Make assumptions about data source
- [ ] Skip normalization without justification
- [ ] Use hard-coded thresholds

## 16. Extended Implementation Examples

### 16.1 Value Dictionary Creation (Common Utility)

**REQ-MET-027 [MUST]** Provide common utility for creating aggregated value dictionaries:

```python
# commons/aggregation.py

def create_value_dictionary(df: pd.DataFrame,
                          key_fields: List[str],
                          value_field: Optional[str] = None,
                          aggregation: str = 'sum',
                          show_progress: bool = True) -> Dict[str, float]:
    """
    Create aggregated value dictionary similar to VBA CreateValueDictionary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    key_fields : List[str]
        Fields to use as composite key
    value_field : str, optional
        Field to aggregate. If None, performs count
    aggregation : str
        Aggregation function: 'sum', 'mean', 'min', 'max', 'count', 'first', 'last'
    show_progress : bool
        Show progress bar for large datasets
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with composite keys and aggregated values
    """
    if value_field is None or aggregation == 'count':
        # Count occurrences
        grouped = df.groupby(key_fields).size()
    else:
        # Group and aggregate
        grouped = df.groupby(key_fields)[value_field].agg(aggregation)
    
    # Convert to dictionary with composite string keys
    result = {}
    for key, value in grouped.items():
        if isinstance(key, tuple):
            composite_key = "_".join(str(k) for k in key)
        else:
            composite_key = str(key)
        result[composite_key] = float(value)
    
    return result

def calculate_distribution_metrics(dict1: Dict[str, float],
                                 dict2: Dict[str, float],
                                 metric_type: str = 'ks') -> Dict[str, Any]:
    """
    Calculate distribution metrics from value dictionaries.
    
    Parameters:
    -----------
    dict1, dict2 : Dict[str, float]
        Value dictionaries to compare
    metric_type : str
        Type of metric: 'ks', 'kl', 'js', 'wasserstein'
    
    Returns:
    --------
    Dict[str, Any]
        Metric results including value and additional statistics
    """
    if metric_type == 'ks':
        return _calculate_ks_from_dicts(dict1, dict2)
    elif metric_type == 'kl':
        return _calculate_kl_from_dicts(dict1, dict2)
    elif metric_type == 'js':
        return _calculate_js_from_dicts(dict1, dict2)
    elif metric_type == 'wasserstein':
        return _calculate_wasserstein_from_dicts(dict1, dict2)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
```

### 16.2 Distribution Saving Utility

**REQ-MET-028 [SHOULD]** Provide utility for saving distribution comparisons:

```python
# commons/distribution_utils.py

def save_distribution_comparison(dict1: Dict[str, float],
                               dict2: Dict[str, float],
                               output_path: str,
                               dataset1_name: str = "original",
                               dataset2_name: str = "transformed",
                               key_fields: List[str] = None,
                               include_cumulative: bool = True) -> pd.DataFrame:
    """
    Save distribution comparison to file, similar to VBA WriteDistribTable.
    
    Parameters:
    -----------
    dict1, dict2 : Dict[str, float]
        Value dictionaries to compare
    output_path : str
        Path to save the comparison
    dataset1_name, dataset2_name : str
        Names for the datasets
    key_fields : List[str]
        Original key field names
    include_cumulative : bool
        Include cumulative distributions
    
    Returns:
    --------
    pd.DataFrame
        Distribution comparison dataframe
    """
    # Create comparison dataframe
    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))
    
    rows = []
    total1 = sum(dict1.values())
    total2 = sum(dict2.values())
    
    cumulative1 = 0.0
    cumulative2 = 0.0
    
    for key in all_keys:
        val1 = dict1.get(key, 0.0)
        val2 = dict2.get(key, 0.0)
        
        prob1 = val1 / total1 if total1 > 0 else 0
        prob2 = val2 / total2 if total2 > 0 else 0
        
        cumulative1 += prob1
        cumulative2 += prob2
        
        row = {
            'key': key,
            f'{dataset1_name}_value': val1,
            f'{dataset2_name}_value': val2,
            f'{dataset1_name}_probability': prob1,
            f'{dataset2_name}_probability': prob2,
            'probability_diff': abs(prob1 - prob2)
        }
        
        if include_cumulative:
            row.update({
                f'{dataset1_name}_cumulative': cumulative1,
                f'{dataset2_name}_cumulative': cumulative2,
                'cumulative_diff': abs(cumulative1 - cumulative2)
            })
        
        # Decode composite key if key_fields provided
        if key_fields and '_' in key:
            key_values = key.split('_')[1:]  # Skip first empty element
            for i, field in enumerate(key_fields):
                if i < len(key_values):
                    row[field] = key_values[i]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to file
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)
    
    return df
```

### 16.3 Metric Result Formatting

**REQ-MET-029 [MUST]** Provide consistent metric result formatting:

```python
# commons/formatting.py

def format_metric_result(metric_name: str,
                        metric_value: float,
                        tables: List[str],
                        fields: List[str],
                        key_fields: Optional[List[str]] = None,
                        aggregation: Optional[str] = None,
                        execution_time: float = None,
                        additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Format metric results consistently, similar to VBA WriteToMetric.
    
    Returns:
    --------
    Dict[str, Any]
        Formatted metric result with metadata
    """
    result = {
        "metric_name": metric_name,
        "value": round(metric_value, 4),
        "datasets": {
            "original": tables[0] if len(tables) > 0 else None,
            "transformed": tables[1] if len(tables) > 1 else None
        },
        "fields": fields,
        "configuration": {
            "key_fields": key_fields,
            "aggregation": aggregation
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time
        }
    }
    
    if additional_info:
        result["additional_info"] = additional_info
    
    # Add parameter string similar to VBA sParamName
    param_parts = [metric_name]
    if len(tables) >= 2:
        param_parts.append(f"{tables[0]}-{tables[1]}")
    if fields:
        param_parts.append("-".join(fields))
    if key_fields:
        param_parts.append(f"({';'.join(key_fields)})")
    
    result["parameter_name"] = "_".join(param_parts)
    
    return result
```

## 17. Glossary

**Fidelity**: Statistical and structural similarity between original and transformed data

**Utility**: Usefulness of transformed data for downstream analytical and ML tasks

**Privacy Metric**: Quantitative measure of disclosure risk or information loss

**Normalization**: Scaling metric values to a standard range (typically [0,1])

**Composite Index**: Weighted combination of multiple metrics into a single score

**DCR**: Distance to Closest Record - minimum distance from synthetic to real records

**SDQI**: Synthetic Data Quality Index - composite measure of synthetic data quality

**PUTI**: Privacy-Utility Tradeoff Index - balanced measure of privacy and utility

## 18. References

1. Synthetic Data Evaluation Framework (IEEE, 2024)
2. Privacy Metrics for Synthetic Data (Journal of Privacy Technology, 2023)
3. Statistical Similarity Measures (Computational Statistics Review, 2024)
4. Machine Learning Performance Metrics (MLOps Best Practices, 2025)

## 19. Appendix: Metric Catalog Summary

| Metric | Category | Range | Higher is Better | MVP Priority |
|--------|----------|-------|------------------|--------------|
| KS Test p-value | Fidelity | [0,1] | Yes | MUST |
| KL Divergence | Fidelity | [0,∞) | No | MUST |
| JS Divergence | Fidelity | [0,1] | No | MUST |
| Hellinger Distance | Fidelity | [0,1] | No | MUST |
| Chi-squared Test | Fidelity | p-value [0,1] | Yes | MUST |
| Correlation Diff | Fidelity | [0,∞) | No | MUST |
| Mahalanobis Analysis | Fidelity | Various | Context-dependent | SHOULD |
| Classification F1 | Utility | [0,1] | Yes | MUST |
| Regression R² | Utility | [0,1] | Yes | MUST |
| DCR | Privacy | [0,∞) | Yes | MUST |
| NNDR | Privacy | [0,1] | Context-dependent | MUST |
| k-anonymity | Privacy | [1,∞) | Yes | MUST |
| Information Loss | Privacy | [0,1] | No | SHOULD |

## 20. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-06-15 | PAMOLA Team | Initial draft aligned with framework standards |