# PAMOLA.CORE Noise Operations Implementation Plan (MVP)

## 1. Package Structure

```
pamola_core/anonymization/noise/
├── __init__.py
├── uniform_numeric_op.py     # Uniform noise for numeric fields
└── uniform_temporal_op.py    # Uniform noise for datetime fields

pamola_core/anonymization/commons/   # Shared utilities (extend existing)
├── noise_utils.py           # Noise-specific utilities
└── statistical_utils.py     # Statistical analysis helpers
```

## 2. Implementation Phases

### Phase 1: Commons Utilities Foundation

#### 1.1 `noise_utils.py` - Core Noise Utilities
**Priority:** High  
**Dependencies:** NumPy, Pandas, SciPy  
**Location:** `pamola_core/anonymization/commons/`

**Core Functionality:**
- `calculate_noise_impact()` - Measure noise effect on data utility
- `calculate_distribution_preservation()` - Analyze distribution changes
- `suggest_noise_range()` - Recommend noise levels based on SNR
- `validate_noise_bounds()` - Check if noise parameters are reasonable
- `create_secure_rng()` - Factory for secure random generators

**Key Implementation Details:**
```python
# Secure random number generation wrapper
class SecureRandomGenerator:
    """Thread-safe secure random generator."""
    def __init__(self, use_secure: bool = True, seed: Optional[int] = None):
        self._lock = threading.Lock()
        if use_secure:
            self._rng = secrets.SystemRandom()
        else:
            self._rng = np.random.default_rng(seed)
    
    def uniform(self, low: float, high: float, size: Optional[int] = None):
        """Generate uniform random values."""
        with self._lock:
            if isinstance(self._rng, secrets.SystemRandom):
                if size is None:
                    return self._rng.uniform(low, high)
                return np.array([self._rng.uniform(low, high) for _ in range(size)])
            else:
                return self._rng.uniform(low, high, size)
```

**Integration Points:**
- Use existing `metric_utils` for standard metrics
- Leverage `validation` framework for parameter validation
- Compatible with `DataWriter` for metrics export

#### 1.2 `statistical_utils.py` - Statistical Analysis
**Priority:** Medium  
**Dependencies:** Phase 1.1  
**Location:** `pamola_core/anonymization/commons/`

**Core Functionality:**
- `calculate_utility_metrics()` - Comprehensive utility preservation metrics
- `analyze_noise_distribution()` - Verify noise follows expected distribution
- `calculate_correlation_preservation()` - Multi-field correlation analysis
- `estimate_information_loss()` - Quantify privacy-utility tradeoff

**Key Implementation Details:**
- Focus on lightweight, fast calculations
- Support both single-field and multi-field analysis
- Provide interpretable metrics for non-technical users

### Phase 2: Base Noise Infrastructure

#### 2.1 Abstract Base Class (Optional for MVP)
**Priority:** Low (can be skipped for MVP)  
**Dependencies:** `base_anonymization_op.py`

**Consideration:** For MVP, directly inherit from `AnonymizationOperation` in each operation. Abstract base can be added later for code reuse.

### Phase 3: Uniform Numeric Noise Operation

#### 3.1 `uniform_numeric_op.py` - Numeric Noise Implementation
**Priority:** High  
**Dependencies:** Phase 1, `base_anonymization_op.py`  
**Location:** `pamola_core/anonymization/noise/`

**Core Implementation Components:**

1. **Constructor & Configuration**
   - Parameter validation using commons validators
   - Auto-detection of integer fields
   - Noise range normalization (symmetric/asymmetric)

2. **Noise Generation Engine**
   ```python
   def _initialize_generator(self):
       """Initialize random generator once per execution."""
       self._generator = SecureRandomGenerator(
           use_secure=self.use_secure_random,
           seed=self.random_seed
       )
   
   def _generate_noise_batch(self, size: int) -> np.ndarray:
       """Generate noise values for a batch."""
       if isinstance(self.noise_range, tuple):
           return self._generator.uniform(
               self.noise_range[0], 
               self.noise_range[1], 
               size
           )
       else:
           return self._generator.uniform(
               -self.noise_range, 
               self.noise_range, 
               size
           )
   ```

3. **Constraint Application**
   - Boundary enforcement (min/max)
   - Integer rounding with proper dtype preservation
   - Zero preservation logic
   - Statistical scaling implementation

4. **Batch Processing**
   - Efficient vectorized operations
   - Memory-conscious processing for large batches
   - Proper null handling per strategy

5. **Metrics Collection**
   - Signal-to-noise ratio calculation
   - Actual vs. configured noise analysis
   - Boundary violation counts
   - Performance metrics integration

**Key Features:**
- Support for both additive and multiplicative noise
- Automatic dtype preservation
- Configurable random generation (secure vs. fast)
- Comprehensive validation and error handling

### Phase 4: Uniform Temporal Noise Operation

#### 4.1 `uniform_temporal_op.py` - Temporal Noise Implementation
**Priority:** High  
**Dependencies:** Phase 1, Phase 3.1  
**Location:** `pamola_core/anonymization/noise/`

**Core Implementation Components:**

1. **Temporal Shift Generation**
   ```python
   def _calculate_total_shift_range(self) -> float:
       """Convert all time units to seconds."""
       total = 0
       if self.noise_range_days:
           total += self.noise_range_days * 86400
       if self.noise_range_hours:
           total += self.noise_range_hours * 3600
       if self.noise_range_minutes:
           total += self.noise_range_minutes * 60
       if self.noise_range_seconds:
           total += self.noise_range_seconds
       return total
   ```

2. **Datetime Handling**
   - Robust datetime parsing and validation
   - Timezone-aware operations
   - Support for various datetime formats

3. **Constraint System**
   - Boundary datetime enforcement
   - Weekend preservation algorithm
   - Special date preservation
   - Time-of-day preservation

4. **Granularity Control**
   - Efficient rounding to day/hour/minute/second
   - Maintain datetime validity after rounding

5. **Advanced Features**
   - Business day calculations for weekend preservation
   - Holiday calendar integration (post-MVP)
   - Directional bias (forward/backward only)

**Implementation Challenges:**
- Handle edge cases (leap years, DST transitions)
- Efficient weekend adjustment algorithm
- Memory-efficient processing of datetime objects

### Phase 5: Integration and Testing

#### 5.1 Package Initialization
**Priority:** High  
**Dependencies:** All operations implemented  
**Location:** `pamola_core/anonymization/noise/__init__.py`

```python
"""PAMOLA.CORE Noise Operations Package."""

from .uniform_numeric_op import UniformNumericNoiseOperation
from .uniform_temporal_op import UniformTemporalNoiseOperation

__all__ = [
    'UniformNumericNoiseOperation',
    'UniformTemporalNoiseOperation'
]

# Register operations with framework
from pamola_core.utils.ops.op_registry import register_operation
register_operation(UniformNumericNoiseOperation)
register_operation(UniformTemporalNoiseOperation)
```

#### 5.2 Integration Testing Suite
**Priority:** High  
**Dependencies:** All components complete

**Test Categories:**

1. **Unit Tests**
   - Noise generation correctness
   - Constraint enforcement
   - Statistical properties
   - Edge case handling

2. **Integration Tests**
   - Framework integration (DataWriter, metrics, etc.)
   - Large dataset processing
   - Memory usage validation
   - Progress tracking

3. **Statistical Tests**
   - Uniform distribution validation
   - Utility preservation metrics
   - Correlation analysis

4. **Performance Tests**
   - Benchmark different data sizes
   - Memory profiling
   - Concurrent processing

## 3. Key Implementation Considerations

### 3.1 Random Number Generation Strategy

```python
# Centralized RNG management
class NoiseOperationMixin:
    """Mixin for consistent RNG handling across noise operations."""
    
    def _get_random_generator(self) -> SecureRandomGenerator:
        """Get or create random generator."""
        if not hasattr(self, '_generator'):
            self._generator = SecureRandomGenerator(
                use_secure=self.use_secure_random,
                seed=self.random_seed
            )
        return self._generator
    
    def _cleanup_generator(self):
        """Clean up generator after execution."""
        if hasattr(self, '_generator'):
            del self._generator
```

### 3.2 Memory Optimization

```python
# Chunked noise generation for large datasets
def _apply_noise_chunked(self, series: pd.Series, chunk_size: int = 100000):
    """Apply noise in chunks to manage memory."""
    result = series.copy()
    
    for start in range(0, len(series), chunk_size):
        end = min(start + chunk_size, len(series))
        chunk_mask = slice(start, end)
        
        # Generate noise only for chunk
        noise = self._generate_noise_batch(end - start)
        
        # Apply to chunk
        result.iloc[chunk_mask] += noise
        
        # Force garbage collection for large chunks
        if chunk_size > 500000:
            gc.collect()
    
    return result
```

### 3.3 Validation Integration

```python
def validate_configuration(self):
    """Comprehensive configuration validation."""
    # Use commons validators
    if isinstance(self, UniformNumericNoiseOperation):
        # Validate numeric field
        validator = NumericFieldValidator(
            allow_null=(self.null_strategy != "ERROR")
        )
        result = validator.validate(self.df[self.field_name])
        if not result.is_valid:
            raise ValidationError(result.errors[0])
        
        # Validate noise parameters
        if self.noise_type == "multiplicative" and self.preserve_zero:
            self.logger.warning(
                "Multiplicative noise with preserve_zero may not preserve zeros effectively"
            )
```

### 3.4 Metrics Collection Pattern

```python
def _collect_comprehensive_metrics(self, original: pd.Series, 
                                 noisy: pd.Series) -> Dict[str, Any]:
    """Collect all metrics using framework utilities."""
    # Standard metrics via commons
    metrics = collect_operation_metrics(
        operation_type="noise_addition",
        original_data=original,
        processed_data=noisy,
        operation_params=self._get_params_dict(),
        timing_info={"start": self.start_time, "end": time.time()}
    )
    
    # Noise-specific metrics
    noise_metrics = self._collect_specific_metrics(original, noisy)
    metrics.update(noise_metrics)
    
    # Utility metrics via noise_utils
    utility = calculate_noise_impact(original, noisy)
    metrics["utility_preservation"] = utility
    
    # Distribution metrics if requested
    if self.calculate_distribution_metrics:
        dist_metrics = calculate_distribution_preservation(original, noisy)
        metrics["distribution"] = dist_metrics
    
    return metrics
```

## 4. Dependencies and Prerequisites

### 4.1 External Dependencies
- **NumPy**: Core numerical operations
- **Pandas**: DataFrame operations
- **SciPy**: Statistical tests and distributions
- **secrets**: Cryptographically secure random generation

### 4.2 Internal Dependencies
- `base_anonymization_op.py`: Base class functionality
- `op_field_utils.py`: Field naming and utilities
- `op_data_processing.py`: Batch processing and memory optimization
- `commons/validation/`: Field and parameter validation
- `commons/metric_utils.py`: Standard metrics collection
- `commons/visualization_utils.py`: Visualization generation

### 4.3 Optional Dependencies
- **matplotlib/plotly**: For enhanced visualizations (via visualization_utils)
- **numba**: For performance optimization (future enhancement)

## 5. Testing Strategy

### 5.1 Test Structure
```
tests/anonymization/noise/
├── test_uniform_numeric_op.py
├── test_uniform_temporal_op.py
├── test_noise_utils.py
├── test_statistical_utils.py
└── test_integration.py
```

### 5.2 Critical Test Cases

1. **Boundary Testing**
   ```python
   def test_output_bounds_enforcement():
       """Ensure output never exceeds specified bounds."""
       data = pd.Series([50.0] * 1000)
       op = UniformNumericNoiseOperation(
           field_name="value",
           noise_range=100,  # Large noise
           output_min=0,
           output_max=100
       )
       result = op.process_batch(pd.DataFrame({"value": data}))
       assert result["value"].min() >= 0
       assert result["value"].max() <= 100
   ```

2. **Distribution Testing**
   ```python
   def test_uniform_distribution_statistical():
       """Verify noise follows uniform distribution."""
       # Generate large sample
       # Apply Kolmogorov-Smirnov test
       # Check p-value > 0.05
   ```

3. **Preservation Testing**
   ```python
   def test_zero_preservation():
       """Verify zeros remain zeros when preserve_zero=True."""
       data = pd.Series([0, 10, 0, 20, 0])
       # Process and verify zeros unchanged
   ```

## 6. Performance Optimization Strategies

### 6.1 Vectorization
- Use NumPy operations instead of loops
- Batch random number generation
- Vectorized constraint application

### 6.2 Memory Management
- Process in chunks for large datasets
- In-place operations when mode="REPLACE"
- Explicit garbage collection for large operations

### 6.3 Caching Strategy
- Cache random generator per execution
- Cache statistical calculations (std, mean) for scaling
- Clear caches after execution

## 7. Error Handling Patterns

### 7.1 Graceful Degradation
```python
try:
    # Try secure random
    generator = secrets.SystemRandom()
except NotImplementedError:
    # Fall back to NumPy
    self.logger.warning("Secure random not available, using NumPy")
    generator = np.random.default_rng()
```

### 7.2 Informative Errors
```python
if self.output_min >= self.output_max:
    raise ValueError(
        f"Invalid output bounds: min ({self.output_min}) must be less than "
        f"max ({self.output_max})"
    )
```

## 8. Documentation Requirements

### 8.1 Module Documentation
- Comprehensive docstrings following NumPy style
- Usage examples in docstrings
- Mathematical formulas for noise generation

### 8.2 User Guide
- Basic usage examples
- Parameter selection guidelines
- Privacy-utility tradeoff discussion

## 9. Future Extension Points

### 9.1 Differential Privacy
- Abstract base class for DP mechanisms
- Sensitivity calculation framework
- Privacy budget tracking

### 9.2 Additional Noise Types
- Laplace mechanism
- Gaussian mechanism
- Exponential mechanism
- Custom distribution support

### 9.3 Advanced Features
- Correlated noise for multiple fields
- Adaptive noise based on local density
- Query-specific noise calibration

## 10. Success Criteria

### 10.1 Functional Requirements
- [ ] Both operations process 100K records in <5 seconds
- [ ] Noise follows specified distribution (statistical tests pass)
- [ ] All constraints are enforced (bounds, preservation rules)
- [ ] Metrics are accurately calculated
- [ ] Integration with framework is seamless

### 10.2 Quality Requirements
- [ ] >90% test coverage
- [ ] No memory leaks in large dataset processing
- [ ] Thread-safe operations
- [ ] Comprehensive error messages
- [ ] Clear documentation with examples

### 10.3 Performance Requirements
- [ ] <10% overhead compared to simple NumPy operations
- [ ] Memory usage scales linearly with data size
- [ ] Efficient batch processing (>1M records/second for numeric)