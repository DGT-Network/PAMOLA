# PAMOLA.CORE Anonymization Module Implementation Checklist

## Architecture and Inheritance

- [ ] Module extends `BaseOperation` from `pamola_core/utils/ops/op_base.py` via `AnonymizationOperation`
- [ ] Constructor properly initializes base class with required parameters
- [ ] Module implements required abstract methods:
  - [ ] `process_batch(self, batch: pd.DataFrame)`
  - [ ] `process_value(self, value, **params)`
  - [ ] `_collect_specific_metrics(self, original_data, anonymized_data)`
- [ ] Module overrides the `run()` method to properly handle direct DataFrame modification

## Code Organization and Style

- [ ] All documentation and comments are in English
- [ ] Module has comprehensive docstrings with parameter descriptions
- [ ] Code follows PEP 8 style guidelines (line length, naming conventions)
- [ ] Type hints are used for all method parameters and return values
- [ ] Complex operations have inline comments explaining logic
- [ ] Maximum file length is ~700 lines for maintainability

## Pamola Core Utilities Integration

- [ ] File I/O operations use `pamola_core/utils/io.py` utilities
- [ ] Visualization generation uses `pamola_core/utils/visualization.py` utilities only
- [ ] Progress reporting uses `pamola_core/utils/progress.py` for long-running operations
- [ ] Logging uses `pamola_core/utils/logging.py` with appropriate log levels
- [ ] DataFrame handling uses `DataSource` from `pamola_core/utils/ops/op_data_source.py`
- [ ] Results are returned as `OperationResult` from `pamola_core/utils/ops/op_result.py`
- [ ] Caching support uses `pamola_core/utils/ops/op_cache.py` when appropriate

## Error Handling and Input Validation

- [ ] Parameter validation occurs before processing starts
  - [ ] Field existence is verified before processing
  - [ ] Data types are validated appropriately 
  - [ ] Strategy parameters are validated (bin_count, precision, range_limits)
- [ ] Non-numeric fields are detected and handled gracefully
- [ ] Null values are handled according to the null_strategy parameter
- [ ] All errors are captured in the `OperationResult` with appropriate error codes
- [ ] Critical errors are logged using the pamola core logging module
- [ ] Specific exception classes are used for different error types

## Metrics and Visualization

- [ ] Basic metrics are collected for all anonymization operations
- [ ] Strategy-specific metrics are collected based on the generalization approach
- [ ] Metrics are properly structured and saved as JSON
- [ ] Visualizations show "before and after" effects of anonymization
- [ ] Visualization files use consistent naming conventions
- [ ] All visualizations are created using only `pamola_core/utils/visualization.py` functions
- [ ] Metrics and visualizations are correctly added to both result and reporter

## Large Data Processing

- [ ] Support for chunked processing through batch_size parameter
- [ ] Memory usage is optimized for large datasets
- [ ] Progress tracking is implemented for long-running operations
- [ ] Graceful degradation is implemented when processing very large datasets

## Testing Requirements 

- [ ] Unit tests cover all functionality and edge cases
- [ ] Tests handle non-numeric fields that were already processed
- [ ] Tests verify correct error conditions (like null_strategy="ERROR")
- [ ] Tests validate metrics and visualization generation
- [ ] Tests check both REPLACE and ENRICH modes
- [ ] Test-specific warnings (like Kaleido deprecation warnings) are suppressed at the test level

## Lessons Learned from Testing

- [ ] When modifying DataFrames directly, ensure changes are visible to tests by operating on the original DataFrame
- [ ] Add special handling for already-processed fields to avoid validation errors in subsequent operations
- [ ] Artifacts must be added to both the result object AND the reporter object
- [ ] When validating data types, check field type before processing to avoid type conversion errors
- [ ] For test-specific warnings (like Kaleido), use pytest's filterwarnings mechanism rather than modifying source code
- [ ] When handling null values, verify null counts match expectations before and after processing
- [ ] Track metrics properly even when using direct DataFrame modification
- [ ] Ensure visualization paths are properly constructed relative to task_dir

## Security and Privacy

- [ ] Output files are properly secured with encryption support
- [ ] Sensitive intermediate data is not unnecessarily stored
- [ ] Privacy metrics are calculated to evaluate effectiveness of anonymization
- [ ] Field has proper masking/generalization for expected privacy outcomes
- [ ] Validate generalization is applied consistently across all records

## Operation Behavior Requirements

- [ ] ENRICH mode creates a new field with the generalized values
- [ ] REPLACE mode modifies the original field with generalized values
- [ ] Non-processed values are properly preserved (e.g., nulls with PRESERVE strategy)
- [ ] Correctly handles numeric fields with different ranges and distributions
- [ ] Provides factory function for easy operation creation