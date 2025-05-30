# PAMOLA.CORE Operation Templates

This directory contains templates and examples to help you quickly create new operations
that follow the PAMOLA.CORE standards and best practices.

## Contents

- `operation_skeleton.py` - Boilerplate code for creating a new operation
- `config_example.json` - Example configuration file with schema annotations
- `README.md` - This file with usage instructions

## Creating a New Operation

1. **Copy the skeleton**: 
   ```bash
   cp pamola_core/utils/ops/templates/operation_skeleton.py pamola_core/[your_module_path]/[your_operation_name].py
   ```

2. **Customize the skeleton**:
   - Replace `MyOperation` with your operation name (e.g., `GeneralizeNumericOperation`)
   - Replace `MyOperationConfig` with your operation config name (e.g., `GeneralizeNumericConfig`)
   - Update docstrings and class-level comments
   - Replace TODO markers with your business logic
   - Update parameters in `__init__` method to match your operation's needs
   - Implement your processing logic in the `execute()` method

3. **Follow the operation lifecycle phases**:
   1. Initialize result objects and writer
   2. Save operation configuration
   3. Set up progress tracking
   4. Load input data
   5. Validate inputs
   6. Process data (your business logic)
   7. Calculate metrics
   8. Write outputs
   9. Register artifacts with the result

4. **Update schema validation**:
   - Modify the `schema` in the config class to match your parameters
   - Add required fields to the `required` list
   - Add validation constraints (min/max/enum) for parameters

## Using Configuration Files

The `config_example.json` provides a template for operation configurations. When creating configurations:

1. **Basic structure**:
   ```json
   {
     "operation_name": "your_operation_name",
     "version": "1.0.0",
     "parameters": {
       "param1": "value1",
       "param2": 123
     }
   }
   ```

2. **Required fields**:
   - `operation_name`: String identifier for your operation
   - `version`: Semantic version string (MAJOR.MINOR.PATCH)
   - Any other parameters required by your operation

3. **Common parameter types**:
   - Field/column names (strings)
   - Thresholds (floats between 0-1)
   - Counts/limits (integers)
   - Method selectors (enum strings)
   - Output path modifiers (prefixes/suffixes)

## Testing Your Operation

Use the provided test helpers from `pamola_core/utils/ops/op_test_helpers.py` to write effective unit tests:

1. **Create test environment**:
   ```python
   def test_my_operation(tmp_path):
       # Create test environment
       task_dir, config = create_test_operation_env(
           tmp_path, 
           {"parameters": {"column_name": "test_col", "threshold": 0.5}}
       )
       
       # Create test data
       test_df = pd.DataFrame({"test_col": [1, 2, 3, 4, 5]})
       data_source = MockDataSource.from_dataframe(test_df)
       
       # Create operation instance
       operation = MyOperation(
           column_name="test_col",
           threshold=0.5
       )
       
       # Execute operation with stub writer
       result = operation.run(
           data_source=data_source,
           task_dir=task_dir,
           reporter=None
       )
       
       # Assert results
       assert result.status == OperationStatus.SUCCESS
       assert_artifact_exists(task_dir, "output", r"processed_data\.csv")
       assert_metrics_content(task_dir, {"row_count": 5})
   ```

2. **Key test helpers**:
   - `MockDataSource`: Provides in-memory DataFrames without I/O
   - `StubDataWriter`: Records write operations and writes to a temp directory
   - `assert_artifact_exists()`: Verifies that a file was created
   - `assert_metrics_content()`: Verifies metrics content with partial matching
   - `create_test_operation_env()`: Sets up a test task directory with config

3. **Run tests** with:
   ```bash
   pytest tests/your_operation_test.py -v
   ```

## Best Practices

1. **Error handling**: Use try/except and return appropriate error status
2. **Progress tracking**: Update the progress tracker at key phases
3. **Metrics**: Calculate and store relevant statistics about the operation
4. **Artifacts**: Register all output files as artifacts with descriptions
5. **Logging**: Use `self.logger` to record important events and errors
6. **Parameter validation**: Validate inputs before processing
7. **Config saving**: Always call `self.save_config(task_dir)` before processing

## Troubleshooting

- If imports fail, ensure your file is in the correct module structure
- If registration fails, check that `register_operation()` is called after the class definition
- For testing issues, ensure `op_test_helpers.py` is in your Python path
- Use logging with `DEBUG` level to trace operation execution