"""
File: test_full_masking_op.py
Test Target: masking/full_masking_op.py
Version: 1.0
Coverage Status: Partial (codebase limitation)
Last Updated: 2025-07-28
"""

import pytest
import pandas as pd
from pathlib import Path
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

# Helper to create a DataSource from a DataFrame
def make_data_source(df):
    return DataSource(dataframes={"main": df})

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "ssn": ["123-45-6789", "987-65-4321", "555-55-5555"],
        "salary": [1000, 2000, 3000],
        "date": ["2021-01-01", "2021-02-01", "2021-03-01"]
    })

@pytest.fixture
def task_dir(tmp_path):
    return tmp_path

@pytest.fixture
def reporter():
    class DummyReporter:
        def add_operation(self, *args, **kwargs):
            pass
    return DummyReporter()

# Test: error on invalid field name
def test_full_masking_invalid_field(sample_df, task_dir, reporter):
    field_name = "not_a_field"
    use_encryption = False
    op = FullMaskingOperation(field_name=field_name, use_encryption=use_encryption)
    op.preset_type = None
    op.preset_name = None
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.ERROR

# Test: error on invalid mask_char
def test_full_masking_invalid_mask_char(sample_df, task_dir, reporter):
    field_name = "name"
    mask_char = "**"
    use_encryption = False
    op = FullMaskingOperation(field_name=field_name, mask_char=mask_char, use_encryption=use_encryption)
    op.preset_type = None
    op.preset_name = None
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.ERROR

# Test: get_operation_summary returns expected keys
def test_get_operation_summary_keys():
    field_name = "name"
    use_encryption = False
    op = FullMaskingOperation(field_name=field_name, use_encryption=use_encryption)
    op.preset_type = None
    op.preset_name = None
    summary = op.get_operation_summary()
    assert "field_name" in summary and "mask_character" in summary

# Test: valid masking on string field
def test_full_masking_valid_string_field(sample_df, task_dir, reporter):
    field_name = "name"
    mask_char = "*"
    use_encryption = False
    op = FullMaskingOperation(field_name=field_name, mask_char=mask_char, use_encryption=use_encryption)
    op.preset_type = None
    op.preset_name = None
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    # Load output artifact
    output_artifacts = result.get_artifacts_by_type("csv")
    assert output_artifacts, "No output artifact of type 'csv' found."
    masked_df = pd.read_csv(output_artifacts[0].path)
    # Validate all values in the field are fully masked (same length, all mask_char)
    # Accept either all values are the same masked string (fixed length), or each value is masked to the length of the original
    masked_values = masked_df[field_name].tolist()
    # If all values are the same, accept (fixed-length masking)
    if all(masked_values[0] == v for v in masked_values):
        assert set(masked_values[0]) == {mask_char}, f"Expected only '{mask_char}' in masked value, got {masked_values[0]}"
    else:
        for orig, masked in zip(sample_df[field_name], masked_values):
            assert masked == mask_char * len(str(orig)), f"Expected {mask_char * len(str(orig))}, got {masked}"
    # Validate artifact metadata
    assert output_artifacts[0].category == "output"
    assert output_artifacts[0].description.lower().startswith(field_name)

# Test: valid masking on numeric field
def test_full_masking_valid_numeric_field(sample_df, task_dir, reporter):
    field_name = "salary"
    mask_char = "#"
    use_encryption = False
    op = FullMaskingOperation(field_name=field_name, mask_char=mask_char, use_encryption=use_encryption)
    op.preset_type = None
    op.preset_name = None
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    output_artifacts = result.get_artifacts_by_type("csv")
    assert output_artifacts, "No output artifact of type 'csv' found."
    masked_df = pd.read_csv(output_artifacts[0].path)
    # Numeric fields may be masked as strings of mask_char or a fixed value
    for orig, masked in zip(sample_df[field_name], masked_df[field_name]):
        masked_str = str(masked)
        # Accept either all mask_char or a fixed value (all values equal)
        assert masked_str == mask_char * len(str(orig)) or all(masked_df[field_name][0] == v for v in masked_df[field_name])
    assert output_artifacts[0].category == "output"
    assert output_artifacts[0].description.lower().startswith(field_name)

# Test: valid masking on date field
def test_full_masking_valid_date_field(sample_df, task_dir, reporter):
    field_name = "date"
    mask_char = "X"
    use_encryption = False
    op = FullMaskingOperation(field_name=field_name, mask_char=mask_char, use_encryption=use_encryption)
    op.preset_type = None
    op.preset_name = None
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    output_artifacts = result.get_artifacts_by_type("csv")
    assert output_artifacts, "No output artifact of type 'csv' found."
    masked_df = pd.read_csv(output_artifacts[0].path)
    # Dates may be masked as strings of Xs or a fixed value
    for orig, masked in zip(sample_df[field_name], masked_df[field_name]):
        assert masked == mask_char * len(str(orig)) or all(masked_df[field_name][0] == v for v in masked_df[field_name])
    assert output_artifacts[0].category == "output"
    assert output_artifacts[0].description.lower().startswith(field_name)
