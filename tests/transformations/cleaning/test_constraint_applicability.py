"""ASB-OPS-14: a constraint that cannot apply is rejected, never silently skipped.

``process_batch`` applies each constraint type to one kind of field (numeric, categorical, date,
text). A constraint outside its kind, an unknown or missing ``constraint_type`` and missing
parameters used to produce SUCCESS with the data unchanged. They now raise
``InvalidParameterError`` naming the field and the fix, before any data is processed.
"""

import pandas as pd
import pytest

from pamola_core.errors.exceptions import InvalidParameterError
from pamola_core.transformations.cleaning.clean_invalid_values import CleanInvalidValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frame():
    return pd.DataFrame({
        "city": ["Paris", "Rome", "Oslo"] * 4,
        "age": list(range(12)),
        "when": pd.date_range("2020-01-01", periods=12),
    })


def _validate(constraints):
    op = CleanInvalidValuesOperation(field_constraints=constraints, use_cache=False)
    op._validate_constraints(_frame())


@pytest.mark.parametrize(("constraints", "must_name"), [
    ({"city": {"constraint_type": "allowed_values", "allowed_values": ["Paris"]}}, ['"data_type": "categorical"', "'city'"]),
    ({"city": {"data_type": "string", "constraint_type": "disallowed_values", "disallowed_values": ["Rome"]}}, ['"data_type": "categorical"']),
    ({"city": {"constraint_type": "min_value", "min_value": 3}}, ['"data_type": "numeric"', "'city'"]),
    ({"age": {"constraint_type": "pattern", "pattern": "^1"}}, ['"data_type": "categorical"', "'age'"]),
    ({"age": {"constraint_type": "min_length", "min_length": 2}}, ['"data_type": "text"']),
    ({"city": {"constraint_type": "min_date", "min_date": "2020-01-05"}}, ['"data_type": "date"']),
    ({"city": {"constraint_type": "allowed_value", "allowed_values": ["Paris"]}}, ["no valid constraint_type", "allowed_values"]),
    ({"age": {"min": 0, "max": 5}}, ["no valid constraint_type", "'age'"]),
    ({"age": {"constraint_type": "min_value"}}, ["needs 'min_value'", "'age'"]),
    ({"age": {"constraint_type": "valid_range", "min_value": 1}}, ["needs 'max_value'"]),
    ({"city": {"data_type": "categorical", "constraint_type": "allowed_values"}}, ["needs 'allowed_values'"]),
])
def test_a_constraint_that_cannot_apply_is_rejected_with_the_fix(constraints, must_name):
    with pytest.raises(InvalidParameterError) as info:
        _validate(constraints)
    message = str(info.value)
    for text in must_name:
        assert text in message, message


@pytest.mark.parametrize("constraints", [
    {"age": {"constraint_type": "min_value", "min_value": 3}},
    {"age": {"constraint_type": "valid_range", "min_value": 1, "max_value": 9}},
    {"city": {"data_type": "categorical", "constraint_type": "allowed_values", "allowed_values": ["Paris"]}},
    {"city": {"data_type": "categorical", "constraint_type": "pattern", "pattern": "^P"}},
    {"city": {"constraint_type": "min_length", "min_length": 2}},
    {"city": {"constraint_type": "valid_pattern", "valid_pattern": "^P"}},
    {"when": {"constraint_type": "min_date", "min_date": "2020-01-05"}},
    {"when": {"constraint_type": "valid_format", "valid_format": "%Y-%m-%d"}},
    {"city": {"constraint_type": "valid_format", "valid_format": "%Y-%m-%d"}},
    {"missing_field": {"constraint_type": "min_value", "min_value": 3}},      # absent field: unchanged behaviour
])
def test_constraints_that_apply_are_accepted(constraints):
    _validate(constraints)


def test_the_error_reaches_the_caller_as_a_failed_result_before_any_output(tmp_path):
    op = CleanInvalidValuesOperation(
        field_constraints={"city": {"constraint_type": "allowed_values", "allowed_values": ["Paris"]}},
        use_cache=False,
    )
    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
    assert result.status == OperationStatus.ERROR
    assert '"data_type": "categorical"' in result.error_message
    assert not list((tmp_path / "output").glob("*.csv")) if (tmp_path / "output").exists() else True
