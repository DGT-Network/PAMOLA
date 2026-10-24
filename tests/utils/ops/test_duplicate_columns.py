"""DATA-03: duplicate column names are rejected with a typed, actionable error.

With repeated column names ``df["name"]`` is a DataFrame, not a Series, and every
operation failed somewhere downstream with an unrelated pandas message
(``'DataFrame' object has no attribute 'dtype'``, "truth value of a DataFrame is
ambiguous"...). The check lives in ``load_data_operation`` - every operation loads
its input there - so it is one check, not one per operation.
"""

import pandas as pd
import pytest

from pamola_core.anonymization.generalization.numeric_op import (
    NumericGeneralizationOperation,
)
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.errors.exceptions import DuplicateColumnError, ValidationError
from pamola_core.fake_data import FakeNameOperation
from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
from pamola_core.utils.io import load_data_operation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _duplicated() -> pd.DataFrame:
    df = pd.DataFrame(
        {"name": list("abcd") * 5, "age": range(20), "other": range(20)}
    )
    df.columns = ["name", "age", "age"]  # 'age' now occurs twice
    return df


_CASES = {
    "full_masking": lambda: FullMaskingOperation(field_name="name"),
    "numeric_generalization": lambda: NumericGeneralizationOperation(
        field_name="age", strategy="binning", bin_count=3
    ),
    "cell_suppression": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null"
    ),
    "remove_fields": lambda: RemoveFieldsOperation(fields_to_remove=["name"]),
    "fake_name": lambda: FakeNameOperation(field_name="name"),
    "numeric_profile": lambda: NumericOperation("age", use_cache=False),
}


def test_loader_raises_a_typed_error_naming_the_columns():
    source = DataSource(dataframes={"main": _duplicated()})
    with pytest.raises(DuplicateColumnError) as excinfo:
        load_data_operation(source, "main")
    assert isinstance(excinfo.value, ValidationError)
    message = str(excinfo.value)
    assert "'age'" in message
    assert "Rename or drop" in message
    assert excinfo.value.details["duplicated_columns"] == ["age"]


def test_loader_accepts_unique_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert load_data_operation(DataSource(dataframes={"main": df}), "main") is df


def test_loader_rejects_a_bare_dataframe_too():
    with pytest.raises(DuplicateColumnError):
        load_data_operation(_duplicated(), "main")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_operations_report_the_duplicate_instead_of_a_pandas_error(case, tmp_path):
    result = _CASES[case]().execute(
        DataSource(dataframes={"main": _duplicated()}), tmp_path, None
    )

    assert result.status == OperationStatus.ERROR
    assert "Duplicate column names" in result.error_message, result.error_message
    assert "'age'" in result.error_message
    assert "dtype" not in result.error_message
    assert "truth value" not in result.error_message


def test_metrics_operation_reports_the_duplicate(tmp_path):
    original = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    transformed = pd.concat([original, original], axis=1)  # 'x' twice
    result = FidelityOperation(fidelity_metrics=["ks"], columns=["x"]).execute(
        DataSource(dataframes={"main": original, "synth": transformed}),
        tmp_path,
        None,
        original_dataset_name="main",
        transformed_dataset_name="synth",
    )
    assert result.status == OperationStatus.ERROR
    assert "Duplicate column names" in result.error_message
