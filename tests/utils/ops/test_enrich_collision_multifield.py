"""CDX-AC-10: the ENRICH guard covers operations that write several columns.

The first version of the guard returned when ``field_name`` was empty, so
multi-field operations (per-field strategy tables) still overwrote existing
``<prefix><field>`` columns silently. Operations now declare their targets with
``get_enrich_targets`` and every one is validated before any work is done.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.errors.exceptions import OutputFieldExistsError
from pamola_core.transformations.cleaning.clean_invalid_values import (
    CleanInvalidValuesOperation,
)
from pamola_core.transformations.field_ops.add_modify_fields import (
    AddOrModifyFieldsOperation,
)
from pamola_core.transformations.imputation.impute_missing_values import (
    ImputeMissingValuesOperation,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frame():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, np.nan, 40.0],
            "_a": [99.0, 98.0, 97.0, 96.0],  # an earlier output that must survive
        }
    )


_CASES = {
    "impute": (
        lambda **kw: ImputeMissingValuesOperation(
            field_strategies={
                "a": {"data_type": "numeric", "imputation_strategy": "mean"},
                "b": {"data_type": "numeric", "imputation_strategy": "mean"},
            },
            mode="ENRICH",
            column_prefix="_",
            **kw,
        ),
        "_a",
    ),
    "clean": (
        lambda **kw: CleanInvalidValuesOperation(
            field_constraints={
                "a": {"data_type": "numeric", "constraint_type": "min_value", "min_value": 2},
                "b": {"data_type": "numeric", "constraint_type": "min_value", "min_value": 15},
            },
            mode="ENRICH",
            column_prefix="_",
            **kw,
        ),
        "_a",
    ),
    "add_modify": (
        lambda **kw: AddOrModifyFieldsOperation(
            field_operations={"a": {"operation_type": "modify_constant", "constant_value": 5}},
            mode="ENRICH",
            column_prefix="_",
            **kw,
        ),
        "_a",
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_multi_field_operation_fails_before_touching_an_existing_target(case, tmp_path):
    make, target = _CASES[case]
    df = _frame()
    reference = df.copy(deep=True)

    result = make().execute(DataSource(dataframes={"main": df}), tmp_path, None)

    assert result.status == OperationStatus.ERROR
    assert f"'{target}'" in result.error_message
    assert "overwrite_existing_output=True" in result.error_message
    pd.testing.assert_frame_equal(df, reference)  # and the caller's frame is intact


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_option_lets_a_multi_field_operation_overwrite(case, tmp_path):
    make, target = _CASES[case]
    result = make(overwrite_existing_output=True).execute(
        DataSource(dataframes={"main": _frame()}), tmp_path, None
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    written = pd.read_csv(next((tmp_path / "output").glob("*_output_*.csv")))
    assert target in written.columns
    assert list(written[target]) != [99.0, 98.0, 97.0, 96.0]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_free_targets_are_unaffected(case, tmp_path):
    make, _ = _CASES[case]
    result = make().execute(
        DataSource(dataframes={"main": _frame().drop(columns=["_a"])}), tmp_path, None
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message


def test_a_collision_on_a_later_field_is_found_too(tmp_path):
    """Not only the first field: '_b' exists, '_a' does not."""
    df = _frame().drop(columns=["_a"])
    df["_b"] = 0.0
    make, _ = _CASES["impute"]
    result = make().execute(DataSource(dataframes={"main": df}), tmp_path, None)
    assert result.status == OperationStatus.ERROR
    assert "'_b'" in result.error_message


def test_declared_targets():
    df = _frame()
    make, _ = _CASES["impute"]
    assert make().get_enrich_targets(df) == ["_a", "_b"]
    # fields that are not in the frame are not written, so not targets
    op = ImputeMissingValuesOperation(
        field_strategies={"zzz": {"imputation_strategy": "mean"}}, mode="ENRICH", column_prefix="_"
    )
    assert op.get_enrich_targets(df) == []


def test_error_type_is_shared_with_the_single_field_guard():
    assert issubclass(OutputFieldExistsError, Exception)
