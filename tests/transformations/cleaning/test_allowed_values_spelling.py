"""ASB-OPS-13: how an ``allowed_values`` / ``disallowed_values`` constraint is applied.

The constraint is applied to a field that is categorical: either the column has the pandas
``category`` dtype or the field is declared ``"data_type": "categorical"``. A text column declared
``"data_type": "string"`` is NOT covered by these two constraint types (the probe that suggested a
no-op used ``"string"``). Invalid values are nullified; the replacement of nulls is the separate
``null_replacement`` parameter (the constraint has no ``replacement_*`` keys).
"""

import numpy as np
import pandas as pd

from pamola_core.transformations.cleaning.clean_invalid_values import CleanInvalidValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"city": rng.choice(["Paris", "Rome", "Oslo", "Kyiv"], 400)})


def _run(tmp_path, constraint, frame):
    op = CleanInvalidValuesOperation(field_constraints={"city": constraint}, use_cache=False)
    result = op.execute(DataSource(dataframes={"main": frame}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    return pd.read_csv(next((tmp_path / "output").glob("*_output_*.csv")))["city"]


def test_allowed_values_nullifies_everything_else_for_a_categorical_field(tmp_path):
    frame = _frame()
    out = _run(tmp_path, {"data_type": "categorical", "constraint_type": "allowed_values",
                          "allowed_values": ["Paris", "Rome"]}, frame)
    assert set(out.dropna()) == {"Paris", "Rome"}
    assert out.isna().sum() == frame["city"].isin(["Oslo", "Kyiv"]).sum() > 0


def test_allowed_values_applies_to_the_pandas_category_dtype_without_a_data_type(tmp_path):
    frame = _frame().astype("category")
    out = _run(tmp_path, {"constraint_type": "allowed_values", "allowed_values": ["Paris"]}, frame)
    assert set(out.dropna()) == {"Paris"}


def test_disallowed_values_nullifies_just_those(tmp_path):
    frame = _frame()
    out = _run(tmp_path, {"data_type": "categorical", "constraint_type": "disallowed_values",
                          "disallowed_values": ["Oslo"]}, frame)
    assert "Oslo" not in set(out.dropna())
    assert out.isna().sum() == (frame["city"] == "Oslo").sum() > 0


def test_a_string_declared_field_is_not_touched_by_allowed_values(tmp_path):
    """Pins the current, documented limit: 'string' is not 'categorical'."""
    frame = _frame()
    out = _run(tmp_path, {"data_type": "string", "constraint_type": "allowed_values",
                          "allowed_values": ["Paris"]}, frame)
    assert out.isna().sum() == 0 and set(out) == set(frame["city"])


def test_invalid_values_are_replaced_with_null_replacement(tmp_path):
    frame = _frame()
    op = CleanInvalidValuesOperation(
        field_constraints={"city": {"data_type": "categorical", "constraint_type": "allowed_values",
                                    "allowed_values": ["Paris"]}},
        null_replacement={"city": "INVALID"}, use_cache=False,
    )
    result = op.execute(DataSource(dataframes={"main": frame}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    out = pd.read_csv(next((tmp_path / "output").glob("*_output_*.csv")))["city"]
    assert set(out) == {"Paris", "INVALID"}
