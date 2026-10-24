"""ASB-OPS-8: operations that assign NEW labels work on the "category" columns the engine creates.

The memory optimization turns low-cardinality text columns of frames above 10,000 rows into
``category``. Assigning a label that is not yet a category raised "Cannot setitem on a
Categorical"; ``CellSuppression(constant)`` failed that way and ``ImputeMissingValues`` assumed
the ``.cat`` accessor on any column declared categorical. ``identify_rare_categories`` returned
"no rare categories" on any exception, which makes a caller change nothing.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.anonymization.commons.category_utils import identify_rare_categories
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _big_frame(n=12_000):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"city": rng.choice(["Paris", "Rome", "Oslo"], n)})
    frame.loc[::10, "city"] = None
    return frame


def _output(tmp_path):
    return pd.read_csv(next((tmp_path / "output").glob("*_output_*.csv")))


@pytest.mark.parametrize("n", [500, 12_000])
def test_cell_suppression_constant_works_on_any_size(n, tmp_path):
    frame = _big_frame(n)
    op = CellSuppressionOperation(field_name="city", suppression_strategy="constant",
                                  suppression_value="SUPPRESSED", use_cache=False,
                                  condition_field="city", condition_values=["Oslo"], condition_operator="in")
    result = op.execute(DataSource(dataframes={"main": frame}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    out = _output(tmp_path)
    assert (out["city"] == "SUPPRESSED").sum() == (frame["city"] == "Oslo").sum() > 0
    assert "Oslo" not in set(out["city"].dropna())


@pytest.mark.parametrize("n", [500, 12_000])
def test_imputation_of_a_categorical_field_works_on_any_size(n, tmp_path):
    frame = _big_frame(n)
    op = ImputeMissingValuesOperation(
        field_strategies={"city": {"data_type": "categorical", "imputation_strategy": "constant",
                                   "constant_value": "UNKNOWN"}},
        use_cache=False,
    )
    result = op.execute(DataSource(dataframes={"main": frame}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    out = _output(tmp_path)
    assert (out["city"] == "UNKNOWN").sum() == frame["city"].isna().sum()


def test_identify_rare_categories_raises_instead_of_answering_nothing():
    with pytest.raises(Exception):
        identify_rare_categories(pd.Series(["a", "b"]), value_counts="not a series")
