"""DATA-01: an operation never writes into the DataFrame the caller handed over.

``DataSource`` stores and returns the caller's object itself. Three operations
(``NumericGeneralizationOperation``, ``UniformNumericNoiseOperation``,
``AddOrModifyFieldsOperation``) assigned columns on that object, so after
``execute()`` the caller's frame had different dtypes (and, for
``AddOrModifyFields``, different columns and values). The contract now lives in
the transform bases' ``_validate_and_get_dataframe``; these tests pin it both for
the three reported operations and for the contract itself.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.generalization.numeric_op import (
    NumericGeneralizationOperation,
)
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.noise.uniform_numeric_op import (
    UniformNumericNoiseOperation,
)
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.fake_data import FakeNameOperation
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.transformations.field_ops.add_modify_fields import (
    AddOrModifyFieldsOperation,
)
from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.arange(20, 120, dtype=float),  # float64: dtype normalisation target
            "other": np.arange(100),
            "name": ["Alice", "Bob", "Carol", "Dave"] * 25,
        }
    )


_ADD_MODIFY = {
    "new_field": {"operation_type": "add_constant", "constant_value": 42},
    "age": {"operation_type": "modify_constant", "constant_value": 7},
}

# The first three are the confirmed offenders; the rest guard the contract on
# every transform base (anonymization, transformation, generator) and on ENRICH.
_CASES = {
    "numeric_generalization": lambda: NumericGeneralizationOperation(
        field_name="age", strategy="binning", bin_count=4
    ),
    "uniform_numeric_noise": lambda: UniformNumericNoiseOperation(
        field_name="age", noise_range=2.0, noise_type="additive"
    ),
    "add_or_modify_fields": lambda: AddOrModifyFieldsOperation(
        field_operations=_ADD_MODIFY, field_name="age", mode="REPLACE"
    ),
    "numeric_generalization_enrich": lambda: NumericGeneralizationOperation(
        field_name="age", strategy="binning", bin_count=4, mode="ENRICH"
    ),
    "uniform_numeric_noise_enrich": lambda: UniformNumericNoiseOperation(
        field_name="age", noise_range=2.0, noise_type="additive", mode="ENRICH"
    ),
    "full_masking": lambda: FullMaskingOperation(field_name="name"),
    "cell_suppression": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null"
    ),
    "remove_fields": lambda: RemoveFieldsOperation(fields_to_remove=["other"]),
    "fake_name": lambda: FakeNameOperation(field_name="name"),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_execute_leaves_the_callers_dataframe_untouched(case, tmp_path):
    df = _frame()
    reference = df.copy(deep=True)
    original_id = id(df)
    data_source = DataSource(dataframes={"main": df})

    result = _CASES[case]().execute(data_source, tmp_path, None)

    assert result.status == OperationStatus.SUCCESS, result.error_message
    # Identity: the source still holds the very object the caller passed in ...
    assert data_source.dataframes["main"] is df
    assert id(df) == original_id
    # ... and deep equality: same columns, same order, same dtypes, same values.
    pd.testing.assert_frame_equal(df, reference, check_exact=True)


@pytest.mark.parametrize(
    ("base", "make_op"),
    [
        (AnonymizationOperation, lambda: FullMaskingOperation(field_name="name")),
        (
            TransformationOperation,
            lambda: RemoveFieldsOperation(fields_to_remove=["other"]),
        ),
        (GeneratorOperation, lambda: FakeNameOperation(field_name="name")),
    ],
    ids=["anonymization", "transformation", "generator"],
)
def test_transform_bases_hand_out_a_private_frame(base, make_op):
    """The contract itself: what an operation loads is never the caller's frame."""
    op = make_op()
    assert isinstance(op, base)
    df = _frame()
    reference = df.copy(deep=True)
    data_source = DataSource(dataframes={"main": df})

    loaded = op._validate_and_get_dataframe(data_source, "main")

    assert loaded is not df
    assert not np.shares_memory(loaded["other"].to_numpy(), df["other"].to_numpy())
    # In-place value writes and column rebinding must both stay private.
    loaded.loc[loaded["other"] < 10, "other"] = -1
    loaded["age"] = loaded["age"].astype("Int64")
    loaded["extra"] = 1
    pd.testing.assert_frame_equal(df, reference, check_exact=True)
