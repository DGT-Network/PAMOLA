"""DATA-04: an existing ENRICH target is never silently overwritten (SRS 6.8).

Before, every operation logged a WARNING and overwrote the column. Now the
operation fails with ``OutputFieldExistsError`` unless ``overwrite_existing_output``
is set. The error names columns only, never values.
"""

import logging

import pandas as pd
import pytest

from pamola_core.anonymization.generalization.numeric_op import (
    NumericGeneralizationOperation,
)
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.noise.uniform_numeric_op import (
    UniformNumericNoiseOperation,
)
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.errors.exceptions import OutputFieldExistsError, ValidationError
from pamola_core.fake_data import FakeNameOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

SECRET = "TOPSECRET-VALUE"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [SECRET, "Bob", "Carol", "Dave"] * 5,
            "age": [float(v) for v in range(20, 40)],
            "_name": ["old"] * 20,  # what ENRICH with the default prefix would write
            "_age": [0.0] * 20,
            "masked": ["old"] * 20,
        }
    )


_CASES = {
    "full_masking": (
        lambda **kw: FullMaskingOperation(field_name="name", mode="ENRICH", **kw),
        "_name",
    ),
    "full_masking_named": (
        lambda **kw: FullMaskingOperation(
            field_name="name", mode="ENRICH", output_field_name="masked", **kw
        ),
        "masked",
    ),
    "numeric_generalization": (
        lambda **kw: NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=3, mode="ENRICH", **kw
        ),
        "_age",
    ),
    "uniform_numeric_noise": (
        lambda **kw: UniformNumericNoiseOperation(
            field_name="age", noise_range=2.0, noise_type="additive", mode="ENRICH", **kw
        ),
        "_age",
    ),
    "cell_suppression": (
        lambda **kw: CellSuppressionOperation(
            field_name="name", suppression_strategy="null", mode="ENRICH", **kw
        ),
        "_name",
    ),
    "fake_name": (
        lambda **kw: FakeNameOperation(field_name="name", mode="ENRICH", **kw),
        "_name",
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_existing_target_fails_by_default(case, tmp_path):
    make, target = _CASES[case]
    result = make().execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert result.status == OperationStatus.ERROR
    assert f"'{target}'" in result.error_message
    assert "overwrite_existing_output=True" in result.error_message
    assert SECRET not in result.error_message
    assert list((tmp_path / "output").glob("*")) == [] if (tmp_path / "output").exists() else True


@pytest.mark.parametrize("case", sorted(_CASES))
def test_explicit_option_overwrites_without_warning_spam(case, tmp_path, caplog):
    make, target = _CASES[case]
    with caplog.at_level(logging.DEBUG):
        result = make(overwrite_existing_output=True).execute(
            DataSource(dataframes={"main": _frame()}), tmp_path, None
        )

    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert not [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "already exists" in r.getMessage()
    ]
    written = pd.read_csv(next((tmp_path / "output").glob("*_output_*.csv")))
    assert target in written.columns
    assert not (written[target].astype(str) == "old").all(), "column was not replaced"


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_free_target_is_unaffected(case, tmp_path):
    make, target = _CASES[case]
    df = _frame().drop(columns=[target])
    result = make().execute(DataSource(dataframes={"main": df}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message


def test_replace_mode_never_checks_the_target(tmp_path):
    result = FullMaskingOperation(field_name="name").execute(
        DataSource(dataframes={"main": _frame()}), tmp_path, None
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message


def test_the_option_defaults_to_false_and_is_in_the_config_schema():
    from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig

    assert FullMaskingOperation(field_name="name").overwrite_existing_output is False
    prop = BaseOperationConfig.schema["properties"]["overwrite_existing_output"]
    assert prop["default"] is False


def test_error_is_a_typed_validation_error():
    err = OutputFieldExistsError("_name", "FullMaskingOperation")
    assert isinstance(err, ValidationError)
    assert err.details["output_field"] == "_name"
