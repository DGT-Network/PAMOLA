"""DATA-02: cell and record suppression on a 0-row DataFrame.

A 0-row frame is a normal shape - an upstream filter step removes every record.
Both operations chunked the input and concatenated the chunks, and with no chunks
that is a raw ``ValueError: No objects to concatenate`` surfacing as an opaque
``ERROR``. The defined behaviour is now the one ``AttributeSuppressionOperation``
and the profilers already had: ``SUCCESS``, an output with the input's schema and
no rows, and zero counts in the metrics.
"""

import json

import pandas as pd
import pytest

from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {"name": pd.Series([], dtype=object), "age": pd.Series([], dtype=float)}
    )


_CASES = {
    "cell_null": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null"
    ),
    "cell_mean": lambda: CellSuppressionOperation(
        field_name="age", suppression_strategy="mean"
    ),
    "cell_small_chunks": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null", chunk_size=10
    ),
    "cell_enrich": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null", mode="ENRICH"
    ),
    "cell_dask": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null", use_dask=True, npartitions=2
    ),
    "cell_joblib": lambda: CellSuppressionOperation(
        field_name="name",
        suppression_strategy="null",
        use_vectorization=True,
        parallel_processes=2,
    ),
    "record_null": lambda: RecordSuppressionOperation(
        field_name="name", suppression_condition="null"
    ),
    "record_value": lambda: RecordSuppressionOperation(
        field_name="name", suppression_condition="value", suppression_values=["x"]
    ),
    "record_save_suppressed": lambda: RecordSuppressionOperation(
        field_name="name", suppression_condition="null", save_suppressed_records=True
    ),
    "record_dask": lambda: RecordSuppressionOperation(
        field_name="name", suppression_condition="null", use_dask=True, npartitions=2
    ),
    "record_joblib": lambda: RecordSuppressionOperation(
        field_name="name",
        suppression_condition="null",
        use_vectorization=True,
        parallel_processes=2,
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_empty_dataframe_succeeds_with_an_empty_output(case, tmp_path):
    result = _CASES[case]().execute(
        DataSource(dataframes={"main": _empty()}), tmp_path, None
    )

    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert "concatenate" not in (result.error_message or "")

    outputs = sorted((tmp_path / "output").glob("*_output_*.csv"))
    assert len(outputs) == 1
    written = pd.read_csv(outputs[0])
    assert len(written) == 0
    # The schema survives, so a downstream step still finds its columns.
    assert {"name", "age"} <= set(written.columns)


@pytest.mark.parametrize("case", ["cell_null", "record_null"])
def test_empty_dataframe_reports_zero_in_the_metrics(case, tmp_path):
    result = _CASES[case]().execute(
        DataSource(dataframes={"main": _empty()}), tmp_path, None
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message

    metrics_file = next((tmp_path / "metrics").glob("*Operation_metrics_*.json"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))["metrics"]
    assert metrics["name_effectiveness_ratio"] == 0.0
    assert metrics["name_original_unique"] == 0
    assert metrics["name_anonymized_unique"] == 0
