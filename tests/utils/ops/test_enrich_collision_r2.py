"""CDX-R2-5: every column an ENRICH run writes is declared and guarded.

Reproductions: probe_more.py (whitelist cleaning) and probe_final.py (k-anonymity profiling,
empty prefix). Before: whitelist-only cleaning declared no targets and replaced an existing
``_x``; k-anonymity ENRICH replaced an existing ``_KA_x_k``; an empty prefix exempted the guard.
"""

import pandas as pd
import pytest

from pamola_core.profiling.analyzers.anonymity import (
    KAnonymityProfilerOperation,
    generate_ka_field_name,
    generate_output_field_name,
)
from pamola_core.transformations.cleaning.clean_invalid_values import CleanInvalidValuesOperation
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _run(op, df, task_dir):
    return op.execute(DataSource(dataframes={"main": df}), task_dir, None)


def _written(task_dir):
    return pd.read_csv(next((task_dir / "output").glob("*_output_*.csv")))


@pytest.fixture
def allow_file(tmp_path):
    path = tmp_path / "allow.txt"
    path.write_text("alice\n", encoding="utf-8")
    return path


def test_whitelist_only_cleaning_declares_its_target_and_fails_on_collision(tmp_path, allow_file):
    df = pd.DataFrame({"x": ["alice", "bob"], "_x": ["KEEP1", "KEEP2"]})
    op = CleanInvalidValuesOperation(field_constraints={}, whitelist_path={"x": str(allow_file)},
                                     mode="ENRICH", use_cache=False)
    assert op.get_enrich_targets(df) == ["_x"]
    result = _run(op, df, tmp_path)
    assert result.status == OperationStatus.ERROR and "'_x'" in result.error_message
    assert list(df["_x"]) == ["KEEP1", "KEEP2"]


def test_whitelist_cleaning_with_the_option_replaces_the_target(tmp_path, allow_file):
    df = pd.DataFrame({"x": ["alice", "bob"], "_x": ["KEEP1", "KEEP2"]})
    op = CleanInvalidValuesOperation(field_constraints={}, whitelist_path={"x": str(allow_file)},
                                     mode="ENRICH", use_cache=False, overwrite_existing_output=True)
    result = _run(op, df, tmp_path)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert "KEEP1" not in set(_written(tmp_path)["_x"].astype(str))


def test_blacklist_and_null_replacement_targets_are_declared():
    df = pd.DataFrame({"a": [1.0, None], "b": ["u", "v"], "_b": ["k", "k"]})
    op = CleanInvalidValuesOperation(field_constraints={}, blacklist_path={"b": "x.txt"},
                                     null_replacement={"a": "mean"}, mode="ENRICH", use_cache=False)
    assert sorted(op.get_enrich_targets(df)) == ["_a", "_b"]
    every = CleanInvalidValuesOperation(field_constraints={}, null_replacement="mode", mode="ENRICH", use_cache=False)
    assert sorted(every.get_enrich_targets(df)) == ["__b", "_a", "_b"]


def test_null_replacement_with_the_option_replaces_instead_of_cleaning_the_old_column(tmp_path):
    df = pd.DataFrame({"a": [1.0, None, 3.0], "_a": [99.0, 99.0, 99.0]})
    op = CleanInvalidValuesOperation(field_constraints={}, null_replacement={"a": "mean"}, mode="ENRICH",
                                     use_cache=False, overwrite_existing_output=True)
    result = _run(op, df, tmp_path)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert list(_written(tmp_path)["_a"]) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("make", [
    lambda: CleanInvalidValuesOperation(
        field_constraints={"x": {"data_type": "numeric", "constraint_type": "min_value", "min_value": 0}},
        mode="ENRICH", column_prefix="", use_cache=False),
    lambda: ImputeMissingValuesOperation(
        field_strategies={"x": {"data_type": "numeric", "imputation_strategy": "mean"}},
        mode="ENRICH", column_prefix="", use_cache=False),
])
def test_an_empty_prefix_is_not_an_exemption(make, tmp_path):
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    op = make()
    assert op.get_enrich_targets(df) == ["x"]
    result = _run(op, df, tmp_path)
    assert result.status == OperationStatus.ERROR and "'x'" in result.error_message
    assert df["x"].isna().sum() == 1  # the caller's column is untouched


def _ka_column(suffix="k"):
    return generate_output_field_name(field_name=generate_ka_field_name(["x"]), mode="ENRICH",
                                      operation_suffix=suffix)


def test_k_anonymity_enrichment_is_guarded_like_every_other_enrichment(tmp_path):
    col = _ka_column()
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], col: [99] * 4})
    op = KAnonymityProfilerOperation(quasi_identifiers=["x"], analysis_mode="ENRICH",
                                     output_field_suffix="k", use_cache=False)
    assert op.get_enrich_targets(df) == [col]
    result = _run(op, df, tmp_path)
    assert result.status == OperationStatus.ERROR and f"'{col}'" in result.error_message
    assert list(df[col]) == [99] * 4


def test_k_anonymity_enrichment_with_the_option_and_without_a_collision(tmp_path):
    col = _ka_column()
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], col: [99] * 4})
    ok = KAnonymityProfilerOperation(quasi_identifiers=["x"], analysis_mode="ENRICH", output_field_suffix="k",
                                     use_cache=False, overwrite_existing_output=True)
    assert _run(ok, df, tmp_path).status == OperationStatus.SUCCESS
    fresh_dir = tmp_path / "fresh"
    fresh = KAnonymityProfilerOperation(quasi_identifiers=["x"], analysis_mode="ENRICH", output_field_suffix="k",
                                        use_cache=False)
    assert _run(fresh, df.drop(columns=col), fresh_dir).status == OperationStatus.SUCCESS
    # analysis-only runs write nothing, so they never collide
    only = KAnonymityProfilerOperation(quasi_identifiers=["x"], analysis_mode="ANALYZE", use_cache=False)
    assert only.get_enrich_targets(df) and not only._enrich_requested()
