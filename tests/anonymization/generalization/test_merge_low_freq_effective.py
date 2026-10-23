"""PERF-FIX-1: strategy="merge_low_freq" actually merges the rare categories.

Before: the strategy protected the `max_categories` (default 1000) most frequent categories
from grouping, so any field with fewer categories came back unchanged (SUCCESS, 0 values
changed) - found by the PERF-01 correctness check.
"""

import pandas as pd

from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

COMMON = [f"common-{i}" for i in range(10)]
RARE = [f"rare-{i}" for i in range(6)]


def _frame():
    values = [c for c in COMMON for _ in range(300)] + [r for r in RARE for _ in range(2)]
    return pd.DataFrame({"c": values})


def _run(tmp_path, **kwargs):
    op = CategoricalGeneralizationOperation(field_name="c", use_cache=False, **kwargs)
    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    path = next(a.path for a in result.artifacts if str(a.path).endswith(".csv") and "output" in str(a.path))
    return pd.read_csv(path)["c"]


def test_merge_low_freq_merges_the_rare_categories(tmp_path):
    out = _run(tmp_path, strategy="merge_low_freq", min_group_size=5, freq_threshold=0.01)
    assert not set(RARE) & set(out)               # every rare category is gone
    assert (out == "OTHER").sum() == 12           # merged into the one group
    assert set(COMMON) <= set(out)                # frequent ones are untouched
    assert out.nunique() == 11


def test_either_threshold_makes_a_category_rare(tmp_path):
    # count threshold alone: 2 rows < 5, although the share (0.07%) is above a 0.01% threshold
    out = _run(tmp_path, strategy="merge_low_freq", min_group_size=5, freq_threshold=0.0001)
    assert not set(RARE) & set(out)
    # frequency threshold alone: 300 rows >= min_group_size, but the share (~10%) is below 20%
    out = _run(tmp_path, strategy="merge_low_freq", min_group_size=5, freq_threshold=0.20)
    assert not set(COMMON) & set(out) and set(out) == {"OTHER"}


def test_large_datasets_are_not_exempt(tmp_path):
    # every category has far more than min_group_size rows (the PERF-01 100K/1M shape)
    values = [f"c{i}" for i in range(50) for _ in range(100)] + ["big"] * 5000
    op = CategoricalGeneralizationOperation(field_name="c", strategy="merge_low_freq", min_group_size=5,
                                            freq_threshold=0.05, use_cache=False)
    result = op.execute(DataSource(dataframes={"main": pd.DataFrame({"c": values})}), tmp_path, None)
    out = pd.read_csv(next(a.path for a in result.artifacts if str(a.path).endswith(".csv") and "output" in str(a.path)))["c"]
    assert set(out) == {"OTHER", "big"}


def test_numbered_groups_are_applied_too(tmp_path):
    out = _run(tmp_path, strategy="merge_low_freq", min_group_size=5, freq_threshold=0.01, group_rare_as="RARE_N")
    assert not set(RARE) & set(out)


def test_sibling_frequency_based_strategy_also_groups(tmp_path):
    out = _run(tmp_path, strategy="frequency_based", max_categories=10, min_group_size=5)
    assert not set(RARE) & set(out)


def test_categorical_dtype_input_is_grouped_not_silently_skipped(tmp_path):
    # the engine converts large text columns to category; assigning the group label used to
    # raise inside the grouping helper, which swallowed it and returned the data unchanged
    from pamola_core.anonymization.commons.category_utils import group_rare_categories

    series = _frame()["c"].astype("category")
    grouped, info = group_rare_categories(series, threshold=5, categories_to_group=set(RARE))
    assert not set(RARE) & set(grouped.astype(str))
    assert (grouped.astype(str) == "OTHER").sum() == 12
    assert info["categories_grouped"] == 6


def test_a_grouping_failure_is_raised_not_swallowed():
    import pytest
    from pamola_core.anonymization.commons.category_utils import group_rare_categories

    with pytest.raises(Exception):
        group_rare_categories(_frame()["c"], threshold="not-a-number")


def test_the_documented_rule_is_stated_where_the_parameters_are_described():
    from pamola_core.anonymization.schemas.categorical_op_core_schema import CategoricalGeneralizationConfig

    text = repr(CategoricalGeneralizationConfig.schema)
    assert "count < min_group_size OR share < freq_threshold" in text
    assert "frequency_based: number of most frequent categories that are kept" in text
    docstring = CategoricalGeneralizationOperation.__init__.__doc__
    assert "count < min_group_size OR share < freq_threshold" in docstring.replace("``", "")
