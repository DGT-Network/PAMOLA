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


def test_a_category_that_is_small_but_not_rare_by_frequency_stays(tmp_path):
    # both criteria must hold (documented "combined" rule): count < min_group_size AND share < threshold
    out = _run(tmp_path, strategy="merge_low_freq", min_group_size=5, freq_threshold=0.0001)
    assert set(RARE) <= set(out)


def test_numbered_groups_are_applied_too(tmp_path):
    out = _run(tmp_path, strategy="merge_low_freq", min_group_size=5, freq_threshold=0.01, group_rare_as="RARE_N")
    assert not set(RARE) & set(out)


def test_sibling_frequency_based_strategy_also_groups(tmp_path):
    out = _run(tmp_path, strategy="frequency_based", max_categories=10, min_group_size=5)
    assert not set(RARE) & set(out)
