"""CDX-R2-3b: the cache key identifies the exact, typed input - every returned column.

Reproductions: C:/wt-core-env/codex_scratch/r2/probe_cache_identity.py. Before: object columns
[1, '1'] and ['1', '1'] had the same fingerprint (served identity-disclosure 1.0 for a dataset
whose fresh value is 0.0); a masking run keyed only on the masked field served an old CSV after an
untouched column changed; unhashable values fell back to their text form.
"""

import subprocess
import sys

import pandas as pd
import pytest

from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.metrics.operations.privacy_ops import PrivacyMetricOperation
from pamola_core.utils.helpers import UnhashableDataError, generate_data_hash
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _obj(values):
    return pd.DataFrame({"x": pd.Series(values, dtype=object)})


@pytest.mark.parametrize(("a", "b"), [
    ([1, "1"], ["1", "1"]),
    ([1, 2], [1.0, 2.0]),
    ([1, 0], [True, False]),
    ([1, None], ["1", None]),
    (["a", 1], ["a", 1.5]),
])
def test_object_values_of_different_types_have_different_fingerprints(a, b):
    assert generate_data_hash(_obj(a)) != generate_data_hash(_obj(b))


def test_equal_typed_values_have_equal_fingerprints_across_processes():
    here = generate_data_hash(_obj([1, "1", 2.5, None, "x"]))
    code = (
        "import pandas as pd;"
        "from pamola_core.utils.helpers import generate_data_hash;"
        "print(generate_data_hash(pd.DataFrame({'x': pd.Series([1,'1',2.5,None,'x'], dtype=object)})))"
    )
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert other.returncode == 0, other.stderr
    assert other.stdout.strip() == here


@pytest.mark.parametrize("bad", [[[1, 2], [3]], [{"a": 1}, {"a": 2}]])
def test_unhashable_values_raise_instead_of_falling_back_to_text(bad):
    with pytest.raises(UnhashableDataError):
        generate_data_hash(_obj(bad))


def _privacy_run(task_dir, original):
    op = PrivacyMetricOperation(
        privacy_metrics=["uniqueness"], columns=["x"], use_cache=True,
        metric_params={"uniqueness": {"quasi_identifiers": ["x"], "sensitives": [],
                                      "l_diversity": False, "t_closeness": False}},
    )
    return op.execute(
        DataSource(dataframes={"main": original, "synth": original}), task_dir, None,
        original_dataset_name="main", transformed_dataset_name="synth",
    )


def test_metric_cache_does_not_alias_typed_values(tmp_path):
    """Through the real metrics cache: [1,'1'] has two unique records, ['1','1'] has none."""
    a, b = _obj([1, "1"]), _obj(["1", "1"])
    first = _privacy_run(tmp_path, a)
    second = _privacy_run(tmp_path, b)
    assert first.status != OperationStatus.ERROR and second.status != OperationStatus.ERROR
    assert not second.metrics.get("cached")
    third = _privacy_run(tmp_path, a)
    assert third.metrics.get("cached") is True


def _mask(task_dir, df, **kw):
    op = FullMaskingOperation(field_name="name", use_cache=True, **kw)
    return op.execute(DataSource(dataframes={"main": df}), task_dir, None)


def _frame(context):
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol"] * 4, "context": [context] * 12})


def test_masking_cache_keys_on_every_returned_column(tmp_path):
    first = _mask(tmp_path, _frame("OLD"))
    second = _mask(tmp_path, _frame("NEW"))
    assert first.status == second.status == OperationStatus.SUCCESS
    assert not second.metrics.get("cached"), "an untouched column changed: the cache must miss"
    written = [p for p in (tmp_path / "output").glob("*_output_*.csv")]
    assert any("NEW" in p.read_text(encoding="utf-8") for p in written)
    # identical input still hits
    again = _mask(tmp_path, _frame("NEW"))
    assert again.metrics.get("cached") is True


def test_unhashable_column_disables_caching_without_failing_the_run(tmp_path):
    df = _frame("x")
    df["blob"] = [[1, 2]] * 12
    result = _mask(tmp_path, df)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert not list((tmp_path / "cache").rglob("*.json"))
