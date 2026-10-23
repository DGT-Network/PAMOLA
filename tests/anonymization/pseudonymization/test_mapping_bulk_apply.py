"""PERF-FIX-4: the mapping is applied in bulk with the same results as the row-by-row pass.

The per-row loop (three lock acquisitions, a clock and a running mean per row) made 1M rows
~15x the cost of 100K. Distinct values are now resolved once, in order of first appearance, and
the results are placed by position.
"""

import json

import numpy as np
import pandas as pd

from pamola_core.anonymization.commons.mapping_storage import MappingStorage
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

KEY = "ab" * 32


def _op(path, **kwargs):
    return ConsistentMappingPseudonymizationOperation(
        field_name="v", mapping_encryption_key=KEY, mapping_file=str(path),
        pseudonym_type="sequential", use_cache=False, **kwargs,
    )


LAST = {}


def _find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _run(op, df, task_dir):
    result = op.execute(DataSource(dataframes={"main": df}), task_dir, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    LAST.clear()  # per-run counters are reset after execute(); they survive in the metrics file
    for artifact in result.get_artifacts_by_type("json"):
        data = json.loads(open(artifact.path, encoding="utf-8").read())
        for key in ("total_lookups", "mapping_hits", "new_mappings_created"):
            value = _find(data, key)
            if value is not None:
                LAST[key] = value
    return pd.read_csv(result.get_artifacts_by_type("csv")[0].path, dtype=str, keep_default_na=False)["v"].tolist()


def test_new_pseudonyms_follow_first_appearance_order(tmp_path):
    out = _run(_op(tmp_path / "m.enc"), pd.DataFrame({"v": ["b", "a", "b", "c", "a"]}), tmp_path / "t")
    # sequential counter: b first, then a, then c - repeated values reuse theirs
    assert out == ["000001", "000002", "000001", "000003", "000002"]


def test_counters_match_a_row_by_row_pass(tmp_path):
    op = _op(tmp_path / "m.enc")
    values = ["x", "y", "x", "x", "z", "y"]
    _run(op, pd.DataFrame({"v": values}), tmp_path / "t")
    assert LAST["total_lookups"] == 6
    assert LAST["new_mappings_created"] == 3
    assert LAST["mapping_hits"] == 3   # the three repeats find the value in the mapping


def test_existing_mappings_count_as_hits_and_keep_their_pseudonym(tmp_path):
    path = tmp_path / "m.enc"
    first = _run(_op(path), pd.DataFrame({"v": ["p", "q"]}), tmp_path / "a")
    op = _op(path)
    second = _run(op, pd.DataFrame({"v": ["q", "r", "p", "q"]}), tmp_path / "b")
    assert second[0] == first[1] and second[2] == first[0] and second[3] == first[1]
    assert LAST["mapping_hits"] == 3 and LAST["new_mappings_created"] == 1


def test_duplicate_index_labels_and_nulls_are_positional(tmp_path):
    df = pd.DataFrame({"v": ["a", None, "b", "a"]}, index=[0, 0, 0, 0])
    out = _run(_op(tmp_path / "m.enc"), df, tmp_path / "t")
    assert out[0] == out[3] != out[2]
    assert out[1] == ""   # the null row stays null


def test_persisting_at_the_threshold_keeps_every_mapping(tmp_path):
    path = tmp_path / "m.enc"
    values = [f"v{i}" for i in range(25)] * 2
    _run(_op(path, persist_frequency=4), pd.DataFrame({"v": values}), tmp_path / "t")
    stored = MappingStorage(path, bytes.fromhex(KEY), format="csv").load()
    assert set(stored) == {f"v{i}" for i in range(25)}
    assert len(set(stored.values())) == 25


def test_bulk_matches_a_reference_loop_on_random_data(tmp_path):
    rng = np.random.default_rng(5)
    values = [f"id-{int(i)}" for i in rng.integers(0, 300, size=5000)]
    out = _run(_op(tmp_path / "m.enc"), pd.DataFrame({"v": values}), tmp_path / "t")
    expected, seen = [], {}
    for v in values:
        seen.setdefault(v, f"{len(seen) + 1:06d}")
        expected.append(seen[v])
    assert out == expected
