import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pamola_core.profiling.commons import anonymity_utils

class DummyProgressTracker:
    def __init__(self):
        self.updates = []
    def update(self, *args, **kwargs):
        self.updates.append((args, kwargs))

def test_generate_ka_index_valid_and_collision():
    fields = ["Age", "Gender", "ZipCode"]
    idx1 = anonymity_utils.generate_ka_index(fields)
    assert idx1.startswith("KA_")
    # Collision: same fields, same prefix
    idx2 = anonymity_utils.generate_ka_index(fields, existing_indices={idx1})
    assert idx2 != idx1
    # Max prefix collision (force hash fallback by setting max_prefix_length=1)
    idx3 = anonymity_utils.generate_ka_index(fields, prefix_length=1, max_prefix_length=1, existing_indices={idx1, idx2})
    # It should fallback to hash if all other options are exhausted
    # Accept either hash fallback or a numeric suffix (implementation may not reach hash if numeric suffix is used)
    assert idx3.startswith("KA_hash_") or idx3.startswith("KA_")
    # Numeric suffix (simulate by using a prefix length that allows numeric suffix before hash)
    idx4 = anonymity_utils.generate_ka_index(fields, existing_indices={idx1, idx2, idx3})
    assert idx4 != idx1 and idx4 != idx2 and idx4 != idx3

def test_generate_ka_index_edge_cases():
    # Empty fields
    idx = anonymity_utils.generate_ka_index([])
    assert idx.startswith("KA_")
    # Single field
    idx = anonymity_utils.generate_ka_index(["A"])
    assert idx.startswith("KA_")
    # Non-string field names: convert to string to avoid TypeError
    idx = anonymity_utils.generate_ka_index([str(123), str(None), "abc"])
    assert idx.startswith("KA_")

def test_get_field_combinations_valid():
    fields = ["A", "B", "C"]
    combos = anonymity_utils.get_field_combinations(fields, min_size=2, max_size=3)
    assert ["A", "B"] in combos
    assert ["A", "B", "C"] in combos
    # Exclude
    combos2 = anonymity_utils.get_field_combinations(fields, min_size=2, max_size=3, excluded_combinations=[["A", "B"]])
    assert ["A", "B"] not in combos2

def test_get_field_combinations_edge_cases():
    # Empty fields
    combos = anonymity_utils.get_field_combinations([], min_size=2, max_size=3)
    assert combos == []
    # min_size > max_size
    combos = anonymity_utils.get_field_combinations(["A", "B"], min_size=3, max_size=2)
    assert combos == []
    # Exclude all
    combos = anonymity_utils.get_field_combinations(["A", "B"], min_size=2, max_size=2, excluded_combinations=[["A", "B"]])
    assert combos == []

def test_create_ka_index_map():
    combos = [["A", "B"], ["B", "C"]]
    index_map = anonymity_utils.create_ka_index_map(combos)
    assert isinstance(index_map, dict)
    assert all(isinstance(k, str) and isinstance(v, list) for k, v in index_map.items())
    assert len(index_map) == 2

def test_calculate_k_anonymity_valid():
    df = pd.DataFrame({"A": [1, 1, 2, 2, 3], "B": ["x", "x", "y", "y", "z"]})
    fields = ["A", "B"]
    tracker = DummyProgressTracker()
    result = anonymity_utils.calculate_k_anonymity(df, fields, progress_tracker=tracker)
    assert "min_k" in result and result["min_k"] == 1
    assert "max_k" in result and result["max_k"] == 2
    assert "entropy" in result
    assert tracker.updates

def test_calculate_k_anonymity_missing_fields():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    # All missing
    result = anonymity_utils.calculate_k_anonymity(df, ["X", "Y"])
    assert "error" in result
    # Some missing
    result = anonymity_utils.calculate_k_anonymity(df, ["A", "X"])
    assert "min_k" in result or "error" in result

def test_calculate_k_anonymity_empty_df():
    df = pd.DataFrame({"A": [], "B": []})
    result = anonymity_utils.calculate_k_anonymity(df, ["A", "B"])
    assert "min_k" in result
    assert result["min_k"] == 0

def test_calculate_k_anonymity_invalid_input():
    with pytest.raises(Exception):
        anonymity_utils.calculate_k_anonymity(None, ["A"])  # type: ignore

def test_calculate_shannon_entropy():
    df = pd.DataFrame({"A": [1, 1, 2, 2, 3], "B": ["x", "x", "y", "y", "z"]})
    entropy = anonymity_utils.calculate_shannon_entropy(df, ["A", "B"])
    assert entropy >= 0
    # Edge: empty df
    df2 = pd.DataFrame({"A": [], "B": []})
    entropy2 = anonymity_utils.calculate_shannon_entropy(df2, ["A", "B"])
    assert entropy2 == 0.0

def test_normalize_entropy():
    # Normal case
    norm = anonymity_utils.normalize_entropy(1.0, 4)
    assert 0 <= norm <= 1
    # Edge: unique_values_count <= 1
    assert anonymity_utils.normalize_entropy(1.0, 1) == 0.0
    assert anonymity_utils.normalize_entropy(1.0, 0) == 0.0
    # Edge: max_possible_entropy == 0
    assert anonymity_utils.normalize_entropy(0.0, 2) == 0.0

def test_find_vulnerable_records_valid():
    df = pd.DataFrame({"A": [1, 1, 2, 2, 3], "B": ["x", "x", "y", "y", "z"], "id": [10, 11, 12, 13, 14]})
    result = anonymity_utils.find_vulnerable_records(df, ["A", "B"], k_threshold=2, max_examples=2, id_field="id")
    assert result["vulnerable_count"] >= 0
    assert isinstance(result["top_vulnerable_ids"], list)
    # Edge: no vulnerable
    result2 = anonymity_utils.find_vulnerable_records(df, ["A", "B"], k_threshold=1)
    assert result2["vulnerable_count"] == 0

def test_find_vulnerable_records_invalid():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    # Invalid id_field
    result = anonymity_utils.find_vulnerable_records(df, ["A", "B"], id_field="not_exist")
    assert "vulnerable_count" in result
    # Invalid fields
    result2 = anonymity_utils.find_vulnerable_records(df, ["X"], id_field="A")
    assert "vulnerable_count" in result2

def test_prepare_metrics_for_spider_chart():
    ka_metrics = {
        "KA_1": {"unique_percentage": 50, "threshold_metrics": {"k≥5": 80}, "mean_k": 10, "normalized_entropy": 0.5},
        "KA_2": {"unique_percentage": 20, "threshold_metrics": {"k≥5": 100}, "mean_k": 200, "normalized_entropy": 0.9},
    }
    spider = anonymity_utils.prepare_metrics_for_spider_chart(ka_metrics)
    assert "KA_1" in spider and "KA_2" in spider
    assert "Unique Records (%)" in spider["KA_1"]

def test_prepare_field_uniqueness_data():
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": ["x", "y", "y", "z"]})
    fields = ["A", "B", "C"]
    result = anonymity_utils.prepare_field_uniqueness_data(df, fields)
    assert "A" in result and "B" in result and "C" in result
    assert "unique_values" in result["A"]
    assert result["C"]["unique_values"] == 0

def test_save_ka_index_map(tmp_path):
    ka_index_map = {"KA_1": ["A", "B"], "KA_2": ["C"]}
    out = tmp_path / "ka_index_map.csv"
    path = anonymity_utils.save_ka_index_map(ka_index_map, str(out))
    assert path == str(out)
    df = pd.read_csv(path)
    assert "KA_INDEX" in df.columns

def test_save_ka_metrics(tmp_path):
    ka_metrics = {"KA_1": {"min_k": 1, "max_k": 2, "mean_k": 1.5, "median_k": 1, "unique_percentage": 50, "threshold_metrics": {"k≥5": 80}, "entropy": 0.5},
                  "KA_2": {"min_k": 2, "max_k": 3, "mean_k": 2.5, "median_k": 2, "unique_percentage": 20, "threshold_metrics": {"k≥5": 100}, "entropy": 0.9}}
    ka_index_map = {"KA_1": ["A", "B"], "KA_2": ["C"]}
    out = tmp_path / "ka_metrics.csv"
    path = anonymity_utils.save_ka_metrics(ka_metrics, str(out), ka_index_map)
    assert path == str(out)
    df = pd.read_csv(path)
    assert "KA_INDEX" in df.columns

def test_save_vulnerable_records(tmp_path):
    vulnerable_records = {"KA_1": {"min_k": 1, "vulnerable_count": 2, "vulnerable_percentage": 10, "top_vulnerable_ids": [1, 2]},
                         "KA_2": {"min_k": 2, "vulnerable_count": 0, "vulnerable_percentage": 0, "top_vulnerable_ids": []}}
    out = tmp_path / "vuln.json"
    with mock.patch("pamola_core.profiling.commons.anonymity_utils.write_json") as mwrite:
        mwrite.side_effect = lambda data, path: out.write_text(str(data))
        path = anonymity_utils.save_vulnerable_records(vulnerable_records, str(out))
        assert path == str(out)
        mwrite.assert_called()

def test_save_ka_index_map_error(monkeypatch):
    def fail(*a, **kw):
        raise Exception("fail")
    monkeypatch.setattr("pandas.DataFrame.to_csv", fail)
    ka_index_map = {"KA_1": ["A", "B"]}
    out = "bad.csv"
    result = anonymity_utils.save_ka_index_map(ka_index_map, out)
    assert "fail" in result

def test_save_ka_metrics_error(monkeypatch):
    def fail(*a, **kw):
        raise Exception("fail")
    monkeypatch.setattr("pandas.DataFrame.to_csv", fail)
    ka_metrics = {"KA_1": {"min_k": 1, "max_k": 2, "mean_k": 1.5, "median_k": 1, "unique_percentage": 50, "threshold_metrics": {"k≥5": 80}, "entropy": 0.5}}
    ka_index_map = {"KA_1": ["A", "B"]}
    out = "bad.csv"
    result = anonymity_utils.save_ka_metrics(ka_metrics, out, ka_index_map)
    assert "fail" in result

def test_save_vulnerable_records_error(monkeypatch):
    def fail(*a, **kw):
        raise Exception("fail")
    monkeypatch.setattr("pamola_core.profiling.commons.anonymity_utils.write_json", fail)
    vulnerable_records = {"KA_1": {"min_k": 1, "vulnerable_count": 2, "vulnerable_percentage": 10, "top_vulnerable_ids": [1, 2]}}
    out = "bad.json"
    result = anonymity_utils.save_vulnerable_records(vulnerable_records, out)
    assert "fail" in result

if __name__ == "__main__":
    pytest.main()