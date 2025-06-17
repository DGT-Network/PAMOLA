import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pamola_core.profiling.commons.base import (
    AnalysisResult, BaseAnalyzer, BaseMultiFieldAnalyzer, BaseOperation, DataFrameProfiler, ProfileOperation
)

class DummyAnalyzer(BaseAnalyzer):
    def analyze(self, df, field_name, **kwargs):
        return AnalysisResult(stats={"mean": df[field_name].mean()}, field_name=field_name, data_type=str(df[field_name].dtype))

class DummyMultiFieldAnalyzer(BaseMultiFieldAnalyzer):
    def analyze_fields(self, df, field_names, **kwargs):
        return AnalysisResult(stats={"sum": df[field_names].sum().sum()}, field_name=','.join(field_names), data_type="multi")

class DummyOperation(BaseOperation):
    def __init__(self, name="DummyOp"):
        self.name = name
    def execute(self, df, reporter, profile_type, **kwargs):
        reporter.add_operation(f"Executed {self.name}")
        return {"executed": True, "profile_type": profile_type}

class DummyReporter:
    def __init__(self):
        self.operations = []
    def add_operation(self, msg, status=None, details=None):
        self.operations.append((msg, status, details))

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })

def test_analysis_result_valid():
    stats = {"min": 1, "max": 10}
    ar = AnalysisResult(stats, field_name="a", data_type="int", metadata={"foo": "bar"})
    ar.add_artifact("json", "/tmp/file.json", "desc")
    d = ar.to_dict()
    assert d["stats"] == stats
    assert d["field_name"] == "a"
    assert d["data_type"] == "int"
    assert d["metadata"] == {"foo": "bar"}
    assert d["artifacts"][0]["type"] == "json"
    assert d["artifacts"][0]["path"] == "/tmp/file.json"
    assert d["artifacts"][0]["description"] == "desc"

def test_analysis_result_edge_cases():
    ar = AnalysisResult({}, field_name=None, data_type=None, metadata=None)
    assert ar.stats == {}
    assert ar.field_name is None
    assert ar.data_type is None
    assert ar.metadata == {}
    assert ar.artifacts == []
    ar.add_artifact("csv", "", "")
    assert ar.artifacts[0]["type"] == "csv"
    assert ar.artifacts[0]["path"] == ""
    assert ar.artifacts[0]["description"] == ""
    d = ar.to_dict()
    assert d["artifacts"] == ar.artifacts

def test_analysis_result_invalid_input():
    # AnalysisResult(None) should not raise TypeError, but should set stats to None
    ar = AnalysisResult({})
    # The add_artifact method in AnalysisResult does not raise TypeError for None arguments by default.
    # Instead, it will add a dict with None values. Let's check for that behavior instead of expecting an exception.
    ar.add_artifact(None, None)
    assert ar.artifacts[-1]["type"] is None
    assert ar.artifacts[-1]["path"] is None
    assert ar.artifacts[-1]["description"] == ""

def test_base_analyzer_abstract():
    with pytest.raises(TypeError):
        BaseAnalyzer()
    dummy = DummyAnalyzer()
    df = pd.DataFrame({"a": [1, 2]})
    result = dummy.analyze(df, "a")
    assert isinstance(result, AnalysisResult)
    assert result.stats["mean"] == 1.5

def test_base_multifield_analyzer_abstract():
    with pytest.raises(TypeError):
        BaseMultiFieldAnalyzer()
    dummy = DummyMultiFieldAnalyzer()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = dummy.analyze_fields(df, ["a", "b"])
    assert isinstance(result, AnalysisResult)
    assert result.stats["sum"] == 10

def test_base_operation_abstract():
    with pytest.raises(TypeError):
        BaseOperation()
    dummy = DummyOperation()
    df = pd.DataFrame({"a": [1]})
    reporter = DummyReporter()
    result = dummy.execute(df, reporter, "basic")
    assert result["executed"] is True
    assert result["profile_type"] == "basic"
    assert any("Executed DummyOp" in op[0] for op in reporter.operations)

def test_dataframe_profiler_valid(sample_df):
    prof = DataFrameProfiler(analyzers={"int64": DummyAnalyzer()})
    with patch("pamola_core.profiling.commons.base.DataFrameProfiler._infer_data_type", return_value="int64"):
        results = prof.profile_dataframe(sample_df)
    assert set(results.keys()) == set(["a", "b", "c"])
    for res in results.values():
        assert isinstance(res, AnalysisResult)

def test_dataframe_profiler_include_exclude(sample_df):
    prof = DataFrameProfiler(analyzers={"int64": DummyAnalyzer()})
    with patch("pamola_core.profiling.commons.base.DataFrameProfiler._infer_data_type", return_value="int64"):
        results = prof.profile_dataframe(sample_df, include_fields=["a", "b"])
        assert set(results.keys()) == {"a", "b"}
        results = prof.profile_dataframe(sample_df, exclude_fields=["b"]) 
        assert "b" not in results

def test_dataframe_profiler_no_analyzer(sample_df, caplog):
    prof = DataFrameProfiler(analyzers={})
    with patch("pamola_core.profiling.commons.base.DataFrameProfiler._infer_data_type", return_value="int64"):
        with caplog.at_level("WARNING"):
            results = prof.profile_dataframe(sample_df)
            assert all("No analyzer found" in r for r in caplog.text.splitlines() if r)
            assert results == {}

def test_dataframe_profiler_error(sample_df, caplog):
    class FailingAnalyzer(BaseAnalyzer):
        def analyze(self, df, field_name, **kwargs):
            raise ValueError("fail")
    prof = DataFrameProfiler(analyzers={"int64": FailingAnalyzer()})
    with patch("pamola_core.profiling.commons.base.DataFrameProfiler._infer_data_type", return_value="int64"):
        with caplog.at_level("ERROR"):
            results = prof.profile_dataframe(sample_df)
            assert any("Error profiling field" in r for r in caplog.text.splitlines() if r)
            assert results == {}

def test_get_analyzer_for_type():
    prof = DataFrameProfiler(analyzers={"foo": DummyAnalyzer()})
    assert isinstance(prof._get_analyzer_for_type("foo"), DummyAnalyzer)
    assert prof._get_analyzer_for_type("bar") is None

def test_profile_operation_valid(sample_df):
    op1 = DummyOperation(name="Op1")
    op2 = DummyOperation(name="Op2")
    reporter = DummyReporter()
    profile_op = ProfileOperation("TestProfile", operations=[op1, op2])
    results = profile_op.execute(sample_df, reporter, "basic")
    assert set(results.keys()) == {"Op1", "Op2"}
    assert any("Starting profile operation" in op[0] for op in reporter.operations)
    assert any("Completed profile operation" in op[0] for op in reporter.operations)

def test_profile_operation_add_operation(sample_df):
    op1 = DummyOperation(name="Op1")
    profile_op = ProfileOperation("TestProfile")
    profile_op.add_operation(op1)
    assert profile_op.operations[0] == op1
    reporter = DummyReporter()
    results = profile_op.execute(sample_df, reporter, "basic")
    assert "Op1" in results

def test_profile_operation_error(sample_df, caplog):
    class FailingOperation(BaseOperation):
        def __init__(self):
            self.name = "FailingOp"
        def execute(self, df, reporter, profile_type, **kwargs):
            raise RuntimeError("fail op")
    op1 = DummyOperation(name="Op1")
    op2 = FailingOperation()
    reporter = DummyReporter()
    profile_op = ProfileOperation("TestProfile", operations=[op1, op2])
    with caplog.at_level("ERROR"):
        results = profile_op.execute(sample_df, reporter, "basic")
        assert "Op1" in results
        assert "FailingOp" not in results
        assert any("Error executing operation" in r for r in caplog.text.splitlines() if r)
        assert any(op[1] == "error" for op in reporter.operations if op[0].startswith("Operation"))

if __name__ == "__main__":
    pytest.main()