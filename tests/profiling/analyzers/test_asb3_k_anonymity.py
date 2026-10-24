"""ASB-3: k-anonymity percentages count RECORDS; a failed enrichment fails the run; the caller's
DataSource keeps the frame it registered."""

import pandas as pd

from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _df():
    return pd.DataFrame({"a": ["x"] * 6 + ["y"] * 4, "v": range(10)})


def _op(**kw):
    return KAnonymityProfilerOperation(
        quasi_identifiers=["a"], threshold_k=5, generate_visualization=False, use_cache=False, **kw)


def test_asb3_compliance_and_distribution_are_shares_of_records():
    m = _op()._calculate_k_anonymity(_df(), ["a"])
    # classes of 6 and 4 records: every record has k>=4, six of ten have k>=5
    assert m["threshold_compliance"]["k≥2"] == 100.0
    assert m["threshold_compliance"]["k≥5"] == 60.0
    assert m["threshold_compliance"]["k≥10"] == 0.0
    assert m["k_distribution"]["k=2-4"] == 40.0 and m["k_distribution"]["k=5-9"] == 60.0
    assert abs(sum(m["k_distribution"].values()) - 100.0) < 1e-9
    assert m["vulnerable_percent"] == 40.0  # records with k < 5 (was already right)


def test_asb3_dead_chunked_metrics_method_is_gone():
    assert not hasattr(KAnonymityProfilerOperation, "_calculate_k_anonymity_chunked")


def test_asb3_a_failed_enrichment_fails_the_operation(tmp_path, monkeypatch):
    def boom(self, df, fields):
        raise RuntimeError("enrichment broke")

    monkeypatch.setattr(KAnonymityProfilerOperation, "_calculate_k_values", boom)
    ds = DataSource(dataframes={"main": _df()})
    out = _op(analysis_mode="ENRICH").execute(ds, tmp_path / "t", _Reporter())
    assert out.status == OperationStatus.ERROR
    assert "enrichment broke" in str(out.error_message)


def test_asb3_enrichment_does_not_replace_the_callers_frame(tmp_path):
    df = _df()
    ds = DataSource(dataframes={"main": df})
    out = _op(analysis_mode="ENRICH").execute(ds, tmp_path / "t", _Reporter())
    assert out.status == OperationStatus.SUCCESS, out.error_message
    assert ds.dataframes["main"] is df
    assert list(df.columns) == ["a", "v"]
    enriched = [p for p in (tmp_path / "t").rglob("enriched_data_*.csv")]
    assert enriched and "k_anon" in enriched[0].read_text(encoding="utf-8").splitlines()[0]
