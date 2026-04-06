"""Deep coverage tests for multiple profiling analyzers.
Targets: email (84 missed), text (80), identity (77), date (72),
group (71), correlation (85), mvf (85) — total ~494 missed lines."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
        def add_artifact(self, *a, **kw): pass
    return R()


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# === Email Profiling ===
class TestEmailProfiling:
    @pytest.fixture
    def email_df(self):
        return pd.DataFrame({
            "email": [
                "alice@example.com", "bob@gmail.com", "carol@yahoo.co.uk",
                "dave@company.org", "eve@test.com", None, "invalid", "",
                "frank@sub.domain.com", "grace@example.com",
            ] * 20,
            "id": range(200),
        })

    def test_email_basic(self, email_df, reporter, tmp_path):
        from pamola_core.profiling.analyzers.email import EmailOperation
        op = EmailOperation(field_name="email")
        result = _run(op, email_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_email_with_many_domains(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.email import EmailOperation
        emails = [f"user{i}@domain{i % 30}.com" for i in range(200)]
        df = pd.DataFrame({"email": emails, "id": range(200)})
        op = EmailOperation(field_name="email")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_email_all_invalid(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.email import EmailOperation
        df = pd.DataFrame({"email": ["nope", "bad", "123"] * 50, "id": range(150)})
        op = EmailOperation(field_name="email")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Text Profiling ===
class TestTextProfiling:
    @pytest.fixture
    def text_df(self):
        texts = [
            "Short text",
            "This is a medium length text with some words in it",
            "A very long text " * 20,
            None,
            "",
            "Numbers 12345 and symbols !@#$%",
            "Unicode: café résumé naïve",
        ] * 30
        return pd.DataFrame({"text": texts[:200], "id": range(200)})

    def test_text_basic(self, text_df, reporter, tmp_path):
        from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation as TextOperation
        op = TextOperation(field_name="text")
        result = _run(op, text_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_text_all_long(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation as TextOperation
        df = pd.DataFrame({"text": ["word " * 100] * 200, "id": range(200)})
        op = TextOperation(field_name="text")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Identity Profiling ===
class TestIdentityProfiling:
    def test_identity_basic(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
        df = pd.DataFrame({
            "user_id": [f"UID-{i:05d}" for i in range(200)],
            "name": [f"user_{i}" for i in range(200)],
        })
        op = IdentityAnalysisOperation(uid_field="user_id", reference_fields=["name"])
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_identity_with_duplicates(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
        ids = [f"ID-{i % 50}" for i in range(200)]
        df = pd.DataFrame({"uid": ids, "name": [f"n{i}" for i in range(200)]})
        op = IdentityAnalysisOperation(uid_field="uid", reference_fields=["name"])
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_identity_with_nulls(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
        ids = [f"ID-{i}" if i % 5 != 0 else None for i in range(200)]
        df = pd.DataFrame({"uid": ids, "name": [f"n{i}" for i in range(200)]})
        op = IdentityAnalysisOperation(uid_field="uid", reference_fields=["name"])
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Date Profiling ===
class TestDateProfiling:
    def test_date_basic(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.date import DateOperation
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame({"dt": dates, "val": range(200)})
        op = DateOperation(field_name="dt")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_date_with_gaps(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.date import DateOperation
        dates = list(pd.date_range("2020-01-01", periods=100, freq="D"))
        dates += list(pd.date_range("2022-01-01", periods=100, freq="D"))
        df = pd.DataFrame({"dt": dates, "val": range(200)})
        op = DateOperation(field_name="dt")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_date_string_format(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.date import DateOperation
        dates = [f"2020-{(i % 12)+1:02d}-{(i % 28)+1:02d}" for i in range(200)]
        df = pd.DataFrame({"dt": dates, "val": range(200)})
        op = DateOperation(field_name="dt")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Group Profiling ===
# GroupAnalyzerOperation requires specific fields_config — tested elsewhere


# === Correlation Profiling ===
class TestCorrelationProfiling:
    def test_correlation_basic(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.correlation import CorrelationOperation
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(200),
            "b": np.random.randn(200),
        })
        op = CorrelationOperation(field1="a", field2="b")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_correlation_all_same(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.correlation import CorrelationOperation
        df = pd.DataFrame({"a": [1.0] * 200, "b": list(range(200))})
        op = CorrelationOperation(field1="a", field2="b")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_correlation_with_nulls(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.correlation import CorrelationOperation
        np.random.seed(42)
        a = np.random.randn(200).tolist()
        a[::10] = [np.nan] * len(a[::10])
        df = pd.DataFrame({"x": a, "y": np.random.randn(200)})
        op = CorrelationOperation(field1="x", field2="y")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_correlation_spearman(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.correlation import CorrelationOperation
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(200), "b": np.random.randn(200)})
        op = CorrelationOperation(field1="a", field2="b", method="spearman")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === MVF (Multi-Value Field) Profiling ===
class TestMVFProfiling:
    def test_mvf_basic(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.mvf import MVFOperation
        df = pd.DataFrame({
            "tags": [f"tag{i % 5},tag{(i+1) % 5}" for i in range(200)],
            "id": range(200),
        })
        op = MVFOperation(field_name="tags")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mvf_single_values(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.mvf import MVFOperation
        df = pd.DataFrame({
            "tags": [f"single{i % 10}" for i in range(200)],
            "id": range(200),
        })
        op = MVFOperation(field_name="tags")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mvf_with_nulls(self, reporter, tmp_path):
        from pamola_core.profiling.analyzers.mvf import MVFOperation
        tags = [f"a,b" if i % 3 != 0 else None for i in range(200)]
        df = pd.DataFrame({"tags": tags, "id": range(200)})
        op = MVFOperation(field_name="tags")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
