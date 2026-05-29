"""
File: test_hash_based_op.py
Test Target: pseudonymization/hash_based_op.py
Version: 1.1
Coverage Status: Smoke + security + mode + edge-case tests
"""

import json
import pickle

import pytest
import pandas as pd
from pamola_core.anonymization.pseudonymization.hash_based_op import (
    HashBasedPseudonymizationOperation,
)
from pamola_core.errors.exceptions import InvalidParameterError
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_data_source(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "carol@example.com",
                "dave@example.com",
                "eve@example.com",
            ],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        }
    )


@pytest.fixture
def task_dir(tmp_path):
    return tmp_path


@pytest.fixture
def reporter():
    class DummyReporter:
        def add_operation(self, *args, **kwargs):
            pass

        def add_artifact(self, *args, **kwargs):
            pass

    return DummyReporter()


def _make_op(**overrides):
    """Create HashBasedPseudonymizationOperation with deterministic salt."""
    defaults = dict(
        field_name="email",
        algorithm="sha3_256",
        salt_config={"source": "parameter", "value": "ab" * 32},
        use_pepper=False,
        hash_output_format="hex",
    )
    defaults.update(overrides)
    return HashBasedPseudonymizationOperation(**defaults)


# ----------------------- Initialization -----------------------


def test_init_sets_attributes():
    op = _make_op()
    assert op.field_name == "email"
    assert op.algorithm == "sha3_256"
    assert op.hash_output_format == "hex"
    assert op.use_pepper is False


def test_init_defaults_salt_config():
    op = HashBasedPseudonymizationOperation(field_name="email")
    assert op.salt_config is not None
    assert op.salt_config.get("source") == "parameter"


# ----------------------- Error paths -----------------------


def test_execute_invalid_field_returns_error(sample_df, task_dir, reporter):
    op = _make_op(field_name="not_a_field")
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.ERROR


def test_execute_invalid_algorithm_raises():
    with pytest.raises(Exception):
        HashBasedPseudonymizationOperation(
            field_name="email", algorithm="not_a_real_algo"
        )


# ----------------------- Happy path -----------------------


def test_execute_success_produces_csv_artifact(sample_df, task_dir, reporter):
    op = _make_op(field_name="email")
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    artifacts = result.get_artifacts_by_type("csv")
    assert artifacts, "No output CSV artifact produced."


def test_execute_produces_deterministic_hashes(sample_df, task_dir, reporter):
    """Same salt + pepper disabled → same input hashes to same output."""
    r1 = _make_op(field_name="email").execute(
        make_data_source(sample_df.copy()), task_dir / "run1", reporter
    )
    r2 = _make_op(field_name="email").execute(
        make_data_source(sample_df.copy()), task_dir / "run2", reporter
    )
    assert r1.status == OperationStatus.SUCCESS
    assert r2.status == OperationStatus.SUCCESS
    df1 = pd.read_csv(r1.get_artifacts_by_type("csv")[0].path)
    df2 = pd.read_csv(r2.get_artifacts_by_type("csv")[0].path)
    # Default mode=REPLACE → "email" column now holds the hash
    assert list(df1["email"]) == list(df2["email"])
    # Hashes differ from original plaintext values
    assert list(df1["email"]) != list(sample_df["email"])


def test_execute_output_format_hex_produces_hex_strings(
    sample_df, task_dir, reporter
):
    op = _make_op(field_name="email", hash_output_format="hex")
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    for val in df_out["email"].dropna():
        s = str(val)
        assert all(ch in "0123456789abcdef" for ch in s.lower()), (
            f"Non-hex char in output: {s}"
        )


# ----------------------- Security / validation (H2) -----------------------


def test_weak_salt_with_pepper_disabled_raises_on_execute(
    sample_df, task_dir, reporter
):
    """All-zero default salt + pepper disabled must be rejected at execute()."""
    op = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": "0" * 64},
        use_pepper=False,
    )
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    # Validation happens inside execute(); error is captured as result.ERROR.
    assert result.status == OperationStatus.ERROR


def test_weak_salt_with_pepper_enabled_is_allowed(sample_df, task_dir, reporter):
    """Default zero salt is acceptable if pepper provides per-session entropy."""
    op = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": "0" * 64},
        use_pepper=True,
    )
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS


def test_empty_salt_with_pepper_disabled_raises_on_execute(
    sample_df, task_dir, reporter
):
    op = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": ""},
        use_pepper=False,
    )
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.ERROR


# ----------------------- Cache invalidation (H1) -----------------------


def test_pepper_enabled_produces_different_hashes_across_runs(
    sample_df, task_dir, reporter
):
    """With use_pepper=True, each run generates a fresh pepper → different hashes.

    Verifies that the disk cache key includes the per-run session id so
    previous-run pseudonyms are not served from cache.
    """
    op1 = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": "ab" * 32},
        use_pepper=True,
    )
    op2 = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": "ab" * 32},
        use_pepper=True,
    )
    r1 = op1.execute(make_data_source(sample_df.copy()), task_dir / "run1", reporter)
    r2 = op2.execute(make_data_source(sample_df.copy()), task_dir / "run2", reporter)
    assert r1.status == OperationStatus.SUCCESS
    assert r2.status == OperationStatus.SUCCESS
    df1 = pd.read_csv(r1.get_artifacts_by_type("csv")[0].path)
    df2 = pd.read_csv(r2.get_artifacts_by_type("csv")[0].path)
    # Different pepper → different hashes for the same plaintext.
    assert list(df1["email"]) != list(df2["email"])


# ----------------------- process_batch defensive guard (H4) -----------------------


def test_process_batch_tolerates_none_additional_fields(sample_df):
    op = _make_op(field_name="email")
    op.additional_fields = None  # simulate config-reload edge case
    # Must initialize crypto components before calling process_batch directly.
    op._initialize_crypto_components()
    try:
        out = op.process_batch(sample_df.copy())
        assert "email" in out.columns
        # All emails should now be hashes (non-empty, non-original values)
        assert all(out["email"] != sample_df["email"])
    finally:
        # Clean up crypto memory the way execute()'s finally would
        op._salt = None
        op._hash_generator = None


# ----------------------- M2: class marker -----------------------


def test_is_pseudonymization_class_marker_set():
    """Hash-based op opts in to the pseudonymization null-handling branch."""
    assert HashBasedPseudonymizationOperation._is_pseudonymization is True


# ----------------------- M1: metric naming -----------------------


def _load_pseudo_metrics(result):
    """Helper: extract the pseudonymization metrics dict from result artifacts."""
    json_artifacts = [
        a for a in result.get_artifacts_by_type("json")
        if "metrics" in str(a.path).lower() and "data_types" not in str(a.path).lower()
    ]
    assert json_artifacts, "No metrics JSON artifact produced"
    blob = json.loads(json_artifacts[0].path.read_text(encoding="utf-8"))
    # Metrics file format: {"metadata": ..., "metrics": { ... "pseudonymization": {...} }}
    return blob.get("metrics", {}).get("pseudonymization", {})


def test_metrics_include_rows_processed_and_unique_values_hashed(
    sample_df, task_dir, reporter
):
    op = _make_op(field_name="email")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    pseudo = _load_pseudo_metrics(result)
    assert "rows_processed" in pseudo
    assert "unique_values_hashed" in pseudo
    # Old misleading key should be gone (catches regressions).
    assert "values_pseudonymized" not in pseudo


def test_pseudonymization_rate_full_when_all_non_null(
    sample_df, task_dir, reporter
):
    """All non-null originals get hashed → rate == 1.0."""
    op = _make_op(field_name="email")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    pseudo = _load_pseudo_metrics(result)
    assert pseudo.get("pseudonymization_rate") == 1.0


def test_pseudonymization_rate_with_null_preserve(task_dir, reporter):
    """null_strategy=PRESERVE keeps nulls untouched.

    Rate is computed only over non-null originals, so it should still be
    1.0 when every non-null input is hashed (nulls are excluded from
    both numerator and denominator).
    """
    df = pd.DataFrame({"email": ["a@x.com", None, "b@x.com", None, "c@x.com"]})
    op = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": "ab" * 32},
        use_pepper=False,
        null_strategy="PRESERVE",
    )
    result = op.execute(make_data_source(df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    pseudo = _load_pseudo_metrics(result)
    assert pseudo.get("pseudonymization_rate") == 1.0


def test_pseudonymization_rate_zero_when_all_null(task_dir, reporter):
    """Empty/all-null input → rate == 0.0 (nothing to pseudonymize)."""
    df = pd.DataFrame({"email": [None, None, None]})
    op = HashBasedPseudonymizationOperation(
        field_name="email",
        salt_config={"source": "parameter", "value": "ab" * 32},
        use_pepper=False,
        null_strategy="PRESERVE",
    )
    result = op.execute(make_data_source(df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    pseudo = _load_pseudo_metrics(result)
    assert pseudo.get("pseudonymization_rate") == 0.0


# ----------------------- Mode coverage -----------------------


def test_enrich_mode_adds_new_column_and_preserves_original(
    sample_df, task_dir, reporter
):
    op = _make_op(field_name="email", mode="ENRICH")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # Original field preserved
    assert list(df_out["email"]) == list(sample_df["email"])
    # Output column added (name set by op.output_field_name)
    assert op.output_field_name in df_out.columns
    # Pseudonyms differ from originals
    assert list(df_out[op.output_field_name]) != list(sample_df["email"])


def test_compound_mode_pseudonymizes_combined_fields(task_dir, reporter):
    df = pd.DataFrame(
        {
            "first": ["a", "a", "b", "b"],
            "last": ["x", "y", "x", "y"],
        }
    )
    op = HashBasedPseudonymizationOperation(
        field_name="first",
        additional_fields=["last"],
        compound_mode=True,
        salt_config={"source": "parameter", "value": "ab" * 32},
        use_pepper=False,
    )
    result = op.execute(make_data_source(df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # Each (first, last) compound is unique → 4 distinct pseudonyms.
    assert df_out["first"].nunique() == 4
    # additional_fields are nulled in REPLACE-compound mode.
    assert df_out["last"].isna().all()


# ----------------------- Output format coverage -----------------------


def test_base64_output_format(sample_df, task_dir, reporter):
    op = _make_op(field_name="email", hash_output_format="base64")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS


def test_uuid_output_format(sample_df, task_dir, reporter):
    import re

    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    op = _make_op(field_name="email", hash_output_format="uuid")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    for val in df_out["email"].dropna():
        assert uuid_re.match(str(val)), f"Not UUID format: {val}"


# ----------------------- Dask pickle round-trip -----------------------


def test_pickle_round_trip_preserves_config():
    """Op must survive pickle/unpickle for Dask distributed processing."""
    op = _make_op(field_name="email", hash_output_format="hex")
    data = pickle.dumps(op)
    restored = pickle.loads(data)
    assert restored.field_name == op.field_name
    assert restored.algorithm == op.algorithm
    assert restored.hash_output_format == op.hash_output_format
