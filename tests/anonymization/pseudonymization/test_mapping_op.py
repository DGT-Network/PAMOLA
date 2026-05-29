"""
File: test_mapping_op.py
Test Target: pseudonymization/mapping_op.py
Version: 1.1
Coverage Status: Smoke + security + mode + persistence + edge-case tests
"""

import json
import pickle

import pytest
import pandas as pd
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.errors.exceptions import ValidationError
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


# 256-bit encryption key (64 hex chars)
TEST_KEY_HEX = "0" * 64
VALID_KEY_HEX = "ab" * 32


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
    defaults = dict(
        field_name="email",
        mapping_encryption_key=VALID_KEY_HEX,
        pseudonym_type="uuid",
    )
    defaults.update(overrides)
    return ConsistentMappingPseudonymizationOperation(**defaults)


# ----------------------- Initialization -----------------------


def test_init_with_hex_key():
    op = _make_op()
    assert op.field_name == "email"
    assert op.pseudonym_type == "uuid"
    # Public hex attribute is deleted after init (security)
    assert not hasattr(op, "mapping_encryption_key")
    assert op._mapping_encryption_key == bytes.fromhex(VALID_KEY_HEX)


def test_init_with_bytes_key():
    key_bytes = bytes.fromhex(VALID_KEY_HEX)
    op = _make_op(mapping_encryption_key=key_bytes)
    assert op._mapping_encryption_key == key_bytes


def test_init_invalid_hex_key_raises():
    with pytest.raises(ValidationError):
        _make_op(mapping_encryption_key="not_hex")


def test_init_invalid_key_size_raises():
    # 128-bit key instead of 256
    with pytest.raises(ValidationError):
        _make_op(mapping_encryption_key="ab" * 16)


# ----------------------- Error paths -----------------------


def test_execute_invalid_field_returns_error(sample_df, task_dir, reporter):
    op = _make_op(field_name="not_a_field")
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.ERROR


# ----------------------- Happy path -----------------------


def test_execute_success_produces_csv_artifact(sample_df, task_dir, reporter):
    op = _make_op(field_name="email")
    ds = make_data_source(sample_df)
    result = op.execute(ds, task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    artifacts = result.get_artifacts_by_type("csv")
    assert artifacts, "No output CSV artifact produced."


def test_execute_consistent_mapping_same_input_same_pseudonym(
    sample_df, task_dir, reporter
):
    """Same value appearing multiple times → same pseudonym."""
    df = pd.DataFrame({"email": ["a@x.com", "b@x.com", "a@x.com", "b@x.com"]})
    op = _make_op(field_name="email")
    result = op.execute(make_data_source(df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # Default mode=REPLACE → "email" now holds the pseudonym
    # Row 0 and row 2 share same input → same pseudonym
    assert df_out["email"].iloc[0] == df_out["email"].iloc[2]
    # Row 1 and row 3 share same input → same pseudonym
    assert df_out["email"].iloc[1] == df_out["email"].iloc[3]
    # Different inputs → different pseudonyms
    assert df_out["email"].iloc[0] != df_out["email"].iloc[1]


def test_execute_uuid_pseudonym_type_produces_uuid_format(
    sample_df, task_dir, reporter
):
    import re

    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )
    op = _make_op(field_name="email", pseudonym_type="uuid")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    for val in df_out["email"].dropna():
        assert uuid_re.match(str(val)), f"Not a valid UUID: {val}"


# ----------------------- C1: encryption key NOT on disk -----------------------


def test_config_json_does_not_contain_encryption_key(sample_df, task_dir, reporter):
    """Saved config.json must NOT contain the AES-256 mapping encryption key."""
    op = _make_op(field_name="email")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    config_path = task_dir / "config.json"
    assert config_path.exists(), "save_config() did not write config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    # Hex key MUST NOT appear anywhere in the serialized config (defense
    # against accidental nesting / future schema changes).
    raw = config_path.read_text(encoding="utf-8")
    assert VALID_KEY_HEX not in raw, "Encryption key leaked into config.json"
    # The redacted placeholder should be present for the key field.
    assert saved.get("mapping_encryption_key") == "*REDACTED*"


def test_to_safe_dict_redacts_sensitive_keys():
    op = _make_op()
    safe = op.config.to_safe_dict()
    assert safe["mapping_encryption_key"] == "*REDACTED*"
    # Non-sensitive keys must remain untouched.
    assert safe["field_name"] == "email"


# ----------------------- M2: class marker -----------------------


def test_is_pseudonymization_class_marker_set():
    assert ConsistentMappingPseudonymizationOperation._is_pseudonymization is True


# ----------------------- Pseudonym type coverage -----------------------


def test_sequential_pseudonym_type(sample_df, task_dir, reporter):
    op = _make_op(field_name="email", pseudonym_type="sequential")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # Each distinct input gets a distinct sequential pseudonym; sample has 5 unique emails.
    assert df_out["email"].nunique() == 5


def test_random_string_pseudonym_type(sample_df, task_dir, reporter):
    op = _make_op(
        field_name="email", pseudonym_type="random_string", pseudonym_length=12
    )
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # Distinct inputs → distinct pseudonyms.
    assert df_out["email"].nunique() == 5


# ----------------------- Mode coverage -----------------------


def test_enrich_mode_preserves_original_and_adds_pseudonym_column(
    sample_df, task_dir, reporter
):
    op = _make_op(field_name="email", mode="ENRICH")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    assert list(df_out["email"]) == list(sample_df["email"])
    assert op.output_field_name in df_out.columns


def test_compound_mode_pseudonymizes_combined_identifiers(task_dir, reporter):
    df = pd.DataFrame(
        {
            "first": ["a", "a", "b", "b"],
            "last": ["x", "y", "x", "y"],
        }
    )
    op = ConsistentMappingPseudonymizationOperation(
        field_name="first",
        additional_fields=["last"],
        compound_mode=True,
        mapping_encryption_key=VALID_KEY_HEX,
    )
    result = op.execute(make_data_source(df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # 4 unique (first, last) combos → 4 unique pseudonyms.
    assert df_out["first"].nunique() == 4
    assert df_out["last"].isna().all()


# ----------------------- Reverse mapping -----------------------


def test_reverse_mapping_recovers_original_values(sample_df, task_dir, reporter):
    op = _make_op(field_name="email")
    result = op.execute(make_data_source(sample_df), task_dir, reporter)
    assert result.status == OperationStatus.SUCCESS
    df_out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)
    # Reverse mapping should map every pseudonym back to its original.
    reverse = op._reverse_mapping
    assert len(reverse) >= sample_df["email"].nunique()
    for original, pseudonym in zip(sample_df["email"], df_out["email"]):
        assert reverse[pseudonym] == original


# ----------------------- additional_fields normalization (H4 / M4) -----------------------


def test_none_additional_fields_normalized_to_empty_list():
    op = _make_op(additional_fields=None, quasi_identifiers=None)
    # Constructor must normalize None → [] so process_batch and metrics don't crash.
    assert op.additional_fields == []
    assert op.quasi_identifiers == []


def test_process_batch_tolerates_none_additional_fields_runtime(sample_df):
    op = _make_op(field_name="email")
    op.additional_fields = None  # simulate config-reload edge case
    # process_batch must not crash with None.
    op._initialize_runtime_state = lambda *a, **k: None  # bypass crypto init if any
    # Need the basic runtime state used by the lookup path.
    out = op.process_batch(sample_df.copy())
    assert "email" in out.columns


# ----------------------- Dask pickle round-trip -----------------------


def test_pickle_round_trip_preserves_op_and_key():
    op = _make_op()
    data = pickle.dumps(op)
    restored = pickle.loads(data)
    assert restored.field_name == op.field_name
    assert restored.pseudonym_type == op.pseudonym_type
    # Private encryption key bytes survive pickle (needed for distributed Dask).
    assert restored._mapping_encryption_key == op._mapping_encryption_key
    # Public hex attribute is NOT restored (security).
    assert not hasattr(restored, "mapping_encryption_key")
