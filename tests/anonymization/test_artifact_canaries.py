"""
SEC-03 / SEC-04 / SEC-05 (A4): PII and secret canaries must not reach non-data artifacts.

Representative masking and pseudonymization operations run on data that contains
unique canary strings (raw PII values, the hash salt, the mapping encryption
key). Every file the operation writes outside its data output (``config.json``,
``metrics/*.json``, reports, logs) and every log record is then searched,
recursively, for those canaries.

Pre-fix behaviour reproduced by these tests:

* ``HashBasedPseudonymizationOperation`` wrote ``salt_config.value`` in plaintext
  to ``config.json`` (SEC-03).
* ``metrics/*.json`` carried up to 10 raw pre-transformation values per field in
  ``field_info.original.top_values`` for every masking and pseudonymization
  operation (SEC-04).
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
    get_value_distribution_summary,
)
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.masking.partial_masking_op import (
    PartialMaskingOperation,
)
from pamola_core.anonymization.pseudonymization.hash_based_op import (
    HashBasedPseudonymizationOperation,
)
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

# Unique, greppable canaries. Low cardinality on purpose: the pre-fix leak only
# triggered for fields with <= 20 unique values, which is the common PII shape.
PII_VALUES = [
    "zqcanary.alice@pii-canary.test",
    "zqcanary.bob@pii-canary.test",
    "zqcanary.carol@pii-canary.test",
]
SALT_CANARY = "5a17ca4a47c0ffee5a17ca4a47beefc0de5a17ca4a47c0ffee5a17ca4a47beef"
MAPPING_KEY_CANARY = "6b2dcb5b58d1ff0f6b2dcb5b58cf0fd16b2dcb5b58d1ff0f6b2dcb5b58cf0fd1"

SECRET_CANARIES = [SALT_CANARY, MAPPING_KEY_CANARY]
# Substrings that identify a raw value (the local part is enough: masking may
# keep or alter the domain).
PII_FRAGMENTS = [v.split("@")[0] for v in PII_VALUES]


class _Reporter:
    def add_operation(self, *args, **kwargs):
        pass

    def add_artifact(self, *args, **kwargs):
        pass


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": range(1, 7),
            "email": PII_VALUES * 2,
        }
    )


def _data_source(df):
    return DataSource(dataframes={"main": df})


def _operations():
    return {
        "full_masking": lambda: FullMaskingOperation(field_name="email"),
        "partial_masking": lambda: PartialMaskingOperation(
            field_name="email",
            mask_strategy="fixed",
            unmasked_prefix=3,
            unmasked_suffix=0,
        ),
        "hash_pseudonymization": lambda: HashBasedPseudonymizationOperation(
            field_name="email",
            salt_config={"source": "parameter", "value": SALT_CANARY},
            use_pepper=False,
        ),
        "mapping_pseudonymization": lambda: ConsistentMappingPseudonymizationOperation(
            field_name="email",
            mapping_encryption_key=MAPPING_KEY_CANARY,
            pseudonym_type="uuid",
        ),
    }


def _non_data_artifacts(task_dir: Path):
    """Every file the operation wrote, except its data output directory."""
    output_dir = task_dir / "output"
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        if output_dir in path.parents:
            continue  # the transformed data itself is the operation's product
        yield path


def _contains(path: Path, needles):
    data = path.read_bytes()
    text = data.decode("utf-8", errors="ignore")
    return [n for n in needles if n in text or n.encode() in data]


@pytest.mark.parametrize("name", list(_operations()))
def test_secret_and_pii_canaries_absent_from_non_data_artifacts(
    name, df, tmp_path, caplog
):
    task_dir = tmp_path / "task"
    op = _operations()[name]()

    with caplog.at_level(logging.DEBUG):
        result = op.execute(_data_source(df), task_dir, _Reporter())

    assert result.status == OperationStatus.SUCCESS, result.error_message

    scanned = list(_non_data_artifacts(task_dir))
    assert any(p.name == "config.json" for p in scanned), "config.json not written"
    assert any(p.parent.name == "metrics" for p in scanned), "no metrics artifact"

    leaks = {
        str(p.relative_to(task_dir)): hits
        for p in scanned
        if (hits := _contains(p, SECRET_CANARIES + PII_FRAGMENTS))
    }
    assert leaks == {}, f"canaries leaked into non-data artifacts: {leaks}"

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert [c for c in SECRET_CANARIES + PII_FRAGMENTS if c in logged] == []

    # Also the result object that callers serialize into reports.
    reported = json.dumps(result.metrics, default=str)
    assert [c for c in SECRET_CANARIES + PII_FRAGMENTS if c in reported] == []


def test_hash_salt_is_redacted_in_config_json(df, tmp_path):
    """SEC-03: the salt lives under salt_config.value; config.json must not carry it."""
    task_dir = tmp_path / "task"
    op = _operations()["hash_pseudonymization"]()
    result = op.execute(_data_source(df), task_dir, _Reporter())
    assert result.status == OperationStatus.SUCCESS

    cfg = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
    assert SALT_CANARY not in json.dumps(cfg)
    assert cfg["salt_config"] == "***REDACTED***" or "REDACT" in str(cfg["salt_config"]).upper()
    # The operation itself must still use the real salt.
    assert op.salt_config["value"] == SALT_CANARY


def test_hash_config_declares_salt_as_sensitive():
    from pamola_core.anonymization.schemas.hash_based_op_core_schema import (
        HashBasedPseudonymizationConfig,
    )

    assert "salt_config" in HashBasedPseudonymizationConfig.SENSITIVE_KEYS


# --- SEC-04 / SEC-05: raw samples in metrics are opt-in ---------------------


def test_distribution_summary_has_no_values_by_default():
    series = pd.Series(PII_VALUES * 3)
    summary = get_value_distribution_summary(series)
    assert "top_values" not in summary
    assert summary["unique_count"] == 3
    assert PII_FRAGMENTS[0] not in json.dumps(summary)


def test_distribution_summary_raw_samples_only_when_requested():
    series = pd.Series(PII_VALUES * 3)
    summary = get_value_distribution_summary(series, include_raw_samples=True)
    assert set(summary["top_values"]) == set(PII_VALUES)


def test_collect_operation_metrics_default_has_no_raw_values():
    original = pd.Series(PII_VALUES * 2)
    processed = pd.Series(["***"] * 6)
    metrics = collect_operation_metrics(
        operation_type="masking",
        original_data=original,
        processed_data=processed,
        operation_params={},
    )
    assert PII_FRAGMENTS[0] not in json.dumps(metrics, default=str)
    assert "top_values" not in metrics["field_info"]["original"]


@pytest.mark.parametrize("name", list(_operations()))
def test_raw_samples_option_defaults_false_and_is_in_schema(name):
    op = _operations()[name]()
    assert op.include_raw_samples_in_metrics is False
    prop = op.config.schema["allOf"][1]["properties"]["include_raw_samples_in_metrics"]
    assert prop["type"] == "boolean"
    assert prop["default"] is False
    assert "WARNING" in prop["description"]


def test_raw_samples_opt_in_reaches_metrics_artifact(df, tmp_path):
    """The diagnostic option still works, and is the only way to get raw values."""
    task_dir = tmp_path / "task"
    op = FullMaskingOperation(field_name="email", include_raw_samples_in_metrics=True)
    result = op.execute(_data_source(df), task_dir, _Reporter())
    assert result.status == OperationStatus.SUCCESS

    metrics_files = [p for p in task_dir.rglob("*.json") if p.parent.name == "metrics"]
    assert metrics_files
    combined = "".join(p.read_text(encoding="utf-8") for p in metrics_files)
    assert PII_FRAGMENTS[0] in combined
