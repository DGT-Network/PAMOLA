"""CDX-AC-8: output-encryption keys never reach the (unencrypted) cache metadata.

``_get_base_parameters()`` carried the raw ``encryption_key`` and the cache layer
wrote those parameters to a plain JSON file, bypassing the redaction that protects
``config.json``. The key now influences the cache identity only through a
non-secret digest, and the stored parameters are redacted centrally.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.utils.helpers import (
    build_base_cache,
    is_sensitive_parameter,
    redact_sensitive_parameters,
    secret_identity,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

CANARY = "KEY-CANARY-0123456789abcdef0123456789abcdef"


def _tree_text(root: Path) -> str:
    chunks = []
    for path in root.rglob("*"):
        if path.is_file():
            chunks.append(path.name)
            chunks.append(path.read_bytes().decode("utf-8", errors="ignore"))
    return "\n".join(chunks)


def test_base_parameters_hold_a_digest_not_the_key():
    op = FullMaskingOperation(
        field_name="name", use_encryption=True, encryption_key=CANARY, use_cache=True
    )
    params = op._get_base_parameters()
    assert "encryption_key" not in params
    assert CANARY not in json.dumps(params, default=str)
    assert params["encryption_key_id"] == secret_identity(CANARY)
    assert params["encryption_key_id"] != CANARY


def test_a_different_key_gives_a_different_cache_identity():
    a = FullMaskingOperation(field_name="name", use_encryption=True, encryption_key="k1" * 16)
    b = FullMaskingOperation(field_name="name", use_encryption=True, encryption_key="k2" * 16)
    assert a._get_base_parameters() != b._get_base_parameters()


def test_cache_files_never_contain_the_key(tmp_path):
    op = FullMaskingOperation(
        field_name="name", use_encryption=True, encryption_key=CANARY, use_cache=True
    )
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"] * 4})
    result = op.execute(DataSource(dataframes={"main": df}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message

    assert list((tmp_path / "cache").rglob("*.json")), "the test must have written a cache entry"
    assert CANARY not in _tree_text(tmp_path / "cache")


def test_store_level_redaction_catches_a_secret_the_caller_forgot(tmp_path):
    cache = OperationCache(cache_dir=tmp_path)
    cache.save_cache(
        data={
            "parameters": {
                "encryption_key": CANARY,
                "mapping_encryption_key": CANARY,
                "nested": {"api_key": CANARY, "ok": 1},
                "visualization_theme": "dark",
            },
            "status": "SUCCESS",
        },
        cache_key="abc",
        operation_type="X",
    )
    text = _tree_text(tmp_path)
    assert CANARY not in text
    assert "dark" in text  # non-secret parameters survive


def test_build_base_cache_redacts():
    cache = build_base_cache(
        {"password": CANARY, "salt": CANARY, "chunk_size": 10},
        OperationResult(status=OperationStatus.SUCCESS),
    )
    assert CANARY not in json.dumps(cache)
    assert cache["parameters"]["chunk_size"] == 10


@pytest.mark.parametrize(
    ("name", "sensitive"),
    [
        ("encryption_key", True),
        ("mapping_encryption_key", True),
        ("password", True),
        ("db_password", True),
        ("api_key", True),
        ("salt", True),
        ("pepper", True),
        ("token", True),
        ("key_fields", False),
        ("encryption_key_id", False),
        ("encryption_mode", False),
        ("chunk_size", False),
    ],
)
def test_sensitive_name_rule(name, sensitive):
    assert is_sensitive_parameter(name) is sensitive


def test_declared_sensitive_keys_are_redacted_too():
    out = redact_sensitive_parameters({"custom_secret_field": "v", "x": 1}, extra_keys=["custom_secret_field"])
    assert out == {"custom_secret_field": "*REDACTED*", "x": 1}


def test_metrics_cache_does_not_persist_the_key_either(tmp_path):
    original = pd.DataFrame({"x": [float(i) for i in range(50)]})
    op = FidelityOperation(
        fidelity_metrics=["ks"], use_cache=True, columns=["x"],
        use_encryption=True, encryption_key=CANARY,
    )
    op.execute(
        DataSource(dataframes={"main": original, "synth": original.copy()}),
        tmp_path, None,
        original_dataset_name="main", transformed_dataset_name="synth",
    )
    assert CANARY not in _tree_text(tmp_path / "cache")
