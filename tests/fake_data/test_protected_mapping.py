"""
fake_data ``save_mapping=True`` writes a PROTECTED_MAPPING: encrypted with the task key,
refused without one, never plain text.

Before: ``MappingStore.save_json`` wrote ``maps/*.json`` in clear text, original values
included.
"""

import pandas as pd
import pytest

from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.fake_data.operations.email_op import FakeEmailOperation
from pamola_core.utils.artifact_policy import ArtifactClass, ArtifactPolicyError
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

PII = [
    "zqcanary.alice@pii-canary.test",
    "zqcanary.bob@pii-canary.test",
    "zqcanary.carol@pii-canary.test",
]
KEY = "unit-test-task-key"


class _Reporter:
    def __init__(self):
        self.calls = []

    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        self.calls.append(k)


def _op(**kwargs):
    return FakeEmailOperation(
        field_name="email", consistency_mechanism="mapping", key="k", **kwargs
    )


def _run(tmp_path, op, reporter=None):
    df = pd.DataFrame({"id": range(6), "email": PII * 2})
    task_dir = tmp_path / "task"
    result = op.execute(DataSource(dataframes={"main": df}), task_dir, reporter or _Reporter())
    assert result.status == OperationStatus.SUCCESS, result.error_message
    return result, task_dir


def test_save_mapping_without_the_task_key_is_a_typed_error():
    with pytest.raises(ArtifactPolicyError, match="encrypted"):
        _op(save_mapping=True)
    with pytest.raises(ArtifactPolicyError):
        _op(save_mapping=True, use_encryption=True)  # no key


def test_save_mapping_with_the_key_writes_no_plain_text(tmp_path):
    reporter = _Reporter()
    result, task_dir = _run(
        tmp_path, _op(save_mapping=True, use_encryption=True, encryption_key=KEY), reporter
    )
    maps = list((task_dir / "maps").glob("*"))
    assert maps, "the mapping file was not written"
    for path in maps:
        assert b"zqcanary" not in path.read_bytes(), "original value found in the mapping file"
    mapping = [a for a in result.artifacts if a.category == "mapping"]
    assert len(mapping) == 1
    assert mapping[0].artifact_class is ArtifactClass.PROTECTED_MAPPING
    assert mapping[0].contains_source_values is True
    assert any(c.get("artifact_class") is ArtifactClass.PROTECTED_MAPPING for c in reporter.calls)


def test_the_encrypted_mapping_round_trips_with_the_key_only(tmp_path):
    op = _op(save_mapping=True, use_encryption=True, encryption_key=KEY)
    _, task_dir = _run(tmp_path, op)
    path = next((task_dir / "maps").glob("*"))
    store = MappingStore()
    store.load(path, format="json", encryption_key=KEY)
    assert sum(len(m) for m in store.mappings.values()) == 3
    other = MappingStore()
    with pytest.raises(Exception):
        other.load(path, format="json", encryption_key="a-wrong-key")


def test_default_run_writes_no_mapping_file(tmp_path):
    _, task_dir = _run(tmp_path, _op())
    assert not (task_dir / "maps").exists()
