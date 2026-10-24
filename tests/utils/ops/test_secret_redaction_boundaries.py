"""CDX-R2-4: one redaction, applied at every diagnostic serialization boundary.

Before: ``config.json`` of masking / hashing / mapping operations and the reporter's
``parameters`` event carried the output-encryption key (only the cache writers were fixed).
"""

import json

import pandas as pd
import pytest

from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.pseudonymization.hash_based_op import (
    HashBasedPseudonymizationOperation,
)
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.redaction import (
    REDACTED,
    is_sensitive_parameter,
    redact_sensitive_parameters,
)

CANARY = "KEY-CANARY-r2-0123456789abcdef0123456789abcdef"


class _Recorder:
    def __init__(self):
        self.events = []

    def add_operation(self, *args, **kwargs):
        self.events.append((args, kwargs))

    def add_artifact(self, *args, **kwargs):
        self.events.append((args, kwargs))


def _everything(task_dir, reporter):
    text = [repr(reporter.events)]
    for path in task_dir.rglob("*"):
        if path.is_file():
            text.append(path.name)
            text.append(path.read_bytes().decode("utf-8", errors="ignore"))
    return "\n".join(text)


_OPS = {
    "masking": lambda: FullMaskingOperation(field_name="name", use_encryption=True, encryption_key=CANARY),
    "hash": lambda: HashBasedPseudonymizationOperation(
        field_name="name", use_encryption=True, encryption_key=CANARY
    ),
}


@pytest.mark.parametrize("case", sorted(_OPS))
def test_config_json_and_reporter_events_never_hold_the_key(case, tmp_path):
    op = _OPS[case]()
    reporter = _Recorder()
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"] * 4})
    op.run(
        data_source=DataSource(dataframes={"main": df}), task_dir=tmp_path, reporter=reporter,
        track_progress=False, encryption_key=CANARY, api_token=CANARY, label="ok",
    )
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert CANARY not in json.dumps(config)
    assert CANARY not in _everything(tmp_path, reporter)
    params = reporter.events[0][1]["details"]["parameters"]
    assert params["encryption_key"] == REDACTED and params["api_token"] == REDACTED
    assert params["label"] == "ok"  # non-secrets survive


def test_config_safe_dict_redacts_by_name_and_by_declaration():
    class Cfg(OperationConfig):
        schema = {"type": "object"}
        SENSITIVE_KEYS = frozenset({"custom_field"})

    cfg = Cfg(encryption_key=CANARY, custom_field="declared", nested={"api_key": CANARY, "ok": 1}, tau=0.5)
    safe = cfg.to_safe_dict()
    assert safe["encryption_key"] == safe["custom_field"] == REDACTED
    assert safe["nested"] == {"api_key": REDACTED, "ok": 1}
    assert safe["tau"] == 0.5
    assert cfg.to_dict()["encryption_key"] == CANARY  # the live value is untouched


def test_task_report_parameters_are_redacted():
    from pamola_core.utils.tasks.task_runner import TaskRunner

    runner = TaskRunner.__new__(TaskRunner)
    runner.lst_result = []
    runner._add_operation_result(
        class_name="FullMaskingOperation", operation_type="masking", operation={},
        scope={}, parameters={"encryption_key": CANARY, "mode": "REPLACE"},
    )
    assert CANARY not in json.dumps(runner.lst_result, default=str)
    assert runner.lst_result[0]["parameters"]["mode"] == "REPLACE"


@pytest.mark.parametrize(("name", "expected"), [
    ("encryption_key", True), ("mapping_encryption_key", True), ("api_token", True), ("db_password", True),
    ("user_secret", True), ("Salt", True), ("key_fields", False), ("tau", False), ("encryption_mode", False),
])
def test_the_name_rule(name, expected):
    assert is_sensitive_parameter(name) is expected


def test_redaction_reaches_lists_and_leaves_none_alone():
    out = redact_sensitive_parameters({"a": [{"password": "x"}, 1], "token": None})
    assert out == {"a": [{"password": REDACTED}, 1], "token": None}
