"""F1 end to end: a diagnostic profiling artifact keeps its class and both flags through the
operation result, the reporter, the real ``BaseTask.finalize`` and ``manifest.json``."""

import json
import logging
import time

import pandas as pd

from pamola_core.profiling.analyzers.group import GroupAnalyzerOperation
from pamola_core.utils.artifact_policy import ArtifactClass
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.tasks.task_reporting import TaskReporter

MARK = "R1TASKSECRET"


def test_diagnostic_class_and_flags_survive_reporter_finalize_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(tmp_path / "execution_log.json"))
    frame = pd.DataFrame(
        {"group": [MARK + "A"] * 10 + [MARK + "B"] * 10, "age": list(range(20))}
    )
    out = tmp_path / "group"
    out.mkdir()
    reporter = TaskReporter("group", "review", "finalization", out / "task_report.json")
    op = GroupAnalyzerOperation(field_name="group", fields_config={"age": 1})
    result = op.execute(DataSource(dataframes={"main": frame}), out, reporter)
    assert result.status.name == "SUCCESS", result.error_message

    holders = [a for a in result.artifacts if MARK.encode() in a.path.read_bytes()]
    assert holders, "the probe produced no artifact with source values"
    for art in holders:  # at the operation boundary
        assert art.artifact_class is ArtifactClass.DIAGNOSTIC
        assert art.contains_source_values is True

    task = BaseTask("group", "review", "finalization", task_dir=str(out))
    task.task_dir = out
    task.start_time = time.time()
    task.logger = logging.getLogger("task-finalize-test")
    task.reporter = reporter
    task.artifacts = result.artifacts
    task.metrics = {"group": result.metrics}
    task.finalize(True)

    manifest = {
        e["path"]: e
        for e in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
    }
    for art in holders:
        path = str(art.path)
        entries = [e for e in reporter.artifacts if e["path"] == path]
        assert len(entries) == 1, "one reporter entry per path"
        for entry in (entries[0], manifest[path]):
            assert entry["artifact_class"] == "diagnostic"
            assert entry["contains_source_values"] is True
            assert entry["not_for_release"] is True
