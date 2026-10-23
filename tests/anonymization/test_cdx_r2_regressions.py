"""
Regressions for the second adversarial review (CDX-R2-1, 2, 3a, 11).

Reproductions follow the review's probes: real Dask execution with the process
scheduler, creation policy, cache identity of the mapping file, fail-open
fake-data generation.
"""

import logging

import pandas as pd
import pytest

from pamola_core.anonymization.commons.mapping_storage import MappingStorage
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.fake_data.operations.name_op import FakeNameOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

KEY = "ab" * 32


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _op(path, **kwargs):
    return ConsistentMappingPseudonymizationOperation(
        field_name="email",
        mapping_encryption_key=KEY,
        mapping_file=str(path),
        **kwargs,
    )


def _run(op, df, task_dir):
    return op.execute(DataSource(dataframes={"main": df}), task_dir, _Reporter())


def _out(result):
    return pd.read_csv(result.get_artifacts_by_type("csv")[0].path)["email"].tolist()


# --- CDX-R2-1: Dask --------------------------------------------------------


@pytest.mark.parametrize("scheduler", ["threads", "processes"])
def test_dask_mapping_is_reversible_and_has_no_phantom_entry(tmp_path, scheduler):
    dask = pytest.importorskip("dask")
    path = tmp_path / "m.enc"
    values = ["id-a", "id-b", "id-c", "id-d"]
    df = pd.DataFrame({"email": values})
    op = _op(path, pseudonym_type="sequential", use_dask=True, npartitions=2)

    with dask.config.set(scheduler=scheduler):
        result = _run(op, df, tmp_path / "t")

    assert result.status == OperationStatus.SUCCESS, result.error_message
    stored = MappingStorage(path, bytes.fromhex(KEY)).load()
    assert set(stored) == set(values), f"unexpected entries: {sorted(stored)}"
    out = _out(result)
    assert len(set(out)) == 4
    assert out == [int(stored[v]) for v in values]  # sequential ids, one per source


# --- CDX-R2-2: creation policy ----------------------------------------------


def test_missing_mapping_with_creation_disabled_is_an_error(tmp_path):
    path = tmp_path / "must-exist.enc"
    op = _op(path, create_if_not_exists=False, pseudonym_type="sequential")

    result = _run(op, pd.DataFrame({"email": ["a", "b"]}), tmp_path / "t")

    assert result.status == OperationStatus.ERROR
    assert not path.exists(), "a replacement mapping must not be created"
    assert not result.get_artifacts_by_type("csv")


def test_existing_mapping_with_creation_disabled_is_used(tmp_path):
    path = tmp_path / "m.enc"
    first = _run(_op(path), pd.DataFrame({"email": ["a", "b"]}), tmp_path / "t1")
    assert first.status == OperationStatus.SUCCESS
    second = _run(
        _op(path, create_if_not_exists=False),
        pd.DataFrame({"email": ["b", "a"]}),
        tmp_path / "t2",
    )
    assert second.status == OperationStatus.SUCCESS, second.error_message
    assert _out(second) == _out(first)[::-1]


# --- CDX-R2-3a: result cache cannot alias mapping files ----------------------


def test_changing_mapping_file_never_reuses_cached_pseudonyms(tmp_path):
    df = pd.DataFrame({"email": ["alice", "bob"]})
    seen = {}
    for name in ("a.enc", "b.enc"):
        path = tmp_path / name
        op = _op(path, pseudonym_type="sequential", use_cache=True)
        result = _run(op, df, tmp_path / f"t-{name}")
        assert result.status == OperationStatus.SUCCESS, result.error_message
        assert not result.metrics.get("cached")
        # the table in use really issued the pseudonyms that were returned
        stored = MappingStorage(path, bytes.fromhex(KEY)).load()
        assert set(stored) == {"alice", "bob"}
        seen[name] = _out(result)
        assert seen[name] == [int(stored["alice"]), int(stored["bob"])]
    assert op.use_cache is False


# --- CDX-R2-11: fake data fails closed --------------------------------------


@pytest.mark.parametrize("failure", [RuntimeError("dictionary unavailable")])
def test_fake_name_generation_failure_never_writes_or_logs_originals(
    tmp_path, caplog, monkeypatch, failure
):
    canary = ["Zq Canary Alice", "Zq Canary Bob"]
    df = pd.DataFrame({"name": canary})
    op = FakeNameOperation(field_name="name")

    def boom(self, value, **params):
        raise failure

    monkeypatch.setattr(FakeNameOperation, "process_value", boom)
    with caplog.at_level(logging.DEBUG):
        result = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())

    assert result.status == OperationStatus.ERROR
    assert not result.get_artifacts_by_type("csv")
    assert "Canary" not in "\n".join(r.getMessage() for r in caplog.records)
    assert "Canary" not in str(result.error_message)
    for path in (tmp_path / "t").rglob("*"):
        if path.is_file():
            assert b"Canary" not in path.read_bytes(), path


@pytest.mark.parametrize(
    "module,cls,field",
    [
        ("email_op", "FakeEmailOperation", "email"),
        ("organization_op", "FakeOrganizationOperation", "org"),
        ("phone_op", "FakePhoneOperation", "phone"),
    ],
)
def test_other_fake_operations_fail_closed_too(tmp_path, caplog, monkeypatch, module, cls, field):
    import importlib

    klass = getattr(importlib.import_module(f"pamola_core.fake_data.operations.{module}"), cls)
    df = pd.DataFrame({field: ["CANARY-ONE", "CANARY-TWO"]})
    monkeypatch.setattr(klass, "process_value", lambda self, v, **p: (_ for _ in ()).throw(RuntimeError("x")))
    with caplog.at_level(logging.DEBUG):
        result = klass(field_name=field).execute(
            DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter()
        )
    assert result.status == OperationStatus.ERROR
    assert "CANARY" not in "\n".join(r.getMessage() for r in caplog.records)
