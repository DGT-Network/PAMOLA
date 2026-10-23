"""
MAP-01 / MAP-02 / MAP-03: reversible mapping pseudonymization across instances.

A mapping persisted by one operation instance must be loadable by a fresh
instance (a new process in practice) with the same key, in every supported
format, and must fail closed when the key is wrong or the file is damaged.

Pre-fix behaviour reproduced by these tests: with the documented default
``mapping_format="csv"`` a fresh instance could not reload the mapping
(``AttributeError: 'str' object has no attribute 'items'``), and the JSON format
worked only because two wrapping layers happened to use the same key names.
"""

import json
import struct
from pathlib import Path

import pandas as pd
import pytest

from pamola_core.anonymization.commons import mapping_storage as ms
from pamola_core.anonymization.commons.mapping_storage import (
    MAGIC,
    MappingIntegrityError,
    MappingStorage,
)
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.errors.exceptions import MappingStorageError
from pamola_core.utils.crypto_helpers.pseudonymization import MappingEncryption
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

KEY = "ab" * 32
OTHER_KEY = "cd" * 32


class _Reporter:
    def add_operation(self, *args, **kwargs):
        pass

    def add_artifact(self, *args, **kwargs):
        pass


def _op(mapping_file, key=KEY, **kwargs):
    return ConsistentMappingPseudonymizationOperation(
        field_name="email",
        mapping_encryption_key=key,
        mapping_file=str(mapping_file),
        **kwargs,
    )


def _run(op, values, task_dir):
    df = pd.DataFrame({"email": list(values)})
    result = op.execute(DataSource(dataframes={"main": df}), task_dir, _Reporter())
    return result, df


def _output(result):
    return pd.read_csv(result.get_artifacts_by_type("csv")[0].path)["email"].tolist()


def _stored(path, fmt):
    return MappingStorage(path, bytes.fromhex(KEY), format=fmt).load()


FORMATS = [None, "csv", "json"]  # None = the documented default


def _fmt_kwargs(fmt):
    return {} if fmt is None else {"mapping_format": fmt}


def _fmt_name(fmt):
    return fmt or "csv"


# --------------------------------------------------------------------------
# MAP-01: cross-instance round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f or "default")
def test_fresh_instance_reloads_the_exact_mapping(fmt, tmp_path):
    path = tmp_path / "shared.map.enc"
    a, _ = _run(
        _op(path, **_fmt_kwargs(fmt)), ["ann", "bob", "cy", "bob"], tmp_path / "a"
    )
    assert a.status == OperationStatus.SUCCESS, a.error_message
    out_a = _output(a)

    # A brand new instance and a different task directory: nothing in memory.
    b, _ = _run(_op(path, **_fmt_kwargs(fmt)), ["cy", "ann", "bob"], tmp_path / "b")
    assert b.status == OperationStatus.SUCCESS, b.error_message
    out_b = _output(b)

    assert out_b == [out_a[2], out_a[0], out_a[1]]  # same pseudonym per value
    assert len(set(out_a)) == 3
    assert out_a[1] == out_a[3]  # duplicate source value -> one pseudonym
    stored = _stored(path, _fmt_name(fmt))
    assert set(stored) == {"ann", "bob", "cy"}
    assert {stored[v] for v in stored} == set(out_a)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f or "default")
def test_second_instance_extends_without_changing_existing_pseudonyms(fmt, tmp_path):
    path = tmp_path / "shared.map.enc"
    a, _ = _run(_op(path, **_fmt_kwargs(fmt)), ["ann", "bob"], tmp_path / "a")
    before = _stored(path, _fmt_name(fmt))

    op_b = _op(path, **_fmt_kwargs(fmt))
    b, _ = _run(op_b, ["bob", "dee", "eve"], tmp_path / "b")
    assert b.status == OperationStatus.SUCCESS, b.error_message

    after = _stored(path, _fmt_name(fmt))
    assert {k: after[k] for k in before} == before  # continuity
    assert set(after) == {"ann", "bob", "dee", "eve"}
    assert len(set(after.values())) == 4  # still one-to-one
    # Reversal works on the fresh instance too.
    assert op_b.get_reverse_mapping(after["dee"]) == "dee"


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f or "default")
def test_sequential_numbering_continues_across_instances(fmt, tmp_path):
    path = tmp_path / "seq.map.enc"
    kwargs = dict(pseudonym_type="sequential", **_fmt_kwargs(fmt))
    a, _ = _run(_op(path, **kwargs), ["x", "y", "z"], tmp_path / "a")
    assert _output(a) == [1, 2, 3]

    b, _ = _run(_op(path, **kwargs), ["z", "w"], tmp_path / "b")
    assert _output(b) == [3, 4]


def test_export_mappings_round_trips_through_storage(tmp_path):
    path = tmp_path / "shared.map.enc"
    op = _op(path, mapping_format="json")
    _run(op, ["ann", "bob"], tmp_path / "a")
    expected = _stored(path, "json")

    export = tmp_path / "export.enc"
    op._mapping_storage = None
    op.export_mappings(export)

    assert MappingStorage(export, bytes.fromhex(KEY), format="json").load() == expected


# --------------------------------------------------------------------------
# MAP-02: wrong key and damaged files fail closed
# --------------------------------------------------------------------------


def _make_mapping_file(tmp_path, fmt="csv", persist_frequency=2):
    path = tmp_path / "m.map.enc"
    result, _ = _run(
        _op(path, mapping_format=fmt, persist_frequency=persist_frequency),
        [f"user{i}" for i in range(7)],
        tmp_path / "a",
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    return path


def _frames(path):
    """(offset, length) of every frame in a version 2 file."""
    raw = path.read_bytes()
    pos, out = len(MAGIC), []
    while pos < len(raw):
        (n,) = struct.unpack_from(">I", raw, pos)
        out.append((pos, n))
        pos += 4 + n
    return out


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_wrong_key_fails_closed_with_a_typed_actionable_error(fmt, tmp_path):
    path = _make_mapping_file(tmp_path, fmt)

    with pytest.raises(MappingIntegrityError) as exc:
        MappingStorage(path, bytes.fromhex(OTHER_KEY), format=fmt).load()
    assert "key" in str(exc.value).lower()
    assert "No mapping was loaded" in str(exc.value)

    result, _ = _run(_op(path, key=OTHER_KEY, mapping_format=fmt), ["user1"], tmp_path / "b")
    assert result.status == OperationStatus.ERROR
    assert "key" in result.error_message.lower()
    assert not result.artifacts  # no output produced from a half-loaded mapping


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_wrong_key_does_not_modify_the_file(fmt, tmp_path):
    path = _make_mapping_file(tmp_path, fmt)
    before = path.read_bytes()
    _run(_op(path, key=OTHER_KEY, mapping_format=fmt), ["new"], tmp_path / "b")
    assert path.read_bytes() == before


def _flip_byte(path, offset):
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 0x01
    path.write_bytes(bytes(raw))


def test_modified_ciphertext_is_detected(tmp_path):
    path = _make_mapping_file(tmp_path)
    offset, length = _frames(path)[1]
    _flip_byte(path, offset + 4 + length // 2)

    with pytest.raises(MappingIntegrityError):
        _stored(path, "csv")
    result, _ = _run(_op(path), ["user1"], tmp_path / "b")
    assert result.status == OperationStatus.ERROR


def test_truncated_file_is_detected(tmp_path):
    path = _make_mapping_file(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[:-10])  # cut inside the last frame

    with pytest.raises(MappingIntegrityError) as exc:
        _stored(path, "csv")
    assert "truncated" in str(exc.value)

    path.write_bytes(raw[: len(MAGIC) + 2])  # cut inside a frame header
    with pytest.raises(MappingIntegrityError):
        _stored(path, "csv")


def test_dropped_middle_frame_is_detected(tmp_path):
    path = _make_mapping_file(tmp_path)
    raw = path.read_bytes()
    (o1, n1) = _frames(path)[1]
    path.write_bytes(raw[:o1] + raw[o1 + 4 + n1 :])

    with pytest.raises(MappingIntegrityError):
        _stored(path, "csv")


def test_reordered_frames_are_detected(tmp_path):
    path = _make_mapping_file(tmp_path)
    raw = path.read_bytes()
    (o1, n1), (o2, n2) = _frames(path)[1:3]
    f1, f2 = raw[o1 : o1 + 4 + n1], raw[o2 : o2 + 4 + n2]
    path.write_bytes(raw[:o1] + f2 + f1 + raw[o2 + 4 + n2 :])

    with pytest.raises(MappingIntegrityError):
        _stored(path, "csv")


def test_garbage_file_is_rejected(tmp_path):
    path = tmp_path / "junk.enc"
    path.write_bytes(b"this is not an encrypted mapping at all, just some bytes....")
    with pytest.raises(MappingIntegrityError):
        MappingStorage(path, bytes.fromhex(KEY), create_if_missing=False).load()


def _craft(path, frames, fmt="csv"):
    """Write a version 2 file from explicit per-frame mappings (bypassing the op)."""
    storage = MappingStorage(path.with_name("scratch.enc"), bytes.fromhex(KEY), format=fmt)
    data = MAGIC + b"".join(
        storage._encode_frame(i, m, None) for i, m in enumerate(frames)
    )
    path.write_bytes(data)
    return MappingStorage(path, bytes.fromhex(KEY), format=fmt, create_if_missing=False)


def test_source_value_with_two_pseudonyms_is_rejected(tmp_path):
    storage = _craft(tmp_path / "c.enc", [{"a": "p1"}, {"a": "p2"}])
    with pytest.raises(MappingIntegrityError) as exc:
        storage.load()
    assert "two different pseudonyms" in str(exc.value)


def test_pseudonym_shared_by_two_sources_is_rejected(tmp_path):
    storage = _craft(tmp_path / "c.enc", [{"a": "p1"}, {"b": "p1"}])
    with pytest.raises(MappingIntegrityError) as exc:
        storage.load()
    assert "ambiguous" in str(exc.value)


def test_repeating_the_same_pair_is_harmless(tmp_path):
    storage = _craft(tmp_path / "c.enc", [{"a": "p1"}, {"a": "p1", "b": "p2"}])
    assert storage.load() == {"a": "p1", "b": "p2"}


def test_save_rejects_wrapped_mapping(tmp_path):
    """The old op passed {"mappings": {...}, "_metadata": {...}}: refuse that shape."""
    storage = MappingStorage(tmp_path / "w.enc", bytes.fromhex(KEY))
    with pytest.raises(MappingStorageError):
        storage.save({"mappings": {"a": "p"}, "_metadata": {"x": 1}})


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_awkward_values_survive_the_round_trip(fmt, tmp_path):
    awkward = {
        "comma,inside": "p1",
        'quote"inside': "p2",
        "new\nline": "p3",
        "tab\tchar": "p4",
        "юникод ✓": "p5",
        "": "p6",
        "=cmd()": "p7",
    }
    storage = MappingStorage(tmp_path / "a.enc", bytes.fromhex(KEY), format=fmt)
    storage.save(awkward)
    assert MappingStorage(tmp_path / "a.enc", bytes.fromhex(KEY), format=fmt).load() == awkward


# --------------------------------------------------------------------------
# Previous releases' files
# --------------------------------------------------------------------------


def _v1_file(path, plaintext):
    path.write_bytes(MappingEncryption(bytes.fromhex(KEY)).encrypt(plaintext))


def test_version1_json_file_written_by_the_old_op_is_still_readable(tmp_path):
    envelope = {
        "mappings": {"ann": "P-1", "bob": "P-2"},
        "_metadata": {"last_sequential": 2, "pseudonym_type": "sequential"},
    }
    old_json = {"_metadata": {"version": "1.0"}, "mappings": envelope}
    path = tmp_path / "old.enc"
    _v1_file(path, json.dumps(old_json).encode())

    storage = MappingStorage(path, bytes.fromhex(KEY), format="json")
    mapping, meta = storage.load_with_metadata()
    assert mapping == {"ann": "P-1", "bob": "P-2"}
    assert meta["last_sequential"] == 2


def test_version1_csv_file_with_the_old_envelope_rows_is_recovered(tmp_path):
    """The old CSV writer stored the envelope as two rows holding dict reprs."""
    envelope = {
        "mappings": {"ann": "P-1", "bob": "P-2"},
        "_metadata": {"last_sequential": 2},
    }
    csv_text = "original,pseudonym\r\n"
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    for k, v in sorted(envelope.items()):
        w.writerow([k, str(v)])
    _v1_file(tmp_path / "old.enc", (csv_text + buf.getvalue()).encode())

    storage = MappingStorage(tmp_path / "old.enc", bytes.fromhex(KEY), format="csv")
    assert storage.load() == {"ann": "P-1", "bob": "P-2"}


def test_version1_plain_csv_file_is_readable_and_upgraded_on_append(tmp_path):
    path = tmp_path / "old.enc"
    _v1_file(path, b"original,pseudonym\r\nann,P-1\r\nbob,P-2\r\n")
    storage = MappingStorage(path, bytes.fromhex(KEY), format="csv")
    assert storage.load() == {"ann": "P-1", "bob": "P-2"}

    storage.append({"cy": "P-3"})

    assert path.read_bytes().startswith(MAGIC)
    assert MappingStorage(path, bytes.fromhex(KEY), format="csv").load() == {
        "ann": "P-1",
        "bob": "P-2",
        "cy": "P-3",
    }
    assert list(tmp_path.glob("old.bak.*")), "a backup is taken before the upgrade"


# --------------------------------------------------------------------------
# MAP-03: persistence is incremental
# --------------------------------------------------------------------------


def test_append_does_not_rewrite_existing_bytes(tmp_path):
    path = tmp_path / "inc.enc"
    storage = MappingStorage(path, bytes.fromhex(KEY))
    storage.append({f"k{i}": f"p{i}" for i in range(50)})
    first = path.read_bytes()

    storage.append({"extra": "px"})
    second = path.read_bytes()

    assert second.startswith(first)  # strictly appended
    assert len(second) - len(first) < 200  # proportional to the one new entry


def test_append_cost_does_not_grow_with_table_size(tmp_path):
    path = tmp_path / "inc.enc"
    storage = MappingStorage(path, bytes.fromhex(KEY))
    storage.append({"seed": "p"})
    deltas = []
    for batch in range(5):
        before = path.stat().st_size
        storage.append({f"b{batch}-{i}": f"p{batch}-{i}" for i in range(100)})
        deltas.append(path.stat().st_size - before)
    # Same-size batches add (about) the same number of bytes however large the
    # table already is; a full rewrite would add the whole table each time.
    assert max(deltas) - min(deltas) < 64


def test_backup_is_taken_once_per_instance_not_per_persist(tmp_path, monkeypatch):
    path = tmp_path / "bk.enc"
    storage = MappingStorage(path, bytes.fromhex(KEY))
    calls = []
    original = storage._create_backup
    monkeypatch.setattr(storage, "_create_backup", lambda: calls.append(1) or original())

    for i in range(10):
        storage.append({f"k{i}": f"p{i}"})

    assert len(calls) == 1


def test_operation_persists_in_frames_of_persist_frequency(tmp_path):
    path = tmp_path / "freq.map.enc"
    op = _op(path, persist_frequency=3)
    result, _ = _run(op, [f"v{i}" for i in range(10)], tmp_path / "a")
    assert result.status == OperationStatus.SUCCESS, result.error_message
    # empty snapshot + ceil(10/3) appends (the final one flushes the remainder)
    assert len(_frames(path)) >= 4
    assert len(_stored(path, "csv")) == 10


def test_failed_persist_keeps_pending_entries_for_retry(tmp_path, monkeypatch):
    path = tmp_path / "retry.map.enc"
    op = _op(path, persist_frequency=1000)
    _run(op, ["a"], tmp_path / "a")

    op2 = _op(path, persist_frequency=1000)
    (tmp_path / "b").mkdir()
    op2._initialize_mapping(tmp_path / "b")
    op2._mapping["n"] = "pn"
    op2._pending_mappings["n"] = "pn"
    op2._new_mappings_count = 1

    real_append = op2._mapping_storage.append
    state = {"fail": True}

    def flaky(entries, metadata=None):
        if state["fail"]:
            state["fail"] = False
            raise MappingStorageError("disk full")
        return real_append(entries, metadata)

    monkeypatch.setattr(op2._mapping_storage, "append", flaky)
    with pytest.raises(MappingStorageError):
        op2._persist_mappings()
    assert op2._pending_mappings == {"n": "pn"}

    op2._persist_mappings()
    assert op2._pending_mappings == {}
    assert _stored(path, "csv")["n"] == "pn"


def test_deleted_mapping_file_is_rewritten_in_full_not_partially(tmp_path):
    path = tmp_path / "gone.map.enc"
    _run(_op(path), ["a", "b"], tmp_path / "a")

    op = _op(path, persist_frequency=1000)
    (tmp_path / "b").mkdir()
    op._initialize_mapping(tmp_path / "b")
    path.unlink()
    op._mapping["c"] = "pc"
    op._pending_mappings["c"] = "pc"
    op._new_mappings_count = 1
    op._persist_mappings()

    assert set(_stored(path, "csv")) == {"a", "b", "c"}


# --------------------------------------------------------------------------
# Storage edges
# --------------------------------------------------------------------------


def test_append_from_a_fresh_instance_continues_the_frame_sequence(tmp_path):
    path = tmp_path / "f.enc"
    MappingStorage(path, bytes.fromhex(KEY)).append({"a": "p1"})
    fresh = MappingStorage(path, bytes.fromhex(KEY))  # has not loaded anything
    fresh.append({"b": "p2"})
    assert MappingStorage(path, bytes.fromhex(KEY)).load() == {"a": "p1", "b": "p2"}
    assert len(_frames(path)) == 3  # snapshot of {} + two appends


def test_append_to_a_truncated_file_fails_closed(tmp_path):
    path = tmp_path / "f.enc"
    MappingStorage(path, bytes.fromhex(KEY)).append({"a": "p1"})
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(MappingIntegrityError):
        MappingStorage(path, bytes.fromhex(KEY)).append({"b": "p2"})


def test_missing_and_empty_files_load_as_empty(tmp_path):
    storage = MappingStorage(tmp_path / "x.enc", bytes.fromhex(KEY), create_if_missing=False)
    assert storage.load() == {}
    (tmp_path / "x.enc").write_bytes(b"")
    assert storage.load_with_metadata() == ({}, {})


def test_update_merges_and_keeps_metadata(tmp_path):
    storage = MappingStorage(tmp_path / "u.enc", bytes.fromhex(KEY))
    storage.save({"a": "p1"}, {"last_sequential": 7})
    merged = storage.update({"a": "p9", "b": "p2"})
    assert merged == {"a": "p9", "b": "p2"}
    assert storage.load_with_metadata() == (merged, {"last_sequential": 7})


def test_invalid_format_is_rejected(tmp_path):
    from pamola_core.errors.exceptions import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        MappingStorage(tmp_path / "x.enc", bytes.fromhex(KEY), format="xml")


def test_metadata_helpers(tmp_path):
    storage = MappingStorage(tmp_path / "m.enc", bytes.fromhex(KEY))
    storage.save({"a": "p1"})
    info = storage.get_metadata()
    assert info["exists"] and info["format"] == "csv" and info["size_bytes"] > 0
    report = storage.validate_mappings({"a": "p1", "b": "p1", "": "p3"})
    assert report["valid"] is False
    assert report["duplicate_values"] == ["p1"]
    assert report["empty_keys"] == 1
