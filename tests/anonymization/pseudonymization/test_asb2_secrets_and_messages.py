"""ASB-2: no row content in mapping-storage errors; no key/salt/pepper/plaintext mapping in the
pickled state of the pseudonymization operations; low-entropy warning for the hash operation."""

import copy
import logging
import pickle

import pandas as pd
import pytest

from pamola_core.anonymization.commons.mapping_storage import MappingIntegrityError, MappingStorage
from pamola_core.anonymization.pseudonymization.hash_based_op import HashBasedPseudonymizationOperation
from pamola_core.anonymization.pseudonymization.mapping_op import ConsistentMappingPseudonymizationOperation
from pamola_core.utils.crypto_helpers.pseudonymization import MappingEncryption
from pamola_core.utils.ops.op_data_source import DataSource

KEY_HEX = "ab" * 32
SALT_HEX = "c3" * 32
CANARY = "CANARY-ORIGINAL-9f2e"


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _v1_file(tmp_path, payload: bytes):
    path = tmp_path / "legacy.enc"
    path.write_bytes(MappingEncryption(bytes.fromhex(KEY_HEX)).encrypt(payload))
    return path


def test_asb2_corrupt_csv_row_message_has_no_row_content(tmp_path):
    path = _v1_file(tmp_path, f"original,pseudonym\n{CANARY},PSEUDO-SECRET,extra\n".encode())
    with pytest.raises(MappingIntegrityError) as exc:
        MappingStorage(path, bytes.fromhex(KEY_HEX), format="csv", create_if_missing=False).load()
    text = str(exc.value)
    assert "CANARY" not in text and "PSEUDO" not in text
    assert "row 1" in text and "3 column" in text


def test_asb2_wrong_header_message_has_no_content(tmp_path):
    path = _v1_file(tmp_path, f"{CANARY},PSEUDO-SECRET\nx,y\n".encode())
    with pytest.raises(MappingIntegrityError) as exc:
        MappingStorage(path, bytes.fromhex(KEY_HEX), format="csv", create_if_missing=False).load()
    assert "CANARY" not in str(exc.value) and "PSEUDO" not in str(exc.value)


def _run_mapping(tmp_path):
    op = ConsistentMappingPseudonymizationOperation(
        field_name="email", mapping_encryption_key=KEY_HEX, generate_visualization=False)
    df = pd.DataFrame({"email": [f"{CANARY}-{i}" for i in range(5)]})
    out = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert str(out.status).endswith("SUCCESS"), out.error_message
    assert op._mapping, "the run must have produced a plaintext mapping in memory"
    return op


def test_asb2_pickled_mapping_operation_holds_no_key_and_no_plaintext_mapping(tmp_path):
    op = _run_mapping(tmp_path)
    for blob in (pickle.dumps(op), pickle.dumps(copy.deepcopy(op))):
        assert bytes.fromhex(KEY_HEX) not in blob and KEY_HEX.encode() not in blob
        assert CANARY.encode() not in blob
        for pseudonym in op._reverse_mapping:
            assert pseudonym.encode() not in blob
    restored = pickle.loads(pickle.dumps(op))
    assert restored.field_name == "email" and restored._mapping == {} and restored._mapping_encryption_key is None
    assert op._mapping and op._mapping_encryption_key is not None  # the original is untouched


def _hash_op(**kw):
    return HashBasedPseudonymizationOperation(
        field_name="email", salt_config={"source": "parameter", "value": SALT_HEX}, generate_visualization=False, **kw)


def test_asb2_pickled_hash_operation_holds_no_salt_and_no_pepper():
    op = _hash_op()
    op._initialize_crypto_components()
    pepper = op._pepper.get()
    assert op._salt and pepper
    blob = pickle.dumps(op)
    assert bytes.fromhex(SALT_HEX) not in blob and SALT_HEX.encode() not in blob
    assert pepper not in blob
    restored = pickle.loads(blob)
    assert restored._salt is None and restored._pepper is None
    assert "value" not in restored.salt_config


def test_asb2_secrets_reach_the_workers_only_during_the_operations_own_parallel_run():
    op = _hash_op()
    op._initialize_crypto_components()
    pepper = op._pepper.get()
    op._secret_transfer = True  # what _process_data_with_config sets around its parallel run
    blob = pickle.dumps(op)
    assert pepper in blob  # every worker must hash identically
    op._secret_transfer = False
    assert pepper not in pickle.dumps(op)


def test_asb2_hash_operation_still_works_with_chunked_parallel_processing(tmp_path):
    df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(40)]})
    ds = lambda: DataSource(dataframes={"main": df.copy()})  # noqa: E731
    base = _hash_op(use_pepper=False)
    seq = base.execute(ds(), tmp_path / "a", _Reporter())
    par = _hash_op(use_pepper=False, use_vectorization=True, parallel_processes=2, chunk_size=10)
    out = par.execute(ds(), tmp_path / "b", _Reporter())
    assert str(seq.status).endswith("SUCCESS") and str(out.status).endswith("SUCCESS")


def test_asb2_low_entropy_field_is_warned_about_once_per_run(tmp_path, caplog):
    df = pd.DataFrame({"status": ["A", "B", "C"] * 20})
    op = HashBasedPseudonymizationOperation(
        field_name="status", salt_config={"source": "parameter", "value": SALT_HEX}, generate_visualization=False)
    with caplog.at_level(logging.WARNING):
        out = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    msgs = [r.getMessage() for r in caplog.records if "low-entropy" in r.getMessage()]
    assert len(msgs) == 1 and "reversible by enumeration" in msgs[0]
    assert "3 distinct" in msgs[0]  # a count, never the values
    metrics = [p for p in (tmp_path / "t").rglob("*_metrics_*.json")]
    assert metrics and '"low_entropy_source": true' in metrics[0].read_text(encoding="utf-8")


def test_asb2_high_cardinality_field_gets_no_warning(tmp_path, caplog):
    df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(1500)]})
    op = _hash_op()
    with caplog.at_level(logging.WARNING):
        op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert not [r for r in caplog.records if "low-entropy" in r.getMessage()]
