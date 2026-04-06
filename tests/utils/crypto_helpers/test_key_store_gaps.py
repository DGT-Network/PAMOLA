"""
Gap tests for key_store module.
Covers: EncryptedKeyStore init, _get_master_key, _generate_master_key,
_initialize_keys_db, _load_keys_db, _save_keys_db, store_task_key,
load_task_key, delete_task_key, list_task_keys, generate_task_key,
is_master_key_exposed, get_key_for_task function.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.utils.crypto_helpers.key_store import (
    EncryptedKeyStore,
    get_key_for_task,
)
from pamola_core.errors.exceptions import TaskKeyError, KeyStoreError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path):
    """Create an EncryptedKeyStore using tmp_path for isolation."""
    keys_db = tmp_path / "keys.db"
    master_key = tmp_path / "master.key"
    return EncryptedKeyStore(keys_db_path=keys_db, master_key_path=master_key)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_key_store_init_creates_db(tmp_path):
    store = _make_store(tmp_path)
    assert (tmp_path / "keys.db").exists()


def test_key_store_init_creates_master_key(tmp_path):
    store = _make_store(tmp_path)
    assert (tmp_path / "master.key").exists()


def test_key_store_init_existing_db(tmp_path):
    # First init creates db; second init should reuse it
    store1 = _make_store(tmp_path)
    store2 = _make_store(tmp_path)
    assert store2 is not None


# ---------------------------------------------------------------------------
# _get_master_key
# ---------------------------------------------------------------------------

def test_get_master_key_generates_when_missing(tmp_path):
    keys_db = tmp_path / "keys.db"
    master_key_path = tmp_path / "master.key"
    store = EncryptedKeyStore(keys_db_path=keys_db, master_key_path=master_key_path)
    # delete and re-request
    master_key_path.unlink()
    key = store._get_master_key()
    assert isinstance(key, str)
    assert len(key) > 0


def test_get_master_key_reads_existing(tmp_path):
    store = _make_store(tmp_path)
    key1 = store._get_master_key()
    key2 = store._get_master_key()
    assert key1 == key2


def test_get_master_key_invalid_base64_triggers_regeneration(tmp_path):
    store = _make_store(tmp_path)
    (tmp_path / "master.key").write_text("not-valid-base64!!!")
    # Should regenerate without raising
    key = store._get_master_key()
    assert isinstance(key, str)


def test_get_master_key_wrong_length_triggers_regeneration(tmp_path):
    import base64
    store = _make_store(tmp_path)
    short_key = base64.b64encode(b"short").decode()
    (tmp_path / "master.key").write_text(short_key)
    key = store._get_master_key()
    assert isinstance(key, str)


# ---------------------------------------------------------------------------
# _generate_master_key
# ---------------------------------------------------------------------------

def test_generate_master_key_returns_base64_string(tmp_path):
    store = _make_store(tmp_path)
    key = store._generate_master_key()
    import base64
    decoded = base64.b64decode(key)
    assert len(decoded) == 32  # MASTER_KEY_LENGTH


def test_generate_master_key_saves_to_file(tmp_path):
    store = _make_store(tmp_path)
    key = store._generate_master_key()
    assert (tmp_path / "master.key").exists()
    assert (tmp_path / "master.key").read_text().strip() == key


# ---------------------------------------------------------------------------
# store_task_key / load_task_key
# ---------------------------------------------------------------------------

def test_store_and_load_task_key(tmp_path):
    store = _make_store(tmp_path)
    store.store_task_key("task-001", "my-secret-key")
    retrieved = store.load_task_key("task-001")
    assert retrieved == "my-secret-key"


def test_store_task_key_with_metadata(tmp_path):
    store = _make_store(tmp_path)
    store.store_task_key("task-002", "key-value", metadata={"owner": "test"})
    retrieved = store.load_task_key("task-002")
    assert retrieved == "key-value"


def test_load_task_key_not_found_raises(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(TaskKeyError):
        store.load_task_key("nonexistent-task")


def test_store_multiple_task_keys(tmp_path):
    store = _make_store(tmp_path)
    store.store_task_key("task-a", "key-a")
    store.store_task_key("task-b", "key-b")
    assert store.load_task_key("task-a") == "key-a"
    assert store.load_task_key("task-b") == "key-b"


# ---------------------------------------------------------------------------
# delete_task_key
# ---------------------------------------------------------------------------

def test_delete_task_key_success(tmp_path):
    store = _make_store(tmp_path)
    store.store_task_key("task-del", "key-to-delete")
    store.delete_task_key("task-del")
    with pytest.raises(TaskKeyError):
        store.load_task_key("task-del")


def test_delete_task_key_not_found_raises(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(TaskKeyError):
        store.delete_task_key("nonexistent-task")


# ---------------------------------------------------------------------------
# list_task_keys
# ---------------------------------------------------------------------------

def test_list_task_keys_empty(tmp_path):
    store = _make_store(tmp_path)
    keys = store.list_task_keys()
    assert isinstance(keys, list)
    assert len(keys) == 0


def test_list_task_keys_returns_metadata(tmp_path):
    store = _make_store(tmp_path)
    store.store_task_key("task-list-1", "k1", metadata={"env": "test"})
    store.store_task_key("task-list-2", "k2")
    keys = store.list_task_keys()
    assert len(keys) == 2
    task_ids = [k["task_id"] for k in keys]
    assert "task-list-1" in task_ids
    assert "task-list-2" in task_ids


def test_list_task_keys_no_actual_key_exposed(tmp_path):
    store = _make_store(tmp_path)
    store.store_task_key("task-secure", "super-secret")
    keys = store.list_task_keys()
    for entry in keys:
        assert "key" not in entry


# ---------------------------------------------------------------------------
# generate_task_key
# ---------------------------------------------------------------------------

def test_generate_task_key_returns_string(tmp_path):
    store = _make_store(tmp_path)
    key = store.generate_task_key("task-gen-1")
    assert isinstance(key, str)
    assert len(key) > 0


def test_generate_task_key_is_retrievable(tmp_path):
    store = _make_store(tmp_path)
    generated = store.generate_task_key("task-gen-2")
    retrieved = store.load_task_key("task-gen-2")
    assert generated == retrieved


def test_generate_task_key_with_metadata(tmp_path):
    store = _make_store(tmp_path)
    key = store.generate_task_key("task-gen-3", metadata={"purpose": "testing"})
    assert key is not None


def test_generate_task_key_produces_different_keys_each_time(tmp_path):
    store = _make_store(tmp_path)
    key1 = store.generate_task_key("task-a1")
    key2 = store.generate_task_key("task-a2")
    assert key1 != key2


# ---------------------------------------------------------------------------
# is_master_key_exposed
# ---------------------------------------------------------------------------

def test_is_master_key_exposed_no_file(tmp_path):
    store = _make_store(tmp_path)
    master_key_path = tmp_path / "master.key"
    master_key_path.unlink()
    result = store.is_master_key_exposed()
    assert result is False


def test_is_master_key_exposed_returns_bool(tmp_path):
    store = _make_store(tmp_path)
    result = store.is_master_key_exposed()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _load_keys_db
# ---------------------------------------------------------------------------

def test_load_keys_db_returns_dict(tmp_path):
    store = _make_store(tmp_path)
    db = store._load_keys_db()
    assert isinstance(db, dict)
    assert "keys" in db


def test_load_keys_db_contains_version(tmp_path):
    store = _make_store(tmp_path)
    db = store._load_keys_db()
    assert "version" in db


# ---------------------------------------------------------------------------
# get_key_for_task (module-level function)
# ---------------------------------------------------------------------------

def test_get_key_for_task_returns_string(tmp_path):
    with patch(
        "pamola_core.utils.crypto_helpers.key_store._get_env_path",
        side_effect=lambda var, default: str(tmp_path / "configs/") if "CONFIGS" in var
            else str(tmp_path / ("keys.db" if "KEY_STORE" in var else "master.key")),
    ):
        key = get_key_for_task("test-task-module")
        assert isinstance(key, str)
        assert len(key) > 0


def test_get_key_for_task_same_task_same_key(tmp_path):
    with patch(
        "pamola_core.utils.crypto_helpers.key_store._get_env_path",
        side_effect=lambda var, default: str(tmp_path / "configs/") if "CONFIGS" in var
            else str(tmp_path / ("keys.db" if "KEY_STORE" in var else "master.key")),
    ):
        key1 = get_key_for_task("stable-task")
        key2 = get_key_for_task("stable-task")
        assert key1 == key2


def test_get_key_for_task_different_tasks_different_keys(tmp_path):
    with patch(
        "pamola_core.utils.crypto_helpers.key_store._get_env_path",
        side_effect=lambda var, default: str(tmp_path / "configs/") if "CONFIGS" in var
            else str(tmp_path / ("keys.db" if "KEY_STORE" in var else "master.key")),
    ):
        key1 = get_key_for_task("task-x")
        key2 = get_key_for_task("task-y")
        assert key1 != key2
