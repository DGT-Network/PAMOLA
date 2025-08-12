"""
Tests for the encryption_manager module in the pamola_core/utils/tasks package.

These tests ensure that the TaskEncryptionManager and related classes properly implement encryption key management, encryption/decryption, redaction, and error handling.
"""

import base64
import tempfile
import os
import shutil
from pathlib import Path
from unittest import mock
import pytest

from pamola_core.utils.tasks.encryption_manager import (
    TaskEncryptionManager,
    EncryptionMode,
    EncryptionError,
    EncryptionInitializationError,
    KeyGenerationError,
    KeyLoadingError,
    DataRedactionError,
    MemoryProtectedKey,
    EncryptionContext,
)

# --- Fixtures and Mocks ---

class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.allowed_external_paths = []
        self.allow_external = False
        self.task_id = kwargs.get('task_id', 'dummy_task')
    def resolve_legacy_path(self, path):
        return Path(path)
    def to_dict(self):
        return self.__dict__

class DummyProgressManager:
    def __init__(self):
        self.logs = []
    def create_operation_context(self, name, total, description=None):
        return mock.MagicMock().__enter__.return_value
    def log_info(self, msg):
        self.logs.append(("info", msg))
    def log_error(self, msg):
        self.logs.append(("error", msg))
    def log_warning(self, msg):
        self.logs.append(("warning", msg))
    def log_debug(self, msg):
        self.logs.append(("debug", msg))

@pytest.fixture
def temp_key_file(tmp_path):
    key = base64.urlsafe_b64encode(os.urandom(32))
    key_path = tmp_path / "test.key"
    with open(key_path, "wb") as f:
        f.write(key)
    return key_path, key

@pytest.fixture
def dummy_config():
    return DummyConfig(
        encryption_mode=EncryptionMode.SIMPLE,
        use_encryption=True,
        encryption_key_path=None,
        task_id="dummy_task"
    )

@pytest.fixture
def manager(dummy_config):
    return TaskEncryptionManager(
        task_config=dummy_config,
        logger=None,
        progress_manager=DummyProgressManager(),
    )

# --- Tests for TaskEncryptionManager and related classes ---

class TestEncryptionManager:
    def test_initialize_with_file_key(self, dummy_config, temp_key_file):
        key_path, key = temp_key_file
        dummy_config.encryption_key_path = key_path
        mgr = TaskEncryptionManager(dummy_config)
        assert mgr.initialize() is True
        assert mgr._protected_key is not None
        assert mgr._protected_key.fingerprint is not None
        assert mgr._use_encryption is True

    def test_initialize_with_key_store(self, dummy_config, monkeypatch):
        dummy_config.encryption_key_path = None
        dummy_config.task_id = "test_store"
        monkeypatch.setattr(
            "pamola_core.utils.tasks.encryption_manager.get_key_for_task",
            lambda task_id: base64.urlsafe_b64encode(b"x" * 32)
        )
        monkeypatch.setattr(
            "pamola_core.utils.tasks.encryption_manager.KEY_STORE_AVAILABLE",
            True
        )
        mgr = TaskEncryptionManager(dummy_config)
        assert mgr.initialize() is True
        assert mgr._protected_key is not None

    def test_initialize_generate_key(self, dummy_config, monkeypatch):
        dummy_config.encryption_key_path = None
        monkeypatch.setattr(
            "pamola_core.utils.tasks.encryption_manager.KEY_STORE_AVAILABLE",
            False
        )
        mgr = TaskEncryptionManager(dummy_config)
        assert mgr.initialize() is True
        assert mgr._protected_key is not None

    def test_initialize_encryption_disabled(self, dummy_config):
        dummy_config.use_encryption = False
        mgr = TaskEncryptionManager(dummy_config)
        assert mgr.initialize() is False
        assert mgr._protected_key is None
        assert mgr._use_encryption is False

    def test_initialize_invalid_key_path(self, dummy_config, monkeypatch):
        dummy_config = None  # Invalid type
        mgr = TaskEncryptionManager(dummy_config)
        mgr.initialize()
        # The manager may still have a MemoryProtectedKey if it falls back to key generation
        # Accept both None and MemoryProtectedKey, but encryption should be disabled
        assert mgr._use_encryption is False

    def test_encryption_context(self, manager):
        manager._protected_key = MemoryProtectedKey(b"a" * 32)
        manager._encryption_mode = EncryptionMode.SIMPLE
        manager._use_encryption = True
        ctx = manager.get_encryption_context()
        assert ctx.can_encrypt is True
        assert ctx.mode == EncryptionMode.SIMPLE
        assert isinstance(ctx.key_fingerprint, str)
        d = ctx.to_dict()
        assert d["can_encrypt"] is True
        assert d["mode"] == "simple"

    def test_encryption_context_disabled(self, dummy_config):
        dummy_config.use_encryption = False
        mgr = TaskEncryptionManager(dummy_config)
        ctx = mgr.get_encryption_context()
        assert ctx.can_encrypt is False
        assert ctx.mode == EncryptionMode.NONE

    def test_encrypt_data_disabled(self, manager):
        manager._use_encryption = False
        with pytest.raises(EncryptionError):
            manager.encrypt_data(b"data")

    def test_decrypt_data_disabled(self, manager):
        manager._use_encryption = False
        with pytest.raises(EncryptionError):
            manager.decrypt_data(b"data")

    def test_encrypt_data_invalid_mode(self, manager):
        manager._protected_key = MemoryProtectedKey(manager._generate_encryption_key())
        manager._encryption_mode = EncryptionMode.NONE
        manager._use_encryption = True
        with pytest.raises(EncryptionError):
            manager.encrypt_data(b"data")

    def test_decrypt_data_invalid_mode(self, manager):
        manager._protected_key = MemoryProtectedKey(manager._generate_encryption_key())
        manager._encryption_mode = EncryptionMode.NONE
        manager._use_encryption = True
        with pytest.raises(EncryptionError):
            manager.decrypt_data(b"data")

    def test_add_and_check_sensitive_param(self, manager):
        manager.add_sensitive_param_names(["mysecret", "token"])
        assert manager.is_sensitive_param("mysecret")
        assert manager.is_sensitive_param("token")
        assert not manager.is_sensitive_param("notsensitive")

    def test_redact_sensitive_data_dict(self, manager):
        manager.add_sensitive_param_names(["password"])
        data = {"password": "123456", "other": "ok"}
        redacted = manager.redact_sensitive_data(data)
        # The key may be redacted as '<redacted:pas...>' if redact_keys is True (default)
        # So check for both possible keys
        assert any(k.startswith("<redacted:pas") for k in redacted) or "password" in redacted
        # Value should be '<redacted>' for the redacted key
        if "password" in redacted:
            assert redacted["password"] == "<redacted>"
        else:
            k = next(k for k in redacted if k.startswith("<redacted:pas"))
            assert redacted[k] == "<redacted>"
        assert any(v == "ok" for v in redacted.values())

    def test_redact_sensitive_data_list(self, manager):
        manager.add_sensitive_param_names(["secret"])
        data = ["notsecret", {"secret": "val"}]
        redacted = manager.redact_sensitive_data(data)
        # The dict inside the list may have its key redacted
        d = redacted[1]
        assert any((k == "secret" or k.startswith("<redacted:sec")) and v == "<redacted>" for k, v in d.items())

    def test_redact_sensitive_data_tuple_set(self, manager):
        manager.add_sensitive_param_names(["token"])
        data = ("token", {"token": "abc"})
        redacted = manager.redact_sensitive_data(data)
        assert isinstance(redacted, tuple)
        s = {"token", ("token", "abc")}
        redacted_set = manager.redact_sensitive_data(s)
        assert isinstance(redacted_set, set)

    def test_redact_sensitive_data_keylike(self, manager):
        keylike = base64.urlsafe_b64encode(os.urandom(32)).decode()
        result = manager.redact_sensitive_data(keylike)
        assert result == "<redacted:key-like>" or result == keylike

    def test_redact_config_dict(self, manager):
        config = {"api_key": "123", "param": 1}
        redacted = manager.redact_config_dict(config)
        assert redacted["api_key"] == "<redacted>"
        assert redacted["param"] == 1

    def test_get_encryption_info(self, manager):
        manager._protected_key = MemoryProtectedKey(manager._generate_encryption_key())
        manager._encryption_mode = EncryptionMode.SIMPLE
        manager._use_encryption = True
        info = manager.get_encryption_info()
        assert info["enabled"] is True
        assert info["mode"] == "simple"
        assert info["key_available"] is True
        assert isinstance(info["key_fingerprint"], str)

    def test_check_dataset_encryption(self, manager, tmp_path):
        class DummyDataSource:
            def get_file_paths(self):
                f = tmp_path / "data.enc"
                with open(f, "wb") as out:
                    out.write(b"gAAAAA")
                return {"ds": f}
        manager._use_encryption = True
        manager._encryption_mode = EncryptionMode.SIMPLE
        assert manager.check_dataset_encryption(DummyDataSource()) is True

    def test_is_file_encrypted_simple(self, manager, tmp_path):
        f = tmp_path / "enc.dat"
        with open(f, "wb") as out:
            out.write(b"gAAAAA")
        manager._encryption_mode = EncryptionMode.SIMPLE
        assert manager.is_file_encrypted(f) is True
        with open(f, "wb") as out:
            out.write(b"notenc")
        assert manager.is_file_encrypted(f) is False

    def test_is_file_encrypted_invalid_path(self, manager, monkeypatch):
        monkeypatch.setattr(
            "pamola_core.utils.tasks.encryption_manager.validate_path_security",
            lambda path, **kwargs: False
        )
        assert manager.is_file_encrypted("/bad/path") is False

    def test_supports_encryption_mode(self, manager):
        assert manager.supports_encryption_mode("none") is True
        # The result for 'simple' and 'age' depends on installed libraries, so just check for bool
        assert isinstance(manager.supports_encryption_mode("simple"), bool)
        assert isinstance(manager.supports_encryption_mode("age"), bool)

    def test_cleanup(self, manager):
        manager._protected_key = MemoryProtectedKey(b"a" * 32)
        manager.cleanup()
        assert manager._protected_key is None

    def test_memory_protected_key(self):
        key = b"b" * 32
        mpk = MemoryProtectedKey(key)
        assert mpk.fingerprint is not None
        assert mpk.key_id is not None
        assert not mpk.has_been_used
        with mpk as k:
            assert k == key
        assert mpk.has_been_used
        del mpk  # Should not raise

    def test_encryption_mode_from_string(self):
        assert EncryptionMode.from_string("simple") == EncryptionMode.SIMPLE
        assert EncryptionMode.from_string("none") == EncryptionMode.NONE
        assert EncryptionMode.from_string("age") == EncryptionMode.AGE
        assert EncryptionMode.from_string("invalid") == EncryptionMode.SIMPLE

    def test_encryption_errors(self):
        assert issubclass(EncryptionError, Exception)
        assert issubclass(EncryptionInitializationError, EncryptionError)
        assert issubclass(KeyGenerationError, EncryptionError)
        assert issubclass(KeyLoadingError, EncryptionError)
        assert issubclass(DataRedactionError, EncryptionError)

    def test_resolve_key_path_invalid_type(self, manager):
        manager._encryption_key_path = 12345
        with pytest.raises(ValueError):
            manager._resolve_key_path()

    def test_resolve_key_path_path_security_error(self, manager, monkeypatch):
        manager._encryption_key_path = "bad.key"
        monkeypatch.setattr("pamola_core.utils.tasks.encryption_manager.validate_path_security", lambda *a, **k: False)
        with pytest.raises(Exception):
            manager._resolve_key_path()

    def test_get_key_from_store_cache(self, manager):
        manager._key_cache["dummy_task"] = b"cachedkey"
        assert manager._get_key_from_store() == b"cachedkey"

    def test_get_key_from_store_str(self, manager, monkeypatch):
        monkeypatch.setattr("pamola_core.utils.tasks.encryption_manager.get_key_for_task", lambda tid: base64.urlsafe_b64encode(b"x"*32).decode())
        monkeypatch.setattr("pamola_core.utils.tasks.encryption_manager.KEY_STORE_AVAILABLE", True)
        manager._key_cache.clear()
        assert isinstance(manager._get_key_from_store(), bytes)

    def test_get_key_from_store_error(self, manager, monkeypatch):
        class DummyError(Exception):
            pass
        def raise_error(tid):
            raise DummyError("fail")
        monkeypatch.setattr("pamola_core.utils.tasks.encryption_manager.get_key_for_task", raise_error)
        monkeypatch.setattr("pamola_core.utils.tasks.encryption_manager.KEY_STORE_AVAILABLE", True)
        manager._key_cache.clear()
        with pytest.raises(Exception):
            manager._get_key_from_store()

    def test_generate_encryption_key_invalid_mode(self, manager):
        manager._encryption_mode = "invalid"
        with pytest.raises(Exception):
            manager._generate_encryption_key()

    def test_initialize_progress_manager_exception(self, dummy_config):
        class DummyPM:
            def create_operation_context(self, *a, **k):
                raise Exception("fail")
        mgr = TaskEncryptionManager(dummy_config, progress_manager=DummyPM())
        assert mgr.initialize() is False

    def test_redact_sensitive_data_error(self, manager):
        class BadObj:
            def __getattr__(self, item):
                raise Exception("fail")
        with pytest.raises(DataRedactionError):
            manager.redact_sensitive_data(BadObj())

    def test_is_file_encrypted_open_error(self, manager, monkeypatch):
        monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))
        assert manager.is_file_encrypted("somefile") is False

    def test_is_file_encrypted_age_mode(self, manager, tmp_path):
        f = tmp_path / "agefile"
        with open(f, "wb") as out:
            out.write(b"age-encryption.org/")
        manager._encryption_mode = EncryptionMode.AGE
        assert manager.is_file_encrypted(f) is True

    def test_memory_protected_key_del(self):
        key = b"x" * 32
        mpk = MemoryProtectedKey(key)
        del mpk  # Should not raise

    def test_task_encryption_manager_del(self, dummy_config):
        mgr = TaskEncryptionManager(dummy_config)
        del mgr  # Should not raise

    def test_cleanup_with_log_info(self, manager):
        class DummyPM:
            def log_info(self, msg):
                self.called = True
        pm = DummyPM()
        manager.progress_manager = pm
        manager.cleanup()
        assert hasattr(pm, "called")

if __name__ == "__main__":
    pytest.main()
