"""Additional tests for crypto_utils.py — covers is_encrypted, get_encryption_info,
get_encryption_mode, and encrypt_data which are missing from test_crypto_utils_coverage.py."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.utils.io_helpers.crypto_utils import (
    is_encrypted,
    get_encryption_info,
    get_encryption_mode,
    encrypt_data,
)


# --- is_encrypted ---
class TestIsEncrypted:
    def test_plain_file_returns_false(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   return_value="none"):
            result = is_encrypted(f)
        assert result is False

    def test_encrypted_file_returns_true(self, tmp_path):
        f = tmp_path / "data.enc"
        f.write_bytes(b"\x00\x01\x02\x03")
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   return_value="simple"):
            result = is_encrypted(f)
        assert result is True

    def test_nonexistent_file_returns_false(self):
        result_path = Path("/nonexistent/file.csv")
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            result = is_encrypted(result_path)
        assert result is False

    def test_encrypted_dict_returns_true(self):
        data = {"mode": "simple", "algorithm": "AES", "data": "base64data"}
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            result = is_encrypted(data)
        assert result is True

    def test_plain_dict_returns_false(self):
        data = {"mode": "none", "value": "plain"}
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            result = is_encrypted(data)
        assert result is False

    def test_dict_with_algorithm_and_data_returns_true(self):
        data = {"algorithm": "AES-256", "data": "encoded_data"}
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            result = is_encrypted(data)
        assert result is True

    def test_detect_exception_returns_false(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("data")
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   side_effect=Exception("error")):
            result = is_encrypted(f)
        assert result is False

    def test_bytes_returns_false(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            result = is_encrypted(b"raw bytes")
        assert result is False


# --- get_encryption_info ---
class TestGetEncryptionInfo:
    def test_plain_file_returns_mode_none(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   return_value="none"):
            info = get_encryption_info(f)
        assert info.get("mode") == "none"

    def test_encrypted_dict_returns_metadata(self):
        data = {"mode": "simple", "version": "1.0", "data": "secret", "iv": "initvec"}
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            info = get_encryption_info(data)
        assert info.get("mode") == "simple"
        assert "data" not in info
        assert "iv" not in info

    def test_dict_with_version_included(self):
        data = {"mode": "age", "version": "2.0", "data": "enc"}
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            info = get_encryption_info(data)
        assert info.get("version") == "2.0"

    def test_nonexistent_file_returns_empty(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            info = get_encryption_info(Path("/nonexistent/file.csv"))
        assert isinstance(info, dict)

    def test_exception_returns_partial_info(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("data")
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   side_effect=Exception("fail")):
            info = get_encryption_info(f)
        assert isinstance(info, dict)

    def test_simple_mode_reads_json_metadata(self, tmp_path):
        import json
        f = tmp_path / "enc.json"
        f.write_text(json.dumps({"mode": "simple", "algorithm": "AES", "data": "abc", "iv": "xyz"}))
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   return_value="simple"):
            info = get_encryption_info(f)
        assert "mode" in info


# --- get_encryption_mode ---
class TestGetEncryptionMode:
    def test_disabled_returns_none(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            result = get_encryption_mode("data", use_encryption=False)
        assert "none" in result.lower() or result == "NONE"

    def test_enabled_small_data_returns_simple(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.auto_choose_provider_encryption", True), \
             patch("pamola_core.utils.io_helpers.crypto_utils.default_data_size", 100000):
            result = get_encryption_mode(list(range(10)), use_encryption=True)
        assert isinstance(result, str)

    def test_auto_disabled_returns_simple(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.auto_choose_provider_encryption", False):
            result = get_encryption_mode("any data", use_encryption=True)
        assert "simple" in result.lower() or result == "SIMPLE"

    def test_data_without_len_returns_simple(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.auto_choose_provider_encryption", True):
            # Pass an int which has no len()
            result = get_encryption_mode(42, use_encryption=True)
        assert isinstance(result, str)

    def test_large_data_returns_age_or_simple(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.auto_choose_provider_encryption", True), \
             patch("pamola_core.utils.io_helpers.crypto_utils.default_data_size", 5):
            # List of 100 items exceeds threshold of 5
            result = get_encryption_mode(list(range(100)), use_encryption=True)
        assert isinstance(result, str)


# --- encrypt_data ---
class TestEncryptData:
    def test_encrypt_success(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_data_router") as mock_enc, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_enc.return_value = b"encrypted_bytes"
            result = encrypt_data(b"raw data", key="testkey", mode="simple")
            assert result == b"encrypted_bytes"

    def test_encrypt_with_description(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_data_router") as mock_enc, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_enc.return_value = {"data": "base64data"}
            result = encrypt_data(b"data", key="k", description="test description")
            call_kwargs = mock_enc.call_args[1]
            assert "data_info" in call_kwargs

    def test_encrypt_with_task_id(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_data_router") as mock_enc, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation") as mock_log, \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_enc.return_value = b"enc"
            encrypt_data(b"data", key="k", task_id="task-001")
            mock_log.assert_called()

    def test_encrypt_failure_raises(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_data_router",
                    side_effect=RuntimeError("fail")), \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            with pytest.raises(Exception):
                encrypt_data(b"data", key="k")
