"""Coverage tests for crypto_utils.py — targets 72 missed lines.
Tests encrypt_file, decrypt_file, decrypt_data paths with mocked crypto providers."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.utils.io_helpers.crypto_utils import (
    encrypt_file,
    decrypt_file,
    decrypt_data,
)


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    return f


# --- encrypt_file ---
class TestEncryptFile:
    def test_encrypt_success(self, sample_file, tmp_path):
        dest = tmp_path / "encrypted.bin"
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_file_router") as mock_enc, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_enc.return_value = dest
            result = encrypt_file(sample_file, dest, key="testkey", mode="simple")
            assert result == dest

    def test_encrypt_with_description(self, sample_file, tmp_path):
        dest = tmp_path / "enc.bin"
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_file_router") as mock_enc, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_enc.return_value = dest
            result = encrypt_file(sample_file, dest, key="k", mode="simple", description="test file")
            assert result == dest
            # Verify file_info was passed
            call_kwargs = mock_enc.call_args[1]
            assert "file_info" in call_kwargs

    def test_encrypt_with_task_id(self, sample_file, tmp_path):
        dest = tmp_path / "enc.bin"
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_file_router") as mock_enc, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation") as mock_log, \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_enc.return_value = dest
            encrypt_file(sample_file, dest, key="k", task_id="task-001")
            mock_log.assert_called()

    def test_encrypt_failure_logs_error(self, sample_file, tmp_path):
        dest = tmp_path / "enc.bin"
        with patch("pamola_core.utils.io_helpers.crypto_utils.encrypt_file_router",
                    side_effect=RuntimeError("encrypt fail")), \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation") as mock_log, \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            with pytest.raises(RuntimeError):
                encrypt_file(sample_file, dest, key="k")
            # Should log failure
            assert any("failure" in str(c) for c in mock_log.call_args_list)


# --- decrypt_file ---
class TestDecryptFile:
    def test_decrypt_success(self, sample_file, tmp_path):
        dest = tmp_path / "decrypted.csv"
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_file_router") as mock_dec, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_dec.return_value = dest
            result = decrypt_file(sample_file, dest, key="testkey", mode="simple")
            assert result == dest

    def test_decrypt_auto_detect_mode(self, sample_file, tmp_path):
        dest = tmp_path / "decrypted.csv"
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_file_router") as mock_dec, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"), \
             patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                    return_value="simple"):
            mock_dec.return_value = dest
            result = decrypt_file(sample_file, dest, key="testkey")
            assert result == dest

    def test_decrypt_failure_logs(self, sample_file, tmp_path):
        dest = tmp_path / "out.csv"
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_file_router",
                    side_effect=RuntimeError("decrypt fail")), \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation") as mock_log, \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            with pytest.raises(RuntimeError):
                decrypt_file(sample_file, dest, key="k")
            assert any("failure" in str(c) for c in mock_log.call_args_list)

    def test_decrypt_with_task_id(self, sample_file, tmp_path):
        dest = tmp_path / "out.csv"
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_file_router") as mock_dec, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_dec.return_value = dest
            decrypt_file(sample_file, dest, key="k", task_id="t-1")


# --- decrypt_data ---
class TestDecryptData:
    def test_decrypt_data_success(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_data_router") as mock_dec, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_dec.return_value = b"decrypted data"
            result = decrypt_data(b"encrypted", key="testkey", mode="simple")
            assert result == b"decrypted data"

    def test_decrypt_data_failure(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_data_router",
                    side_effect=RuntimeError("fail")), \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            with pytest.raises(RuntimeError):
                decrypt_data(b"encrypted", key="k")

    def test_decrypt_data_with_task_id(self):
        with patch("pamola_core.utils.io_helpers.crypto_utils.decrypt_data_router") as mock_dec, \
             patch("pamola_core.utils.io_helpers.crypto_utils.log_crypto_operation"), \
             patch("pamola_core.utils.io_helpers.crypto_utils._ensure_crypto_setup"):
            mock_dec.return_value = b"data"
            decrypt_data(b"enc", key="k", task_id="t-1")
