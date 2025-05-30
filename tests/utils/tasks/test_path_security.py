"""
Tests for the path_security module in the pamola_core/utils/tasks package.

These tests ensure that the path security utilities properly validate, normalize, and secure file paths against traversal, symlink, and system path attacks.
"""

import os
import sys
import tempfile
import shutil
import platform
from pathlib import Path
from unittest import mock
import pytest

from pamola_core.utils.tasks import path_security
from pamola_core.utils.tasks.path_security import (
    validate_path_security,
    is_within_allowed_paths,
    get_system_specific_dangerous_paths,
    validate_paths,
    is_potentially_dangerous_path,
    normalize_and_validate_path,
    PathSecurityError,
)

class DummyLogger:
    def __init__(self):
        self.warnings = []
    def warning(self, msg):
        self.warnings.append(msg)

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(path_security, "logger", dummy)
    return dummy

@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d

@pytest.fixture
def temp_file(temp_dir):
    f = temp_dir / "file.txt"
    f.write_text("test")
    return f

class TestValidatePathSecurity:
    def test_valid_path(self, temp_file):
        assert validate_path_security(temp_file)

    def test_path_traversal(self, temp_dir):
        bad = temp_dir / ".." / "etc" / "passwd"
        with pytest.raises(PathSecurityError):
            validate_path_security(str(bad), strict_mode=True)
        assert not validate_path_security(str(bad), strict_mode=False)

    def test_home_tilde(self, temp_dir):
        bad = "~/secret.txt"
        with pytest.raises(PathSecurityError):
            validate_path_security(bad, strict_mode=True)
        assert not validate_path_security(bad, strict_mode=False)

    def test_command_injection_patterns(self, temp_dir):
        for pattern in ["|", ";", "&", "$", "`", "\\x", "\\u"]:
            bad = f"/tmp/file{pattern}"
            with pytest.raises(PathSecurityError):
                validate_path_security(bad, strict_mode=True)
            assert not validate_path_security(bad, strict_mode=False)

    def test_absolute_system_path(self):
        sys_paths = get_system_specific_dangerous_paths()
        for sys_path in sys_paths:
            if Path(sys_path).is_absolute():
                with pytest.raises(PathSecurityError):
                    validate_path_security(sys_path, strict_mode=True)
                assert not validate_path_security(sys_path, strict_mode=False)

    def test_external_path_not_allowed(self, temp_file, temp_dir):
        ext = temp_file.parent.parent / "external.txt"
        ext.write_text("external")
        allowed = [str(temp_dir)]
        with pytest.raises(PathSecurityError):
            validate_path_security(ext, allowed_paths=allowed, allow_external=False, strict_mode=True)
        assert not validate_path_security(ext, allowed_paths=allowed, allow_external=False, strict_mode=False)

    def test_external_path_allowed(self, temp_file, temp_dir):
        ext = temp_file.parent.parent / "external.txt"
        ext.write_text("external")
        allowed = [str(ext.parent)]
        assert validate_path_security(ext, allowed_paths=allowed, allow_external=True)

    def test_symlink_outside_allowed(self, temp_dir, tmp_path):
        target = tmp_path / "outside.txt"
        target.write_text("outside")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        allowed = [str(temp_dir)]
        with pytest.raises(PathSecurityError):
            validate_path_security(link, allowed_paths=allowed, allow_external=False, strict_mode=True)
        assert not validate_path_security(link, allowed_paths=allowed, allow_external=False, strict_mode=False)

    def test_symlink_inside_allowed(self, temp_dir):
        target = temp_dir / "target.txt"
        target.write_text("ok")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        allowed = [str(temp_dir)]
        assert validate_path_security(link, allowed_paths=allowed, allow_external=False)

    def test_broken_symlink(self, temp_dir):
        link = temp_dir / "broken.txt"
        link.symlink_to(temp_dir / "doesnotexist.txt")
        allowed = [str(temp_dir)]
        # For a broken symlink, validate_path_security returns True regardless of strict_mode
        assert validate_path_security(link, allowed_paths=allowed, allow_external=False, strict_mode=False)
        assert validate_path_security(link, allowed_paths=allowed, allow_external=False, strict_mode=True)

    def test_nonexistent_path(self, temp_dir):
        p = temp_dir / "nope.txt"
        allowed = [str(temp_dir)]
        # Should not raise, just return True (does not exist, so no symlink check)
        assert validate_path_security(p, allowed_paths=allowed, allow_external=False)

    def test_invalid_type(self):
        with pytest.raises(Exception):
            validate_path_security(12345)

    def test_symlink_oserror_strict_false(self, temp_dir, monkeypatch):
        # Simulate OSError in resolve, strict_mode=False
        target = temp_dir / "target.txt"
        target.write_text("ok")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        allowed = [str(temp_dir)]
        monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("fail")))
        assert not validate_path_security(link, allowed_paths=allowed, allow_external=False, strict_mode=False)

    def test_symlink_oserror_strict_true(self, temp_dir, monkeypatch):
        # Simulate OSError in resolve, strict_mode=True
        target = temp_dir / "target.txt"
        target.write_text("ok")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        allowed = [str(temp_dir)]
        monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("fail")))
        with pytest.raises(PathSecurityError):
            validate_path_security(link, allowed_paths=allowed, allow_external=False, strict_mode=True)

    def test_allowed_paths_none(self, temp_file):
        # allowed_paths is None, allow_external is False
        assert validate_path_security(temp_file, allowed_paths=None, allow_external=False)

    def test_real_path_diff_allow_external_true(self, temp_dir, tmp_path):
        # real_path != path_obj but allow_external=True
        target = tmp_path / "target.txt"
        target.write_text("ok")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        allowed = [str(temp_dir)]
        assert validate_path_security(link, allowed_paths=allowed, allow_external=True)

class TestIsWithinAllowedPaths:
    def test_path_within(self, temp_dir):
        allowed = [str(temp_dir)]
        p = temp_dir / "foo.txt"
        assert is_within_allowed_paths(p, allowed)

    def test_path_not_within(self, temp_dir, tmp_path):
        allowed = [str(temp_dir)]
        p = tmp_path / "bar.txt"
        assert not is_within_allowed_paths(p, allowed)

    def test_path_within_symlink(self, temp_dir):
        target = temp_dir / "target.txt"
        target.write_text("ok")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        allowed = [str(temp_dir)]
        assert is_within_allowed_paths(link, allowed)

    def test_path_resolution_error(self, temp_dir, monkeypatch):
        p = temp_dir / "bad.txt"
        allowed = [str(temp_dir)]
        monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("fail")))
        assert is_within_allowed_paths(p, allowed)

    def test_path_and_allowed_path_both_resolve_fail(self, temp_dir, monkeypatch):
        p = temp_dir / "bad.txt"
        allowed = [temp_dir / "other"]
        monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("fail")))
        # Both path and allowed_path resolve fail, fallback to string, which will not match, so should be False
        assert not is_within_allowed_paths(p, allowed)

class TestGetSystemSpecificDangerousPaths:
    def test_returns_list(self):
        paths = get_system_specific_dangerous_paths()
        assert isinstance(paths, list)
        assert all(isinstance(p, str) for p in paths)

    def test_warns_on_unknown(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "AlienOS")
        paths = get_system_specific_dangerous_paths()
        assert "/bin" in paths or "C:\\Windows" in paths

    def test_logger_warning_on_unknown(self, monkeypatch, patch_logger):
        monkeypatch.setattr(platform, "system", lambda: "AlienOS")
        get_system_specific_dangerous_paths()
        assert patch_logger.warnings

class TestValidatePaths:
    def test_all_valid(self, temp_file):
        ok, errors = validate_paths([temp_file])
        assert ok
        assert errors == []

    def test_some_invalid(self, temp_file, temp_dir):
        bad = temp_dir / ".." / "etc" / "passwd"
        ok, errors = validate_paths([temp_file, bad])
        assert not ok
        assert any("Invalid path" in e or "Error validating path" in e for e in errors)

    def test_empty_list(self):
        ok, errors = validate_paths([])
        assert ok
        assert errors == []

    def test_validate_paths_exception(self, temp_file, monkeypatch):
        # Simulate validate_path_security raising
        monkeypatch.setattr(path_security, "validate_path_security", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))
        ok, errors = validate_paths([temp_file])
        assert not ok
        assert any("Error validating path" in e for e in errors)

class TestIsPotentiallyDangerousPath:
    def test_safe(self, temp_file):
        assert not is_potentially_dangerous_path(temp_file)

    def test_dangerous(self, temp_dir):
        bad = temp_dir / ".." / "etc" / "passwd"
        assert is_potentially_dangerous_path(bad)

    def test_exception_in_validate_path_security(self, monkeypatch):
        monkeypatch.setattr(path_security, "validate_path_security", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))
        assert is_potentially_dangerous_path("foo.txt")

class TestNormalizeAndValidatePath:
    def test_relative_path_with_base(self, temp_dir):
        rel = Path("foo.txt")
        out = normalize_and_validate_path(rel, base_dir=temp_dir)
        assert out.parent == temp_dir

    def test_absolute_path(self, temp_file):
        out = normalize_and_validate_path(temp_file)
        assert out == temp_file

    def test_invalid_path_raises(self, temp_dir):
        bad = temp_dir / ".." / "etc" / "passwd"
        with pytest.raises(PathSecurityError):
            normalize_and_validate_path(bad, base_dir=temp_dir)

    def test_invalid_type(self):
        with pytest.raises(Exception):
            normalize_and_validate_path(12345)

    def test_relative_path_no_base(self):
        rel = Path("foo.txt")
        out = normalize_and_validate_path(rel)
        assert out == rel
            
if __name__ == "__main__":
    pytest.main()
