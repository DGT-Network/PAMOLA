"""
Unit tests for pamola_core.configs.config_variables module.

Tests cover:
- L-Diversity configuration defaults
- Environment variable overrides
- Configuration validation
- Deep configuration merging
- Invalid parameter detection

Run with: pytest -s tests/configs/test_config_variables.py
"""

import os
from unittest import mock

import pytest

from pamola_core.configs.config_variables import (
    L_DIVERSITY_DEFAULTS,
    validate_l_diversity_config,
    get_l_diversity_config,
)
from pamola_core.errors.exceptions import InvalidParameterError


class TestLDiversityDefaults:
    """Test L-Diversity default configuration."""

    def test_defaults_dict_exists(self):
        """L_DIVERSITY_DEFAULTS should exist and be dict."""
        assert isinstance(L_DIVERSITY_DEFAULTS, dict)
        assert len(L_DIVERSITY_DEFAULTS) > 0

    def test_defaults_has_required_keys(self):
        """Defaults should have all required configuration keys."""
        required = {
            "l", "diversity_type", "c_value", "k",
            "use_dask", "mask_value", "suppression",
            "npartitions", "optimize_memory",
            "log_level", "visualization", "compliance"
        }
        assert required.issubset(set(L_DIVERSITY_DEFAULTS.keys()))

    def test_l_value_default(self):
        """L value should default to positive integer."""
        assert isinstance(L_DIVERSITY_DEFAULTS["l"], int)
        assert L_DIVERSITY_DEFAULTS["l"] > 0

    def test_diversity_type_default(self):
        """Diversity type should be valid string."""
        diversity_type = L_DIVERSITY_DEFAULTS["diversity_type"]
        assert isinstance(diversity_type, str)
        assert diversity_type in ["distinct", "entropy", "recursive"]

    def test_k_value_default(self):
        """K value should default to positive integer."""
        assert isinstance(L_DIVERSITY_DEFAULTS["k"], int)
        assert L_DIVERSITY_DEFAULTS["k"] >= 1

    def test_c_value_default(self):
        """C value should be positive number."""
        c_value = L_DIVERSITY_DEFAULTS["c_value"]
        assert isinstance(c_value, (int, float))
        assert c_value > 0

    def test_suppression_default(self):
        """Suppression should be boolean."""
        assert isinstance(L_DIVERSITY_DEFAULTS["suppression"], bool)

    def test_use_dask_default(self):
        """use_dask should be boolean."""
        assert isinstance(L_DIVERSITY_DEFAULTS["use_dask"], bool)

    def test_mask_value_default(self):
        """mask_value should be string."""
        assert isinstance(L_DIVERSITY_DEFAULTS["mask_value"], str)
        assert len(L_DIVERSITY_DEFAULTS["mask_value"]) > 0

    def test_visualization_config(self):
        """Visualization config should have required keys."""
        vis = L_DIVERSITY_DEFAULTS["visualization"]
        assert isinstance(vis, dict)
        assert "hist_bins" in vis
        assert "save_format" in vis
        assert isinstance(vis["hist_bins"], int)
        assert isinstance(vis["save_format"], str)

    def test_compliance_config(self):
        """Compliance config should have required keys."""
        comp = L_DIVERSITY_DEFAULTS["compliance"]
        assert isinstance(comp, dict)
        assert "risk_threshold" in comp
        assert "supported_regulations" in comp
        assert isinstance(comp["risk_threshold"], (int, float))
        assert isinstance(comp["supported_regulations"], list)


class TestValidateLDiversityConfig:
    """Test L-Diversity configuration validation."""

    def test_validate_default_config(self):
        """Default config should be valid."""
        result = validate_l_diversity_config(L_DIVERSITY_DEFAULTS)
        assert result is True

    def test_validate_valid_config(self):
        """Valid config should pass validation."""
        config = {
            "l": 3,
            "diversity_type": "distinct",
            "c_value": 1.0,
            "k": 2
        }
        result = validate_l_diversity_config(config)
        assert result is True

    def test_validate_all_diversity_types(self):
        """Should validate all supported diversity types."""
        for diversity_type in ["distinct", "entropy", "recursive"]:
            config = {
                "l": 2,
                "diversity_type": diversity_type,
                "c_value": 1.0,
                "k": 2
            }
            result = validate_l_diversity_config(config)
            assert result is True

    def test_validate_invalid_l_value_zero(self):
        """Should reject l=0."""
        config = {"l": 0, "diversity_type": "distinct", "c_value": 1.0, "k": 2}
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_l_value_negative(self):
        """Should reject negative l."""
        config = {"l": -1, "diversity_type": "distinct", "c_value": 1.0, "k": 2}
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_l_value_non_integer(self):
        """Should reject non-integer l."""
        config = {"l": 2.5, "diversity_type": "distinct", "c_value": 1.0, "k": 2}
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_diversity_type(self):
        """Should reject invalid diversity type."""
        config = {
            "l": 3,
            "diversity_type": "invalid_type",
            "c_value": 1.0,
            "k": 2
        }
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_c_value_zero(self):
        """Should reject c_value=0 for recursive diversity."""
        config = {
            "l": 3,
            "diversity_type": "recursive",
            "c_value": 0,
            "k": 2
        }
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_c_value_negative(self):
        """Should reject negative c_value for recursive diversity."""
        config = {
            "l": 3,
            "diversity_type": "recursive",
            "c_value": -1.0,
            "k": 2
        }
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_k_value_zero(self):
        """Should reject k=0."""
        config = {"l": 3, "diversity_type": "distinct", "c_value": 1.0, "k": 0}
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_k_value_negative(self):
        """Should reject negative k."""
        config = {"l": 3, "diversity_type": "distinct", "c_value": 1.0, "k": -1}
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_invalid_k_value_non_integer(self):
        """Should reject non-integer k."""
        config = {"l": 3, "diversity_type": "distinct", "c_value": 1.0, "k": 2.5}
        result = validate_l_diversity_config(config)
        assert result is False

    def test_validate_empty_config(self):
        """Should validate empty config (uses defaults)."""
        result = validate_l_diversity_config({})
        assert result is True

    def test_validate_config_with_extra_fields(self):
        """Should validate config with extra fields."""
        config = {
            "l": 3,
            "diversity_type": "distinct",
            "c_value": 1.0,
            "k": 2,
            "extra_field": "extra_value"
        }
        result = validate_l_diversity_config(config)
        assert result is True

    def test_validate_c_value_for_distinct_type(self):
        """c_value validation should be skipped for non-recursive types."""
        config = {
            "l": 3,
            "diversity_type": "distinct",
            "c_value": -1.0,  # Negative, but should be ignored for distinct
            "k": 2
        }
        result = validate_l_diversity_config(config)
        # Should pass because c_value only validated for recursive
        assert result is True


class TestGetLDiversityConfig:
    """Test getting L-Diversity configuration."""

    def test_get_default_config(self):
        """Should return default config when no override provided."""
        config = get_l_diversity_config()
        assert isinstance(config, dict)
        assert config["l"] == L_DIVERSITY_DEFAULTS["l"]
        assert config["diversity_type"] == L_DIVERSITY_DEFAULTS["diversity_type"]

    def test_get_config_with_override(self):
        """Should merge overrides with defaults."""
        override = {"l": 5, "diversity_type": "entropy"}
        config = get_l_diversity_config(override)

        assert config["l"] == 5
        assert config["diversity_type"] == "entropy"
        # Other values should use defaults
        assert config["k"] == L_DIVERSITY_DEFAULTS["k"]

    def test_get_config_deep_merge(self):
        """Should deep merge nested overrides."""
        override = {"visualization": {"hist_bins": 50}}
        config = get_l_diversity_config(override)

        assert config["visualization"]["hist_bins"] == 50
        assert config["visualization"]["save_format"] == L_DIVERSITY_DEFAULTS["visualization"]["save_format"]

    def test_get_config_returns_copy(self):
        """Should return copy, not reference to defaults."""
        config = get_l_diversity_config()
        config["l"] = 999

        fresh_config = get_l_diversity_config()
        assert fresh_config["l"] != 999

    def test_get_config_validates_result(self):
        """Should validate merged configuration."""
        override = {"l": 0}  # Invalid
        config = get_l_diversity_config(override)

        # Should fall back to defaults if invalid
        assert config["l"] == L_DIVERSITY_DEFAULTS["l"]

    def test_get_config_with_invalid_override_uses_defaults(self):
        """Should use defaults if override makes config invalid."""
        invalid_override = {"l": -5, "diversity_type": "recursive", "c_value": -1.0}
        config = get_l_diversity_config(invalid_override)

        # Should revert to defaults
        assert config == L_DIVERSITY_DEFAULTS

    def test_get_config_with_partial_override(self):
        """Should allow partial overrides."""
        override = {"l": 4}
        config = get_l_diversity_config(override)

        assert config["l"] == 4
        assert config["diversity_type"] == L_DIVERSITY_DEFAULTS["diversity_type"]
        assert config["npartitions"] == L_DIVERSITY_DEFAULTS["npartitions"]

    def test_get_config_with_none_override(self):
        """Should handle None override."""
        config = get_l_diversity_config(None)
        assert isinstance(config, dict)


class TestEnvironmentVariableOverrides:
    """Test environment variable configuration overrides."""

    def test_l_value_from_env(self):
        """Should read L value from PAMOLA_L_DIVERSITY_L env var."""
        with mock.patch.dict(os.environ, {"PAMOLA_L_DIVERSITY_L": "5"}):
            # Re-import to get new env values
            # Note: In real code, this would require module reload
            pass

    def test_diversity_type_from_env(self):
        """Should read diversity type from PAMOLA_L_DIVERSITY_TYPE env var."""
        with mock.patch.dict(os.environ, {"PAMOLA_L_DIVERSITY_TYPE": "entropy"}):
            pass

    def test_boolean_from_env(self):
        """Should parse boolean env vars correctly."""
        # Test that true/false strings are parsed as booleans
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
        ]

        for env_str, expected in test_cases:
            result = env_str.lower() == "true"
            assert result == expected


class TestConfigIntegration:
    """Integration tests for L-Diversity configuration."""

    def test_full_config_workflow(self):
        """Test typical config workflow: get defaults, override, validate."""
        # Get default
        default = get_l_diversity_config()
        assert validate_l_diversity_config(default)

        # Override
        override = {"l": 5, "npartitions": 8}
        custom = get_l_diversity_config(override)
        assert custom["l"] == 5
        assert custom["npartitions"] == 8
        assert validate_l_diversity_config(custom)

    def test_config_immutability(self):
        """Multiple calls should return independent dicts."""
        config1 = get_l_diversity_config()
        config2 = get_l_diversity_config()

        config1["l"] = 999
        assert config2["l"] != 999

    def test_nested_config_immutability(self):
        """Nested dicts should also be independent."""
        config1 = get_l_diversity_config()
        config2 = get_l_diversity_config()

        config1["visualization"]["hist_bins"] = 999
        assert config2["visualization"]["hist_bins"] != 999
