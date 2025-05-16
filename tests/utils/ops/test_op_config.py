"""
PAMOLA.CORE - Unit Tests for Operation Configuration
----------------------------------------------------
Module: tests.utils.ops.test_op_config
Description: Unit tests for the operation configuration module
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Run with pytest -xvs tests/utils/ops/test_op_config.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from pamola_core.utils.ops.op_config import OperationConfig, OperationConfigRegistry, ConfigError, OpsError


class TestOperationConfig:
    """Tests for the OperationConfig base class."""

    def test_init_basic(self):
        """Test basic initialization with valid parameters."""
        config = OperationConfig(name="test", value=42, enabled=True)
        assert config.get("name") == "test"
        assert config.get("value") == 42
        assert config.get("enabled") is True

    def test_init_empty(self):
        """Test initialization with no parameters."""
        config = OperationConfig()
        assert config.to_dict() == {}

    def test_get_existing(self):
        """Test get() with existing parameter."""
        config = OperationConfig(name="test", value=42)
        assert config.get("name") == "test"
        assert config.get("value") == 42

    def test_get_default(self):
        """Test get() with non-existing parameter and default."""
        config = OperationConfig(name="test")
        assert config.get("missing", "default") == "default"
        assert config.get("missing") is None

    def test_getitem(self):
        """Test __getitem__ access."""
        config = OperationConfig(name="test", value=42)
        assert config["name"] == "test"
        assert config["value"] == 42

    def test_getitem_missing(self):
        """Test __getitem__ with missing key raises KeyError."""
        config = OperationConfig(name="test")
        with pytest.raises(KeyError):
            _ = config["missing"]

    def test_contains(self):
        """Test __contains__ method."""
        config = OperationConfig(name="test", value=None)
        assert "name" in config
        assert "value" in config
        assert "missing" not in config

    def test_to_dict(self):
        """Test to_dict() returns a copy of parameters."""
        params = {"name": "test", "value": 42}
        config = OperationConfig(**params)

        result = config.to_dict()
        assert result == params
        assert result is not params  # Should be a copy, not the same object

    def test_update_valid(self):
        """Test update() with valid parameters."""
        config = OperationConfig(name="test", value=42)
        config.update(value=100, new_param="added")

        assert config.get("name") == "test"
        assert config.get("value") == 100
        assert config.get("new_param") == "added"

    def test_repr(self):
        """Test __repr__ method."""
        config = OperationConfig(name="test", value=42)
        repr_str = repr(config)

        assert "OperationConfig" in repr_str
        assert "name='test'" in repr_str
        assert "value=42" in repr_str


class TestSchemaValidation:
    """Tests for schema validation functionality."""

    class ValidatedConfig(OperationConfig):
        """Config class with schema for testing."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer", "minimum": 0, "maximum": 100},
                "mode": {"type": "string", "enum": ["a", "b", "c"]}
            },
            "required": ["name", "value"]
        }

    def test_valid_params(self):
        """Test initialization with valid parameters against schema."""
        config = self.ValidatedConfig(name="test", value=42, mode="a")
        assert config.get("name") == "test"
        assert config.get("value") == 42
        assert config.get("mode") == "a"

    def test_missing_required(self):
        """Test initialization with missing required parameter."""
        with pytest.raises(ConfigError):
            _ = self.ValidatedConfig(name="test")  # Missing required 'value'

    def test_wrong_type(self):
        """Test initialization with wrong parameter type."""
        with pytest.raises(ConfigError):
            _ = self.ValidatedConfig(name="test", value="not-an-integer")

    def test_out_of_range(self):
        """Test initialization with value out of allowed range."""
        with pytest.raises(ConfigError):
            _ = self.ValidatedConfig(name="test", value=200)  # > max 100

    def test_invalid_enum(self):
        """Test initialization with invalid enum value."""
        with pytest.raises(ConfigError):
            _ = self.ValidatedConfig(name="test", value=42, mode="invalid")

    def test_update_invalid(self):
        """Test update() with invalid parameters."""
        config = self.ValidatedConfig(name="test", value=42)

        with pytest.raises(ConfigError):
            config.update(value="not-an-integer")


class TestSerialization:
    """Tests for serialization and deserialization."""

    def test_save_load(self):
        """Test saving and loading configuration."""
        config = OperationConfig(
            name="test_operation",
            value=42,
            nested={"a": 1, "b": 2},
            list_value=[1, 2, 3]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"

            # Save the config
            config.save(path)
            assert path.exists()

            # Verify file contents directly
            with open(path, 'r') as f:
                saved_data = json.load(f)
                assert saved_data == config.to_dict()

            # Load the config
            loaded = OperationConfig.load(path)
            assert loaded.to_dict() == config.to_dict()

    def test_save_load_with_schema(self):
        """Test saving and loading with schema validation."""

        class TestConfig(OperationConfig):
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "integer"}
                },
                "required": ["name"]
            }

        config = TestConfig(name="test", value=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"

            # Save the config
            config.save(path)

            # Load with correct class
            loaded = TestConfig.load(path)
            assert loaded.get("name") == "test"
            assert loaded.get("value") == 42

    def test_load_invalid_json(self):
        """Test loading from invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"

            # Create an invalid JSON file
            with open(path, 'w') as f:
                f.write("{not valid json")

            with pytest.raises(json.JSONDecodeError):
                _ = OperationConfig.load(path)

    def test_load_missing_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            _ = OperationConfig.load(Path("/tmp/does-not-exist.json"))


class TestOperationConfigRegistry:
    """Tests for the OperationConfigRegistry class."""

    @pytest.mark.skip
    class Config1(OperationConfig):
        """Test configuration class 1."""
        pass

    @pytest.mark.skip
    class Config2(OperationConfig):
        """Test configuration class 2."""
        schema = {
            "type": "object",
            "properties": {
                "specific": {"type": "string"}
            }
        }

    def setup_method(self):
        """Set up registry before each test."""
        # Clear registry to avoid test interference
        OperationConfigRegistry._registry = {}

    def test_register_get(self):
        """Test registering and retrieving config classes."""
        # Register configurations
        OperationConfigRegistry.register("type1", self.Config1)
        OperationConfigRegistry.register("type2", self.Config2)

        # Retrieve and verify
        assert OperationConfigRegistry.get_config_class("type1") == self.Config1
        assert OperationConfigRegistry.get_config_class("type2") == self.Config2
        assert OperationConfigRegistry.get_config_class("unknown") is None

    def test_register_override(self):
        """Test registering with the same name overrides."""
        OperationConfigRegistry.register("test", self.Config1)
        assert OperationConfigRegistry.get_config_class("test") == self.Config1

        # Override
        OperationConfigRegistry.register("test", self.Config2)
        assert OperationConfigRegistry.get_config_class("test") == self.Config2

    def test_create_config(self):
        """Test creating config instances from registry."""
        OperationConfigRegistry.register("test", self.Config2)

        # Create with parameters
        config = OperationConfigRegistry.create_config("test", specific="value", other=42)

        assert isinstance(config, self.Config2)
        assert config.get("specific") == "value"
        assert config.get("other") == 42

    def test_create_unregistered(self):
        """Test creating config for unregistered type returns None."""
        config = OperationConfigRegistry.create_config("unknown", param="value")
        assert config is None


class TestErrorClasses:
    """Tests for error classes."""

    def test_ops_error_base(self):
        """Test OpsError is an Exception subclass."""
        assert issubclass(OpsError, Exception)

        # Test instantiation
        error = OpsError("Test error")
        assert str(error) == "Test error"

    def test_config_error(self):
        """Test ConfigError is an OpsError subclass."""
        assert issubclass(ConfigError, OpsError)

        # Test instantiation
        error = ConfigError("Configuration error")
        assert str(error) == "Configuration error"


class TestCustomConfigurations:
    """Tests for creating custom configuration classes."""

    # Define CustomConfig at the class level - without __init__ override
    class CustomConfig(OperationConfig):
        """Custom configuration with additional methods."""

        schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            },
            "required": ["param1"]
        }

        def get_combined(self):
            """Example of custom method."""
            return f"{self.get('param1')}_{self.get('param2')}"

    def test_custom_config_class(self):
        """Test creating and using a custom config class."""
        # Create instance
        config = self.CustomConfig(param1="value1", param2=100)

        # Test standard methods
        assert config.get("param1") == "value1"
        assert config.get("param2") == 100

        # Test custom method
        assert config.get_combined() == "value1_100"

        # Test validation still works
        with pytest.raises(ConfigError):
            _ = self.CustomConfig(param2=100)  # Missing required param1


class TestIntegrationWithJSONUtils:
    """Tests for integration with json_utils.validate_json_schema."""

    def test_validate_json_schema_call(self, monkeypatch):
        """Test that _validate_params calls json_utils.validate_json_schema."""
        # Create a mock for validate_json_schema
        called_with = {}

        def mock_validate(data, schema, error_class):
            called_with['data'] = data
            called_with['schema'] = schema
            called_with['error_class'] = error_class

        # Mock the import and function
        monkeypatch.setattr(
            'pamola_core.utils.io_helpers.json_utils.validate_json_schema',
            mock_validate
        )

        # Create config with schema
        class TestConfig(OperationConfig):
            schema = {"type": "object", "properties": {"test": {"type": "string"}}}

        # Initialize config to trigger validation
        test_params = {"test": "value"}
        config = TestConfig(**test_params)

        # Check the mock was called correctly
        assert called_with['data'] == test_params
        assert called_with['schema'] == TestConfig.schema
        assert called_with['error_class'] == ConfigError


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])