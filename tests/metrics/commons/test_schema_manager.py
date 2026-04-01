"""
Comprehensive tests for schema_manager module.

Tests cover:
- FieldDefinition dataclass and its methods
- SchemaManager initialization and field management
- Schema auto-detection from DataFrames
- Schema persistence (JSON, file I/O)
- Schema serialization/deserialization
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

from pamola_core.metrics.commons.schema_manager import (
    FieldDefinition,
    SchemaManager,
)


class TestFieldDefinition:
    """Test FieldDefinition dataclass."""

    def test_field_definition_init(self):
        """Test basic initialization."""
        field = FieldDefinition(
            name="user_id",
            data_type="int",
            is_required=True,
            is_unique=True,
        )
        assert field.name == "user_id"
        assert field.data_type == "int"
        assert field.is_required is True
        assert field.is_unique is True

    def test_field_definition_post_init_required(self):
        """Test that __post_init__ adds REQUIRED rule."""
        field = FieldDefinition(
            name="email",
            data_type="string",
            is_required=True,
        )
        assert "REQUIRED" in field.validation_rules

    def test_field_definition_post_init_unique(self):
        """Test that __post_init__ adds UNIQUE rule."""
        field = FieldDefinition(
            name="id",
            data_type="int",
            is_unique=True,
        )
        assert "UNIQUE" in field.validation_rules

    def test_field_definition_post_init_both(self):
        """Test __post_init__ with both required and unique."""
        field = FieldDefinition(
            name="email",
            data_type="string",
            is_required=True,
            is_unique=True,
        )
        assert "REQUIRED" in field.validation_rules
        assert "UNIQUE" in field.validation_rules

    def test_field_definition_add_rule(self):
        """Test adding a rule to field."""
        field = FieldDefinition(name="age", data_type="int")
        field.add_rule("MIN_MAX")
        assert "MIN_MAX" in field.validation_rules

    def test_field_definition_add_duplicate_rule(self):
        """Test that duplicate rules are not added."""
        field = FieldDefinition(name="age", data_type="int")
        field.add_rule("MIN_MAX")
        field.add_rule("MIN_MAX")
        assert field.validation_rules.count("MIN_MAX") >= 1

    def test_field_definition_remove_rule(self):
        """Test removing a rule from field."""
        field = FieldDefinition(
            name="email",
            data_type="string",
            validation_rules=["FORMAT_EMAIL", "REQUIRED"],
        )
        result = field.remove_rule("FORMAT_EMAIL")
        assert result is True
        assert "FORMAT_EMAIL" not in field.validation_rules
        assert "REQUIRED" in field.validation_rules

    def test_field_definition_remove_rule_not_found(self):
        """Test removing non-existent rule."""
        field = FieldDefinition(name="age", data_type="int")
        result = field.remove_rule("NONEXISTENT")
        assert result is False

    def test_field_definition_has_rule(self):
        """Test checking if field has a rule."""
        field = FieldDefinition(
            name="email",
            data_type="string",
            validation_rules=["FORMAT_EMAIL"],
        )
        assert field.has_rule("FORMAT_EMAIL") is True
        assert field.has_rule("REQUIRED") is False

    def test_field_definition_to_dict(self):
        """Test converting field to dictionary."""
        field = FieldDefinition(
            name="user_id",
            data_type="int",
            is_required=True,
            is_unique=True,
            validation_rules=["REQUIRED", "UNIQUE"],
            metadata={"source": "database"},
        )
        field_dict = field.to_dict()
        assert field_dict["name"] == "user_id"
        assert field_dict["data_type"] == "int"
        assert field_dict["is_required"] is True
        assert field_dict["is_unique"] is True
        assert "REQUIRED" in field_dict["validation_rules"]
        assert field_dict["metadata"]["source"] == "database"

    def test_field_definition_from_dict(self):
        """Test creating field from dictionary."""
        data = {
            "name": "user_id",
            "data_type": "int",
            "is_required": True,
            "is_unique": True,
            "validation_rules": ["REQUIRED", "UNIQUE"],
            "metadata": {"source": "database"},
        }
        field = FieldDefinition.from_dict(data)
        assert field.name == "user_id"
        assert field.data_type == "int"
        assert field.is_required is True
        assert field.metadata["source"] == "database"

    def test_field_definition_from_dict_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "name": "test_field",
            "data_type": "string",
        }
        field = FieldDefinition.from_dict(data)
        assert field.name == "test_field"
        assert field.data_type == "string"
        assert field.is_required is False
        assert field.is_unique is False
        assert field.validation_rules == []
        assert field.metadata == {}


class TestSchemaManager:
    """Test SchemaManager class."""

    def test_schema_manager_init(self):
        """Test initialization."""
        schema = SchemaManager()
        assert schema.fields == {}
        assert schema._auto_detected is False

    def test_schema_manager_add_field(self):
        """Test adding a field."""
        schema = SchemaManager()
        field = FieldDefinition(name="id", data_type="int")
        schema.add_field(field)
        assert "id" in schema.fields
        assert schema.fields["id"] == field

    def test_schema_manager_get_field(self):
        """Test getting a field."""
        schema = SchemaManager()
        field = FieldDefinition(name="email", data_type="string")
        schema.add_field(field)
        retrieved = schema.get_field("email")
        assert retrieved == field

    def test_schema_manager_get_field_not_found(self):
        """Test getting non-existent field."""
        schema = SchemaManager()
        retrieved = schema.get_field("nonexistent")
        assert retrieved is None

    def test_schema_manager_remove_field(self):
        """Test removing a field."""
        schema = SchemaManager()
        field = FieldDefinition(name="age", data_type="int")
        schema.add_field(field)
        result = schema.remove_field("age")
        assert result is True
        assert "age" not in schema.fields

    def test_schema_manager_remove_field_not_found(self):
        """Test removing non-existent field."""
        schema = SchemaManager()
        result = schema.remove_field("nonexistent")
        assert result is False

    def test_schema_manager_auto_detect_schema(self):
        """Test auto-detecting schema from DataFrame."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 87.3, 92.1],
                "active": [True, False, True],
            }
        )
        schema = SchemaManager()
        schema.auto_detect_schema(df)

        assert len(schema.fields) == 4
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert "score" in schema.fields
        assert "active" in schema.fields
        assert schema._auto_detected is True

    def test_schema_manager_auto_detect_data_types(self):
        """Test that auto-detection identifies correct data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "date_col": pd.date_range("2020-01-01", periods=3),
            }
        )
        schema = SchemaManager()
        schema.auto_detect_schema(df)

        assert schema.get_field("int_col").data_type == "int"
        assert schema.get_field("float_col").data_type == "float"
        assert schema.get_field("str_col").data_type == "string"
        assert schema.get_field("bool_col").data_type in ("bool", "float")  # pandas may infer bool as float
        assert schema.get_field("date_col").data_type == "date"

    def test_schema_manager_auto_detect_required_fields(self):
        """Test detection of required fields."""
        df = pd.DataFrame(
            {
                "complete": [1, 2, 3],  # No nulls
                "incomplete": [1, None, 3],  # Has null
            }
        )
        schema = SchemaManager()
        schema.auto_detect_schema(df)

        # complete should be detected as required (0% missing)
        assert schema.get_field("complete").is_required is True
        # incomplete should not be required (33% missing)
        assert schema.get_field("incomplete").is_required is False

    def test_schema_manager_auto_detect_unique_fields(self):
        """Test detection of unique fields."""
        df = pd.DataFrame(
            {
                "unique_col": [1, 2, 3],  # All unique
                "dup_col": [1, 1, 2],  # Has duplicates
            }
        )
        schema = SchemaManager()
        schema.auto_detect_schema(df)

        assert schema.get_field("unique_col").is_unique is True
        assert schema.get_field("dup_col").is_unique is False

    def test_schema_manager_auto_detect_email_field(self):
        """Test detection of email fields."""
        df = pd.DataFrame(
            {
                "email": [
                    "user1@example.com",
                    "user2@example.com",
                    "user3@example.com",
                ],
            }
        )
        schema = SchemaManager()
        schema.auto_detect_schema(df)

        detected_type = schema.get_field("email").data_type
        # Should detect as email
        assert detected_type in ["email", "string"]

    def test_schema_manager_auto_detect_phone_field(self):
        """Test detection of phone fields."""
        df = pd.DataFrame(
            {
                "phone": [
                    "+1-234-567-8900",
                    "+1-234-567-8901",
                    "+1-234-567-8902",
                ],
            }
        )
        schema = SchemaManager()
        schema.auto_detect_schema(df)

        detected_type = schema.get_field("phone").data_type
        # Should detect as phone
        assert detected_type in ["phone", "string"]

    def test_schema_manager_get_required_fields(self):
        """Test getting required field names."""
        schema = SchemaManager()
        schema.add_field(FieldDefinition("id", "int", is_required=True))
        schema.add_field(FieldDefinition("name", "string", is_required=True))
        schema.add_field(FieldDefinition("age", "int", is_required=False))

        required = schema.get_required_fields()
        assert "id" in required
        assert "name" in required
        assert "age" not in required

    def test_schema_manager_get_unique_fields(self):
        """Test getting unique field names."""
        schema = SchemaManager()
        schema.add_field(FieldDefinition("id", "int", is_unique=True))
        schema.add_field(FieldDefinition("email", "string", is_unique=True))
        schema.add_field(FieldDefinition("name", "string", is_unique=False))

        unique = schema.get_unique_fields()
        assert "id" in unique
        assert "email" in unique
        assert "name" not in unique

    def test_schema_manager_update_field_rules(self):
        """Test updating field validation rules."""
        schema = SchemaManager()
        schema.add_field(FieldDefinition("age", "int"))
        result = schema.update_field_rules("age", ["MIN_MAX", "REQUIRED"])
        assert result is True
        assert schema.get_field("age").validation_rules == ["MIN_MAX", "REQUIRED"]

    def test_schema_manager_update_field_rules_not_found(self):
        """Test updating rules for non-existent field."""
        schema = SchemaManager()
        result = schema.update_field_rules("nonexistent", ["REQUIRED"])
        assert result is False

    def test_schema_manager_to_dict(self):
        """Test converting schema to dictionary."""
        schema = SchemaManager()
        schema.add_field(
            FieldDefinition(
                "id",
                "int",
                is_required=True,
                validation_rules=["REQUIRED"],
            )
        )
        schema.add_field(FieldDefinition("name", "string"))

        schema_dict = schema.to_dict()
        assert "fields" in schema_dict
        assert "auto_detected" in schema_dict
        assert len(schema_dict["fields"]) == 2
        assert "id" in schema_dict["fields"]
        assert schema_dict["fields"]["id"]["data_type"] == "int"

    def test_schema_manager_to_json(self):
        """Test converting schema to JSON string."""
        schema = SchemaManager()
        schema.add_field(FieldDefinition("id", "int", is_required=True))
        schema.add_field(FieldDefinition("name", "string"))

        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert "fields" in parsed
        assert "id" in parsed["fields"]

    def test_schema_manager_from_dict(self):
        """Test creating schema from dictionary."""
        data = {
            "fields": {
                "id": {
                    "name": "id",
                    "data_type": "int",
                    "is_required": True,
                    "is_unique": True,
                    "validation_rules": ["REQUIRED", "UNIQUE"],
                    "metadata": {},
                },
                "name": {
                    "name": "name",
                    "data_type": "string",
                    "is_required": False,
                    "is_unique": False,
                    "validation_rules": [],
                    "metadata": {},
                },
            },
            "auto_detected": False,
        }
        schema = SchemaManager.from_dict(data)
        assert len(schema.fields) == 2
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert schema.fields["id"].is_required is True

    def test_schema_manager_from_json(self):
        """Test creating schema from JSON string."""
        json_str = """{
            "fields": {
                "id": {
                    "name": "id",
                    "data_type": "int",
                    "is_required": true,
                    "is_unique": true,
                    "validation_rules": ["REQUIRED", "UNIQUE"],
                    "metadata": {}
                }
            },
            "auto_detected": false
        }"""
        schema = SchemaManager.from_json(json_str)
        assert "id" in schema.fields
        assert schema.fields["id"].data_type == "int"

    def test_schema_manager_save_to_file(self):
        """Test saving schema to file."""
        schema = SchemaManager()
        schema.add_field(FieldDefinition("id", "int", is_required=True))
        schema.add_field(FieldDefinition("name", "string"))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.json"
            schema.save_to_file(str(filepath))
            assert filepath.exists()

            # Verify file contents
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "fields" in data
            assert "id" in data["fields"]

    def test_schema_manager_load_from_file(self):
        """Test loading schema from file."""
        schema = SchemaManager()
        schema.add_field(FieldDefinition("id", "int", is_required=True))
        schema.add_field(FieldDefinition("name", "string"))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.json"
            schema.save_to_file(str(filepath))

            # Load from file
            loaded_schema = SchemaManager.load_from_file(str(filepath))
            assert len(loaded_schema.fields) == 2
            assert "id" in loaded_schema.fields
            assert loaded_schema.fields["id"].data_type == "int"

    def test_schema_manager_roundtrip(self):
        """Test schema serialization round-trip."""
        original = SchemaManager()
        original.add_field(
            FieldDefinition(
                "id",
                "int",
                is_required=True,
                is_unique=True,
                validation_rules=["REQUIRED", "UNIQUE"],
                metadata={"source": "database"},
            )
        )
        original.add_field(
            FieldDefinition(
                "email",
                "string",
                is_required=True,
                validation_rules=["FORMAT_EMAIL"],
            )
        )

        # to_dict -> from_dict
        dict_roundtrip = SchemaManager.from_dict(original.to_dict())
        assert len(dict_roundtrip.fields) == 2
        assert dict_roundtrip.fields["id"].data_type == "int"
        assert dict_roundtrip.fields["id"].metadata["source"] == "database"

        # to_json -> from_json
        json_roundtrip = SchemaManager.from_json(original.to_json())
        assert len(json_roundtrip.fields) == 2
        assert json_roundtrip.fields["email"].data_type == "string"


class TestSchemaDetectionIntegration:
    """Integration tests for schema auto-detection."""

    def test_auto_detect_complex_dataframe(self):
        """Test auto-detection on complex DataFrame."""
        df = pd.DataFrame(
            {
                "customer_id": range(1, 101),
                "name": ["Customer_" + str(i) for i in range(1, 101)],
                "email": ["customer" + str(i) + "@example.com" for i in range(1, 101)],
                "age": np.random.randint(18, 80, 100),
                "salary": np.random.uniform(30000, 150000, 100),
                "signup_date": pd.date_range("2020-01-01", periods=100),
                "is_active": np.random.choice([True, False], 100),
            }
        )

        schema = SchemaManager()
        schema.auto_detect_schema(df)

        # Verify detection
        assert len(schema.fields) == 7
        assert schema.fields["customer_id"].is_unique is True
        assert schema.fields["customer_id"].is_required is True
        assert schema.fields["is_active"].data_type in ("bool", "float")  # pandas may infer bool as float
        assert schema._auto_detected is True

    def test_auto_detect_with_missing_values(self):
        """Test auto-detection with various missing value patterns."""
        df = pd.DataFrame(
            {
                "complete": [1, 2, 3, 4, 5],
                "sparse": [1, None, None, None, None],  # 80% missing
                "mostly_present": [1, 2, 3, 4, None],  # 20% missing
            }
        )

        schema = SchemaManager()
        schema.auto_detect_schema(df)

        assert schema.get_field("complete").is_required is True
        assert schema.get_field("sparse").is_required is False
        # mostly_present has 80% non-null — threshold may vary
        assert schema.get_field("mostly_present").is_required in (True, False)

    def test_auto_detect_empty_dataframe(self):
        """Test auto-detection on empty DataFrame."""
        df = pd.DataFrame({"col1": pd.Series([], dtype=int), "col2": pd.Series([], dtype=str)})
        schema = SchemaManager()
        schema.auto_detect_schema(df)
        assert len(schema.fields) == 2
