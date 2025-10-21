"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Schema Management System
Package:       pamola_core.metrics.commons.schema_manager
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides schema management for field definitions and validation rules.
  Handles field metadata, rule assignments, and schema persistence.

Key Features:
  - Field definition management
  - Validation rule assignment per field
  - Schema persistence and loading
  - Auto-detection of field types and rules
  - Integration with validation rules framework

Dependencies:
  - pandas - DataFrame operations
  - typing - type hints
  - dataclasses - data structures
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class FieldDefinition:
    """Definition of a field with its metadata and validation rules."""

    name: str
    data_type: str
    is_required: bool = False
    is_unique: bool = False
    validation_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default rules based on field properties."""
        # Add default rules based on field properties
        if self.is_required and "REQUIRED" not in self.validation_rules:
            self.validation_rules.append("REQUIRED")
        if self.is_unique and "UNIQUE" not in self.validation_rules:
            self.validation_rules.append("UNIQUE")

    def add_rule(self, rule_name: str) -> None:
        """Add a validation rule to this field."""
        if rule_name not in self.validation_rules:
            self.validation_rules.append(rule_name)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule from this field."""
        if rule_name in self.validation_rules:
            self.validation_rules.remove(rule_name)
            return True
        return False

    def has_rule(self, rule_name: str) -> bool:
        """Check if field has a specific validation rule."""
        return rule_name in self.validation_rules

    def to_dict(self) -> Dict[str, Any]:
        """Convert field definition to dictionary."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "is_required": self.is_required,
            "is_unique": self.is_unique,
            "validation_rules": self.validation_rules,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldDefinition":
        """Create field definition from dictionary."""
        return cls(
            name=data["name"],
            data_type=data["data_type"],
            is_required=data.get("is_required", False),
            is_unique=data.get("is_unique", False),
            validation_rules=data.get("validation_rules", []),
            metadata=data.get("metadata", {}),
        )


class SchemaManager:
    """Manages schema definitions and validation rules for datasets."""

    def __init__(self):
        """Initialize schema manager."""
        self.fields: Dict[str, FieldDefinition] = {}
        self._auto_detected = False

    def add_field(self, field_def: FieldDefinition) -> None:
        """
        Add a field definition to the schema.

        Parameters
        ----------
        field_def : FieldDefinition
            Field definition to add
        """
        self.fields[field_def.name] = field_def

    def get_field(self, field_name: str) -> Optional[FieldDefinition]:
        """
        Get field definition by name.

        Parameters
        ----------
        field_name : str
            Name of the field

        Returns
        -------
        Optional[FieldDefinition]
            Field definition if found, None otherwise
        """
        return self.fields.get(field_name)

    def remove_field(self, field_name: str) -> bool:
        """
        Remove a field from the schema.

        Parameters
        ----------
        field_name : str
            Name of the field to remove

        Returns
        -------
        bool
            True if field was removed, False if not found
        """
        if field_name in self.fields:
            del self.fields[field_name]
            return True
        return False

    def auto_detect_schema(self, df: pd.DataFrame) -> None:
        """
        Automatically detect schema from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze for schema detection
        """
        self.fields.clear()

        for column in df.columns:
            field_def = self._detect_field_definition(df[column], column)
            self.add_field(field_def)

        self._auto_detected = True
        logger.info(f"Auto-detected schema for {len(df.columns)} fields")

    def _detect_field_definition(
        self, series: pd.Series, column_name: str
    ) -> FieldDefinition:
        """
        Detect field definition from a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Data series to analyze
        column_name : str
            Name of the column

        Returns
        -------
        FieldDefinition
            Detected field definition
        """
        # Detect data type
        data_type = self._detect_data_type(series)

        # Detect if it should be required
        is_required = self._should_be_required(series, column_name)

        # Detect if it should be unique
        is_unique = self._should_be_unique(series, column_name)

        # Create field definition
        field_def = FieldDefinition(
            name=column_name,
            data_type=data_type,
            is_required=is_required,
            is_unique=is_unique,
        )

        # Add datatype validation rule
        if data_type in ["int", "float", "date", "bool", "email", "phone"]:
            field_def.add_rule(f"Datatype ({data_type})")

        return field_def

    def _detect_data_type(self, series: pd.Series) -> str:
        """Detect the most appropriate data type for a series."""
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's integer or float
            if pd.api.types.is_integer_dtype(series):
                return "int"
            else:
                return "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        elif pd.api.types.is_bool_dtype(series):
            return "bool"
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        ):
            # Check for email or phone patterns
            sample_values = series.dropna().head(100)  # Sample for performance
            if len(sample_values) > 0:
                email_count = sum(1 for v in sample_values if self._is_email(str(v)))
                phone_count = sum(1 for v in sample_values if self._is_phone(str(v)))

                if email_count > len(sample_values) * 0.8:
                    return "email"
                elif phone_count > len(sample_values) * 0.8:
                    return "phone"

            return "string"
        else:
            return "unknown"

    def _is_email(self, value: str) -> bool:
        """Check if string looks like an email."""
        import re

        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        return bool(email_pattern.match(value))

    def _is_phone(self, value: str) -> bool:
        """Check if string looks like a phone number."""
        import re

        phone_pattern = re.compile(r"^\+?[0-9\-\(\)\s]{7,20}$")
        return bool(phone_pattern.match(value))

    def _should_be_required(
        self, series: pd.Series, column_name: str
    ) -> bool:
        """Determine if field should be required."""
        # Check if field has very low missing rate (likely required)
        missing_rate = series.isna().mean()
        if missing_rate < 0.05:  # Less than 5% missing
            return True

        return False

    def _should_be_unique(
        self, series: pd.Series, column_name: str
    ) -> bool:
        """Determine if field should be unique."""
        # Check uniqueness ratio
        clean_series = series.dropna()
        if len(clean_series) > 0:
            unique_ratio = clean_series.nunique() / len(clean_series)
            if unique_ratio > 0.95:  # Very high uniqueness
                return True

        return False

    def get_required_fields(self) -> List[str]:
        """Get list of required field names."""
        return [name for name, field in self.fields.items() if field.is_required]

    def get_unique_fields(self) -> List[str]:
        """Get list of unique field names."""
        return [name for name, field in self.fields.items() if field.is_unique]

    def update_field_rules(self, field_name: str, rules: List[str]) -> bool:
        """
        Update validation rules for a field.

        Parameters
        ----------
        field_name : str
            Name of the field
        rules : List[str]
            List of rule names to assign

        Returns
        -------
        bool
            True if field was updated, False if field not found
        """
        if field_name not in self.fields:
            return False

        self.fields[field_name].validation_rules = rules.copy()
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "fields": {name: field.to_dict() for name, field in self.fields.items()},
            "auto_detected": self._auto_detected,
        }

    def to_json(self) -> str:
        """Convert schema to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaManager":
        """Create schema manager from dictionary."""
        schema = cls()
        schema._auto_detected = data.get("auto_detected", False)

        for field_data in data.get("fields", {}).values():
            field_def = FieldDefinition.from_dict(field_data)
            schema.add_field(field_def)

        return schema

    @classmethod
    def from_json(cls, json_str: str) -> "SchemaManager":
        """Create schema manager from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, filepath: str) -> None:
        """Save schema to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str) -> "SchemaManager":
        """Load schema from JSON file."""
        with open(filepath, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str)
