"""
Unit tests for pamola_core.configs.field_definitions module.

Tests cover:
- Field type enumeration
- Privacy category enumeration
- Anonymization strategy enumeration
- Profiling task enumeration
- Table definitions
- Enum member values and iteration

Run with: pytest -s tests/configs/test_field_definitions.py
"""

import pytest

from pamola_core.configs.field_definitions import (
    FieldType,
    PrivacyCategory,
    AnonymizationStrategy,
    ProfilingTask,
    TABLES,
)


class TestFieldType:
    """Test FieldType enumeration."""

    def test_field_type_has_required_members(self):
        """FieldType should have all required members."""
        required = {"SHORT_TEXT", "LONG_TEXT", "DOUBLE", "LONG", "DATE"}
        members = {m.name for m in FieldType}
        assert required.issubset(members)

    def test_short_text_value(self):
        """SHORT_TEXT should have correct string value."""
        assert FieldType.SHORT_TEXT.value == "short_text"

    def test_long_text_value(self):
        """LONG_TEXT should have correct string value."""
        assert FieldType.LONG_TEXT.value == "long_text"

    def test_double_value(self):
        """DOUBLE should have correct string value."""
        assert FieldType.DOUBLE.value == "double"

    def test_long_value(self):
        """LONG should have correct string value."""
        assert FieldType.LONG.value == "long"

    def test_date_value(self):
        """DATE should have correct string value."""
        assert FieldType.DATE.value == "date"

    def test_field_type_all_are_strings(self):
        """All FieldType values should be strings."""
        for member in FieldType:
            assert isinstance(member.value, str)

    def test_field_type_iteration(self):
        """Should be able to iterate over all FieldType members."""
        count = len(list(FieldType))
        assert count >= 5

    def test_field_type_by_value(self):
        """Should be able to lookup by value."""
        ft = FieldType("short_text")
        assert ft == FieldType.SHORT_TEXT

    def test_field_type_invalid_value(self):
        """Should raise ValueError for invalid value."""
        with pytest.raises(ValueError):
            FieldType("invalid_type")


class TestPrivacyCategory:
    """Test PrivacyCategory enumeration."""

    def test_privacy_category_has_required_members(self):
        """PrivacyCategory should have all required members."""
        required = {
            "DIRECT_IDENTIFIER",
            "INDIRECT_IDENTIFIER",
            "QUASI_IDENTIFIER",
            "SENSITIVE_ATTRIBUTE",
            "NON_SENSITIVE"
        }
        members = {m.name for m in PrivacyCategory}
        assert required.issubset(members)

    def test_direct_identifier_value(self):
        """DIRECT_IDENTIFIER should have correct value."""
        assert PrivacyCategory.DIRECT_IDENTIFIER.value == "Direct Identifier"

    def test_indirect_identifier_value(self):
        """INDIRECT_IDENTIFIER should have correct value."""
        assert PrivacyCategory.INDIRECT_IDENTIFIER.value == "Indirect Identifier"

    def test_quasi_identifier_value(self):
        """QUASI_IDENTIFIER should have correct value."""
        assert PrivacyCategory.QUASI_IDENTIFIER.value == "Quasi-Identifier"

    def test_sensitive_attribute_value(self):
        """SENSITIVE_ATTRIBUTE should have correct value."""
        assert PrivacyCategory.SENSITIVE_ATTRIBUTE.value == "Sensitive Attribute"

    def test_non_sensitive_value(self):
        """NON_SENSITIVE should have correct value."""
        assert PrivacyCategory.NON_SENSITIVE.value == "Non-Sensitive"

    def test_privacy_category_all_are_strings(self):
        """All PrivacyCategory values should be strings."""
        for member in PrivacyCategory:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    def test_privacy_category_iteration(self):
        """Should be able to iterate over all members."""
        count = len(list(PrivacyCategory))
        assert count == 5

    def test_privacy_category_by_value(self):
        """Should be able to lookup by value."""
        pc = PrivacyCategory("Direct Identifier")
        assert pc == PrivacyCategory.DIRECT_IDENTIFIER

    def test_privacy_category_invalid_value(self):
        """Should raise ValueError for invalid value."""
        with pytest.raises(ValueError):
            PrivacyCategory("Invalid Category")


class TestAnonymizationStrategy:
    """Test AnonymizationStrategy enumeration."""

    def test_anonymization_strategy_has_required_members(self):
        """AnonymizationStrategy should have all required members."""
        required = {
            "PSEUDONYMIZATION",
            "GENERALIZATION",
            "SUPPRESSION",
            "KEEP_ORIGINAL",
            "NOISE_ADDITION",
            "NER_LLM_RECONSTRUCTION"
        }
        members = {m.name for m in AnonymizationStrategy}
        assert required.issubset(members)

    def test_pseudonymization_value(self):
        """PSEUDONYMIZATION should have correct value."""
        assert AnonymizationStrategy.PSEUDONYMIZATION.value == "PSEUDONYMIZATION"

    def test_generalization_value(self):
        """GENERALIZATION should have correct value."""
        assert AnonymizationStrategy.GENERALIZATION.value == "GENERALIZATION"

    def test_suppression_value(self):
        """SUPPRESSION should have correct value."""
        assert AnonymizationStrategy.SUPPRESSION.value == "SUPPRESSION"

    def test_keep_original_value(self):
        """KEEP_ORIGINAL should have correct value."""
        assert AnonymizationStrategy.KEEP_ORIGINAL.value == "KEEP_ORIGINAL"

    def test_noise_addition_value(self):
        """NOISE_ADDITION should have correct value."""
        assert AnonymizationStrategy.NOISE_ADDITION.value == "NOISE_ADDITION"

    def test_ner_llm_reconstruction_value(self):
        """NER_LLM_RECONSTRUCTION should have correct value."""
        assert AnonymizationStrategy.NER_LLM_RECONSTRUCTION.value == "NER/LLM RECONSTRUCTION"

    def test_anonymization_strategy_all_strings(self):
        """All AnonymizationStrategy values should be strings."""
        for member in AnonymizationStrategy:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    def test_anonymization_strategy_iteration(self):
        """Should be able to iterate over all members."""
        count = len(list(AnonymizationStrategy))
        assert count == 6

    def test_anonymization_strategy_by_value(self):
        """Should be able to lookup by value."""
        strategy = AnonymizationStrategy("PSEUDONYMIZATION")
        assert strategy == AnonymizationStrategy.PSEUDONYMIZATION

    def test_anonymization_strategy_invalid_value(self):
        """Should raise ValueError for invalid value."""
        with pytest.raises(ValueError):
            AnonymizationStrategy("INVALID_STRATEGY")


class TestProfilingTask:
    """Test ProfilingTask enumeration."""

    def test_profiling_task_has_required_members(self):
        """ProfilingTask should have required members."""
        required = {
            "COMPLETENESS",
            "UNIQUENESS",
            "DISTRIBUTION",
            "FREQUENCY",
            "OUTLIERS",
            "RARE_VALUES",
            "FORMAT_VALIDATION",
            "CORRELATION",
            "TEXT_ANALYSIS",
            "LENGTH_ANALYSIS",
            "PATTERN_DETECTION"
        }
        members = {m.name for m in ProfilingTask}
        assert required.issubset(members)

    def test_completeness_value(self):
        """COMPLETENESS should have correct value."""
        assert ProfilingTask.COMPLETENESS.value == "completeness"

    def test_uniqueness_value(self):
        """UNIQUENESS should have correct value."""
        assert ProfilingTask.UNIQUENESS.value == "uniqueness"

    def test_distribution_value(self):
        """DISTRIBUTION should have correct value."""
        assert ProfilingTask.DISTRIBUTION.value == "distribution"

    def test_frequency_value(self):
        """FREQUENCY should have correct value."""
        assert ProfilingTask.FREQUENCY.value == "frequency"

    def test_outliers_value(self):
        """OUTLIERS should have correct value."""
        assert ProfilingTask.OUTLIERS.value == "outliers"

    def test_rare_values_value(self):
        """RARE_VALUES should have correct value."""
        assert ProfilingTask.RARE_VALUES.value == "rare_values"

    def test_format_validation_value(self):
        """FORMAT_VALIDATION should have correct value."""
        assert ProfilingTask.FORMAT_VALIDATION.value == "format_validation"

    def test_correlation_value(self):
        """CORRELATION should have correct value."""
        assert ProfilingTask.CORRELATION.value == "correlation"

    def test_text_analysis_value(self):
        """TEXT_ANALYSIS should have correct value."""
        assert ProfilingTask.TEXT_ANALYSIS.value == "text_analysis"

    def test_length_analysis_value(self):
        """LENGTH_ANALYSIS should have correct value."""
        assert ProfilingTask.LENGTH_ANALYSIS.value == "length_analysis"

    def test_pattern_detection_value(self):
        """PATTERN_DETECTION should have correct value."""
        assert ProfilingTask.PATTERN_DETECTION.value == "pattern_detection"

    def test_profiling_task_all_strings(self):
        """All ProfilingTask values should be strings."""
        for member in ProfilingTask:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    def test_profiling_task_iteration(self):
        """Should be able to iterate over all members."""
        count = len(list(ProfilingTask))
        assert count >= 11

    def test_profiling_task_by_value(self):
        """Should be able to lookup by value."""
        task = ProfilingTask("completeness")
        assert task == ProfilingTask.COMPLETENESS

    def test_profiling_task_invalid_value(self):
        """Should raise ValueError for invalid value."""
        with pytest.raises(ValueError):
            ProfilingTask("invalid_task")


class TestTableDefinitions:
    """Test TABLES constant definitions."""

    def test_tables_is_dict(self):
        """TABLES should be a dictionary."""
        assert isinstance(TABLES, dict)

    def test_tables_has_required_keys(self):
        """TABLES should have required table keys."""
        required = {
            "IDENTIFICATION",
            "RESUME_DETAILS",
            "CONTACTS",
            "SPECIALIZATION",
            "ATTESTATION"
        }
        assert required.issubset(set(TABLES.keys()))

    def test_each_table_is_list(self):
        """Each table should be a list of column names."""
        for table_name, columns in TABLES.items():
            assert isinstance(columns, list), f"{table_name} should be a list"
            assert len(columns) > 0, f"{table_name} should not be empty"

    def test_each_column_is_string(self):
        """Each column name should be a string."""
        for table_name, columns in TABLES.items():
            for col in columns:
                assert isinstance(col, str), f"Column in {table_name} should be string"
                assert len(col) > 0, f"Column in {table_name} should not be empty"

    def test_identification_table_columns(self):
        """IDENTIFICATION table should have expected columns."""
        cols = TABLES["IDENTIFICATION"]
        expected = {"resume_id", "first_name", "last_name"}
        assert expected.issubset(set(cols))

    def test_resume_details_table_columns(self):
        """RESUME_DETAILS table should have expected columns."""
        cols = TABLES["RESUME_DETAILS"]
        expected = {"post", "salary", "area_name"}
        assert expected.issubset(set(cols))

    def test_contacts_table_columns(self):
        """CONTACTS table should have expected columns."""
        cols = TABLES["CONTACTS"]
        expected = {"email", "cell_phone"}
        assert expected.issubset(set(cols))

    def test_no_duplicate_columns_per_table(self):
        """Each table should not have duplicate columns."""
        for table_name, columns in TABLES.items():
            assert len(columns) == len(set(columns)), \
                f"{table_name} has duplicate columns"

    def test_tables_not_empty(self):
        """TABLES should contain at least one table."""
        assert len(TABLES) > 0


class TestEnumIntegration:
    """Integration tests for field definition enums."""

    def test_all_enums_have_members(self):
        """All enum classes should have at least one member."""
        enums = [FieldType, PrivacyCategory, AnonymizationStrategy, ProfilingTask]
        for enum_class in enums:
            members = list(enum_class)
            assert len(members) > 0, f"{enum_class.__name__} has no members"

    def test_enum_values_are_unique(self):
        """Each enum should have unique values."""
        enums = [FieldType, PrivacyCategory, AnonymizationStrategy, ProfilingTask]
        for enum_class in enums:
            values = [m.value for m in enum_class]
            assert len(values) == len(set(values)), \
                f"{enum_class.__name__} has duplicate values"

    def test_can_compare_enum_members(self):
        """Should be able to compare enum members."""
        ft1 = FieldType.SHORT_TEXT
        ft2 = FieldType.SHORT_TEXT
        ft3 = FieldType.LONG_TEXT

        assert ft1 == ft2
        assert ft1 != ft3

    def test_can_access_enum_by_name(self):
        """Should be able to access enum members by name."""
        assert FieldType["SHORT_TEXT"] == FieldType.SHORT_TEXT
        assert PrivacyCategory["DIRECT_IDENTIFIER"] == PrivacyCategory.DIRECT_IDENTIFIER
        assert AnonymizationStrategy["SUPPRESSION"] == AnonymizationStrategy.SUPPRESSION
        assert ProfilingTask["COMPLETENESS"] == ProfilingTask.COMPLETENESS
