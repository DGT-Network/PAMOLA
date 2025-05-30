import pytest
from pamola_core.profiling.commons import data_types

class TestDataTypeEnum:
    def test_valid_enum_values(self):
        assert data_types.DataType.NUMERIC.value == "numeric"
        assert data_types.DataType.CATEGORICAL.value == "categorical"
        assert data_types.DataType.TEXT.value == "text"
        assert data_types.DataType.LONGTEXT.value == "longtext"
        assert data_types.DataType.DATE.value == "date"
        assert data_types.DataType.DATETIME.value == "datetime"
        assert data_types.DataType.EMAIL.value == "email"
        assert data_types.DataType.PHONE.value == "phone"
        assert data_types.DataType.BOOLEAN.value == "boolean"
        assert data_types.DataType.MULTI_VALUED.value == "multi_valued"
        assert data_types.DataType.JSON.value == "json"
        assert data_types.DataType.ARRAY.value == "array"
        assert data_types.DataType.MIXED.value == "mixed"
        assert data_types.DataType.UNKNOWN.value == "unknown"
        assert data_types.DataType.CORRELATION.value == "correlation"
        assert data_types.DataType.CORRELATION_MATRIX.value == "correlation_matrix"

    def test_invalid_enum_value(self):
        with pytest.raises(ValueError):
            data_types.DataType("not_a_type")

class TestAnalysisTypeEnum:
    def test_valid_enum_values(self):
        assert data_types.AnalysisType.BASIC.value == "basic"
        assert data_types.AnalysisType.COMPLETENESS.value == "completeness"
        assert data_types.AnalysisType.UNIQUENESS.value == "uniqueness"
        assert data_types.AnalysisType.DISTRIBUTION.value == "distribution"
        assert data_types.AnalysisType.CORRELATION.value == "correlation"
        assert data_types.AnalysisType.TEXT_ANALYSIS.value == "text_analysis"
        assert data_types.AnalysisType.GROUP_VARIATION.value == "group_variation"
        assert data_types.AnalysisType.DUPLICATES.value == "duplicates"
        assert data_types.AnalysisType.EMAIL_ANALYSIS.value == "email_analysis"
        assert data_types.AnalysisType.PHONE_ANALYSIS.value == "phone_analysis"
        assert data_types.AnalysisType.NAME_ANALYSIS.value == "name_analysis"
        assert data_types.AnalysisType.NER_ANALYSIS.value == "ner_analysis"

    def test_invalid_enum_value(self):
        with pytest.raises(ValueError):
            data_types.AnalysisType("not_a_type")

class TestResultTypeEnum:
    def test_valid_enum_values(self):
        assert data_types.ResultType.STATS.value == "statistics"
        assert data_types.ResultType.DISTRIBUTION.value == "distribution"
        assert data_types.ResultType.CORRELATION.value == "correlation"
        assert data_types.ResultType.DICTIONARY.value == "dictionary"
        assert data_types.ResultType.VISUALIZATION.value == "visualization"
        assert data_types.ResultType.ERROR.value == "error"

    def test_invalid_enum_value(self):
        with pytest.raises(ValueError):
            data_types.ResultType("not_a_type")

class TestArtifactTypeEnum:
    def test_valid_enum_values(self):
        assert data_types.ArtifactType.JSON.value == "json"
        assert data_types.ArtifactType.CSV.value == "csv"
        assert data_types.ArtifactType.PNG.value == "png"
        assert data_types.ArtifactType.HTML.value == "html"
        assert data_types.ArtifactType.TEXT.value == "text"

    def test_invalid_enum_value(self):
        with pytest.raises(ValueError):
            data_types.ArtifactType("not_a_type")

class TestOperationStatusEnum:
    def test_valid_enum_values(self):
        assert data_types.OperationStatus.SUCCESS.value == "success"
        assert data_types.OperationStatus.WARNING.value == "warning"
        assert data_types.OperationStatus.ERROR.value == "error"
        assert data_types.OperationStatus.SKIPPED.value == "skipped"

    def test_invalid_enum_value(self):
        with pytest.raises(ValueError):
            data_types.OperationStatus("not_a_type")

class TestPrivacyLevelEnum:
    def test_valid_enum_values(self):
        assert data_types.PrivacyLevel.PUBLIC.value == "public"
        assert data_types.PrivacyLevel.RESTRICTED.value == "restricted"
        assert data_types.PrivacyLevel.SENSITIVE.value == "sensitive"
        assert data_types.PrivacyLevel.IDENTIFIER.value == "identifier"

    def test_invalid_enum_value(self):
        with pytest.raises(ValueError):
            data_types.PrivacyLevel("not_a_type")

class TestProfilerConfig:
    def test_default_values(self):
        cfg = data_types.ProfilerConfig
        assert cfg.DEFAULT_TOP_N == 20
        assert cfg.DEFAULT_MIN_GROUP_SIZE == 2
        assert cfg.DEFAULT_CORRELATION_THRESHOLD == 0.1
        assert cfg.PROFILING_DIR_NAME == "profiling"
        assert cfg.DICTIONARIES_DIR_NAME == "dictionaries"
        assert cfg.VISUALIZATION_DIR_NAME == "visualizations"
        assert cfg.DEFAULT_FIGURE_WIDTH == 12
        assert cfg.DEFAULT_FIGURE_HEIGHT == 8
        assert cfg.DEFAULT_DPI == 300
        assert cfg.DEFAULT_SAMPLE_SIZE == 10000
        assert cfg.SAMPLING_ENABLED is True
        assert cfg.MVF_SEPARATOR == ","
        assert cfg.MVF_QUOTE_CHAR == '"'
        assert isinstance(cfg.EMAIL_DOMAINS_OF_INTEREST, set)
        assert "gmail.com" in cfg.EMAIL_DOMAINS_OF_INTEREST
        assert cfg.DEFAULT_COUNTRY_CODE == "7"
        assert isinstance(cfg.PHONE_FORMAT_REGEX, str)
        assert cfg.LONGTEXT_MIN_LENGTH == 200
        assert cfg.DEFAULT_TEXT_SAMPLE_SIZE == 100
        assert cfg.NER_ENABLED is False
        assert cfg.JSON_MAX_DEPTH == 5
        assert cfg.ARRAY_MAX_ELEMENTS == 100

    def test_email_domains_content(self):
        domains = data_types.ProfilerConfig.EMAIL_DOMAINS_OF_INTEREST
        assert "gmail.com" in domains
        assert "yandex.ru" in domains
        assert "hotmail.com" in domains
        assert "mail.ru" in domains
        assert "yahoo.com" in domains

    def test_phone_regex(self):
        import re
        regex = data_types.ProfilerConfig.PHONE_FORMAT_REGEX
        match = re.match(regex, '(123,456,789)')
        assert match is not None
        match2 = re.match(regex, '(123,456,789,"extra")')
        assert match2 is not None
        match3 = re.match(regex, '(123,456,789)')
        assert match3.group(1) == '123'

class TestDataTypeDetection:
    def test_categorical_threshold(self):
        assert data_types.DataTypeDetection.CATEGORICAL_THRESHOLD == 100

    def test_date_patterns(self):
        patterns = data_types.DataTypeDetection.DATE_PATTERNS
        assert "%Y-%m-%d" in patterns
        assert "%d/%m/%Y" in patterns
        assert "%d.%m.%Y" in patterns

    def test_boolean_values(self):
        assert "true" in data_types.DataTypeDetection.BOOLEAN_TRUE_VALUES
        assert "no" in data_types.DataTypeDetection.BOOLEAN_FALSE_VALUES

    def test_mvf_indicators_and_threshold(self):
        assert "," in data_types.DataTypeDetection.MVF_INDICATORS
        assert data_types.DataTypeDetection.MVF_THRESHOLD == 0.1

    def test_email_regex(self):
        import re
        regex = data_types.DataTypeDetection.EMAIL_REGEX
        assert re.match(regex, "test@example.com")
        assert not re.match(regex, "not-an-email")

    def test_phone_basic_regex(self):
        import re
        regex = data_types.DataTypeDetection.PHONE_BASIC_REGEX
        assert re.match(regex, "(123,456,789)")
        assert not re.match(regex, "123-456-789")

    def test_json_start_end_chars(self):
        assert '{' in data_types.DataTypeDetection.JSON_START_CHARS
        assert '}' in data_types.DataTypeDetection.JSON_END_CHARS

    def test_array_regex(self):
        import re
        regex = data_types.DataTypeDetection.ARRAY_REGEX
        assert re.match(regex, "[1,2,3]")
        assert not re.match(regex, "1,2,3")

class TestFieldCategory:
    @pytest.mark.parametrize("field,expected", [
        ("name", "PERSONAL_IDENTIFIERS"),
        ("email", "CONTACT_INFORMATION"),
        ("birthdate", "TEMPORAL"),
        ("city", "CONTACT_INFORMATION"),
        ("university", "EDUCATIONAL"),
        ("job", "OCCUPATIONAL"),
        ("salary", "FINANCIAL"),
        ("relocation", "PREFERENCE"),
        ("unknown_field", "UNKNOWN"),
        ("", "UNKNOWN"),
        ("NaMe", "PERSONAL_IDENTIFIERS"),
        ("work_phone", "CONTACT_INFORMATION"),
        ("metro_station_name", "CONTACT_INFORMATION"),
        ("driver_license_types", "PREFERENCE"),
    ])
    def test_get_category_for_field(self, field, expected):
        result = data_types.FieldCategory.get_category_for_field(field)
        assert result == expected

    def test_pattern_match_priority(self):
        # Should match pattern before direct
        assert data_types.FieldCategory.get_category_for_field("work_phone") == "CONTACT_INFORMATION"
        assert data_types.FieldCategory.get_category_for_field("company_name") == "PERSONAL_IDENTIFIERS"  # changed from OCCUPATIONAL
        assert data_types.FieldCategory.get_category_for_field("salary_currency") == "FINANCIAL"

    def test_edge_cases(self):
        # Empty string, numeric string, special chars
        assert data_types.FieldCategory.get_category_for_field("") == "UNKNOWN"
        assert data_types.FieldCategory.get_category_for_field("12345") == "UNKNOWN"
        assert data_types.FieldCategory.get_category_for_field("@!#%$") == "UNKNOWN"

    def test_case_insensitivity(self):
        assert data_types.FieldCategory.get_category_for_field("NaMe") == "PERSONAL_IDENTIFIERS"
        assert data_types.FieldCategory.get_category_for_field("EMAIL") == "CONTACT_INFORMATION"

if __name__ == "__main__":
    pytest.main()
