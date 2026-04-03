"""
Extended tests for FullMaskingOperation targeting missed coverage lines.

Focus areas:
- preserve_length=True / False / fixed_length
- random_mask with and without mask_char_pool
- preserve_format=True with format_patterns
- numeric_output: "string", "numeric", "preserve"
- date_format parameter
- _mask_value edge cases: NaN, int, float, string
- _mask_to_numeric: integer, float, scientific notation
- _mask_with_format and _reconstruct_format
- _is_string_field
- _can_vectorize
- _vectorized_mask
- process_batch: various field types
- process_batch_dask
- _validate_configuration error paths
- get_operation_summary
- _get_cache_parameters
- _collect_specific_metrics
"""

import pytest
import pandas as pd
import numpy as np

from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
    return R()


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "ssn": ["123-45-6789", "987-65-4321", "555-55-5555", "111-22-3333", "444-55-6666"],
        "email": ["a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co"],
        "salary": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "score": [1.5, 2.5, 3.5, 4.5, 5.5],
        "date": ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01"],
    })


@pytest.fixture
def numeric_df():
    return pd.DataFrame({
        "value": [100, 200, 300, 400, 500],
        "price": [9.99, 19.99, 29.99, 39.99, 49.99],
    })


# ---------------------------------------------------------------------------
# 1. Basic masking variants
# ---------------------------------------------------------------------------

class TestBasicMasking:
    def test_default_mask_char(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_custom_mask_char(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", mask_char="#", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_preserve_length_true(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", preserve_length=True, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_preserve_length_false(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", preserve_length=False, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_fixed_length(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", fixed_length=5, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_random_mask_no_pool(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", random_mask=True, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_random_mask_with_pool(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="name", random_mask=True,
            mask_char_pool="ABCDEF", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_mode(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="name", mode="ENRICH",
            output_field_name="name_masked", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 2. Numeric field masking
# ---------------------------------------------------------------------------

class TestNumericMasking:
    def test_numeric_field_string_output(self, numeric_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="value", numeric_output="string", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(numeric_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_numeric_field_numeric_output(self, numeric_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="value", numeric_output="numeric", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(numeric_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_numeric_field_preserve_output(self, numeric_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="value", numeric_output="preserve", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(numeric_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_float_field_numeric_output(self, numeric_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="price", numeric_output="numeric", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(numeric_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 3. Date format masking
# ---------------------------------------------------------------------------

class TestDateFormatMasking:
    def test_date_format_specified(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="date", date_format="%Y-%m-%d", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_date_format_invalid_field(self, base_df, reporter, tmp_path):
        """date_format on a non-date field should fall back gracefully."""
        op = FullMaskingOperation(
            field_name="name", date_format="%Y-%m-%d", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 4. Format preservation
# ---------------------------------------------------------------------------

class TestFormatPreservation:
    def test_preserve_format_ssn(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="ssn", preserve_format=True, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_preserve_format_email(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="email", preserve_format=True, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_preserve_format_with_custom_patterns(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="ssn",
            preserve_format=True,
            format_patterns={"ssn": r"(\d{3})-(\d{2})-(\d{4})"},
            use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 5. _mask_value unit tests
# ---------------------------------------------------------------------------

class TestMaskValue:
    def setup_method(self):
        self.op = FullMaskingOperation(field_name="f", use_encryption=False)
        self.op.preset_type = None
        self.op.preset_name = None

    def test_nan_preserved(self):
        self.op.null_strategy = "PRESERVE"
        result = self.op._mask_value(float("nan"))
        assert pd.isna(result)

    def test_string_value(self):
        self.op.preserve_length = True
        self.op.mask_char = "*"
        self.op.fixed_length = None
        self.op.random_mask = False
        self.op.numeric_output = "string"
        result = self.op._mask_value("hello")
        assert result == "*****"

    def test_fixed_length_mask(self):
        self.op.fixed_length = 4
        self.op.preserve_length = False
        self.op.mask_char = "#"
        self.op.random_mask = False
        self.op.numeric_output = "string"
        result = self.op._mask_value("hello")
        assert result == "####"

    def test_default_length_8(self):
        self.op.fixed_length = None
        self.op.preserve_length = False
        self.op.mask_char = "*"
        self.op.random_mask = False
        self.op.numeric_output = "string"
        result = self.op._mask_value("abc")
        assert result == "********"

    def test_random_mask_with_pool(self):
        self.op.random_mask = True
        self.op.mask_char_pool = "ABC"
        self.op.preserve_length = True
        self.op.fixed_length = None
        self.op.numeric_output = "string"
        result = self.op._mask_value("hello")
        assert len(result) == 5
        assert all(c in "ABC" for c in result)

    def test_random_mask_without_pool(self):
        self.op.random_mask = True
        self.op.mask_char_pool = None
        self.op.preserve_length = True
        self.op.fixed_length = None
        self.op.numeric_output = "string"
        result = self.op._mask_value("hello")
        assert len(result) == 5

    def test_integer_numeric_output(self):
        self.op.numeric_output = "numeric"
        self.op.preserve_length = True
        self.op.fixed_length = None
        self.op.random_mask = False
        self.op.mask_char = "*"
        result = self.op._mask_value(1234)
        assert isinstance(result, (int, float))

    def test_float_numeric_output(self):
        self.op.numeric_output = "numeric"
        self.op.preserve_length = True
        self.op.fixed_length = None
        self.op.random_mask = False
        self.op.mask_char = "*"
        result = self.op._mask_value(12.34)
        assert isinstance(result, (int, float))

    def test_numeric_preserve_returns_original(self):
        self.op.numeric_output = "preserve"
        self.op.preserve_length = True
        self.op.fixed_length = None
        self.op.random_mask = False
        result = self.op._mask_value(999)
        assert result == 999


# ---------------------------------------------------------------------------
# 6. _mask_to_numeric unit tests
# ---------------------------------------------------------------------------

class TestMaskToNumeric:
    def setup_method(self):
        self.op = FullMaskingOperation(field_name="f", use_encryption=False)
        self.op.preset_type = None
        self.op.preset_name = None
        self.op.random_mask = False
        self.op.mask_char = "*"

    def test_integer_input(self):
        result = self.op._mask_to_numeric("****", "1234")
        assert isinstance(result, int)

    def test_float_input(self):
        result = self.op._mask_to_numeric("*****", "12.34")
        assert isinstance(result, float)

    def test_scientific_notation(self):
        result = self.op._mask_to_numeric("*****", "1.2e+04")
        assert isinstance(result, float)

    def test_random_mask_to_numeric(self):
        self.op.random_mask = True
        result = self.op._mask_to_numeric("****", "1234")
        assert isinstance(result, (int, float))

    def test_digit_mask_char(self):
        self.op.mask_char = "9"
        result = self.op._mask_to_numeric("9999", "1234")
        assert isinstance(result, int)

    def test_short_digit_str(self):
        result = self.op._mask_to_numeric("*", "1.23456")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 7. _mask_with_format
# ---------------------------------------------------------------------------

class TestMaskWithFormat:
    def setup_method(self):
        self.op = FullMaskingOperation(
            field_name="ssn",
            preserve_format=True,
            format_patterns={"ssn": r"(\d{3})-(\d{2})-(\d{4})"},
            use_encryption=False
        )
        self.op.preset_type = None
        self.op.preset_name = None

    def test_matching_pattern(self):
        result = self.op._mask_with_format("123-45-6789")
        assert "-" in result  # structure preserved
        assert "1" not in result or result != "123-45-6789"

    def test_non_matching_falls_back_to_mask_value(self):
        result = self.op._mask_with_format("no_match_here")
        assert isinstance(result, str)

    def test_nan_returns_nan(self):
        result = self.op._mask_with_format(float("nan"))
        assert pd.isna(result)

    def test_no_groups_in_pattern(self):
        op = FullMaskingOperation(
            field_name="f",
            preserve_format=True,
            format_patterns={"plain": r"\d+"},
            use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op._mask_with_format("12345")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 8. _is_string_field
# ---------------------------------------------------------------------------

class TestIsStringField:
    def setup_method(self):
        self.op = FullMaskingOperation(field_name="f", use_encryption=False)
        self.op.preset_type = None; self.op.preset_name = None

    def test_string_series(self):
        series = pd.Series(["a", "b", "c"])
        assert self.op._is_string_field(series) is True

    def test_numeric_series(self):
        series = pd.Series([1, 2, 3])
        assert self.op._is_string_field(series) is False

    def test_mixed_object_with_non_strings(self):
        series = pd.Series([1, "a", 2.0])
        assert self.op._is_string_field(series) is False

    def test_empty_object_series(self):
        series = pd.Series([], dtype=object)
        # Should not crash
        result = self.op._is_string_field(series)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 9. _can_vectorize
# ---------------------------------------------------------------------------

class TestCanVectorize:
    def test_can_vectorize_simple(self):
        op = FullMaskingOperation(
            field_name="f", preserve_format=False, random_mask=False,
            numeric_output="string", date_format=None, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        assert op._can_vectorize() is True

    def test_cannot_vectorize_preserve_format(self):
        op = FullMaskingOperation(field_name="f", preserve_format=True, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        assert op._can_vectorize() is False

    def test_cannot_vectorize_random_mask(self):
        op = FullMaskingOperation(field_name="f", random_mask=True, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        assert op._can_vectorize() is False

    def test_cannot_vectorize_date_format(self):
        op = FullMaskingOperation(
            field_name="f", date_format="%Y-%m-%d", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        assert op._can_vectorize() is False

    def test_cannot_vectorize_numeric_output(self):
        op = FullMaskingOperation(field_name="f", numeric_output="numeric", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        assert op._can_vectorize() is False


# ---------------------------------------------------------------------------
# 10. _vectorized_mask
# ---------------------------------------------------------------------------

class TestVectorizedMask:
    def test_fixed_length_vectorized(self):
        op = FullMaskingOperation(field_name="f", fixed_length=4, use_encryption=False)
        op.preset_type = None; op.preset_name = None
        series = pd.Series(["Alice", "Bob", "Carol"])
        result = op._vectorized_mask(series)
        assert all(v == "****" for v in result)

    def test_preserve_length_vectorized(self):
        op = FullMaskingOperation(
            field_name="f", preserve_length=True, fixed_length=None, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        series = pd.Series(["Alice", "Bob", "Carol"])
        result = op._vectorized_mask(series)
        assert result.tolist() == ["*****", "***", "*****"]

    def test_default_length_8_vectorized(self):
        op = FullMaskingOperation(
            field_name="f", preserve_length=False, fixed_length=None, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        series = pd.Series(["a", "b", "c"])
        result = op._vectorized_mask(series)
        assert all(v == "********" for v in result)


# ---------------------------------------------------------------------------
# 11. _validate_configuration error paths
# ---------------------------------------------------------------------------

class TestValidateConfiguration:
    def test_invalid_mask_char_empty(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", mask_char="", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_invalid_mask_char_two_chars(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(field_name="name", mask_char="**", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_invalid_fixed_length_negative(self, base_df, reporter, tmp_path):
        with pytest.raises(Exception):  # jsonschema.ValidationError or ConfigurationError
            FullMaskingOperation(field_name="name", fixed_length=-1, use_encryption=False)

    def test_invalid_numeric_output(self, base_df, reporter, tmp_path):
        with pytest.raises(Exception):  # jsonschema.ValidationError or ConfigurationError
            FullMaskingOperation(field_name="name", numeric_output="bad_value", use_encryption=False)

    def test_invalid_format_pattern_regex(self, base_df, reporter, tmp_path):
        op = FullMaskingOperation(
            field_name="name",
            format_patterns={"bad": "[invalid_regex("},
            use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR


# ---------------------------------------------------------------------------
# 12. process_batch direct tests
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_string_field_replace_mode(self, base_df):
        op = FullMaskingOperation(field_name="name", mode="REPLACE", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "name"
        result = op.process_batch(base_df.copy())
        assert "name" in result.columns

    def test_string_field_enrich_mode(self, base_df):
        op = FullMaskingOperation(
            field_name="name", mode="ENRICH",
            output_field_name="name_masked", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "name_masked"
        result = op.process_batch(base_df.copy())
        assert "name_masked" in result.columns

    def test_numeric_field(self, numeric_df):
        op = FullMaskingOperation(field_name="value", mode="REPLACE", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "value"
        result = op.process_batch(numeric_df.copy())
        assert "value" in result.columns

    def test_with_date_format(self, base_df):
        op = FullMaskingOperation(
            field_name="date", date_format="%Y-%m-%d",
            mode="REPLACE", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "date"
        result = op.process_batch(base_df.copy())
        assert "date" in result.columns

    def test_preserve_format_string(self, base_df):
        op = FullMaskingOperation(
            field_name="ssn", preserve_format=True,
            mode="REPLACE", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "ssn"
        result = op.process_batch(base_df.copy())
        assert "ssn" in result.columns

    def test_field_not_found_raises(self, base_df):
        from pamola_core.errors.exceptions import FieldNotFoundError
        op = FullMaskingOperation(field_name="nonexistent", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "nonexistent"
        with pytest.raises(Exception):
            op.process_batch(base_df.copy())


# ---------------------------------------------------------------------------
# 13. process_batch_dask
# ---------------------------------------------------------------------------

class TestProcessBatchDask:
    def test_dask_string_field(self, base_df):
        import dask.dataframe as dd
        op = FullMaskingOperation(field_name="name", mode="REPLACE", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "name"
        ddf = dd.from_pandas(base_df, npartitions=2)
        result = op.process_batch_dask(ddf)
        computed = result.compute()
        assert "name" in computed.columns

    def test_dask_numeric_field(self, numeric_df):
        import dask.dataframe as dd
        op = FullMaskingOperation(field_name="value", mode="REPLACE", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        op.output_field_name = "value"
        ddf = dd.from_pandas(numeric_df, npartitions=2)
        result = op.process_batch_dask(ddf)
        computed = result.compute()
        assert "value" in computed.columns


# ---------------------------------------------------------------------------
# 14. get_operation_summary and _get_cache_parameters
# ---------------------------------------------------------------------------

class TestSummaryAndCacheParams:
    def test_get_operation_summary_all_keys(self):
        op = FullMaskingOperation(
            field_name="name", mask_char="#", preserve_length=True,
            random_mask=False, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        summary = op.get_operation_summary()
        expected_keys = [
            "field_name", "mask_character", "preserve_length",
            "fixed_length", "random_mask", "mask_char_pool",
            "preserve_format", "format_patterns", "numeric_output", "date_format"
        ]
        for key in expected_keys:
            assert key in summary

    def test_get_cache_parameters(self):
        op = FullMaskingOperation(field_name="name", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        params = op._get_cache_parameters()
        assert "mask_char" in params
        assert "preserve_length" in params
        assert "numeric_output" in params


# ---------------------------------------------------------------------------
# 15. Large data / batch processing paths
# ---------------------------------------------------------------------------

class TestLargeData:
    def test_large_df_execute(self, reporter, tmp_path):
        n = 1000
        df = pd.DataFrame({
            "name": [f"Person_{i}" for i in range(n)],
            "val": range(n),
        })
        op = FullMaskingOperation(field_name="name", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_df_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({
            "name": ["Alice", None, "Carol", float("nan"), "Eve"],
            "val": range(5),
        })
        op = FullMaskingOperation(
            field_name="name", null_strategy="PRESERVE", use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mixed_types_field(self, reporter, tmp_path):
        df = pd.DataFrame({
            "mixed": [1, "hello", 3.14, None, "world"],
            "val": range(5),
        })
        op = FullMaskingOperation(field_name="mixed", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 16. Special character fields
# ---------------------------------------------------------------------------

class TestSpecialCharFields:
    def test_field_with_special_chars(self, reporter, tmp_path):
        df = pd.DataFrame({
            "data": ["hello@world.com", "test#123", "foo$bar", "100%", "a&b"],
            "val": range(5),
        })
        op = FullMaskingOperation(field_name="data", use_encryption=False)
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_empty_string_field(self, reporter, tmp_path):
        df = pd.DataFrame({
            "data": ["", "abc", "", "xy", ""],
            "val": range(5),
        })
        op = FullMaskingOperation(
            field_name="data", preserve_length=True, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
