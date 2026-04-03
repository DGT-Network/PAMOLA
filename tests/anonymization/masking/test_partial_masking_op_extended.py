"""
Extended tests for PartialMaskingOperation targeting missed coverage lines.

Focus areas:
- All mask_strategy values: fixed, pattern, random, words
- Position-based masking: unmasked_prefix, unmasked_suffix, unmasked_positions
- Pattern-based masking: pattern_type, mask_pattern, preserve_pattern
- Word-based masking: preserve_word_boundaries=True/False
- Random percentage masking with mask_percentage
- _validate_configuration error paths
- _apply_partial_mask with preset_type/preset_name
- _random_percentage_mask edge cases
- _word_based_mask
- _pattern_based_mask variants
- _position_based_mask with unmasked_positions and separators
- _create_consistency_map
- consistency_fields parameter
- process_batch: REPLACE and ENRICH modes
- process_batch_dask
- execute() with various configs
- Null handling
"""

import pytest
import pandas as pd

from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


# ---------------------------------------------------------------------------
# Helpers
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
        "email": ["alice@example.com", "bob@example.com", "carol@example.com",
                  "dave@example.com", "eve@example.com"],
        "phone": ["555-1234", "555-5678", "555-9999", "555-0000", "555-1111"],
        "notes": ["Hello World", "Short", "A B C D E F G", "Test Value", "Another Note"],
    })


# ---------------------------------------------------------------------------
# 1. mask_strategy=fixed (position-based)
# ---------------------------------------------------------------------------

class TestFixedStrategy:
    def test_default_prefix_suffix_zero(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(field_name="name", use_encryption=False)
        op.output_field_name = "name"
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_prefix_only(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="ssn", unmasked_prefix=3, use_encryption=False
        )
        op.output_field_name = "ssn"
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_suffix_only(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="ssn", unmasked_suffix=4, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_prefix_and_suffix(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="ssn", unmasked_prefix=3, unmasked_suffix=4, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_prefix_suffix_sum_exceeds_length(self):
        """When prefix+suffix >= value length, original value is returned."""
        op = PartialMaskingOperation(
            field_name="name", unmasked_prefix=5, unmasked_suffix=5, use_encryption=False
        )
        op.output_field_name = "name"
        result = op._position_based_mask("Alice")
        assert result == "Alice"

    def test_unmasked_positions(self):
        op = PartialMaskingOperation(
            field_name="name", unmasked_positions=[0, 2, 4], use_encryption=False
        )
        op.output_field_name = "name"
        result = op._position_based_mask("Hello")
        assert result[0] == "H"
        assert result[2] == "l"
        assert result[4] == "o"

    def test_unmasked_positions_default(self):
        """Test default without unmasked_positions."""
        op = PartialMaskingOperation(
            field_name="name", use_encryption=False
        )
        assert hasattr(op, "field_name")

    def test_preserve_separators_true(self):
        op = PartialMaskingOperation(
            field_name="ssn", unmasked_prefix=0, unmasked_suffix=0,
            preserve_separators=True, use_encryption=False
        )
        op.output_field_name = "ssn"
        result = op._position_based_mask("123-45-6789")
        assert "-" in result  # separators preserved

    def test_preserve_separators_false(self):
        op = PartialMaskingOperation(
            field_name="ssn", unmasked_prefix=0, unmasked_suffix=0,
            preserve_separators=False, use_encryption=False
        )
        op.output_field_name = "ssn"
        result = op._position_based_mask("123-45-6789")
        assert isinstance(result, str)

    def test_enrich_mode(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mode="ENRICH",
            output_field_name="name_partial", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_replace_mode(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mode="REPLACE", unmasked_prefix=2, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 2. mask_strategy=random
# ---------------------------------------------------------------------------

class TestRandomStrategy:
    def test_random_strategy_execute(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=50.0, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_random_percentage_100(self):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=100.0, use_encryption=False
        )
        result = op._random_percentage_mask("hello")
        assert len(result) == 5
        assert all(c == "*" for c in result)

    def test_random_percentage_0(self):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=0.0, use_encryption=False
        )
        result = op._random_percentage_mask("hello")
        assert result == "hello"

    def test_random_percentage_empty_value(self):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=50.0, use_encryption=False
        )
        result = op._random_percentage_mask("")
        assert result == ""

    def test_random_percentage_no_percentage(self):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=None, use_encryption=False
        )
        # mask_percentage=None → value returned unchanged
        result = op._random_percentage_mask("hello")
        assert result == "hello"

    def test_random_mask_true(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=50.0, random_mask=True,
            mask_char_pool="ABCDE", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 3. mask_strategy=words
# ---------------------------------------------------------------------------

class TestWordsStrategy:
    def test_word_strategy_no_preserve_boundaries(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="notes", mask_strategy="words",
            preserve_word_boundaries=False, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_word_strategy_with_preserve_boundaries(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="notes", mask_strategy="words",
            preserve_word_boundaries=True, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_word_based_mask_short_words(self):
        op = PartialMaskingOperation(
            field_name="notes", mask_strategy="words",
            preserve_word_boundaries=True, use_encryption=False
        )
        result = op._word_based_mask("Hi A Bob")
        # "Hi" (2) → "**", "A" (1) → "*", "Bob" (3) → "***"
        words = result.split()
        assert words[0] == "**"
        assert words[1] == "*"
        assert words[2] == "***"

    def test_word_based_mask_long_words(self):
        op = PartialMaskingOperation(
            field_name="notes", mask_strategy="words",
            preserve_word_boundaries=True, use_encryption=False
        )
        result = op._word_based_mask("Alexander Elizabeth")
        # Long words go through position-based masking
        assert isinstance(result, str)

    def test_word_based_mask_no_boundaries_falls_back(self):
        op = PartialMaskingOperation(
            field_name="notes", mask_strategy="words",
            preserve_word_boundaries=False, use_encryption=False
        )
        result = op._word_based_mask("Hello World")
        assert isinstance(result, str)

    def test_word_based_mask_empty_string(self):
        op = PartialMaskingOperation(
            field_name="notes", mask_strategy="words",
            preserve_word_boundaries=True, use_encryption=False
        )
        result = op._word_based_mask("")
        assert result == ""


# ---------------------------------------------------------------------------
# 4. mask_strategy=pattern
# ---------------------------------------------------------------------------

class TestPatternStrategy:
    def test_pattern_type_email(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="email", mask_strategy="pattern",
            pattern_type="email", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_pattern_type_ssn(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="ssn", mask_strategy="pattern",
            pattern_type="ssn", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_pattern_type_phone(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="phone", mask_strategy="pattern",
            pattern_type="phone", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_custom_mask_pattern(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="ssn", mask_strategy="pattern",
            mask_pattern=r"\d", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_custom_mask_pattern_direct(self):
        op = PartialMaskingOperation(
            field_name="f", mask_strategy="pattern",
            mask_pattern=r"\d", use_encryption=False
        )
        result = op._pattern_based_mask("abc123")
        assert "1" not in result or result != "abc123"

    def test_preserve_pattern(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="email", mask_strategy="pattern",
            preserve_pattern=r"@[\w.]+", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_preserve_pattern_direct(self):
        op = PartialMaskingOperation(
            field_name="f", mask_strategy="pattern",
            preserve_pattern=r"@\w+", use_encryption=False
        )
        result = op._pattern_based_mask("user@domain")
        assert isinstance(result, str)

    def test_no_pattern_config_returns_original(self):
        op = PartialMaskingOperation(
            field_name="f", mask_strategy="pattern", use_encryption=False
        )
        # No pattern_type, no mask_pattern, no preserve_pattern
        op.mask_pattern = None
        op.preserve_pattern = None
        op._pattern_config = None
        op.pattern_type = None
        result = op._pattern_based_mask("hello")
        assert result == "hello"

    def test_random_mask_in_pattern(self):
        op = PartialMaskingOperation(
            field_name="f", mask_strategy="pattern",
            mask_pattern=r"\d", random_mask=True,
            mask_char_pool="XYZ", use_encryption=False
        )
        result = op._pattern_based_mask("abc123")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. _validate_configuration error paths
# ---------------------------------------------------------------------------

class TestValidateConfiguration:
    def test_invalid_mask_char_empty(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(field_name="name", mask_char="", use_encryption=False)
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_invalid_mask_char_two_chars(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(field_name="name", mask_char="**", use_encryption=False)
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_invalid_mask_strategy(self, base_df, reporter, tmp_path):
        with pytest.raises(Exception):
            PartialMaskingOperation(
                field_name="name", mask_strategy="invalid_strategy", use_encryption=False
            )

    def test_random_strategy_missing_percentage(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=None, use_encryption=False
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_invalid_unmasked_prefix_type(self, base_df, reporter, tmp_path):
        with pytest.raises(Exception):
            PartialMaskingOperation(
                field_name="name", unmasked_prefix=-1, use_encryption=False
            )

    def test_invalid_unmasked_suffix_type(self, base_df, reporter, tmp_path):
        with pytest.raises(Exception):
            PartialMaskingOperation(
                field_name="name", unmasked_suffix=-2, use_encryption=False
            )

    def test_invalid_mask_pattern_regex(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="pattern",
            mask_pattern="[invalid(", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_invalid_preserve_pattern_regex(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="pattern",
            preserve_pattern="[bad(", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_field_not_found(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="nonexistent_field", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR


# ---------------------------------------------------------------------------
# 6. Null handling
# ---------------------------------------------------------------------------

class TestNullHandling:
    def test_null_strategy_preserve(self, reporter, tmp_path):
        df = pd.DataFrame({
            "name": ["Alice", None, "Carol", float("nan"), "Eve"],
            "val": range(5),
        })
        op = PartialMaskingOperation(
            field_name="name", null_strategy="PRESERVE", use_encryption=False
        )
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_apply_partial_mask_with_nan(self):
        op = PartialMaskingOperation(field_name="name", use_encryption=False)
        import math
        result = op._apply_partial_mask(float("nan"))
        assert result is None or (isinstance(result, float) and math.isnan(result))

    def test_apply_partial_mask_with_none(self):
        op = PartialMaskingOperation(field_name="name", use_encryption=False)
        result = op._apply_partial_mask(None)
        assert result is None


# ---------------------------------------------------------------------------
# 7. _apply_partial_mask: case sensitivity
# ---------------------------------------------------------------------------

class TestCaseSensitivity:
    def test_case_insensitive_masking(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", case_sensitive=False, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_case_sensitive_masking(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", case_sensitive=True, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 8. Consistency fields
# ---------------------------------------------------------------------------

class TestConsistencyFields:
    def test_consistency_fields_execute(self, reporter, tmp_path):
        df = pd.DataFrame({
            "first_name": ["Alice", "Alice", "Bob", "Bob", "Carol"],
            "last_name": ["Smith", "Jones", "Smith", "Jones", "Smith"],
            "val": range(5),
        })
        op = PartialMaskingOperation(
            field_name="first_name",
            consistency_fields=["last_name"],
            use_encryption=False
        )
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_create_consistency_map(self):
        df = pd.DataFrame({
            "name": ["Alice", "Alice", "Bob"],
            "alias": ["Al", "Alice", "Bobby"],
        })
        op = PartialMaskingOperation(
            field_name="name",
            consistency_fields=["alias"],
            use_encryption=False
        )
        op.output_field_name = "name"
        cmap = op._create_consistency_map(df)
        # All unique values across both columns should be in the map
        assert isinstance(cmap, dict)
        assert len(cmap) > 0

    def test_create_consistency_map_random(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "alias": ["A", "B", "C"],
        })
        op = PartialMaskingOperation(
            field_name="name",
            consistency_fields=["alias"],
            random_mask=True,
            use_encryption=False
        )
        op.output_field_name = "name"
        cmap = op._create_consistency_map(df)
        assert isinstance(cmap, dict)


# ---------------------------------------------------------------------------
# 9. process_batch direct tests
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_replace_mode(self, base_df):
        op = PartialMaskingOperation(
            field_name="name", mode="REPLACE",
            unmasked_prefix=1, use_encryption=False
        )
        op.output_field_name = "name"
        result = op.process_batch(base_df.copy())
        assert "name" in result.columns

    def test_enrich_mode_new_column(self, base_df):
        op = PartialMaskingOperation(
            field_name="name", mode="ENRICH",
            output_field_name="name_masked", use_encryption=False
        )
        op.output_field_name = "name_masked"
        result = op.process_batch(base_df.copy())
        assert "name_masked" in result.columns

    def test_field_not_found_raises(self, base_df):
        from pamola_core.errors.exceptions import FieldNotFoundError
        op = PartialMaskingOperation(field_name="nonexistent", use_encryption=False)
        op.output_field_name = "nonexistent"
        with pytest.raises(Exception):
            op.process_batch(base_df.copy())

    def test_nan_values_in_batch(self):
        df = pd.DataFrame({
            "name": ["Alice", None, "Carol"],
            "val": range(3),
        })
        op = PartialMaskingOperation(
            field_name="name", mode="REPLACE", use_encryption=False
        )
        op.output_field_name = "name"
        result = op.process_batch(df)
        assert "name" in result.columns

    def test_consistency_fields_in_process_batch(self):
        df = pd.DataFrame({
            "name": ["Alice", "Alice", "Bob"],
            "alias": ["Al", "Alice", "Bobby"],
        })
        op = PartialMaskingOperation(
            field_name="name", mode="REPLACE",
            consistency_fields=["alias"],
            use_encryption=False
        )
        op.output_field_name = "name"
        result = op.process_batch(df)
        assert "masked_alias" in result.columns


# ---------------------------------------------------------------------------
# 10. process_batch_dask
# ---------------------------------------------------------------------------

class TestProcessBatchDask:
    def test_dask_basic(self, base_df):
        import dask.dataframe as dd
        op = PartialMaskingOperation(
            field_name="name", mode="REPLACE",
            unmasked_prefix=1, use_encryption=False
        )
        op.output_field_name = "name"
        ddf = dd.from_pandas(base_df, npartitions=2)
        result = op.process_batch_dask(ddf)
        computed = result.compute()
        assert "name" in computed.columns

    def test_dask_enrich_mode(self, base_df):
        import dask.dataframe as dd
        op = PartialMaskingOperation(
            field_name="name", mode="ENRICH",
            output_field_name="name_m", use_encryption=False
        )
        op.output_field_name = "name_m"
        ddf = dd.from_pandas(base_df, npartitions=2)
        result = op.process_batch_dask(ddf)
        computed = result.compute()
        assert "name_m" in computed.columns


# ---------------------------------------------------------------------------
# 11. Varied mask_char options
# ---------------------------------------------------------------------------

class TestMaskCharVariants:
    def test_hash_mask_char(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", mask_char="#", unmasked_prefix=1, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_x_mask_char(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="ssn", mask_char="X", unmasked_suffix=4, use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_random_mask_with_pool(self, base_df, reporter, tmp_path):
        op = PartialMaskingOperation(
            field_name="name", random_mask=True,
            mask_char_pool="0123456789", use_encryption=False
        )
        result = op.execute(make_ds(base_df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 12. Large data
# ---------------------------------------------------------------------------

class TestLargeData:
    def test_large_df(self, reporter, tmp_path):
        n = 500
        df = pd.DataFrame({
            "ssn": [f"{i:03d}-{i%99:02d}-{i*3:04d}" for i in range(n)],
            "val": range(n),
        })
        op = PartialMaskingOperation(
            field_name="ssn", unmasked_prefix=3, unmasked_suffix=4,
            use_encryption=False
        )
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_single_row_df(self, reporter, tmp_path):
        df = pd.DataFrame({"name": ["Alice"], "val": [1]})
        op = PartialMaskingOperation(
            field_name="name", unmasked_prefix=1, use_encryption=False
        )
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
