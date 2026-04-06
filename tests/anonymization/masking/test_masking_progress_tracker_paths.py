"""Tests targeting progress_tracker code paths in masking operations.
Covers: full_masking_op lines 274-287, 291-292, 323-353, 357-358, 367-368, 393-400
        partial_masking_op lines 328-341, 345-346, 376-406
These lines are only exercised when progress_tracker is provided."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
        def add_artifact(self, *a, **kw): pass
    return R()


@pytest.fixture
def progress_tracker():
    pt = MagicMock()
    pt.total = 0
    pt.update = MagicMock()
    pt.create_sub_tracker = MagicMock(return_value=MagicMock())
    return pt


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"] * 40,
        "ssn": ["123-45-6789", "987-65-4321", "555-55-5555", "111-22-3333", "444-55-6666"] * 40,
        "salary": [50000, 60000, 70000, 80000, 90000] * 40,
        "email": ["a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co"] * 40,
    })


# --- Full Masking with progress_tracker ---
class TestFullMaskingWithProgress:
    def test_string_field_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = FullMaskingOperation(field_name="name", use_encryption=False)
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS
        assert progress_tracker.update.called

    def test_numeric_field_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = FullMaskingOperation(field_name="salary", use_encryption=False)
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_mask_char_custom_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = FullMaskingOperation(field_name="name", mask_char="#", use_encryption=False)
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_preserve_length_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = FullMaskingOperation(field_name="name", preserve_length=True, use_encryption=False)
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_fixed_length_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = FullMaskingOperation(field_name="name", fixed_length=5, use_encryption=False)
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_ssn_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = FullMaskingOperation(field_name="ssn", use_encryption=False)
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS


# --- Partial Masking with progress_tracker ---
class TestPartialMaskingWithProgress:
    def test_fixed_strategy_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="fixed",
            unmasked_prefix=1, unmasked_suffix=1, use_encryption=False,
        )
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS
        assert progress_tracker.update.called

    def test_pattern_email_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = PartialMaskingOperation(
            field_name="email", mask_strategy="pattern",
            pattern_type="email", use_encryption=False,
        )
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_random_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=50, use_encryption=False,
        )
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_prefix_only_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="fixed",
            unmasked_prefix=2, use_encryption=False,
        )
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_suffix_only_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="fixed",
            unmasked_suffix=2, use_encryption=False,
        )
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS

    def test_ssn_pattern_with_tracker(self, sample_df, reporter, tmp_path, progress_tracker):
        op = PartialMaskingOperation(
            field_name="ssn", mask_strategy="pattern",
            pattern_type="ssn", use_encryption=False,
        )
        op.preset_type = None
        op.preset_name = None
        result = op.execute(make_ds(sample_df), tmp_path, reporter, progress_tracker=progress_tracker)
        assert result.status == OperationStatus.SUCCESS
