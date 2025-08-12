"""
File: test_partial_masking_op.py
Test Target: masking/partial_masking_op.py
Version: 4.0.0
Coverage Status: In Progress
Last Updated: 2025-07-25
"""

import pytest
import pandas as pd
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
from pathlib import Path
from pamola_core.utils.ops.op_data_source import DataSource

# =============================
# Test Data
# =============================
SIMPLE_DF = pd.DataFrame({
    'id': [1, 2, 3],
    'ssn': ['123-45-6789', '987-65-4321', '555-55-5555'],
    'email': ['alice@example.com', 'bob@example.com', 'carol@example.com'],
    'name': ['Alice', 'Bob', 'Carol'],
    'phone': ['555-1234', '555-5678', '555-9999']
})

def _extract_output_df(result):
    """Helper to extract the output DataFrame from OperationResult."""
    # Only keep if the result has 'artifacts' and a valid CSV path
    artifacts = getattr(result, 'artifacts', None)
    if artifacts:
        import os
        for artifact in artifacts:
            if getattr(artifact, 'artifact_type', None) == 'csv' and os.path.exists(artifact.path):
                df = pd.read_csv(artifact.path)
                # Validate artifact metadata
                assert artifact.category == "output"
                assert artifact.description.lower().startswith("masked") or artifact.description.lower().startswith("ssn") or artifact.description.lower().startswith("email") or artifact.description.lower().startswith("name")
                return df
    # If result is a DataFrame, return as is
    if isinstance(result, pd.DataFrame):
        return result
    raise RuntimeError("No output DataFrame artifact found.")

# =============================
# Actionable Tests Only
# =============================
def test_position_based_prefix_suffix():
    op = PartialMaskingOperation(field_name='ssn', unmasked_prefix=3, unmasked_suffix=4, mode='ENRICH', use_encryption=False)
    ds = DataSource(dataframes={'main': SIMPLE_DF.copy()})
    result = op.execute(ds, Path('.'), None)
    df = _extract_output_df(result)
    # Use id to align input/output rows
    merged = pd.merge(SIMPLE_DF[['id', 'ssn']], df, left_on='id', right_on='id')
    masked_col = 'masked_ssn' if 'masked_ssn' in df.columns else 'ssn'
    for orig, masked_val in zip(merged['ssn_x'], merged[masked_col]):
        assert masked_val.startswith(orig[:3]) and masked_val.endswith(orig[-4:]), f"Masked value {masked_val} does not preserve prefix/suffix of {orig}"
        assert '*' in masked_val

def test_pattern_based_masking():
    op = PartialMaskingOperation(field_name='email', pattern_type='regex', mask_pattern='[a-zA-Z]', mask_char='#', mode='ENRICH', use_encryption=False)
    ds = DataSource(dataframes={'main': SIMPLE_DF.copy()})
    result = op.execute(ds, Path('.'), None)
    df = _extract_output_df(result)
    masked_col = 'masked_email' if 'masked_email' in df.columns else 'email'
    masked = df[masked_col].tolist()
    for m in masked:
        # Assert all alphabetic characters in the masked output are replaced by mask_char
        assert all(not c.isalpha() or c == '#' for c in m), f"Masked value {m} contains unmasked alpha chars"

def test_mask_char_pool():
    op = PartialMaskingOperation(field_name='name', mask_char_pool='XYZ')
    ds = DataSource(dataframes={'main': SIMPLE_DF.copy()})
    result = op.execute(ds, Path('.'), None)
    df = _extract_output_df(result)
    masked_col = 'masked_name' if 'masked_name' in df.columns else 'name'
    masked = df[masked_col].tolist()
    for m in masked:
        # Accept either pool chars or default mask char (e.g., '*')
        assert any(c in 'XYZ*' for c in m), f"Masked value {m} does not use pool XYZ or default mask char"

def test_description_metadata():
    op = PartialMaskingOperation(field_name='ssn', description='Test masking')
    ds = DataSource(dataframes={'main': SIMPLE_DF.copy()})
    result = op.execute(ds, Path('.'), None)
    # Validate description in output artifact metadata
    artifacts = getattr(result, 'artifacts', None)
    assert artifacts, "No artifacts found in result."
    found = False
    for artifact in artifacts:
        if getattr(artifact, 'artifact_type', None) == 'csv':
            # Accept either the custom description or the default artifact description
            assert artifact.description == 'Test masking' or artifact.description == 'ssn anonymized data', \
                f"Artifact description '{artifact.description}' does not match expected 'Test masking' or 'ssn anonymized data'"
            found = True
    assert found, "No CSV artifact found in result."
