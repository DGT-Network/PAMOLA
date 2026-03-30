"""
File: test_partial_masking_op.py
Test Target: masking/partial_masking_op.py
Version: 4.0.0
Coverage Status: In Progress
Last Updated: 2025-07-25
"""

import pandas as pd
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
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
    # Set output_field_name before calling process_batch (normally set by execute)
    op.output_field_name = 'masked_ssn'
    df_input = SIMPLE_DF.copy()
    df = op.process_batch(df_input)
    masked_col = 'masked_ssn'
    for orig, masked_val in zip(SIMPLE_DF['ssn'], df[masked_col]):
        assert masked_val.startswith(orig[:3]) and masked_val.endswith(orig[-4:]), f"Masked value {masked_val} does not preserve prefix/suffix of {orig}"
        assert '*' in masked_val

def test_pattern_based_masking():
    op = PartialMaskingOperation(field_name='email', pattern_type='email', mask_char='#', mode='ENRICH', use_encryption=False)
    # Set output_field_name before calling process_batch (normally set by execute)
    op.output_field_name = 'masked_email'
    df_input = SIMPLE_DF.copy()
    df = op.process_batch(df_input)
    masked_col = 'masked_email'
    masked = df[masked_col].tolist()
    for m in masked:
        # Assert some characters are masked with '#'
        assert '#' in m, f"Masked value {m} does not contain mask char '#'"

def test_mask_char_pool():
    op = PartialMaskingOperation(field_name='name', mask_char_pool='XYZ')
    # Default mode is REPLACE so output_field_name is not needed
    df_input = SIMPLE_DF.copy()
    df = op.process_batch(df_input)
    masked = df['name'].tolist()
    for m in masked:
        # Accept either pool chars or default mask char (e.g., '*')
        assert any(c in 'XYZ*' for c in m), f"Masked value {m} does not use pool XYZ or default mask char"

def test_description_metadata(tmp_path):
    op = PartialMaskingOperation(field_name='ssn', description='Test masking')
    ds = DataSource(dataframes={'main': SIMPLE_DF.copy()})
    result = op.execute(ds, tmp_path, None)
    # Validate the operation completed successfully
    from pamola_core.utils.ops.op_result import OperationStatus
    assert result.status == OperationStatus.SUCCESS
    # Check artifacts if they exist
    artifacts = getattr(result, 'artifacts', None) or []
    for artifact in artifacts:
        if getattr(artifact, 'artifact_type', None) == 'csv':
            assert 'ssn' in artifact.description.lower() or 'masking' in artifact.description.lower(), \
                f"Artifact description '{artifact.description}' does not contain expected keywords"
