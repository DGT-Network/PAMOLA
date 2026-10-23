"""CDX-R2-7b: a failed chart of a pseudonymization run is reported in the result.

Before: the handler logged a warning and the result carried no trace of the failure.
"""

import pandas as pd
import pytest

from pamola_core.anonymization.pseudonymization.hash_based_op import HashBasedPseudonymizationOperation
from pamola_core.anonymization.pseudonymization.mapping_op import ConsistentMappingPseudonymizationOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

_MAKERS = {
    "hash": lambda: HashBasedPseudonymizationOperation(
        field_name="name", use_cache=False, generate_visualization=True, visualization_backend="matplotlib"
    ),
    "mapping": lambda: ConsistentMappingPseudonymizationOperation(
        field_name="name", use_cache=False, generate_visualization=True, visualization_backend="matplotlib",
        mapping_encryption_key="ab" * 32,
    ),
}


@pytest.mark.parametrize("case", sorted(_MAKERS))
def test_visualization_error_appears_in_the_result(case, tmp_path, monkeypatch):
    op = _MAKERS[case]()

    def boom(self, *args, **kwargs):
        raise RuntimeError("R2_RENDERER_FAILED")

    monkeypatch.setattr(type(op), "_handle_visualizations", boom)
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol", "Dave"] * 5})
    result = op.execute(DataSource(dataframes={"main": df}), tmp_path, None)

    assert "R2_RENDERER_FAILED" in result.metrics.get("visualization_error", ""), result.metrics
    # the pseudonymized data itself is unaffected
    assert result.status == OperationStatus.SUCCESS, result.error_message
