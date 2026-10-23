"""ASB-4: the category mapping artifact holds source categories by design (a generalization DATA
artifact) - only in the task's mappings/ directory, never in metrics, results or logs."""

import json
import logging

import pandas as pd

from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

CANARY = "ZQCATEGORY"


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def test_asb4_category_mapping_stays_in_mappings_dir_and_out_of_metrics_and_logs(tmp_path, caplog):
    cats = [f"{CANARY}{i}" for i in range(6)]
    df = pd.DataFrame({"category": [c for c in cats for _ in range(5)] + [f"{CANARY}rare"]})
    op = CategoricalGeneralizationOperation(
        field_name="category", strategy="merge_low_freq", min_group_size=6, generate_visualization=False,
        use_cache=False)
    with caplog.at_level(logging.DEBUG):
        out = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert out.status == OperationStatus.SUCCESS, out.error_message
    files = [p for p in (tmp_path / "t").rglob("*") if p.is_file()]
    holders = sorted({p.parent.name for p in files if CANARY.encode() in p.read_bytes()})
    assert "mappings" in holders or not [p for p in files if p.parent.name == "mappings"]
    for name in holders:  # source categories only in the mapping artifact and the data output
        assert name in ("mappings", "output"), holders
    assert CANARY not in json.dumps(out.metrics, default=str)
    assert CANARY not in "\n".join(r.getMessage() for r in caplog.records)
