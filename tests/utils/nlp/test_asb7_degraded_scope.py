"""ASB-7: the degraded-NLP-resource registry is scoped to one operation run, and every operation
reports it in its result."""

import pandas as pd

from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.utils.nlp import remote_resources as rr
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _ds():
    return DataSource(dataframes={"main": pd.DataFrame({"name": ["a", "b", "c", "a"]})})


class _NeedsModel(FullMaskingOperation):
    """A masking operation that, mid-run, fails to obtain an NLP resource."""

    def execute(self, *a, **k):
        rr.record_degraded("spacy:probe_model", "not installed locally and downloads are off")
        return super().execute(*a, **k)


def test_asb7_any_operation_reports_a_resource_it_could_not_obtain(tmp_path):
    out = _NeedsModel(field_name="name", generate_visualization=False).execute(_ds(), tmp_path / "t", _Reporter())
    assert out.status == OperationStatus.SUCCESS, out.error_message
    assert out.metrics["nlp_resources_degraded"] == {
        "spacy:probe_model": "not installed locally and downloads are off"}


def test_asb7_the_registry_does_not_leak_from_one_run_into_the_next(tmp_path):
    rr.record_degraded("fasttext:stale", "from an earlier operation")
    out = CategoricalOperation(field_name="name").execute(_ds(), tmp_path / "t", _Reporter())
    assert out.status == OperationStatus.SUCCESS, out.error_message
    assert "nlp_resources_degraded" not in out.metrics  # the stale entry was cleared at the start
    assert rr.get_degraded_resources() == {}


def test_asb7_a_clean_run_has_no_metric_and_a_nested_execute_reports_once(tmp_path):
    out = FullMaskingOperation(field_name="name", generate_visualization=False).execute(_ds(), tmp_path / "t", _Reporter())
    assert "nlp_resources_degraded" not in out.metrics
    class _Spy(_NeedsModel):
        pass

    res = _Spy(field_name="name", generate_visualization=False).execute(_ds(), tmp_path / "u", _Reporter())
    assert list(res.metrics["nlp_resources_degraded"]) == ["spacy:probe_model"]
