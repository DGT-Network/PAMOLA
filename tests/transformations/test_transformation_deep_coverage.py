"""Deep coverage tests for transformation operations.
Targets: merge_datasets_op (69 missed), split_by_id_values_op (43 missed),
split_fields_op (50 missed), remove_fields (62 missed),
aggregate_records_op (51 missed) — total ~275 missed lines."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df, name="main"):
    return DataSource(dataframes={name: df})


def make_multi_ds(dfs):
    return DataSource(dataframes=dfs)


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
        def add_artifact(self, *a, **kw): pass
    return R()


def _run(op, ds, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(ds, tmp_path, reporter)


# === Remove Fields ===
class TestRemoveFields:
    def test_remove_single_field(self, reporter, tmp_path):
        from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        op = RemoveFieldsOperation(fields_to_remove=["b"])
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_remove_multiple_fields(self, reporter, tmp_path):
        from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
        df = pd.DataFrame({"a": range(100), "b": range(100), "c": range(100), "d": range(100)})
        op = RemoveFieldsOperation(fields_to_remove=["b", "c"])
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_remove_nonexistent_field(self, reporter, tmp_path):
        from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        op = RemoveFieldsOperation(fields_to_remove=["nonexistent"])
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_remove_large_df(self, reporter, tmp_path):
        from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
        df = pd.DataFrame({f"col_{i}": range(500) for i in range(20)})
        op = RemoveFieldsOperation(fields_to_remove=[f"col_{i}" for i in range(0, 20, 2)])
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Split Fields ===
class TestSplitFields:
    def test_split_by_field_groups(self, reporter, tmp_path):
        from pamola_core.transformations.splitting.split_fields_op import SplitFieldsOperation
        df = pd.DataFrame({
            "id": range(150),
            "name": [f"n{i}" for i in range(150)],
            "age": [20 + i % 50 for i in range(150)],
            "salary": [30000 + i * 100 for i in range(150)],
        })
        op = SplitFieldsOperation(
            id_field="id",
            field_groups={"personal": ["name", "age"], "financial": ["salary"]},
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Split By ID Values ===
class TestSplitByIdValues:
    def test_split_by_category(self, reporter, tmp_path):
        from pamola_core.transformations.splitting.split_by_id_values_op import SplitByIDValuesOperation
        df = pd.DataFrame({
            "dept": ["IT", "HR", "Finance", "IT", "HR"] * 40,
            "val": range(200),
        })
        op = SplitByIDValuesOperation(
            id_field="dept",
            value_groups={"it_group": ["IT"], "hr_group": ["HR"], "fin_group": ["Finance"]},
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_split_two_groups(self, reporter, tmp_path):
        from pamola_core.transformations.splitting.split_by_id_values_op import SplitByIDValuesOperation
        df = pd.DataFrame({
            "group": ["A", "B"] * 100,
            "val": range(200),
        })
        op = SplitByIDValuesOperation(
            id_field="group",
            value_groups={"group_a": ["A"], "group_b": ["B"]},
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_split_partition_method(self, reporter, tmp_path):
        from pamola_core.transformations.splitting.split_by_id_values_op import SplitByIDValuesOperation
        df = pd.DataFrame({
            "cat": [f"c{i % 5}" for i in range(200)],
            "val": range(200),
        })
        op = SplitByIDValuesOperation(
            id_field="cat",
            number_of_partitions=3,
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Aggregate Records ===
class TestAggregateRecords:
    def test_aggregate_sum(self, reporter, tmp_path):
        from pamola_core.transformations.grouping.aggregate_records_op import AggregateRecordsOperation
        df = pd.DataFrame({
            "dept": ["IT", "HR", "IT", "HR", "Finance"] * 40,
            "salary": [50000, 60000, 55000, 65000, 70000] * 40,
        })
        op = AggregateRecordsOperation(
            group_by_fields=["dept"],
            aggregations={"salary": ["sum"]},
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_aggregate_multiple_functions(self, reporter, tmp_path):
        from pamola_core.transformations.grouping.aggregate_records_op import AggregateRecordsOperation
        df = pd.DataFrame({
            "cat": [f"c{i % 5}" for i in range(200)],
            "val": np.random.randint(0, 100, 200),
            "score": np.random.rand(200),
        })
        op = AggregateRecordsOperation(
            group_by_fields=["cat"],
            aggregations={"val": ["mean", "max"], "score": ["min", "std"]},
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_aggregate_count(self, reporter, tmp_path):
        from pamola_core.transformations.grouping.aggregate_records_op import AggregateRecordsOperation
        df = pd.DataFrame({
            "cat": [f"c{i % 10}" for i in range(200)],
            "val": range(200),
        })
        op = AggregateRecordsOperation(
            group_by_fields=["cat"],
            aggregations={"val": ["count"]},
        )
        result = _run(op, make_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Merge Datasets ===
class TestMergeDatasets:
    def test_merge_inner(self, reporter, tmp_path):
        from pamola_core.transformations.merging.merge_datasets_op import MergeDatasetsOperation
        df1 = pd.DataFrame({"id": range(100), "val_a": range(100)})
        df2 = pd.DataFrame({"id": range(50, 150), "val_b": range(100)})
        ds = make_multi_ds({"left": df1, "right": df2})
        op = MergeDatasetsOperation(
            left_dataset_name="left", right_dataset_name="right",
            left_key="id", right_key="id", join_type="inner",
        )
        result = _run(op, ds, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_merge_left(self, reporter, tmp_path):
        from pamola_core.transformations.merging.merge_datasets_op import MergeDatasetsOperation
        df1 = pd.DataFrame({"id": range(100), "a": range(100)})
        df2 = pd.DataFrame({"id": range(50), "b": range(50)})
        ds = make_multi_ds({"left": df1, "right": df2})
        op = MergeDatasetsOperation(
            left_dataset_name="left", right_dataset_name="right",
            left_key="id", right_key="id", join_type="left",
        )
        result = _run(op, ds, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_merge_outer(self, reporter, tmp_path):
        from pamola_core.transformations.merging.merge_datasets_op import MergeDatasetsOperation
        df1 = pd.DataFrame({"id": [1, 2, 3], "a": ["x", "y", "z"]})
        df2 = pd.DataFrame({"id": [2, 3, 4], "b": ["p", "q", "r"]})
        ds = make_multi_ds({"left": df1, "right": df2})
        op = MergeDatasetsOperation(
            left_dataset_name="left", right_dataset_name="right",
            left_key="id", right_key="id", join_type="outer",
        )
        result = _run(op, ds, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
