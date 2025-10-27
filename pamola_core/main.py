import os
from pathlib import Path
from pamola_core.anonymization.generalization.categorical_op import (
    CategoricalGeneralizationOperation,
)
from pamola_core.anonymization.generalization.datetime_op import (
    DateTimeGeneralizationOperation,
)
from pamola_core.anonymization.generalization.numeric_op import (
    NumericGeneralizationOperation,
)
from pamola_core.io.csv import read_csv
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.tasks.task_reporting import TaskReporter


def validate_and_read(file_path: str, file_description: str, **kwargs):
    """Validates the existence and format of a file, then loads it as a DataFrame."""
    if not file_path:
        # Raise an exception if the file_path is not provided
        raise "File path is required but was not provided."

    # Supported file extensions
    file_handlers = {
        ".csv": read_csv,
    }

    # Get the file extension in lowercase
    ext = file_path.suffix.lower()

    # Validate and read the file
    if ext not in file_handlers:
        raise f"{file_description} is not a supported file type. Supported types are: {', '.join(file_handlers.keys())}."

    try:
        # Get the appropriate function for reading the file
        read_function = file_handlers.get(ext)

        # Full file read
        dataframe = read_function(file_path, **kwargs)

    except Exception as e:
        raise (f"Failed to load {file_description}. Error: {str(e)}")
    return dataframe, file_path.stem, ext


def create_base_config_kwargs():
    return {
        "generate_visualization": True,
        "save_output": True,
        "force_recalculation": False,
        "dataset_name": "partial_dataset",
    }


# -------------------------------------------------------------------------
# 🚀 Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    task_dir = Path("D:/AIShowRoom/DGT-Network/PAMOLA/pamola_datasets/result_datasets/")
    os.makedirs(task_dir, exist_ok=True)
    task_reporter = TaskReporter(
        "123", "test_operation", "test operation", report_path=task_dir
    )
    tracker = HierarchicalProgressTracker(
        total=6,
        description="Processing data",
        unit="steps",
        track_memory=True,
        level=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    # use test_datetime_data.csv for datetime generalization testing and categorical_sample_data.csv for categorical generalization testing
    file_path = Path("pamola_datasets/test_numeric_generalization.csv")
    file_description = "Original Dataset"
    df, _, _ = validate_and_read(file_path, file_description)
    kwargs = create_base_config_kwargs()

    data_source = DataSource(
        dataframes={"partial_dataset": df},
    )

    # Numeric Generalization Operation
    start_op = NumericGeneralizationOperation(
        field_name="salary",
        strategy="binning",
        precision=2,
        bin_count=3,
        binning_method="equal_width",
        mode="ENRICH",
        output_field_name="salary_generalized",
        quasi_identifiers=["age", "gender"],
        null_strategy="PRESERVE",
        output_format="csv",
        use_cache=False,
        use_encryption=False,
        use_dask=False,
        npartitions=4,
        use_vectorization=False,
        parallel_processes=4,
        visualization_backend="plotly",
    )

    # # DateTime Generalization Operation
    # start_op = DateTimeGeneralizationOperation(
    #     field_name="timestamp",
    #     strategy="rounding",
    #     rounding_unit="month",
    #     output_field_name="timestamp_generalized",
    #     mode="ENRICH",
    #     output_format="csv",
    #     use_cache=False,
    #     use_encryption=False,
    #     use_dask=False,
    #     npartitions=4,
    #     use_vectorization=False,
    #     parallel_processes=4,
    #     visualization_backend="plotly",
    # )

    # # Categorical Generalization Operation
    # start_op = CategoricalGeneralizationOperation(
    #     field_name="category",
    #     strategy="hierarchy",
    #     hierarchy_level=2,  # Generalize to level 2: e.g., 'Fruit > Apple'
    #     external_dictionary_path="D:/AIShowRoom/DGT-Network/PAMOLA/pamola_datasets/category_hierarchy_mapping.csv",
    #     merge_low_freq=True,
    #     min_group_size=3,
    #     allow_unknown=True,
    #     group_rare_as="OTHER",
    #     risk_threshold=4.0,
    #     ka_risk_field="risk_score",
    #     condition_field="region",
    #     condition_values=["North"],
    #     condition_operator="in",
    #     quasi_identifiers=["gender"],
    #     output_field_name="category_generalized",
    #     vulnerable_record_strategy="suppress",
    #     case_sensitive=True,
    #     mode="ENRICH",
    #     output_format="csv",
    #     use_cache=False,
    #     use_encryption=False,
    #     use_dask=False,
    #     npartitions=4,
    #     use_vectorization=False,
    #     parallel_processes=2,
    #     visualization_backend="plotly",
    # )

    result = start_op.execute(
        data_source=data_source,
        task_dir=task_dir,
        reporter=task_reporter,
        progress_tracker=tracker,
        **kwargs,
    )
