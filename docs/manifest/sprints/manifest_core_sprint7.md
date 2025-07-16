# PR Manifest – Sprint 7 (05/12/2025 - 05/30/2025)

## Target Branch
`mvp_epic2` ← `mvp_sprint7_epic2`

---

## Key points
- The transformation package provides a set of operations for modifying datasets to prepare them for anonymization, synthetic generation, and privacy analysis. The operations under the transformation package are listed below.
- Encryption is applied to intermediate artifacts and task results, with integrating focused on the `fake_data`, `profiling`, `tranformation` packages.
- Apply `Dask` and `JobLib` for large dataset and parallel processing. This affects in the `fake_data`, `profiling`, `tranformation` packages. Their flags are controlled via Task configuration file. 
- Integrate Caching and progress tracking for the `fake_data`, `profiling`, `tranformation` packages. Currently, the `HierarchicalProgressTracker` class is used for progress tracking.
- Apply Visualization context for thread safety, not only to avoid inconsistent outputs or failures, but also to improve performance during concurrent execution. This affects in the `fake_data`, `profiling`, `tranformation` packages.

---

## Performance options

### Dask
- `use_dask`: Whether to use Dask for large dataset processing
- `chunk_size`: Size of data chunks for processing
- `npartition`: number of jobs

### JobLib
- `use_vectorization`: Whether to use vectorized (parallel) processing
- `parallel_processes`: Number of parallel processes to use
- `chunk_size`: Size of data chunks for processing

### Chunk
- `chunk_size`: Size of data chunks for processing

### Acknowledgement
- Based on the specific logical inner process per operation, some of them can not be applied for JobLib and/or Dask that leads incurrate calculation. Therefore, the parallel processing is forcibly ignored for those operations. The table below details the operations for which parallel processing is marked as 'No'.

---

## Remaining tasks
- Complete encryption and performance optimization with applying the libraries of Dask and JobLib for the remaining operations of the `fake_data`, `profiling`, `tranformation` packages are listed in the table below.
- Bug fixing for integration tests.
- Update Unit tests and documentations.

---

## New package: Transformation

```
pamola_core/
└── transformations/
    ├── base_transformation_op.py
    ├── __init__.py
    ├── cleaning/
    │   ├── clean_invalid_values.py
    │   └── __init__.py
    ├── commons/
    │   ├── aggregation_utils.py
    │   ├── merging_utils.py
    │   ├── metric_utils.py
    │   ├── processing_utils.py
    │   ├── validation_utils.py
    │   ├── visualization_utils.py
    │   └── __init__.py
    ├── field_ops/
    │   ├── add_modify_fields.py
    │   ├── remove_fields.py
    │   └── __init__.py
    ├── grouping/
    │   ├── aggregate_records_op.py
    │   └── __init__.py
    ├── imputation/
    │   ├── impute_missing_values.py
    │   └── __init__.py
    ├── merging/
    │   ├── merge_datasets_op.py
    │   └── __init__.py
    └── splitting/
        ├── split_by_id_values_op.py
        ├── split_fields_op.py
        └── __init__.py
```

---

## Functionalities
|Packages            |Operations            |Encryption |Dask       |JobLib     |Visualization context|Cache      |Tracker Progress|Reporter|Notes                                                                                                                                                                                                                                                                                                                                          |
|--------------------|----------------------|-----------|-----------|-----------|-------------|-----------|----------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|core/profiling/     |Anonymity             |Yes        |No         |No         |In-Progress  |Yes        |Yes             |Yes     |Reason: Dask chunking and joblist cannot be applied by splitting the dataframe because k-anonymity and attribute profiling require global statistics across the entire dataset. To enable Dask or parallel processing, each calculation must operate on the full dataframe, not on separate chunks, and results should be aggregated afterward.|
|core/profiling/     |Attribute             |Yes        |No         |No         |In-Progress  |Yes        |Yes             |Yes     |Reason: Dask chunking and joblist cannot be applied by splitting the dataframe because k-anonymity and attribute profiling require global statistics across the entire dataset. To enable Dask or parallel processing, each calculation must operate on the full dataframe, not on separate chunks, and results should be aggregated afterward.|
|core/profiling/     |Categorical           |Yes        |No         |No         |In-Progress  |In-Progress|No              |Yes     |Parallelizing categorical by splitting rows is not feasible because the function requires the entire column to compute accurate statistics such as value_counts, entropy, and top_n. Row-wise chunking would result in incorrect results since partial calculations from chunks cannot be simply merged to match the global column statistics. |
|core/profiling/     |Correlation           |Yes        |No         |No         |In-Progress  |In-Progress|No              |Yes     |Parallelizing correlation analysis by splitting rows using Dask or joblib (joblist) is not feasible because accurate correlation statistics—such as value_counts, contingency tables, and correlation coefficients—require access to the entire columns or all relevant data for each field.                                                   |
|core/profiling/     |Date                  |Yes        |No         |In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Email                 |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Group                 |Yes        |No         |No         |In-Progress  |In-Progress|No              |Yes     |Parallelizing group-based analysis by splitting rows or partitions is not feasible because group metrics such as variance and duplication require access to all records of each group. If groups are fragmented across partitions, the calculated statistics will be incorrect                                                                 |
|core/profiling/     |Identity              |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Mvf                   |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Numeric               |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Phone                 |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Text                  |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/profiling/     |Currency              |Yes        |In-Progress|In-Progress|In-Progress  |In-Progress|No              |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/fake_data/     |Email                 |Yes        |Yes        |Yes        |Yes          |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/fake_data/     |Name                  |Yes        |Yes        |Yes        |Yes          |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/fake_data/     |Organization          |Yes        |Yes        |Yes        |Yes          |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/fake_data/     |Phone                 |Yes        |Yes        |Yes        |Yes          |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|split_fields          |Yes        |In-Progress|In-Progress|In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|split_by_id_values    |Yes        |In-Progress|In-Progress|In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|clean_invalid_values  |Yes        |Yes        |Yes        |In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|impute_missing_values |Yes        |Yes        |Yes        |In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|aggregate_records     |Yes        |In-Progress|No         |In-Progress  |Yes        |Yes             |Yes     |Only using dask (Using joblib requires manually dividing the data into chunks and later concatenating the results, which may lead to inaccuracies in the final processed data.)                                                                                                                                                                |
|core/transformation/|merge_datasets        |Yes        |In-Progress|No         |In-Progress  |Yes        |Yes             |Yes     |Only using dask (Using joblib requires manually dividing the data into chunks and later concatenating the results, which may lead to inaccuracies in the final processed data.)                                                                                                                                                                |
|core/transformation/|remove_fields         |Yes        |Yes        |Yes        |In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|add_modify_fields     |Yes        |Yes        |Yes        |In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/transformation/|base_transformation_op|Yes        |In-Progress|In-Progress|In-Progress  |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/anonymization  |numeric               |In-Progress|Yes        |Yes        |Yes          |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|core/anonymization  |base_anonymization_op |In-Progress|Yes        |Yes        |Yes          |Yes        |Yes             |Yes     |                                                                                                                                                                                                                                                                                                                                               |
|                    |                      |           |           |           |             |           |                |        |                                                                                                                                                                                                                                                                                                                                               |

---

## Improvement for consistency (if needed)
- Handle the parameter `include_timestamp` consistently across all packages. Currently, handling is inconsistent among packages/operations, and it's also missing from the `fake_data` package.
- Handle the parameter `output_format` across all packages. It is currently required for the `transformation` package but is not mentioned or updated for other packages.

---

## End of the sprint 7

---

# PREVIOUS CHANGES

---

# Previous Manifest – Sprint 7 - Week 1 (05/12/2025 - 05/23/2025)

## Key points

- The transformation package provides a set of operations for modifying datasets to prepare them for anonymization, synthetic generation, and privacy analysis.
- Encryption is applied to intermediate artifacts and task results, with initial integrating focused on the fake_data package.
- Dask is used for parallel, distributed processing, with initial implementation focused on the fake_data package.

---

## Implementations changed

### Added
- pamola/pamola_core/common/enum/relationship_type.py
- pamola/pamola_core/common/helpers/custom_aggregations_helper.py
- pamola/pamola_core/transformations/__init__.py
- pamola/pamola_core/transformations/base_transformation_op.py
- pamola/pamola_core/transformations/cleaning/clean_invalid_values.py
- pamola/pamola_core/transformations/cleaning__old_02_05/__init__.py
- pamola/pamola_core/transformations/commons/__init__.py
- pamola/pamola_core/transformations/commons/aggregation_utils.py
- pamola/pamola_core/transformations/commons/merging_utils.py
- pamola/pamola_core/transformations/commons/metric_utils.py
- pamola/pamola_core/transformations/commons/processing_utils.py
- pamola/pamola_core/transformations/commons/validation_utils.py
- pamola/pamola_core/transformations/commons/visualization_utils.py
- pamola/pamola_core/transformations/field_ops/__init__.py
- pamola/pamola_core/transformations/field_ops/add_modify_fields.py
- pamola/pamola_core/transformations/field_ops/remove_fields.py
- pamola/pamola_core/transformations/grouping/__init__.py
- pamola/pamola_core/transformations/grouping/aggregate_records_op.py
- pamola/pamola_core/transformations/imputation/__init__.py
- pamola/pamola_core/transformations/imputation/impute_missing_values.py
- pamola/pamola_core/transformations/merging/__init__.py
- pamola/pamola_core/transformations/merging/merge_datasets_op.py
- pamola/pamola_core/transformations/splitting/__init__.py
- pamola/pamola_core/transformations/splitting/split_by_id_values_op.py
- pamola/pamola_core/transformations/splitting/split_fields_op.py
- pamola/pamola_core/utils/vis_helpers/venn_diagram.py

#### Modified
- pamola/pamola_core/anonymization/base_anonymization_op.py
- pamola/pamola_core/anonymization/commons/visualization_utils.py
- pamola/pamola_core/anonymization/generalization/numeric.py
- pamola/pamola_core/common/constants.py
- pamola/pamola_core/fake_data/commons/operations.py
- pamola/pamola_core/fake_data/operations/email_op.py
- pamola/pamola_core/fake_data/operations/name_op.py
- pamola/pamola_core/fake_data/operations/organization_op.py
- pamola/pamola_core/fake_data/operations/phone_op.py
- pamola/pamola_core/profiling/analyzers/anonymity.py
- pamola/pamola_core/profiling/analyzers/attribute.py
- pamola/pamola_core/profiling/analyzers/categorical.py
- pamola/pamola_core/profiling/analyzers/correlation.py
- pamola/pamola_core/profiling/analyzers/currency.py
- pamola/pamola_core/profiling/analyzers/date.py
- pamola/pamola_core/profiling/analyzers/email.py
- pamola/pamola_core/profiling/analyzers/group.py
- pamola/pamola_core/profiling/analyzers/identity.py
- pamola/pamola_core/profiling/analyzers/mvf.py
- pamola/pamola_core/profiling/analyzers/numeric.py
- pamola/pamola_core/profiling/analyzers/phone.py
- pamola/pamola_core/profiling/analyzers/text.py
- pamola/pamola_core/profiling/commons/anonymity_utils.py
- pamola/pamola_core/transformations/cleaning/__init__.py
- pamola/pamola_core/utils/io.py
- pamola/pamola_core/utils/ops/op_data_writer.py
- pamola/pamola_core/utils/tasks/base_task.py
- pamola/pamola_core/utils/vis_helpers/__init__.py
- pamola/pamola_core/utils/visualization.py

---

## Public API Changes

### Added

#### Transformation
- `MergeDatasetsOperation`
- `ImputeMissingValuesOperation`
- `AddOrModifyFieldsOperation`
- `RemoveFieldsOperation`
- `SplitByIDValuesOperation`
- `SplitFieldsOperation`
- `AggregateRecordsOperation`  
- `CleanInvalidValuesOperation` 

### Changed
- ...

### Deprecated
- None

---

## Bug fixing
- None


## Unit tests

### Test Coverage Summary

| Module         | Coverage (%) |
| -------------- | ------------ |
| Tasks utils    | 94           |
| Nlp utils      | 94           |
| profiling      | 99           |
| fake_data      | 99           |
| transformation | 97           |
| **Total**      | 96.6         |
 
#### fake_data
coverage run -m pytest tests/fake_data
coverage report --include="tests/fake_data/**/*.py"

#### profiling
coverage run -m pytest tests/profiling
coverage report --include="tests/profiling/**/*.py"

#### nlp
coverage run -m pytest tests/utils/nlp
coverage report --include="tests/utils/nlp/**/*.py"

#### transformations
coverage run -m pytest tests/transformations
coverage report --include="tests/transformations/**/*.py"

### Added
- pamola/tests/profiling/analyzers/test_anonymity.py
- pamola/tests/profiling/analyzers/test_categorical.py
- pamola/tests/profiling/analyzers/test_correlation.py
- pamola/tests/profiling/analyzers/test_currency.py
- pamola/tests/profiling/analyzers/test_group.py
- pamola/tests/profiling/analyzers/test_mvf.py
- pamola/tests/profiling/analyzers/test_numeric.py
- pamola/tests/profiling/analyzers/test_phone.py
- pamola/tests/profiling/analyzers/test_text.py
- pamola/tests/profiling/commons/test_anonymity_utils.py
- pamola/tests/profiling/commons/test_attribute_utils.py
- pamola/tests/profiling/commons/test_base.py
- pamola/tests/profiling/commons/test_categorical_utils.py
- pamola/tests/profiling/commons/test_correlation_utils.py
- pamola/tests/profiling/commons/test_currency_analysis.py
- pamola/tests/profiling/commons/test_currency_utils.py
- pamola/tests/profiling/commons/test_data_processing.py
- pamola/tests/profiling/commons/test_data_types.py
- pamola/tests/profiling/commons/test_date_utils.py
- pamola/tests/profiling/commons/test_dtype_helpers.py
- pamola/tests/profiling/commons/test_email_utils.py
- pamola/tests/profiling/commons/test_group_utils.py
- pamola/tests/profiling/commons/test_helpers.py
- pamola/tests/profiling/commons/test_identity_utils.py
- pamola/tests/profiling/commons/test_mvf_utils.py
- pamola/tests/profiling/commons/test_numeric_utils.py
- pamola/tests/profiling/commons/test_phone_utils.py
- pamola/tests/profiling/commons/test_statistical_analysis.py
- pamola/tests/profiling/commons/test_text_utils.py
- pamola/tests/profiling/commons/test_visualization_utils.py
- pamola/tests/transformations/cleaning/test_clean_invalid_values.py
- pamola/tests/transformations/commons/test_aggregation_utils.py
- pamola/tests/transformations/commons/test_merging_utils.py
- pamola/tests/transformations/commons/test_metric_utils.py
- pamola/tests/transformations/commons/test_processing_utils.py
- pamola/tests/transformations/commons/test_validation_utils.py
- pamola/tests/transformations/commons/test_visualization_utils.py
- pamola/tests/transformations/field_ops/test_add_modify_fields.py
- pamola/tests/transformations/field_ops/test_remove_fields.py
- pamola/tests/transformations/grouping/test_aggregate_records_op.py
- pamola/tests/transformations/imputation/test_impute_missing_values.py
- pamola/tests/transformations/merging/test_merge_datasets_op.py
- pamola/tests/transformations/splitting/test_split_by_id_values_op.py
- pamola/tests/transformations/splitting/test_split_fields_op.py
- pamola/tests/transformations/test_base_transformation_op.py
- pamola/tests/utils/tasks/test_context_manager.py
- pamola/tests/utils/tasks/test_dependency_manager.py
- pamola/tests/utils/tasks/test_directory_manager.py
- pamola/tests/utils/tasks/test_encryption_manager.py
- pamola/tests/utils/tasks/test_operation_executor.py
- pamola/tests/utils/tasks/test_path_security.py
- pamola/tests/utils/tasks/test_progress_manager.py
- pamola/tests/utils/tasks/test_project_config_loader.py

### Modified
- None

---

## Documentations

### Summary
- Tasks utils (DONE)
- NLP utils (DONE)
- Transformation package (DONE)

### Added
- pamola/docs/en/core/transformations/base_transformation_op.md
- pamola/docs/en/core/transformations/cleaning/clean_invalid_values.md
- pamola/docs/en/core/transformations/commons/aggregation_utils.md
- pamola/docs/en/core/transformations/commons/merging_utils.md
- pamola/docs/en/core/transformations/commons/metric_utils.md
- pamola/docs/en/core/transformations/commons/processing_utils.md
- pamola/docs/en/core/transformations/commons/validation_utils.md
- pamola/docs/en/core/transformations/commons/visualization_utils.md
- pamola/docs/en/core/transformations/field_ops/add_modify_fields.md
- pamola/docs/en/core/transformations/field_ops/remove_fields.md
- pamola/docs/en/core/transformations/grouping/aggregate_records_op.md
- pamola/docs/en/core/transformations/imputation/impute_missing_values.md
- pamola/docs/en/core/transformations/merging/merge_datasets_op.md
- pamola/docs/en/core/transformations/splitting/split_by_id_values_op.md
- pamola/docs/en/core/transformations/splitting/split_fields_op.md
- pamola/docs/en/core/utils/nlp/base.md
- pamola/docs/en/core/utils/nlp/cache.md
- pamola/docs/en/core/utils/nlp/category_matching.md
- pamola/docs/en/core/utils/nlp/clustering.md
- pamola/docs/en/core/utils/nlp/compatibility.md
- pamola/docs/en/core/utils/nlp/entity/base.md
- pamola/docs/en/core/utils/nlp/entity/dictionary.md
- pamola/docs/en/core/utils/nlp/entity/job.md
- pamola/docs/en/core/utils/nlp/entity/organization.md
- pamola/docs/en/core/utils/nlp/entity/skill.md
- pamola/docs/en/core/utils/nlp/entity/transaction.md
- pamola/docs/en/core/utils/nlp/entity_extraction.md
- pamola/docs/en/core/utils/nlp/model_manager.md
- pamola/docs/en/core/utils/nlp/tokenization_ext.md
- pamola/docs/en/core/utils/nlp/tokenization_helpers.md
- pamola/docs/sprints/manifest_s7.md
- pamola/docs/sprints/mr_manifest_per_sprint.md

### Modified
- None

---

## Remaining tasks (Week 2)
- Integrate encryption across all operations in the implemented packages, including `fake_data`, `profiling`, and `transformation`.
- Apply `Dask` and `JobLib` for large dataset and parallel processing.
- Migrate code of Visualization utils from HHR and adopt code changes as needed.
- Integrate other packages if delivered.

---
