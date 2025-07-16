# PR Manifest – Sprint 8 (02/06/2025 - 13/06/2025)

## Key points
- Encryption integration: fake_data, profiling, transformation packages
- Parallel processing with Dask and JobLib: fake_data, profiling, transformation packages
- Cache and progress tracking: fake_data, profiling, transformation packages
- Thread-safe visualization context: fake_data, profiling, transformation packages
- Enhancement log to file for Task execution, including operation log

---

## Summary of Changes

| Category | Package | MRs | Changes |
|----------|---------|-----|---------|
| Package & Integration | fake_data | [MR-902, MR-963, MR-907, MR-948] | - Applied encryption across operations<br>- Applied parallel processing using Dask and JobLib |
| | profiling | [MR-931, MR-963, MR-929, MR-962, MR-948, MR-1031, MR-1035] | - Applied encryption across operations<br>- Applied parallel processing using Dask and JobLib |
| | transformation | [MR-902, MR-905, MR-924, MR-948] | - Applied encryption across operations<br>- Applied parallel processing using Dask and JobLib |
| | anonymization | [MR-1027] | - Applied encryption across operations |
| Bug Fixing | transformation | [MR-910, MR-980, MR-1008, MR-1038, MR-1039] | - Fixed issues in transformation operations<br>- Fixed import encryption mode |
| | fake_data | [MR-941, MR-961, MR-967, MR-1038, MR-1039] | - Fixed bugs in operations<br>- Fixed cache application issues<br>- Fixed import encryption mode |
| | profiling | [MR-941, MR-980, MR-1008, MR-1038, MR-1039] | - Fixed bugs in operations<br>- Fixed import encryption mode |
| Logging Improvements | transformation | [MR-954, MR-962, MR-948] | - Enhanced logging to file instead of console-only output<br>- Added logging for operations |
| | profiling | [MR-962, MR-948] | - Added logging for operations |
| | fake_data | [MR-948] | - Added logging for operations |
| Visualization & Context | transformation | [MR-902] | - Implemented visualization context with thread safety |
| | profiling | [MR-931, MR-963] | - Applied context to packages<br>- Enhanced safety measures for thread handling |
| Cache & Progress Tracking | profiling | [MR-915, MR-936] | - Integrated caching mechanism across operations<br>- Implemented progress tracking<br>- Enhanced caching with encryption key parameter |
| | fake_data | [MR-967] | - Integrated caching mechanism across operations<br>- Implemented progress tracking |
| | transformation | [MR-967] | - Integrated caching mechanism across operations<br>- Implemented progress tracking |
| Code Improvements | profiling | [MR-963] | - Removed unused parameters<br>- Standardized parameter handling across packages<br>- Optimized code organization |
| Unit Tests | fake_data | [MR-907] | - Added tests for parallel processing using Dask and JobLib<br>- Added tests for transformation operations<br>- Maintained high test coverage (96.6% total coverage) |

---

### MRs (Incorporated into Summary Table Above)
All the MRs below have been integrated into the Summary of Changes table above:

- MR-902: Package & Integration and Visualization & Context for fake_data and transformation packages
- MR-905: Package & Integration for transformation package
- MR-907: Package & Integration and Unit Tests for fake_data package
- MR-910: Bug Fixing for transformation package
- MR-915: Cache & Progress Tracking for profiling package
- MR-924: Package & Integration for transformation package
- MR-929: Package & Integration for profiling package
- MR-931: Package & Integration and Visualization & Context for profiling package
- MR-936: Cache & Progress Tracking for profiling package
- MR-941: Bug Fixing for fake_data and profiling packages
- MR-948: Package & Integration and Logging Improvements for fake_data, profiling, and transformation packages
- MR-954: Logging Improvements for transformation package
- MR-961: Bug Fixing for fake_data package
- MR-962: Package & Integration and Logging Improvements for profiling package
- MR-963: Package & Integration, Visualization & Context, and Code Improvements for fake_data and profiling packages
- MR-967: Bug Fixing for fake_data and Cache & Progress Tracking for fake_data and transformation packages
- MR-980: Bug Fixing for profiling and transformation packages
- MR-1008: Bug Fixing for profiling and transformation packages
- MR-1027: Package & Integration and encryption across operations for anonymization package
- MR-1031: Package & Integration for profiling package
- MR-1035: Package & Integration for profiling package
- MR-1038: Bug Fixing for import encryption mode in profiling, transformation, and fake_data packages
- MR-1039: Bug Fixing for profiling, transformation, and fake_data packages

## Functionalities

### Summary of functionalities per each package
| Functionality | Fake Data | Profiling | Transformation |
| --- | --- | --- | --- |
| Encryption | 100% | 100% | 100% |
| Dask | 100% | 100% | 100% |
| JobLib | 100% | 100% | 100% |
| Metrics & Visualization | 100% | 100% | 100% |
| Threads for Visualization | 0% | 85% | 78% |
| Cache | 100% | 100% | 100% |
| Standardized Error Handling | 100% | 100% | 100% |
| Progress Tracker | 100% | 100% | 100% |
| Reporter | 100% | 100% | 100% |


### Details of functionalities

| Packages | Operations | DataSource utils | Encryption | Encryption mode per Task | Encryption provider for output dataset | Encryption provider for artifacts | Dask | JobLib | Metrics & Visualization | Threads for Visualization | Cache | Standardized Error Handling | Tracker Progress | Reporter | Log to files | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core/profiling/ | Anonymity | Yes | Yes | Yes | Auto-determined  | Simple | No | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | - Dask chunking and joblist cannot be applied by splitting the dataframe because k-anonymity and attribute profiling require global statistics across the entire dataset. - To enable Dask or parallel processing, each calculation must operate on the full dataframe, not on separate chunks, and results should be aggregated afterward. |
| core/profiling/ | Attribute | Yes | Yes | Yes | Auto-determined  | Simple | No | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | - Dask chunking and joblist cannot be applied by splitting the dataframe because k-anonymity and attribute profiling require global statistics across the entire dataset. - To enable Dask or parallel processing, each calculation must operate on the full dataframe, not on separate chunks, and results should be aggregated afterward. |
| core/profiling/ | Categorical | Yes | Yes | Yes | Auto-determined  | Simple | No | No | Yes | Todo | Yes | Yes | Yes | Yes | Yes | - Parallelizing categorical by splitting rows is not feasible because the function requires the entire column to compute accurate statistics such as value_counts, entropy, and top_n. Row-wise chunking would result in incorrect results since partial calculations from chunks cannot be simply merged to match the global column statistics. |
| core/profiling/ | Correlation | Yes | Yes | Yes | Auto-determined  | Simple | No | No | Yes | Todo | Yes | Yes | Yes | Yes | Yes | - Parallelizing correlation analysis by splitting rows using Dask or joblib (joblist) is not feasible because accurate correlation statisticssuch as value_counts, contingency tables, and correlation coefficientsrequire access to the entire columns or all relevant data for each field. |
| core/profiling/ | Date | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/profiling/ | Email | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/profiling/ | Group | Yes | Yes | Yes | Auto-determined  | Simple | No | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | - Parallelizing group-based analysis by splitting rows or partitions is not feasible because group metrics such as variance and duplication require access to all records of each group. If groups are fragmented across partitions, the calculated statistics will be incorrect |
| core/profiling/ | Identity | Yes | Yes | Yes | Simple | Simple | No | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | - Identity analysis cannot be parallelized by splitting rows using Dask or joblib. This is because accurate statisticssuch as value_counts, groupby, distribution, and top_nrequire access to the entire column or all groups to compute correct results. If the data is split by rows, partial results from each chunk cannot be simply merged to produce the correct global statistics. - Do not use output_format because the data only returns analysis results. |
| core/profiling/ | Mvf | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/profiling/ | Numeric | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/profiling/ | Phone | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/profiling/ | Text | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | The flow will get a sample to detect if it's a large dataset, so there's no need to apply parallelization |
| core/profiling/ | Currency | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/fake_data/ | Email | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Todo | Yes | Yes | Yes | Yes | Yes | nan |
| core/fake_data/ | Name | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Todo | Yes | Yes | Yes | Yes | Yes | nan |
| core/fake_data/ | Organization | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Todo | Yes | Yes | Yes | Yes | Yes | nan |
| core/fake_data/ | Phone | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Todo | Yes | Yes | Yes | Yes | Yes | nan |
| core/transformation/ | split_fields | Yes | Yes | Yes | Auto-determined  | Simple | No | No | Yes | Todo | Yes | Yes | Yes | Yes | Yes | - Dask and Joblib are not used here because we are only splitting the DataFrame by column groups, which is a lightweight operation. Pandas handles column selection efficiently even with millions of rows, so adding parallelism would introduce unnecessary overhead without performance benefits. |
| core/transformation/ | split_by_id_values | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Todo | Yes | Yes | Yes | Yes | Yes | nan |
| core/transformation/ | clean_invalid_values | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/transformation/ | impute_missing_values | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/transformation/ | aggregate_records | Yes | Yes | Yes | Auto-determined  | Simple | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | - Only using dask (Using joblib requires manually dividing the data into chunks and later concatenating the results, which may lead to inaccuracies in the final processed data.) |
| core/transformation/ | merge_datasets | Yes | Yes | Yes | Auto-determined  | Simple | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | - Only using dask (Using joblib requires manually dividing the data into chunks and later concatenating the results, which may lead to inaccuracies in the final processed data.) |
| core/transformation/ | remove_fields | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/transformation/ | add_modify_fields | Yes | Yes | Yes | Auto-determined  | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |
| core/transformation/ | base_transformation_op | Yes | Yes | Yes | Simple | Simple | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | nan |

---
