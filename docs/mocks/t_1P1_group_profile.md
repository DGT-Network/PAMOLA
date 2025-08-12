# Group Profiler Module - Software Requirements Specification

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for the Group Profiler module (t_1P1_group_profile.py), which analyzes variability within grouped resume data to determine optimal aggregation strategies for anonymization.

As part of the development:

- The task script `scripts/mock/t_1P1_group_profile.py` must be developed to work with configuration and call the pamola core operation in the library (pamola_core/profiling/analyzers/group.py)
- The pamola core operation `pamola_core/profiling/analyzers/group.py` must be refactored (reimplemented) to receive all parameters from the t_1P1 task, process data independently, create metrics, and be based on the pamola_core/utils/ops framework
- The operation must integrate with the OPS framework to leverage standardized data handling, result management, and artifact generation capabilities

This dual-component structure ensures separation between the high-level task management and the pamola core analysis operation, allowing the operation to be reused across different tasks while maintaining consistent behavior and integration with the framework.

### 1.2 Scope
The Group Profiler analyzes vertical subsets of resume data produced by the Scissors module (t_1I_scissors.py). It evaluates the variance and duplication within groups of records that share the same resume_id, enabling informed decisions about which groups can be safely aggregated during anonymization.

### 1.3 Context
In the resume anonymization process, determining which records can be aggregated is crucial for:
- Reducing data volume while preserving analytical value
- Identifying groups with high internal consistency for safe aggregation (группировки)
- Maintaining diversity in groups with significant variability
- Supporting downstream anonymization techniques with appropriate grouped data

### 1.4 Module Identification

- **Task ID**: t_1P1
- **Module Name**: Group Profiler (accesses datasets created by the previous task)
- **Module Path**: scripts/mock/t_1P1_group_profile.py
- **Home Directory**: {task_dir} (determined by task_id) - created in the data repository DATA/processed/t_1P1 (relative to the project root, which is also specified in the configuration as D:\VK_DEVEL\PAMOLA.CORE)
- **Preceded By**: t_1I_scissors.py (which splits raw data into vertical subsets)

The task operates within the established project structure, creating its output directory in the processed data section of the data repository. It processes the outputs from the scissors task and produces analytics that will be used by subsequent anonymization processes.
## 2. System Architecture

### 2.1 Overall Data Flow and Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│          scripts/mock/t_1P1_group_profile.py                │
│            (High-level Task Module)                         │
│                                                             │
│ - Loads configuration (configs/t_1P1.json)                  │
│ - Processes list of subsets for analysis                    │
│ - Manages sequential or parallel operation execution         │
│ - Aggregates and summarizes results                         │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│         pamola_core/profiling/analyzers/group.py                 │
│                 (Group Analysis Operation)                │
│                                                           │
│ - Extends BaseOperation from pamola_core/utils/ops/op_base.py    │
│ - Uses DataSource for data reading                        │
│ - Groups records and calculates variability metrics       │
│ - Creates visualizations                                  │
│ - Returns OperationResult with metrics and artifacts      │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│               Operations Framework (OPS)                  │
│                                                           │
│ - op_base.py: Base operation classes                      │
│ - op_cache.py: Operation result caching                   │
│ - op_data_source.py: Data source abstraction              │
│ - op_data_writer.py: Result and artifact writing          │
│ - op_registry.py: Operation registration                  │
│ - op_result.py: Structured operation result               │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              Support Utilities                            │
│                                                           │
│ - pamola_core/utils/io.py: I/O operations                        │
│ - pamola_core/utils/logging.py: Logging                          │
│ - pamola_core/utils/progress.py: Progress tracking               │
│ - pamola_core/utils/visualization.py: Visualization creation     │
│ - pamola_core/utils/nlp/minhash.py: Text comparison              │
│ - pamola_core/utils/nlp/stopwords.py: Stopword processing        │
└───────────────────────────────────────────────────────────┘
```

### 2.2 Task Directory Structure
I'll rewrite the Task Directory Structure section in English to match the detailed structure you've provided:

### 2.2 Task Directory Structure

The task runs from the scripts/mock directory within the project root {project_root}, which is specified in the configuration. It processes and stores data within the repository {project_root}\DATA, taking input data from the output directory of the previous task: {project_root}\DATA\processed\t_1I\output\

This task does not create output data files, but instead writes all metrics and visualizations directly to the root of the current task directory: {project_root}\DATA\processed\t_1P1

All logs are written to the {project_root}\logs directory, and the task configuration is read from and written to the top-level directory {project_root}\configs\

```
{project_root}/
├── DATA/                                # DATA REPOSITORY
│    ├── raw                             # Initial source
│    │    ├── ...                        # Other data sources 
│    │    └── 10K.csv                    # Primary source
│    └── processed                       # For all processed data
│         ├── t_1I                       # Previous task directory
│         │     ├── output
│         │           ├── IDENT.csv      # Datasets processed by previous task
│         │           ├── DETAILS.csv    # - inputs for current task
│         │           ├── ...            # (EXPIRIENCE.csv, IDENT.csv,   
│         │           │                  #  PRIMARY_EDU.csv, SPECS.csv,...)
│         ├── t_1P1                      # {task_dir} Current task !!!!!!!!!!!!!
│              ├── ...                   # (output ALL json and png here)
│              ├── ADDITIONAL_EDU_metrics.json      # Metrics for subset
│              ├── ATTESTATION_metrics.json
│              ├── summary_metrics.json        # Summary metrics across all subsets
│              ├── ADDITIONAL_EDU_variance_dist.png # Visualizations for subset
│              ├── ADDITIONAL_EDU_field_heatmap.png
│              └── output                # (NOTHING is output here - ANALYSIS ONLY)
│
├── configs/                             # All configurations stored here
     └── t_1P1.json                      # Configuration for current task
```

### 2.3 Input Data

- **Source**: DATA/processed/t_1I/output/
- **Input Datasets**:
    - ADDITIONAL_EDU.csv (resume_id| additional_education_names=3; additional_education_end_dates=1)
    - ATTESTATION.csv (resume_id| attestation_education_organizations=3; attestation_education_names=2; attestation_education_end_dates=1)
    - CONTACTS.csv (resume_id| cell_phone=3; email=2)
    - DETAILS.csv (resume_id| post=3; salary=1; area_name=2; employments=2; education_level=1)
    - EXPIRIENCE.csv (resume_id| experience_start_dates=3; experience_organizations=3; experience_posts=2)
    - IDENT.csv (resume_id| file_as=3; birth_day=1)
    - PRIMARY_EDU.csv (resume_id| primary_education_names=3; primary_education_end_dates=1)
    - SPECS.csv (resume_id| key_skill_names=3; specialization_names=2)

For each dataset, the information in parentheses specifies the grouping field (resume_id), the fields to be analyzed for variation within each group, and their relative weights for calculating weighted variation across the group. These weights indicate the importance of each field in determining overall group variability.

### 2.4 Configuration

- **Config File**: {project_root}/configs/t_1P1.json
- **Default Configuration**: Hard-coded default values used if config file is not found
- **Repository Root**: Configurable absolute path (e.g., D:\VK\_DEVEL\PAMOLA.CORE\)
- **DATA repository**: Configurable absolute or relative path (e.g., {project_root}\DATA)

The configuration file provides parameters for the task execution, including paths, processing options, and analysis settings. If the configuration file is not found, the task falls back to default values defined in the code to ensure operation can continue.
## 3. Components and Responsibilities

### 3.1 Task Module (t_1P1_group_profile.py)

**Purpose**: Coordinates analysis of all data subsets, configures and executes operations, integrates results.

**Key Responsibilities**:
1. Load and validate configuration from JSON file or use default values
2. Organize subset list for analysis
3. Load and validate field weights for each subset
4. Execute analysis operations for each subset (sequentially or in parallel)
5. Collect and aggregate results from all operations
6. Create summary report
7. Manage logging and progress tracking

**OPS Integration Points**:
- Uses `DataSource` for data retrieval
- Creates and executes `GroupOperation` instances
- Processes `OperationResult` objects
- Utilizes `DataWriter` for output
- Optionally uses `OperationCache` for performance optimization

### 3.2 Group Analysis Operation (pamola_core/profiling/analyzers/group.py)

**Purpose**: Analyzes variability within groups of records with the same identifier in a specific data subset.

**Key Responsibilities**:
1. Extend the BaseOperation class from OPS framework
2. Process a subset through DataSource abstraction
3. Group data by specified field (resume_id)
4. Calculate variability and duplication metrics for analyzed fields
5. Create visualizations through visualization utilities
6. Save results using DataWriter
7. Return standardized OperationResult

**OPS Integration Points**:
- Inherits from `BaseOperation`
- Implements `execute()` method per OPS contract
- Returns `OperationResult` with metrics and artifacts
- Uses `DataWriter` for file output
- Employs `OperationCache` for result caching
- Registers with `OperationRegistry`

## 4. Input and Output Specifications

### 4.1 Input Data
- **Source**: DATA/processed/t_1I/output/
- **Input Subsets**:
  - ADDITIONAL_EDU.csv
  - ATTESTATION.csv
  - CONTACTS.csv
  - DETAILS.csv
  - EXPIRIENCE.csv
  - IDENT.csv
  - PRIMARY_EDU.csv
  - SPECS.csv

### 4.2 Input Data Validation
- Validate field presence and types in each subset
- Verify resume_id field exists in each dataset
- Confirm numeric fields contain valid data
- Log warnings for any inconsistencies

### 4.3 Output Metrics Structure
```json
{
  "subset": "EXPIRIENCE",
  "total_groups": 10000,
  "total_records": 165450,
  "avg_variance": 0.0244,
  "max_variance": 1.0,
  "groups_to_aggregate": 8765,
  "field_metrics": {
    "experience_posts": {
      "avg_variance": 0.0235,
      "max_variance": 0.67,
      "avg_duplication_ratio": 15.32,
      "unique_values_total": 12500
    },
    "experience_organizations": {
      "avg_variance": 0.0252,
      "max_variance": 0.89,
      "avg_duplication_ratio": 14.87,
      "unique_values_total": 18700
    },
    "experience_start_dates": {
      "avg_variance": 0.0246,
      "max_variance": 0.74,
      "avg_duplication_ratio": 15.10,
      "unique_values_total": 8900
    }
  },
  "group_metrics": {
    "242621053": {
      "weighted_variance": 0.0085,
      "max_field_variance": 0.0088,
      "total_records": 2160,
      "field_variances": {
        "experience_posts": 0.0083,
        "experience_organizations": 0.0088,
        "experience_start_dates": 0.0083
      },
      "duplication_ratios": {
        "experience_posts": 16.50,
        "experience_organizations": 15.82,
        "experience_start_dates": 16.48
      },
      "should_aggregate": true
    }
  },
  "threshold_metrics": {
    "below_0.1": 7800,
    "0.1_to_0.2": 965,
    "0.2_to_0.5": 800,
    "0.5_to_0.8": 300,
    "above_0.8": 135
  },
  "algorithm_info": {
    "hash_algorithm_used": "md5",
    "text_length_threshold": 100,
    "variance_threshold": 0.2,
    "large_group_threshold": 100,
    "large_group_variance_threshold": 0.05
  },
  "execution_stats": {
    "execution_time_seconds": 12.5,
    "memory_usage_mb": 128.4,
    "records_per_second": 13236
  }
}
```

### 4.4 Summary Metrics Structure
```json
{
  "total_subsets_analyzed": 8,
  "total_groups": 10000,
  "total_records": 1254780,
  "subsets_summary": {
    "EXPIRIENCE": {
      "avg_variance": 0.0244,
      "groups_to_aggregate": 8765,
      "fields_analyzed": ["experience_posts", "experience_organizations", "experience_start_dates"]
    },
    "CONTACTS": {
      "avg_variance": 0.0186,
      "groups_to_aggregate": 9150,
      "fields_analyzed": ["email", "cell_phone"]
    },
    "IDENT": {
      "avg_variance": 0.0092,
      "groups_to_aggregate": 9450,
      "fields_analyzed": ["first_name", "last_name"]
    }
  },
  "lowest_variance_subset": "IDENT",
  "highest_variance_subset": "EXPIRIENCE",
  "global_aggregation_candidates": 8650,
  "configuration": {
    "variance_threshold": 0.2,
    "large_group_threshold": 100,
    "large_group_variance_threshold": 0.05,
    "hash_algorithm": "md5"
  },
  "execution_stats": {
    "total_execution_time_seconds": 86.2,
    "parallel_execution": true,
    "max_workers": 4
  }
}
```

### 4.5 Visualization Outputs

The module must generate the following visualizations:

#### 4.5.1 Variance Distribution Histogram
- **Purpose**: Show distribution of variance values across groups
- **X-axis**: Variance values (0-1)
- **Y-axis**: Number of groups
- **Color-coding**: Aggregation candidates vs. non-candidates
- **Annotations**: Threshold lines and key statistics

#### 4.5.2 Field Variability Heatmap
- **Purpose**: Compare variability across different fields
- **X-axis**: Analyzed fields
- **Y-axis**: Intensity representing variance
- **Color scheme**: Sequential (low to high variability)
- **Annotations**: Numeric values in cells

#### 4.5.3 Group Size vs. Variance Scatter Plot
- **Purpose**: Visualize relationship between group size and variance
- **X-axis**: Group size (log scale)
- **Y-axis**: Weighted variance
- **Color-coding**: Aggregation threshold
- **Annotations**: Size distribution indicators

#### 4.5.4 Aggregation Recommendation Pie Chart
- **Purpose**: Summarize aggregation recommendations
- **Segments**: Groups to aggregate vs. preserve
- **Annotations**: Percentages and counts
- **Color scheme**: Categorical distinction

## 5. Functional Requirements

### 5.1 Core Functionality Requirements

#### FR-1: Subset Data Processing
1. Read vertical subset files produced by Scissors module
2. For each subset, analyze specified fields (excluding ID and grouping fields)
3. Process subsets independently to enable parallel execution
4. Support partial processing when not all subsets are available

#### FR-2: Group Variance Analysis
1. Group records by resume_id for each subset
2. Calculate variance for each field in analysis list using (unique_count - 1) / (total_records - 1)
3. Calculate duplication ratio for each field using total_records / unique_count
4. Compute weighted variance across all fields based on provided field weights
5. Apply different variance thresholds based on group size

#### FR-3: Aggregation Determination
1. Mark groups with weighted variance below threshold as candidates for aggregation
2. Apply stricter thresholds for larger groups (>100 records)
3. Generate recommendations based on variance analysis
4. Create detailed metrics for both group-level and field-level variance

#### FR-4: Text Field Handling
1. For fields exceeding configured length threshold, use hashing for comparison
2. Support both MD5 (default) and MinHash algorithms for text comparison
3. Optional stopword removal for text fields via pamola_core/utils/nlp/stopwords.py
4. Normalize text before comparison (case, whitespace, etc.)

#### FR-5: Visualization Generation
1. Create variance distribution histogram for each subset
2. Generate field variability heatmap comparing fields within subset
3. Produce group size vs. variance scatter plots
4. Create aggregation recommendation pie charts
5. Save visualizations in standardized formats and locations

#### FR-6: Results Aggregation
1. Combine metrics from all analyzed subsets
2. Create summary report with overall statistics
3. Identify subsets with highest and lowest variability
4. Generate global aggregation recommendations

### 5.2 OPS Framework Integration Requirements

#### FR-7: BaseOperation Implementation
1. Inherit from BaseOperation class
2. Implement required execute() method with standardized parameters
3. Support task_dir, reporter, and progress_tracker parameters
4. Return proper OperationResult with status and metrics

#### FR-8: DataSource Usage
1. Accept DataSource objects for data retrieval
2. Handle errors when data is unavailable
3. Support schema validation for input data
4. Process dataframe chunks for large datasets

#### FR-9: Result and Artifact Management
1. Use DataWriter for standardized file output
2. Create structured OperationResult with metrics and artifacts
3. Register all output files and visualizations as artifacts
4. Add performance metrics to operation results

#### FR-10: Caching and Optimization
1. Support operation caching via OperationCache
2. Generate deterministic cache keys based on inputs
3. Validate cached results before using
4. Cache metrics and artifact references

## 6. Algorithmic Requirements

### 6.1 Variance Calculation Algorithm

The variability within each group for a specific field must be calculated using:

**Algorithm**: For each field in each group:
1. Count distinct values in the field (unique_count)
2. Count total records in the group (total_records)
3. Calculate variance using: (unique_count - 1) / (total_records - 1)
   - Output of 0: All values are identical
   - Output of 1: All values are unique
   - Intermediate values: Proportion of unique values

**Implementation Considerations**:
- Handle NULL values according to null_strategy parameter
- Handle empty groups (variance = 0)
- Handle single-record groups (variance = 0)

### 6.2 Text Similarity Algorithm

For text fields exceeding the text_length_threshold:

**MD5 Algorithm**:
1. Normalize the text (uppercase, trim whitespace)
2. Generate MD5 hash of normalized text
3. Compare hash values instead of original text

**MinHash Algorithm** (optional):
1. Break text into n-grams (shingles) of specified size
2. Apply hash functions to create a signature
3. Compare signatures to estimate similarity
4. Use Jaccard similarity for threshold decisions

### 6.3 Weighted Variance Algorithm

For determining overall group variance:

1. For each field in the group:
   a. Calculate field variance using the variance algorithm
   b. Apply weight according to field type (numeric, list, text)
2. Calculate weighted average: Σ(field_variance × field_weight) / Σ(field_weight)
3. Compare to threshold based on group size:
   a. For groups > large_group_threshold: Use large_group_variance_threshold
   b. Otherwise: Use standard variance_threshold

## 7. Non-Functional Requirements

### 7.1 Performance
- Process large datasets efficiently (180,000 records across 10,000 groups)
- Complete analysis within 5 minutes on standard hardware
- Support parallel processing for multi-subset analysis
- Implement chunking for very large datasets (>1M records)
- Memory usage below 4GB RAM for standard dataset sizes

### 7.2 Scalability
- Support linear scaling with increasing data volume
- Maintain performance with datasets up to 10x expected size
- Allow configuration of batch size and parallelism parameters
- Support distributed processing capabilities for very large datasets

### 7.3 Logging and Monitoring
- Provide comprehensive logging at appropriate detail levels
- Log progress for long-running operations
- Track memory usage during execution
- Record execution time and performance metrics
- Support configurable log verbosity levels

### 7.4 Reliability
- Handle missing or incomplete data gracefully
- Implement proper error handling for all operations
- Provide detailed error messages for troubleshooting
- Support recovery from partial failures
- Validate all inputs before processing

### 7.5 Maintainability
- Follow consistent coding standards
- Provide comprehensive documentation
- Implement unit tests for key functionality
- Support configuration via external files
- Use descriptive metric and artifact naming

## 8. Configuration Parameters

### 8.1 Task Module Parameters

| Parameter | Description | Default Value | Required |
|-----------|-------------|---------------|----------|
| repo_root | Absolute path to repository root | D:\VK\_DEVEL\PAMOLA.CORE\ | Yes |
| data_root | Path to data directory | DATA | Yes |
| input_dir | Path to input data | processed/t_1I/output | Yes |
| output_dir | Path to output directory | output | Yes |
| parallel_processing | Use parallel processing | true | No |
| max_workers | Maximum worker processes | 4 | No |
| subsets | List of subsets to analyze | All available subsets | No |
| fields_to_analyze | Dictionary mapping subsets to fields | Subset-specific selections | Yes |

### 8.2 Operation Parameters

| Parameter | Description | Default Value | Required |
|-----------|-------------|---------------|----------|
| text_length_threshold | Threshold for long text fields | 100 | No |
| variance_threshold | Threshold for aggregation decision | 0.2 | Yes |
| large_group_threshold | Threshold for large group size | 100 | No |
| large_group_variance_threshold | Variance threshold for large groups | 0.05 | No |
| field_weights | Weights for different field types | numeric: 3, list: 2, text: 1 | Yes |
| hash_algorithm | Algorithm for text comparison | md5 | No |

### 8.3 Example Configuration

```json
{
  "repo_root": "D:\\VK\\_DEVEL\\PAMOLA.CORE\\",
  "variance_threshold": 0.2,
  "large_group_threshold": 100,
  "large_group_variance_threshold": 0.05,
  "field_weights": {
    "numeric": 3,
    "list": 2,
    "text": 1
  },
  "parallel_processing": true,
  "max_workers": 4,
  "subsets": [
    "ADDITIONAL_EDU",
    "ATTESTATION",
    "CONTACTS",
    "DETAILS",
    "EXPIRIENCE",
    "IDENT",
    "PRIMARY_EDU",
    "SPECS"
  ],
  "fields_to_analyze": {
    "EXPIRIENCE": [
      "experience_posts",
      "experience_organizations",
      "experience_start_dates"
    ],
    "CONTACTS": [
      "email",
      "cell_phone"
    ],
    "IDENT": [
      "first_name",
      "last_name"
    ]
  }
}
```

## 9. OPS Framework Integration

### 9.1 BaseOperation Implementation

The GroupOperation class must properly extend the BaseOperation class:
1. Register itself via the operation_registry mechanism
2. Implement the execute() method with the standard signature
3. Follow the OPS lifecycle for initialization, execution, and result handling
4. Support proper error propagation through OperationResult

### 9.2 DataSource Integration

1. Use the DataSource abstraction to retrieve data
2. Support different data source types (files, in-memory)
3. Handle data access errors appropriately
4. Support chunked processing for large datasets

### 9.3 DataWriter Integration

1. Use DataWriter for all file operations
2. Follow the standard directory structure
3. Use appropriate formats for different artifact types
4. Include timestamps and metadata for traceability

### 9.4 OperationResult Creation

1. Create OperationResult with appropriate status code
2. Add metrics for performance and processing statistics
3. Register all artifacts with correct types and categories
4. Include error information when failures occur

### 9.5 Caching Support

1. Implement cache key generation from input parameters
2. Store and retrieve cached results appropriately
3. Validate cached results before using
4. Include cache metadata in operation metrics

## 10. Testing Requirements

### 10.1 Unit Testing
1. Test variance calculation algorithm with different input scenarios
2. Test text similarity algorithms (MD5 and MinHash)
3. Test weighted variance calculation
4. Verify aggregation threshold application
5. Validate handling of edge cases (empty groups, single records)

### 10.2 Integration Testing
1. Test integration with DataSource and DataWriter
2. Verify end-to-end processing with test datasets
3. Validate parallel processing functionality
4. Test handling of missing or incomplete data
5. Verify visualization generation

### 10.3 Performance Testing
1. Measure execution time with datasets of increasing size
2. Monitor memory usage during execution
3. Validate parallel processing efficiency
4. Test caching mechanism effectiveness
5. Measure visualization generation performance

## 11. Acceptance Criteria

1. The module correctly calculates variance and duplication metrics for all specified fields
2. Variance calculations account for field weights and group sizes
3. Visualizations clearly represent the distribution of variability
4. Output artifacts are properly stored in the task directory
5. The module handles all input datasets correctly
6. All required metrics are present in the output JSON files
7. The module integrates properly with the OPS framework
8. Performance meets the specified requirements
9. Logging provides comprehensive information for debugging and analysis
10. The module correctly identifies candidate groups for aggregation

## 12. Dependencies

### 12.1 Internal Dependencies
- pamola_core/profiling/analyzers/group.py - Main analysis logic
- pamola_core/utils/ops/op_base.py - Base operation class
- pamola_core/utils/ops/op_data_source.py - Data source abstraction
- pamola_core/utils/ops/op_data_writer.py - Data writer class
- pamola_core/utils/ops/op_result.py - Operation result classes
- pamola_core/utils/ops/op_cache.py - Operation caching utilities
- pamola_core/utils/io.py - I/O utilities
- pamola_core/utils/progress.py - Progress tracking utilities
- pamola_core/utils/visualization.py - Visualization utilities
- pamola_core/utils/nlp/minhash.py - MinHash implementation (optional)
- pamola_core/utils/nlp/stopwords.py - Stopwords utilities (optional)

### 12.2 External Dependencies
- pandas - Data manipulation library
- numpy - Numerical operations library
- matplotlib/seaborn - Visualization libraries
- datasketch - MinHash implementation (if not using internal)