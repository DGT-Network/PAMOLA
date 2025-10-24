# 3. FUNCTIONAL REQUIREMENTS (Enhanced Version)

## 3.1. Supported Data Types

The fake data generation system must support processing various data types, classified into three priority levels.

### 3.1.1. Priority 1 (Critical Identifiers) _(High Priority)_

| Data Type              | Validation                                                                                      | Formats                                                                                     | Generation Specifics                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Names (Full Names)** | Valid alphabet characters; length 2-50 characters; gender correspondence; no special characters | Full names: "Last Name First Name Patronymic" (Russia), "First Name Last Name" (US, Canada) | Consider gender, nationality; frequency characteristics; transliteration; preserve case and formatting |
| **Email Addresses**    | Compliance with RFC 5322; uniqueness; domain validation                                         | username@domain.com; username+tag@domain.com                                                | Generation based on full names; relationship with first/last name; realistic domains                   |
| **Phone Numbers**      | Regional code compliance; valid length; permitted characters                                    | +7 (XXX) XXX-XX-XX; +1 (XXX) XXX-XXXX                                                       | Consider regional codes; geographic consistency; proper formatting                                     |
| **National IDs**       | Format constraints; checksums; regional templates                                               | SSN: XXX-XX-XXXX; Russian passport: XXXX XXXXXX                                             | Compliance with national standards; correct check digits; regional consistency                         |
| **Bank Accounts**      | Number mask; BIN verification; bank consistency                                                 | XXXX XXXX XXXX XXXX                                                                         | Real BIN ranges; country-specific formatting; correct check digits                                     |
| **User Logins**        | Permitted characters; length 3-20 characters; uniqueness                                        | [a-zA-Z0-9_.-]+                                                                             | Connection to full name; name/surname variations; transliteration; digit addition                      |

### 3.1.2. Priority 2 (Sensitive Data) _(Medium Priority)_

|Data Type|Validation|Formats|Generation Specifics|
|---|---|---|---|
|**Patronymics**|Consistency with first name and gender; language rules|Ivanovich/Ivanovna (Russia)|Derivative from male first name; handling exceptions|
|**Birth Dates, Age**|Realism (0-120 years); statistical distribution|DD.MM.YYYY; YYYY-MM-DD|Consider demographics; correlation with education and work experience|
|**Job Titles**|Compliance with job title dictionary; industry specifics|"Manager"; "Senior Developer"|Consistency with age and education; correlation with salary|
|**Employers**|Real or plausible names; regional affiliation|"LLC Horns and Hooves"; "Acme Corp"|Connection with region and position; real or templated names|
|**Medical Data**|ICD compliance; diagnostic realism|ICD-10: A00.0-Z99.9|Consider disease statistics; correlation with age and gender|

### 3.1.3. Priority 3 (Indirect Identifiers) _(Low Priority)_

|Data Type|Validation|Formats|Generation Specifics|
|---|---|---|---|
|**Postal Addresses**|Country-specific address format; geographic correctness|"15 Lenin St., apt. 42, Moscow, 115114"|Component consistency; use of classifiers|
|**Geographic Coordinates**|Valid values; binding to real location|55.7558° N, 37.6173° E|Connection with address; random deviation within a given radius|
|**Demographic Data**|Statistical correctness; inter-field consistency|Distribution by age, gender, education|Correspondence to real demographics; preservation of statistical properties|
|**Social Media URLs**|Correct structure; user relationship|facebook.com/username; vk.com/id12345|Connection with full name and login; consideration of platform popularity in the region|

## 3.2. Data Generation _(High Priority)_

### 3.2.1. Sources and Structures for Generation

#### 3.2.1.1. Dictionaries

**Dictionary requirements:**

- Structure: CSV/JSON with fields id, value, gender (opt.), region (opt.), weight (opt.)
- Support for loading from local files, databases, remote sources
- Dictionary versioning and mapping migration
- Possibility to extract and create dictionaries from existing data

#### 3.2.1.2. Statistical Profiles _(Medium Priority)_

**Requirements:**

- Region profiles: demographic characteristics by region
- Data profiles: results of source data analysis
- Correlation matrices: dependencies between attributes

### 3.2.2. Generation Algorithms for Specific Data Types _(High Priority)_

#### 3.2.2.1. Name and Patronymic Generation

**Algorithm requirements:**

- Consider national and language specifics
- Consider gender characteristics
- Use frequency distribution characteristics
- Generate patronymics based on male first names (for Russian language)
- Support various formats depending on region
- Transliteration for multilingual systems

#### 3.2.2.2. Address Data Generation _(Medium Priority)_

**Algorithm requirements:**

- Support for national classifiers (KLADR/FIAS for Russia)
- Geographic consistency of components
- Consider regional formats and specifics
- Realistic house and apartment numbers
- Maintain internal address consistency

#### 3.2.2.3. Data Generation Considering Interdependencies _(Medium Priority)_

**Algorithm requirements:**

- Correlation between related fields (position-salary)
- Age dependencies (education-age-position)
- Geographic connections (address-phone-IP)
- Preservation of statistical characteristics of source data

### 3.2.3. Algorithm Customization Mechanisms _(Medium Priority)_

#### 3.2.3.1. Generator Configuration

**Requirements:**

- Configuration of generator type for each field
- Regional settings and localization parameters
- Paths to dictionaries and data sources
- Weight coefficients and formatting templates
- Empty value handling strategies
- Support for relationships between fields
- Deterministic generation through seed

#### 3.2.3.2. Generation Strategies in Absence of Dictionaries

**Requirements:**

- Use of Faker library with locale configuration
- Generation based on patterns and rules
- Random generation with format preservation
- Switching between strategies for different data types

#### 3.2.3.3. Performance Parameters _(Medium Priority)_

**Requirements:**

- Batch processing size configuration
- Vectorization of operations where possible
- Parallel processing with configurable number of processes
- Caching of dictionaries and frequent values
- Memory consumption control and data compression

## 3.3. Mapping and Transitivity _(High Priority)_

### 3.3.1. Basic Implementation of 1-to-1 Mapping

**Mapping store requirements:**

- Bidirectional mappings (original→synthetic and synthetic→original)
- Support for various field types
- Marking of transitive links
- Optional value encryption
- Serialization/deserialization to various formats
- Ability to restore original values

### 3.3.2. Conflict Resolution Algorithms for Transitive Replacements _(High Priority)_

**Algorithm requirements:**

- Detection of cyclic replacements (A → B → C → A)
- Marking of transitive mappings
- Conflict resolution when a synthetic value exists as an original
- Value modification to break cyclic dependencies
- Support for various conflict resolution strategies

### 3.3.3. Examples of Complex Cases of Transitive Replacements _(Medium Priority)_

#### 3.3.3.1. Cyclic Replacements

**Scenario:**

- A → B (alice@example.com is replaced with bob@example.com)
- B → C (bob@example.com is replaced with charlie@example.com)
- C → A (attempt to replace charlie@example.com with alice@example.com)

**Requirement:** Cycle detection must occur and modification of the last replacement.

#### 3.3.3.2. Resolution of Existing Value Conflicts

**Scenario:** Synthetic value already exists as an original in another record.

**Requirements:**

- Conflict detection
- Synthetic value modification (suffix addition)
- Preservation of connection with original
- Ensuring uniqueness of new value

## 3.4. Empty Values and NULL _(High Priority)_

### 3.4.1. NULL Handling Strategies

**Requirements:**

- **PRESERVE**: preserve NULL values without changes
- **RANDOM**: replace with a random plausible value
- **DEFAULT**: use a default value
- **EXCLUDE**: exclude NULL values from processing

**Configuration:**

- Default global strategy
- Specific strategies for each data type
- Default values for DEFAULT strategy
- Combination with generators for RANDOM strategy

### 3.4.2. Specialized Configurations _(Medium Priority)_

**Banking data:**

- Exclusion of NULL from critical fields (client ID, account number)
- Default values for ratings and limits
- Preservation of NULL in fields with semantic meaning (end date)

**Medical data:**

- Exclusion of NULL from identification fields
- Standard values for physical parameters
- Preservation of NULL in clinical fields meaning absence of diseases

## 3.5. Performance and Resource Requirements _(High Priority)_

### 3.5.1. General Performance Metrics

|Operation Type|Required Performance|Maximum Memory Consumption|Scalability|
|---|---|---|---|
|Data Generation|10,000 records/sec for simple types, 1,000 records/sec for complex|4 GB RAM per 1 million records|Linear scaling with number of cores|
|Mapping and Saving|5,000 records/sec|2 GB RAM per 1 million records|Limited by I/O operations|
|Data Restoration|20,000 records/sec|1 GB RAM per 1 million records|Linear scaling|

### 3.5.2. Metrics by Data Type

|Data Type|Time for 100K Records|Memory Consumption|Optimization Specifics|
|---|---|---|---|
|**Names (Full Names)**|60 sec|500 MB|Dictionary caching, template preprocessing|
|**Email Addresses**|45 sec|400 MB|Pre-computation of patterns, regexp optimization|
|**Phone Numbers**|30 sec|300 MB|Format templates, code caching|
|**National IDs**|90 sec|600 MB|Validation algorithm optimization, batch processing|

### 3.5.3. Scaling Strategies _(Medium Priority)_

**Requirements:**

- Division of large sets into batches of optimal size
- Calculation of optimal batch size based on available memory
- Incremental processing without loading the entire set into memory
- Multi-process and multi-thread processing
- Data division into parts for parallel processing
- Synchronization of access to shared resources

## 3.6. Output Artifacts and Integration _(High Priority)_

### 3.6.1. Operation Artifacts

|Artifact|File Type|Path|Description|
|---|---|---|---|
|**Output Data Set**|CSV, JSON, parquet|`{task_dir}/output/{operation_name}_output.{format}`|Main operation result - data with replaced values|
|**Mapping Maps**|JSON, pickle|`{task_dir}/mapping/{field_name}_mapping.{format}`|Correspondences between original and synthetic values|
|**Operation Metrics**|JSON|`{task_dir}/metrics/{operation_name}_metrics.json`|Operation execution statistics (time, memory, volume)|
|**Replacement Log**|JSONL|`{task_dir}/logs/{operation_name}_replacements.jsonl`|Detailed log of replacements made for audit|
|**Quality Metrics**|JSON|`{task_dir}/quality/{field_name}_quality.json`|Anonymization quality metrics|
|**Visualizations**|PNG, HTML|`{task_dir}/visualizations/{field_name}_distribution.png`|Distribution graphs before and after anonymization|

### 3.6.2. Output Data Structure

**Requirements:**

- Standard structure for output data
- Operation metadata (name, time, version)
- Processed records with original and replaced values
- Processing status information and statistics

### 3.6.3. Field Replacement or Enrichment Strategy _(High Priority)_

**Requirements:**

- Each operation must support two modes of data transformation:
    - **REPLACE**: Replace the existing column with fake data (default)
    - **ENRICH**: Add a new column with fake data, keeping the original
- When in ENRICH mode, the new column name should default to `{original_field_name}_fake`
- Operations should provide configuration parameter for output field naming
- Audit logs must clearly indicate whether data was replaced or enriched
- When enriching, original-to-fake mappings must still be properly maintained

**Example configuration:**

```python
operation = NameGenerationOperation(
    field_name="first_name",
    mode="ENRICH",  # Options: "REPLACE" or "ENRICH"
    output_field_name="first_name_anonymized"  # Optional, defaults to "first_name_fake" in ENRICH mode
)
```

### 3.6.4. Integration with PAMOLA.CORE Infrastructure _(High Priority)_

**Requirements:**

- Inheritance from PAMOLA.CORE base operation classes (`BaseOperation`)
- Use of standard data sources (`DataSource`)
- Return of results in `OperationResult` format
- Integration with logging system (`Logger`)
- Support for progress tracking (`ProgressTracker`)
- Standardized saving of results and artifacts

### 3.6.5. Integration with External Sources _(Medium Priority)_

**Requirements:**

- Connection to various source types (databases, APIs, files)
- Unified interface for access to classifiers
- Support for KLADR/FIAS for Russia and similar for other countries
- Caching and optimization of access to frequently used data

## 3.7. Integration with Formal Anonymization Models _(Low Priority for MVP)_

### 3.7.1. Integration with k-anonymity, l-diversity and t-closeness

**Requirements:**

- Calculation of metrics for original and anonymized data
- Visualization and interpretation of metrics
- Setting target values
- Display of metrics as additional columns

### 3.7.2. Data Generation Considering Formal Models

**Requirements:**

- Data generation complying with given constraints
- Automatic adjustment to achieve set parameters
- Identification of problematic record groups
- Application of additional generalization and masking methods

## 3.8. Dictionary Building and Management _(High Priority)_

### 3.8.1. Creating Dictionaries Based on Existing Sets

**Requirements:**

- Building name dictionaries based on source data
- Extraction of frequency characteristics
- Preservation of gender and regional features
- Filtering by minimum frequency of occurrence
- Normalization of weight coefficients

### 3.8.2. Dictionary Validation and Updating _(Medium Priority)_

**Requirements:**

- Structure checking and compliance with requirements
- Validation of mandatory fields and data types
- Control of identifier uniqueness
- Normalization of weight coefficients
- Supplementation with missing fields and values

### 3.8.3. Dictionary Versioning and Change Control _(Medium Priority)_

**Requirements:**

- Registration of dictionaries in version control system
- Storage of metadata (type, description, creation date)
- Determination of change type (backward compatibility)
- Incremental version updates
- History storage and rollback capability

## 3.9. Additional Functional Capabilities _(Medium Priority)_

### 3.9.1. Deterministic Generation Using Seed

**Requirements:**

- Use of fixed initial value (seed)
- Saving seed in metadata for reproducibility
- Generation of unique seeds for each record
- Saving current state of PRNG
- Documenting seeds for regression testing

### 3.9.2. Generation of Related Data _(Low Priority for MVP)_

**Requirements:**

- Generation of primary and related subjects (e.g., family members)
- Calculation of number of related subjects
- Different types of relationships (one-to-one, one-to-many)
- Consistency between related subjects
- For family relationships: consistent surnames, ages, contact information

### 3.9.3. Intelligent Data Type Recognition _(Low Priority for MVP)_

**Requirements:**

- Automatic type determination based on column name and value samples
- Application of specialized detectors for different types
- Calculation of correspondence to various types with determination of most probable
- Use of data profiling results

## 3.10. Audit and Security Requirements _(High Priority)_

### 3.10.1. Operation Logging

**Requirements:**

- Recording information about each operation in the audit log
- Hashing of original and synthetic values
- Structured storage of logs (JSONL or CSV)
- Inclusion of timestamps, operation type, fields, and mapping references
- Ensuring log integrity and immutability

### 3.10.2. Encryption of Sensitive Data _(Medium Priority)_

**Requirements:**

- Encryption of sensitive data with modern algorithms
- Encryption key management
- Protection of mapping dictionaries
- Encryption during storage and transmission
- Secure storage and key rotation

### 3.10.3. Mapping Access Management _(Low Priority for MVP)_

**Requirements:**

- Access policies for each field
- Restriction by users and roles
- Policy expiration period
- Access verification before each operation
- Detailed access logging

## 3.11. Output Documentation and Reporting _(Medium Priority)_

### 3.11.1. Generation of Operation Execution Reports

**Requirements:**

- Reports in various formats (HTML, PDF, Markdown)
- Information about operation performed and results
- Key metrics and statistics
- List of artifacts with descriptions
- Visualizations for visual representation
- Errors and warnings
- Customizable report templates

### 3.11.2. Documentation of Dictionaries and Configurations _(Medium Priority)_

**Requirements:**

- Automatic creation of documentation for dictionaries
- General information (name, type, number of records)
- Structure and record examples
- Statistical information about content
- Usage instructions
- Documentation in various formats

## 3.12. Output Artifacts _(High Priority)_

**Requirements for standard operation artifacts:**

|Artifact|Format|Content|Purpose|
|---|---|---|---|
|**Output Data Set**|CSV, JSON, parquet|Anonymized data with structure preservation|Main result, data for use|
|**Mapping File**|JSON, pickle|Correspondences between original and synthetic values|Original restoration, audit, control|
|**Performance Metrics**|JSON|Statistics of time, memory, performance|Optimization, efficiency analysis|
|**Quality Metrics**|JSON|K-anonymity, reidentification risks|Evaluation of anonymization quality|
|**Visualizations**|PNG, HTML|Distribution graphs and comparisons|Visual analysis of results|
|**Operation Log**|JSONL|Detailed log of replacements made|Audit and security|
|**Final Report**|HTML, PDF|Summary document about the operation|Providing results to stakeholders|