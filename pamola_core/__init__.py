"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Package: pamola_core
Type: Canonical Public API

Author: Realm Inveo Inc. & DGT Network Inc.
"""

__all__ = [
    # version
    "__version__",
    # anonymization
    "AttributeSuppressionOperation",
    "CategoricalGeneralizationOperation",
    "CellSuppressionOperation",
    "DateTimeGeneralizationOperation",
    "FullMaskingOperation",
    "NumericGeneralizationOperation",
    "PartialMaskingOperation",
    "RecordSuppressionOperation",
    "UniformNumericNoiseOperation",
    "UniformTemporalNoiseOperation",
    # fake_data
    "FakeEmailOperation",
    "FakeNameOperation",
    "FakeOrganizationOperation",
    "FakePhoneOperation",
    # profiling
    "CategoricalOperation",
    "CorrelationMatrixOperation",
    "CorrelationOperation",
    "CurrencyOperation",
    "DataAttributeProfilerOperation",
    "DateOperation",
    "EmailOperation",
    "GroupAnalyzerOperation",
    "IdentityAnalysisOperation",
    "KAnonymityProfilerOperation",
    "MVFOperation",
    "NumericOperation",
    "PhoneOperation",
    "TextSemanticCategorizerOperation",
    # transformations
    "AddOrModifyFieldsOperation",
    "AggregateRecordsOperation",
    "CleanInvalidValuesOperation",
    "ImputeMissingValuesOperation",
    "MergeDatasetsOperation",
    "RemoveFieldsOperation",
    "SplitByIDValuesOperation",
    "SplitFieldsOperation",
    # metrics
    "FidelityOperation",
    "UtilityMetricOperation",
    "PrivacyMetricOperation",
    "RuleCode",
    "SchemaManager",
    "calculate_quality_with_rules",
    "calculate_provisional_risk",
    "calculate_predicted_utility",
    # analysis
    "analyze_dataset_summary",
    "calculate_full_risk",
    "analyze_descriptive_stats",
    "visualize_distribution_df",
    "analyze_correlation",
    # io
    "read_csv",
    "read_json",
    "read_excel",
    "read_parquet",
    # common
    "EncryptionMode",
    # errors
    "TaskInitializationError",
    # utilities
    "decrypt_file",
    "decrypt_data",
    "encrypt_file",
    "safe_remove_temp_file",
    "detect_encryption_mode",
    "get_key_for_task",
    "TaskConfig",
    "BaseTask",
    "load_data_operation",
    "load_settings_operation",
    "optimize_dataframe_memory",
    "BaseOperation",
    "DataSource",
    "OperationStatus",
]
# version
from pamola_core._version import __version__

# ── anonymization ──────────────────────────────────────────────────────────
from pamola_core.anonymization import (
    AttributeSuppressionOperation,
    CategoricalGeneralizationOperation,
    CellSuppressionOperation,
    DateTimeGeneralizationOperation,
    FullMaskingOperation,
    NumericGeneralizationOperation,
    PartialMaskingOperation,
    RecordSuppressionOperation,
    UniformNumericNoiseOperation,
    UniformTemporalNoiseOperation,
)

# ── fake_data ──────────────────────────────────────────────────────────────
from pamola_core.fake_data import (
    FakeEmailOperation,
    FakeNameOperation,
    FakeOrganizationOperation,
    FakePhoneOperation,
)

# ── profiling ──────────────────────────────────────────────────────────────
from pamola_core.profiling import (
    CategoricalOperation,
    CorrelationMatrixOperation,
    CorrelationOperation,
    CurrencyOperation,
    DataAttributeProfilerOperation,
    DateOperation,
    EmailOperation,
    GroupAnalyzerOperation,
    IdentityAnalysisOperation,
    KAnonymityProfilerOperation,
    MVFOperation,
    NumericOperation,
    PhoneOperation,
    TextSemanticCategorizerOperation,
)

# ── transformations ────────────────────────────────────────────────────────
from pamola_core.transformations import (
    AddOrModifyFieldsOperation,
    AggregateRecordsOperation,
    CleanInvalidValuesOperation,
    ImputeMissingValuesOperation,
    MergeDatasetsOperation,
    RemoveFieldsOperation,
    SplitByIDValuesOperation,
    SplitFieldsOperation,
)

# ── metrics ────────────────────────────────────────────────────────────────
from pamola_core.metrics import (
    FidelityOperation,
    PrivacyMetricOperation,
    RuleCode,
    SchemaManager,
    UtilityMetricOperation,
    calculate_predicted_utility,
    calculate_provisional_risk,
    calculate_quality_with_rules,
)

# ── analysis ───────────────────────────────────────────────────────────────
from pamola_core.analysis import (
    analyze_correlation,
    analyze_dataset_summary,
    analyze_descriptive_stats,
    calculate_full_risk,
    visualize_distribution_df,
)

# ── io ─────────────────────────────────────────────────────────────────────
from pamola_core.io import (
    read_csv,
    read_excel,
    read_json,
    read_parquet,
)

# ── common ─────────────────────────────────────────────────────────────────
from pamola_core.common import EncryptionMode

# ── errors ─────────────────────────────────────────────────────────────────
from pamola_core.errors import TaskInitializationError

# ── utilities ──────────────────────────────────────────────────────────────
from pamola_core.utils import (
    BaseOperation,
    BaseTask,
    DataSource,
    OperationStatus,
    TaskConfig,
    decrypt_data,
    decrypt_file,
    detect_encryption_mode,
    encrypt_file,
    get_key_for_task,
    load_data_operation,
    load_settings_operation,
    optimize_dataframe_memory,
    safe_remove_temp_file,
)

