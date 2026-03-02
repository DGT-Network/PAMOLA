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

from pamola_core._version import __version__

import importlib
from typing import Dict

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

_LAZY_IMPORTS: Dict[str, str] = {
    # anonymization
    "AttributeSuppressionOperation": "pamola_core.anonymization",
    "CategoricalGeneralizationOperation": "pamola_core.anonymization",
    "CellSuppressionOperation": "pamola_core.anonymization",
    "DateTimeGeneralizationOperation": "pamola_core.anonymization",
    "FullMaskingOperation": "pamola_core.anonymization",
    "NumericGeneralizationOperation": "pamola_core.anonymization",
    "PartialMaskingOperation": "pamola_core.anonymization",
    "RecordSuppressionOperation": "pamola_core.anonymization",
    "UniformNumericNoiseOperation": "pamola_core.anonymization",
    "UniformTemporalNoiseOperation": "pamola_core.anonymization",
    # fake_data
    "FakeEmailOperation": "pamola_core.fake_data",
    "FakeNameOperation": "pamola_core.fake_data",
    "FakeOrganizationOperation": "pamola_core.fake_data",
    "FakePhoneOperation": "pamola_core.fake_data",
    # profiling
    "CategoricalOperation": "pamola_core.profiling",
    "CorrelationMatrixOperation": "pamola_core.profiling",
    "CorrelationOperation": "pamola_core.profiling",
    "CurrencyOperation": "pamola_core.profiling",
    "DataAttributeProfilerOperation": "pamola_core.profiling",
    "DateOperation": "pamola_core.profiling",
    "EmailOperation": "pamola_core.profiling",
    "GroupAnalyzerOperation": "pamola_core.profiling",
    "IdentityAnalysisOperation": "pamola_core.profiling",
    "KAnonymityProfilerOperation": "pamola_core.profiling",
    "MVFOperation": "pamola_core.profiling",
    "NumericOperation": "pamola_core.profiling",
    "PhoneOperation": "pamola_core.profiling",
    "TextSemanticCategorizerOperation": "pamola_core.profiling",
    # transformations
    "AddOrModifyFieldsOperation": "pamola_core.transformations",
    "AggregateRecordsOperation": "pamola_core.transformations",
    "CleanInvalidValuesOperation": "pamola_core.transformations",
    "ImputeMissingValuesOperation": "pamola_core.transformations",
    "MergeDatasetsOperation": "pamola_core.transformations",
    "RemoveFieldsOperation": "pamola_core.transformations",
    "SplitByIDValuesOperation": "pamola_core.transformations",
    "SplitFieldsOperation": "pamola_core.transformations",
    # metrics
    "FidelityOperation": "pamola_core.metrics",
    "UtilityMetricOperation": "pamola_core.metrics",
    "PrivacyMetricOperation": "pamola_core.metrics",
    "RuleCode": "pamola_core.metrics",
    "SchemaManager": "pamola_core.metrics",
    "calculate_quality_with_rules": "pamola_core.metrics",
    "calculate_provisional_risk": "pamola_core.metrics",
    "calculate_predicted_utility": "pamola_core.metrics",
    # analysis
    "analyze_dataset_summary": "pamola_core.analysis",
    "calculate_full_risk": "pamola_core.analysis",
    "analyze_descriptive_stats": "pamola_core.analysis",
    "visualize_distribution_df": "pamola_core.analysis",
    "analyze_correlation": "pamola_core.analysis",
    # io
    "read_csv": "pamola_core.io",
    "read_json": "pamola_core.io",
    "read_excel": "pamola_core.io",
    "read_parquet": "pamola_core.io",
    # common
    "EncryptionMode": "pamola_core.common",
    # errors
    "TaskInitializationError": "pamola_core.errors",
    # utilities
    "decrypt_file": "pamola_core.utils",
    "decrypt_data": "pamola_core.utils",
    "encrypt_file": "pamola_core.utils",
    "safe_remove_temp_file": "pamola_core.utils",
    "detect_encryption_mode": "pamola_core.utils",
    "get_key_for_task": "pamola_core.utils",
    "TaskConfig": "pamola_core.utils",
    "BaseTask": "pamola_core.utils",
    "load_data_operation": "pamola_core.utils",
    "load_settings_operation": "pamola_core.utils",
    "optimize_dataframe_memory": "pamola_core.utils",
    "BaseOperation": "pamola_core.utils",
    "DataSource": "pamola_core.utils",
    "OperationStatus": "pamola_core.utils",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
