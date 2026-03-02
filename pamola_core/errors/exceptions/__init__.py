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

Package: pamola_core.errors.exceptions
Type: Internal (Non-Public API)

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    # core_domain.py
    "DataError",
    "DataWriteError",
    "DataFrameProcessingError",
    "ConfigurationError",
    "ConfigSaveError",
    "ProcessingError",
    "BatchProcessingError",
    "ChunkProcessingError",
    "CacheError",
    "ArtifactError",
    "VisualizationError",
    # crypto.py
    "EncryptionError",
    "DecryptionError",
    "EncryptionInitializationError",
    "KeyGenerationError",
    "KeyLoadingError",
    "MasterKeyError",
    "TaskKeyError",
    "CryptoKeyError",
    "KeyStoreError",
    "CryptoError",
    "PseudonymizationError",
    "HashCollisionError",
    "DataRedactionError",
    "FormatError",
    "ProviderError",
    "ModeError",
    "LegacyMigrationError",
    "AgeToolError",
    # tasks.py
    "TaskError",
    "TaskInitializationError",
    "TaskExecutionError",
    "TaskFinalizationError",
    "TaskDependencyError",
    "DependencyError",
    "DependencyMissingError",
    "DependencyFailedError",
    "TaskRegistryError",
    "ExecutionError",
    "ExecutionLogError",
    "CheckpointError",
    "StateSerializationError",
    "StateRestorationError",
    "ContextManagerError",
    "MaxRetriesExceededError",
    "NonRetriableError",
    "OpsError",
    "FeatureNotImplementedError",
    # validation.py
    "ValidationError",
    "FieldNotFoundError",
    "FieldTypeError",
    "FieldValueError",
    "ColumnNotFoundError",
    "InvalidParameterError",
    "MissingParameterError",
    "TypeValidationError",
    "InvalidStrategyError",
    "FileValidationError",
    "PamolaFileNotFoundError",
    "InvalidFileFormatError",
    "InvalidDataFormatError",
    "RangeValidationError",
    "ConditionalValidationError",
    "MarkerValidationError",
    "MultipleValidationErrors",
    "ValidationErrorInfo",
    "raise_if_errors",
    # resources.py
    "ResourceError",
    "DateTimeParsingError",
    "DateTimeGeneralizationError",
    "InsufficientPrivacyError",
    "MappingError",
    "MappingStorageError",
    "FakeDataError",
    "ReportingError",
    # nlp.py
    "NLPError",
    "PromptValidationError",
    "LLMError",
    "LLMConnectionError",
    "LLMGenerationError",
    "LLMResponseError",
    "ResourceNotFoundError",
    "ModelNotAvailableError",
    "ModelLoadError",
    "UnsupportedLanguageError",
    # filesystem.py
    "PathValidationError",
    "PathSecurityError",
    "DirectoryManagerError",
    "DirectoryCreationError",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "DataError": "pamola_core.errors.exceptions.core_domain",
    "DataWriteError": "pamola_core.errors.exceptions.core_domain",
    "DataFrameProcessingError": "pamola_core.errors.exceptions.core_domain",
    "ConfigurationError": "pamola_core.errors.exceptions.core_domain",
    "ConfigSaveError": "pamola_core.errors.exceptions.core_domain",
    "ProcessingError": "pamola_core.errors.exceptions.core_domain",
    "BatchProcessingError": "pamola_core.errors.exceptions.core_domain",
    "ChunkProcessingError": "pamola_core.errors.exceptions.core_domain",
    "CacheError": "pamola_core.errors.exceptions.core_domain",
    "ArtifactError": "pamola_core.errors.exceptions.core_domain",
    "VisualizationError": "pamola_core.errors.exceptions.core_domain",
    "EncryptionError": "pamola_core.errors.exceptions.crypto",
    "DecryptionError": "pamola_core.errors.exceptions.crypto",
    "EncryptionInitializationError": "pamola_core.errors.exceptions.crypto",
    "KeyGenerationError": "pamola_core.errors.exceptions.crypto",
    "KeyLoadingError": "pamola_core.errors.exceptions.crypto",
    "MasterKeyError": "pamola_core.errors.exceptions.crypto",
    "TaskKeyError": "pamola_core.errors.exceptions.crypto",
    "CryptoKeyError": "pamola_core.errors.exceptions.crypto",
    "KeyStoreError": "pamola_core.errors.exceptions.crypto",
    "CryptoError": "pamola_core.errors.exceptions.crypto",
    "PseudonymizationError": "pamola_core.errors.exceptions.crypto",
    "HashCollisionError": "pamola_core.errors.exceptions.crypto",
    "DataRedactionError": "pamola_core.errors.exceptions.crypto",
    "FormatError": "pamola_core.errors.exceptions.crypto",
    "ProviderError": "pamola_core.errors.exceptions.crypto",
    "ModeError": "pamola_core.errors.exceptions.crypto",
    "LegacyMigrationError": "pamola_core.errors.exceptions.crypto",
    "AgeToolError": "pamola_core.errors.exceptions.crypto",
    "TaskError": "pamola_core.errors.exceptions.tasks",
    "TaskInitializationError": "pamola_core.errors.exceptions.tasks",
    "TaskExecutionError": "pamola_core.errors.exceptions.tasks",
    "TaskFinalizationError": "pamola_core.errors.exceptions.tasks",
    "TaskDependencyError": "pamola_core.errors.exceptions.tasks",
    "DependencyError": "pamola_core.errors.exceptions.tasks",
    "DependencyMissingError": "pamola_core.errors.exceptions.tasks",
    "DependencyFailedError": "pamola_core.errors.exceptions.tasks",
    "TaskRegistryError": "pamola_core.errors.exceptions.tasks",
    "ExecutionError": "pamola_core.errors.exceptions.tasks",
    "ExecutionLogError": "pamola_core.errors.exceptions.tasks",
    "CheckpointError": "pamola_core.errors.exceptions.tasks",
    "StateSerializationError": "pamola_core.errors.exceptions.tasks",
    "StateRestorationError": "pamola_core.errors.exceptions.tasks",
    "ContextManagerError": "pamola_core.errors.exceptions.tasks",
    "MaxRetriesExceededError": "pamola_core.errors.exceptions.tasks",
    "NonRetriableError": "pamola_core.errors.exceptions.tasks",
    "OpsError": "pamola_core.errors.exceptions.tasks",
    "FeatureNotImplementedError": "pamola_core.errors.exceptions.tasks",
    "ValidationError": "pamola_core.errors.exceptions.validation",
    "FieldNotFoundError": "pamola_core.errors.exceptions.validation",
    "FieldTypeError": "pamola_core.errors.exceptions.validation",
    "FieldValueError": "pamola_core.errors.exceptions.validation",
    "ColumnNotFoundError": "pamola_core.errors.exceptions.validation",
    "InvalidParameterError": "pamola_core.errors.exceptions.validation",
    "MissingParameterError": "pamola_core.errors.exceptions.validation",
    "TypeValidationError": "pamola_core.errors.exceptions.validation",
    "InvalidStrategyError": "pamola_core.errors.exceptions.validation",
    "FileValidationError": "pamola_core.errors.exceptions.validation",
    "PamolaFileNotFoundError": "pamola_core.errors.exceptions.validation",
    "InvalidFileFormatError": "pamola_core.errors.exceptions.validation",
    "InvalidDataFormatError": "pamola_core.errors.exceptions.validation",
    "RangeValidationError": "pamola_core.errors.exceptions.validation",
    "ConditionalValidationError": "pamola_core.errors.exceptions.validation",
    "MarkerValidationError": "pamola_core.errors.exceptions.validation",
    "MultipleValidationErrors": "pamola_core.errors.exceptions.validation",
    "ValidationErrorInfo": "pamola_core.errors.exceptions.validation",
    "raise_if_errors": "pamola_core.errors.exceptions.validation",
    "ResourceError": "pamola_core.errors.exceptions.resources",
    "DateTimeParsingError": "pamola_core.errors.exceptions.resources",
    "DateTimeGeneralizationError": "pamola_core.errors.exceptions.resources",
    "InsufficientPrivacyError": "pamola_core.errors.exceptions.resources",
    "MappingError": "pamola_core.errors.exceptions.resources",
    "MappingStorageError": "pamola_core.errors.exceptions.resources",
    "FakeDataError": "pamola_core.errors.exceptions.resources",
    "ReportingError": "pamola_core.errors.exceptions.resources",
    "NLPError": "pamola_core.errors.exceptions.nlp",
    "PromptValidationError": "pamola_core.errors.exceptions.nlp",
    "LLMError": "pamola_core.errors.exceptions.nlp",
    "LLMConnectionError": "pamola_core.errors.exceptions.nlp",
    "LLMGenerationError": "pamola_core.errors.exceptions.nlp",
    "LLMResponseError": "pamola_core.errors.exceptions.nlp",
    "ResourceNotFoundError": "pamola_core.errors.exceptions.nlp",
    "ModelNotAvailableError": "pamola_core.errors.exceptions.nlp",
    "ModelLoadError": "pamola_core.errors.exceptions.nlp",
    "UnsupportedLanguageError": "pamola_core.errors.exceptions.nlp",
    "PathValidationError": "pamola_core.errors.exceptions.filesystem",
    "PathSecurityError": "pamola_core.errors.exceptions.filesystem",
    "DirectoryManagerError": "pamola_core.errors.exceptions.filesystem",
    "DirectoryCreationError": "pamola_core.errors.exceptions.filesystem",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        target = _LAZY_IMPORTS[name]
        if isinstance(target, tuple):
            module_name, attr_name = target
        else:
            module_name = target
            attr_name = name
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
