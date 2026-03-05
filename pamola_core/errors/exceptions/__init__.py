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

from pamola_core.errors.exceptions.core_domain import DataError
from pamola_core.errors.exceptions.core_domain import DataWriteError
from pamola_core.errors.exceptions.core_domain import DataFrameProcessingError
from pamola_core.errors.exceptions.core_domain import ConfigurationError
from pamola_core.errors.exceptions.core_domain import ConfigSaveError
from pamola_core.errors.exceptions.core_domain import ProcessingError
from pamola_core.errors.exceptions.core_domain import BatchProcessingError
from pamola_core.errors.exceptions.core_domain import ChunkProcessingError
from pamola_core.errors.exceptions.core_domain import CacheError
from pamola_core.errors.exceptions.core_domain import ArtifactError
from pamola_core.errors.exceptions.core_domain import VisualizationError

from pamola_core.errors.exceptions.crypto import EncryptionError
from pamola_core.errors.exceptions.crypto import DecryptionError
from pamola_core.errors.exceptions.crypto import EncryptionInitializationError
from pamola_core.errors.exceptions.crypto import KeyGenerationError
from pamola_core.errors.exceptions.crypto import KeyLoadingError
from pamola_core.errors.exceptions.crypto import MasterKeyError
from pamola_core.errors.exceptions.crypto import TaskKeyError
from pamola_core.errors.exceptions.crypto import CryptoKeyError
from pamola_core.errors.exceptions.crypto import KeyStoreError
from pamola_core.errors.exceptions.crypto import CryptoError
from pamola_core.errors.exceptions.crypto import PseudonymizationError
from pamola_core.errors.exceptions.crypto import HashCollisionError
from pamola_core.errors.exceptions.crypto import DataRedactionError
from pamola_core.errors.exceptions.crypto import FormatError
from pamola_core.errors.exceptions.crypto import ProviderError
from pamola_core.errors.exceptions.crypto import ModeError
from pamola_core.errors.exceptions.crypto import LegacyMigrationError
from pamola_core.errors.exceptions.crypto import AgeToolError

from pamola_core.errors.exceptions.tasks import TaskError
from pamola_core.errors.exceptions.tasks import TaskInitializationError
from pamola_core.errors.exceptions.tasks import TaskExecutionError
from pamola_core.errors.exceptions.tasks import TaskFinalizationError
from pamola_core.errors.exceptions.tasks import TaskDependencyError
from pamola_core.errors.exceptions.tasks import DependencyError
from pamola_core.errors.exceptions.tasks import DependencyMissingError
from pamola_core.errors.exceptions.tasks import DependencyFailedError
from pamola_core.errors.exceptions.tasks import TaskRegistryError
from pamola_core.errors.exceptions.tasks import ExecutionError
from pamola_core.errors.exceptions.tasks import ExecutionLogError
from pamola_core.errors.exceptions.tasks import CheckpointError
from pamola_core.errors.exceptions.tasks import StateSerializationError
from pamola_core.errors.exceptions.tasks import StateRestorationError
from pamola_core.errors.exceptions.tasks import ContextManagerError
from pamola_core.errors.exceptions.tasks import MaxRetriesExceededError
from pamola_core.errors.exceptions.tasks import NonRetriableError
from pamola_core.errors.exceptions.tasks import OpsError
from pamola_core.errors.exceptions.tasks import FeatureNotImplementedError

from pamola_core.errors.exceptions.validation import ValidationError
from pamola_core.errors.exceptions.validation import FieldNotFoundError
from pamola_core.errors.exceptions.validation import FieldTypeError
from pamola_core.errors.exceptions.validation import FieldValueError
from pamola_core.errors.exceptions.validation import ColumnNotFoundError
from pamola_core.errors.exceptions.validation import InvalidParameterError
from pamola_core.errors.exceptions.validation import MissingParameterError
from pamola_core.errors.exceptions.validation import TypeValidationError
from pamola_core.errors.exceptions.validation import InvalidStrategyError
from pamola_core.errors.exceptions.validation import FileValidationError
from pamola_core.errors.exceptions.validation import PamolaFileNotFoundError
from pamola_core.errors.exceptions.validation import InvalidFileFormatError
from pamola_core.errors.exceptions.validation import InvalidDataFormatError
from pamola_core.errors.exceptions.validation import RangeValidationError
from pamola_core.errors.exceptions.validation import ConditionalValidationError
from pamola_core.errors.exceptions.validation import MarkerValidationError
from pamola_core.errors.exceptions.validation import MultipleValidationErrors
from pamola_core.errors.exceptions.validation import ValidationErrorInfo
from pamola_core.errors.exceptions.validation import raise_if_errors

from pamola_core.errors.exceptions.resources import ResourceError
from pamola_core.errors.exceptions.resources import DateTimeParsingError
from pamola_core.errors.exceptions.resources import DateTimeGeneralizationError
from pamola_core.errors.exceptions.resources import InsufficientPrivacyError
from pamola_core.errors.exceptions.resources import MappingError
from pamola_core.errors.exceptions.resources import MappingStorageError
from pamola_core.errors.exceptions.resources import FakeDataError
from pamola_core.errors.exceptions.resources import ReportingError

from pamola_core.errors.exceptions.nlp import NLPError
from pamola_core.errors.exceptions.nlp import PromptValidationError
from pamola_core.errors.exceptions.nlp import LLMError
from pamola_core.errors.exceptions.nlp import LLMConnectionError
from pamola_core.errors.exceptions.nlp import LLMGenerationError
from pamola_core.errors.exceptions.nlp import LLMResponseError
from pamola_core.errors.exceptions.nlp import ResourceNotFoundError
from pamola_core.errors.exceptions.nlp import ModelNotAvailableError
from pamola_core.errors.exceptions.nlp import ModelLoadError
from pamola_core.errors.exceptions.nlp import UnsupportedLanguageError

from pamola_core.errors.exceptions.filesystem import PathValidationError
from pamola_core.errors.exceptions.filesystem import PathSecurityError
from pamola_core.errors.exceptions.filesystem import DirectoryManagerError
from pamola_core.errors.exceptions.filesystem import DirectoryCreationError

