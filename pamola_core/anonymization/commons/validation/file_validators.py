"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        File and Path Validators
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  File and path validation functions for anonymization operations.
  Provides validators for file paths, directories, file formats,
  and external configuration files used in anonymization.

Key Features:
  - File path validation with existence checking
  - Directory path validation with creation options
  - File extension and format validation
  - Configuration file structure validation
  - Hierarchy dictionary file validation
  - Size and permission checks
  - Multi-file validation support

Design Principles:
  - Consistent validation interface using ValidationResult
  - Clear error messages with actionable information
  - Support for both strict and lenient validation modes
  - Integration with base validators and decorators

Usage:
  Used by anonymization operations to validate input files,
  output directories, and configuration files before processing.

Dependencies:
  - pathlib - Path manipulation
  - os - File system operations
  - json - JSON file validation
  - csv - CSV file validation
  - typing - Type hints
  - pandas - DataFrame operations

Changelog:
  1.0.0 - Initial implementation extracted from validation_utils
"""

import csv
import json
# Configure module logger
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseValidator, ValidationResult
from .decorators import standard_validator, validation_handler
from .exceptions import (
    FileNotFoundError as FileNotFoundValidationError,
    FileValidationError,
    InvalidFileFormatError,
    ValidationError
)

logger = logging.getLogger(__name__)


# =============================================================================
# File Path Validators
# =============================================================================

class FilePathValidator(BaseValidator):
    """
    Validator for file paths with existence and format checks.

    Attributes:
        must_exist: Whether file must exist
        valid_extensions: List of allowed extensions
        max_size_mb: Maximum file size in MB
        check_permissions: Whether to check read permissions
    """

    def __init__(self,
                 must_exist: bool = True,
                 valid_extensions: Optional[List[str]] = None,
                 max_size_mb: Optional[float] = None,
                 check_permissions: bool = True):
        """
        Initialize file path validator.

        Args:
            must_exist: Whether file must exist
            valid_extensions: Allowed file extensions (e.g., ['.csv', '.json'])
            max_size_mb: Maximum allowed file size in MB
            check_permissions: Whether to check read permissions
        """
        super().__init__()
        self.must_exist = must_exist
        self.valid_extensions = valid_extensions
        self.max_size_mb = max_size_mb
        self.check_permissions = check_permissions

    @validation_handler
    def validate(self, file_path: Union[str, Path], **kwargs) -> ValidationResult:
        """
        Validate file path.

        Args:
            file_path: Path to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)

        # Convert to Path object
        try:
            path = Path(file_path)
        except Exception as e:
            raise FileValidationError(
                file_path=str(file_path),
                reason=f"Invalid path format: {e}"
            )

        # Check existence
        if self.must_exist and not path.exists():
            raise FileNotFoundValidationError(str(path))

        if path.exists():
            # Check if it's actually a file
            if not path.is_file():
                raise FileValidationError(
                    file_path=str(path),
                    reason="Path exists but is not a file"
                )

            # Check extension
            if self.valid_extensions:
                ext = path.suffix.lower()
                valid_exts = [e.lower() for e in self.valid_extensions]
                if ext not in valid_exts:
                    raise InvalidFileFormatError(
                        file_path=str(path),
                        expected_formats=self.valid_extensions,
                        actual_format=ext
                    )

            # Check size
            if self.max_size_mb is not None:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    raise FileValidationError(
                        file_path=str(path),
                        reason=f"File size {size_mb:.2f}MB exceeds limit {self.max_size_mb}MB"
                    )

            # Check permissions
            if self.check_permissions and not os.access(path, os.R_OK):
                raise FileValidationError(
                    file_path=str(path),
                    reason="File is not readable"
                )

            # Add file info to result
            result.details['path'] = str(path)
            result.details['size_mb'] = path.stat().st_size / (1024 * 1024)
            result.details['extension'] = path.suffix

        return result


class DirectoryPathValidator(BaseValidator):
    """
    Validator for directory paths with creation options.

    Attributes:
        must_exist: Whether directory must exist
        create_if_missing: Whether to create missing directories
        check_permissions: Whether to check write permissions
    """

    def __init__(self,
                 must_exist: bool = True,
                 create_if_missing: bool = False,
                 check_permissions: bool = True):
        """
        Initialize directory path validator.

        Args:
            must_exist: Whether directory must exist
            create_if_missing: Whether to create if missing
            check_permissions: Whether to check write permissions
        """
        super().__init__()
        self.must_exist = must_exist
        self.create_if_missing = create_if_missing
        self.check_permissions = check_permissions

    @validation_handler
    def validate(self, dir_path: Union[str, Path], **kwargs) -> ValidationResult:
        """
        Validate directory path.

        Args:
            dir_path: Directory path to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)

        # Convert to Path object
        try:
            path = Path(dir_path)
        except Exception as e:
            raise FileValidationError(
                file_path=str(dir_path),
                reason=f"Invalid path format: {e}"
            )

        # Handle non-existent directories
        if not path.exists():
            if self.create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    result.details['created'] = True
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    raise FileValidationError(
                        file_path=str(path),
                        reason=f"Failed to create directory: {e}"
                    )
            elif self.must_exist:
                raise FileValidationError(
                    file_path=str(path),
                    reason="Directory does not exist",
                    error_type="DIRECTORY_NOT_FOUND"
                )

        # Check if it's actually a directory
        if path.exists() and not path.is_dir():
            raise FileValidationError(
                file_path=str(path),
                reason="Path exists but is not a directory"
            )

        # Check permissions
        if path.exists() and self.check_permissions:
            if not os.access(path, os.W_OK):
                raise FileValidationError(
                    file_path=str(path),
                    reason="Directory is not writable"
                )

        result.details['path'] = str(path)
        return result


# =============================================================================
# Configuration File Validators
# =============================================================================

class JSONFileValidator(BaseValidator):
    """
    Validator for JSON configuration files.

    Attributes:
        schema: Optional JSON schema for validation
        required_keys: Required top-level keys
    """

    def __init__(self,
                 schema: Optional[Dict[str, Any]] = None,
                 required_keys: Optional[List[str]] = None):
        """
        Initialize JSON file validator.

        Args:
            schema: JSON schema for validation
            required_keys: Required top-level keys
        """
        super().__init__()
        self.schema = schema
        self.required_keys = required_keys or []

    @validation_handler
    def validate(self, file_path: Union[str, Path], **kwargs) -> ValidationResult:
        """
        Validate JSON file structure and content.

        Args:
            file_path: Path to JSON file
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        path = Path(file_path)

        # First validate file path
        file_validator = FilePathValidator(
            valid_extensions=['.json'],
            must_exist=True
        )
        file_result = file_validator.validate(path)
        if not file_result.is_valid:
            return file_result

        # Load and validate JSON content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise FileValidationError(
                file_path=str(path),
                reason=f"Invalid JSON format: {e}"
            )
        except Exception as e:
            raise FileValidationError(
                file_path=str(path),
                reason=f"Failed to read JSON file: {e}"
            )

        # Check required keys
        if self.required_keys:
            missing_keys = [k for k in self.required_keys if k not in data]
            if missing_keys:
                raise ValidationError(
                    f"Missing required keys in JSON: {missing_keys}",
                    error_code="MISSING_JSON_KEYS"
                )

        # Schema validation would go here if schema provided
        # For MVP, we keep it simple

        result.details['num_keys'] = len(data)
        result.details['keys'] = list(data.keys())[:10]  # First 10 keys

        return result


class CSVFileValidator(BaseValidator):
    """
    Validator for CSV files.

    Attributes:
        required_columns: Required column names
        delimiter: Expected delimiter
        has_header: Whether file should have header
        encoding: Expected encoding
    """

    def __init__(self,
                 required_columns: Optional[List[str]] = None,
                 delimiter: str = ',',
                 has_header: bool = True,
                 encoding: str = 'utf-8'):
        """
        Initialize CSV file validator.

        Args:
            required_columns: Required column names
            delimiter: Expected delimiter
            has_header: Whether file should have header
            encoding: Expected file encoding
        """
        super().__init__()
        self.required_columns = required_columns or []
        self.delimiter = delimiter
        self.has_header = has_header
        self.encoding = encoding

    @validation_handler
    def validate(self, file_path: Union[str, Path], **kwargs) -> ValidationResult:
        """
        Validate CSV file structure.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        path = Path(file_path)

        # First validate file path
        file_validator = FilePathValidator(
            valid_extensions=['.csv', '.tsv', '.txt'],
            must_exist=True
        )
        file_result = file_validator.validate(path)
        if not file_result.is_valid:
            return file_result

        # Check CSV structure
        try:
            # Read first few rows to validate structure
            with open(path, 'r', encoding=self.encoding) as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 5:  # Check first 5 rows
                        break

            if not rows:
                raise FileValidationError(
                    file_path=str(path),
                    reason="CSV file is empty"
                )

            # Check header
            if self.has_header:
                headers = rows[0]
                result.details['columns'] = headers
                result.details['num_columns'] = len(headers)

                # Check required columns
                if self.required_columns:
                    missing_cols = set(self.required_columns) - set(headers)
                    if missing_cols:
                        raise ValidationError(
                            f"Missing required columns: {list(missing_cols)}",
                            error_code="MISSING_CSV_COLUMNS"
                        )

            # Check consistency
            if len(rows) > 1:
                expected_cols = len(rows[0])
                for i, row in enumerate(rows[1:], 1):
                    if len(row) != expected_cols:
                        result.warnings.append(
                            f"Row {i} has {len(row)} columns, expected {expected_cols}"
                        )

        except UnicodeDecodeError:
            raise FileValidationError(
                file_path=str(path),
                reason=f"File encoding does not match expected '{self.encoding}'"
            )
        except Exception as e:
            raise FileValidationError(
                file_path=str(path),
                reason=f"Failed to read CSV file: {e}"
            )

        return result


class HierarchyFileValidator(BaseValidator):
    """
    Validator for hierarchy dictionary files.

    Validates files containing hierarchical mappings used in
    categorical generalization operations.
    """

    def __init__(self,
                 file_format: str = 'auto',
                 validate_structure: bool = True):
        """
        Initialize hierarchy file validator.

        Args:
            file_format: Expected format ('json', 'csv', 'auto')
            validate_structure: Whether to validate hierarchy structure
        """
        super().__init__()
        self.file_format = file_format
        self.validate_structure = validate_structure

    @validation_handler
    def validate(self, file_path: Union[str, Path], **kwargs) -> ValidationResult:
        """
        Validate hierarchy dictionary file.

        Args:
            file_path: Path to hierarchy file
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        path = Path(file_path)

        # Determine format
        if self.file_format == 'auto':
            ext = path.suffix.lower()
            if ext == '.json':
                format_type = 'json'
            elif ext in ['.csv', '.tsv']:
                format_type = 'csv'
            else:
                raise InvalidFileFormatError(
                    file_path=str(path),
                    expected_formats=['.json', '.csv'],
                    actual_format=ext
                )
        else:
            format_type = self.file_format

        # Validate based on format
        if format_type == 'json':
            json_validator = JSONFileValidator()
            result = json_validator.validate(path)

            if result.is_valid and self.validate_structure:
                # Load and check hierarchy structure
                with open(path, 'r') as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    raise ValidationError(
                        "Hierarchy JSON must be a dictionary",
                        error_code="INVALID_HIERARCHY_FORMAT"
                    )

                # Basic structure validation
                result.details['num_mappings'] = len(data)
                result.details['sample_mappings'] = dict(list(data.items())[:5])

        elif format_type == 'csv':
            csv_validator = CSVFileValidator(
                required_columns=['value', 'parent'] if self.validate_structure else None
            )
            result = csv_validator.validate(path)

        return result


# =============================================================================
# Multi-File Validators
# =============================================================================

class MultiFileValidator(BaseValidator):
    """
    Validator for multiple related files.

    Used when operations require multiple input files that
    should be validated together.
    """

    def __init__(self,
                 file_validator: Optional[BaseValidator] = None,
                 min_files: int = 1,
                 max_files: Optional[int] = None,
                 consistent_format: bool = True):
        """
        Initialize multi-file validator.

        Args:
            file_validator: Validator to use for each file
            min_files: Minimum number of files required
            max_files: Maximum number of files allowed
            consistent_format: Whether all files must have same format
        """
        super().__init__()
        self.file_validator = file_validator or FilePathValidator()
        self.min_files = min_files
        self.max_files = max_files
        self.consistent_format = consistent_format

    @validation_handler
    def validate(self, file_paths: List[Union[str, Path]], **kwargs) -> ValidationResult:
        """
        Validate multiple files.

        Args:
            file_paths: List of file paths to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)

        if not isinstance(file_paths, list):
            raise ValidationError(
                "file_paths must be a list",
                error_code="INVALID_INPUT_TYPE"
            )

        # Check count constraints
        num_files = len(file_paths)
        if num_files < self.min_files:
            raise ValidationError(
                f"Too few files: {num_files} < {self.min_files}",
                error_code="INSUFFICIENT_FILES"
            )

        if self.max_files and num_files > self.max_files:
            raise ValidationError(
                f"Too many files: {num_files} > {self.max_files}",
                error_code="EXCESS_FILES"
            )

        # Validate each file
        extensions = set()
        valid_files = []

        for i, file_path in enumerate(file_paths):
            try:
                file_result = self.file_validator.validate(file_path)
                if file_result.is_valid:
                    valid_files.append(str(Path(file_path)))
                    extensions.add(Path(file_path).suffix.lower())
                else:
                    result.errors.append(f"File {i}: {file_result.errors}")
                    result.is_valid = False
            except Exception as e:
                result.errors.append(f"File {i}: {str(e)}")
                result.is_valid = False

        # Check format consistency
        if self.consistent_format and len(extensions) > 1:
            result.warnings.append(
                f"Inconsistent file formats: {list(extensions)}"
            )

        result.details['num_files'] = num_files
        result.details['valid_files'] = len(valid_files)
        result.details['formats'] = list(extensions)

        return result


# =============================================================================
# Convenience Functions (for backward compatibility)
# =============================================================================

@standard_validator()
def validate_file_path(file_path: Union[str, Path],
                       must_exist: bool = True,
                       valid_extensions: Optional[List[str]] = None) -> ValidationResult:
    """
    Validate file path (convenience function).

    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        valid_extensions: Allowed extensions

    Returns:
        ValidationResult
    """
    validator = FilePathValidator(
        must_exist=must_exist,
        valid_extensions=valid_extensions
    )
    return validator.validate(file_path)


@standard_validator()
def validate_directory_path(dir_path: Union[str, Path],
                            must_exist: bool = True,
                            create_if_missing: bool = False) -> ValidationResult:
    """
    Validate directory path (convenience function).

    Args:
        dir_path: Directory path to validate
        must_exist: Whether directory must exist
        create_if_missing: Whether to create if missing

    Returns:
        ValidationResult
    """
    validator = DirectoryPathValidator(
        must_exist=must_exist,
        create_if_missing=create_if_missing
    )
    return validator.validate(dir_path)


# Module exports
__all__ = [
    # Validators
    'FilePathValidator',
    'DirectoryPathValidator',
    'JSONFileValidator',
    'CSVFileValidator',
    'HierarchyFileValidator',
    'MultiFileValidator',

    # Convenience functions
    'validate_file_path',
    'validate_directory_path'
]