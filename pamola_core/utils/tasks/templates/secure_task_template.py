"""
PAMOLA Project - PAMOLA Core
------------------------
Encryption Task Template
Version: 1.0.0
Created: 2025-05-10
Author: PAMOLA Core Team

Description:
This template provides a secure framework for creating PAMOLA tasks with robust encryption
features. It demonstrates best practices for handling sensitive data, including key
management, secure file operations, and encrypted data processing with multiple encryption
modes.

Usage:
1. Rename this file to match your task ID (e.g., "t_1E_encrypt_data.py")
2. Update the task_id, task_type, and description fields
3. Configure encryption settings according to your security requirements
4. Implement your specialized encryption operations in configure_operations()
5. Run the task with appropriate encryption parameters

Example:
python t_1E_encrypt_data.py --data_repository /path/to/data --encryption_mode age --encryption_key_path /secure/keys/master.key
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import PAMOLA Core modules
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.tasks.task_config import EncryptionMode
from pamola_core.utils.tasks.task_utils import ensure_secure_directory

# Safe import of utility functions - moved to module level
try:
    from pamola_core.utils.tasks.task_utils import is_master_key_exposed
except ImportError:
    # Define fallback function if import fails
    def is_master_key_exposed() -> bool:
        """Fallback implementation when the real function is not available."""
        return False


class EncryptionTask(BaseTask):
    """
    Encryption task template for secure data processing with different encryption modes.

    This template demonstrates how to:
    1. Configure different encryption modes (simple, age)
    2. Manage encryption keys securely
    3. Process sensitive data with appropriate security measures
    4. Generate and handle encrypted outputs
    5. Implement secure operation execution patterns

    @author: PAMOLA Core Team
    @version: 1.0.0
    """

    def __init__(self):
        """Initialize the task with encryption settings."""
        super().__init__(
            task_id="t_1E_encryption_demo",  # Replace with your task ID
            task_type="encryption",  # Replace as appropriate
            description="Secure data processing with encryption capabilities",

            # Define input datasets with sensitive data
            input_datasets={
                "sensitive_data": "DATA/raw/sensitive_dataset.csv",
                # Additional datasets can be added as needed
            },

            # Optional reference datasets
            auxiliary_datasets={
                "lookup_table": "DATA/dictionaries/field_mappings.csv",
            },

            version="1.0.0"
        )

        # Define sensitive field patterns for extra protection
        self._sensitive_field_patterns = [
            "ssn", "social", "password", "secret", "key", "credit",
            "passport", "license", "dob", "birth", "address", "phone", "email"
        ]

        # Track encrypted artifacts for secure handling
        self._encrypted_artifacts = []

    def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the task with enhanced security measures.

        This method extends the base initialization process with additional
        security checks and encryption setup, including key management and
        security validation.

        Args:
            args: Command line arguments to override configuration

        Returns:
            True if initialization is successful, False otherwise
        """
        # First perform standard initialization
        if not super().initialize(args):
            self.logger.error("Base initialization failed")
            return False

        # Ensure encryption is always enabled for this task type
        if not self.use_encryption:
            self.logger.warning("Encryption was disabled; enabling with default settings")
            self.use_encryption = True
            # Default to AGE encryption for better security if available
            try:
                # Check if AGE encryption is available
                supports_age = self.encryption_manager.supports_encryption_mode(EncryptionMode.AGE)
                self.encryption_mode = EncryptionMode.AGE if supports_age else EncryptionMode.SIMPLE
                self.logger.info(f"Using encryption mode: {self.encryption_mode.value}")
            except Exception as e:
                self.logger.warning(f"Error checking encryption mode support: {e}")
                self.encryption_mode = EncryptionMode.SIMPLE

        # Perform comprehensive security checks
        security_status = self._validate_security_configuration()

        # Log detailed security status
        if self.reporter:
            self.reporter.add_operation(
                name="Security Configuration",
                status="success" if security_status else "warning",
                details={
                    "encryption_enabled": self.use_encryption,
                    "encryption_mode": self.encryption_mode.value,
                    "encryption_configured": self.encryption_manager.get_encryption_info()
                }
            )

        # Create dedicated encrypted output directory
        self._create_encrypted_output_dir()

        return True

    def _validate_security_configuration(self) -> bool:
        """
        Validate the security configuration for this task.

        Performs comprehensive checks on:
        - Encryption setup
        - Directory permissions
        - Input data security
        - Key management

        Returns:
            True if security configuration is valid, False otherwise
        """
        self.logger.info("Validating security configuration")

        valid_config = True

        # Check encryption manager initialization
        encryption_info = self.encryption_manager.get_encryption_info()
        if not encryption_info.get("enabled", False):
            self.logger.error("Encryption manager reports encryption is not enabled")
            valid_config = False

        if not encryption_info.get("key_available", False):
            self.logger.error("No encryption key available")
            valid_config = False

        # Check directory permissions on Unix-like systems
        if os.name == 'posix':
            try:
                import stat

                # Check task directory permissions
                task_dir_stat = os.stat(self.task_dir)
                task_dir_mode = task_dir_stat.st_mode

                # Group or others should not have write/execute permissions
                if task_dir_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IXGRP | stat.S_IXOTH):
                    self.logger.warning(f"Task directory {self.task_dir} has potentially insecure permissions")
                    valid_config = False
            except Exception as e:
                self.logger.warning(f"Could not check directory permissions: {e}")

        # Check if master key exposure is a risk
        # is_master_key_exposed() is already defined at module level
        if is_master_key_exposed():
            self.logger.warning("Master key may be exposed with insecure permissions")
            valid_config = False

        return valid_config

    def _create_encrypted_output_dir(self) -> Path:
        """
        Create a dedicated directory for encrypted outputs.

        Returns:
            Path to the encrypted output directory
        """
        encrypted_dir = self.task_dir / "encrypted_output"

        try:
            # Create directory with secure permissions
            secure_dir = ensure_secure_directory(encrypted_dir)
            self.logger.info(f"Created secure directory for encrypted outputs: {secure_dir}")

            # Register as a standard directory for reference
            self.directories["encrypted_output"] = secure_dir

            return secure_dir
        except Exception as e:
            self.logger.error(f"Failed to create secure directory: {e}")
            # Fallback to regular output directory
            return self.task_dir / "output"

    def configure_operations(self) -> None:
        """
        Configure encryption-focused operations.

        This method sets up a sequence of operations designed to work
        with encrypted data and produce secure outputs.
        """
        self.logger.info("Configuring encryption operations")

        # Operation 1: Sensitive Field Detection
        # This identifies potentially sensitive fields in the dataset
        self.add_operation(
            "SensitiveFieldDetectionOperation",
            dataset_name="sensitive_data",
            detection_patterns=self._sensitive_field_patterns,
            high_precision_mode=True,
            output_file="sensitive_fields_report.json"
        )

        # Operation 2: Field Encryption Operation
        # This encrypts specific sensitive fields while preserving structure
        self.add_operation(
            "FieldEncryptionOperation",
            dataset_name="sensitive_data",
            fields_to_encrypt=[
                {"field": "ssn", "method": "fernet", "preserve_format": False},
                {"field": "email", "method": "hmac", "preserve_format": False, "params": {"salt": "unique_salt"}},
                {"field": "address", "method": "format_preserving", "preserve_format": True}
            ],
            output_file="field_encrypted_data.csv"
        )

        # Operation 3: Data Anonymization with Encryption
        # This performs k-anonymity with encrypted sensitive attributes
        self.add_operation(
            "SecureAnonymizationOperation",
            dataset_name="sensitive_data",
            method="k-anonymity",
            k=5,
            quasi_identifiers=["age", "zipcode", "gender"],
            sensitive_attributes=["diagnosis", "income"],
            encrypt_sensitive=True,
            output_file="anonymized_data.csv"
        )

        # Operation 4: Full Encryption of Output Data
        # This performs full file encryption for maximum security
        self.add_operation(
            "FullEncryptionOperation",
            input_file="anonymized_data.csv",
            # Encryption settings will be passed from the task
            output_file="fully_encrypted_data.enc"
        )

        # Operation 5: Encrypted Backup Creation
        # This creates an encrypted backup of all sensitive outputs
        self.add_operation(
            "EncryptedBackupOperation",
            input_files=[
                "field_encrypted_data.csv",
                "anonymized_data.csv"
            ],
            backup_name="sensitive_data_backup",
            compression_level=9,
            output_dir="encrypted_output"
        )

        self.logger.info(f"Configured {len(self.operations)} encryption-focused operations")

    def execute(self) -> bool:
        """
        Execute operations with additional security measures.

        This method extends the default execution with additional security
        logging and validation to ensure sensitive data is properly protected.

        Returns:
            True if execution is successful, False otherwise
        """
        self.logger.info(f"Executing {len(self.operations)} secure operations")

        # Track which operations have encryption enabled
        encryption_status = []

        # Pre-execution validation
        for i, operation in enumerate(self.operations):
            op_name = operation.__class__.__name__

            # Check if operation supports encryption
            if hasattr(operation, 'supports_encryption'):
                supports_encryption = operation.supports_encryption
            else:
                # Determine through parameter inspection
                supports_encryption = 'use_encryption' in dir(operation)

            encryption_status.append({
                "operation": op_name,
                "index": i,
                "supports_encryption": supports_encryption
            })

            # Log warning if encryption not supported for critical operation
            if not supports_encryption and "Encryption" in op_name:
                self.logger.warning(f"Operation {op_name} has 'Encryption' in name but doesn't support encryption")

        # Use the standard execution logic from the parent class
        execution_result = super().execute()

        # Post-execution security check
        if execution_result:
            self._post_execution_security_check()

        return execution_result

    def _post_execution_security_check(self) -> None:
        """
        Perform security checks after execution completes.

        This validates that sensitive artifacts are properly encrypted
        and warns about any potential security issues.
        """
        self.logger.info("Performing post-execution security checks")

        # Identify artifacts that should be encrypted
        potentially_sensitive_artifacts = []
        encrypted_artifacts = []

        for artifact in self.artifacts:
            # Check artifact metadata for encryption status
            is_encrypted = (
                    hasattr(artifact, 'metadata') and
                    artifact.metadata and
                    artifact.metadata.get('encrypted', False)
            )

            # Track artifact encryption status
            artifact_path = str(artifact.path) if hasattr(artifact, 'path') else "unknown"

            if is_encrypted:
                encrypted_artifacts.append(artifact_path)
            else:
                # Check if artifact might contain sensitive data based on name
                is_potentially_sensitive = any(
                    pattern in artifact_path.lower()
                    for pattern in self._sensitive_field_patterns
                )

                if is_potentially_sensitive:
                    potentially_sensitive_artifacts.append(artifact_path)

        # Log security warnings for potentially sensitive unencrypted artifacts
        if potentially_sensitive_artifacts:
            warning_msg = f"Found {len(potentially_sensitive_artifacts)} potentially sensitive artifacts that are not encrypted"
            self.logger.warning(warning_msg)
            for path in potentially_sensitive_artifacts:
                self.logger.warning(f"- Unencrypted potentially sensitive artifact: {path}")

            if self.reporter:
                self.reporter.add_operation(
                    name="Security Warning",
                    status="warning",
                    details={
                        "warning_type": "unencrypted_sensitive_artifacts",
                        "artifact_count": len(potentially_sensitive_artifacts),
                        "artifacts": potentially_sensitive_artifacts
                    }
                )

        # Record encryption metrics
        if self.reporter:
            self.reporter.add_nested_metric(
                "security",
                "encryption_summary",
                {
                    "encrypted_artifacts": len(encrypted_artifacts),
                    "total_artifacts": len(self.artifacts),
                    "potentially_sensitive_unencrypted": len(potentially_sensitive_artifacts)
                }
            )

    def finalize(self, success: bool) -> bool:
        """
        Finalize the task with secure cleanup operations.

        This method extends the standard finalization with secure
        cleanup procedures to ensure no sensitive data remains in
        temporary storage.

        Args:
            success: Whether the task executed successfully

        Returns:
            True if finalization was successful, False otherwise
        """
        self.logger.info(f"Finalizing encryption task (success: {success})")

        # Perform secure cleanup before standard finalization
        if self.task_dir:
            # Securely clean temporary directory
            temp_dir = self.task_dir / "temp"
            if temp_dir.exists():
                self._secure_cleanup_temp_dir(temp_dir)

        # Call the parent class finalization
        finalization_result = super().finalize(success)

        # For tasks with encryption, add encryption summary to the report
        if finalization_result and self.reporter:
            encryption_info = self.encryption_manager.get_encryption_info()
            self.reporter.add_metric("encryption_configuration", encryption_info)

        return finalization_result

    def _secure_cleanup_temp_dir(self, temp_dir: Path) -> None:
        """
        Perform secure cleanup of temporary files.

        This method securely removes temporary files by overwriting
        their content before deletion to prevent data recovery.

        Args:
            temp_dir: Path to the temporary directory
        """
        self.logger.info(f"Performing secure cleanup of {temp_dir}")

        try:
            # Identify all files in temp directory
            for item in temp_dir.iterdir():
                if item.is_file():
                    try:
                        # Securely overwrite file content before deletion
                        file_size = item.stat().st_size

                        # Skip overwriting very large files (>100MB) for performance
                        if file_size < 100 * 1024 * 1024:  # 100MB
                            with open(item, 'wb') as f:
                                # Overwrite with zeros
                                f.write(b'\0' * min(file_size, 1024 * 1024))  # Max 1MB at a time
                                f.flush()
                                os.fsync(f.fileno())

                        # Delete the file
                        item.unlink()
                    except Exception as e:
                        self.logger.warning(f"Error securely cleaning file {item}: {e}")

                elif item.is_dir():
                    # Recursively clean subdirectories
                    self._secure_cleanup_temp_dir(item)

                    # Remove empty directory
                    try:
                        item.rmdir()
                    except Exception as e:
                        self.logger.warning(f"Error removing directory {item}: {e}")

        except Exception as e:
            self.logger.error(f"Error during secure temp directory cleanup: {e}")


def parse_arguments():
    """Parse command line arguments for encryption task configuration."""
    parser = argparse.ArgumentParser(description="Encryption Task for Secure Data Processing")

    # Standard task arguments
    parser.add_argument("--data_repository", help="Path to data repository")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Log level")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue execution if an operation fails")

    # Encryption-specific arguments
    parser.add_argument("--use_encryption", action="store_true", default=True,
                        help="Enable encryption for task outputs")
    parser.add_argument("--encryption_mode", choices=["simple", "age"], default="simple",
                        help="Encryption mode to use")
    parser.add_argument("--encryption_key_path",
                        help="Path to encryption key file (will be generated if it doesn't exist)")
    parser.add_argument("--secure_delete", action="store_true", default=True,
                        help="Enable secure deletion of temporary files")

    # Task-specific arguments
    parser.add_argument("--fields_to_encrypt",
                        help="Comma-separated list of fields to encrypt")
    parser.add_argument("--k_value", type=int, default=5,
                        help="K value for k-anonymity")

    return parser.parse_args()


def main():
    """Main entry point for the encryption task."""
    # Parse command line arguments
    args = parse_arguments()

    # Create task instance
    task = EncryptionTask()

    # Convert arguments to dictionary for task configuration
    arg_dict = vars(args)

    # Process fields to encrypt if provided
    if args.fields_to_encrypt:
        arg_dict["fields_to_encrypt"] = [f.strip() for f in args.fields_to_encrypt.split(",")]

    # Provide information about encryption
    print("Starting encryption task with the following security settings:")
    print(f"- Encryption Enabled: {args.use_encryption}")
    print(f"- Encryption Mode: {args.encryption_mode}")
    print(f"- Key Path: {args.encryption_key_path or 'Auto-generated'}")

    # Run the task with the provided arguments
    start_time = datetime.now()
    success = task.run(arg_dict)
    execution_time = (datetime.now() - start_time).total_seconds()

    # Handle task completion
    if success:
        print(f"Encryption task completed successfully in {execution_time:.2f} seconds")
        print(f"Secure report saved to: {task.reporter.report_path}")
        sys.exit(0)
    else:
        print(f"Encryption task failed: {task.error_info.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()