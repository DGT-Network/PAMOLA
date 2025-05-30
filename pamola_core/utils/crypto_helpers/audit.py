"""
Audit logging for cryptographic operations in PAMOLA.

This module provides functions for logging cryptographic operations
for audit purposes. Each operation is logged with relevant metadata
while ensuring sensitive information is not exposed.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any, Optional

# Configure logger
logger = logging.getLogger("pamola_core.utils.crypto_helpers.audit")

# Constants
AUDIT_LOG_PATH = os.environ.get("PAMOLA_AUDIT_LOG_PATH", "logs/crypto_audit.log")


def setup_audit_logging(log_path: Optional[Union[str, Path]] = None) -> None:
    """
    Set up audit logging for cryptographic operations.

    Parameters:
    -----------
    log_path : str or Path, optional
        Path to the audit log file. If not provided, uses AUDIT_LOG_PATH env var
        or default location.
    """
    global logger

    # Use provided path or default
    log_path = log_path or AUDIT_LOG_PATH
    log_path = Path(log_path)

    # Create directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure handler with specific format for audit logs
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(process)d | %(thread)d | %(message)s'
    )
    handler.setFormatter(formatter)

    # Set up a specific logger for audit
    logger = logging.getLogger("pamola_core.utils.crypto_helpers.audit")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Audit logging initialized")


def log_crypto_operation(operation: str,
                         mode: str,
                         status: str,
                         source: Optional[Union[str, Path]] = None,
                         destination: Optional[Union[str, Path]] = None,
                         task_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a cryptographic operation for audit purposes.

    Parameters:
    -----------
    operation : str
        Type of operation (e.g., 'encrypt', 'decrypt', 'key_generation')
    mode : str
        Encryption mode used (e.g., 'none', 'simple', 'age')
    status : str
        Status of the operation (e.g., 'success', 'failure')
    source : str or Path, optional
        Source file or data identifier
    destination : str or Path, optional
        Destination file or data identifier
    task_id : str, optional
        Identifier for the task that triggered the operation
    user_id : str, optional
        Identifier for the user that triggered the operation
    metadata : Dict[str, Any], optional
        Additional metadata about the operation
    """
    # Prepare metadata
    log_data = {
        "operation": operation,
        "mode": mode,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
    }

    # Add optional fields if provided
    if source:
        log_data["source"] = str(source)
    if destination:
        log_data["destination"] = str(destination)
    if task_id:
        log_data["task_id"] = task_id
    if user_id:
        log_data["user_id"] = user_id
    if metadata:
        # Filter out any sensitive information from metadata
        safe_metadata = {k: v for k, v in metadata.items()
                         if not any(sensitive in k.lower()
                                    for sensitive in ["key", "password", "secret", "token"])}
        log_data["metadata"] = safe_metadata

    # Log the operation
    message = f"CRYPTO_AUDIT | {operation} | {mode} | {status}"
    if task_id:
        message += f" | Task: {task_id}"
    if user_id:
        message += f" | User: {user_id}"

    logger.info(message, extra={"audit_data": log_data})


def log_key_access(operation: str,
                   key_id: str,
                   status: str,
                   user_id: Optional[str] = None,
                   task_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a key access operation for audit purposes.

    Parameters:
    -----------
    operation : str
        Type of operation (e.g., 'create', 'read', 'update', 'delete')
    key_id : str
        Identifier for the key being accessed (not the key itself)
    status : str
        Status of the operation (e.g., 'success', 'failure')
    user_id : str, optional
        Identifier for the user that triggered the operation
    task_id : str, optional
        Identifier for the task that triggered the operation
    metadata : Dict[str, Any], optional
        Additional metadata about the operation
    """
    log_crypto_operation(
        operation=f"key_{operation}",
        mode="keystore",
        status=status,
        source=key_id,  # Using key_id as the source
        task_id=task_id,
        user_id=user_id,
        metadata=metadata
    )