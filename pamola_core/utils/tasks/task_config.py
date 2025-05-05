"""
Task Configuration Module for HHR project.

This module provides functionality for loading and managing task configurations,
including loading from project configuration files and overriding with
command line arguments.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from pamola_core.utils.io import read_json

# Set up logger
logger = logging.getLogger(__name__)


class TaskConfig:
    """
    Task configuration container and manager.

    Holds all configuration parameters for a task and provides
    methods for accessing them.
    """

    def __init__(self, config_dict: Dict[str, Any], task_id: str, task_type: str):
        """
        Initialize configuration with values from dictionary.

        Parameters:
        -----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration values
        task_id : str
            ID of the task this configuration is for
        task_type : str
            Type of the task this configuration is for
        """
        self.task_id = task_id
        self.task_type = task_type

        # Extract base configuration
        self.data_repository = config_dict.get('data_repository', './data')
        self.log_level = config_dict.get('log_level', 'INFO')

        # Extract directory structure
        dir_structure = config_dict.get('directory_structure', {})
        self.raw_dir = dir_structure.get('raw', 'raw')
        self.processed_dir = dir_structure.get('processed', 'processed')
        self.logs_dir = dir_structure.get('logs', 'logs')

        # Extract task-specific configuration
        task_config = config_dict.get('tasks', {}).get(task_id, {})
        self.dependencies = task_config.get('dependencies', [])
        self.continue_on_error = task_config.get('continue_on_error', False)
        self.use_encryption = task_config.get('use_encryption', False)

        # Extract encryption configuration
        encryption_config = config_dict.get('encryption', {})
        self.encryption_key_path = task_config.get('encryption_key_path') or encryption_config.get('key_path')

        # Extract vectorization configuration
        self.use_vectorization = task_config.get('use_vectorization',
                                                 config_dict.get('use_vectorization', False))
        self.parallel_processes = task_config.get('parallel_processes',
                                                  config_dict.get('parallel_processes', 1))

        # Extract scope configuration
        self.scope = task_config.get('scope', {})
        if 'fields' in task_config:
            self.scope['fields'] = task_config['fields']
        if 'datasets' in task_config:
            self.scope['datasets'] = task_config['datasets']
        if 'field_groups' in task_config:
            self.scope['field_groups'] = task_config['field_groups']

        # Set up directories
        self.output_directory = self._resolve_path(
            self.data_repository,
            self.processed_dir,
            task_type,
            task_id
        )

        self.report_path = self._resolve_path(
            self.data_repository,
            'reports',
            task_type,
            f"{task_id}.json"
        )

        # Set additional properties from task_config
        for key, value in task_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @staticmethod
    def _resolve_path(*parts):
        """
        Resolve a path from components.

        Parameters:
        -----------
        *parts : str
            Path components to join

        Returns:
        --------
        Path
            Resolved path
        """
        # Convert all parts to strings first
        str_parts = [str(part) for part in parts]
        # Use Path to create the path object
        return Path(*str_parts)

    def override_with_args(self, args: Dict[str, Any]):
        """
        Override configuration with command line arguments.

        Parameters:
        -----------
        args : Dict[str, Any]
            Command line arguments
        """
        if not args:
            return

        # Override data repository if specified
        if 'data_repository' in args and args['data_repository']:
            self.data_repository = args['data_repository']

            # Update dependent paths
            self.output_directory = self._resolve_path(
                self.data_repository,
                self.processed_dir,
                self.task_type,
                self.task_id
            )

            self.report_path = self._resolve_path(
                self.data_repository,
                'reports',
                self.task_type,
                f"{self.task_id}.json"
            )

        # Override log level if specified
        if 'log_level' in args and args['log_level']:
            self.log_level = args['log_level']

        # Override use_encryption if specified
        if 'use_encryption' in args:
            self.use_encryption = args['use_encryption']

        # Override encryption_key_path if specified
        if 'encryption_key_path' in args:
            self.encryption_key_path = args['encryption_key_path']

        # Override use_vectorization if specified
        if 'use_vectorization' in args:
            self.use_vectorization = args['use_vectorization']

        # Override parallel_processes if specified
        if 'parallel_processes' in args:
            self.parallel_processes = args['parallel_processes']

        # Override scope fields if specified
        if 'fields' in args:
            if not hasattr(self, 'scope'):
                self.scope = {}
            self.scope['fields'] = args['fields']

        # Override scope datasets if specified
        if 'datasets' in args:
            if not hasattr(self, 'scope'):
                self.scope = {}
            self.scope['datasets'] = args['datasets']

        # Override continue_on_error if specified
        if 'continue_on_error' in args:
            self.continue_on_error = args['continue_on_error']

        # Process any additional arguments that might match attributes
        for key, value in args.items():
            if key not in ['data_repository', 'log_level', 'use_encryption',
                           'encryption_key_path', 'use_vectorization',
                           'parallel_processes', 'fields', 'datasets',
                           'continue_on_error'] and value is not None:
                setattr(self, key, value)

    def __str__(self):
        """String representation of the configuration."""
        attributes = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"TaskConfig({', '.join(attributes)})"

    def get_scope_fields(self) -> List[str]:
        """
        Get fields defined in the scope.

        Returns:
        --------
        List[str]
            List of field names in the scope
        """
        if hasattr(self, 'scope') and isinstance(self.scope, dict):
            return self.scope.get('fields', [])
        return []

    def get_scope_datasets(self) -> List[str]:
        """
        Get datasets defined in the scope.

        Returns:
        --------
        List[str]
            List of dataset names in the scope
        """
        if hasattr(self, 'scope') and isinstance(self.scope, dict):
            return self.scope.get('datasets', [])
        return []

    def get_scope_field_groups(self) -> Dict[str, List[str]]:
        """
        Get field groups defined in the scope.

        Returns:
        --------
        Dict[str, List[str]]
            Dictionary mapping group names to lists of field names
        """
        if hasattr(self, 'scope') and isinstance(self.scope, dict):
            return self.scope.get('field_groups', {})
        return {}


def load_task_config(task_id: str, task_type: str, args: Optional[Dict[str, Any]] = None) -> TaskConfig:
    """
    Load task configuration from project configuration file and override
    with command line arguments.

    Parameters:
    -----------
    task_id : str
        ID of the task
    task_type : str
        Type of the task
    args : Dict[str, Any], optional
        Command line arguments

    Returns:
    --------
    TaskConfig
        Task configuration
    """
    # Default config path
    config_path = Path("configs/hhr_config.json")

    # Try to load config file
    try:
        config_dict = read_json(config_path)
        logger.debug(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.warning(f"Failed to load configuration from {config_path}: {e}")
        logger.warning("Using default configuration")
        config_dict = {}

    # Create task configuration
    task_config = TaskConfig(config_dict, task_id, task_type)

    # Override with command line arguments
    if args:
        task_config.override_with_args(args)
        logger.debug("Configuration overridden with command line arguments")

    return task_config