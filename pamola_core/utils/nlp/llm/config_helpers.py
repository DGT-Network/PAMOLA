"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        LLM Configuration Helpers
Package:       ppamola_core.utils.nlp.llm.config_helpers
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides high-level configuration management for LLM tasks.
It builds upon the base configuration classes in config.py to provide
task-specific configuration handling, CLI argument merging, and model
change detection.

Key Features:
- TaskConfig dataclass for complete task configuration
- ConfigManager for configuration lifecycle management
- CLI argument merging with validation
- Model change detection with cache management
- Configuration persistence and versioning
- Default configuration templates for common tasks
- Automatic configuration history tracking
- Path resolution and validation
- Thread-safe cache and history operations

Framework:
Part of PAMOLA.CORE LLM subsystem, providing task-level configuration
management on top of the base configuration infrastructure.

Changelog:
1.2.0 - Added thread safety for cache and history operations
     - Fixed path handling for absolute paths outside repository
     - Added require_existing_dataset to legacy mapping
     - Improved diff detection to avoid duplicate history entries
     - Fixed default value consistency for skip_processed
     - Enhanced path resolution in from_dict
1.1.0 - Fixed configuration serialization issues
     - Added automatic history tracking
     - Fixed skip_processed field location
     - Improved path handling in to_dict/from_dict
     - Added require_existing_dataset flag
     - Fixed reset operation order
     - Added configuration diff tracking
1.0.0 - Initial implementation
     - Basic TaskConfig and ConfigManager
     - CLI argument merging
     - Model change detection

Dependencies:
- pamola_core.utils.nlp.llm.config - Base configuration classes
- pamola_core.utils.nlp.cache - Cache management
- dataclasses - Configuration structures
- pathlib - File path handling
- json - Configuration persistence
- threading - Thread safety

TODO:
- Add configuration validation schemas
- Implement configuration inheritance
- Add configuration migration between versions
- Support for environment variable overrides
- Add configuration templates for common tasks
- Implement configuration versioning with migrations
"""

import argparse
import json
import logging
import shutil
import threading
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pamola_core.utils.nlp.cache import get_cache
from pamola_core.utils.nlp.llm.config import (
    LLMConfig, ProcessingConfig, GenerationConfig, CacheConfig, MonitoringConfig,
    resolve_model_name, get_model_info
)

# Configure logger
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Task Configuration Components
# ------------------------------------------------------------------------------

@dataclass
class TaskPaths:
    """
    Paths configuration for task execution.

    Attributes
    ----------
    data_repository : Path
        Root directory for data
    dataset_path : Path
        Path to input dataset
    task_dir : Path
        Task working directory
    output_dir : Path
        Output files directory
    checkpoint_dir : Path
        Checkpoints directory
    reports_dir : Path
        Reports directory
    """
    data_repository: Path
    dataset_path: Path
    task_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    reports_dir: Path

    def __post_init__(self):
        """Convert strings to Path objects."""
        for field_name in ['data_repository', 'dataset_path', 'task_dir',
                           'output_dir', 'checkpoint_dir', 'reports_dir']:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

    def ensure_directories(self):
        """Create all necessary directories."""
        for path_name in ['task_dir', 'output_dir', 'checkpoint_dir', 'reports_dir']:
            path = getattr(self, path_name)
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ColumnConfig:
    """
    Column configuration for dataframe processing.

    Attributes
    ----------
    source : str
        Source column name
    target : str
        Target column name
    id_column : str, optional
        ID column for tracking
    error_column : str, optional
        Error logging column
    backup_suffix : str
        Suffix for backup columns
    """
    source: str
    target: str
    id_column: Optional[str] = None
    error_column: Optional[str] = None
    backup_suffix: str = "_original"

    @property
    def is_in_place(self) -> bool:
        """Check if processing is in-place."""
        return self.source == self.target


@dataclass
class DataConfig:
    """
    Data processing configuration.

    Attributes
    ----------
    encoding : str
        File encoding
    separator : str
        CSV separator
    text_qualifier : str
        Text qualifier character
    start_id : int, optional
        Starting record ID
    end_id : int, optional
        Ending record ID
    max_records : int, optional
        Maximum records to process
    create_backup : bool
        Create backup before processing
    warn_on_in_place : bool
        Warn about in-place operations
    require_existing_dataset : bool
        Whether dataset must exist at validation
    """
    encoding: str = "UTF-16"
    separator: str = ","
    text_qualifier: str = '"'
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    max_records: Optional[int] = None
    create_backup: bool = True
    warn_on_in_place: bool = True
    require_existing_dataset: bool = True


@dataclass
class RuntimeConfig:
    """
    Runtime behavior configuration.

    Attributes
    ----------
    dry_run : bool
        Run without saving results
    test_connection_critical : bool
        Make connection test failures critical
    clear_cache_on_model_change : bool
        Clear cache when model changes
    max_errors : int
        Maximum allowed errors
    error_threshold : float
        Error rate threshold
    force_reprocess : bool
        Force reprocessing of all records
    clear_target : bool
        Clear target column before processing
    """
    dry_run: bool = False
    test_connection_critical: bool = False
    clear_cache_on_model_change: bool = True
    max_errors: int = 5
    error_threshold: float = 0.2
    force_reprocess: bool = False
    clear_target: bool = False


@dataclass
class TaskConfig:
    """
    Complete task configuration with validation.

    This class combines all configuration aspects needed for LLM task execution.

    Attributes
    ----------
    task_id : str
        Task identifier
    project_root : Path
        Project root directory
    llm : LLMConfig
        LLM connection configuration
    processing : ProcessingConfig
        Processing behavior configuration
    generation : GenerationConfig
        Text generation parameters
    cache : CacheConfig
        Cache configuration
    monitoring : MonitoringConfig
        Monitoring configuration
    columns : ColumnConfig
        Column mapping configuration
    paths : TaskPaths
        File system paths
    data : DataConfig
        Data processing configuration
    runtime : RuntimeConfig
        Runtime behavior configuration
    prompt : dict
        Prompt template configuration
    metadata : dict
        Task metadata
    """
    task_id: str
    project_root: Path
    llm: LLMConfig
    processing: ProcessingConfig
    generation: GenerationConfig
    cache: CacheConfig
    monitoring: MonitoringConfig
    columns: ColumnConfig
    paths: TaskPaths
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    prompt: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure Path objects
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

        # Ensure paths are properly configured
        self.paths.ensure_directories()

        # Add task metadata
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
        if 'version' not in self.metadata:
            self.metadata['version'] = '1.0.0'

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskConfig':
        """
        Create TaskConfig from dictionary with validation.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary

        Returns
        -------
        TaskConfig
            Validated task configuration

        Raises
        ------
        ValueError
            If required fields are missing
        """
        # Extract required fields
        task_id = config_dict.get('task_id')
        if not task_id:
            raise ValueError("task_id is required in configuration")

        project_root = config_dict.get('project_root')
        if not project_root:
            raise ValueError("project_root is required in configuration")

        # Create base configs from dictionaries
        llm_dict = config_dict.get('llm', {})
        llm_config = LLMConfig(**llm_dict) if isinstance(llm_dict, dict) else llm_dict

        processing_dict = config_dict.get('processing', {})
        processing_config = ProcessingConfig(**processing_dict) if isinstance(processing_dict,
                                                                              dict) else processing_dict

        generation_dict = config_dict.get('generation', {})
        generation_config = GenerationConfig(**generation_dict) if isinstance(generation_dict,
                                                                              dict) else generation_dict

        cache_dict = config_dict.get('cache', {})
        cache_config = CacheConfig(**cache_dict) if isinstance(cache_dict, dict) else cache_dict

        monitoring_dict = config_dict.get('monitoring', {})
        monitoring_config = MonitoringConfig(**monitoring_dict) if isinstance(monitoring_dict,
                                                                              dict) else monitoring_dict

        # Create column config
        columns_dict = config_dict.get('columns', {})
        column_config = ColumnConfig(**columns_dict) if isinstance(columns_dict, dict) else columns_dict

        # Helper function to resolve paths
        def resolve_path(path_str: str, base: Path) -> Path:
            """Resolve path string to Path object, handling absolute paths."""
            if not path_str:
                return base
            path = Path(path_str)
            # If absolute path, use as-is
            if path.is_absolute():
                return path
            # Otherwise, relative to base
            return base / path

        # Create paths - handle both new and legacy format
        data_repo = Path(config_dict.get('data_repository', '.'))
        paths_dict = config_dict.get('paths', {})

        if paths_dict:
            # New format with explicit paths
            base_repo = Path(paths_dict.get('data_repository', data_repo))

            paths = TaskPaths(
                data_repository=base_repo,
                dataset_path=resolve_path(paths_dict.get('dataset_path', config_dict.get('dataset_path', '')),
                                          base_repo),
                task_dir=resolve_path(paths_dict.get('task_dir', f'processed/{task_id}'), base_repo),
                output_dir=resolve_path(paths_dict.get('output_dir', f'processed/{task_id}/output'), base_repo),
                checkpoint_dir=resolve_path(paths_dict.get('checkpoint_dir', f'processed/{task_id}/checkpoints'),
                                            base_repo),
                reports_dir=resolve_path(paths_dict.get('reports_dir', 'reports'), base_repo)
            )
        else:
            # Legacy format - construct paths
            paths = TaskPaths(
                data_repository=data_repo,
                dataset_path=data_repo / config_dict.get('dataset_path', ''),
                task_dir=data_repo / config_dict.get('task_dir', f'processed/{task_id}'),
                output_dir=data_repo / config_dict.get('task_dir', f'processed/{task_id}') / 'output',
                checkpoint_dir=data_repo / config_dict.get('task_dir', f'processed/{task_id}') / 'checkpoints',
                reports_dir=data_repo / 'reports'
            )

        # Create data config with legacy field mapping
        data_dict = config_dict.get('data', {})
        # Map legacy fields including new require_existing_dataset
        for old_key, new_key in [('encoding', 'encoding'), ('separator', 'separator'),
                                 ('text_qualifier', 'text_qualifier'), ('start_id', 'start_id'),
                                 ('end_id', 'end_id'), ('max_records', 'max_records'),
                                 ('create_backup', 'create_backup'), ('warn_on_in_place', 'warn_on_in_place'),
                                 ('require_existing_dataset', 'require_existing_dataset')]:
            if old_key in config_dict and old_key not in data_dict:
                data_dict[old_key] = config_dict[old_key]
        data_config = DataConfig(**data_dict)

        # Create runtime config with legacy field mapping
        runtime_dict = config_dict.get('runtime', {})
        # Map legacy fields
        for old_key in ['dry_run', 'test_connection_critical', 'clear_cache_on_model_change',
                        'max_errors', 'error_threshold']:
            if old_key in config_dict and old_key not in runtime_dict:
                runtime_dict[old_key] = config_dict[old_key]
        runtime_config = RuntimeConfig(**runtime_dict)

        # Get prompt config
        prompt = config_dict.get('prompt', {})

        # Get metadata
        metadata = config_dict.get('metadata', {})

        return cls(
            task_id=task_id,
            project_root=project_root,
            llm=llm_config,
            processing=processing_config,
            generation=generation_config,
            cache=cache_config,
            monitoring=monitoring_config,
            columns=column_config,
            paths=paths,
            data=data_config,
            runtime=runtime_config,
            prompt=prompt,
            metadata=metadata
        )

    def merge_cli_args(self, args: argparse.Namespace) -> 'TaskConfig':
        """
        Merge command line arguments into configuration.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command line arguments

        Returns
        -------
        TaskConfig
            New configuration with merged arguments
        """
        # Create a copy to avoid modifying original
        new_config = replace(self)

        # Model configuration
        if hasattr(args, 'model') and args.model:
            resolved_model = resolve_model_name(args.model)
            new_config.llm = replace(new_config.llm, model_name=resolved_model)
            # Update generation config with model defaults
            new_config.generation = new_config.generation.merge_with_model_defaults(resolved_model)

        # Data selection
        if hasattr(args, 'start_id') and args.start_id is not None:
            new_config.data = replace(new_config.data, start_id=args.start_id)
        if hasattr(args, 'end_id') and args.end_id is not None:
            new_config.data = replace(new_config.data, end_id=args.end_id)
        if hasattr(args, 'max_records') and args.max_records is not None:
            new_config.data = replace(new_config.data, max_records=args.max_records)

        # Processing options - skip_processed correctly goes to processing config
        if hasattr(args, 'skip_processed'):
            if args.skip_processed:
                new_config.processing = replace(new_config.processing, skip_processed=True)
        if hasattr(args, 'no_skip_processed') and args.no_skip_processed:
            new_config.processing = replace(new_config.processing, skip_processed=False)

        # Runtime options
        if hasattr(args, 'dry_run') and args.dry_run:
            new_config.runtime = replace(new_config.runtime, dry_run=True)
        if hasattr(args, 'force_reprocess') and args.force_reprocess:
            new_config.runtime = replace(new_config.runtime, force_reprocess=True)
        if hasattr(args, 'clear_target') and args.clear_target:
            new_config.runtime = replace(new_config.runtime, clear_target=True)
        if hasattr(args, 'test_critical') and args.test_critical:
            new_config.runtime = replace(new_config.runtime, test_connection_critical=True)

        # Cache options
        if hasattr(args, 'no_cache') and args.no_cache:
            new_config.cache = replace(new_config.cache, enabled=False)
        if hasattr(args, 'no_clear_cache_on_model_change') and args.no_clear_cache_on_model_change:
            new_config.runtime = replace(new_config.runtime, clear_cache_on_model_change=False)

        # Column configuration
        if hasattr(args, 'in_place') and args.in_place:
            new_config.columns = replace(new_config.columns, target=new_config.columns.source)

        # Backup options
        if hasattr(args, 'no_backup') and args.no_backup:
            new_config.data = replace(new_config.data, create_backup=False)
        elif hasattr(args, 'create_backup') and args.create_backup:
            new_config.data = replace(new_config.data, create_backup=True)

        # Debug options
        if hasattr(args, 'debug_llm') and args.debug_llm:
            new_config.monitoring = replace(new_config.monitoring, debug_mode=True)
        if hasattr(args, 'debug_log_file') and args.debug_log_file:
            new_config.monitoring = replace(new_config.monitoring,
                                            debug_log_file=Path(args.debug_log_file))

        # Prompt template
        if hasattr(args, 'prompt_template') and args.prompt_template:
            new_config.prompt['template'] = args.prompt_template

        # Update metadata
        new_config.metadata['cli_args'] = vars(args)
        new_config.metadata['updated_at'] = datetime.now().isoformat()

        return new_config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict
            Configuration as nested dictionary
        """

        def safe_relative_path(path: Path, base: Path) -> str:
            """Get relative path or absolute if outside base."""
            try:
                return str(path.relative_to(base))
            except ValueError:
                # Path is outside base directory, return absolute
                return str(path)

        return {
            'task_id': self.task_id,
            'project_root': str(self.project_root),
            'llm': asdict(self.llm),
            'processing': asdict(self.processing),
            'generation': asdict(self.generation),
            'cache': asdict(self.cache),
            'monitoring': asdict(self.monitoring),
            'columns': asdict(self.columns),
            'paths': {
                'data_repository': str(self.paths.data_repository),
                'dataset_path': safe_relative_path(self.paths.dataset_path, self.paths.data_repository),
                'task_dir': safe_relative_path(self.paths.task_dir, self.paths.data_repository),
                # Include all paths to avoid configuration drift
                'output_dir': safe_relative_path(self.paths.output_dir, self.paths.data_repository),
                'checkpoint_dir': safe_relative_path(self.paths.checkpoint_dir, self.paths.data_repository),
                'reports_dir': safe_relative_path(self.paths.reports_dir, self.paths.data_repository),
            },
            'data': asdict(self.data),
            'runtime': asdict(self.runtime),
            'prompt': self.prompt,
            'metadata': self.metadata
        }

    def to_resolved_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with fully resolved absolute paths.

        Returns
        -------
        dict
            Configuration with absolute paths
        """
        config_dict = self.to_dict()

        # Resolve relative paths to absolute
        if 'paths' in config_dict:
            for key in ['dataset_path', 'task_dir', 'output_dir', 'checkpoint_dir', 'reports_dir']:
                if key in config_dict['paths']:
                    rel_path = config_dict['paths'][key]
                    # Check if already absolute
                    path = Path(rel_path)
                    if not path.is_absolute():
                        abs_path = self.paths.data_repository / rel_path
                        config_dict['paths'][key] = str(abs_path)

        return config_dict

    def validate(self) -> None:
        """
        Validate configuration completeness and consistency.

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Check required paths exist
        if not self.paths.data_repository.exists():
            raise ValueError(f"Data repository not found: {self.paths.data_repository}")

        # Check dataset only if required
        if self.data.require_existing_dataset and not self.paths.dataset_path.exists():
            raise ValueError(f"Dataset not found: {self.paths.dataset_path}")

        # Validate column configuration
        if not self.columns.source:
            raise ValueError("Source column must be specified")

        if not self.columns.target:
            raise ValueError("Target column must be specified")

        # Validate prompt
        if 'template' not in self.prompt:
            raise ValueError("Prompt template must be specified")

        # Warn about in-place mode
        if self.columns.is_in_place and self.data.warn_on_in_place:
            logger.warning(f"In-place mode enabled: column '{self.columns.source}' will be overwritten")


# ------------------------------------------------------------------------------
# Configuration Manager
# ------------------------------------------------------------------------------

class ConfigManager:
    """
    Manages task configuration lifecycle.

    This class handles loading, saving, and tracking configuration changes
    for LLM tasks with thread-safe operations.

    Attributes
    ----------
    task_id : str
        Task identifier
    project_root : Path
        Project root directory
    config_dir : Path
        Configuration directory
    config_path : Path
        Main configuration file path
    """

    # Class-level locks for thread safety
    _cache_lock = threading.Lock()
    _history_lock = threading.Lock()

    def __init__(self, task_id: str, project_root: Path):
        """
        Initialize configuration manager.

        Parameters
        ----------
        task_id : str
            Task identifier
        project_root : Path
            Project root directory
        """
        self.task_id = task_id
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "configs"
        self.config_path = self.config_dir / f"{task_id}.json"

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for task.

        This method should be overridden by task-specific implementations
        to provide appropriate defaults.

        Returns
        -------
        dict
            Default configuration dictionary
        """
        return {
            "task_id": self.task_id,
            "project_root": str(self.project_root),
            "data_repository": str(self.project_root / "DATA"),
            "dataset_path": "raw/data.csv",
            "task_dir": f"processed/{self.task_id}",
            "llm": {
                "provider": "lmstudio",
                "model_name": "LLM1",
                "api_url": "http://localhost:1234/v1"
            },
            "processing": {
                "batch_size": 1,
                "use_processing_marker": True,
                "processing_marker": "~",
                "skip_processed": True  # Consistent with ProcessingConfig default
            },
            "generation": {
                "temperature": 0.3,
                "max_tokens": 512
            },
            "cache": {
                "enabled": True,
                "type": "memory"
            },
            "monitoring": {
                "monitor_performance": True
            },
            "columns": {
                "source": "text",
                "target": "text_processed",
                "id_column": "id"
            },
            "data": {
                "encoding": "UTF-8",
                "max_records": 10,
                "require_existing_dataset": True
            },
            "prompt": {
                "template": "Process: {text}"
            }
        }

    def load_config(self,
                    reset: bool = False,
                    ignore: bool = False,
                    default_config: Optional[Dict[str, Any]] = None) -> TaskConfig:
        """
        Load configuration with fallback to defaults.

        Parameters
        ----------
        reset : bool
            Reset configuration to defaults
        ignore : bool
            Ignore saved config and use defaults
        default_config : dict, optional
            Custom default configuration

        Returns
        -------
        TaskConfig
            Loaded configuration
        """
        config_dict: Dict[str, Any] = {}
        if ignore:
            logger.info("Using default configuration (--ignore-config flag)")
            config_dict = default_config or self.get_default_config()
            return TaskConfig.from_dict(config_dict)

        # Determine which config to use
        if reset or not self.config_path.exists():
            # Use default config
            config_dict = default_config or self.get_default_config()

            # If reset and file exists - backup first
            if reset and self.config_path.exists():
                logger.info(f"Resetting configuration to default at {self.config_path}")
                backup_path = self.config_path.with_suffix('.json.bak')
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Previous config backed up to {backup_path}")

                # Record reset in history
                self.save_config_change('reset', {
                    'reason': 'manual_reset',
                    'backup_path': str(backup_path)
                })

            # Save new config
            logger.info(f"Creating {'reset' if reset else 'new'} config at {self.config_path}")
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            # Load existing config
            logger.info(f"Loading config from {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

        # Create and validate TaskConfig
        config = TaskConfig.from_dict(config_dict)

        try:
            config.validate()
        except ValueError as e:
            logger.warning(f"Configuration validation warning: {e}")

        return config

    def save_config(self, config: TaskConfig, suffix: Optional[str] = None):
        """
        Save configuration to file.

        Parameters
        ----------
        config : TaskConfig
            Configuration to save
        suffix : str, optional
            Suffix to add to filename (e.g., "_backup")
        """
        # Get old config for comparison
        old_config = None
        if self.config_path.exists() and not suffix:
            try:
                with open(self.config_path, 'r') as f:
                    old_config = json.load(f)
            except:
                pass

        # Determine save path
        if suffix:
            save_path = self.config_path.with_name(f"{self.config_path.stem}{suffix}.json")
        else:
            save_path = self.config_path

        config_dict = config.to_dict()

        # Save config
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to {save_path}")

        # Record change in history if main config updated and changed
        if not suffix and old_config:
            changes = self._diff_configs(old_config, config_dict)
            # Only record if there are actual changes
            if changes:
                self.save_config_change('config_update', {
                    'changes': changes
                })

    def check_model_change(self,
                           current_model: str,
                           task_dir: Path,
                           clear_on_change: bool = True) -> bool:
        """
        Check if model changed and optionally clear cache.

        Parameters
        ----------
        current_model : str
            Current model name or alias
        task_dir : Path
            Task directory for model tracking
        clear_on_change : bool
            Whether to clear cache on model change

        Returns
        -------
        bool
            True if model changed
        """
        model_file = task_dir / ".last_model"
        model_changed = False

        # Resolve current model
        current_resolved = resolve_model_name(current_model)

        if model_file.exists():
            try:
                # Read last model
                last_model_data = json.loads(model_file.read_text())
                last_model = last_model_data.get('model', '')
                last_resolved = last_model_data.get('resolved', '')

                # Check if resolved model changed
                if last_resolved != current_resolved:
                    model_changed = True
                    logger.warning(
                        f"Model changed from '{last_model}' ({last_resolved}) "
                        f"to '{current_model}' ({current_resolved})"
                    )

                    if clear_on_change:
                        self._clear_cache()

                    # Record change in history
                    self.save_config_change('model_change', {
                        'from': {'model': last_model, 'resolved': last_resolved},
                        'to': {'model': current_model, 'resolved': current_resolved},
                        'cache_cleared': clear_on_change
                    })

            except Exception as e:
                logger.warning(f"Could not read last model file: {e}")

        # Save current model info
        try:
            model_data = {
                'model': current_model,
                'resolved': current_resolved,
                'timestamp': datetime.now().isoformat(),
                'info': get_model_info(current_model)
            }
            model_file.write_text(json.dumps(model_data, indent=2))
        except Exception as e:
            logger.warning(f"Could not save model file: {e}")

        return model_changed

    def _clear_cache(self):
        """Clear the text cache (thread-safe)."""
        with self._cache_lock:
            logger.warning("Clearing cache due to model change...")
            try:
                # Try text cache first
                cache = get_cache('text')
                if hasattr(cache, 'clear'):
                    cache.clear()
                    logger.info("Text cache cleared successfully")
            except Exception:
                # Fall back to memory cache
                try:
                    cache = get_cache('memory')
                    if hasattr(cache, 'clear'):
                        cache.clear()
                        logger.info("Memory cache cleared successfully")
                except Exception as e:
                    logger.error(f"Failed to clear cache: {e}")

    def get_config_history(self) -> list:
        """
        Get configuration change history (thread-safe).

        Returns
        -------
        list
            List of configuration changes
        """
        with self._history_lock:
            history_file = self.config_dir / f"{self.task_id}_history.json"

            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

            return []

    def save_config_change(self, change_type: str, details: Dict[str, Any]):
        """
        Record configuration change in history (thread-safe).

        Parameters
        ----------
        change_type : str
            Type of change (e.g., "model_change", "reset", "cli_override")
        details : dict
            Change details
        """
        with self._history_lock:
            history = self.get_config_history()

            change_record = {
                'timestamp': datetime.now().isoformat(),
                'type': change_type,
                'details': details
            }

            history.append(change_record)

            # Keep only last 100 changes
            if len(history) > 100:
                history = history[-100:]

            history_file = self.config_dir / f"{self.task_id}_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

    def _diff_configs(self, old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find differences between configurations.

        Parameters
        ----------
        old : dict
            Old configuration
        new : dict
            New configuration

        Returns
        -------
        dict
            Configuration differences (empty if no changes)
        """
        diff = {}

        def _recursive_diff(old_val: Any, new_val: Any, path: str = "") -> Dict[str, Any]:
            """Recursively find differences."""
            local_diff = {}

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Compare dictionaries
                all_keys = set(old_val.keys()) | set(new_val.keys())
                for key in all_keys:
                    key_path = f"{path}.{key}" if path else key
                    if key not in old_val:
                        local_diff[key_path] = {'added': new_val[key]}
                    elif key not in new_val:
                        local_diff[key_path] = {'removed': old_val[key]}
                    else:
                        nested_diff = _recursive_diff(old_val[key], new_val[key], key_path)
                        local_diff.update(nested_diff)
            elif old_val != new_val:
                local_diff[path] = {'from': old_val, 'to': new_val}

            return local_diff

        return _recursive_diff(old, new)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def merge_nested_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge nested dictionaries.

    Parameters
    ----------
    base : dict
        Base dictionary
    update : dict
        Dictionary with updates

    Returns
    -------
    dict
        Merged dictionary
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_nested_dicts(result[key], value)
        else:
            result[key] = value

    return result


def validate_task_config(config: Dict[str, Any], required_fields: list) -> None:
    """
    Validate task configuration has required fields.

    Parameters
    ----------
    config : dict
        Configuration to validate
    required_fields : list
        List of required field paths (e.g., ["llm.model_name", "columns.source"])

    Raises
    ------
    ValueError
        If required fields are missing
    """
    missing = []

    for field_path in required_fields:
        parts = field_path.split('.')
        current = config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                missing.append(field_path)
                break

    if missing:
        raise ValueError(f"Missing required configuration fields: {missing}")


def get_field_from_path(config: Dict[str, Any], field_path: str, default: Any = None) -> Any:
    """
    Get field value from nested configuration using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    field_path : str
        Dot-separated field path (e.g., "llm.model_name")
    default : Any
        Default value if field not found

    Returns
    -------
    Any
        Field value or default
    """
    parts = field_path.split('.')
    current = config

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def set_field_from_path(config: Dict[str, Any], field_path: str, value: Any) -> Dict[str, Any]:
    """
    Set field value in nested configuration using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    field_path : str
        Dot-separated field path (e.g., "llm.model_name")
    value : Any
        Value to set

    Returns
    -------
    dict
        Updated configuration
    """
    result = config.copy()
    parts = field_path.split('.')
    current = result

    # Navigate to parent
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Set value
    current[parts[-1]] = value

    return result