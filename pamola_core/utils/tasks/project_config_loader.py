# project_config_loader.py

"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Project Configuration Loader
Description: YAML project configuration loading and management
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for loading and managing project-level
configurations in YAML format, including variable substitution, caching,
and standardized access to project structure.

Key features:
- YAML configuration parsing with comments support
- JSON fallback for backward compatibility
- Variable substitution using Jinja2 templates
- Configuration caching for performance
- Default value handling
- Structured validation
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Initialize with default values
Template = None  # initial value
JINJA2_AVAILABLE = False

try:
    from jinja2 import Template as _JinjaTemplate
    Template = _JinjaTemplate  # reassign to the imported class
    JINJA2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JinjaTemplate = None  # Define the variable that would have been imported
    Template = None  # explicit reassignment for linters/typing
    # JINJA2_AVAILABLE remains False

# Set up logger
logger = logging.getLogger(__name__)

# Default filename and extensions
PROJECT_CONFIG_FILENAME = "prj_config.yaml"
PROJECT_CONFIG_LEGACY_FILENAME = "prj_config.json"

# Cache for parsed configurations
_config_cache: Dict[str, Dict[str, Any]] = {}


def find_project_root() -> Path:
    """
    Locate the project root directory.

    Priority:
    1) PAMOLA_PROJECT_ROOT environment variable
    2) .pamolaProject marker file or presence of configs/prj_config.{yaml,json}
    3) Git repository root (if GitPython is available; otherwise search for .git directory)
    4) Current directory (with warning)

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    cwd = Path.cwd().resolve()

    # -- 1. Check environment variable ------------------------------
    env_root = os.getenv("PAMOLA_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if root.is_dir() and ((root / ".pamolaProject").exists() or (root / "configs").is_dir()):
            logger.debug("Using project root from environment variable: %s", root)
            return root
        logger.warning("PAMOLA_PROJECT_ROOT points to invalid directory: %s", root)

    # -- 2. Look for marker file or configs directory up the hierarchy --
    max_depth = int(os.getenv("PAMOLA_MAX_SEARCH_DEPTH", "20"))
    current = cwd
    for _ in range(max_depth):
        if (current / ".pamolaProject").exists():
            logger.debug("Found project root by marker file: %s", current)
            return current

        cfg = current / "configs"
        if cfg.is_dir() and ((cfg / PROJECT_CONFIG_FILENAME).exists() or
                             (cfg / PROJECT_CONFIG_LEGACY_FILENAME).exists()):
            logger.debug("Found project root by configs directory: %s", current)
            return current

        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # -- 3a. Try to use GitPython to find repository root (if available) --
    try:
        import git  # Try to import
    except ImportError:
        git = None  # Define the name to exist
        logger.debug("GitPython not installed - skipping Git repository detection")
    else:
        try:
            repo = git.Repo(cwd, search_parent_directories=True)
            git_root = Path(repo.git.rev_parse("--show-toplevel")).resolve()
            logger.debug("Using Git repository root (GitPython): %s", git_root)
            return git_root
        except git.InvalidGitRepositoryError:
            logger.debug("Not inside a Git repository")
        except Exception as exc:  # Catch any git-related error
            logger.debug("GitPython error: %s", exc)

    # -- 3b. Fallback - manually search for .git directory ----------
    current = cwd
    for _ in range(max_depth):
        if (current / ".git").is_dir():
            logger.debug("Found Git repository root by .git directory: %s", current)
            return current

        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # -- 4. Fallback to current working directory ------------------
    logger.warning(
        "Could not determine project root - using current directory: %s. "
        "Set PAMOLA_PROJECT_ROOT or create .pamolaProject marker file.", cwd
    )
    return cwd


def substitute_variables(config_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform variable substitution in configuration values using Jinja2.

    Args:
        config_data: Configuration dictionary to process
        context: Dictionary of variables for substitution

    Returns:
        Dict[str, Any]: Configuration with variables substituted
    """
    if not JINJA2_AVAILABLE:
        logger.warning("Jinja2 not available. Variable substitution will be skipped.")
        return config_data

    result = {}

    # Process config dictionary
    for key, value in config_data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = substitute_variables(value, context)
        elif isinstance(value, list):
            # Process lists
            result[key] = [
                # Process each item in the list
                substitute_item(item, context)
                for item in value
            ]
        elif isinstance(value, str):
            # Substitute variables in strings
            template = Template(value)
            try:
                result[key] = template.render(**context)
            except Exception as e:
                logger.warning(f"Error during variable substitution for '{key}': {e}")
                result[key] = value
        else:
            # Keep other types unchanged
            result[key] = value

    return result


def substitute_item(item: Any, context: Dict[str, Any]) -> Any:
    """
    Substitute variables in a single configuration item.

    Args:
        item: Item to process
        context: Dictionary of variables for substitution

    Returns:
        Processed item
    """
    if isinstance(item, dict):
        # Process dictionary
        return substitute_variables(item, context)
    elif isinstance(item, list):
        # Process list
        return [substitute_item(subitem, context) for subitem in item]
    elif isinstance(item, str):
        # Process string
        template = Template(item)
        try:
            return template.render(**context)
        except Exception as e:
            logger.warning(f"Error during variable substitution: {e}")
            return item
    else:
        # Keep other types unchanged
        return item


def load_project_config(
        project_root: Optional[Path] = None,
        config_filename: Optional[str] = None,
        use_cache: bool = True
) -> Dict[str, Any]:
    """
    Load the project configuration from a YAML file with JSON fallback.

    Args:
        project_root: Path to the project root directory. If None, it will be auto-detected.
        config_filename: Name of the configuration file. If None, uses default names.
        use_cache: Whether to use cached configuration if available.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary with defaults applied.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration file is invalid.
    """
    # Determine project root if not provided
    if project_root is None:
        project_root = find_project_root()

    # Use default filename if not specified
    if config_filename is None:
        config_filename = PROJECT_CONFIG_FILENAME

    # Resolve config paths
    config_path_yaml = project_root / "configs" / config_filename
    config_path_json = project_root / "configs" / config_filename.replace('.yaml', '.json')

    # Check cache first for YAML path
    cache_key_yaml = str(config_path_yaml)
    if use_cache and cache_key_yaml in _config_cache:
        logger.debug(f"Using cached project configuration from {config_path_yaml}")
        return _config_cache[cache_key_yaml]

    # Check cache for JSON path
    cache_key_json = str(config_path_json)
    if use_cache and cache_key_json in _config_cache:
        logger.debug(f"Using cached project configuration from {config_path_json}")
        return _config_cache[cache_key_json]

    # Try to load configuration
    config_data = None
    cache_key = None

    # Try YAML first
    if config_path_yaml.exists():
        try:
            # Load YAML configuration
            with open(config_path_yaml, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            logger.debug(f"Loaded project configuration from YAML: {config_path_yaml}")
            cache_key = cache_key_yaml
        except yaml.YAMLError as e:
            error_msg = f"Error parsing project configuration YAML: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    # If YAML loading didn't succeed, try JSON
    if config_data is None and config_path_json.exists():
        try:
            # Load JSON configuration
            with open(config_path_json, 'r', encoding='utf-8') as f:
                config_data = json.load(f) or {}

            logger.warning(f"Using legacy JSON configuration from: {config_path_json}. Consider migrating to YAML.")
            logger.debug(f"Loaded project configuration from JSON: {config_path_json}")
            cache_key = cache_key_json
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing project configuration JSON: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    # If neither YAML nor JSON was loaded successfully, raise error
    if config_data is None:
        raise FileNotFoundError(f"Project configuration file not found at {config_path_yaml} or {config_path_json}")

    # Apply default values to loaded configuration
    config_data = apply_default_values(config_data)

    # Prepare context for variable substitution
    context = {
        "project_root": str(project_root),
        **{k: v for k, v in config_data.items() if not isinstance(v, (dict, list))}
    }

    # Perform variable substitution
    processed_config = substitute_variables(config_data, context)

    # Cache the processed configuration
    if use_cache and cache_key:
        _config_cache[cache_key] = processed_config

    return processed_config


def apply_default_values(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values to the configuration where values are missing.

    Args:
        config_data: Original configuration dictionary

    Returns:
        Dict[str, Any]: Configuration with defaults applied
    """
    # Define default values for critical sections
    defaults = {
        "directory_structure": {
            "raw": "raw",
            "processed": "processed",
            "reports": "reports",
            "logs": "logs",
            "configs": "configs"
        },
        "task_dir_suffixes": [
            "input",
            "output",
            "temp",
            "logs",
            "dictionaries"
        ],
        "logging": {
            "level": "INFO",
        },
        "performance": {
            "chunk_size": 100000,
            "default_encoding": "utf-8",
            "default_delimiter": ",",
            "default_quotechar": "\"",  # Added missing parameter
            "memory_limit_mb": 1000,
            "use_dask": False,  
            "npartitions": 4
        },
        "encryption": {
            "use_encryption": False,
            "encryption_mode": "none",
            "key_path": None
        },
        "task_defaults": {
            "continue_on_error": True,
            "parallel_processes": 4,
            "use_vectorization": False
        }
    }

    # Copy configuration to avoid modifying the original
    result = config_data.copy()

    # Apply defaults for sections
    for section, section_defaults in defaults.items():
        if section not in result:
            result[section] = {}

        if isinstance(section_defaults, dict):
            # Apply defaults for keys in this section
            for key, default_value in section_defaults.items():
                if key not in result[section]:
                    result[section][key] = default_value
        else:
            # Handle non-dict defaults (like lists)
            if not result[section]:
                result[section] = section_defaults

    # Ensure data repository is set
    if "data_repository" not in result:
        # Default to DATA directory in project root
        result["data_repository"] = "DATA"

    return result


def clear_config_cache() -> None:
    """Clear the configuration cache."""
    global _config_cache
    _config_cache = {}
    logger.debug("Project configuration cache cleared")


def get_project_paths(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get standard project paths from configuration.

    Args:
        config: Project configuration dictionary
        project_root: Project root path (auto-detected if None)

    Returns:
        Dict[str, Path]: Dictionary of standard project paths
    """
    if project_root is None:
        project_root = find_project_root()

    # Get directory structure from config
    dir_structure = config.get("directory_structure", {})

    # Get data repository
    data_repo = config.get("data_repository", "DATA")
    if not isinstance(data_repo, Path):
        if os.path.isabs(data_repo):
            data_repo = Path(data_repo)
        else:
            data_repo = project_root / data_repo

    # Build standard paths
    paths = {
        "project_root": project_root,
        "data_repository": data_repo,
        "configs_dir": project_root / dir_structure.get("configs", "configs"),
        "logs_dir": project_root / dir_structure.get("logs", "logs"),
        "raw_dir": data_repo / dir_structure.get("raw", "raw"),
        "processed_dir": data_repo / dir_structure.get("processed", "processed"),
        "reports_dir": data_repo / dir_structure.get("reports", "reports")
    }

    return paths


def save_project_config(
        config_data: Dict[str, Any],
        project_root: Optional[Path] = None,
        format: str = "yaml"
) -> Path:
    """
    Save the project configuration to a file.

    Args:
        config_data: Configuration dictionary to save
        project_root: Project root path (auto-detected if None)
        format: Format to save in - "yaml" or "json"

    Returns:
        Path to saved configuration file
    """
    if project_root is None:
        project_root = find_project_root()

    # Determine file extension based on format
    extension = ".yaml" if format.lower() == "yaml" else ".json"
    filename = "prj_config" + extension

    # Ensure configs directory exists
    config_dir = project_root / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)

    # Full path to config file
    config_path = config_dir / filename

    try:
        # Save in appropriate format
        if format.lower() == "yaml":
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                logger.error(f"Error saving YAML configuration: {e}")
                raise
        else:
            # Use JSON format
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving JSON configuration: {e}")
                raise

        logger.info(f"Project configuration saved to {config_path}")

        # Clear cache for this config
        if str(config_path) in _config_cache:
            del _config_cache[str(config_path)]

        return config_path
    except Exception as e:
        logger.error(f"Failed to save project configuration: {e}")
        raise


def is_valid_project_root(path: Path) -> bool:
    """
    Check if a path is a valid project root.

    Args:
        path: Path to check

    Returns:
        True if the path is a valid project root, False otherwise
    """
    # Check if path exists and is a directory
    if not path.exists() or not path.is_dir():
        return False

    # Check for marker file
    if (path / ".pamolaProject").exists():
        return True

    # Check for configs directory with project config
    yaml_config = path / "configs" / PROJECT_CONFIG_FILENAME
    json_config = path / "configs" / PROJECT_CONFIG_LEGACY_FILENAME

    return yaml_config.exists() or json_config.exists()


def create_default_project_structure(root_path: Path, data_path: Optional[Path] = None) -> Dict[str, Path]:
    """
    Create a default project structure at the specified location.

    This creates the standard directories and configuration files for a new project.

    Args:
        root_path: Root path for the new project
        data_path: Path for data repository (defaults to DATA under root_path)

    Returns:
        Dictionary of created paths
    """
    # Ensure root path exists
    root_path.mkdir(exist_ok=True, parents=True)

    # Create marker file
    (root_path / ".pamolaProject").touch()

    # Set data path
    if data_path is None:
        data_path = root_path / "DATA"

    # Create standard directories
    dirs = {
        "configs": root_path / "configs",
        "logs": root_path / "logs",
        "core": root_path / "core",
        "scripts": root_path / "scripts",
        "data": data_path,
        "raw": data_path / "raw",
        "processed": data_path / "processed",
        "reports": data_path / "reports"
    }

    # Create each directory
    for name, path in dirs.items():
        path.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Created directory: {path}")

    # Create default project config
    default_config = {
        "project_root": str(root_path),
        "data_repository": str(data_path),
        "directory_structure": {
            "raw": "raw",
            "processed": "processed",
            "reports": "reports",
            "logs": "logs",
            "configs": "configs"
        },
        "task_dir_suffixes": [
            "input",
            "output",
            "temp",
            "logs",
            "dictionaries"
        ],
        "logging": {
            "level": "INFO"
        },
        "performance": {
            "chunk_size": 100000,
            "default_encoding": "utf-8",
            "default_delimiter": ",",
            "default_quotechar": "\"",
            "memory_limit_mb": 1000,
            "use_dask": False
        },
        "encryption": {
            "use_encryption": False,
            "encryption_mode": "none",
            "key_path": None
        },
        "task_defaults": {
            "continue_on_error": True,
            "parallel_processes": 4
        }
    }

    # Save config in YAML format
    config_path = save_project_config(default_config, root_path, "yaml")
    logger.info(f"Created default project configuration at {config_path}")

    return dirs


def get_recursive_variables(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all variables from configuration data that can be used in substitution.

    This extracts both top-level variables and those inside sections, flattening
    them for use in templates.

    Args:
        config_data: Configuration dictionary

    Returns:
        Dictionary of flattened variables for substitution
    """
    result = {}

    # First, collect top-level variables
    for key, value in config_data.items():
        if not isinstance(value, (dict, list)):
            result[key] = value

    # Then, collect variables from sections with prefixes
    for section_name, section_data in config_data.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if not isinstance(value, (dict, list)):
                    # Add with section prefix
                    result[f"{section_name}.{key}"] = value

    return result