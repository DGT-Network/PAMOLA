"""
PAMOLA.CORE - Configuration Module
----------------------------------------------
This module handles configuration loading and provides access to project settings.

Features:
- Multiple configuration sources (environment variables, config files)
- Support for development and production environments
- Default settings with override capabilities
- Configuration validation

(C) 2025 BDA

Author: V.Khvatov
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Configure module logger
logger = logging.getLogger("pamola_core.configs")

# Default configuration
DEFAULT_CONFIG = {
    "data_repository": None,  # Will be auto-detected
    "directory_structure": {
        "raw": "raw",
        "processed": "processed",
        "logs": "logs",
        "configs": "configs"
    },
    "logging": {
        "level": "INFO",
        "file": "pamola_processing.log",
        "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    },
    "performance": {
        "chunk_size": 100000,
        "default_encoding": "utf-8",
        "default_delimiter": ",",
        "memory_limit_mb": 1000
    }
}

# Config singleton
_config = None


def get_config_file_paths() -> List[Path]:
    """
    Get list of potential config file paths in order of priority.

    Returns:
    --------
    List[Path]
        List of potential config file paths
    """
    paths = []  # Инициализация списка

    # 1. From environment variable
    if 'PAMOLA_CONFIG_PATH' in os.environ:
        paths.append(Path(os.environ['PAMOLA_CONFIG_PATH']))

    # 2. From project directory
    try:
        project_dir = Path(__file__).resolve().parent.parent.parent
        paths.append(project_dir / "configs" / "prj_config.json")
    except:
        pass

    # 3. From user home directory
    try:
        home_dir = Path.home()
        paths.append(home_dir / ".pamola_core" / "config.json")  # type: ignore
    except:
        pass

    # 4. From current directory
    paths.append(Path.cwd() / "prj_config.json")

    return paths


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or use defaults.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to configuration file

    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    global _config

    if _config is not None:
        return _config

    config = DEFAULT_CONFIG.copy()
    config_loaded = False

    # Try to load from specified path
    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                config = update_nested_dict(config, loaded_config)
                config_loaded = True
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # If no path specified or loading failed, try standard locations
    if not config_loaded:
        for path in get_config_file_paths():
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                        config = update_nested_dict(config, loaded_config)
                        config_loaded = True
                        logger.info(f"Loaded configuration from {path}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")

    # Auto-detect data repository if not specified
    if not config["data_repository"]:  # type: ignore
        # First check environment variable
        if 'PAMOLA_DATA_REPOSITORY' in os.environ:
            config["data_repository"] = os.environ['PAMOLA_DATA_REPOSITORY']  # type: ignore
            logger.info(f"Using data repository from environment: {config['data_repository']}")  # type: ignore
        else:
            # Try to detect based on current file location
            try:
                module_dir = Path(__file__).resolve().parent.parent.parent
                potential_data_dir = module_dir / "data"
                if potential_data_dir.exists() and potential_data_dir.is_dir():
                    config["data_repository"] = str(potential_data_dir)  # type: ignore
                    logger.info(f"Auto-detected data repository: {config['data_repository']}")  # type: ignore
                else:
                    logger.warning(f"Could not auto-detect data repository. Using current directory.")
                    config["data_repository"] = str(Path.cwd())  # type: ignore
            except Exception as e:
                logger.warning(f"Error detecting data repository: {e}. Using current directory.")
                config["data_repository"] = str(Path.cwd())  # type: ignore

    # Set log level from config
    if "logging" in config and "level" in config["logging"]:
        level_name = config["logging"]["level"].upper()
        try:
            level = getattr(logging, level_name)
            logger.setLevel(level)
        except AttributeError:
            logger.warning(f"Invalid log level: {level_name}. Using INFO.")

    _config = config
    return config


def update_nested_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a nested dictionary with values from another dictionary.

    Parameters:
    -----------
    d1 : Dict[str, Any]
        Base dictionary to update
    d2 : Dict[str, Any]
        Dictionary with update values

    Returns:
    --------
    Dict[str, Any]
        Updated dictionary
    """
    result = d1.copy()
    for k, v in d2.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = update_nested_dict(result[k], v)
        else:
            result[k] = v
    return result


def get_config() -> Dict[str, Any]:
    """
    Get the configuration.

    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
        if _config is None:  # Если по какой-то причине load_config вернул None
            _config = DEFAULT_CONFIG.copy()  # Используем значения по умолчанию
    return _config


def get_data_repository() -> Path:
    """
    Get the path to the data repository.

    Returns:
    --------
    Path
        Path to the data repository
    """
    config = get_config()
    return Path(config["data_repository"])


def set_data_repository(path: Union[str, Path]) -> None:
    """
    Set the data repository path explicitly.

    Parameters:
    -----------
    path : str or Path
        Path to the data repository
    """
    global _config
    if _config is None:
        _config = load_config()
    _config["data_repository"] = str(Path(path))
    logger.info(f"Data repository set to: {_config['data_repository']}")


def get_directory_structure() -> Dict[str, str]:
    """
    Get the directory structure configuration.

    Returns:
    --------
    Dict[str, str]
        Directory structure configuration
    """
    config = get_config()
    return config["directory_structure"]


def get_performance_settings() -> Dict[str, Any]:
    """
    Get performance-related settings.

    Returns:
    --------
    Dict[str, Any]
        Performance settings
    """
    config = get_config()
    return config["performance"]


def get_logging_settings() -> Dict[str, Any]:
    """
    Get logging-related settings.

    Returns:
    --------
    Dict[str, Any]
        Logging settings
    """
    config = get_config()
    return config["logging"]


def save_config(config_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Save current configuration to a file.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to save configuration file to

    Returns:
    --------
    Path
        Path to the saved configuration file
    """
    config = get_config()

    if config_path is None:
        # Use the first path from standard locations if none specified
        try:
            project_dir = Path(__file__).resolve().parent.parent.parent
            configs_dir = project_dir / "configs"
            if not configs_dir.exists():
                configs_dir.mkdir(parents=True, exist_ok=True)
            config_path = configs_dir / "prj_config.json"
        except Exception as e:
            logger.warning(f"Failed to determine config path: {e}. Using current directory.")
            config_path = Path.cwd() / "prj_config.json"

    config_path = Path(config_path)

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False) # type: ignore

    logger.info(f"Configuration saved to {config_path}")
    return config_path


# Initialize configuration when module is imported
if __name__ != "__main__":
    load_config()