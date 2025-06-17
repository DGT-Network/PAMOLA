"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Templates
Description: Templates for creating new PAMOLA tasks
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This package provides templates and utilities for creating new tasks
in the PAMOLA ecosystem, simplifying the process of creating standardized
task scripts that follow best practices.

Key features:
- Standard task template without encryption
- Secure task template with encryption
- Template generator utility
- Directory structure creation
"""

import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Package version
__version__ = "1.0.0"

# Path to template directory
TEMPLATE_DIR = Path(__file__).parent

# Template paths
SIMPLE_TEMPLATE_PATH = TEMPLATE_DIR / "simple_task_template.py"
SECURE_TEMPLATE_PATH = TEMPLATE_DIR / "secure_task_template.py"
GENERATOR_PATH = TEMPLATE_DIR / "task_template_generator.py"


def create_task(template: str = "simple",
                task_id: str = "",
                task_type: str = "",
                description: str = "",
                author: Optional[str] = None,
                output_dir: Union[str, Path] = ".",
                create_directories: bool = True,
                data_repository: str = "DATA") -> Path:
    """
    Create a new task file from a template.

    Args:
        template: Template to use ('simple' or 'secure')
        task_id: Task ID (e.g., t_1A_profile)
        task_type: Task type (e.g., profiling, anonymization)
        description: Task description
        author: Author name (defaults to current user)
        output_dir: Directory where the task file will be created
        create_directories: Whether to create the necessary directory structure
        data_repository: Path to the data repository

    Returns:
        Path to the created task file

    Example:
        ```python
        from pamola_core.utils.tasks.templates import create_task

        task_file = create_task(
            template="simple",
            task_id="t_1A_my_task",
            task_type="profiling",
            description="My custom profiling task",
            create_directories=True
        )

        print(f"Created task file: {task_file}")
        ```
    """
    # Import the generator module
    spec = importlib.util.spec_from_file_location("generator", GENERATOR_PATH)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    # Set default author if not provided
    if author is None:
        author = os.environ.get("USERNAME", os.environ.get("USER", "PAMOLA User"))

    # Create output directory path
    output_dir = Path(output_dir)

    # Create directory structure if requested
    if create_directories:
        generator.create_directory_structure(output_dir, data_repository, task_id)

    # Determine template path
    if template.lower() == "simple":
        template_path = SIMPLE_TEMPLATE_PATH
    else:
        template_path = SECURE_TEMPLATE_PATH

    # Generate task file
    output_path = output_dir / f"{task_id}.py"
    task_file = generator.generate_task_file(
        template_path,
        output_path,
        task_id,
        task_type,
        description,
        author
    )

    # Create basic config
    generator.create_basic_config(
        output_dir,
        task_id,
        task_type,
        description
    )

    return task_file