"""
PAMOLA Task Template Generator
-----------------------------
Utility script to generate new PAMOLA tasks from templates.

This script creates a new task file from a template, customizing it with
the specified task ID, type, and description. It also ensures that the
necessary directory structure exists.

Usage:
    python create_task.py --template simple --task_id t_1X_my_task --task_type profiling --description "My task description"
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path
import re

# Default template paths
TEMPLATE_DIR = Path(__file__).parent
SIMPLE_TEMPLATE = TEMPLATE_DIR / "simple_task_template.py"
SECURE_TEMPLATE = TEMPLATE_DIR / "secure_task_template.py"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a new PAMOLA task from template")

    parser.add_argument("--template", choices=["simple", "secure"], default="simple",
                        help="Template to use (simple or secure)")
    parser.add_argument("--task_id", required=True,
                        help="Task ID (e.g., t_1A_profile)")
    parser.add_argument("--task_type", required=True,
                        help="Task type (e.g., profiling, anonymization)")
    parser.add_argument("--description", required=True,
                        help="Task description")
    parser.add_argument("--author", default=os.environ.get("USERNAME", os.environ.get("USER", "PAMOLA User")),
                        help="Author name")
    parser.add_argument("--output_dir", default=".",
                        help="Directory where the task file will be created")
    parser.add_argument("--create_directories", action="store_true", default=True,
                        help="Create the necessary directory structure")
    parser.add_argument("--data_repository", default="DATA",
                        help="Path to the data repository")

    return parser.parse_args()


def create_directory_structure(project_root, data_repository, task_id):
    """
    Create the necessary directory structure for the task.

    Args:
        project_root: Path to the project root
        data_repository: Path to the data repository
        task_id: Task ID

    Returns:
        Dictionary with paths to created directories
    """
    project_root = Path(project_root)

    # Convert data_repository to absolute path if it's not already
    data_repo_path = Path(data_repository)
    if not data_repo_path.is_absolute():
        data_repo_path = project_root / data_repo_path

    # Define directories to create
    directories = {
        "configs": project_root / "configs",
        "logs": project_root / "logs",
        "data_repo": data_repo_path,
        "raw": data_repo_path / "raw",
        "processed": data_repo_path / "processed",
        "reports": data_repo_path / "reports",
        "task_dir": data_repo_path / "processed" / task_id,
        "task_output": data_repo_path / "processed" / task_id / "output",
        "task_dictionaries": data_repo_path / "processed" / task_id / "dictionaries",
        "task_visualizations": data_repo_path / "processed" / task_id / "visualizations",
    }

    # Create directories
    for name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

    return directories


def generate_task_file(template_path, output_path, task_id, task_type, description, author):
    """
    Generate a new task file from the template.

    Args:
        template_path: Path to the template file
        output_path: Path where the new file will be created
        task_id: Task ID
        task_type: Task type
        description: Task description
        author: Author name

    Returns:
        Path to the created file
    """
    # Read template content
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Determine class name from task_id
    if task_id.startswith('t_'):
        # Convert t_1A_profile to T1AProfile
        class_name_parts = task_id.split('_')
        if len(class_name_parts) >= 2:
            class_name = ''.join(part.capitalize() for part in class_name_parts)
        else:
            class_name = task_id.replace('_', '').capitalize() + 'Task'
    else:
        # Convert my_task to MyTask
        class_name = ''.join(part.capitalize() for part in task_id.split('_'))

    # Ensure class name ends with 'Task'
    if not class_name.endswith('Task'):
        class_name += 'Task'

    # Set class name in template
    if 'secure' in template_path.name.lower():
        content = re.sub(r'class MySecureTask', f'class {class_name}', content)
    else:
        content = re.sub(r'class MyTask', f'class {class_name}', content)

    # Replace placeholder values
    replacements = {
        'task_id="t_1X_my_task"': f'task_id="{task_id}"',
        'task_id="t_1X_my_secure_task"': f'task_id="{task_id}"',
        'task_type="template"': f'task_type="{task_type}"',
        'task_type="secure_template"': f'task_type="{task_type}"',
        'description="Example task template for standard processing"': f'description="{description}"',
        'description="Example secure task template with encryption"': f'description="{description}"',
        '@author: Your Name': f'@author: {author}',
        'MyTask()': f'{class_name}()',
        'MySecureTask()': f'{class_name}()',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    # Add creation timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content = content.replace('@version: 1.0.0', f'@version: 1.0.0\n    @created: {timestamp}')

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return output_path


def create_basic_config(project_root, task_id, task_type, description):
    """
    Create a basic configuration file for the task.

    Args:
        project_root: Path to the project root
        task_id: Task ID
        task_type: Task type
        description: Task description

    Returns:
        Path to the created config file
    """
    import json

    config_dir = Path(project_root) / "configs"
    os.makedirs(config_dir, exist_ok=True)

    config_path = config_dir / f"{task_id}.json"

    # Check if project config exists
    project_config_path = config_dir / "prj_config.json"
    if not project_config_path.exists():
        # Create basic project config
        project_config = {
            "data_repository": "DATA",
            "log_level": "INFO",
            "directory_structure": {
                "raw": "raw",
                "processed": "processed",
                "logs": "logs",
                "reports": "reports"
            },
            "tasks": {}
        }

        with open(project_config_path, 'w', encoding='utf-8') as f:
            json.dump(project_config, f, indent=2)

        print(f"✓ Created project configuration: {project_config_path}")

    # Create task config
    task_config = {
        "task_type": task_type,
        "description": description,
        "continue_on_error": False,
        "use_encryption": False,
        "encryption_mode": "none",
        "dependencies": []
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(task_config, f, indent=2)

    print(f"✓ Created task configuration: {config_path}")

    return config_path


def main():
    """Main entry point."""
    args = parse_arguments()

    # Determine template path
    if args.template == "simple":
        template_path = SIMPLE_TEMPLATE
    else:
        template_path = SECURE_TEMPLATE

    # Check if template exists
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        sys.exit(1)

    # Determine output path
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.task_id}.py"

    # Check if output file already exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    # Create directory structure if requested
    if args.create_directories:
        create_directory_structure(output_dir, args.data_repository, args.task_id)

    # Generate task file
    try:
        task_file = generate_task_file(
            template_path,
            output_path,
            args.task_id,
            args.task_type,
            args.description,
            args.author
        )
        print(f"✓ Generated task file: {task_file}")

        # Create basic config
        config_file = create_basic_config(
            output_dir,
            args.task_id,
            args.task_type,
            args.description
        )

        print("\nTask created successfully!")
        print(f"You can run it with: python {output_path}")

    except Exception as e:
        print(f"Error generating task file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()