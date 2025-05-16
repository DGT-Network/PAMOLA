# MkDocs Navigation Generator

## Overview

The MkDocs Navigation Generator is a Python script designed to automate and synchronize navigation configurations for multilingual documentation projects using MkDocs. The script provides intelligent management of navigation structures across different language versions of documentation.

## Key Features

- Automatic navigation structure generation
- Bilingual (English and Russian) documentation support
- Intelligent file and directory path tracking
- Preservation of manually added navigation entries
- Automatic commenting of top-level sections in Russian documentation

## Script Purpose

In multilingual documentation projects, maintaining consistent navigation across different language versions can be challenging. This script solves several key problems:

1. Automatically generate navigation structure based on existing markdown files
2. Ensure consistency between English and Russian documentation
3. Add helpful comments to Russian navigation to track original folder names
4. Remove non-existent paths
5. Preserve manually added navigation entries

## Algorithm

The script follows these main steps:

1. Scan documentation directories for markdown files
2. Generate a navigation structure based on file and directory hierarchy
3. Compare navigation structures between languages
4. Update MkDocs YAML configuration files

### Navigation Generation Process

- Recursively traverse documentation directories
- Create navigation entries for directories and files
- Match entries between English and Russian versions
- Add comments for top-level sections in Russian documentation
- Remove entries for non-existent paths

## Usage

### Prerequisites

- Python 3.7+
- PyYAML library
- MkDocs project structure with separate language directories

### Installation

```bash
pip install pyyaml
```

### Running the Script

Place the script in your project's `scripts` directory and run:

```bash
python update_mkdocs_nav.py
```

## Configuration Parameters

### Main Function Parameters

- `project_root`: Automatically detected root directory of the project
- `en_docs_path`: Path to English documentation directory
- `ru_docs_path`: Path to Russian documentation directory
- `en_yml_path`: Path to English MkDocs configuration file
- `ru_yml_path`: Path to Russian MkDocs configuration file

### Customization Options

- Modify `extract_original_folder_name()` to change folder name extraction logic
- Adjust `process_nav_item()` to customize navigation processing rules

## Example Project Structure

```
project_root/
│
├── docs/
│   ├── en/
│   │   ├── index.md
│   │   ├── section1/
│   │   └── section2/
│   │
│   └── ru/
│       ├── index.md
│       ├── section1/
│       └── section2/
│
├── mkdocs.en.yml
├── mkdocs.ru.yml
└── scripts/
    └── update_mkdocs_nav.py
```

## Limitations and Considerations

- Requires consistent file structure across language versions
- Assumes markdown (.md) files for documentation
- Does not support complex nested navigation beyond two levels

## Troubleshooting

- Ensure PyYAML is installed
- Check file permissions
- Verify documentation directory structure
- Use verbose logging for detailed information

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the project repository.

## License

[Specify your project's license]

## Author

[Your Name/Organization]