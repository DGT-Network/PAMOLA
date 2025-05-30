# PAMOLA CORE Project Documentation Guide

This guide explains the documentation system for the PAMOLA.CORE project, including its structure, how to maintain and update it, and best practices for documenting new modules.

## What is MkDocs?

[MkDocs](https://www.mkdocs.org/) is a fast, simple, and static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file.

Key features of MkDocs:
- **Markdown-based**: Write documentation in simple Markdown syntax
- **Configurable**: Easy customization with a single YAML configuration file
- **Preview**: Built-in development server for previewing documentation as you write
- **Themes**: Multiple themes available, including ReadTheDocs, Material, and others
- **Search**: Full-text search across your documentation
- **Static output**: Generate static HTML files for deployment on any web server

## ReadTheDocs Theme

The PAMOLA.CORE project uses the [ReadTheDocs theme](https://mkdocs.readthedocs.io/), a clean, responsive theme designed specifically for technical documentation.

Key features of the ReadTheDocs theme:
- **Collapsible navigation**: Hierarchical navigation with expandable sections
- **Mobile-friendly**: Responsive design that works well on all devices
- **Search integration**: Built-in search functionality
- **Table of contents**: Automatic generation of table of contents
- **Code highlighting**: Syntax highlighting for code blocks
- **Admonitions**: Support for note, warning, and other callout blocks

## Project Structure

The PAMOLA CORE documentation project is organized as follows:

```
pamola_core/
├── docs/                      # Root documentation directory
│   ├── en/                    # English documentation
│   │   ├── index.md           # Main English page
│   │   ├── profiling/         # Profiling module docs
│   │   │   ├── null_analysis.md
│   │   │   └── ...
│   │   ├── utils/             # Utils module docs
│   │   └── tests/             # Tests docs
│   ├── ru/                    # Russian documentation
│   │   ├── index.md           # Main Russian page
│   │   ├── profiling/         # Russian profiling docs
│   │   └── ...
│   └── common/                # Shared resources
│       ├── assets/            # Images, CSS, etc.
│       └── examples/          # Code examples
├── theme_overrides/           # Custom theme adjustments
├── mkdocs.en.yml              # English config
├── mkdocs.ru.yml              # Russian config
├── index.html                 # Main landing page
├── serve-docs.bat             # Windows server script
└── serve-docs.sh              # Linux/Mac server script
```

## Multilingual Documentation System

The PAMOLA.CORE project uses a multilingual documentation system with separate builds for each language:

1. **Separate configuration files**:
   - `mkdocs.en.yml` for English
   - `mkdocs.ru.yml` for Russian

2. **Separate language directories**:
   - `/docs/en/` for English content
   - `/docs/ru/` for Russian content
   
3. **Shared resources**:
   - `/docs/common/` for assets and examples used in both languages
   
4. **Custom main page**:
   - `index.html` serves as a language selector landing page

## How to Update Documentation

### Adding or Updating Module Documentation

1. **Create markdown files** in both language directories:
   ```
   docs/en/your_module/your_feature.md
   docs/ru/your_module/your_feature.md
   ```

2. **Update navigation** in both configuration files:
   ```yaml
   # In mkdocs.en.yml
   nav:
     - Home: index.md
     - ...
     - Your Module:
       - Feature: your_module/your_feature.md
   
   # In mkdocs.ru.yml (with translated titles)
   nav:
     - Главная: index.md
     - ...
     - Ваш Модуль:
       - Функция: your_module/your_feature.md
   ```

3. **Build and test** the documentation:
   ```
   # Run the documentation server
   ./serve-docs.sh  # or serve-docs.bat on Windows
   ```

### Working with Shared Resources

1. **Add images** to the common assets directory:
   ```
   docs/common/assets/images/your_image.png
   ```

2. **Reference images** in your Markdown files:
   ```markdown
   ![Image description](../../common/assets/images/your_image.png)
   ```

3. **Add code examples** to the examples directory:
   ```
   docs/common/examples/your_example.py
   ```

4. **Reference code examples** using the include syntax:
   ```markdown
   ```python
   --8<-- "docs/common/examples/your_example.py"
   ```
   ```

## Documentation Module Template

When documenting a module, follow this structure as demonstrated in the `null_analysis.md` module:

### 1. Module Header

Start with a title and brief introduction:

```markdown
# Module Name

## Overview

Brief description of what the module does and its role in the PAMOLA.CORE project.
```

### 2. Purpose Section

Explain why this module exists:

```markdown
## Purpose

The module serves these key functions:

- Function 1
- Function 2 
- ...
```

### 3. Key Features

List and explain major features:

```markdown
## Key Features

### 1. Feature Name

Detailed explanation of the feature.

### 2. Another Feature

Example code showing usage:
```python
# Example code
```
```

### 4. Class/Function Documentation

Document main classes/functions:

```markdown
## Class: YourClass

Description of the class and its role.

```python
@Decorator
class YourClass(BaseClass):
    """
    Class docstring.
    """
    # ...implementation...
```
```

### 5. Usage Examples

Provide complete usage examples:

```markdown
## Usage Examples

### Basic Usage

```python
from pamola_coremodule import YourClass

# Create instance
instance = YourClass(param1, param2)

# Use methods
result = instance.method()
```

### Command Line Usage

```bash
python -m pamola_coremodule --param value
```
```

### 6. Output Examples

Show example outputs:

```markdown
## Output

### Example JSON Output

```json
{
  "result": "value",
  "metrics": {
    "accuracy": 0.95
  }
}
```
```

### 7. Integration Section

Explain how the module integrates with others:

```markdown
## Integration

This module integrates with:
- ModuleA for feature X
- ModuleB for feature Y
```

## Best Practices

1. **Update documentation alongside code changes**
   - Document new features as you implement them
   - Keep documentation in sync with code

2. **Use consistent formatting**
   - Follow the template structure
   - Use consistent heading levels
   - Maintain consistent code block formatting

3. **Write clear, concise explanations**
   - Use simple language
   - Explain complex concepts step by step
   - Provide examples for clarification

4. **Include working examples**
   - Make sure code examples run correctly
   - Use realistic, but simple examples

5. **Add visuals when appropriate**
   - Include diagrams for complex concepts
   - Add screenshots for UI components
   - Use flowcharts for processes

6. **Ensure multilingual consistency**
   - Keep both language versions in sync
   - Translate technical terms consistently
   - Maintain parallel structure in both languages

## Building and Serving Documentation

The project includes scripts to build and serve documentation:

- **Windows**: `serve-docs.bat`
- **Linux/Mac**: `serve-docs.sh`

These scripts offer four options:
1. Start English documentation server
2. Start Russian documentation server
3. Build both and start main documentation server
4. Build documentation only

For development, use options 1 or 2 to preview changes in a specific language.
For testing the complete documentation site, use option 3.
For deployment, use option 4 to generate static files only.

## Adding a New Language

To add a new language (e.g., Spanish):

1. **Create language directory**:
   ```
   mkdir -p docs/es
   ```

2. **Copy and translate files** from the English version
   
3. **Create configuration file**:
   ```
   cp mkdocs.en.yml mkdocs.es.yml
   ```
   
4. **Update configuration** with Spanish settings
   
5. **Add language to landing page**:
   ```html
   <a href="es/" class="lang-btn">Español</a>
   ```
   
6. **Update build scripts** to include the new language

## Conclusion

Following these guidelines will help maintain a high-quality, consistent, and comprehensive documentation for the PAMOLA.CORE project. Remember that good documentation is essential for user adoption, developer onboarding, and long-term project maintenance.

If you have questions or need assistance with documentation, please contact the documentation maintainer.