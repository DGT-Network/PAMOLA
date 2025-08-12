# Visualization Module Refactoring Specification

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for refactoring the PAMOLA visualization subsystem to address concurrency issues. The current implementation experiences state conflicts when multiple visualization operations are executed concurrently, leading to inconsistent outputs or failures.

### 1.2 Scope
The refactoring will focus on ensuring thread safety and context isolation within the PAMOLA visualization API without changing its external interfaces. It includes modifying pamola core components to eliminate global state dependencies while preserving backward compatibility.

#### In Scope:
- Refactoring theme and backend management for thread safety
- Isolating visualization context between concurrent executions
- Ensuring compatibility with existing visualization types and configurations
- Implementation of context-aware state management

#### Out of Scope:
- Changes to the public visualization API signatures
- Modifications to actual visualization algorithms or styles
- Creation of new visualization types
- Performance optimizations unrelated to concurrency

## 2. Glossary

- **Backend**: The underlying visualization library (Plotly or Matplotlib) used to render graphics
- **Context**: The execution environment for a visualization operation, including theme and backend settings
- **ContextVar**: Python's context variable mechanism for isolating state within execution contexts
- **FigureFactory**: Factory class that creates specific visualization implementations
- **GIL**: Global Interpreter Lock in Python
- **Theme**: A collection of visual styling parameters (colors, fonts, etc.) for visualizations
- **Thread-safe**: Code that functions correctly during simultaneous execution by multiple threads

## 3. References & Dependencies

### 3.1 Dependencies
- Python ≥ 3.7 (required for `contextvars` module)
- Plotly (primary visualization backend)
- Matplotlib (secondary/fallback visualization backend)
- NumPy and Pandas (data handling)
- PIL/Pillow (image processing)

### 3.2 References
- [Python contextvars documentation](https://docs.python.org/3/library/contextvars.html)
- [Plotly API reference](https://plotly.com/python-api-reference/)
- [Matplotlib API reference](https://matplotlib.org/stable/api/index.html)
- [PAMOLA code style guide] (internal reference)
- [PEP 567 – Context Variables](https://peps.python.org/pep-0567/)

## 4. Assumptions & Constraints

### 4.1 Assumptions
- The application may execute visualizations concurrently from multiple threads
- External visualization libraries (Plotly, Matplotlib) remain unchanged
- Users expect consistent behavior regardless of concurrency
- Data validation is handled prior to visualization

### 4.2 Constraints
- Must maintain backward compatibility with existing code
- Must work on all platforms where PAMOLA is supported (Windows, Linux, macOS)
- Visualization operations should not increase in execution time by more than 5%
- Memory usage should not increase by more than 10%

## 5. External Interface Requirements

### 5.1 Input Data
- All visualization functions must continue to accept the same data formats:
  - Pandas DataFrames 
  - Dictionaries (various formats as specified per function)
  - NumPy arrays
  - Primitive lists
- All should handle empty or null input data gracefully

### 5.2 Output Data
- Functions must maintain current return value types:
  - Paths to saved files (as strings)
  - Error messages (as strings) for failures
  - Figure objects when requested

### 5.3 Error Handling
- Must log errors with appropriate severity levels
- Must return standardized error messages that include:
  - Type of error (data validation, rendering, saving)
  - Context of the failure (theme, backend, visualization type)
  - Suggested remediation when applicable

## 6. Detailed Functional Requirements

### 6.1 Context Management

#### 6.1.1 Theme Context
- Replace global theme state with context-isolated state
- Themes must be applied only within the context of the current visualization operation
- Theme changes must not affect other concurrent visualization operations

#### 6.1.2 Backend Context
- Replace global backend state with context-isolated state
- Default to "plotly" when backend not explicitly specified
- Backend selection must not affect other concurrent visualization operations

#### 6.1.3 Initialization
- Module imports must not have side effects that modify global state
- Remove global backend initialization from module import time

### 6.2 Visualization Functions

For all visualization functions (`create_bar_plot`, `create_histogram`, etc.):

- Must apply theme and backend settings locally within the function scope
- Must handle empty or invalid input data by:
  - Creating an appropriate "empty" visualization
  - Including a descriptive error message within the visualization
  - Returning an appropriate error message or placeholder path
- Must isolate their execution context from other concurrent operations
- Must release all resources properly after use
- Must maintain identical parameter signatures

### 6.3 Matplotlib Handling

- Must use a headless backend (e.g., 'Agg') for file-only output
- Must properly manage figure creation and cleanup
- Must ensure figures are not shared between concurrent operations

## 7. Non-Functional Requirements

### 7.1 Performance
- Visualization operations must complete within 1.5x of the current execution time
- Memory usage must not exceed 1.2x of the current implementation
- Must support at least 10 concurrent visualization operations

### 7.2 Reliability
- Success rate must be ≥ 99.5% under normal operating conditions
- Must gracefully handle resource limitations with informative error messages
- Must ensure consistent visual output regardless of concurrency

### 7.3 Maintainability
- All modified code must follow PAMOLA coding standards
- Test coverage of modified code must be ≥ 90%
- All functions must include updated docstrings in English
- Implementation must include detailed comments explaining concurrency handling

### 7.4 Logging
- Must log context transitions at DEBUG level
- Must log theme and backend selections at DEBUG level
- Must log visualization creation/completion at INFO level
- Must log errors and exceptions at ERROR level with sufficient context

## 8. Code Structure Requirements

### 8.1 Module Header
All modified files must include the standard PAMOLA header:

```python
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Visualization System
Description: Thread-safe visualization capabilities for data analysis and privacy metrics
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides a comprehensive visualization system for generating standardized data visualizations
with thread-safe operation for concurrent execution contexts. It implements multiple visualization types
through a unified interface with configurable backends and themes.

Key features:
1. Context-isolated visualization state for thread-safe parallel execution
2. Dual-backend architecture with Plotly (primary) and Matplotlib (fallback)
3. Configurable theming system with context-specific application
4. Robust error handling with appropriate fallbacks
5. Memory-efficient operation with explicit resource cleanup
6. Support for multiple visualization types (bar, line, scatter, histogram, etc.)
7. Standardized saving mechanisms with consistent filepath handling
8. Graceful handling of empty or invalid input data

Implementation follows the PAMOLA.CORE framework with standardized interfaces for data processing,
visualization generation, and output management while ensuring concurrent operations do not interfere
with each other through proper context isolation.
"""
```

### 8.2 Documentation
- All public functions must have clear docstrings with parameters, return values, and examples
- All changes to the concurrency model must be documented in comments
- Complex operations must include inline comments explaining their purpose

### 8.3 Testing Requirements
- Must include tests for concurrent visualization operations
- Must include tests for theme and backend isolation
- Must include tests for all error handling cases
- Must include performance comparison tests

## 9. Implementation Recommendations

### 9.1 Matplotlib Configuration
For file-only output, configure Matplotlib to use the 'Agg' backend:

```python
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
```

### 9.2 Context Implementation
Implement a context manager for visualization operations:

```python
@contextmanager
def visualization_context(theme=None, backend=None):
    # Implementation details...
    try:
        yield
    finally:
        # Cleanup resources...
```

### 9.3 Error Handling Pattern
Standardize error handling across visualization functions:

```python
try:
    # Visualization code...
except Exception as e:
    logger.error(f"Visualization error: {e}", exc_info=True)
    return f"Error creating visualization: {str(e)}"
```

## 10. Testing Strategy

### 10.1 Unit Tests
- Test each context-aware function individually
- Test theme and backend isolation
- Test proper error handling

### 10.2 Integration Tests
- Test multiple visualization types in sequence
- Test with various data types and edge cases

### 10.3 Concurrency Tests
- Test parallel visualization creation with different themes
- Test parallel visualization creation with different backends
- Test high concurrency scenarios (10+ simultaneous operations)

### 10.4 Performance Tests
- Compare execution time before and after refactoring
- Measure memory usage before and after refactoring
- Test resource cleanup after visualization operations

## 11 # Priority Modules for Refactoring

Based on the provided information and the pamola core issue of thread safety in visualization, the following modules should be prioritized for refactoring in this order:

## Phase 1: Core Infrastructure (Highest Priority)

1. **pamola_core/utils/vis_helpers/base.py**
    
    - Contains global backend settings and pamola core infrastructure
    - Foundational for all other modules
    - Needs conversion to `contextvars` for thread-safe backend management
2. **pamola_core/utils/vis_helpers/theme.py**
    
    - Contains global theme settings
    - Used across all visualization types
    - Needs conversion to `contextvars` for thread-safe theme management
3. **pamola_core/utils/visualization.py**
    
    - Main entry point for visualization API
    - Initializes global state during import
    - Needs refactoring to eliminate side effects and use context-aware helpers

## Phase 2: High-Usage Visualization Types

4. **pamola_core/utils/vis_helpers/bar_plots.py**
    
    - Common visualization type
    - Demonstrates pattern for Plotly/Matplotlib dual backend
5. **pamola_core/utils/vis_helpers/line_plots.py**
    
    - Another common visualization type
    - Tests different context patterns
6. **pamola_core/utils/vis_helpers/boxplot.py**
    
    - More complex visualization with multiple data handling paths
    - Good test case for context isolation

## Phase 3: Additional Core Components

7. **pamola_core/utils/vis_helpers/**init**.py**
    
    - May contain imports that trigger side effects
    - Entry point for the helpers package
8. **pamola_core/utils/vis_helpers/scatter_plots.py**
    
    - Interactive elements with potential state issues
9. **pamola_core/utils/vis_helpers/histograms.py**
    
    - Statistical visualization with binning logic

## Phase 4: Specialized Visualizations

10. **pamola_core/utils/vis_helpers/heatmap.py**
11. **pamola_core/utils/vis_helpers/pie_charts.py**
12. **pamola_core/utils/vis_helpers/combined_charts.py**
13. **pamola_core/utils/vis_helpers/cor_matrix.py**
14. **pamola_core/utils/vis_helpers/cor_pair.py**
15. **pamola_core/utils/vis_helpers/cor_utils.py**
16. **pamola_core/utils/vis_helpers/spider_charts.py**
17. **pamola_core/utils/vis_helpers/word_clouds.py**

## Implementation Strategy

1. Create a new module for context management first (e.g., `pamola_core/utils/vis_helpers/context.py`)
2. Modify `base.py` and `theme.py` to use context variables
3. Update `visualization.py` to leverage the new context-aware approach
4. Progressively update individual visualization type implementations
5. Add comprehensive tests focusing on concurrent execution after each phase

This phased approach allows for incremental testing and validation, ensuring that the pamola core infrastructure works correctly before proceeding to more specialized visualizations.