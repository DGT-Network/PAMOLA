# fake_data.commons.base Module Documentation

## Overview

The `base.py` module serves as the foundation for the entire `fake_data` package, providing abstract base classes, interfaces, and pamola core structures that define the architecture of the system. It establishes the contract for how fake data generation, mapping, and operations are implemented throughout the package. The module enforces consistency, enables extensibility, and promotes code reuse by defining standard interfaces that all concrete implementations must follow.

This architecture allows for a modular, maintainable system that can be easily extended with new data types, generation algorithms, and mapping strategies while maintaining consistent behavior across the package.

## Architecture

The module implements a layered architecture with clearly defined abstractions:

```
┌─────────────────────────────────────────────────────────────────┐
│                          base.py                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Core Exceptions │  │     Enums       │  │   Type Defs     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Abstract Base Classes                   │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │  │
│  │  │BaseGenerator│  │BaseMapper   │  │BaseOperation    │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │  │
│  │                                                           │  │
│  │              ┌─────────────────┐  ┌─────────────────┐     │  │
│  │              │FieldOperation   │  │MappingStore     │     │  │
│  │              └─────────────────┘  └─────────────────┘     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

This architectural approach provides a clear separation of concerns:

1. **BaseGenerator**: Responsible for producing synthetic data values
2. **BaseMapper**: Handles mapping between original and synthetic values
3. **BaseOperation/FieldOperation**: Provides standardized execution flow for data operations
4. **MappingStore**: Centralizes storage and retrieval of mappings
5. **Supporting structures**: Enums, exceptions, and type definitions

## Key Components

### Pamola Core Exceptions

| Exception | Purpose |
|-----------|---------|
| `FakeDataError` | Base exception for the package |
| `ValidationError` | Raised for data validation errors |
| `ResourceError` | Raised for resource-related issues (memory, etc.) |
| `MappingError` | Raised for mapping conflicts or issues |

### Enums

| Enum | Purpose |
|------|---------|
| `ResourceType` | Types of resources that can be estimated (MEMORY, TIME, CPU, DISK) |
| `NullStrategy` | Strategies for handling NULL values (PRESERVE, REPLACE, EXCLUDE, ERROR) |

### Abstract Base Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `BaseGenerator` | Template for all data generators | `generate()`, `generate_like()`, `analyze_value()`, `estimate_resources()` |
| `BaseMapper` | Template for mapping components | `map()`, `restore()`, `add_mapping()`, `check_conflicts()` |
| `BaseOperation` | Core interface for all operations | `execute()` |
| `FieldOperation` | Extended interface for field-specific operations | `process_batch()`, `preprocess_data()`, `postprocess_data()`, `handle_null_values()` |

### Concrete Utility Classes

| Class | Purpose | Key Methods |
|-------|---------|------------|
| `MappingStore` | Repository for mappings between original and synthetic values | `add_mapping()`, `get_mapping()`, `restore_original()`, `save()`, `load()` |

## Usage Examples

### Defining a Custom Generator

```python
from pamola_core.fake_data.commons.base import BaseGenerator, ResourceType


class NameGenerator(BaseGenerator):
    """Generator for fake names."""

    def __init__(self, dictionary, language="en"):
        self.dictionary = dictionary
        self.language = language

    def generate(self, count, **params):
        # Extract parameters with defaults
        gender = params.get("gender")
        region = params.get("region", "US")

        # Implementation specific to name generation
        result = []
        # ... generation logic here ...

        return result

    def generate_like(self, original_value, **params):
        # Analyze the original value
        properties = self.analyze_value(original_value)

        # Generate a similar value
        # ... implementation ...

        return generated_value

    def analyze_value(self, value):
        # Extract properties from the value
        properties = {
            "type": "name",
            "length": len(value),
            "has_space": " " in value,
            # ... other properties ...
        }

        return properties

    def estimate_resources(self, count, **params):
        # Estimate memory and time requirements
        return {
            ResourceType.MEMORY.value: count * 0.1,  # MB
            ResourceType.TIME.value: count * 0.005,  # seconds
        }
```

### Implementing a Custom Mapper

```python
from pamola_core.fake_data.commons.base import BaseMapper, MappingError


class OneToOneMapper(BaseMapper):
    """Maps original values to synthetic ones with 1:1 correspondence."""

    def __init__(self, fallback_generator=None):
        self.mapping = {}
        self.reverse_mapping = {}
        self.fallback_generator = fallback_generator

    def map(self, original_value, **params):
        # Check if value is already mapped
        if original_value in self.mapping:
            return self.mapping[original_value]

        # Generate new value if not mapped yet
        if self.fallback_generator:
            synthetic_value = self.fallback_generator.generate_like(
                original_value, **params
            )
            # Add to mapping store
            self.add_mapping(original_value, synthetic_value)
            return synthetic_value
        else:
            raise MappingError("Value not mapped and no fallback generator provided")

    def restore(self, synthetic_value):
        return self.reverse_mapping.get(synthetic_value)

    def add_mapping(self, original, synthetic, is_transitive=False):
        # Check for conflicts
        conflicts = self.check_conflicts(original, synthetic)
        if conflicts["has_conflicts"]:
            raise MappingError(f"Mapping conflict: {conflicts['conflict_type']}")

        # Add mappings
        self.mapping[original] = synthetic
        self.reverse_mapping[synthetic] = original

    def check_conflicts(self, original, synthetic):
        has_conflicts = False
        conflict_type = None
        affected_values = []

        # Check if original already mapped to different value
        if original in self.mapping and self.mapping[original] != synthetic:
            has_conflicts = True
            conflict_type = "original_already_mapped"
            affected_values.append(original)

        # Check if synthetic already mapped from different original
        if synthetic in self.reverse_mapping and self.reverse_mapping[synthetic] != original:
            has_conflicts = True
            conflict_type = "synthetic_already_used"
            affected_values.append(synthetic)

        return {
            "has_conflicts": has_conflicts,
            "conflict_type": conflict_type,
            "affected_values": affected_values
        }
```

### Creating a Field Operation

```python
from pamola_core.fake_data.commons.base import FieldOperation, NullStrategy
import pandas as pd


class NameGenerationOperation(FieldOperation):
    """Operation to generate fake names for a dataset."""

    name = "generate_names"
    description = "Generates synthetic names based on original values"

    def __init__(self, field_name, generator, mapper, **kwargs):
        super().__init__(field_name, **kwargs)
        self.generator = generator
        self.mapper = mapper

    def process_batch(self, batch):
        # Create a copy of the batch
        result = batch.copy()

        # Process each non-null value
        mask = result[self.field_name].notna()

        if self.mode == "REPLACE":
            # Replace original values with synthetic ones
            result.loc[mask, self.field_name] = result.loc[mask, self.field_name].apply(
                lambda x: self.mapper.map(x)
            )
        else:  # ENRICH mode
            # Add new column with synthetic values
            result[self.output_field_name] = result[self.field_name].apply(
                lambda x: self.mapper.map(x) if pd.notna(x) else None
            )

        return result
```

### Using MappingStore

```python
from pamola_core.fake_data.commons.base import MappingStore

# Create a mapping store
mapping_store = MappingStore()

# Add mappings for different fields
mapping_store.add_mapping("first_name", "John", "David")
mapping_store.add_mapping("first_name", "Mary", "Susan")
mapping_store.add_mapping("last_name", "Smith", "Jones")

# Retrieve mappings
synthetic_name = mapping_store.get_mapping("first_name", "John")  # Returns "David"

# Restore original values
original_name = mapping_store.restore_original("first_name", "David")  # Returns "John"

# Get all mappings for a field
first_name_mappings = mapping_store.get_all_mappings_for_field("first_name")
# Returns {"John": "David", "Mary": "Susan"}

# Check if a mapping is transitive
is_transitive = mapping_store.is_transitive("first_name", "John")  # Returns False
```

## Parameters and Return Values

### BaseGenerator

```python
class BaseGenerator:
    @abc.abstractmethod
    def generate(
        self,
        count: int,   # Number of values to generate
        **params      # Additional parameters (gender, region, language, seed, etc.)
    ) -> List[Any]:   # List of generated values
        pass
        
    @abc.abstractmethod
    def generate_like(
        self,
        original_value: Any,  # Original value to base generation on
        **params             # Additional parameters
    ) -> Any:                # Generated value
        pass
        
    @abc.abstractmethod
    def analyze_value(
        self,
        value: Any    # Value to analyze
    ) -> Dict[str, Any]:  # Dictionary with analysis results
        pass
        
    def estimate_resources(
        self,
        count: int,   # Number of values to generate
        **params      # Additional parameters affecting resource usage
    ) -> Dict[str, float]:  # Dictionary with resource estimates
        pass
```

### BaseMapper

```python
class BaseMapper:
    @abc.abstractmethod
    def map(
        self,
        original_value: Any,  # Original value to map
        **params             # Additional parameters
    ) -> Any:                # Synthetic value
        pass
        
    @abc.abstractmethod
    def restore(
        self,
        synthetic_value: Any  # Synthetic value to restore from
    ) -> Optional[Any]:       # Original value if available
        pass
        
    @abc.abstractmethod
    def add_mapping(
        self,
        original: Any,        # Original value
        synthetic: Any,       # Synthetic value
        is_transitive: bool = False  # Flag for transitive mappings
    ) -> None:
        pass
        
    @abc.abstractmethod
    def check_conflicts(
        self,
        original: Any,        # Original value
        synthetic: Any        # Synthetic value
    ) -> Dict[str, Any]:      # Conflict information
        pass
```

### FieldOperation

```python
class FieldOperation(BaseOperation):
    def __init__(
        self,
        field_name: str,      # Name of the field to process
        mode: str = "REPLACE",  # Operation mode: "REPLACE" or "ENRICH"
        output_field_name: Optional[str] = None,  # Output field name for ENRICH mode
        null_strategy: NullStrategy = NullStrategy.PRESERVE  # NULL handling strategy
    ):
        pass
        
    @abc.abstractmethod
    def process_batch(
        self,
        batch: pd.DataFrame   # Batch of data to process
    ) -> pd.DataFrame:        # Processed batch
        pass
        
    def preprocess_data(
        self,
        df: pd.DataFrame      # DataFrame to preprocess
    ) -> pd.DataFrame:        # Preprocessed DataFrame
        pass
        
    @staticmethod
    def postprocess_data(
        df: pd.DataFrame      # DataFrame to postprocess
    ) -> pd.DataFrame:        # Postprocessed DataFrame
        pass
        
    def handle_null_values(
        self,
        df: pd.DataFrame      # DataFrame with NULL values
    ) -> pd.DataFrame:        # DataFrame with handled NULL values
        pass
```

### MappingStore

```python
class MappingStore:
    def __init__(self):
        pass
        
    def add_mapping(
        self,
        field_name: str,      # Name of the field
        original: Any,        # Original value
        synthetic: Any,       # Synthetic value
        is_transitive: bool = False  # Whether the mapping is transitive
    ) -> None:
        pass
        
    def get_mapping(
        self,
        field_name: str,      # Name of the field
        original: Any         # Original value
    ) -> Optional[Any]:       # Synthetic value or None if not found
        pass
        
    def restore_original(
        self,
        field_name: str,      # Name of the field
        synthetic: Any        # Synthetic value
    ) -> Optional[Any]:       # Original value or None if not found
        pass
        
    def is_transitive(
        self,
        field_name: str,      # Name of the field
        original: Any         # Original value
    ) -> bool:                # True if the mapping is transitive
        pass
        
    def get_all_mappings_for_field(
        self,
        field_name: str       # Name of the field
    ) -> Dict[Any, Any]:      # Dictionary of original to synthetic mappings
        pass
        
    def get_field_names(
        self
    ) -> Set[str]:            # Set of field names with mappings
        pass
```

## Design Principles

The `base.py` module embodies several key design principles:

1. **Abstract Base Classes**: Uses Python's ABC module to define interfaces
2. **Separation of Concerns**: Clear division between generation, mapping, and operations
3. **Consistent Error Handling**: Well-defined exception hierarchy
4. **Type Hints**: Comprehensive typing for better IDE support and code safety
5. **Resource Awareness**: Built-in resource estimation capabilities
6. **Standardized Interfaces**: Uniform method signatures across related components
7. **Configuration Through Classes**: Uses class constructors for configuration rather than global settings

## Pamola Core Architectural Patterns

The module implements several architectural patterns:

1. **Strategy Pattern**: Different implementations can be selected at runtime
2. **Template Method Pattern**: Base classes define workflow, subclasses implement specifics
3. **Repository Pattern**: MappingStore provides a central repository for mappings
4. **Factory Method Pattern**: Resource creation is delegated to specialized methods
5. **Command Pattern**: Operations encapsulate actions to be performed on data

## Integration with PAMOLA.CORE Framework

The `base.py` module integrates with the broader PAMOLA CORE framework through:

1. **Operation Framework Compatibility**: BaseOperation aligns with PAMOLA operation semantics
2. **Standardized Resource Tracking**: Compatible with PAMOLA.CORE progress reporting
3. **Consistent Error Propagation**: Exceptions designed to work with PAMOLA.CORE error handling
4. **Data Source Integration**: Operations work with PAMOLA.CORE DataSource objects

## Extension Guidelines

To extend the system with new functionality:

1. **New Generator Types**: Subclass BaseGenerator, implement required methods
2. **New Mapping Strategies**: Extend BaseMapper for different mapping approaches
3. **New Operation Types**: Implement BaseOperation or FieldOperation for new operations
4. **Enhanced MappingStore**: Extend MappingStore for additional storage capabilities

## Performance Considerations

The abstract base classes incorporate performance best practices:

1. **Resource Estimation**: Built-in methods to estimate resource requirements
2. **Batched Processing**: Support for processing data in manageable chunks
3. **Memory Management**: NULL value handling to optimize memory usage
4. **Efficient Mapping**: Bidirectional lookups for fast mapping operations

## Error Handling Strategy

The module defines a comprehensive error handling strategy:

1. **Specialized Exceptions**: Hierarchy of exceptions for different error types
2. **Informative Messages**: Detailed error messages for troubleshooting
3. **Predictable Propagation**: Clear rules for when and how exceptions are raised
4. **Graceful Degradation**: Options for fallback behavior when errors occur

## Thread Safety Considerations

Components in `base.py` have been designed with concurrent usage in mind:

1. **MappingStore**: Implementation should be thread-safe for concurrent access
2. **Generators**: Designed to support thread-safe operation with proper implementation
3. **Operations**: Process isolation through batch-based processing

## NULL Value Handling

The module provides a comprehensive strategy for NULL values:

1. **NullStrategy Enum**: Defines standard approaches to NULL handling
2. **Configurable Behavior**: Each operation can specify its NULL strategy
3. **Consistent Implementation**: handle_null_values method for uniform processing
4. **Preservation Option**: Ability to preserve NULL semantics in processed data

## Complete Package Documentation

When combined with other modules in the `commons` package, the `base.py` module provides the foundation for a comprehensive fake data generation system:

```
┌───────────────────────────────────────────────┐
│           pamola_core.fake_data.commons               │
├───────────┬─────────────────┬─────────────────┤
│           │                 │                 │
│  base.py  │   utils.py      │  validators.py  │
│           │                 │                 │
│ Core      │ Helper          │ Data            │
│ classes   │ functions       │ validation      │
│ and       │ and             │ functions       │
│ interfaces│ utilities       │                 │
└───────────┴─────────────────┴─────────────────┘
            │                 │                 │
┌───────────▼─────────────────▼─────────────────▼───────┐
│                                                       │
│             pamola_core.fake_data Package                     │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ generators  │  │  mappers    │  │ operations  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Implementation Considerations

When implementing classes based on the abstract classes in `base.py`, consider the following:

1. **Complete Implementation**: Ensure all abstract methods are implemented
2. **Resource Efficiency**: Pay attention to memory and CPU usage
3. **Error Handling**: Propagate appropriate exceptions with informative messages
4. **Documentation**: Document implementation-specific behavior and assumptions
5. **Testing**: Create comprehensive tests for different edge cases

## Limitations and Constraints

When working with the `base.py` module, be aware of these limitations:

1. **Abstract Classes**: Cannot be instantiated directly, must be subclassed
2. **Save/Load Implementation**: MappingStore requires subclass implementation for serialization
3. **Resource Estimation**: Estimates are approximations and may vary with data characteristics
4. **Thread Safety**: Implementations must ensure their own thread safety

## Best Practices

To effectively use the `base.py` module:

1. **Follow Interface Contracts**: Adhere to the behavior defined in abstract methods
2. **Leverage Abstract Methods**: Use base class utility methods where appropriate
3. **Consistent Error Handling**: Use the provided exception hierarchy
4. **Resource Awareness**: Implement resource estimation accurately
5. **Clean Extension**: Extend functionality without modifying base classes

## Documentation Guidelines

When documenting implementations of these abstract classes:

1. **Clear Class Purpose**: Explain what makes the implementation unique
2. **Parameter Documentation**: Document all parameters, especially implementation-specific ones
3. **Return Value Specification**: Clearly document return values and formats
4. **Exception Documentation**: List all exceptions that may be raised
5. **Example Usage**: Provide example code for common use cases

## Conclusion

The `base.py` module provides the architectural foundation for the entire `fake_data` package. Its abstract classes and interfaces define a clear contract for how generators, mappers, and operations should behave, ensuring consistency across the system while allowing for flexibility in implementation.

By adhering to these interfaces and design principles, developers can extend the system with new capabilities while maintaining compatibility with existing components. The module's focus on resource awareness, proper error handling, and standardized interfaces makes it a solid foundation for building a robust fake data generation system that can scale to handle large datasets while producing high-quality synthetic data.

As the system evolves, the abstractions defined in `base.py` provide a stable reference point, allowing other components to change without disrupting the overall architecture. This makes the module a critical part of the system's long-term maintainability and extensibility strategy.