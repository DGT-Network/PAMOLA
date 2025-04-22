"""
Base abstract classes for the fake_data module.

This module provides the core abstractions and interfaces for fake data generation,
mapping, and operations.
"""

import abc
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, TypeVar, Union, Set, Callable

import pandas as pd

T = TypeVar('T')


class ResourceType(Enum):
    """Types of resources that can be estimated for operations."""
    MEMORY = "memory_mb"
    TIME = "time_seconds"
    CPU = "cpu_percent"
    DISK = "disk_mb"


class NullStrategy(Enum):
    """Strategies for handling NULL values in data."""
    PRESERVE = "preserve"  # Keep NULL values unchanged
    REPLACE = "replace"  # Replace NULL with a default or generated value
    EXCLUDE = "exclude"  # Skip NULL values during processing
    ERROR = "error"  # Raise an error when NULL is encountered


class FakeDataError(Exception):
    """Base exception class for fake_data module errors."""
    pass


class ValidationError(FakeDataError):
    """Exception raised for validation errors."""
    pass


class ResourceError(FakeDataError):
    """Exception raised for resource-related errors (memory, disk, etc.)."""
    pass


class MappingError(FakeDataError):
    """Exception raised for mapping errors."""
    pass


class BaseGenerator(abc.ABC):
    """
    Base abstract class for all fake data generators.

    Generators are responsible for creating synthetic values replacing
    real data, taking into account statistical characteristics and structure.
    """

    @abc.abstractmethod
    def generate(self, count: int, **params) -> List[Any]:
        """
        Generates the specified number of fake values.

        Parameters:
        -----------
        count : int
            Number of values to generate
        **params : dict
            Additional generation parameters, including:
            - gender: gender ('M'/'F')
            - region: region/country
            - language: language
            - seed: initial value for the random number generator

        Returns:
        --------
        List[Any]
            List of generated values

        Raises:
        -------
        ValidationError
            If parameters are invalid
        ResourceError
            If insufficient resources to complete generation
        """
        pass

    @abc.abstractmethod
    def generate_like(self, original_value: Any, **params) -> Any:
        """
        Generates a fake value similar to the original.

        Parameters:
        -----------
        original_value : Any
            Original value to be replaced
        **params : dict
            Additional generation parameters

        Returns:
        --------
        Any
            Generated value

        Raises:
        -------
        ValidationError
            If original value format is invalid or unsupported
        """
        pass

    @abc.abstractmethod
    def analyze_value(self, value: Any) -> Dict[str, Any]:
        """
        Analyzes the structure and characteristics of a value.

        Parameters:
        -----------
        value : Any
            Value to analyze

        Returns:
        --------
        Dict[str, Any]
            Dictionary with analysis results containing at minimum:
            - type: The detected data type
            - format: The detected format (if applicable)
            - complexity: Estimated complexity level
            - special_features: List of detected special features
        """
        pass

    def estimate_resources(self, count: int, **params) -> Dict[str, float]:
        """
        Estimates resources needed to generate the specified number of values.

        This implementation provides a basic estimation. Override for more accurate estimates.

        Parameters:
        -----------
        count : int
            Number of values to generate
        **params : dict
            Additional parameters that may affect resource usage

        Returns:
        --------
        Dict[str, float]
            Dictionary with resource estimates containing at least:
            - memory_mb: Estimated memory usage in MB
            - time_seconds: Estimated time in seconds
        """
        # Default implementation with minimal estimates
        # Subclasses should override with more accurate calculations
        return {
            ResourceType.MEMORY.value: float(count) * 0.01,  # Basic estimate
            ResourceType.TIME.value: float(count) * 0.001,  # Basic estimate
        }


class BaseMapper(abc.ABC):
    """
    Base abstract class for mapping components.

    Mappers are responsible for mapping original values to synthetic ones,
    ensuring deterministic replacements and conflict resolution.
    """

    @abc.abstractmethod
    def map(self, original_value: Any, **params) -> Any:
        """
        Maps an original value to a fake one.

        Parameters:
        -----------
        original_value : Any
            Original value to map
        **params : dict
            Additional mapping parameters, including:
            - force_new: force creation of a new value
            - context: contextual information for ensuring consistency
            - preserve_format: preserve the format of the original value

        Returns:
        --------
        Any
            Fake value

        Raises:
        -------
        ValidationError
            If original value cannot be processed
        MappingError
            If mapping cannot be performed or conflict cannot be resolved
        """
        pass

    @abc.abstractmethod
    def restore(self, synthetic_value: Any) -> Optional[Any]:
        """
        Attempts to restore the original value from a fake one.

        Parameters:
        -----------
        synthetic_value : Any
            Fake value to restore from

        Returns:
        --------
        Optional[Any]
            Original value if available, None if restoration is not possible
            due to non-reversible mapping or other limitations

        Raises:
        -------
        MappingError
            If restoration encounters conflicts (multiple possible originals)
        """
        pass

    @abc.abstractmethod
    def add_mapping(self, original: Any, synthetic: Any, is_transitive: bool = False) -> None:
        """
        Adds a new mapping to the mapper.

        Parameters:
        -----------
        original : Any
            Original value
        synthetic : Any
            Fake value
        is_transitive : bool
            Flag indicating whether the mapping is transitive

        Raises:
        -------
        MappingError
            If mapping addition creates conflicts
        """
        pass

    @abc.abstractmethod
    def check_conflicts(self, original: Any, synthetic: Any) -> Dict[str, Any]:
        """
        Checks for possible conflicts when adding a new mapping.

        Parameters:
        -----------
        original : Any
            Original value
        synthetic : Any
            Fake value

        Returns:
        --------
        Dict[str, Any]
            Information about conflicts:
            - has_conflicts: bool indicating if conflicts exist
            - conflict_type: type of conflict if present
            - affected_values: list of affected values
        """
        pass

    @staticmethod
    def get_conflicts_resolution_strategies() -> Dict[str, Callable]:
        """
        Returns available conflict resolution strategies.

        This method can be used to discover supported strategies for
        resolving mapping conflicts in specific implementations.

        Returns:
        --------
        Dict[str, Callable]
            Dictionary mapping strategy names to handler functions
        """
        # Default minimal implementation - subclasses should enhance this
        return {
            "append_suffix": lambda x: f"{x}_1",
            "use_original": lambda x, y: x,
            "use_latest": lambda x, y: y,
        }


class BaseOperation(abc.ABC):
    """
    Base abstract class for data operations.

    Operations provide a standardized interface for performing
    various actions on data using the operations infrastructure.
    """

    name: str
    description: str

    @abc.abstractmethod
    def execute(self, data_source: Any, task_dir: Any, reporter: Any, **kwargs) -> Any:
        """
        Executes the operation.

        Parameters:
        -----------
        data_source : Any
            Source of data for processing
        task_dir : Any
            Directory for storing operation artifacts
        reporter : Any
            Reporter for operation progress and results
        **kwargs : dict
            Additional operation parameters

        Returns:
        --------
        Any
            Operation result

        Raises:
        -------
        FakeDataError or subclasses
            If operation execution fails
        """
        pass


class FieldOperation(BaseOperation):
    """
    Base abstract class for field-specific operations.

    Provides common functionality for operations that process
    specific fields in a dataset.
    """

    def __init__(self, field_name: str, mode: str = "REPLACE", output_field_name: Optional[str] = None,
                 null_strategy: NullStrategy = NullStrategy.PRESERVE):
        """
        Initializes the field operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to process
        mode : str
            Operation mode: "REPLACE" or "ENRICH"
        output_field_name : str, optional
            Name of the output field (used in ENRICH mode)
        null_strategy : NullStrategy
            Strategy for handling NULL values
        """
        self.field_name = field_name
        self.mode = mode
        self.output_field_name = output_field_name
        self.null_strategy = null_strategy

        # Set default output field name for ENRICH mode
        if self.mode == "ENRICH" and not self.output_field_name:
            self.output_field_name = f"{field_name}_fake"

    @abc.abstractmethod
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a batch of data.

        Parameters:
        -----------
        batch : pd.DataFrame
            Batch of data to process

        Returns:
        --------
        pd.DataFrame
            Processed batch
        """
        pass

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for processing.

        Base implementation performs basic preprocessing.
        Override for specific preprocessing needs.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to preprocess

        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame
        """
        # Basic preprocessing - subclasses can override with more specific logic
        if self.field_name not in df.columns:
            raise ValidationError(f"Field {self.field_name} not found in DataFrame")

        return df.copy()

    @staticmethod
    def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs final processing after all batches are processed.

        Base implementation returns DataFrame as is.
        Override for specific postprocessing needs.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to postprocess

        Returns:
        --------
        pd.DataFrame
            Postprocessed DataFrame
        """
        # Basic postprocessing - subclasses can override with more specific logic
        return df

    def handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles NULL values in the data based on the specified strategy.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with NULL values

        Returns:
        --------
        pd.DataFrame
            DataFrame with handled NULL values
        """
        if self.null_strategy == NullStrategy.PRESERVE:
            return df
        elif self.null_strategy == NullStrategy.EXCLUDE:
            # Create a mask for non-null values
            non_null_mask = df[self.field_name].notna()
            return df[non_null_mask].copy()
        elif self.null_strategy == NullStrategy.ERROR:
            null_count = df[self.field_name].isna().sum()
            if null_count > 0:
                raise ValidationError(f"Found {null_count} NULL values in field {self.field_name}")
            return df
        elif self.null_strategy == NullStrategy.REPLACE:
            # Basic replacement with empty string/0/etc.
            # Subclasses should override this for more sophisticated replacement
            result = df.copy()
            field_dtype = df[self.field_name].dtype

            # Set default replacements based on data type
            if pd.api.types.is_string_dtype(field_dtype):
                result[self.field_name] = result[self.field_name].fillna("")
            elif pd.api.types.is_numeric_dtype(field_dtype):
                result[self.field_name] = result[self.field_name].fillna(0)
            else:
                # For other types, leave as is
                pass

            return result

        # Fallback
        return df


class MappingStore:
    """
    Storage for mappings between original and synthetic values.

    Provides methods for storing, retrieving, and managing mappings
    with support for bidirectional lookup and transitivity marking.
    """

    def __init__(self):
        """
        Initializes the mapping store.
        """
        # Direct mappings: {field_name: {original: synthetic}}
        self.mappings = {}

        # Reverse mappings: {field_name: {synthetic: original}}
        self.reverse_mappings = {}

        # Transitivity markers: {field_name: {original: is_transitive}}
        self.transitivity_markers = {}

    def add_mapping(self, field_name: str, original: Any, synthetic: Any, is_transitive: bool = False) -> None:
        """
        Adds a mapping between original and synthetic values.

        Parameters:
        -----------
        field_name : str
            Name of the field
        original : Any
            Original value
        synthetic : Any
            Synthetic value
        is_transitive : bool
            Whether the mapping is transitive

        Raises:
        -------
        MappingError
            If mapping creates conflicts that cannot be resolved
        """
        # Initialize dictionaries for the field if they don't exist
        if field_name not in self.mappings:
            self.mappings[field_name] = {}
            self.reverse_mappings[field_name] = {}
            self.transitivity_markers[field_name] = {}

        # Check for conflicts
        if original in self.mappings[field_name] and self.mappings[field_name][original] != synthetic:
            raise MappingError(
                f"Conflict: Original value already mapped to a different synthetic value for field {field_name}")

        if synthetic in self.reverse_mappings[field_name] and self.reverse_mappings[field_name][synthetic] != original:
            raise MappingError(
                f"Conflict: Synthetic value already mapped from a different original value for field {field_name}")

        # Add direct and reverse mappings
        self.mappings[field_name][original] = synthetic
        self.reverse_mappings[field_name][synthetic] = original

        # Mark transitivity
        self.transitivity_markers[field_name][original] = is_transitive

    def get_mapping(self, field_name: str, original: Any) -> Optional[Any]:
        """
        Gets the synthetic value for an original value.

        Parameters:
        -----------
        field_name : str
            Name of the field
        original : Any
            Original value

        Returns:
        --------
        Optional[Any]
            Synthetic value or None if not found
        """
        if field_name not in self.mappings:
            return None

        return self.mappings[field_name].get(original)

    def restore_original(self, field_name: str, synthetic: Any) -> Optional[Any]:
        """
        Restores the original value from a synthetic one.

        Parameters:
        -----------
        field_name : str
            Name of the field
        synthetic : Any
            Synthetic value

        Returns:
        --------
        Optional[Any]
            Original value or None if not found
        """
        if field_name not in self.reverse_mappings:
            return None

        return self.reverse_mappings[field_name].get(synthetic)
    

    def is_transitive(self, field_name: str, original: Any) -> bool:
        """
        Checks if a mapping is transitive.

        Parameters:
        -----------
        field_name : str
            Name of the field
        original : Any
            Original value

        Returns:
        --------
        bool
            True if the mapping is transitive, False otherwise
        """
        if field_name not in self.transitivity_markers:
            return False

        return self.transitivity_markers[field_name].get(original, False)
    
    
    def get_field_mappings(self, field_name: str) -> Dict[Any, Any]:
        """
        Gets all mappings for a field.

        Parameters:
        -----------
        field_name : str
            Name of the field

        Returns:
        --------
        Dict[Any, Any]
            Dictionary of original to synthetic mappings
        """

        if field_name not in self.mappings:
            return {}

        return self.mappings[field_name].copy()
    

    def get_field_names(self) -> Set[str]:
        """
        Gets all field names in the mapping store.

        Returns:
        --------
        Set[str]
            Set of field names with mappings
        """
        return set(self.mappings.keys())

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the mapping store to a file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the mapping store

        Raises:
        -------
        IOError
            If saving fails due to I/O errors
        """
        # This implementation should use appropriate I/O utilities
        # and handle serialization of complex types
        raise NotImplementedError("Subclasses must implement save method")

    def load(self, path: Union[str, Path]) -> None:
        """
        Loads the mapping store from a file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to load the mapping store from

        Raises:
        -------
        IOError
            If loading fails due to I/O errors
        MappingError
            If loaded mappings have structural issues
        """
        # This implementation should use appropriate I/O utilities
        # and handle deserialization of complex types
        raise NotImplementedError("Subclasses must implement load method")