"""
One-to-One mapper for deterministic fake data generation.

This module provides an enhanced mapper for maintaining one-to-one relationships
between original and fake values, ensuring consistent replacements with
improved conflict resolution and performance optimizations.
"""

from typing import Dict, List, Optional

from pamola_core.fake_data.commons.base import BaseMapper, MappingError
from pamola_core.fake_data.mappers.transitivity_handler import TransitivityHandler
from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils import logging as pamola_logging
from typing import Callable, Any

# Configure logger
logger = pamola_logging.get_logger("pamola_core.fake_data.mappers.one_to_one")


class OneToOneMapper(BaseMapper):
    """
    Maps original values to synthetic ones with one-to-one relationships.

    Ensures deterministic replacements where each original value always
    maps to the same synthetic value, and each synthetic value only
    has one original value.

    Features:
    - Enhanced conflict detection and resolution
    - Support for custom conflict resolution strategies
    - Optimized performance for large datasets
    - Integration with transitivity handling
    - Type-specific handling for names, emails, and other data types
    """

    def __init__(self,
                 field_name: str,
                 fallback_generator: Optional[Callable[[Any], Any]] = None,
                 mapping_store: Optional[MappingStore] = None,
                 allow_identity: bool = False,
                 transitive_handling: bool = True,
                 transitivity_handler: Optional[TransitivityHandler] = None,
                 conflict_strategies: Optional[Dict[str, Callable]] = None,
                 default_conflict_strategy: str = "append_suffix",
                 data_type: Optional[str] = None):
        """
        Initialize the one-to-one mapper with enhanced options.

        Parameters:
        -----------
        field_name : str
            Name of the field this mapper is for
        fallback_generator : Optional[Callable]
            Function that generates values when not found in mapping
        mapping_store : Optional[MappingStore]
            Store for mappings (created if not provided)
        allow_identity : bool
            Whether to allow mapping a value to itself
        transitive_handling : bool
            Whether to handle transitive mappings
        transitivity_handler : Optional[TransitivityHandler]
            Handler for transitive relationships (created if transitive_handling=True and not provided)
        conflict_strategies : Optional[Dict[str, Callable]]
            Custom strategies for resolving conflicts (overrides defaults)
        default_conflict_strategy : str
            Name of the default strategy to use for conflicts
        data_type : Optional[str]
            Type of data being mapped (e.g., "name", "email", "phone")
            Used for type-specific handling
        """
        self.field_name = field_name
        self.fallback_generator = fallback_generator
        self.mapping_store = mapping_store or MappingStore()
        self.allow_identity = allow_identity
        self.transitive_handling = transitive_handling
        self.data_type = data_type
        self.default_conflict_strategy = default_conflict_strategy

        # Set up transitivity handler if needed
        self._setup_transitivity_handler(transitivity_handler)

        # Set up conflict resolution strategies
        self.conflict_strategies = self._get_default_strategies()
        if conflict_strategies:
            self.conflict_strategies.update(conflict_strategies)

        # Optimize: use dict instead of set for quicker reverse lookups
        self._synthetic_values = {}  # {synthetic_value: original_value}

        # Initialize cache for batch operations
        self._mapping_cache = {}  # {original: synthetic}

        # Initialize from mapping store if provided
        if mapping_store:
            # Load existing mappings into cache
            mappings = mapping_store.get_field_mappings(field_name)
            for original, synthetic in mappings.items():
                self._synthetic_values[synthetic] = original
                self._mapping_cache[original] = synthetic

    def _setup_transitivity_handler(self, transitivity_handler: Optional[TransitivityHandler]) -> None:
        """
        Sets up the transitivity handler.

        Parameters:
        -----------
        transitivity_handler : Optional[TransitivityHandler]
            Externally provided handler or None to create a new one if needed
        """
        if transitivity_handler:
            self.transitivity_handler = transitivity_handler
        elif self.transitive_handling:
            self.transitivity_handler = TransitivityHandler(self.mapping_store)
        else:
            self.transitivity_handler = None

    def _get_default_strategies(self) -> Dict[str, Callable]:
        """
        Gets the default conflict resolution strategies.

        Returns:
        --------
        Dict[str, Callable]
            Dictionary of strategy name to strategy function
        """
        base_strategies = super().get_conflicts_resolution_strategies()

        # Add enhanced strategies
        enhanced_strategies = {
            # Add numeric suffix
            "append_number": lambda x, i=1: f"{x}_{i}",

            # Add random suffix
            "append_random": lambda x, **kwargs: self._generate_random_suffix(x, **kwargs),

            # Generate completely new value
            "generate_new": lambda x, **kwargs: self._generate_new_value(x, **kwargs),

            # Name-specific strategies
            "gender_region_repick": lambda x, **kwargs: self._repick_by_gender_region(x, **kwargs),
            "append_initial": lambda x, **kwargs: self._append_name_initial(x, **kwargs),
            "avoid_duplicate_suffix": lambda x, **kwargs: f"{x}_dup",
        }

        # Combine base and enhanced strategies
        base_strategies.update(enhanced_strategies)
        return base_strategies

    def _generate_random_suffix(self, value: Any, **kwargs) -> Any:
        """
        Generates a value with a random suffix.

        Parameters:
        -----------
        value : Any
            Original value
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Any
            Value with random suffix
        """
        import random
        suffix = kwargs.get('suffix_length', 3)
        if isinstance(value, str):
            return f"{value}_{random.randint(100, 999)}"
        return value

    def _generate_new_value(self, value: Any, **kwargs) -> Any:
        """
        Generates a completely new value.

        Parameters:
        -----------
        value : Any
            Original value
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Any
            New generated value
        """
        if self.fallback_generator:
            # Force a new value by modifying parameters
            params = kwargs.copy()
            params['avoid_collision'] = True
            params['force_new'] = True
            return self._call_fallback_generator(value, **params)
        return value

    def _repick_by_gender_region(self, value: Any, **kwargs) -> Any:
        """
        Re-picks a name considering gender and region constraints.

        Parameters:
        -----------
        value : Any
            Original name
        **kwargs : dict
            Additional parameters including gender and region

        Returns:
        --------
        Any
            New name from the same gender/region pool
        """
        if self.data_type != 'name' or not self.fallback_generator:
            return value

        # Extract gender and region from kwargs
        gender = kwargs.get('gender')
        region = kwargs.get('region', kwargs.get('language'))

        # Generate new name with same constraints
        params = kwargs.copy()
        params['avoid_collision'] = True
        params['force_new'] = True
        params['gender'] = gender
        params['region'] = region

        return self._call_fallback_generator(value, **params)

    def _append_name_initial(self, value: Any, **kwargs) -> Any:
        """
        Appends an initial to a name, typically from another name component.

        Parameters:
        -----------
        value : Any
            Original name
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Any
            Name with appended initial
        """
        if not isinstance(value, str):
            return value

        # Extract initial source from kwargs
        initial_source = kwargs.get('initial_source', '')
        if initial_source and isinstance(initial_source, str) and len(initial_source) > 0:
            initial = initial_source[0].upper()
            return f"{value} {initial}."

        return f"{value} X."



    def _call_fallback_generator(self, original_value: Any, **params) -> Any:
        """
        Safely call the fallback generator with filtered parameters.

        This method ensures that only valid parameters are passed to the generator,
        preventing unexpected argument errors by introspecting the generator's signature.

        Args:
            original_value (Any): The original value to be transformed
            **params: Arbitrary keyword arguments to be passed to the generator

        Returns:
            Any: The generated synthetic value
        """

        def safe_call(generator: Callable, value: Any, **kwargs) -> Any:
            """
            Safely execute a generator by filtering its parameters.

            Args:
                generator (Callable): The generator function to call
                value (Any): The original value to transform
                **kwargs: Arbitrary keyword arguments

            Returns:
                Any: The generated value
            """
            import inspect

            # Introspect the generator's signature
            signature = inspect.signature(generator)

            # Filter parameters to match the generator's expected arguments
            valid_params = {
                k: v for k, v in kwargs.items()
                if k in signature.parameters or
                   k in signature.parameters.values() or
                   any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            }

            # Call the generator with valid parameters
            return generator(value, **valid_params)

        # Call the safe_call wrapper with the fallback generator
        return safe_call(self.fallback_generator, original_value, **params)

    def map(self, original_value: Any, **params) -> Any:
        """
        Maps an original value to a synthetic one with enhanced conflict handling.

        Parameters:
        -----------
        original_value : Any
            Original value to map
        **params : dict
            Additional mapping parameters, including:
            - force_new: force creation of a new value
            - context: contextual information
            - preserve_format: preserve the format of the original value
            - conflict_strategy_override: override default conflict strategy
            - batch_operation: flag indicating this is part of a batch operation
            - gender, region, language: type-specific parameters

        Returns:
        --------
        Any
            Synthetic value

        Raises:
        -------
        MappingError
            If mapping cannot be performed or conflict cannot be resolved
        """
        # Handle None or empty values
        if original_value is None:
            return None

        # Check cache for batch operations
        batch_operation = params.get("batch_operation", False)
        if batch_operation and original_value in self._mapping_cache:
            return self._mapping_cache[original_value]

        # Check if force_new is specified
        force_new = params.get("force_new", False)

        # Check existing mapping if not forcing new
        if not force_new:
            existing = self.mapping_store.get_mapping(self.field_name, original_value)
            if existing is not None:
                # Cache for batch operations
                if batch_operation:
                    self._mapping_cache[original_value] = existing
                return existing

        # No existing mapping or forcing new, generate a synthetic value
        if self.fallback_generator is None:
            raise MappingError(f"No fallback generator for field {self.field_name} and value not in mapping")

        synthetic_value = self._call_fallback_generator(original_value, **params)

        # Check for identity mapping
        if not self.allow_identity and synthetic_value == original_value:
            logger.warning(f"Generated synthetic value is identical to original for {self.field_name}")
            synthetic_value = self._handle_identity_conflict(original_value, synthetic_value, **params)

        # Check for conflicts
        conflict_info = self.check_conflicts(original_value, synthetic_value)
        if conflict_info["has_conflicts"]:
            # Try to resolve the conflict
            synthetic_value = self._resolve_conflict(conflict_info, original_value, synthetic_value, **params)

        # No conflicts or resolved, add the mapping
        self.mapping_store.add_mapping(self.field_name, original_value, synthetic_value)
        self._synthetic_values[synthetic_value] = original_value

        # Cache for batch operations
        if batch_operation:
            self._mapping_cache[original_value] = synthetic_value

        # If transitive handling is enabled, check for cycles
        if self.transitive_handling and self.transitivity_handler:
            self._check_and_fix_transitive_issues()

        return synthetic_value

    def _handle_identity_conflict(self, original_value: Any, synthetic_value: Any, **params) -> Any:
        """
        Handles the case where synthetic value is identical to original value.

        Parameters:
        -----------
        original_value : Any
            Original value
        synthetic_value : Any
            Synthetic value (identical to original)
        **params : dict
            Additional parameters

        Returns:
        --------
        Any
            Modified synthetic value
        """
        # Try again with modified parameters to avoid identity mapping
        retry_params = params.copy()
        retry_params["avoid_identity"] = True

        # Try generator again
        if self.fallback_generator:
            new_value = self._call_fallback_generator(original_value, **retry_params)
            if new_value != original_value:
                return new_value

        # If still identical or no generator, apply simple transformation
        if isinstance(synthetic_value, str):
            return synthetic_value + "_"
        elif isinstance(synthetic_value, (int, float)):
            return synthetic_value + 1
        else:
            # For other types, use string representation
            return f"{synthetic_value}_modified"

    def _resolve_conflict(self, conflict_info: Dict[str, Any], original_value: Any,
                          synthetic_value: Any, **params) -> Any:
        """
        Resolves conflicts using appropriate strategies.

        Parameters:
        -----------
        conflict_info : Dict[str, Any]
            Information about the conflict
        original_value : Any
            Original value
        synthetic_value : Any
            Conflicting synthetic value
        **params : dict
            Additional parameters

        Returns:
        --------
        Any
            Resolved synthetic value
        """
        conflict_type = conflict_info["conflict_type"]
        force_new = params.get("force_new", False)

        # Get the strategy to use
        strategy_name = params.get("conflict_strategy_override", self.default_conflict_strategy)

        if conflict_type == "reverse_mapping_exists":
            # Handle transitive mapping case
            if self.transitive_handling:
                result = self._handle_transitive_mapping(original_value, synthetic_value, conflict_info, **params)
                if result is not None:
                    return result

            # Otherwise, apply conflict resolution strategy
            strategy = self.conflict_strategies.get(strategy_name)
            if not strategy:
                raise MappingError(f"Unknown conflict resolution strategy: {strategy_name}")

            return strategy(synthetic_value, **params)

        elif conflict_type == "direct_mapping_exists":
            # This original already maps to something else
            if force_new:
                # If forcing new, update the mapping
                return synthetic_value
            else:
                # Otherwise, use the existing mapping
                return self.mapping_store.get_mapping(self.field_name, original_value)

        elif conflict_type == "cyclic_mapping_detected":
            # Break the cycle by generating a new value
            logger.warning(f"Cycle detected in mapping for {self.field_name}")
            cycle_break_params = params.copy()
            cycle_break_params["avoid_cycle"] = True
            return self._call_fallback_generator(original_value, **cycle_break_params)

        # For other conflict types
        strategy = self.conflict_strategies.get(strategy_name)
        if strategy:
            return strategy(synthetic_value, **params)

        # Fallback strategy
        if isinstance(synthetic_value, str):
            return f"{synthetic_value}_alt"
        return synthetic_value

    def _handle_transitive_mapping(self, original_value: Any, synthetic_value: Any,
                                   conflict_info: Dict[str, Any], **params) -> Optional[Any]:
        """
        Handles transitive mapping case.

        Parameters:
        -----------
        original_value : Any
            Original value
        synthetic_value : Any
            Synthetic value
        conflict_info : Dict[str, Any]
            Information about the conflict
        **params : dict
            Additional parameters

        Returns:
        --------
        Optional[Any]
            Resolved synthetic value or None if not handled
        """
        # Find what the existing original maps to
        existing_original = self.mapping_store.restore_original(self.field_name, synthetic_value)

        # Check if that creates a cycle
        if existing_original == original_value:
            logger.warning(f"Cyclic mapping detected for {self.field_name}")

            # Generate a different value to break the cycle
            cycle_break_params = params.copy()
            cycle_break_params["avoid_cycle"] = True
            return self._call_fallback_generator(original_value, **cycle_break_params)
        else:
            # Add the mapping with transitive flag
            self.mapping_store.add_mapping(
                self.field_name,
                original_value,
                synthetic_value,
                is_transitive=True
            )
            self._synthetic_values[synthetic_value] = original_value
            return synthetic_value

    def _check_and_fix_transitive_issues(self) -> None:
        """
        Checks and fixes transitive mapping issues.
        """
        if not self.transitivity_handler:
            return

        # Check for cycles
        cycles = self.transitivity_handler.find_cycles(self.field_name)
        if cycles:
            logger.info(f"Found {len(cycles)} cycles in mappings for {self.field_name}")
            self.transitivity_handler.resolve_all_cycles(self.field_name)

        # Fix transitive mappings
        self.transitivity_handler.fix_transitive_mappings(self.field_name)

    def batch_map(self, values: List[Any], **params) -> List[Any]:
        """
        Maps a batch of values efficiently.

        Parameters:
        -----------
        values : List[Any]
            List of original values to map
        **params : dict
            Additional mapping parameters

        Returns:
        --------
        List[Any]
            List of synthetic values
        """
        # Enable batch operation for caching
        batch_params = params.copy()
        batch_params["batch_operation"] = True

        # Clear mapping cache for this batch
        self._mapping_cache = {}

        # Process each value
        result = [self.map(value, **batch_params) for value in values]

        # Clear cache after batch operation
        self._mapping_cache = {}

        return result

    def restore(self, synthetic_value: Any) -> Optional[Any]:
        """
        Attempts to restore the original value from a synthetic one.

        Parameters:
        -----------
        synthetic_value : Any
            Synthetic value to restore from

        Returns:
        --------
        Optional[Any]
            Original value if available, None otherwise
        """
        # Optimized: check local cache first
        if synthetic_value in self._synthetic_values:
            return self._synthetic_values[synthetic_value]

        # Fallback to mapping store
        return self.mapping_store.restore_original(self.field_name, synthetic_value)

    def add_mapping(self, original: Any, synthetic: Any, is_transitive: bool = False) -> None:
        """
        Adds a new mapping to the mapper.

        Parameters:
        -----------
        original : Any
            Original value
        synthetic : Any
            Synthetic value
        is_transitive : bool
            Flag indicating whether the mapping is transitive

        Raises:
        -------
        MappingError
            If mapping addition creates conflicts
        """
        # Check for conflicts
        conflict_info = self.check_conflicts(original, synthetic)
        if conflict_info["has_conflicts"] and not is_transitive:
            raise MappingError(
                f"Conflict adding mapping for {self.field_name}: {conflict_info['conflict_type']}"
            )

        # Add to mapping store
        self.mapping_store.add_mapping(self.field_name, original, synthetic, is_transitive)
        self._synthetic_values[synthetic] = original

        # If transitive handling is enabled, check for cycles
        if self.transitive_handling and self.transitivity_handler and is_transitive:
            self._check_and_fix_transitive_issues()

    def check_conflicts(self, original: Any, synthetic: Any) -> Dict[str, Any]:
        """
        Checks for possible conflicts when adding a new mapping, with enhanced detection.

        Parameters:
        -----------
        original : Any
            Original value
        synthetic : Any
            Synthetic value

        Returns:
        --------
        Dict[str, Any]
            Information about conflicts:
            - has_conflicts: bool indicating if conflicts exist
            - conflict_type: type of conflict if present
            - affected_values: list of affected values
        """
        result = {
            "has_conflicts": False,
            "conflict_type": None,
            "affected_values": []
        }

        # Check if original already has a different mapping
        existing_synthetic = self.mapping_store.get_mapping(self.field_name, original)
        if existing_synthetic is not None and existing_synthetic != synthetic:
            result["has_conflicts"] = True
            result["conflict_type"] = "direct_mapping_exists"
            result["affected_values"].append(existing_synthetic)

        # Check if synthetic is already mapped from a different original
        existing_original = self.mapping_store.restore_original(self.field_name, synthetic)
        if existing_original is not None and existing_original != original:
            result["has_conflicts"] = True
            result["conflict_type"] = "reverse_mapping_exists"
            result["affected_values"].append(existing_original)

        # Check for potential cycles in transitive mappings
        if self.transitive_handling and self.transitivity_handler:
            # If we have an original -> synthetic mapping and synthetic is already an original
            # for some other mapping, check if adding this mapping creates a cycle
            if synthetic in self._synthetic_values:
                # Get the mapping chain starting from synthetic
                chain = self.transitivity_handler.find_mapping_chain(self.field_name, synthetic)

                # Check if original is in the chain (which would create a cycle)
                if original in chain:
                    result["has_conflicts"] = True
                    result["conflict_type"] = "cyclic_mapping_detected"
                    result["affected_values"] = chain

        return result

    def get_all_mappings(self) -> Dict[Any, Any]:
        """
        Gets all mappings for this field.

        Returns:
        --------
        Dict[Any, Any]
            Dictionary mapping original values to synthetic ones
        """
        return self.mapping_store.get_field_mappings(self.field_name)

    def get_mapping_stats(self) -> Dict[str, Any]:
        """
        Gets detailed statistics about the mappings.

        Returns:
        --------
        Dict[str, Any]
            Statistics about the mappings
        """
        mappings = self.get_all_mappings()
        total_mappings = len(mappings)

        # Get transitive mappings count
        transitive_count = sum(
            1 for original in mappings
            if self.mapping_store.is_transitive(self.field_name, original)
        )

        # Get unique synthetic values count
        unique_synthetic = len(set(mappings.values()))

        # Calculate density (ratio of unique synthetics to total mappings)
        density = unique_synthetic / total_mappings if total_mappings > 0 else 0

        # Additional stats for specific data types
        type_specific_stats = {}
        if self.data_type == "name" and total_mappings > 0:
            # Calculate average name length
            if all(isinstance(v, str) for v in mappings.values()):
                avg_length = sum(len(v) for v in mappings.values()) / total_mappings
                type_specific_stats["avg_name_length"] = avg_length

                # Count names with spaces (likely full names)
                names_with_spaces = sum(1 for v in mappings.values() if " " in v)
                type_specific_stats["full_names_percent"] = names_with_spaces / total_mappings

        return {
            "field_name": self.field_name,
            "total_mappings": total_mappings,
            "unique_synthetic_values": unique_synthetic,
            "mapping_density": density,
            "transitive_mappings": transitive_count,
            "transitive_percentage": transitive_count / total_mappings if total_mappings > 0 else 0,
            "data_type": self.data_type,
            "type_specific_stats": type_specific_stats
        }

    def clear_cache(self) -> None:
        """
        Clears all internal caches.
        """
        self._mapping_cache = {}

        # Rebuild synthetic values from mapping store
        self._synthetic_values = {}
        mappings = self.mapping_store.get_field_mappings(self.field_name)
        for original, synthetic in mappings.items():
            self._synthetic_values[synthetic] = original
