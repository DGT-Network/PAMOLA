"""
MappingStore for fake_data package.

This module provides the MappingStore class for storing and managing mappings
between original and synthetic values with support for transitivity.
The store provides comprehensive functionality for serialization/deserialization
and supports incremental updates for large datasets.
"""

import datetime
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from pamola_core.utils import io as pamola_io
from pamola_core.utils import logging as pamola_logging

# Configure module logger
logger = pamola_logging.get_logger("pamola_core.fake_data.mapping_store")


class MappingStore:
    """
    Storage for mappings between original and synthetic values.

    Provides methods for storing, retrieving, and managing mappings
    with support for bidirectional lookup, transitivity marking,
    and serialization to various formats.

    The class supports:
    - Adding, updating, and removing mappings
    - Bidirectional lookups (original→synthetic, synthetic→original)
    - Tracking transitivity for complex mapping chains
    - Serialization to JSON, CSV, and Pickle formats
    - Incremental updates and merging multiple stores
    """

    def __init__(self):
        """
        Initializes the mapping store with empty mappings and metadata.
        """
        # Direct mappings: {field_name: {original: synthetic}}
        self.mappings = {}

        # Reverse mappings: {field_name: {synthetic: original}}
        self.reverse_mappings = {}

        # Transitivity markers: {field_name: {original: is_transitive}}
        self.transitivity_markers = {}

        # Metadata
        self.metadata = {
            "version": "1.0",
            "fields": {},
            "created_at": None,
            "updated_at": None
        }

    def _update_timestamps(self) -> None:
        """
        Internal method to update timestamps in metadata.
        Uses UTC timezone for consistency across systems.
        """
        # Get current time as ISO format string
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Create new metadata dictionary with updated timestamps
        updated_metadata = dict(self.metadata)

        # Update timestamp for creation if not already set
        if not updated_metadata.get("created_at"):
            updated_metadata["created_at"] = timestamp_str

        # Always update the last updated timestamp
        updated_metadata["updated_at"] = timestamp_str

        # Replace the entire metadata dictionary
        self.metadata = updated_metadata

    def add_mapping(self, field_name: str, original: Any, synthetic: Any,
                    is_transitive: bool = False) -> None:
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
        """
        # Initialize dictionaries for the field if they don't exist
        if field_name not in self.mappings:
            self.mappings[field_name] = {}
            self.reverse_mappings[field_name] = {}
            self.transitivity_markers[field_name] = {}
            self.metadata["fields"][field_name] = {
                "count": 0,
                "type": str(type(original).__name__)
            }

        # Add direct and reverse mappings
        self.mappings[field_name][original] = synthetic
        self.reverse_mappings[field_name][synthetic] = original

        # Mark transitivity
        self.transitivity_markers[field_name][original] = is_transitive

        # Update metadata
        self.metadata["fields"][field_name]["count"] = len(self.mappings[field_name])

        # Update timestamp
        self._update_timestamps()

    def update_mapping(self, field_name: str, original: Any, new_synthetic: Any,
                       update_transitivity: bool = True) -> bool:
        """
        Updates an existing mapping with a new synthetic value.

        Parameters:
        -----------
        field_name : str
            Name of the field
        original : Any
            Original value
        new_synthetic : Any
            New synthetic value
        update_transitivity : bool
            Whether to update transitivity markers (default: True)

        Returns:
        --------
        bool
            True if the mapping was updated, False if it didn't exist
        """
        if field_name not in self.mappings or original not in self.mappings[field_name]:
            return False

        # Get the old synthetic value
        old_synthetic = self.mappings[field_name][original]

        # Remove old reverse mapping
        if field_name in self.reverse_mappings and old_synthetic in self.reverse_mappings[field_name]:
            del self.reverse_mappings[field_name][old_synthetic]

        # Update with new synthetic value
        self.mappings[field_name][original] = new_synthetic

        # Update reverse mapping
        if field_name not in self.reverse_mappings:
            self.reverse_mappings[field_name] = {}
        self.reverse_mappings[field_name][new_synthetic] = original

        # Update transitivity if requested
        if update_transitivity:
            # Check if this synthetic value is used in other mappings
            is_transitive = False
            for orig, synth in self.mappings[field_name].items():
                if synth == new_synthetic and orig != original:
                    is_transitive = True
                    break

            self.transitivity_markers[field_name][original] = is_transitive

        # Update timestamp
        self._update_timestamps()

        return True

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
        Any
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
        Any
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

    def mark_as_transitive(self, field_name: str, original: Any) -> bool:
        """
        Explicitly marks a mapping as transitive.

        Parameters:
        -----------
        field_name : str
            Name of the field
        original : Any
            Original value

        Returns:
        --------
        bool
            True if successful, False if mapping doesn't exist
        """
        if field_name not in self.mappings or original not in self.mappings[field_name]:
            return False

        if field_name not in self.transitivity_markers:
            self.transitivity_markers[field_name] = {}

        self.transitivity_markers[field_name][original] = True

        # Update timestamp
        self._update_timestamps()

        return True

    def remove_mapping(self, field_name: str, original: Any) -> bool:
        """
        Removes a mapping from the store.

        Parameters:
        -----------
        field_name : str
            Name of the field
        original : Any
            Original value

        Returns:
        --------
        bool
            True if mapping was removed, False if not found
        """
        if field_name not in self.mappings or original not in self.mappings[field_name]:
            return False

        # Get synthetic value before removal
        synthetic = self.mappings[field_name][original]

        # Remove from mappings
        del self.mappings[field_name][original]

        # Remove from reverse mappings
        if field_name in self.reverse_mappings and synthetic in self.reverse_mappings[field_name]:
            del self.reverse_mappings[field_name][synthetic]

        # Remove from transitivity markers
        if field_name in self.transitivity_markers and original in self.transitivity_markers[field_name]:
            del self.transitivity_markers[field_name][original]

        # Update metadata
        self.metadata["fields"][field_name]["count"] = len(self.mappings[field_name])
        self._update_timestamps()

        return True

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

    def get_field_stats(self, field_name: str) -> Dict[str, Any]:
        """
        Gets statistics for a field.

        Parameters:
        -----------
        field_name : str
            Name of the field

        Returns:
        --------
        Dict[str, Any]
            Dictionary of field statistics
        """
        if field_name not in self.metadata["fields"]:
            return {
                "count": 0,
                "type": None,
                "transitive_count": 0
            }

        stats = self.metadata["fields"][field_name].copy()

        # Count transitive mappings
        if field_name in self.transitivity_markers:
            stats["transitive_count"] = sum(
                1 for is_transitive in self.transitivity_markers[field_name].values()
                if is_transitive
            )
        else:
            stats["transitive_count"] = 0

        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Gets statistics for all fields.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of statistics for all fields
        """
        stats = {
            "total_fields": len(self.mappings),
            "total_mappings": sum(len(mappings) for mappings in self.mappings.values()),
            "created_at": self.metadata["created_at"],
            "updated_at": self.metadata["updated_at"],
            "fields": {}
        }

        # Get stats for each field
        for field_name in self.mappings:
            stats["fields"][field_name] = self.get_field_stats(field_name)

        return stats

    def clear_field(self, field_name: str) -> None:
        """
        Clears all mappings for a field.

        Parameters:
        -----------
        field_name : str
            Name of the field
        """
        if field_name in self.mappings:
            del self.mappings[field_name]

        if field_name in self.reverse_mappings:
            del self.reverse_mappings[field_name]

        if field_name in self.transitivity_markers:
            del self.transitivity_markers[field_name]

        if field_name in self.metadata["fields"]:
            del self.metadata["fields"][field_name]

        # Update timestamp
        self._update_timestamps()

    def clear_all(self) -> None:
        """
        Clears all mappings from the store.
        """
        self.mappings = {}
        self.reverse_mappings = {}
        self.transitivity_markers = {}
        self.metadata["fields"] = {}

        # Reset timestamp
        self._update_timestamps()

    def _prepare_serializable_data(self) -> Dict[str, Any]:
        """
        Prepares the mapping store data in a serializable format.

        Returns:
        --------
        Dict[str, Any]
            Serializable representation of the mapping store
        """
        serializable_data = {
            "metadata": self.metadata,
            "mappings": {}
        }

        # Convert each field's mappings to a list of pairs for JSON serialization
        for field_name, mappings in self.mappings.items():
            serializable_data["mappings"][field_name] = []

            for original, synthetic in mappings.items():
                is_transitive = self.transitivity_markers.get(field_name, {}).get(original, False)

                mapping_item = {
                    "original": original,
                    "synthetic": synthetic,
                    "is_transitive": is_transitive
                }

                serializable_data["mappings"][field_name].append(mapping_item)

        return serializable_data

    def _process_loaded_data(self, data: Dict[str, Any]) -> None:
        """
        Processes loaded data and initializes the mapping store.

        Parameters:
        -----------
        data : Dict[str, Any]
            Data loaded from a file
        """
        # Clear existing data
        self.mappings = {}
        self.reverse_mappings = {}
        self.transitivity_markers = {}

        # Load metadata
        self.metadata = data.get("metadata", {
            "version": "1.0",
            "fields": {},
            "created_at": None,
            "updated_at": None
        })

        # Load mappings
        for field_name, mapping_items in data.get("mappings", {}).items():
            if field_name not in self.mappings:
                self.mappings[field_name] = {}
                self.reverse_mappings[field_name] = {}
                self.transitivity_markers[field_name] = {}
                self.metadata["fields"][field_name] = {
                    "count": 0,
                    "type": "str"  # Default to string type
                }

            for item in mapping_items:
                original = item["original"]
                synthetic = item["synthetic"]
                is_transitive = item.get("is_transitive", False)

                # Add mapping without updating timestamp
                self.mappings[field_name][original] = synthetic
                self.reverse_mappings[field_name][synthetic] = original
                self.transitivity_markers[field_name][original] = is_transitive

                # Update type information if available
                if isinstance(original, (int, float, bool, str)):
                    self.metadata["fields"][field_name]["type"] = type(original).__name__

            # Update count
            self.metadata["fields"][field_name]["count"] = len(self.mappings[field_name])

    def save_json(self, path: Union[str, Path], io_module=None) -> None:
        """
        Saves the mapping store to a JSON file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the JSON file
        io_module : module, optional
            Legacy parameter for backward compatibility (optional)
        """
        # Prepare serializable data
        serializable_data = self._prepare_serializable_data()

        # Use provided io_module if available (backward compatibility)
        if io_module:
            io_module.write_json(serializable_data, path)
        else:
            # Use standard io module from pamola_core.utils
            pamola_io.write_json(serializable_data, path)

        logger.info(f"Saved mapping store to JSON: {path}")

    def load_json(self, path: Union[str, Path], io_module=None) -> None:
        """
        Loads the mapping store from a JSON file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the JSON file
        io_module : module, optional
            Legacy parameter for backward compatibility (optional)
        """
        logger.info(f"Loading mapping store from JSON: {path}")

        # Use provided io_module if available (backward compatibility)
        if io_module:
            data = io_module.read_json(path)
        else:
            # Use standard io module from pamola_core.utils
            data = pamola_io.read_json(path)

        # Process the loaded data
        self._process_loaded_data(data)

        logger.info(f"Loaded mapping store with {sum(len(m) for m in self.mappings.values())} total mappings")

    def update_from_json(self, path: Union[str, Path],
                         overwrite_existing: bool = True,
                         fields_to_update: Optional[List[str]] = None,
                         io_module=None) -> Dict[str, int]:
        """
        Incrementally updates mappings from a JSON file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the JSON file
        overwrite_existing : bool
            Whether to overwrite existing mappings (default: True)
        fields_to_update : Optional[List[str]]
            List of fields to update (if None, all fields are updated)
        io_module : module, optional
            Legacy parameter for backward compatibility (optional)

        Returns:
        --------
        Dict[str, int]
            Statistics of the update: {field_name: count_added}
        """
        logger.info(f"Incrementally updating mapping store from JSON: {path}")

        # Use provided io_module if available (backward compatibility)
        if io_module:
            data = io_module.read_json(path)
        else:
            # Use standard io module from pamola_core.utils
            data = pamola_io.read_json(path)

        # Process the data
        update_stats = {}

        for field_name, mapping_items in data.get("mappings", {}).items():
            # Skip fields not in the update list if provided
            if fields_to_update is not None and field_name not in fields_to_update:
                continue

            # Initialize field mappings if needed
            if field_name not in self.mappings:
                self.mappings[field_name] = {}
                self.reverse_mappings[field_name] = {}
                self.transitivity_markers[field_name] = {}
                self.metadata["fields"][field_name] = {
                    "count": 0,
                    "type": None
                }

            added_count = 0

            for item in mapping_items:
                original = item["original"]
                synthetic = item["synthetic"]
                is_transitive = item.get("is_transitive", False)

                # Skip if exists and not overwriting
                if not overwrite_existing and original in self.mappings[field_name]:
                    continue

                # Add the mapping
                self.add_mapping(field_name, original, synthetic, is_transitive)
                added_count += 1

            update_stats[field_name] = added_count

        logger.info(f"Updated mapping store with {sum(update_stats.values())} new mappings")
        return update_stats

    def save_pickle(self, path: Union[str, Path]) -> None:
        """
        Saves the mapping store to a pickle file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the pickle file
        """
        # Convert path to Path object
        path = Path(path)

        # Ensure directory exists
        pamola_io.ensure_directory(path.parent)

        # Prepare data to pickle
        data_to_pickle = {
            "mappings": self.mappings,
            "reverse_mappings": self.reverse_mappings,
            "transitivity_markers": self.transitivity_markers,
            "metadata": self.metadata
        }

        # Write to file
        with open(path, 'wb') as f:
            pickle.dump(data_to_pickle, f) # type: ignore

        logger.info(f"Saved mapping store to pickle: {path}")

    def load_pickle(self, path: Union[str, Path]) -> None:
        """
        Loads the mapping store from a pickle file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the pickle file
        """
        logger.info(f"Loading mapping store from pickle: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.mappings = data["mappings"]
        self.reverse_mappings = data["reverse_mappings"]
        self.transitivity_markers = data["transitivity_markers"]
        self.metadata = data["metadata"]

        logger.info(f"Loaded mapping store with {sum(len(m) for m in self.mappings.values())} total mappings")

    def to_dataframe(self, field_name: str) -> pd.DataFrame:
        """
        Converts the mappings for a field to a DataFrame.

        Parameters:
        -----------
        field_name : str
            Name of the field

        Returns:
        --------
        pd.DataFrame
            DataFrame with original and synthetic values
        """
        if field_name not in self.mappings:
            return pd.DataFrame(columns=["original", "synthetic", "is_transitive"])

        data = []

        for original, synthetic in self.mappings[field_name].items():
            is_transitive = self.transitivity_markers.get(field_name, {}).get(original, False)

            data.append({
                "original": original,
                "synthetic": synthetic,
                "is_transitive": is_transitive
            })

        return pd.DataFrame(data)

    def from_dataframe(self, df: pd.DataFrame, field_name: str,
                       original_col: str = "original",
                       synthetic_col: str = "synthetic",
                       transitive_col: str = "is_transitive",
                       overwrite_existing: bool = True) -> int:
        """
        Loads mappings for a field from a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with mappings
        field_name : str
            Name of the field
        original_col : str
            Name of the column with original values
        synthetic_col : str
            Name of the column with synthetic values
        transitive_col : str
            Name of the column with transitivity flags
        overwrite_existing : bool
            Whether to overwrite existing mappings (default: True)

        Returns:
        --------
        int
            Number of mappings added
        """
        if overwrite_existing:
            # Clear existing mappings for the field
            self.clear_field(field_name)

        # Initialize counter
        added_count = 0

        # Add mappings from DataFrame
        for _, row in df.iterrows():
            original = row[original_col]
            synthetic = row[synthetic_col]
            is_transitive = row.get(transitive_col, False)

            # Skip if exists and not overwriting
            if not overwrite_existing and field_name in self.mappings and original in self.mappings[field_name]:
                continue

            self.add_mapping(field_name, original, synthetic, is_transitive)
            added_count += 1

        return added_count

    def save_csv(self, path: Union[str, Path]) -> None:
        """
        Saves all mappings to a CSV file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the CSV file
        """
        # Convert all mappings to a single DataFrame
        all_records = []

        for field_name in self.mappings:
            for original, synthetic in self.mappings[field_name].items():
                is_transitive = self.transitivity_markers.get(field_name, {}).get(original, False)

                record = {
                    "field_name": field_name,
                    "original": str(original),
                    "synthetic": str(synthetic),
                    "is_transitive": is_transitive,
                    "original_type": str(type(original).__name__)
                }
                all_records.append(record)

        df = pd.DataFrame(all_records)

        # Save using io utility
        pamola_io.write_dataframe_to_csv(df, path)

        logger.info(f"Saved mapping store to CSV: {path} ({len(all_records)} mappings)")

    def load_csv(self, path: Union[str, Path],
                 overwrite_existing: bool = True,
                 fields_to_load: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Loads mappings from a CSV file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the CSV file
        overwrite_existing : bool
            Whether to overwrite existing mappings (default: True)
        fields_to_load : Optional[List[str]]
            List of fields to load (if None, all fields are loaded)

        Returns:
        --------
        Dict[str, int]
            Statistics of the load: {field_name: count_added}
        """
        logger.info(f"Loading mapping store from CSV: {path}")

        # Read the CSV file
        df = pamola_io.read_full_csv(path)

        load_stats = {}
        processed_fields = set()

        for _, row in df.iterrows():
            field_name = row["field_name"]

            # Skip fields not in the load list if provided
            if fields_to_load is not None and field_name not in fields_to_load:
                continue

            # Initialize stats for this field
            if field_name not in processed_fields:
                processed_fields.add(field_name)
                load_stats[field_name] = 0

            # Extract values
            original_str = row["original"]
            synthetic_str = row["synthetic"]
            is_transitive = row.get("is_transitive", False)
            original_type = row.get("original_type", "str")

            # Convert back to original type if possible
            try:
                if original_type == "int":
                    original = int(original_str)
                elif original_type == "float":
                    original = float(original_str)
                elif original_type == "bool":
                    original = original_str.lower() == "true"
                else:
                    original = original_str
            except (ValueError, TypeError):
                # Fallback to string if conversion fails
                original = original_str

            # Skip if exists and not overwriting
            if not overwrite_existing and field_name in self.mappings and original in self.mappings[field_name]:
                continue

            # Add the mapping
            self.add_mapping(field_name, original, synthetic_str, is_transitive)
            load_stats[field_name] += 1

        logger.info(f"Loaded {sum(load_stats.values())} mappings from CSV")
        return load_stats

    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Saves the mapping store to a file in the specified format.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the file
        format : str
            Format to save in: "json", "pickle", "csv" (default: "json")
        """
        path = Path(path)

        # Ensure directory exists
        pamola_io.ensure_directory(path.parent)

        if format.lower() == "json":
            self.save_json(path)
        elif format.lower() == "pickle":
            self.save_pickle(path)
        elif format.lower() == "csv":
            self.save_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, path: Union[str, Path], format: str = None,
             overwrite_existing: bool = True,
             fields_to_load: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Loads the mapping store from a file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the file
        format : str, optional
            Format to load from: "json", "pickle", "csv"
            If None, inferred from file extension
        overwrite_existing : bool
            Whether to overwrite existing mappings (default: True)
        fields_to_load : Optional[List[str]]
            List of fields to load (if None, all fields are loaded)

        Returns:
        --------
        Dict[str, int]
            Statistics of the load for CSV format: {field_name: count_added}
            Empty dict for other formats
        """
        path = Path(path)

        # Infer format from extension if not specified
        if format is None:
            if path.suffix.lower() == ".json":
                format = "json"
            elif path.suffix.lower() == ".pkl" or path.suffix.lower() == ".pickle":
                format = "pickle"
            elif path.suffix.lower() == ".csv":
                format = "csv"
            else:
                raise ValueError(f"Could not infer format from file extension: {path}")

        # If not overwriting, we need to use incremental methods
        if not overwrite_existing and format.lower() == "json":
            return self.update_from_json(path, overwrite_existing, fields_to_load)
        elif not overwrite_existing and format.lower() == "csv":
            return self.load_csv(path, overwrite_existing, fields_to_load)

        # Otherwise, load the full file (overwriting existing data)
        if format.lower() == "json":
            self.load_json(path)
            return {}
        elif format.lower() == "pickle":
            self.load_pickle(path)
            return {}
        elif format.lower() == "csv":
            return self.load_csv(path, True, fields_to_load)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def merge_with_store(self, other_store: 'MappingStore',
                         overwrite_existing: bool = True,
                         fields_to_merge: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Merges another MappingStore into this one.

        Parameters:
        -----------
        other_store : MappingStore
            Other mapping store to merge from
        overwrite_existing : bool
            Whether to overwrite existing mappings (default: True)
        fields_to_merge : Optional[List[str]]
            List of fields to merge (if None, all fields are merged)

        Returns:
        --------
        Dict[str, int]
            Statistics of the merge: {field_name: count_added}
        """
        logger.info("Merging mapping stores")

        merge_stats = {}

        # Determine which fields to merge
        fields = fields_to_merge or list(other_store.mappings.keys())

        for field_name in fields:
            if field_name not in other_store.mappings:
                merge_stats[field_name] = 0
                continue

            # Initialize field mappings if needed
            if field_name not in self.mappings:
                self.mappings[field_name] = {}
                self.reverse_mappings[field_name] = {}
                self.transitivity_markers[field_name] = {}
                self.metadata["fields"][field_name] = {
                    "count": 0,
                    "type": other_store.metadata.get("fields", {}).get(field_name, {}).get("type")
                }

            added_count = 0

            for original, synthetic in other_store.mappings[field_name].items():
                # Skip if exists and not overwriting
                if not overwrite_existing and original in self.mappings[field_name]:
                    continue

                # Get transitivity from other store
                is_transitive = other_store.transitivity_markers.get(field_name, {}).get(original, False)

                # Add the mapping
                self.add_mapping(field_name, original, synthetic, is_transitive)
                added_count += 1

            merge_stats[field_name] = added_count

        logger.info(f"Merged mapping stores: added {sum(merge_stats.values())} mappings")
        return merge_stats