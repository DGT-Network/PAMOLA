"""
Pseudo-Random Generation module for deterministic fake data generation.

This module provides utilities for generating deterministic values based on
original data, ensuring consistent replacements across multiple runs while
maintaining the appearance of randomness.
"""

import functools
import hashlib
import hmac
import random
import struct
from typing import Any, Dict, List, Optional, Union, Callable

from pamola_core.utils import logging

# Configure logger
logger = logging.get_logger("pamola_core.fake_data.commons.prgn")


class PRNGenerator:
    """
    Pseudo-Random Number Generator for deterministic data generation.

    This class provides methods for generating deterministic values based on
    input seeds, ensuring that the same input always produces the same output
    while maintaining random-like properties.

    Examples:
    ---------
    ```
    # Create a generator with a global seed
    generator = PRNGenerator(global_seed="project-seed-2023")

    # Generate a deterministic selection from a list
    selected_name = generator.select_from_list(
        names_list,
        base_value="John",
        salt="names-v1"
    )
    ```
    """

    def __init__(self, global_seed: Optional[Union[str, bytes, int]] = None):
        """
        Initialize the PRN Generator.

        Parameters:
        -----------
        global_seed : Optional[Union[str, bytes, int]]
            Global seed to use for deterministic generation
        """
        self.global_seed = self._normalize_seed(global_seed)
        self.random_state = random.Random(self.global_seed)

    def _normalize_seed(self, seed: Optional[Union[str, bytes, int]]) -> int:
        """
        Normalizes various seed types to an integer value.

        Parameters:
        -----------
        seed : Optional[Union[str, bytes, int]]
            Seed in various formats

        Returns:
        --------
        int
            Normalized seed as integer
        """
        if seed is None:
            return random.randrange(2 ** 32)

        if isinstance(seed, int):
            return seed

        if isinstance(seed, str):
            seed = seed.encode('utf-8')

        if isinstance(seed, bytes):
            # Create a consistent integer from bytes
            seed_int = 0
            for byte in seed:
                seed_int = ((seed_int << 8) | byte) & 0xFFFFFFFF
            return seed_int

        # Fallback
        return hash(seed) & 0xFFFFFFFF

    @functools.lru_cache(maxsize=1024)
    def generate_with_seed(self,
                           base_value: Any,
                           salt: Union[str, bytes, None] = None,
                           algorithm: str = "hmac") -> int:
        """
        Generates a deterministic integer value based on input value and salt.

        Parameters:
        -----------
        base_value : Any
            Base value to generate from
        salt : Union[str, bytes, None]
            Salt for additional entropy
        algorithm : str
            Algorithm to use ('hmac', 'simple', or 'fast')

        Returns:
        --------
        int
            Deterministic integer value

        Example:
        --------
        ```python
        # Generate a deterministic hash for "John"
        seed_value = generator.generate_with_seed("John", salt="names")
        ```
        """
        # Convert base_value to string if needed
        if not isinstance(base_value, (str, bytes)):
            base_value = str(base_value)

        if isinstance(base_value, str):
            base_value = base_value.encode('utf-8')

        # Apply salt if provided
        if salt is not None:
            if isinstance(salt, str):
                salt = salt.encode('utf-8')

            if algorithm == "hmac":
                # Use HMAC with global seed as key and salt+value as message
                key = struct.pack("I", self.global_seed)
                msg = salt + base_value
                h = hmac.new(key, msg, digestmod=hashlib.sha256)
                digest = h.digest()
                return int.from_bytes(digest[:4], byteorder='big')
            elif algorithm == "fast":
                # Faster but less cryptographically secure algorithm
                combined = salt + base_value
                hash_val = 5381  # Initial value from djb2 hash
                for byte in combined:
                    hash_val = ((hash_val * 33) ^ byte) & 0xFFFFFFFF
                return (hash_val ^ self.global_seed) & 0xFFFFFFFF
            else:  # "simple"
                # Simple algorithm: combine base value and salt with global seed
                combined = salt + base_value
                local_seed = (self._normalize_seed(combined) ^ self.global_seed) & 0xFFFFFFFF
                return local_seed
        else:
            # No salt provided
            if algorithm == "hmac":
                key = struct.pack("I", self.global_seed)
                h = hmac.new(key, base_value, digestmod=hashlib.sha256)
                digest = h.digest()
                return int.from_bytes(digest[:4], byteorder='big')
            elif algorithm == "fast":
                # Fast algorithm without salt
                hash_val = 5381
                for byte in base_value:
                    hash_val = ((hash_val * 33) ^ byte) & 0xFFFFFFFF
                return (hash_val ^ self.global_seed) & 0xFFFFFFFF
            else:  # "simple"
                # Use hash of base_value combined with global seed
                local_seed = (self._normalize_seed(base_value) ^ self.global_seed) & 0xFFFFFFFF
                return local_seed

    def get_random_by_value(self, base_value: Any, salt: Optional[Union[str, bytes]] = None) -> random.Random:
        """
        Gets a random number generator seeded by a deterministic value.

        Parameters:
        -----------
        base_value : Any
            Base value to derive seed from
        salt : Optional[Union[str, bytes]]
            Optional salt for additional entropy

        Returns:
        --------
        random.Random
            Deterministic random generator

        Example:
        --------
        ```python
        # Get a random number generator seeded by a name
        rng = generator.get_random_by_value("John", salt="names")
        random_number = rng.randint(1, 100)
        ```
        """
        seed = self.generate_with_seed(base_value, salt)
        return random.Random(seed)

    def select_from_list(self,
                         items: List[Any],
                         base_value: Any,
                         salt: Optional[Union[str, bytes]] = None) -> Any:
        """
        Selects an item from a list deterministically based on input value.

        Parameters:
        -----------
        items : List[Any]
            List of items to select from
        base_value : Any
            Base value for selection
        salt : Optional[Union[str, bytes]]
            Optional salt for additional entropy

        Returns:
        --------
        Any
            Selected item

        Example:
        --------
        ```python
        # Select a replacement name deterministically
        new_name = generator.select_from_list(
            ["Alice", "Bob", "Charlie"],
            "John",
            salt="first-names"
        )
        ```
        """
        if not items:
            logger.warning("Cannot select from empty list")
            return None

        # Get a random generator for this specific value
        rng = self.get_random_by_value(base_value, salt)

        # Select an item deterministically
        index = rng.randrange(len(items))
        return items[index]

    def select_with_mapping(self,
                            mapping: Dict[Any, Any],
                            base_value: Any,
                            fallback_generator: Optional[Callable[[Any], Any]] = None,
                            salt: Optional[Union[str, bytes]] = None) -> Any:
        """
        Selects from a mapping if available, or generates a new value.

        Parameters:
        -----------
        mapping : Dict[Any, Any]
            Existing mappings
        base_value : Any
            Original value
        fallback_generator : Optional[Callable]
            Function to generate a value if not in mapping
        salt : Optional[Union[str, bytes]]
            Optional salt for additional entropy

        Returns:
        --------
        Any
            Selected or generated value

        Example:
        --------
        ```python
        # Use existing mapping or generate new value
        replacement = generator.select_with_mapping(
            known_mappings,
            original_name,
            lambda x: generator.select_from_list(names_list, x)
        )
        ```
        """
        # Check if value already exists in mapping
        if base_value in mapping:
            return mapping[base_value]

        # If no fallback generator provided, return None
        if fallback_generator is None:
            return None

        # Generate a new value
        return fallback_generator(base_value)

    def shuffle_list(self,
                     items: List[Any],
                     base_value: Any,
                     salt: Optional[Union[str, bytes]] = None) -> List[Any]:
        """
        Shuffles a list deterministically based on input value.

        Parameters:
        -----------
        items : List[Any]
            List to shuffle
        base_value : Any
            Base value for shuffling
        salt : Optional[Union[str, bytes]]
            Optional salt for additional entropy

        Returns:
        --------
        List[Any]
            Shuffled list

        Example:
        --------
        ```python
        # Shuffle a list of names deterministically
        shuffled_names = generator.shuffle_list(
            ["Alice", "Bob", "Charlie"],
            "seed_value",
            salt="shuffle-v1"
        )
        ```
        """
        if not items:
            return []

        # Create a copy to avoid modifying the original
        result = items.copy()

        # Get a random generator for this value
        rng = self.get_random_by_value(base_value, salt)

        # Shuffle the list
        rng.shuffle(result)

        return result

    def select_name_by_gender_region(self,
                                     names_dict: Dict[str, Dict[str, List[str]]],
                                     original_name: str,
                                     gender: str,
                                     region: str,
                                     salt: Optional[str] = None) -> str:
        """
        Deterministically selects a name based on gender and region.

        Parameters:
        -----------
        names_dict : Dict[str, Dict[str, List[str]]]
            Dictionary of name lists organized by region and gender
        original_name : str
            Original name to be replaced
        gender : str
            Gender code (e.g., 'M', 'F')
        region : str
            Region/language code (e.g., 'ru', 'en')
        salt : Optional[str]
            Optional salt for additional entropy

        Returns:
        --------
        str
            Selected replacement name

        Example:
        --------
        ```python
        # Select a name based on gender and region
        new_name = generator.select_name_by_gender_region(
            names_dictionary,
            "Ivan",
            gender="M",
            region="ru",
            salt="names-v2"
        )
        ```
        """
        # Get the appropriate list of names for the region and gender
        candidates = names_dict.get(region, {}).get(gender, [])

        if not candidates:
            logger.warning(f"No names available for region={region}, gender={gender}")
            return original_name

        # Use existing method to select from the list deterministically
        return self.select_from_list(candidates, original_name, salt)


def generate_deterministic_replacement(
        original_value: Any,
        replacement_list: List[Any],
        global_seed: Optional[Union[str, bytes, int]] = None,
        salt: Optional[Union[str, bytes]] = None
) -> Any:
    """
    Generates a deterministic replacement from a list based on original value.

    Parameters:
    -----------
    original_value : Any
        Original value to replace
    replacement_list : List[Any]
        List of potential replacements
    global_seed : Optional[Union[str, bytes, int]]
        Global seed for deterministic generation
    salt : Optional[Union[str, bytes]]
        Optional salt for additional entropy

    Returns:
    --------
    Any
        Deterministic replacement value

    Example:
    --------
    ```python
    # Replace a name deterministically
    new_name = generate_deterministic_replacement(
        "John",
        ["Alice", "Bob", "Charlie"],
        global_seed="global-project-seed",
        salt="names-salt"
    )
    ```
    """
    if not replacement_list:
        return None

    generator = PRNGenerator(global_seed)
    return generator.select_from_list(replacement_list, original_value, salt)

def generate_seed_from_key(key: Union[str, bytes], context: str = "") -> int:
    """
    Generates a seed value from a key and context.

    Parameters:
    -----------
    key : Union[str, bytes]
        Key for seed generation
    context : str
        Optional context for different seeds from same key

    Returns:
    --------
    int
        Seed value

    Example:
    --------
    ```python
    # Generate a seed from a project key and context
    seed = PRNGenerator.generate_seed_from_key(
        "project-api-key",
        context="names-generation"
    )
    ```
    """
    if isinstance(key, str):
        key = key.encode('utf-8')

    if context:
        key = key + context.encode('utf-8')

    # Create a hash of the key
    hash_obj = hashlib.sha256(key)
    hash_bytes = hash_obj.digest()

    # Convert first 4 bytes to integer
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')

    return seed