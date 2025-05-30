"""
Transitivity handler for complex mapping scenarios.

This module provides functionality for handling transitive mappings
and resolving conflicts in mapping chains. It detects and manages cycles,
analyzes mapping relationships, and ensures data consistency.
"""

from typing import Any, Dict, List, Set

from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils import logging as pamola_logging

# Configure logger
logger = pamola_logging.get_logger("pamola_core.fake_data.mappers.transitivity_handler")


class TransitivityHandler:
    """
    Handles transitive relationships in mappings.

    Provides utilities for detecting and resolving complex mapping
    relationships, including cycles and chains. Ensures mapping
    consistency when multiple levels of indirection are present.

    Key features:
    - Detection of mapping chains (A->B->C)
    - Identification of mapping cycles (A->B->C->A)
    - Resolution of cycles using different strategies
    - Analysis of transitive relationships
    - Automatic correction of inconsistent mappings
    """

    def __init__(self, mapping_store: MappingStore):
        """
        Initialize the transitivity handler.

        Parameters:
        -----------
        mapping_store : MappingStore
            Store containing the mappings to analyze and manipulate
        """
        self.mapping_store = mapping_store

    def find_mapping_chain(self, field_name: str, original_value: Any) -> List[Any]:
        """
        Finds the chain of transitive mappings starting from an original value.

        A mapping chain is a sequence of values where each maps to the next,
        potentially creating complex relationships. For example, if A maps to B,
        B maps to C, and C maps to D, the chain is [A, B, C, D].

        Parameters:
        -----------
        field_name : str
            Name of the field to analyze
        original_value : Any
            Starting original value

        Returns:
        --------
        List[Any]
            Chain of values in the mapping sequence, starting with original_value
        """
        chain = [original_value]

        # Get the first mapping
        current_value = original_value
        visited = {original_value}

        while True:
            # Get the next value in the chain
            next_value = self.mapping_store.get_mapping(field_name, current_value)

            if next_value is None:
                # End of chain
                break

            chain.append(next_value)

            # Check if we've reached a value that already maps to something
            original_for_next = self.mapping_store.restore_original(field_name, next_value)

            if original_for_next is not None and original_for_next != current_value:
                # This value is already an original in another mapping
                # Check if we've seen it before (cycle detection)
                if original_for_next in visited:
                    # We've found a cycle
                    cycle_start = chain.index(original_for_next)
                    logger.warning(f"Cycle detected in mapping chain: {chain[cycle_start:]} -> {original_for_next}")
                    chain.append(original_for_next)  # Add it again to show the cycle
                    break

                # We've found a connection to another chain, continue following
                chain.append(original_for_next)
                current_value = original_for_next
                visited.add(original_for_next)
            else:
                # End of chain
                break

            # Update current value
            current_value = next_value
            visited.add(current_value)

        return chain

    def find_cycles(self, field_name: str) -> List[List[Any]]:
        """
        Finds all cycles in the mapping relationships.

        A mapping cycle occurs when following mappings eventually leads back
        to the starting value. For example, A->B->C->A is a cycle. Cycles can
        cause problems in mapping systems and need to be resolved.

        Parameters:
        -----------
        field_name : str
            Name of the field to analyze

        Returns:
        --------
        List[List[Any]]
            List of cycles found, each represented as a list of values
        """
        mappings = self.mapping_store.get_field_mappings(field_name)
        cycles = []

        # Keep track of visited values
        visited = set()

        for original in mappings:
            if original in visited:
                continue

            # Perform DFS to find cycles
            path = []
            path_set = set()

            self._dfs_find_cycles(field_name, original, mappings, path, path_set, visited, cycles)

        return cycles

    def _dfs_find_cycles(self,
                         field_name: str,
                         current: Any,
                         mappings: Dict[Any, Any],
                         path: List[Any],
                         path_set: Set[Any],
                         visited: Set[Any],
                         cycles: List[List[Any]]):
        """
        Helper for cycle detection using depth-first search.

        This recursive method explores the mapping graph depth-first, tracking
        the current path to detect cycles.

        Parameters:
        -----------
        field_name : str
            Field name being analyzed
        current : Any
            Current value being evaluated
        mappings : Dict[Any, Any]
            Mapping dictionary (original->synthetic)
        path : List[Any]
            Current path in the traversal
        path_set : Set[Any]
            Set of values in current path (for O(1) lookups)
        visited : Set[Any]
            Set of all visited values
        cycles : List[List[Any]]
            List of cycles found (modified in-place)
        """
        path.append(current)
        path_set.add(current)
        visited.add(current)

        # Get the next value in the mapping
        next_value = mappings.get(current)

        if next_value is not None:
            if next_value in path_set:
                # Cycle found
                cycle_start = path.index(next_value)
                cycle = path[cycle_start:] + [next_value]
                cycles.append(cycle)
            elif next_value not in visited:
                # Continue DFS
                self._dfs_find_cycles(field_name, next_value, mappings, path, path_set, visited, cycles)

        # Check if the synthetic value maps back to an original value
        orig_of_next = self.mapping_store.restore_original(field_name, next_value) if next_value is not None else None

        if orig_of_next is not None and orig_of_next != current:
            if orig_of_next in path_set:
                # Cycle found
                cycle_start = path.index(orig_of_next)
                cycle = path[cycle_start:] + [next_value, orig_of_next]
                cycles.append(cycle)
            elif orig_of_next not in visited:
                # Continue DFS with the original of the next value
                self._dfs_find_cycles(field_name, orig_of_next, mappings, path, path_set, visited, cycles)

        # Backtrack
        path.pop()
        path_set.remove(current)

    def resolve_cycle(self, field_name: str, cycle: List[Any], strategy: str = "break_at_start") -> None:
        """
        Resolves a cycle in the mappings.

        Breaks a cycle using the specified strategy to ensure mapping consistency.
        Different strategies can be applied based on the specific requirements.

        Parameters:
        -----------
        field_name : str
            Field name
        cycle : List[Any]
            Cycle to resolve
        strategy : str
            Resolution strategy:
            - "break_at_start": Break the cycle at the first element
            - "break_at_end": Break the cycle at the last element
            - "break_longest": Break the link that creates the longest chain

        Raises:
        -------
        ValueError
            If the strategy is not recognized
        """
        if not cycle or len(cycle) < 2:
            logger.warning(f"Cycle too short to resolve: {cycle}")
            return

        # Make sure it's a proper cycle
        if cycle[0] != cycle[-1]:
            cycle.append(cycle[0])

        if strategy == "break_at_start":
            # Remove the mapping from the last to the first element
            to_remove = cycle[-2]
            remove_target = cycle[-1]  # Should be same as cycle[0]

            logger.info(f"Breaking cycle by removing mapping: {to_remove} -> {remove_target}")

            # Remove the mapping
            self.mapping_store.remove_mapping(field_name, to_remove)

        elif strategy == "break_at_end":
            # Remove the mapping from the second-to-last to the last element
            to_remove = cycle[-2]
            remove_target = cycle[-1]

            logger.info(f"Breaking cycle by removing mapping: {to_remove} -> {remove_target}")

            # Remove the mapping
            self.mapping_store.remove_mapping(field_name, to_remove)

        elif strategy == "break_longest":
            # Find the link that, if broken, leaves the longest chain
            longest_chain = 0
            break_from = None
            break_to = None

            for i in range(len(cycle) - 1):
                # Try breaking the link from cycle[i] to cycle[i+1]
                chain_length = len(cycle) - 2  # -2 because we remove one link and the repeated cycle[0]

                if chain_length > longest_chain:
                    longest_chain = chain_length
                    break_from = cycle[i]
                    break_to = cycle[i + 1]

            if break_from is not None and break_to is not None:
                logger.info(f"Breaking cycle at longest chain: {break_from} -> {break_to}")

                # Remove the mapping
                self.mapping_store.remove_mapping(field_name, break_from)
            else:
                logger.warning(f"Could not find suitable link to break in cycle: {cycle}")

        else:
            raise ValueError(f"Unknown cycle resolution strategy: {strategy}")

    def resolve_all_cycles(self, field_name: str, strategy: str = "break_at_start") -> int:
        """
        Finds and resolves all cycles in the mappings.

        Automatically detects and resolves all cycles using the specified strategy.
        This is a higher-level method that wraps find_cycles and resolve_cycle.

        Parameters:
        -----------
        field_name : str
            Field name
        strategy : str
            Resolution strategy

        Returns:
        --------
        int
            Number of cycles resolved
        """
        cycles = self.find_cycles(field_name)

        if not cycles:
            logger.info(f"No cycles found for field {field_name}")
            return 0

        logger.info(f"Found {len(cycles)} cycles in field {field_name}, resolving with strategy '{strategy}'")

        for cycle in cycles:
            self.resolve_cycle(field_name, cycle, strategy)

        return len(cycles)

    def analyze_mapping_relationships(self, field_name: str) -> Dict[str, Any]:
        """
        Analyzes the relationships between mappings for a field.

        Provides detailed information about the mapping structure, including
        chains, cycles, and transitivity statistics.

        Parameters:
        -----------
        field_name : str
            Field name to analyze

        Returns:
        --------
        Dict[str, Any]
            Analysis results including:
            - chains_count: Number of mapping chains
            - max_chain_length: Length of the longest chain
            - cycles_count: Number of cycles detected
            - transitive_mappings: Count of transitive mappings
            - problematic_mappings: List of potentially problematic mappings
        """
        # Get all mappings for the field
        all_mappings = self.mapping_store.get_field_mappings(field_name)

        # Initialize analysis results
        results = {
            "chains_count": 0,
            "max_chain_length": 0,
            "cycles_count": 0,
            "transitive_mappings": 0,
            "problematic_mappings": []
        }

        # Find cycles
        cycles = self.find_cycles(field_name)
        results["cycles_count"] = len(cycles)

        # Analyze chains
        visited = set()
        for original in all_mappings:
            if original in visited:
                continue

            # Find the chain starting from this original
            chain = self.find_mapping_chain(field_name, original)

            # Add all values in the chain to visited
            visited.update(chain)

            # Update statistics
            if len(chain) > 1:  # Only count non-trivial chains
                results["chains_count"] += 1
                results["max_chain_length"] = max(results["max_chain_length"], len(chain))

                # If chain length > 2, intermediate mappings are transitive
                if len(chain) > 2:
                    results["transitive_mappings"] += len(chain) - 2

        # Count explicitly marked transitive mappings
        marked_transitive = 0
        for original in all_mappings:
            if self.mapping_store.is_transitive(field_name, original):
                marked_transitive += 1

        # Check for discrepancies between detected and marked transitive mappings
        results["marked_transitive_mappings"] = marked_transitive
        if marked_transitive != results["transitive_mappings"]:
            results["transitivity_marking_issues"] = True

            # Find mappings that should be marked as transitive but aren't
            for original in all_mappings:
                synthetic = all_mappings[original]
                if self.mapping_store.restore_original(field_name,
                                                       synthetic) is not None and not self.mapping_store.is_transitive(
                        field_name, original):
                    results["problematic_mappings"].append({
                        "original": original,
                        "synthetic": synthetic,
                        "issue": "should_be_transitive"
                    })

        return results

    def fix_transitive_mappings(self, field_name: str) -> int:
        """
        Identifies and fixes issues with transitive mappings.

        Ensures that all mappings that should be marked as transitive are
        correctly marked, and resolves any inconsistencies.

        Parameters:
        -----------
        field_name : str
            Field name

        Returns:
        --------
        int
            Number of issues fixed
        """
        # First, resolve any cycles
        cycles_resolved = self.resolve_all_cycles(field_name)

        # Now check for transitive chains and ensure they're properly marked
        mappings = self.mapping_store.get_field_mappings(field_name)
        issues_fixed = 0

        for original in mappings.keys():
            # Skip if already marked as transitive
            if self.mapping_store.is_transitive(field_name, original):
                continue

            synthetic = mappings[original]

            # Check if the synthetic value is an original in another mapping
            orig_of_synthetic = self.mapping_store.restore_original(field_name, synthetic)

            if orig_of_synthetic is not None and orig_of_synthetic != original:
                # This is a transitive mapping, but it's not marked as such
                logger.info(f"Marking transitive mapping: {original} -> {synthetic} -> ...")
                self.mapping_store.mark_as_transitive(field_name, original)
                issues_fixed += 1

        logger.info(f"Fixed {issues_fixed} transitive mapping issues for field {field_name}")
        return cycles_resolved + issues_fixed

    def get_statistics(self, field_name: str) -> Dict[str, Any]:
        """
        Gets comprehensive statistics about transitivity for a field.

        Provides detailed metrics about the transitivity characteristics
        of the mappings for analysis and monitoring.

        Parameters:
        -----------
        field_name : str
            Field name to analyze

        Returns:
        --------
        Dict[str, Any]
            Statistics including:
            - total_mappings: Total number of mappings
            - transitive_mappings: Number of transitive mappings
            - transitive_percentage: Percentage of transitive mappings
            - chains: Information about mapping chains
            - cycles: Information about mapping cycles
        """
        # Get basic stats from the mapping store
        field_stats = self.mapping_store.get_field_stats(field_name)

        # Initialize statistics
        statistics = {
            "total_mappings": field_stats.get("count", 0),
            "transitive_mappings": field_stats.get("transitive_count", 0),
            "transitive_percentage": 0.0,
            "chains": {
                "count": 0,
                "max_length": 0,
                "avg_length": 0.0
            },
            "cycles": {
                "count": 0
            }
        }

        # Calculate transitive percentage
        if statistics["total_mappings"] > 0:
            statistics["transitive_percentage"] = (statistics["transitive_mappings"] / statistics[
                "total_mappings"]) * 100.0

        # Find cycles
        cycles = self.find_cycles(field_name)
        statistics["cycles"]["count"] = len(cycles)

        # Analyze chains (this is computationally expensive for large mappings)
        if statistics["total_mappings"] <= 10000:  # Limit analysis for very large mappings
            mappings = self.mapping_store.get_field_mappings(field_name)
            visited = set()
            chain_lengths = []

            for original in mappings:
                if original in visited:
                    continue

                # Find the chain starting from this original
                chain = self.find_mapping_chain(field_name, original)

                # Add all values in the chain to visited
                visited.update(chain)

                # Update statistics
                if len(chain) > 1:  # Only count non-trivial chains
                    statistics["chains"]["count"] += 1
                    statistics["chains"]["max_length"] = max(statistics["chains"]["max_length"], len(chain))
                    chain_lengths.append(len(chain))

            # Calculate average chain length
            if chain_lengths:
                statistics["chains"]["avg_length"] = sum(chain_lengths) / float(len(chain_lengths))
        else:
            # For large mappings, use a more limited analysis
            statistics["chains"]["note"] = "Limited analysis due to large mapping count"

            # Sample a subset of mappings for analysis
            sample_size = min(1000, statistics["total_mappings"])
            mappings = self.mapping_store.get_field_mappings(field_name)
            sampled_originals = list(mappings.keys())[:sample_size]

            # Analyze the sample
            longest_chain = 0
            chain_count = 0
            chain_lengths = []

            for original in sampled_originals:
                chain = self.find_mapping_chain(field_name, original)
                if len(chain) > 1:
                    chain_count += 1
                    longest_chain = max(longest_chain, len(chain))
                    chain_lengths.append(len(chain))

            # Extrapolate results
            statistics["chains"]["estimated_count"] = int(
                chain_count * (statistics["total_mappings"] / float(sample_size)))
            statistics["chains"]["max_length"] = longest_chain
            if chain_lengths:
                statistics["chains"]["avg_length"] = sum(chain_lengths) / float(len(chain_lengths))
            else:
                statistics["chains"]["avg_length"] = 0.0

        return statistics