"""
Organization generation operation for fake data system.

This module provides the OrganizationOperation class for generating synthetic
organization names while maintaining statistical properties of the original data
and supporting consistent mapping.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pamola_core.fake_data.generators.organization import OrganizationGenerator
from pamola_core.fake_data.schemas.organization_op_config import FakeOrganizationOperationConfig
from pamola_core.utils import io
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult
from pamola_core.utils.progress import HierarchicalProgressTracker


@register(version="1.0.0")
class FakeOrganizationOperation(GeneratorOperation):
    """
    Operation for generating synthetic organization names.

    This operation processes organization names in datasets, replacing them with
    synthetic alternatives while preserving type characteristics and regional
    specificity.
    """

    def __init__(
        self,
        field_name: str,
        organization_type: str = "general",
        dictionaries: Optional[Dict[str, str]] = None,
        prefixes: Optional[Dict[str, str]] = None,
        suffixes: Optional[Dict[str, str]] = None,
        add_prefix_probability: float = 0.3,
        add_suffix_probability: float = 0.5,
        region: str = "en",
        preserve_type: bool = True,
        industry: Optional[str] = None,
        collect_type_distribution: bool = True,
        type_field: Optional[str] = None,
        region_field: Optional[str] = None,
        detailed_metrics: bool = False,
        max_retries: int = 3,
        key: Optional[str] = None,
        context_salt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize FakeOrganizationOperation.

        Parameters
        ----------
        field_name : str
            Field containing organization names.
        organization_type : str, optional
            Type of organization to generate (e.g., 'general', 'industry', 'government').
        dictionaries : dict, optional
            Custom organization dictionaries.
        prefixes : dict, optional
            Prefix dictionary for organization names.
        suffixes : dict, optional
            Suffix dictionary for organization names.
        add_prefix_probability : float, optional
            Probability of adding a prefix.
        add_suffix_probability : float, optional
            Probability of adding a suffix.
        region : str, optional
            Region code for localized name generation.
        preserve_type : bool, optional
            Whether to preserve the type of organization.
        industry : str, optional
            Specific industry name for contextual organization generation.
        collect_type_distribution : bool, optional
            Whether to collect distribution statistics per organization type.
        type_field : str, optional
            Field in the dataset containing organization type codes.
        region_field : str, optional
            Field in the dataset containing region codes.
        detailed_metrics : bool, optional
            Whether to enable detailed generation statistics.
        max_retries : int, optional
            Maximum retries for failed name generations.
        key : str, optional
            Encryption or PRGN key for consistent pseudonymization.
        context_salt : str, optional
            Contextual salt for PRGN uniqueness.
        **kwargs : dict
            Additional parameters forwarded to GeneratorOperation and BaseOperation.
        """

        # Default description
        kwargs.setdefault(
            "description",
            f"Synthetic organization name generation for '{field_name}' with regional and type preservation",
        )

        # Build configuration object
        config = FakeOrganizationOperationConfig(
            field_name=field_name,
            organization_type=organization_type,
            dictionaries=dictionaries,
            prefixes=prefixes,
            suffixes=suffixes,
            add_prefix_probability=add_prefix_probability,
            add_suffix_probability=add_suffix_probability,
            region=region,
            preserve_type=preserve_type,
            industry=industry,
            collect_type_distribution=collect_type_distribution,
            type_field=type_field,
            region_field=region_field,
            detailed_metrics=detailed_metrics,
            max_retries=max_retries,
            key=key,
            context_salt=context_salt,
            **kwargs,
        )

        # Generator-specific configuration
        generator_params = {
            "organization_type": organization_type,
            "dictionaries": dictionaries or {},
            "prefixes": prefixes or {},
            "suffixes": suffixes or {},
            "add_prefix_probability": add_prefix_probability,
            "add_suffix_probability": add_suffix_probability,
            "region": region,
            "preserve_type": preserve_type,
            "industry": industry,
            "key": key,
            "context_salt": context_salt,
        }

        # Initialize generator
        organization_generator = OrganizationGenerator(config=generator_params)

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize parent class
        super().__init__(
            field_name=field_name,
            generator=organization_generator,
            generator_params=generator_params,
            **kwargs,
        )

        # Expose config fields as attributes
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Initialize runtime stats
        self.operation_name = self.__class__.__name__
        self._original_df = None
        self.error_count = 0
        self.retry_count = 0

        # Optional detailed metrics
        if self.detailed_metrics:
            self._type_stats = Counter()
            self._region_stats = Counter()
            self._prefix_suffix_stats = {
                "with_prefix": 0,
                "with_suffix": 0,
                "with_both": 0,
                "with_neither": 0,
            }
            self._generation_times = []

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the operation with timing and error handling.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Config logger task for operation
        self.logger = kwargs.get("logger", self.logger)

        # Start timing for performance metrics
        self.start_time = time.time()
        self.logger.info(
            f"Starting {self.operation_name} operation at {self.start_time}"
        )

        # Call parent execute method
        result = super().execute(
            data_source, task_dir, reporter, progress_tracker, **kwargs
        )

        # Save mapping if requested
        if self.save_mapping and hasattr(self, "mapping_store") and self.mapping_store:
            mapping_dir = Path(task_dir) / "maps"
            io.ensure_directory(mapping_dir)
            mapping_path = mapping_dir / f"{self.name}_{self.field_name}_mapping.json"
            self.mapping_store.save_json(mapping_path)
            self.logger.info(f"Saved mapping to {Path(mapping_path).name}")

        return result

    def process_batch(self, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of data to generate synthetic organization names.

        Args:
            batch: DataFrame batch to process
            kwargs: Additional parameters

        Returns:
            Processed DataFrame batch
        """
        # Get the field value series
        field_values = batch[self.field_name].copy()

        # Create result dataframe
        result = batch.copy()

        # Extract organization types if field is provided
        org_types = None
        if self.type_field and self.type_field in batch.columns:
            org_types = batch[self.type_field]

        # Extract regions if field is provided
        regions = None
        if self.region_field and self.region_field in batch.columns:
            regions = batch[self.region_field]

        # Prepare a list to store generated values
        generated_values = []

        # Process each value according to null strategy
        for idx, value in enumerate(field_values):
            # Get organization type for this record if available
            org_type = None
            if org_types is not None:
                org_type = org_types.iloc[idx]
                if pd.isna(org_type):
                    org_type = None

            # Get region for this record if available
            region = None
            if regions is not None:
                region = regions.iloc[idx]
                if pd.isna(region):
                    region = None

            # Get record identifier if available (for mapping store)
            record_id = None
            if self.id_field and self.id_field in batch.columns:
                record_id = batch[self.id_field].iloc[idx]

            # Determine parameters for generation
            gen_params = {
                "organization_type": org_type or self.organization_type,
                "region": region or self.region,
                "context_salt": self.context_salt or "org-generation",
                "record_id": record_id,
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
            except Exception as e:
                self.logger.error(
                    f"Error generating organization name for value '{value}': {str(e)}"
                )
                generated_values.append(value if pd.notna(value) else np.nan)

        # Update the dataframe with generated values
        result[self.output_field_name] = generated_values

        return result

    def process_value(self, value, **params):
        """
        Process a single value using the appropriate generation method with retry logic.

        Args:
            value: Original value
            **params: Additional parameters

        Returns:
            Processed value
        """
        # Add time for detailed metrics
        start_time = time.time() if self.detailed_metrics else None

        # If using mapping store, check for existing mapping
        if (
            self.consistency_mechanism == "mapping"
            and hasattr(self, "mapping_store")
            and self.mapping_store
        ):
            # Check if we already have a mapping for this value
            synthetic_value = self.mapping_store.get_mapping(self.field_name, value)

            if synthetic_value is not None:
                # If mapping found, use it
                if self.detailed_metrics and start_time:
                    # Collect performance metrics
                    self._generation_times.append(time.time() - start_time)

                    # Try to collect organization type statistics
                    if isinstance(synthetic_value, str):
                        org_type = self.generator.detect_organization_type(
                            synthetic_value
                        )
                        self._type_stats[org_type] += 1

                return synthetic_value

        # Implement retry mechanism
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Generate new value based on consistency mechanism
                if self.consistency_mechanism == "prgn":
                    # Ensure the generator has PRGN capabilities
                    if (
                        not hasattr(self.generator, "prgn_generator")
                        or self.generator.prgn_generator is None
                    ):
                        # Set up PRGN generator if not already done
                        if not hasattr(self, "_prgn_generator"):
                            from pamola_core.fake_data.commons.prgn import PRNGenerator

                            self._prgn_generator = PRNGenerator(
                                global_seed=self.key or "org-generation"
                            )
                        self.generator.prgn_generator = self._prgn_generator

                    # Generate using PRGN
                    synthetic_value = self.generator.generate_like(value, **params)
                else:
                    # Use standard generation
                    synthetic_value = self.generator.generate_like(value, **params)

                # If using mapping store, store the mapping
                if (
                    self.consistency_mechanism == "mapping"
                    and hasattr(self, "mapping_store")
                    and self.mapping_store
                ):
                    self.mapping_store.add_mapping(
                        self.field_name, value, synthetic_value
                    )

                # Collect metrics if needed
                if self.detailed_metrics and start_time:
                    self._generation_times.append(time.time() - start_time)

                    # Collect organization type statistics
                    if isinstance(synthetic_value, str):
                        org_type = self.generator.detect_organization_type(
                            synthetic_value
                        )
                        self._type_stats[org_type] += 1

                        # Collect region statistics
                        region = self.generator._determine_region_from_name(
                            synthetic_value, org_type
                        )
                        self._region_stats[region] += 1

                        # Collect prefix/suffix statistics
                        has_prefix = any(
                            p.strip() in synthetic_value.split()[:1]
                            for p_list in self.generator._prefixes.values()
                            for region_list in p_list.values()
                            for p in region_list
                        )

                        has_suffix = any(
                            s.strip() in synthetic_value.split()[-1:]
                            for s_list in self.generator._suffixes.values()
                            for region_list in s_list.values()
                            for s in region_list
                        )

                        if has_prefix and has_suffix:
                            self._prefix_suffix_stats["with_both"] += 1
                        elif has_prefix:
                            self._prefix_suffix_stats["with_prefix"] += 1
                        elif has_suffix:
                            self._prefix_suffix_stats["with_suffix"] += 1
                        else:
                            self._prefix_suffix_stats["with_neither"] += 1

                return synthetic_value

            except Exception as e:
                last_error = e
                retries += 1
                self.retry_count += 1

                # Log error with appropriate level
                if retries <= self.max_retries:
                    self.logger.debug(
                        f"Retry {retries}/{self.max_retries} generating organization name for value '{value}': {str(e)}"
                    )
                else:
                    self.logger.error(
                        f"Failed to generate organization name for value '{value}' after {self.max_retries} retries: {str(e)}"
                    )
                    self.error_count += 1

        # If all attempts failed, return original value or None
        if pd.notna(value):
            return value
        return None

    def _analyze_type_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of organization types in generated names.

        Args:
            df: Processed DataFrame

        Returns:
            Organization type distribution metrics
        """
        # Determine which field to analyze based on mode
        if self.mode == "REPLACE":
            field = self.field_name
        else:  # ENRICH
            field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

        if field not in df.columns:
            return {}

        # If we have detailed metrics and they're already collected, use them
        if self.detailed_metrics and hasattr(self, "_type_stats") and self._type_stats:
            top_types = self._type_stats.most_common(10)
            total = sum(self._type_stats.values())
            type_distribution = {
                org_type: count / total for org_type, count in top_types
            }

            # Calculate diversity metrics
            unique_types = len(self._type_stats)
            diversity_ratio = unique_types / total if total > 0 else 0

            return {
                "total_organizations": total,
                "unique_organization_types": unique_types,
                "diversity_ratio": diversity_ratio,
                "top_organization_types": type_distribution,
            }

        # Otherwise detect organization types from generated names
        org_types = []
        for org_name in df[field].dropna():
            if isinstance(org_name, str):
                org_type = self.generator.detect_organization_type(org_name)
                if org_type:
                    org_types.append(org_type)

        if not org_types:
            return {}

        # Calculate organization type frequency
        type_counts = Counter(org_types)
        total = len(org_types)

        # Get top organization types (max 10)
        top_types = type_counts.most_common(10)
        type_distribution = {org_type: count / total for org_type, count in top_types}

        # Calculate diversity metrics
        unique_types = len(type_counts)
        diversity_ratio = unique_types / total if total > 0 else 0

        return {
            "total_organizations": total,
            "unique_organization_types": unique_types,
            "diversity_ratio": diversity_ratio,
            "top_organization_types": type_distribution,
        }

    def _analyze_region_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of regions in generated organization names.

        Args:
            df: Processed DataFrame

        Returns:
            Region distribution metrics
        """
        # Determine which field to analyze based on mode
        if self.mode == "REPLACE":
            field = self.field_name
        else:  # ENRICH
            field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

        if field not in df.columns:
            return {}

        # If we have detailed metrics and they're already collected, use them
        if (
            self.detailed_metrics
            and hasattr(self, "_region_stats")
            and self._region_stats
        ):
            top_regions = self._region_stats.most_common(10)
            total = sum(self._region_stats.values())
            region_distribution = {
                region: count / total for region, count in top_regions
            }

            return {
                "total_organizations": total,
                "unique_regions": len(self._region_stats),
                "region_distribution": region_distribution,
            }

        # Otherwise detect regions from generated names
        regions = []
        for org_name in df[field].dropna():
            if isinstance(org_name, str):
                org_type = self.generator.detect_organization_type(org_name)
                region = self.generator._determine_region_from_name(org_name, org_type)
                if region:
                    regions.append(region)

        if not regions:
            return {}

        # Calculate region frequency
        region_counts = Counter(regions)
        total = len(regions)

        # Get top regions (max 10)
        top_regions = region_counts.most_common(10)
        region_distribution = {region: count / total for region, count in top_regions}

        return {
            "total_organizations": total,
            "unique_regions": len(region_counts),
            "region_distribution": region_distribution,
        }

    def _analyze_prefix_suffix_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of prefixes and suffixes in generated names.

        Args:
            df: Processed DataFrame

        Returns:
            Prefix/suffix distribution metrics
        """
        # If we have detailed metrics and they're already collected, use them
        if (
            self.detailed_metrics
            and hasattr(self, "_prefix_suffix_stats")
            and self._prefix_suffix_stats
        ):
            total = sum(self._prefix_suffix_stats.values())
            if total == 0:
                return {}

            # Calculate percentages
            distribution = {
                key: value / total for key, value in self._prefix_suffix_stats.items()
            }

            return {"prefix_suffix_distribution": distribution, "total_analyzed": total}

        # Otherwise we don't have this data readily available
        return {}

    def _collect_specific_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics specific to the Organization Name Generation operation.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame

        Returns
        -------
        Dict[str, Any]
            Dictionary containing organization-specific metrics.
        """
        metrics_data = {}

        gen = self.generator

        # --- 1. Organization generator info ---
        metrics_data["organization_generator"] = {
            "organization_type": getattr(self, "organization_type", None),
            "region": getattr(self, "region", None),
            "preserve_type": getattr(self, "preserve_type", False),
            "add_prefix_probability": getattr(gen, "add_prefix_probability", 0.3),
            "add_suffix_probability": getattr(gen, "add_suffix_probability", 0.5),
        }

        # --- 2. Dictionary info (if available) ---
        if hasattr(gen, "get_dictionary_info"):
            try:
                dictionary_info = gen.get_dictionary_info()
                if dictionary_info:
                    metrics_data["organization_generator"][
                        "dictionary_info"
                    ] = dictionary_info
            except Exception as e:
                self.logger.warning(f"Error collecting dictionary info: {str(e)}")

        # --- 3. Distributions (type, region, prefix/suffix) ---
        distributions = self._collect_organization_distributions(df)
        if distributions:
            metrics_data["organization_generator"].update(distributions)

        # --- 4. Quality metrics (for ENRICH mode only) ---
        if self.mode == "ENRICH" and getattr(self, "_original_df", None) is not None:
            try:
                output_field = (
                    self.output_field_name or f"{self.column_prefix}{self.field_name}"
                )
                if output_field in self._original_df.columns:
                    quality_metrics = self._calculate_quality_metrics(
                        self._original_df[self.field_name],
                        self._original_df[output_field],
                    )
                    metrics_data["quality_metrics"] = quality_metrics
            except Exception as e:
                self.logger.warning(f"Error calculating quality metrics: {str(e)}")

        return metrics_data

    def _collect_organization_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collects organization-related distributions: type, region, and prefix/suffix.
        """
        distributions = {}

        # Organization type distribution
        if getattr(self, "collect_type_distribution", False):
            try:
                type_metrics = self._analyze_type_distribution(df)
                if type_metrics:
                    distributions["type_distribution"] = type_metrics
            except Exception as e:
                self.logger.warning(
                    f"Error collecting organization type distribution: {str(e)}"
                )

        # Region distribution
        try:
            region_metrics = self._analyze_region_distribution(df)
            if region_metrics:
                distributions["region_distribution"] = region_metrics
        except Exception as e:
            self.logger.warning(f"Error collecting region distribution: {str(e)}")

        # Prefix/suffix distribution
        try:
            prefix_suffix_metrics = self._analyze_prefix_suffix_distribution(df)
            if prefix_suffix_metrics:
                distributions["prefix_suffix_distribution"] = prefix_suffix_metrics
        except Exception as e:
            self.logger.warning(
                f"Error collecting prefix/suffix distribution: {str(e)}"
            )

        return distributions

    def _calculate_quality_metrics(
        self, original_series: pd.Series, generated_series: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics comparing original and generated organization names.

        Args:
            original_series: Series with original organization names
            generated_series: Series with generated organization names

        Returns:
            Dictionary with quality metrics
        """
        metrics = {}

        # Calculate length preservation
        original_lengths = original_series.dropna().astype(str).str.len()
        generated_lengths = generated_series.dropna().astype(str).str.len()

        # Avoid division by zero
        if len(original_lengths) > 0 and len(generated_lengths) > 0:
            orig_mean_len = original_lengths.mean()
            gen_mean_len = generated_lengths.mean()

            # Length similarity (1.0 = perfect match)
            length_similarity = 1.0 - min(
                1.0, abs(orig_mean_len - gen_mean_len) / orig_mean_len
            )
            metrics["length_similarity"] = length_similarity

        # Analyze type preservation
        orig_types = []
        gen_types = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if isinstance(orig, str) and isinstance(gen, str):
                orig_type = self.generator.detect_organization_type(orig)
                gen_type = self.generator.detect_organization_type(gen)

                orig_types.append(orig_type)
                gen_types.append(gen_type)

        if orig_types and gen_types:
            # Calculate percentage of preserved organization types
            type_preservation_count = sum(
                1 for o, g in zip(orig_types, gen_types) if o == g
            )
            type_preservation_ratio = type_preservation_count / len(orig_types)
            metrics["type_preservation_ratio"] = type_preservation_ratio

            # Calculate organization type diversity
            orig_type_count = len(set(orig_types))
            gen_type_count = len(set(gen_types))
            type_diversity_ratio = (
                gen_type_count / orig_type_count if orig_type_count > 0 else 0
            )
            metrics["type_diversity_ratio"] = type_diversity_ratio

        # Word count similarity
        orig_word_counts = original_series.dropna().astype(str).str.split().str.len()
        gen_word_counts = generated_series.dropna().astype(str).str.split().str.len()

        if len(orig_word_counts) > 0 and len(gen_word_counts) > 0:
            orig_mean_words = orig_word_counts.mean()
            gen_mean_words = gen_word_counts.mean()

            # Word count similarity (1.0 = perfect match)
            word_count_similarity = 1.0 - min(
                1.0, abs(orig_mean_words - gen_mean_words) / orig_mean_words
            )
            metrics["word_count_similarity"] = word_count_similarity

        return metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        return {
            "organization_type": self.organization_type,
            "dictionaries": self.dictionaries,
            "prefixes": self.prefixes,
            "suffixes": self.suffixes,
            "add_prefix_probability": self.add_prefix_probability,
            "add_suffix_probability": self.add_suffix_probability,
            "region": self.region,
            "preserve_type": self.preserve_type,
            "industry": self.industry,
            "consistency_mechanism": self.consistency_mechanism,
            "mapping_store_path": self.mapping_store_path,
            "id_field": self.id_field,
            "key": self.key,
            "context_salt": self.context_salt,
            "save_mapping": self.save_mapping,
            "column_prefix": self.column_prefix,
            "collect_type_distribution": self.collect_type_distribution,
            "type_field": self.type_field,
            "region_field": self.region_field,
            "detailed_metrics": self.detailed_metrics,
            "max_retries": self.max_retries,
        }
