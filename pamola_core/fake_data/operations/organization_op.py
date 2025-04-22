"""
Organization generation operation for fake data system.

This module provides the OrganizationOperation class for generating synthetic
organization names while maintaining statistical properties of the original data
and supporting consistent mapping.
"""

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
import pandas as pd

from pamola_core.fake_data.commons import metrics
from pamola_core.fake_data.commons.operations import GeneratorOperation
from pamola_core.fake_data.generators.base_generator import BaseGenerator
from pamola_core.fake_data.generators.organization import OrganizationGenerator
from pamola_core.utils import io
from pamola_core.utils.ops.op_registry import register

# Configure logger
logger = logging.getLogger(__name__)


@register()
class OrganizationOperation(GeneratorOperation):
    """
    Operation for generating synthetic organization names.

    This operation processes organization names in datasets, replacing them with
    synthetic alternatives while preserving type characteristics and regional
    specificity.
    """

    name = "organization_generator"
    description = "Generates synthetic organization names with configurable types and formats"
    category = "fake_data"

    def __init__(self, field_name: str,
                 mode: str = "ENRICH",
                 output_field_name: Optional[str] = None,
                 organization_type: str = "general",
                 dictionaries: Optional[Dict[str, str]] = None,
                 prefixes: Optional[Dict[str, str]] = None,
                 suffixes: Optional[Dict[str, str]] = None,
                 add_prefix_probability: float = 0.3,
                 add_suffix_probability: float = 0.5,
                 region: str = "en",
                 preserve_type: bool = True,
                 industry: Optional[str] = None,
                 batch_size: int = 10000,
                 null_strategy: str = "PRESERVE",
                 consistency_mechanism: str = "prgn",
                 mapping_store_path: Optional[str] = None,
                 id_field: Optional[str] = None,
                 key: Optional[str] = None,
                 context_salt: Optional[str] = None,
                 save_mapping: bool = False,
                 column_prefix: str = "_",
                 # New parameters
                 collect_type_distribution: bool = True,
                 type_field: Optional[str] = None,
                 region_field: Optional[str] = None,
                 detailed_metrics: bool = False,
                 error_logging_level: str = "WARNING",
                 max_retries: int = 3):
        """
        Initialize organization name generation operation.

        Args:
            field_name: Field to process (containing organization names)
            mode: Operation mode (REPLACE or ENRICH)
            output_field_name: Name for the output field (if mode=ENRICH)
            organization_type: Type of organization to generate
            dictionaries: Paths to dictionaries with organization names
            prefixes: Paths to dictionaries with organization name prefixes
            suffixes: Paths to dictionaries with organization name suffixes
            add_prefix_probability: Probability of adding prefix to name
            add_suffix_probability: Probability of adding suffix to name
            region: Default region for generation
            preserve_type: Whether to preserve organization type from original
            industry: Specific industry for 'industry' type organizations
            batch_size: Number of records to process in one batch
            null_strategy: Strategy for handling NULL values
            consistency_mechanism: Method for ensuring consistency (mapping or prgn)
            mapping_store_path: Path to store mappings
            id_field: Field for record identification
            key: Key for encryption/PRGN
            context_salt: Salt for PRGN
            save_mapping: Whether to save mapping to file
            column_prefix: Prefix for new column (if mode=ENRICH)
            collect_type_distribution: Whether to collect organization type distribution
            type_field: Field containing organization types
            region_field: Field containing region codes
            detailed_metrics: Whether to collect detailed metrics
            error_logging_level: Level for error logging (ERROR, WARNING, INFO)
            max_retries: Maximum number of retries for generation on error
        """
        # Save new parameters
        self.detailed_metrics = detailed_metrics
        self.error_logging_level = error_logging_level.upper()
        self.max_retries = max_retries
        self.collect_type_distribution = collect_type_distribution
        self.type_field = type_field
        self.region_field = region_field

        # Configure logging level
        self._configure_logging()

        # Store attributes locally that we need to access directly
        self.column_prefix = column_prefix
        self.id_field = id_field
        self.key = key
        self.context_salt = context_salt
        self.save_mapping = save_mapping
        self.mapping_store_path = mapping_store_path
        self.organization_type = organization_type
        self.region = region
        self.preserve_type = preserve_type

        # Set up organization generator configuration
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
            "context_salt": context_salt
        }

        # Create a temporary BaseGenerator to pass to parent constructor
        # We'll replace it with the real OrganizationGenerator after initialization
        base_generator = BaseGenerator()

        # Initialize parent class first with the base generator
        super().__init__(
            field_name=field_name,
            generator=base_generator,
            mode=mode,
            output_field_name=output_field_name,
            batch_size=batch_size,
            null_strategy=null_strategy,
            consistency_mechanism=consistency_mechanism
        )

        # Now create and store the real organization generator
        self.generator = OrganizationGenerator(config=generator_params)

        # Set up performance metrics
        self.start_time = None
        self.process_count = 0
        self.error_count = 0
        self.retry_count = 0

        # Ensure we have a reference to the original DataFrame for metrics collection
        self._original_df = None

        # Initialize mapping store if path is provided
        if mapping_store_path:
            self._initialize_mapping_store(mapping_store_path)

        # For detailed metrics
        if self.detailed_metrics:
            self._type_stats = Counter()
            self._region_stats = Counter()
            self._prefix_suffix_stats = {
                "with_prefix": 0,
                "with_suffix": 0,
                "with_both": 0,
                "with_neither": 0
            }
            self._generation_times = []

    def _configure_logging(self):
        """
        Configure logging based on error_logging_level.
        """
        log_level = getattr(logging, self.error_logging_level, logging.WARNING)
        logger.setLevel(log_level)

    def _initialize_mapping_store(self, path: Union[str, Path]) -> None:
        """
        Initialize the mapping store if needed.

        Args:
            path: Path to mapping store file
        """
        try:
            from pamola_core.fake_data.commons.mapping_store import MappingStore

            self.mapping_store = MappingStore()

            # Load existing mappings if the file exists
            path_obj = Path(path)
            if path_obj.exists():
                self.mapping_store.load(path_obj)
                logger.info(f"Loaded mapping store from {path}")
        except Exception as e:
            logger.warning(f"Failed to initialize mapping store: {str(e)}")
            self.mapping_store = None

    def execute(self, data_source, task_dir, reporter, **kwargs):
        """
        Execute the organization name generation operation.

        Args:
            data_source: Source of data (DataFrame or path to file)
            task_dir: Directory for storing operation artifacts
            reporter: Reporter for progress updates
            **kwargs: Additional parameters

        Returns:
            Operation result with processed data and metrics
        """
        # Start timing for performance metrics
        self.start_time = time.time()
        self.process_count = 0

        # Load data if needed
        if isinstance(data_source, pd.DataFrame):
            self._original_df = data_source.copy()
        else:
            # If data_source is a path, it will be loaded by the parent class
            pass

        # Call parent execute method
        result = super().execute(data_source, task_dir, reporter, **kwargs)

        # If we didn't set _original_df earlier and the parent loaded it, get it now
        if self._original_df is None and hasattr(self, "_df"):
            self._original_df = self._df.copy()

        # Save mapping if requested
        if self.save_mapping and hasattr(self, "mapping_store") and self.mapping_store:
            mapping_dir = Path(task_dir) / "maps"
            io.ensure_directory(mapping_dir)
            mapping_path = mapping_dir / f"{self.name}_{self.field_name}_mapping.json"
            self.mapping_store.save_json(mapping_path)
            logger.info(f"Saved mapping to {mapping_path}")

        return result

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to generate synthetic organization names.

        Args:
            batch: DataFrame batch to process

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

        # Determine output field based on mode
        if self.mode == "REPLACE":
            output_field = self.field_name
        else:  # ENRICH
            output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

        # Prepare a list to store generated values
        generated_values = []

        # Process each value according to null strategy
        for idx, value in enumerate(field_values):
            # Handle null values according to strategy
            if pd.isna(value):
                if self.null_strategy == "PRESERVE":
                    generated_values.append(np.nan)
                    continue
                elif self.null_strategy == "EXCLUDE":
                    # Skip this value but keep in result with NaN
                    generated_values.append(np.nan)
                    continue
                elif self.null_strategy == "ERROR":
                    raise ValueError(f"Null value found in {self.field_name} at position {idx}")
                # If strategy is REPLACE, we'll generate a value for the null

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
                "record_id": record_id
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
                self.process_count += 1
            except Exception as e:
                logger.error(f"Error generating organization name for value '{value}': {str(e)}")
                generated_values.append(value if pd.notna(value) else np.nan)

        # Update the dataframe with generated values
        result[output_field] = generated_values

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
        if self.consistency_mechanism == "mapping" and hasattr(self, "mapping_store") and self.mapping_store:
            # Check if we already have a mapping for this value
            synthetic_value = self.mapping_store.get_mapping(self.field_name, value)

            if synthetic_value is not None:
                # If mapping found, use it
                if self.detailed_metrics and start_time:
                    # Collect performance metrics
                    self._generation_times.append(time.time() - start_time)

                    # Try to collect organization type statistics
                    if isinstance(synthetic_value, str):
                        org_type = self.generator.detect_organization_type(synthetic_value)
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
                    if not hasattr(self.generator, "prgn_generator") or self.generator.prgn_generator is None:
                        # Set up PRGN generator if not already done
                        if not hasattr(self, "_prgn_generator"):
                            from pamola_core.fake_data.commons.prgn import PRNGenerator
                            self._prgn_generator = PRNGenerator(global_seed=self.key or "org-generation")
                        self.generator.prgn_generator = self._prgn_generator

                    # Generate using PRGN
                    synthetic_value = self.generator.generate_like(value, **params)
                else:
                    # Use standard generation
                    synthetic_value = self.generator.generate_like(value, **params)

                # If using mapping store, store the mapping
                if self.consistency_mechanism == "mapping" and hasattr(self, "mapping_store") and self.mapping_store:
                    self.mapping_store.add_mapping(self.field_name, value, synthetic_value)

                # Collect metrics if needed
                if self.detailed_metrics and start_time:
                    self._generation_times.append(time.time() - start_time)

                    # Collect organization type statistics
                    if isinstance(synthetic_value, str):
                        org_type = self.generator.detect_organization_type(synthetic_value)
                        self._type_stats[org_type] += 1

                        # Collect region statistics
                        region = self.generator._determine_region_from_name(synthetic_value, org_type)
                        self._region_stats[region] += 1

                        # Collect prefix/suffix statistics
                        has_prefix = any(p.strip() in synthetic_value.split()[:1]
                                         for p_list in self.generator._prefixes.values()
                                         for region_list in p_list.values()
                                         for p in region_list)

                        has_suffix = any(s.strip() in synthetic_value.split()[-1:]
                                         for s_list in self.generator._suffixes.values()
                                         for region_list in s_list.values()
                                         for s in region_list)

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
                    logger.debug(
                        f"Retry {retries}/{self.max_retries} generating organization name for value '{value}': {str(e)}")
                else:
                    logger.error(
                        f"Failed to generate organization name for value '{value}' after {self.max_retries} retries: {str(e)}")
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
            type_distribution = {org_type: count / total for org_type, count in top_types}

            # Calculate diversity metrics
            unique_types = len(self._type_stats)
            diversity_ratio = unique_types / total if total > 0 else 0

            return {
                "total_organizations": total,
                "unique_organization_types": unique_types,
                "diversity_ratio": diversity_ratio,
                "top_organization_types": type_distribution
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
            "top_organization_types": type_distribution
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
        if self.detailed_metrics and hasattr(self, "_region_stats") and self._region_stats:
            top_regions = self._region_stats.most_common(10)
            total = sum(self._region_stats.values())
            region_distribution = {region: count / total for region, count in top_regions}

            return {
                "total_organizations": total,
                "unique_regions": len(self._region_stats),
                "region_distribution": region_distribution
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
            "region_distribution": region_distribution
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
        if self.detailed_metrics and hasattr(self, "_prefix_suffix_stats") and self._prefix_suffix_stats:
            total = sum(self._prefix_suffix_stats.values())
            if total == 0:
                return {}

            # Calculate percentages
            distribution = {key: value / total for key, value in self._prefix_suffix_stats.items()}

            return {
                "prefix_suffix_distribution": distribution,
                "total_analyzed": total
            }

        # Otherwise we don't have this data readily available
        return {}

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics for the organization name generation operation.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary with metrics
        """
        # Get basic metrics from parent
        metrics_data = super()._collect_metrics(df)

        # Add organization-specific metrics
        metrics_data["organization_generator"] = {
            "organization_type": self.organization_type,
            "region": self.region,
            "preserve_type": self.preserve_type,
            "add_prefix_probability": getattr(self.generator, "add_prefix_probability", 0.3),
            "add_suffix_probability": getattr(self.generator, "add_suffix_probability", 0.5)
        }

        # Add dictionary info if available
        if hasattr(self.generator, "get_dictionary_info"):
            dictionary_info = self.generator.get_dictionary_info()
            if dictionary_info:
                metrics_data["organization_generator"]["dictionary_info"] = dictionary_info

        # Add performance metrics
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            metrics_data["performance"] = {
                "generation_time": execution_time,
                "records_per_second": int(self.process_count / execution_time) if execution_time > 0 else 0,
                "error_count": self.error_count,
                "retry_count": self.retry_count
            }

            # Add extended performance metrics if collected
            if self.detailed_metrics and hasattr(self, "_generation_times") and self._generation_times:
                metrics_data["performance"]["avg_record_generation_time"] = sum(self._generation_times) / len(
                    self._generation_times)
                metrics_data["performance"]["min_record_generation_time"] = min(self._generation_times)
                metrics_data["performance"]["max_record_generation_time"] = max(self._generation_times)

        # Add organization type distribution
        if self.collect_type_distribution:
            try:
                # Get organization type distribution
                type_metrics = self._analyze_type_distribution(df)
                if type_metrics:
                    metrics_data["organization_generator"]["type_distribution"] = type_metrics
            except Exception as e:
                logger.warning(f"Error collecting organization type distribution: {str(e)}")

        # Add region distribution
        try:
            # Get region distribution
            region_metrics = self._analyze_region_distribution(df)
            if region_metrics:
                metrics_data["organization_generator"]["region_distribution"] = region_metrics
        except Exception as e:
            logger.warning(f"Error collecting region distribution: {str(e)}")

        # Add prefix/suffix distribution
        try:
            # Get prefix/suffix distribution
            prefix_suffix_metrics = self._analyze_prefix_suffix_distribution(df)
            if prefix_suffix_metrics:
                metrics_data["organization_generator"]["prefix_suffix_distribution"] = prefix_suffix_metrics
        except Exception as e:
            logger.warning(f"Error collecting prefix/suffix distribution: {str(e)}")

        # Add quality metrics if we can collect them
        if self.mode == "ENRICH" and hasattr(self, "_original_df") and self._original_df is not None:
            try:
                # Output field name
                output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

                if output_field in self._original_df.columns:
                    quality_metrics = self._calculate_quality_metrics(
                        self._original_df[self.field_name],
                        self._original_df[output_field]
                    )
                    metrics_data["quality_metrics"] = quality_metrics
            except Exception as e:
                logger.warning(f"Error calculating quality metrics: {str(e)}")

        return metrics_data

    def _calculate_quality_metrics(self, original_series: pd.Series, generated_series: pd.Series) -> Dict[str, Any]:
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
            length_similarity = 1.0 - min(1.0, abs(orig_mean_len - gen_mean_len) / orig_mean_len)
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
            type_preservation_count = sum(1 for o, g in zip(orig_types, gen_types) if o == g)
            type_preservation_ratio = type_preservation_count / len(orig_types)
            metrics["type_preservation_ratio"] = type_preservation_ratio

            # Calculate organization type diversity
            orig_type_count = len(set(orig_types))
            gen_type_count = len(set(gen_types))
            type_diversity_ratio = gen_type_count / orig_type_count if orig_type_count > 0 else 0
            metrics["type_diversity_ratio"] = type_diversity_ratio

        # Word count similarity
        orig_word_counts = original_series.dropna().astype(str).str.split().str.len()
        gen_word_counts = generated_series.dropna().astype(str).str.split().str.len()

        if len(orig_word_counts) > 0 and len(gen_word_counts) > 0:
            orig_mean_words = orig_word_counts.mean()
            gen_mean_words = gen_word_counts.mean()

            # Word count similarity (1.0 = perfect match)
            word_count_similarity = 1.0 - min(1.0, abs(orig_mean_words - gen_mean_words) / orig_mean_words)
            metrics["word_count_similarity"] = word_count_similarity

        return metrics

    def _save_metrics(self, metrics_data: Dict[str, Any], task_dir: Path) -> Path:
        """
        Save metrics to a file and generate visualizations.

        Args:
            metrics_data: Metrics data
            task_dir: Directory for storing artifacts

        Returns:
            Path to the saved metrics file
        """
        # Create directories
        metrics_dir = task_dir / "metrics"
        vis_dir = task_dir / "visualizations"
        io.ensure_directory(metrics_dir)
        io.ensure_directory(vis_dir)

        # Create output path
        metrics_path = metrics_dir / f"{self.name}_{self.field_name}_metrics.json"

        # Create metrics collector
        collector = metrics.create_metrics_collector()

        # Generate visualizations if we have both original and generated data
        if self.mode == "ENRICH" and "original_data" in metrics_data and "generated_data" in metrics_data:
            try:
                # Get output field name
                output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

                # Get DataFrame for visualization
                if hasattr(self,
                           "_original_df") and self._original_df is not None and output_field in self._original_df.columns:
                    # Get original data series
                    orig_series = self._original_df[self.field_name]
                    gen_series = self._original_df[output_field]

                    # Create visualizations
                    visualizations = collector.visualize_metrics(
                        metrics_data,
                        self.field_name,
                        vis_dir,
                        self.name
                    )

                    # Add visualization paths to metrics
                    metrics_data["visualizations"] = {
                        name: str(path) for name, path in visualizations.items()
                    }
            except Exception as e:
                logger.warning(f"Error generating visualizations: {str(e)}")

        # Generate metrics report
        report_dir = task_dir / "reports"
        io.ensure_directory(report_dir)
        report = metrics.generate_metrics_report(
            metrics_data,
            report_dir / f"{self.name}_{self.field_name}_report.md",
            self.name,
            self.field_name
        )

        # Save metrics to file
        with open(metrics_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        return metrics_path