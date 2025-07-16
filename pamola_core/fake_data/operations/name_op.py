"""
Name generation operation for fake data system.

This module provides the NameOperation class for generating synthetic names
while maintaining statistical properties of the original data and supporting
consistent mapping.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from pamola_core.fake_data.commons import metrics
from pamola_core.fake_data.commons.operations import GeneratorOperation
from pamola_core.fake_data.commons.prgn import PRNGenerator
from pamola_core.fake_data.generators.name import NameGenerator
from pamola_core.utils import io
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.progress import HierarchicalProgressTracker



@register()
class FakeNameOperation(GeneratorOperation):
    """
    Operation for generating synthetic names.

    This operation processes personal names in datasets, replacing them with
    synthetic alternatives while preserving gender, language, and format characteristics.
    """

    name = "name_generator"
    description = "Generates synthetic names while preserving characteristics"
    category = "fake_data"

    def __init__(self, field_name: str,
                 mode: str = "ENRICH",
                 output_field_name: Optional[str] = None,
                 language: str = "en",
                 gender_field: Optional[str] = None,
                 gender_from_name: bool = False,
                 format: Optional[str] = None,
                 f_m_ratio: float = 0.5,
                 use_faker: bool = False,
                 case: str = "title",
                 dictionaries: Optional[Dict[str, str]] = None,
                 chunk_size: int = 10000,
                 null_strategy: str = "PRESERVE",
                 consistency_mechanism: str = "prgn",
                 mapping_store_path: Optional[str] = None,
                 id_field: Optional[str] = None,
                 key: Optional[str] = None,
                 context_salt: Optional[str] = None,
                 save_mapping: bool = False,
                 column_prefix: str = "_",
                 use_cache: bool = True,
                 force_recalculation: bool = False,
                 use_dask: bool = False,
                 npartitions: int = 1,
                 use_vectorization: bool = False,
                 parallel_processes: int = 1,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 visualization_backend: Optional[str] = None,
                 visualization_theme: Optional[str] = None,
                 visualization_strict: bool = False,
                 encryption_mode: Optional[str] = None):
        """
        Initialize name generation operation.

        Args:
            field_name: Field to process
            mode: Operation mode (REPLACE or ENRICH)
            output_field_name: Name for the output field (if mode=ENRICH)
            language: Language for name generation (en, ru, vn)
            gender_field: Field containing gender information
            gender_from_name: Whether to infer gender from name
            format: Format for name generation (FML, FL, LF, etc.)
            f_m_ratio: Ratio of female/male names for random generation
            use_faker: Whether to use Faker library if available
            case: Case formatting (upper, lower, title)
            dictionaries: Paths to custom dictionaries
            chunk_size: Number of records to process in one batch
            null_strategy: Strategy for handling NULL values
            consistency_mechanism: Method for ensuring consistency (mapping or prgn)
            mapping_store_path: Path to store mappings
            id_field: Field for record identification
            key: Key for encryption/PRGN
            context_salt: Salt for PRGN
            save_mapping: Whether to save mapping to file
            column_prefix: Prefix for new column (if mode=ENRICH)
            use_dask: Whether to use Dask for large datasets
            npartitions: Number of partitions for Dask processing (if use_dask=True)
        """
        # Store attributes locally that we need to access directly
        self.gender_field = gender_field
        self.language = language
        self.column_prefix = column_prefix
        self.id_field = id_field
        self.key = key
        self.context_salt = context_salt
        self.save_mapping = save_mapping
        self.mapping_store_path = mapping_store_path
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.force_recalculation = force_recalculation
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes

        # Set up name generator configuration
        generator_params = {
            "language": language,
            "gender_from_name": gender_from_name,
            "format": format,
            "f_m_ratio": f_m_ratio,
            "use_faker": use_faker,
            "case": case,
            "dictionaries": dictionaries if dictionaries is not None else {},
            "key": key,
            "context_salt": context_salt
        }

        # Create a temporary BaseGenerator to pass to parent constructor
        # We'll replace it with the real NameGenerator after initialization
        base_generator = NameGenerator(config=generator_params)

        # Initialize parent class first with the base generator
        super().__init__(
            field_name=field_name,
            generator=base_generator,
            mode=mode,
            output_field_name=output_field_name,
            chunk_size=chunk_size,
            null_strategy=null_strategy,
            consistency_mechanism=consistency_mechanism,
            use_cache=use_cache,
            force_recalculation=force_recalculation,
            use_dask=use_dask,
            npartitions=npartitions,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
            visualization_backend=visualization_backend,
            visualization_theme=visualization_theme,
            visualization_strict=visualization_strict
        )

        # Set up performance metrics
        self.start_time = None
        self.process_count = 0

        # Ensure we have a reference to the original DataFrame for metrics collection
        self._original_df = None

        # Initialize mapping store if path is provided
        if mapping_store_path:
            self._initialize_mapping_store(mapping_store_path)

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
                self.logger.info(f"Loaded mapping store from {path_obj.name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize mapping store: {str(e)}")
            self.mapping_store = None

    def execute(self, data_source, task_dir, reporter, progress_tracker: Optional[HierarchicalProgressTracker] = None, **kwargs):
        """
        Execute the name generation operation.

        Args:
            data_source: Source of data (DataFrame or path to file)
            task_dir: Directory for storing operation artifacts
            reporter: Reporter for progress updates
            **kwargs: Additional parameters

        Returns:
            Operation result with processed data and metrics
        """
        # Config logger task for operatiions
        self.logger = kwargs.get('logger', self.logger)

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
        result = super().execute(data_source, task_dir, reporter, progress_tracker, **kwargs)

        # If we didn't set _original_df earlier and the parent loaded it, get it now
        if self._original_df is None and hasattr(self, "_df"):
            self._original_df = self._df.copy()

        # Save mapping if requested
        if self.save_mapping and hasattr(self, "mapping_store") and self.mapping_store:
            mapping_dir = Path(task_dir) / "maps"
            io.ensure_directory(mapping_dir)
            mapping_path = mapping_dir / f"{self.name}_{self.field_name}_mapping.json"
            self.mapping_store.save_json(mapping_path)
            self.logger.info(f"Saved mapping to {Path(mapping_path).name}")

        return result

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to generate synthetic names.

        Args:
            batch: DataFrame batch to process

        Returns:
            Processed DataFrame batch
        """
        # Get the field value series
        field_values = batch[self.field_name].copy()

        # Create result dataframe
        result = batch.copy()

        # Get gender information if available
        gender_values = None
        if self.gender_field and self.gender_field in batch.columns:
            gender_values = batch[self.gender_field]

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

            # Get gender for this record
            gender = None
            if gender_values is not None:
                gender_val = gender_values.iloc[idx]
                if not pd.isna(gender_val):
                    # Convert various gender formats to "M" or "F"
                    gender_str = str(gender_val).strip().upper()
                    if gender_str in ["M", "MALE", "МУЖ", "МУЖСКОЙ", "1"]:
                        gender = "M"
                    elif gender_str in ["F", "FEMALE", "ЖЕН", "ЖЕНСКИЙ", "2"]:
                        gender = "F"

            # Get record identifier if available (for mapping store)
            record_id = None
            if self.id_field and self.id_field in batch.columns:
                record_id = batch[self.id_field].iloc[idx]

            # Determine parameters for generation
            gen_params = {
                "gender": gender,
                "language": self.language,
                "field_name": self.field_name,
                "context_salt": self.context_salt or "name-generation",
                "record_id": record_id
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
                self.process_count += 1
            except Exception as e:
                self.logger.error(f"Error generating name for value '{value}': {str(e)}")
                generated_values.append(value if pd.notna(value) else np.nan)

        # Update the dataframe with generated values
        result[output_field] = generated_values

        return result

    def process_value(self, value, **params):
        """
        Process a single value using the appropriate generation method.

        Args:
            value: Original value
            **params: Additional parameters

        Returns:
            Processed value
        """
        # If using mapping store, check for existing mapping
        if self.consistency_mechanism == "mapping" and hasattr(self, "mapping_store") and self.mapping_store:
            # Check if we already have a mapping for this value
            synthetic_value = self.mapping_store.get_mapping(self.field_name, value)

            if synthetic_value is not None:
                return synthetic_value

        # Generate new value based on consistency mechanism
        if self.consistency_mechanism == "prgn":
            # Ensure the generator has PRGN capabilities
            if not hasattr(self.generator, "prgn_generator") or self.generator.prgn_generator is None:
                # Set up PRGN generator if not already done
                if not hasattr(self, "_prgn_generator"):
                    self._prgn_generator = PRNGenerator(global_seed=self.key or "name-generation")
                self.generator.prgn_generator = self._prgn_generator

            # Generate using PRGN
            synthetic_value = self.generator.generate_like(value, **params)
        else:
            # Use standard generation
            synthetic_value = self.generator.generate_like(value, **params)

        # If using mapping store, store the mapping
        if self.consistency_mechanism == "mapping" and hasattr(self, "mapping_store") and self.mapping_store:
            self.mapping_store.add_mapping(self.field_name, value, synthetic_value)

        return synthetic_value

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics for the name generation operation.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary with metrics
        """
        # Get basic metrics from parent
        metrics_data = super()._collect_metrics(df)

        # Add name-specific metrics
        metrics_data["name_generator"] = {
            "language": self.language,
            "gender_field": self.gender_field,
            "gender_from_name": getattr(self.generator, "gender_from_name", False),
            "format": getattr(self.generator, "format", "FL"),
            "use_faker": getattr(self.generator, "use_faker", False)
        }

        # Add performance metrics
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            metrics_data["performance"] = {
                "generation_time": execution_time,
                "records_per_second": int(self.process_count / execution_time) if execution_time > 0 else 0
            }

        # Add dictionary metrics
        dictionary_metrics = {
            "total_dictionary_entries": 0,
            "language_variants": []
        }

        # Try to get dictionary sizes
        for lang in ["en", "ru", "vi"]:
            # Count first names (male and female)
            male_first_count = len(getattr(self.generator, "_first_names_male", {}).get(lang, []))
            female_first_count = len(getattr(self.generator, "_first_names_female", {}).get(lang, []))

            # Count last names
            last_count = len(getattr(self.generator, "_last_names", {}).get(lang, []))

            # Count middle names
            male_middle_count = len(getattr(self.generator, "_middle_names_male", {}).get(lang, []))
            female_middle_count = len(getattr(self.generator, "_middle_names_female", {}).get(lang, []))

            total_lang_count = male_first_count + female_first_count + last_count + male_middle_count + female_middle_count

            if total_lang_count > 0:
                dictionary_metrics["language_variants"].append(lang)
                dictionary_metrics["total_dictionary_entries"] += total_lang_count

        metrics_data["dictionary_metrics"] = dictionary_metrics

        # Try to get input and output field lengths
        try:
            if self.mode == "REPLACE":
                # Just measure the field now
                length_stats = self._calculate_length_stats(df[self.field_name])
                metrics_data["name_generator"]["length_stats"] = length_stats
            else:  # ENRICH
                output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"
                if output_field in df.columns:
                    # Compare original and generated lengths
                    orig_stats = self._calculate_length_stats(df[self.field_name])
                    new_stats = self._calculate_length_stats(df[output_field])

                    metrics_data["name_generator"]["original_length_stats"] = orig_stats
                    metrics_data["name_generator"]["generated_length_stats"] = new_stats
        except Exception as e:
            # Don't fail the operation if metrics collection fails
            self.logger.warning(f"Error collecting length metrics: {str(e)}")
            metrics_data["name_generator"]["length_stats_error"] = str(e)

        return metrics_data

    def _calculate_length_stats(self, series: pd.Series) -> Dict[str, Any]:
        """
        Calculate length statistics for a series of strings.

        Args:
            series: Series of strings

        Returns:
            Dictionary with length statistics
        """
        # Filter out null values
        non_null = series.dropna()

        # Get string lengths
        lengths = non_null.astype(str).str.len()

        if len(lengths) == 0:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0
            }

        return {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median())
        }

    def _save_metrics(self, metrics_data: Dict[str, Any], task_dir: Path, **kwargs) -> Path:
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

                    kwargs_encryption = {
                        "use_encryption": kwargs.get('use_encryption', False),
                        "encryption_key": kwargs.get('encryption_key', None)
                    }
                    # Create visualizations
                    visualizations = collector.visualize_metrics(
                        metrics_data,
                        self.field_name,
                        vis_dir,
                        self.name,
                        **kwargs_encryption
                    )

                    # Add visualization paths to metrics
                    metrics_data["visualizations"] = {
                        name: str(path) for name, path in visualizations.items()
                    }
            except Exception as e:
                self.logger.warning(f"Error generating visualizations: {str(e)}")


        # Save metrics to file
        use_encryption = kwargs.get('use_encryption', False)
        encryption_key= kwargs.get('encryption_key', None) if use_encryption else None
        io.write_json(metrics_data, metrics_path, encryption_key=encryption_key)

        return metrics_path