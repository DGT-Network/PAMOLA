"""
Name generation operation for fake data system.

This module provides the NameOperation class for generating synthetic names
while maintaining statistical properties of the original data and supporting
consistent mapping.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pamola_core.fake_data.schemas.name_op_schema import FakeNameOperationConfig
from pamola_core.fake_data.commons.prgn import PRNGenerator
from pamola_core.fake_data.generators.name import NameGenerator
from pamola_core.utils import io
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult
from pamola_core.utils.progress import HierarchicalProgressTracker


@register(version="1.0.0")
class FakeNameOperation(GeneratorOperation):
    """
    Operation for generating synthetic names.

    This operation processes personal names in datasets, replacing them with
    synthetic alternatives while preserving gender, language, and format characteristics.
    """

    def __init__(
        self,
        field_name: str,
        language: str = "en",
        gender_field: Optional[str] = None,
        gender_from_name: bool = False,
        format: Optional[str] = None,
        f_m_ratio: float = 0.5,
        use_faker: bool = False,
        case: str = "title",
        dictionaries: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        context_salt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize FakeNameOperation.

        Parameters
        ----------
        field_name : str
            Target column containing real names.
        language : str, optional
            Language for synthetic name generation (default "en").
        gender_field : str, optional
            Column name providing gender information (if available).
        gender_from_name : bool, optional
            Whether to infer gender from existing name values.
        format : str, optional
            Output format for synthetic names (e.g., "first_last", "last_first").
        f_m_ratio : float, optional
            Ratio of female-to-male names to preserve balance (default 0.5).
        use_faker : bool, optional
            Whether to use Faker library for name synthesis.
        case : str, optional
            Case formatting for output names ("title", "upper", "lower").
        dictionaries : dict, optional
            Custom dictionaries for localized name generation.
        key : str, optional
            Key for encryption or PRGN consistency (if applicable).
        context_salt : str, optional
                Additional context salt for PRGN to enhance uniqueness.
        **kwargs : dict
            Additional parameters forwarded to GeneratorOperation and BaseOperation.
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Synthetic name generation for '{field_name}' preserving linguistic and gender characteristics",
        )

        # Build config object (if used for schema/validation)
        config = FakeNameOperationConfig(
            field_name=field_name,
            language=language,
            gender_field=gender_field,
            gender_from_name=gender_from_name,
            format=format,
            f_m_ratio=f_m_ratio,
            use_faker=use_faker,
            case=case,
            dictionaries=dictionaries,
            key=key,
            context_salt=context_salt,
            **kwargs,
        )

        # Prepare generator-specific parameters
        generator_params = {
            "language": language,
            "gender_from_name": gender_from_name,
            "format": format,
            "f_m_ratio": f_m_ratio,
            "use_faker": use_faker,
            "case": case,
            "dictionaries": dictionaries or {},
            "key": key,
            "context_salt": context_salt,
        }

        # Instantiate NameGenerator (a subclass of BaseGenerator)
        name_generator = NameGenerator(config=generator_params)

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize parent (GeneratorOperation)
        super().__init__(
            field_name=field_name,
            generator=name_generator,
            generator_params=generator_params,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Operation metadata
        self.operation_name = self.__class__.__name__
        self._original_df = None

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
        Process a batch of data to generate synthetic names.

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

        # Get gender information if available
        gender_values = None
        if self.gender_field and self.gender_field in batch.columns:
            gender_values = batch[self.gender_field]

        # Prepare a list to store generated values
        generated_values = []

        # Process each value according to null strategy
        for idx, value in enumerate(field_values):

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
                "record_id": record_id,
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
            except Exception as e:
                self.logger.error(
                    f"Error generating name for value '{value}': {str(e)}"
                )
                generated_values.append(value if pd.notna(value) else np.nan)

        # Update the dataframe with generated values
        result[self.output_field_name] = generated_values

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
        if (
            self.consistency_mechanism == "mapping"
            and hasattr(self, "mapping_store")
            and self.mapping_store
        ):
            # Check if we already have a mapping for this value
            synthetic_value = self.mapping_store.get_mapping(self.field_name, value)

            if synthetic_value is not None:
                return synthetic_value

        # Generate new value based on consistency mechanism
        if self.consistency_mechanism == "prgn":
            # Ensure the generator has PRGN capabilities
            if (
                not hasattr(self.generator, "prgn_generator")
                or self.generator.prgn_generator is None
            ):
                # Set up PRGN generator if not already done
                if not hasattr(self, "_prgn_generator"):
                    self._prgn_generator = PRNGenerator(
                        global_seed=self.key or "name-generation"
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
            self.mapping_store.add_mapping(self.field_name, value, synthetic_value)

        return synthetic_value

    def _collect_specific_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics specific to the Name Generation operation.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame.

        Returns
        -------
        Dict[str, Any]
            Dictionary of operation-specific metrics.
        """
        metrics_data = {
            "name_generator": {},
            "performance": {},
            "dictionary_metrics": {},
        }

        # === 1. Name generator configuration ===
        gen = self.generator
        metrics_data["name_generator"].update({
            "language": getattr(self, "language", None),
            "gender_field": getattr(self, "gender_field", None),
            "gender_from_name": getattr(gen, "gender_from_name", False),
            "format": getattr(gen, "format", "FL"),
            "use_faker": getattr(gen, "use_faker", False),
        })

        # === 2. Performance metrics ===
        if getattr(self, "start_time", None) is not None:
            try:
                execution_time = time.time() - self.start_time
                records_per_sec = int(self.process_count / execution_time) if execution_time > 0 else 0
                metrics_data["performance"] = {
                    "generation_time": round(execution_time, 4),
                    "records_per_second": records_per_sec,
                }
            except Exception as e:
                self.logger.warning(f"Error calculating performance metrics: {str(e)}")

        # === 3. Dictionary metrics ===
        metrics_data["dictionary_metrics"] = self._collect_dictionary_metrics(gen)

        # === 4. Length statistics (original vs generated) ===
        try:
            if self.mode == "REPLACE":
                # Measure single field
                metrics_data["name_generator"]["length_stats"] = self._calculate_length_stats(df[self.field_name])
            else:  # ENRICH
                output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"
                if output_field in df.columns:
                    metrics_data["name_generator"].update({
                        "original_length_stats": self._calculate_length_stats(df[self.field_name]),
                        "generated_length_stats": self._calculate_length_stats(df[output_field]),
                    })
        except Exception as e:
            self.logger.warning(f"Error collecting length metrics: {str(e)}")
            metrics_data["name_generator"]["length_stats_error"] = str(e)

        return metrics_data

    def _collect_dictionary_metrics(self, generator) -> Dict[str, Any]:
        """Helper to collect dictionary size and language variant info."""
        dictionary_metrics = {"total_dictionary_entries": 0, "language_variants": []}

        for lang in ["en", "ru", "vi"]:
            try:
                male_first = len(getattr(generator, "_first_names_male", {}).get(lang, []))
                female_first = len(getattr(generator, "_first_names_female", {}).get(lang, []))
                last = len(getattr(generator, "_last_names", {}).get(lang, []))
                male_middle = len(getattr(generator, "_middle_names_male", {}).get(lang, []))
                female_middle = len(getattr(generator, "_middle_names_female", {}).get(lang, []))

                total = male_first + female_first + last + male_middle + female_middle
                if total > 0:
                    dictionary_metrics["language_variants"].append(lang)
                    dictionary_metrics["total_dictionary_entries"] += total
            except Exception as e:
                self.logger.warning(f"Error collecting dictionary info for {lang}: {str(e)}")

        return dictionary_metrics

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
            return {"min": 0, "max": 0, "mean": 0, "median": 0}

        return {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        }

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        return {
            "language": self.language,
            "gender_field": self.gender_field,
            "gender_from_name": self.gender_from_name,
            "format": self.format,
            "f_m_ratio": self.f_m_ratio,
            "use_faker": self.use_faker,
            "case": self.case,
            "dictionaries": self.dictionaries,
            "consistency_mechanism": self.consistency_mechanism,
            "mapping_store_path": self.mapping_store_path,
            "id_field": self.id_field,
            "key": self.key,
            "context_salt": self.context_salt,
            "save_mapping": self.save_mapping,
            "column_prefix": self.column_prefix,
        }
