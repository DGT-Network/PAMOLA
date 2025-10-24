"""
Phone number generation operation for fake data system.

This module provides the PhoneOperation class for generating synthetic phone numbers
while maintaining statistical properties of the original data and supporting
consistent mapping.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from pamola_core.fake_data.base_generator_op import GeneratorOperation
from pamola_core.fake_data.generators.phone import PhoneGenerator
from pamola_core.fake_data.schemas.phone_op_config import FakePhoneOperationConfig
from pamola_core.utils import io
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult
from pamola_core.utils.progress import HierarchicalProgressTracker


@register(version="1.0.0")
class FakePhoneOperation(GeneratorOperation):
    """
    Operation for generating synthetic phone numbers.

    This operation processes phone numbers in datasets, replacing them with
    synthetic alternatives while preserving country and operator characteristics.
    """

    def __init__(
        self,
        field_name: str,
        country_codes: Optional[Union[List, Dict]] = None,
        operator_codes_dict: Optional[str] = None,
        format: Optional[str] = None,
        validate_source: bool = True,
        handle_invalid_phone: str = "generate_new",
        default_country: str = "us",
        preserve_country_code: bool = True,
        preserve_operator_code: bool = False,
        region: Optional[str] = None,
        detailed_metrics: bool = False,
        max_retries: int = 3,
        key: Optional[str] = None,
        context_salt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize FakePhoneOperation.

        Parameters
        ----------
        field_name : str
            Column containing phone numbers.
        country_codes : list or dict, optional
            Country codes for generation or weighting.
        operator_codes_dict : str, optional
            Path to operator code dictionary.
        format : str, optional
            Formatting style for generated phone numbers.
        validate_source : bool, optional
            Whether to validate input phone numbers.
        handle_invalid_phone : str, optional
            Strategy for invalid numbers ('generate_new', etc.).
        default_country : str, optional
            Default country to use for generation.
        preserve_country_code : bool, optional
            Preserve original country code.
        preserve_operator_code : bool, optional
            Preserve operator prefix.
        region : str, optional
            Regional formatting specifier.
        detailed_metrics : bool, optional
            Collect detailed generation statistics.
        max_retries : int, optional
            Max retries for generation.
        key : str, optional
            Encryption or PRGN key.
        context_salt : str, optional
            PRGN salt for uniqueness.
        **kwargs : dict
            Forwarded to BaseOperation / GeneratorOperation.
        """
        # Default description
        kwargs.setdefault(
            "description",
            f"Synthetic phone number generation for '{field_name}' with configurable country/operator formats",
        )

        #  Configuration object
        config = FakePhoneOperationConfig(
            field_name=field_name,
            country_codes=country_codes,
            operator_codes_dict=operator_codes_dict,
            format=format,
            validate_source=validate_source,
            handle_invalid_phone=handle_invalid_phone,
            default_country=default_country,
            preserve_country_code=preserve_country_code,
            preserve_operator_code=preserve_operator_code,
            region=region,
            detailed_metrics=detailed_metrics,
            max_retries=max_retries,
            key=key,
            context_salt=context_salt,
            **kwargs,
        )

        # Generator parameters
        generator_params = {
            "country_codes": country_codes,
            "operator_codes_dict": operator_codes_dict,
            "format": format,
            "validate_source": validate_source,
            "handle_invalid_phone": handle_invalid_phone,
            "default_country": default_country,
            "preserve_country_code": preserve_country_code,
            "preserve_operator_code": preserve_operator_code,
            "region": region,
            "key": key,
            "context_salt": context_salt,
        }

        # Initialize generator
        phone_generator = PhoneGenerator(config=generator_params)

        # Pass config to parent
        kwargs["config"] = config

        super().__init__(
            field_name=field_name,
            generator=phone_generator,
            generator_params=generator_params,
            **kwargs,
        )

        # Expose config attributes
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Runtime stats
        self.operation_name = self.__class__.__name__
        self._original_df = None
        self.error_count = 0
        self.retry_count = 0

        # Optional metrics
        if self.detailed_metrics:
            self._country_stats = Counter()
            self._format_stats = Counter()
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
        Process a batch of data to generate synthetic phone numbers.

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

        # Extract country codes if field is provided
        country_codes = None
        if self.country_code_field and self.country_code_field in batch.columns:
            country_codes = batch[self.country_code_field]

        # Prepare a list to store generated values
        generated_values = []

        # Process each value according to null strategy
        for idx, value in enumerate(field_values):
            # Get country code for this record if available
            country_code = None
            if country_codes is not None:
                country_code = country_codes.iloc[idx]
                if pd.isna(country_code):
                    country_code = None
                elif isinstance(country_code, (int, float)):
                    # Convert numeric country code to string
                    country_code = str(int(country_code))

            # Get record identifier if available (for mapping store)
            record_id = None
            if self.id_field and self.id_field in batch.columns:
                record_id = batch[self.id_field].iloc[idx]

            # Determine parameters for generation
            gen_params = {
                "country_code": country_code,
                "region": self.region,
                "international_format": self.international_format,
                "local_formatting": self.local_formatting,
                "context_salt": self.context_salt or "phone-generation",
                "record_id": record_id,
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
            except Exception as e:
                self.logger.error(
                    f"Error generating phone number for value '{value}': {str(e)}"
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

                    # Try to collect country code statistics
                    if isinstance(synthetic_value, str):
                        country_code = self.generator.extract_country_code(
                            synthetic_value
                        )
                        if country_code:
                            self._country_stats[country_code] += 1

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
                                global_seed=self.key or "phone-generation"
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

                    # Collect country code statistics
                    if isinstance(synthetic_value, str):
                        country_code = self.generator.extract_country_code(
                            synthetic_value
                        )
                        if country_code:
                            self._country_stats[country_code] += 1

                    # Try to collect format statistics
                    if isinstance(synthetic_value, str):
                        has_parentheses = (
                            "(" in synthetic_value and ")" in synthetic_value
                        )
                        has_dashes = "-" in synthetic_value
                        has_spaces = " " in synthetic_value
                        has_plus = synthetic_value.startswith("+")

                        format_type = "unknown"
                        if has_plus and has_parentheses and has_dashes:
                            format_type = "international_parentheses_dashes"
                        elif has_plus and has_spaces:
                            format_type = "international_spaces"
                        elif has_plus:
                            format_type = "e164"
                        elif has_parentheses and has_dashes:
                            format_type = "local_parentheses_dashes"

                        self._format_stats[format_type] += 1

                return synthetic_value

            except Exception as e:
                last_error = e
                retries += 1
                self.retry_count += 1

                # Log error with appropriate level
                if retries <= self.max_retries:
                    self.logger.debug(
                        f"Retry {retries}/{self.max_retries} generating phone for value '{value}': {str(e)}"
                    )
                else:
                    self.logger.error(
                        f"Failed to generate phone for value '{value}' after {self.max_retries} retries: {str(e)}"
                    )
                    self.error_count += 1

        # If all attempts failed, return original value or None
        if pd.notna(value):
            return value
        return None

    def _analyze_country_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of country codes in generated phone numbers.

        Args:
            df: Processed DataFrame

        Returns:
            Country distribution metrics
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
            and hasattr(self, "_country_stats")
            and self._country_stats
        ):
            top_countries = self._country_stats.most_common(10)
            total = sum(self._country_stats.values())
            country_distribution = {
                country: count / total for country, count in top_countries
            }

            # Calculate diversity metrics
            unique_countries = len(self._country_stats)
            diversity_ratio = unique_countries / total if total > 0 else 0

            return {
                "total_phones": total,
                "unique_country_codes": unique_countries,
                "diversity_ratio": diversity_ratio,
                "top_country_codes": country_distribution,
            }

        # Otherwise extract country codes from phone numbers
        country_codes = []
        for phone in df[field].dropna():
            if isinstance(phone, str):
                country_code = self.generator.extract_country_code(phone)
                if country_code:
                    country_codes.append(country_code)

        if not country_codes:
            return {}

        # Calculate country code frequency
        country_counts = Counter(country_codes)
        total = len(country_codes)

        # Get top country codes (max 10)
        top_countries = country_counts.most_common(10)
        country_distribution = {
            country: count / total for country, count in top_countries
        }

        # Calculate diversity metrics
        unique_countries = len(country_counts)
        diversity_ratio = unique_countries / total if total > 0 else 0

        return {
            "total_phones": total,
            "unique_country_codes": unique_countries,
            "diversity_ratio": diversity_ratio,
            "top_country_codes": country_distribution,
        }

    def _analyze_formats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of phone number formats in generated numbers.

        Args:
            df: Processed DataFrame

        Returns:
            Format distribution metrics
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
            and hasattr(self, "_format_stats")
            and self._format_stats
        ):
            total = sum(self._format_stats.values())
            format_distribution = {
                format_type: count / total
                for format_type, count in self._format_stats.most_common()
            }

            return {"format_distribution": format_distribution}

        # Otherwise analyze formats from phone numbers
        formats = {
            "international_parentheses_dashes": 0,
            "international_spaces": 0,
            "e164": 0,
            "local_parentheses_dashes": 0,
            "unknown": 0,
        }

        total = 0

        for phone in df[field].dropna():
            if isinstance(phone, str):
                total += 1
                has_parentheses = "(" in phone and ")" in phone
                has_dashes = "-" in phone
                has_spaces = " " in phone
                has_plus = phone.startswith("+")

                if has_plus and has_parentheses and has_dashes:
                    formats["international_parentheses_dashes"] += 1
                elif has_plus and has_spaces:
                    formats["international_spaces"] += 1
                elif has_plus:
                    formats["e164"] += 1
                elif has_parentheses and has_dashes:
                    formats["local_parentheses_dashes"] += 1
                else:
                    formats["unknown"] += 1

        if total == 0:
            return {}

        # Calculate distribution
        format_distribution = {
            format_type: count / total
            for format_type, count in formats.items()
            if count > 0
        }

        return {"format_distribution": format_distribution}

    def _collect_specific_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics specific to the phone number generation operation.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame

        Returns
        -------
        Dict[str, Any]
            Dictionary containing phone-specific metrics.
        """
        metrics_data = {}

        gen = self.generator

        # --- 1. Phone generator info ---
        metrics_data["phone_generator"] = {
            "format": getattr(gen, "format", None),
            "default_country": getattr(self, "default_country", None),
            "region": getattr(self, "region", None),
            "international_format": getattr(self, "international_format", False),
            "local_formatting": getattr(self, "local_formatting", False),
            "preserve_country_code": getattr(gen, "preserve_country_code", True),
            "preserve_operator_code": getattr(gen, "preserve_operator_code", False),
            "validate_source": getattr(gen, "validate_source", True),
            "handle_invalid_phone": getattr(
                gen, "handle_invalid_phone", "generate_new"
            ),
        }

        # --- 2. Distribution metrics ---
        distributions = self._collect_phone_distributions(df)
        if distributions:
            metrics_data["phone_generator"].update(distributions)

        # --- 3. Quality metrics (for ENRICH mode only) ---
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

    def _collect_phone_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collects country and format distributions for phone numbers.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame

        Returns
        -------
        Dict[str, Any]
            Dictionary of phone-related distribution metrics.
        """
        distributions = {}

        # --- Country distribution ---
        try:
            country_metrics = self._analyze_country_distribution(df)
            if country_metrics:
                distributions["country_distribution"] = country_metrics
        except Exception as e:
            self.logger.warning(f"Error collecting country distribution: {str(e)}")

        # --- Format distribution ---
        try:
            format_metrics = self._analyze_formats(df)
            if format_metrics:
                distributions["format_distribution"] = format_metrics
        except Exception as e:
            self.logger.warning(f"Error collecting format distribution: {str(e)}")

        return distributions

    def _calculate_quality_metrics(
        self, original_series: pd.Series, generated_series: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics comparing original and generated phone numbers.

        Args:
            original_series: Series with original phone numbers
            generated_series: Series with generated phone numbers

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

        # Analyze country code preservation
        orig_countries = []
        gen_countries = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if isinstance(orig, str) and isinstance(gen, str):
                orig_country = self.generator.extract_country_code(orig)
                gen_country = self.generator.extract_country_code(gen)

                if orig_country and gen_country:
                    orig_countries.append(orig_country)
                    gen_countries.append(gen_country)

        if orig_countries and gen_countries:
            # Calculate percentage of preserved country codes
            country_preservation_count = sum(
                1 for o, g in zip(orig_countries, gen_countries) if o == g
            )
            country_preservation_ratio = country_preservation_count / len(
                orig_countries
            )
            metrics["country_code_preservation_ratio"] = country_preservation_ratio

            # Calculate country code diversity
            orig_country_count = len(set(orig_countries))
            gen_country_count = len(set(gen_countries))
            country_diversity_ratio = (
                gen_country_count / orig_country_count if orig_country_count > 0 else 0
            )
            metrics["country_code_diversity_ratio"] = country_diversity_ratio

        # Analyze formatting similarity
        orig_formats = []
        gen_formats = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if isinstance(orig, str) and isinstance(gen, str):
                # Determine format by simple characteristics
                orig_format = self._determine_format_type(orig)
                gen_format = self._determine_format_type(gen)

                orig_formats.append(orig_format)
                gen_formats.append(gen_format)

        if orig_formats and gen_formats:
            # Calculate format preservation
            format_preservation_count = sum(
                1 for o, g in zip(orig_formats, gen_formats) if o == g
            )
            format_preservation_ratio = format_preservation_count / len(orig_formats)
            metrics["format_preservation_ratio"] = format_preservation_ratio

        return metrics

    def _determine_format_type(self, phone: str) -> str:
        """
        Determine the format type of a phone number.

        Args:
            phone: Phone number

        Returns:
            Format type identifier
        """
        has_parentheses = "(" in phone and ")" in phone
        has_dashes = "-" in phone
        has_spaces = " " in phone
        has_plus = phone.startswith("+")

        if has_plus and has_parentheses and has_dashes:
            return "international_parentheses_dashes"
        elif has_plus and has_spaces:
            return "international_spaces"
        elif has_plus:
            return "e164"
        elif has_parentheses and has_dashes:
            return "local_parentheses_dashes"
        elif has_dashes:
            return "local_dashes"
        elif has_spaces:
            return "local_spaces"
        else:
            return "numeric"

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        return {
            "format": self.format,
            "country": self.country,
            "region": self.region,
            "area_codes": self.area_codes,
            "preserve_area_code": self.preserve_area_code,
            "handle_invalid_phone": self.handle_invalid_phone,
            "consistency_mechanism": self.consistency_mechanism,
            "mapping_store_path": self.mapping_store_path,
            "id_field": self.id_field,
            "key": self.key,
            "context_salt": self.context_salt,
            "save_mapping": self.save_mapping,
            "column_prefix": self.column_prefix,
            "detailed_metrics": self.detailed_metrics,
        }
