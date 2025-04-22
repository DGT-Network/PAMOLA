"""
Phone number generation operation for fake data system.

This module provides the PhoneOperation class for generating synthetic phone numbers
while maintaining statistical properties of the original data and supporting
consistent mapping.
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
from pamola_core.fake_data.generators.phone import PhoneGenerator
from pamola_core.utils import io
from pamola_core.utils.ops.op_registry import register

# Configure logger
logger = logging.getLogger(__name__)


@register()
class PhoneOperation(GeneratorOperation):
    """
    Operation for generating synthetic phone numbers.

    This operation processes phone numbers in datasets, replacing them with
    synthetic alternatives while preserving country and operator characteristics.
    """

    name = "phone_generator"
    description = "Generates synthetic phone numbers with configurable formats and regions"
    category = "fake_data"

    def __init__(self, field_name: str,
                 mode: str = "ENRICH",
                 output_field_name: Optional[str] = None,
                 country_codes: Optional[Union[List, Dict]] = None,
                 operator_codes_dict: Optional[str] = None,
                 format: Optional[str] = None,
                 validate_source: bool = True,
                 handle_invalid_phone: str = "generate_new",
                 default_country: str = "us",
                 preserve_country_code: bool = True,
                 preserve_operator_code: bool = False,
                 region: Optional[str] = None,
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
                 international_format: bool = True,
                 local_formatting: bool = False,
                 country_code_field: Optional[str] = None,
                 detailed_metrics: bool = False,
                 error_logging_level: str = "WARNING",
                 max_retries: int = 3):
        """
        Initialize phone number generation operation.

        Args:
            field_name: Field to process (containing phone numbers)
            mode: Operation mode (REPLACE or ENRICH)
            output_field_name: Name for the output field (if mode=ENRICH)
            country_codes: Country codes to use (list or dict with weights)
            operator_codes_dict: Path to dictionary of operator codes
            format: Format for phone number generation
            validate_source: Whether to validate source phone numbers
            handle_invalid_phone: How to handle invalid numbers
            default_country: Default country for generation
            preserve_country_code: Whether to preserve country code from original
            preserve_operator_code: Whether to preserve operator code from original
            region: Region/country for formatting
            batch_size: Number of records to process in one batch
            null_strategy: Strategy for handling NULL values
            consistency_mechanism: Method for ensuring consistency (mapping or prgn)
            mapping_store_path: Path to store mappings
            id_field: Field for record identification
            key: Key for encryption/PRGN
            context_salt: Salt for PRGN
            save_mapping: Whether to save mapping to file
            column_prefix: Prefix for new column (if mode=ENRICH)
            international_format: Whether to use international format (with country code)
            local_formatting: Whether to apply local country formatting
            country_code_field: Field containing country codes
            detailed_metrics: Whether to collect detailed metrics
            error_logging_level: Level for error logging (ERROR, WARNING, INFO)
            max_retries: Maximum number of retries for generation on error
        """
        # Save new parameters
        self.detailed_metrics = detailed_metrics
        self.error_logging_level = error_logging_level.upper()
        self.max_retries = max_retries
        self.international_format = international_format
        self.local_formatting = local_formatting
        self.country_code_field = country_code_field

        # Configure logging level
        self._configure_logging()

        # Store attributes locally that we need to access directly
        self.column_prefix = column_prefix
        self.id_field = id_field
        self.key = key
        self.context_salt = context_salt
        self.save_mapping = save_mapping
        self.mapping_store_path = mapping_store_path
        self.default_country = default_country
        self.region = region or default_country

        # Set up phone generator configuration
        generator_params = {
            "country_codes": country_codes,
            "operator_codes_dict": operator_codes_dict,
            "format": format,
            "validate_source": validate_source,
            "handle_invalid_phone": handle_invalid_phone,
            "default_country": default_country,
            "preserve_country_code": preserve_country_code,
            "preserve_operator_code": preserve_operator_code,
            "region": self.region,
            "key": key,
            "context_salt": context_salt
        }

        # Create a temporary BaseGenerator to pass to parent constructor
        # We'll replace it with the real PhoneGenerator after initialization
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

        # Now create and store the real phone generator
        self.generator = PhoneGenerator(config=generator_params)

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
            self._country_stats = Counter()
            self._format_stats = Counter()
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
        Execute the phone number generation operation.

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
        Process a batch of data to generate synthetic phone numbers.

        Args:
            batch: DataFrame batch to process

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
                "record_id": record_id
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
                self.process_count += 1
            except Exception as e:
                logger.error(f"Error generating phone number for value '{value}': {str(e)}")
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

                    # Try to collect country code statistics
                    if isinstance(synthetic_value, str):
                        country_code = self.generator.extract_country_code(synthetic_value)
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
                    if not hasattr(self.generator, "prgn_generator") or self.generator.prgn_generator is None:
                        # Set up PRGN generator if not already done
                        if not hasattr(self, "_prgn_generator"):
                            from pamola_core.fake_data.commons.prgn import PRNGenerator
                            self._prgn_generator = PRNGenerator(global_seed=self.key or "phone-generation")
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

                    # Collect country code statistics
                    if isinstance(synthetic_value, str):
                        country_code = self.generator.extract_country_code(synthetic_value)
                        if country_code:
                            self._country_stats[country_code] += 1

                    # Try to collect format statistics
                    if isinstance(synthetic_value, str):
                        has_parentheses = "(" in synthetic_value and ")" in synthetic_value
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
                    logger.debug(f"Retry {retries}/{self.max_retries} generating phone for value '{value}': {str(e)}")
                else:
                    logger.error(
                        f"Failed to generate phone for value '{value}' after {self.max_retries} retries: {str(e)}")
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
        if self.detailed_metrics and hasattr(self, "_country_stats") and self._country_stats:
            top_countries = self._country_stats.most_common(10)
            total = sum(self._country_stats.values())
            country_distribution = {country: count / total for country, count in top_countries}

            # Calculate diversity metrics
            unique_countries = len(self._country_stats)
            diversity_ratio = unique_countries / total if total > 0 else 0

            return {
                "total_phones": total,
                "unique_country_codes": unique_countries,
                "diversity_ratio": diversity_ratio,
                "top_country_codes": country_distribution
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
        country_distribution = {country: count / total for country, count in top_countries}

        # Calculate diversity metrics
        unique_countries = len(country_counts)
        diversity_ratio = unique_countries / total if total > 0 else 0

        return {
            "total_phones": total,
            "unique_country_codes": unique_countries,
            "diversity_ratio": diversity_ratio,
            "top_country_codes": country_distribution
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
        if self.detailed_metrics and hasattr(self, "_format_stats") and self._format_stats:
            total = sum(self._format_stats.values())
            format_distribution = {format_type: count / total for format_type, count in
                                   self._format_stats.most_common()}

            return {
                "format_distribution": format_distribution
            }

        # Otherwise analyze formats from phone numbers
        formats = {
            "international_parentheses_dashes": 0,
            "international_spaces": 0,
            "e164": 0,
            "local_parentheses_dashes": 0,
            "unknown": 0
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
        format_distribution = {format_type: count / total for format_type, count in formats.items() if count > 0}

        return {
            "format_distribution": format_distribution
        }

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics for the phone generation operation.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary with metrics
        """
        # Get basic metrics from parent
        metrics_data = super()._collect_metrics(df)

        # Add phone-specific metrics
        metrics_data["phone_generator"] = {
            "format": getattr(self.generator, "format", None),
            "default_country": self.default_country,
            "region": self.region,
            "international_format": self.international_format,
            "local_formatting": self.local_formatting,
            "preserve_country_code": getattr(self.generator, "preserve_country_code", True),
            "preserve_operator_code": getattr(self.generator, "preserve_operator_code", False),
            "validate_source": getattr(self.generator, "validate_source", True),
            "handle_invalid_phone": getattr(self.generator, "handle_invalid_phone", "generate_new")
        }

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

        # Add country distribution
        try:
            # Get country distribution
            country_metrics = self._analyze_country_distribution(df)
            if country_metrics:
                metrics_data["phone_generator"]["country_distribution"] = country_metrics
        except Exception as e:
            logger.warning(f"Error collecting country distribution: {str(e)}")

        # Add format distribution
        try:
            # Get format distribution
            format_metrics = self._analyze_formats(df)
            if format_metrics:
                metrics_data["phone_generator"]["format_distribution"] = format_metrics
        except Exception as e:
            logger.warning(f"Error collecting format distribution: {str(e)}")

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
            length_similarity = 1.0 - min(1.0, abs(orig_mean_len - gen_mean_len) / orig_mean_len)
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
            country_preservation_count = sum(1 for o, g in zip(orig_countries, gen_countries) if o == g)
            country_preservation_ratio = country_preservation_count / len(orig_countries)
            metrics["country_code_preservation_ratio"] = country_preservation_ratio

            # Calculate country code diversity
            orig_country_count = len(set(orig_countries))
            gen_country_count = len(set(gen_countries))
            country_diversity_ratio = gen_country_count / orig_country_count if orig_country_count > 0 else 0
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
            format_preservation_count = sum(1 for o, g in zip(orig_formats, gen_formats) if o == g)
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