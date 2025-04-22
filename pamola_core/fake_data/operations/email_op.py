"""
Email address generation operation for fake data system.

This module provides the EmailOperation class for generating synthetic email addresses
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
from pamola_core.fake_data.generators.email import EmailGenerator
from pamola_core.utils import io
from pamola_core.utils.ops.op_registry import register

# Configure logger
logger = logging.getLogger(__name__)


@register()
class EmailOperation(GeneratorOperation):
    """
    Operation for generating synthetic email addresses.

    This operation processes email addresses in datasets, replacing them with
    synthetic alternatives while preserving domain characteristics and/or
    using name information to generate realistic addresses.
    """

    name = "email_generator"
    description = "Generates synthetic email addresses with configurable formats"
    category = "fake_data"

    def __init__(self, field_name: str,
                 mode: str = "ENRICH",
                 output_field_name: Optional[str] = None,
                 domains: Optional[Union[List[str], str]] = None,
                 format: Optional[str] = None,
                 format_ratio: Optional[Dict[str, float]] = None,
                 first_name_field: Optional[str] = None,
                 last_name_field: Optional[str] = None,
                 full_name_field: Optional[str] = None,
                 name_format: Optional[str] = None,
                 validate_source: bool = True,
                 handle_invalid_email: str = "generate_new",
                 nicknames_dict: Optional[str] = None,
                 max_length: int = 254,
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
                 separator_options: Optional[List[str]] = None,
                 number_suffix_probability: float = 0.4,
                 preserve_domain_ratio: float = 0.5,
                 business_domain_ratio: float = 0.2,
                 detailed_metrics: bool = False,  # Collect detailed metrics
                 error_logging_level: str = "WARNING",  # Error logging level
                 max_retries: int = 3):  # Maximum retry attempts on error
        """
        Initialize email generation operation.

        Args:
            field_name: Field to process (containing email addresses)
            mode: Operation mode (REPLACE or ENRICH)
            output_field_name: Name for the output field (if mode=ENRICH)
            domains: List of domains or path to domain dictionary
            format: Format for email generation (name_surname, surname_name, nickname, existing_domain)
            format_ratio: Distribution of format usage (e.g. {'name_surname': 0.4, 'nickname': 0.6})
            first_name_field: Field containing first names
            last_name_field: Field containing last names
            full_name_field: Field containing full names (alternative to separate first/last name fields)
            name_format: Format of full names (FL, FML, LF, etc.)
            validate_source: Whether to validate source email addresses
            handle_invalid_email: How to handle invalid emails (generate_new, keep_empty, generate_with_default_domain)
            nicknames_dict: Path to nickname dictionary
            max_length: Maximum length for email address
            batch_size: Number of records to process in one batch
            null_strategy: Strategy for handling NULL values
            consistency_mechanism: Method for ensuring consistency (mapping or prgn)
            mapping_store_path: Path to store mappings
            id_field: Field for record identification
            key: Key for encryption/PRGN
            context_salt: Salt for PRGN
            save_mapping: Whether to save mapping to file
            column_prefix: Prefix for new column (if mode=ENRICH)
            separator_options: List of separators to use in email generation
            number_suffix_probability: Probability of adding number suffix to email
            preserve_domain_ratio: Probability of preserving original domain
            business_domain_ratio: Probability of using business domains
            detailed_metrics: Whether to collect detailed metrics
            error_logging_level: Level for error logging (ERROR, WARNING, INFO)
            max_retries: Maximum number of retries for generation on error
        """
        # Save new parameters
        self.detailed_metrics = detailed_metrics
        self.error_logging_level = error_logging_level.upper()
        self.max_retries = max_retries

        # Configure logging level
        self._configure_logging()

        # Store attributes locally that we need to access directly
        self.column_prefix = column_prefix
        self.id_field = id_field
        self.key = key
        self.context_salt = context_salt
        self.save_mapping = save_mapping
        self.mapping_store_path = mapping_store_path
        self.first_name_field = first_name_field
        self.last_name_field = last_name_field
        self.full_name_field = full_name_field
        self.name_format = name_format

        # Set up email generator configuration
        generator_params = {
            "domains": domains,
            "format": format,
            "format_ratio": format_ratio,
            "validate_source": validate_source,
            "handle_invalid_email": handle_invalid_email,
            "nicknames_dict": nicknames_dict,
            "max_length": max_length,
            "key": key,
            "context_salt": context_salt,
            # Add new configuration parameters
            "separator_options": separator_options,
            "number_suffix_probability": number_suffix_probability,
            "preserve_domain_ratio": preserve_domain_ratio,
            "business_domain_ratio": business_domain_ratio
        }

        # Create a temporary BaseGenerator to pass to parent constructor
        # We'll replace it with the real EmailGenerator after initialization
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

        # Now create and store the real email generator
        self.generator = EmailGenerator(config=generator_params)

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
            self._domain_stats = Counter()
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
        Execute the email generation operation.

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
        Process a batch of data to generate synthetic email addresses.

        Args:
            batch: DataFrame batch to process

        Returns:
            Processed DataFrame batch
        """
        # Get the field value series
        field_values = batch[self.field_name].copy()

        # Create result dataframe
        result = batch.copy()

        # Extract name information if available
        first_names = None
        last_names = None
        full_names = None

        if self.first_name_field and self.first_name_field in batch.columns:
            first_names = batch[self.first_name_field]

        if self.last_name_field and self.last_name_field in batch.columns:
            last_names = batch[self.last_name_field]

        if self.full_name_field and self.full_name_field in batch.columns:
            full_names = batch[self.full_name_field]

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

            # Get name information for this record
            first_name = None
            last_name = None
            full_name = None

            if first_names is not None:
                first_name = first_names.iloc[idx]
                if pd.isna(first_name):
                    first_name = None

            if last_names is not None:
                last_name = last_names.iloc[idx]
                if pd.isna(last_name):
                    last_name = None

            if full_names is not None:
                full_name = full_names.iloc[idx]
                if pd.isna(full_name):
                    full_name = None

            # Get record identifier if available (for mapping store)
            record_id = None
            if self.id_field and self.id_field in batch.columns:
                record_id = batch[self.id_field].iloc[idx]

            # Determine parameters for generation
            gen_params = {
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "name_format": self.name_format,
                "original_email": value if pd.notna(value) else None,
                "context_salt": self.context_salt or "email-generation",
                "record_id": record_id
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
                self.process_count += 1
            except Exception as e:
                logger.error(f"Error generating email for value '{value}': {str(e)}")
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

                    # Try to collect domain statistics
                    if isinstance(synthetic_value, str) and '@' in synthetic_value:
                        domain = synthetic_value.split('@')[-1]
                        self._domain_stats[domain] += 1

                    # And format statistics
                    if hasattr(self.generator, "parse_email_format"):
                        try:
                            format_used = self.generator.parse_email_format(synthetic_value)
                            self._format_stats[format_used] += 1
                        except:
                            pass

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
                            self._prgn_generator = PRNGenerator(global_seed=self.key or "email-generation")
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

                    # Collect domain statistics
                    if isinstance(synthetic_value, str) and '@' in synthetic_value:
                        domain = synthetic_value.split('@')[-1]
                        self._domain_stats[domain] += 1

                    # And format statistics
                    if hasattr(self.generator, "parse_email_format"):
                        try:
                            format_used = self.generator.parse_email_format(synthetic_value)
                            self._format_stats[format_used] += 1
                        except:
                            pass

                return synthetic_value

            except Exception as e:
                last_error = e
                retries += 1
                self.retry_count += 1

                # Log error with appropriate level
                if retries <= self.max_retries:
                    logger.debug(f"Retry {retries}/{self.max_retries} generating email for value '{value}': {str(e)}")
                else:
                    logger.error(
                        f"Failed to generate email for value '{value}' after {self.max_retries} retries: {str(e)}")
                    self.error_count += 1

        # If all attempts failed, return original value or None
        if pd.notna(value):
            return value
        return None

    def _analyze_domain_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of domains in generated emails.

        Args:
            df: Processed DataFrame

        Returns:
            Domain distribution metrics
        """
        # Determine which field to analyze based on mode
        if self.mode == "REPLACE":
            field = self.field_name
        else:  # ENRICH
            field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

        if field not in df.columns:
            return {}

        # If we have detailed metrics and they're already collected, use them
        if self.detailed_metrics and hasattr(self, "_domain_stats") and self._domain_stats:
            top_domains = self._domain_stats.most_common(20)
            total = sum(self._domain_stats.values())
            domain_distribution = {domain: count / total for domain, count in top_domains}

            # Calculate diversity metrics
            unique_domains = len(self._domain_stats)
            diversity_ratio = unique_domains / total if total > 0 else 0

            # Add domain categorization
            domain_categories = self._categorize_domains_distribution(self._domain_stats)

            return {
                "total_emails": total,
                "unique_domains": unique_domains,
                "diversity_ratio": diversity_ratio,
                "top_domains": domain_distribution,
                "domain_categories": domain_categories
            }

        # Otherwise extract domains from email addresses
        domains = []
        for email in df[field].dropna():
            if isinstance(email, str) and '@' in email:
                domain = email.split('@')[-1]
                domains.append(domain)

        if not domains:
            return {}

        # Calculate domain frequency
        domain_counts = Counter(domains)
        total = len(domains)

        # Get top domains (max 20)
        top_domains = domain_counts.most_common(20)
        domain_distribution = {domain: count / total for domain, count in top_domains}

        # Calculate diversity metrics
        unique_domains = len(domain_counts)
        diversity_ratio = unique_domains / total if total > 0 else 0

        # Add domain categorization
        domain_categories = self._categorize_domains_distribution(domain_counts)

        return {
            "total_emails": total,
            "unique_domains": unique_domains,
            "diversity_ratio": diversity_ratio,
            "top_domains": domain_distribution,
            "domain_categories": domain_categories
        }

    def _categorize_domains_distribution(self, domain_counts: Counter) -> Dict[str, float]:
        """
        Categorize domains into business, personal, educational, etc.

        Args:
            domain_counts: Counter with domain frequencies

        Returns:
            Dictionary with domain category distribution
        """
        categories = {
            "common": 0,
            "business": 0,
            "educational": 0,
            "others": 0
        }

        total = sum(domain_counts.values())
        if total == 0:
            return categories

        # Extract domain dictionaries for comparison
        from pamola_core.fake_data.dictionaries import domains as domain_dicts

        try:
            common_domains = set(domain_dicts.get_common_email_domains())
            business_domains = set(domain_dicts.get_business_email_domains())
            educational_domains = set(domain_dicts.get_educational_email_domains())

            # Count occurrence of each category
            for domain, count in domain_counts.items():
                if domain in common_domains:
                    categories["common"] += count
                elif domain in business_domains:
                    categories["business"] += count
                elif domain in educational_domains:
                    categories["educational"] += count
                else:
                    # Heuristic determination of category
                    if any(business_term in domain for business_term in
                           ['company', 'corp', 'enterprise', 'business', 'inc', 'llc', 'agency', 'consulting']):
                        categories["business"] += count
                    elif any(edu_term in domain for edu_term in
                             ['edu', 'ac.', 'university', 'school', 'college']):
                        categories["educational"] += count
                    else:
                        categories["others"] += count

        except Exception:
            # In case of error, use heuristic determination
            for domain, count in domain_counts.items():
                if any(common_term in domain for common_term in
                       ['gmail', 'yahoo', 'hotmail', 'outlook', 'mail', 'proton', 'aol']):
                    categories["common"] += count
                elif any(business_term in domain for business_term in
                         ['company', 'corp', 'enterprise', 'business', 'inc', 'llc', 'agency', 'consulting']):
                    categories["business"] += count
                elif any(edu_term in domain for edu_term in
                         ['edu', 'ac.', 'university', 'school', 'college']):
                    categories["educational"] += count
                else:
                    categories["others"] += count

        # Normalize percentages
        for category in categories:
            categories[category] = categories[category] / total

        return categories

    def _get_popular_domains(self) -> List[str]:
        """
        Get a list of the most popular domains from the generator's dictionary.

        Returns:
            List of the top domains (max 10)
        """
        domains = getattr(self.generator, "_domain_list", [])
        if not domains:
            return []

        # Return the first 10 domains (assuming they are already sorted by popularity)
        return domains[:min(10, len(domains))]

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect metrics for the email generation operation.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary with metrics
        """
        # Get basic metrics from parent
        metrics_data = super()._collect_metrics(df)

        # Add email-specific metrics
        metrics_data["email_generator"] = {
            "format": getattr(self.generator, "format", None),
            "domains_count": len(getattr(self.generator, "_domain_list", [])),
            "validate_source": getattr(self.generator, "validate_source", True),
            "handle_invalid_email": getattr(self.generator, "handle_invalid_email", "generate_new"),
            "name_fields_used": {
                "first_name_field": self.first_name_field,
                "last_name_field": self.last_name_field,
                "full_name_field": self.full_name_field,
                "name_format": self.name_format
            }
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

        # Add domain distribution
        try:
            # Get domain distribution
            domain_metrics = self._analyze_domain_distribution(df)
            if domain_metrics:
                metrics_data["email_generator"]["domain_distribution"] = domain_metrics
        except Exception as e:
            logger.warning(f"Error collecting domain distribution: {str(e)}")

        # Add domain dictionary metrics
        dictionary_metrics = {
            "total_domains": len(getattr(self.generator, "_domain_list", [])),
            "popular_domains": self._get_popular_domains()
        }

        # Add statistics on formats used
        if self.detailed_metrics and hasattr(self, "_format_stats") and self._format_stats:
            total_formats = sum(self._format_stats.values())
            format_distribution = {
                format_name: count / total_formats
                for format_name, count in self._format_stats.most_common()
            }
            dictionary_metrics["format_distribution"] = format_distribution

        metrics_data["dictionary_metrics"] = dictionary_metrics

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
        Calculate quality metrics comparing original and generated email addresses.

        Args:
            original_series: Series with original emails
            generated_series: Series with generated emails

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

        # Analyze domain preservation
        orig_domains = []
        gen_domains = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if isinstance(orig, str) and '@' in orig and isinstance(gen, str) and '@' in gen:
                orig_domains.append(orig.split('@')[-1])
                gen_domains.append(gen.split('@')[-1])

        if orig_domains and gen_domains:
            # Calculate percentage of preserved domains
            domain_preservation_count = sum(1 for o, g in zip(orig_domains, gen_domains) if o == g)
            domain_preservation_ratio = domain_preservation_count / len(orig_domains)
            metrics["domain_preservation_ratio"] = domain_preservation_ratio

            # Calculate domain diversity
            orig_domain_count = len(set(orig_domains))
            gen_domain_count = len(set(gen_domains))
            domain_diversity_ratio = gen_domain_count / orig_domain_count if orig_domain_count > 0 else 0
            metrics["domain_diversity_ratio"] = domain_diversity_ratio

        # Analyze local part structure
        orig_local_parts = []
        gen_local_parts = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if isinstance(orig, str) and '@' in orig and isinstance(gen, str) and '@' in gen:
                orig_local_parts.append(orig.split('@')[0])
                gen_local_parts.append(gen.split('@')[0])

        if orig_local_parts and gen_local_parts:
            # Calculate average local part length
            orig_lp_mean_len = sum(len(lp) for lp in orig_local_parts) / len(orig_local_parts)
            gen_lp_mean_len = sum(len(lp) for lp in gen_local_parts) / len(gen_local_parts)

            # Local part length similarity
            lp_length_similarity = 1.0 - min(1.0, abs(orig_lp_mean_len - gen_lp_mean_len) / orig_lp_mean_len)
            metrics["local_part_length_similarity"] = lp_length_similarity

            # Analyze separator usage
            orig_separator_stats = Counter()
            gen_separator_stats = Counter()

            for lp in orig_local_parts:
                if '.' in lp:
                    orig_separator_stats['.'] += 1
                if '_' in lp:
                    orig_separator_stats['_'] += 1
                if '-' in lp:
                    orig_separator_stats['-'] += 1

            for lp in gen_local_parts:
                if '.' in lp:
                    gen_separator_stats['.'] += 1
                if '_' in lp:
                    gen_separator_stats['_'] += 1
                if '-' in lp:
                    gen_separator_stats['-'] += 1

            # Calculate separator usage similarity
            separator_similarity = {}
            for sep in ['.', '_', '-']:
                orig_ratio = orig_separator_stats[sep] / len(orig_local_parts)
                gen_ratio = gen_separator_stats[sep] / len(gen_local_parts)
                separator_similarity[sep] = 1.0 - min(1.0, abs(orig_ratio - gen_ratio))

            metrics["separator_similarity"] = separator_similarity

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