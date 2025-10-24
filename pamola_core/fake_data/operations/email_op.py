"""
Email address generation operation for fake data system.

This module provides the EmailOperation class for generating synthetic email addresses
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
from pamola_core.fake_data.generators.email import EmailGenerator
from pamola_core.utils import io
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult
from pamola_core.utils.progress import HierarchicalProgressTracker


class FakeEmailOperationConfig(OperationConfig):
    """Configuration for FakeEmailOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- GeneratorOperation-specific fields ---
                    "generator": {"type": ["object", "null"]},
                    "generator_params": {"type": ["object", "null"]},
                    "consistency_mechanism": {
                        "type": "string",
                        "enum": ["mapping", "prgn"],
                        "default": "prgn",
                    },
                    "id_field": {"type": ["string", "null"]},
                    "mapping_store_path": {"type": ["string", "null"]},
                    "mapping_store": {"type": ["object", "null"]},
                    "save_mapping": {"type": "boolean", "default": False},
                    "output_field_name": {"type": ["string", "null"]},
                    # --- FakeEmailOperation-specific fields ---
                    "field_name": {"type": "string"},
                    "domains": {"type": ["array", "string", "null"]},
                    "format": {"type": ["string", "null"]},
                    "format_ratio": {"type": ["object", "null"]},
                    "first_name_field": {"type": ["string", "null"]},
                    "last_name_field": {"type": ["string", "null"]},
                    "full_name_field": {"type": ["string", "null"]},
                    "name_format": {"type": ["string", "null"]},
                    "validate_source": {"type": "boolean", "default": True},
                    "handle_invalid_email": {
                        "type": "string",
                        "enum": [
                            "generate_new",
                            "keep_empty",
                            "generate_with_default_domain",
                        ],
                        "default": "generate_new",
                    },
                    "nicknames_dict": {"type": ["string", "null"]},
                    "max_length": {"type": "integer", "minimum": 1, "default": 254},
                    # --- Generator fine-tuning fields ---
                    "separator_options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": [".", "_", "-", ""],
                    },
                    "number_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                    },
                    "preserve_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                    },
                    "business_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                    },
                    "detailed_metrics": {"type": "boolean", "default": False},
                    "max_retries": {"type": "integer", "minimum": 0, "default": 3},
                    "key": {"type": ["string", "null"]},
                    "context_salt": {"type": ["string", "null"]},
                },
                "required": ["field_name"],
            },
        ],
    }


@register(version="1.0.0")
class FakeEmailOperation(GeneratorOperation):
    """
    Operation for generating synthetic email addresses.

    This operation processes email addresses in datasets, replacing them with
    synthetic alternatives while preserving domain characteristics and/or
    using name information to generate realistic addresses.
    """

    def __init__(
        self,
        field_name: str,
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
        separator_options: Optional[List[str]] = None,
        number_suffix_probability: float = 0.4,
        preserve_domain_ratio: float = 0.5,
        business_domain_ratio: float = 0.2,
        detailed_metrics: bool = False,
        max_retries: int = 3,
        key: Optional[str] = None,
        context_salt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize FakeEmailOperation.

        Parameters
        ----------
        field_name : str
            Target column containing email addresses.
        domains : list or str, optional
            List of available domains or path to domain dictionary.
        format : str, optional
            Email format (e.g., 'first_last', 'nickname', 'existing_domain').
        format_ratio : dict, optional
            Ratio distribution for format usage.
        first_name_field, last_name_field, full_name_field : str, optional
            Reference fields for name-based email generation.
        name_format : str, optional
            Format of full names (e.g., 'FL', 'LF').
        validate_source : bool, optional
            Whether to validate input email addresses.
        handle_invalid_email : str, optional
            Strategy for invalid emails ('generate_new', 'keep_empty', etc.).
        nicknames_dict : str, optional
            Path to nickname mapping file.
        max_length : int, optional
            Maximum email length (default 254).
        separator_options : list, optional
            Separators used between name parts (default ['.', '_', '']).
        number_suffix_probability : float, optional
            Probability of adding numeric suffixes.
        preserve_domain_ratio : float, optional
            Probability of preserving original domain.
        business_domain_ratio : float, optional
            Probability of using business-related domains.
        detailed_metrics : bool, optional
            Whether to collect detailed per-domain statistics.
        max_retries : int, optional
            Maximum retries on generation failure.
        key : str, optional
            Key for encryption or PRGN consistency (if applicable).
        context_salt : str, optional
                Additional context salt for PRGN to enhance uniqueness.
        **kwargs : dict
            Additional parameters forwarded to GeneratorOperation and BaseOperation.
        """

        # Fallback description for this operation
        kwargs.setdefault(
            "description",
            f"Synthetic email generation for '{field_name}' with configurable format and domain patterns",
        )

        # Build configuration object for schema validation
        config = FakeEmailOperationConfig(
            field_name=field_name,
            domains=domains,
            format=format,
            format_ratio=format_ratio,
            first_name_field=first_name_field,
            last_name_field=last_name_field,
            full_name_field=full_name_field,
            name_format=name_format,
            validate_source=validate_source,
            handle_invalid_email=handle_invalid_email,
            nicknames_dict=nicknames_dict,
            max_length=max_length,
            separator_options=separator_options,
            number_suffix_probability=number_suffix_probability,
            preserve_domain_ratio=preserve_domain_ratio,
            business_domain_ratio=business_domain_ratio,
            detailed_metrics=detailed_metrics,
            max_retries=max_retries,
            key=key,
            context_salt=context_salt,
            **kwargs,
        )

        # Prepare generator-specific configuration
        generator_params = {
            "domains": domains,
            "format": format,
            "format_ratio": format_ratio,
            "validate_source": validate_source,
            "handle_invalid_email": handle_invalid_email,
            "nicknames_dict": nicknames_dict,
            "max_length": max_length,
            "separator_options": separator_options,
            "number_suffix_probability": number_suffix_probability,
            "preserve_domain_ratio": preserve_domain_ratio,
            "business_domain_ratio": business_domain_ratio,
            "key": key,
            "context_salt": context_salt,
        }

        # Initialize generator
        email_generator = EmailGenerator(config=generator_params)

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize parent class
        super().__init__(
            field_name=field_name,
            generator=email_generator,
            generator_params=generator_params,
            **kwargs,
        )

        # Expose config fields as attributes
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Initialize internal metrics
        self.operation_name = self.__class__.__name__
        self._original_df = None
        self.error_count = 0
        self.retry_count = 0

        if self.detailed_metrics:
            self._domain_stats = Counter()
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
        Process a batch of data to generate synthetic email addresses.

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

        # Prepare a list to store generated values
        generated_values = []

        # Process each value according to null strategy
        for idx, value in enumerate(field_values):
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
                "record_id": record_id,
            }

            # Process the value
            try:
                synthetic_value = self.process_value(value, **gen_params)
                generated_values.append(synthetic_value)
            except Exception as e:
                self.logger.error(
                    f"Error generating email for value '{value}': {str(e)}"
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

                    # Try to collect domain statistics
                    if isinstance(synthetic_value, str) and "@" in synthetic_value:
                        domain = synthetic_value.split("@")[-1]
                        self._domain_stats[domain] += 1

                    # And format statistics
                    if hasattr(self.generator, "parse_email_format"):
                        try:
                            format_used = self.generator.parse_email_format(
                                synthetic_value
                            )
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
                    if (
                        not hasattr(self.generator, "prgn_generator")
                        or self.generator.prgn_generator is None
                    ):
                        # Set up PRGN generator if not already done
                        if not hasattr(self, "_prgn_generator"):
                            from pamola_core.fake_data.commons.prgn import PRNGenerator

                            self._prgn_generator = PRNGenerator(
                                global_seed=self.key or "email-generation"
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

                    # Collect domain statistics
                    if isinstance(synthetic_value, str) and "@" in synthetic_value:
                        domain = synthetic_value.split("@")[-1]
                        self._domain_stats[domain] += 1

                    # And format statistics
                    if hasattr(self.generator, "parse_email_format"):
                        try:
                            format_used = self.generator.parse_email_format(
                                synthetic_value
                            )
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
                    self.logger.debug(
                        f"Retry {retries}/{self.max_retries} generating email for value '{value}': {str(e)}"
                    )
                else:
                    self.logger.error(
                        f"Failed to generate email for value '{value}' after {self.max_retries} retries: {str(e)}"
                    )
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
        if (
            self.detailed_metrics
            and hasattr(self, "_domain_stats")
            and self._domain_stats
        ):
            top_domains = self._domain_stats.most_common(20)
            total = sum(self._domain_stats.values())
            domain_distribution = {
                domain: count / total for domain, count in top_domains
            }

            # Calculate diversity metrics
            unique_domains = len(self._domain_stats)
            diversity_ratio = unique_domains / total if total > 0 else 0

            # Add domain categorization
            domain_categories = self._categorize_domains_distribution(
                self._domain_stats
            )

            return {
                "total_emails": total,
                "unique_domains": unique_domains,
                "diversity_ratio": diversity_ratio,
                "top_domains": domain_distribution,
                "domain_categories": domain_categories,
            }

        # Otherwise extract domains from email addresses
        domains = []
        for email in df[field].dropna():
            if isinstance(email, str) and "@" in email:
                domain = email.split("@")[-1]
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
            "domain_categories": domain_categories,
        }

    def _categorize_domains_distribution(
        self, domain_counts: Counter
    ) -> Dict[str, float]:
        """
        Categorize domains into business, personal, educational, etc.

        Args:
            domain_counts: Counter with domain frequencies

        Returns:
            Dictionary with domain category distribution
        """
        categories = {"common": 0, "business": 0, "educational": 0, "others": 0}

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
                    if any(
                        business_term in domain
                        for business_term in [
                            "company",
                            "corp",
                            "enterprise",
                            "business",
                            "inc",
                            "llc",
                            "agency",
                            "consulting",
                        ]
                    ):
                        categories["business"] += count
                    elif any(
                        edu_term in domain
                        for edu_term in [
                            "edu",
                            "ac.",
                            "university",
                            "school",
                            "college",
                        ]
                    ):
                        categories["educational"] += count
                    else:
                        categories["others"] += count

        except Exception:
            # In case of error, use heuristic determination
            for domain, count in domain_counts.items():
                if any(
                    common_term in domain
                    for common_term in [
                        "gmail",
                        "yahoo",
                        "hotmail",
                        "outlook",
                        "mail",
                        "proton",
                        "aol",
                    ]
                ):
                    categories["common"] += count
                elif any(
                    business_term in domain
                    for business_term in [
                        "company",
                        "corp",
                        "enterprise",
                        "business",
                        "inc",
                        "llc",
                        "agency",
                        "consulting",
                    ]
                ):
                    categories["business"] += count
                elif any(
                    edu_term in domain
                    for edu_term in ["edu", "ac.", "university", "school", "college"]
                ):
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
        return domains[: min(10, len(domains))]

    def _collect_specific_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect operation-specific metrics for the Email Generation operation.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame

        Returns
        -------
        Dict[str, Any]
            Dictionary with metrics.
        """
        metrics_data = {
            "email_generator": {},
            "dictionary_metrics": {},
        }

        gen = self.generator

        # === 1. Email generator configuration ===
        metrics_data["email_generator"].update(
            {
                "format": getattr(gen, "format", None),
                "domains_count": len(getattr(gen, "_domain_list", [])),
                "validate_source": getattr(gen, "validate_source", True),
                "handle_invalid_email": getattr(
                    gen, "handle_invalid_email", "generate_new"
                ),
                "name_fields_used": {
                    "first_name_field": getattr(self, "first_name_field", None),
                    "last_name_field": getattr(self, "last_name_field", None),
                    "full_name_field": getattr(self, "full_name_field", None),
                    "name_format": getattr(self, "name_format", None),
                },
            }
        )

        # === 2. Domain distribution ===
        try:
            domain_metrics = self._analyze_domain_distribution(df)
            if domain_metrics:
                metrics_data["email_generator"]["domain_distribution"] = domain_metrics
        except Exception as e:
            self.logger.warning(f"Error collecting domain distribution: {str(e)}")

        # === 3. Dictionary (domain) metrics ===
        metrics_data["dictionary_metrics"] = self._collect_domain_dictionary_metrics()

        # === 4. Format distribution (if detailed metrics enabled) ===
        if (
            getattr(self, "detailed_metrics", False)
            and hasattr(self, "_format_stats")
            and self._format_stats
        ):
            try:
                total_formats = sum(self._format_stats.values())
                format_distribution = {
                    fmt: count / total_formats
                    for fmt, count in self._format_stats.most_common()
                }
                metrics_data["dictionary_metrics"][
                    "format_distribution"
                ] = format_distribution
            except Exception as e:
                self.logger.warning(f"Error calculating format distribution: {str(e)}")

        # === 5. Quality metrics (if enrich mode) ===
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

    def _collect_domain_dictionary_metrics(self) -> Dict[str, Any]:
        """
        Helper method to collect dictionary (domain-level) metrics for email generation.
        """
        dictionary_metrics = {}
        try:
            domain_list = getattr(self.generator, "_domain_list", [])
            dictionary_metrics["total_domains"] = len(domain_list)
            dictionary_metrics["popular_domains"] = self._get_popular_domains()
        except Exception as e:
            self.logger.warning(f"Error collecting domain dictionary metrics: {str(e)}")
            dictionary_metrics["error"] = str(e)
        return dictionary_metrics

    def _calculate_quality_metrics(
        self, original_series: pd.Series, generated_series: pd.Series
    ) -> Dict[str, Any]:
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
            length_similarity = 1.0 - min(
                1.0, abs(orig_mean_len - gen_mean_len) / orig_mean_len
            )
            metrics["length_similarity"] = length_similarity

        # Analyze domain preservation
        orig_domains = []
        gen_domains = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if (
                isinstance(orig, str)
                and "@" in orig
                and isinstance(gen, str)
                and "@" in gen
            ):
                orig_domains.append(orig.split("@")[-1])
                gen_domains.append(gen.split("@")[-1])

        if orig_domains and gen_domains:
            # Calculate percentage of preserved domains
            domain_preservation_count = sum(
                1 for o, g in zip(orig_domains, gen_domains) if o == g
            )
            domain_preservation_ratio = domain_preservation_count / len(orig_domains)
            metrics["domain_preservation_ratio"] = domain_preservation_ratio

            # Calculate domain diversity
            orig_domain_count = len(set(orig_domains))
            gen_domain_count = len(set(gen_domains))
            domain_diversity_ratio = (
                gen_domain_count / orig_domain_count if orig_domain_count > 0 else 0
            )
            metrics["domain_diversity_ratio"] = domain_diversity_ratio

        # Analyze local part structure
        orig_local_parts = []
        gen_local_parts = []

        for orig, gen in zip(original_series.dropna(), generated_series.dropna()):
            if (
                isinstance(orig, str)
                and "@" in orig
                and isinstance(gen, str)
                and "@" in gen
            ):
                orig_local_parts.append(orig.split("@")[0])
                gen_local_parts.append(gen.split("@")[0])

        if orig_local_parts and gen_local_parts:
            # Calculate average local part length
            orig_lp_mean_len = sum(len(lp) for lp in orig_local_parts) / len(
                orig_local_parts
            )
            gen_lp_mean_len = sum(len(lp) for lp in gen_local_parts) / len(
                gen_local_parts
            )

            # Local part length similarity
            lp_length_similarity = 1.0 - min(
                1.0, abs(orig_lp_mean_len - gen_lp_mean_len) / orig_lp_mean_len
            )
            metrics["local_part_length_similarity"] = lp_length_similarity

            # Analyze separator usage
            orig_separator_stats = Counter()
            gen_separator_stats = Counter()

            for lp in orig_local_parts:
                if "." in lp:
                    orig_separator_stats["."] += 1
                if "_" in lp:
                    orig_separator_stats["_"] += 1
                if "-" in lp:
                    orig_separator_stats["-"] += 1

            for lp in gen_local_parts:
                if "." in lp:
                    gen_separator_stats["."] += 1
                if "_" in lp:
                    gen_separator_stats["_"] += 1
                if "-" in lp:
                    gen_separator_stats["-"] += 1

            # Calculate separator usage similarity
            separator_similarity = {}
            for sep in [".", "_", "-"]:
                orig_ratio = orig_separator_stats[sep] / len(orig_local_parts)
                gen_ratio = gen_separator_stats[sep] / len(gen_local_parts)
                separator_similarity[sep] = 1.0 - min(1.0, abs(orig_ratio - gen_ratio))

            metrics["separator_similarity"] = separator_similarity

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
            "domains": self.domains,
            "format": self.format,
            "format_ratio": self.format_ratio,
            "first_name_field": self.first_name_field,
            "last_name_field": self.last_name_field,
            "full_name_field": self.full_name_field,
            "name_format": self.name_format,
            "validate_source": self.validate_source,
            "handle_invalid_email": self.handle_invalid_email,
            "nicknames_dict": self.nicknames_dict,
            "max_length": self.max_length,
            "consistency_mechanism": self.consistency_mechanism,
            "mapping_store_path": self.mapping_store_path,
            "id_field": self.id_field,
            "key": self.key,
            "context_salt": self.context_salt,
            "save_mapping": self.save_mapping,
            "column_prefix": self.column_prefix,
            "separator_options": self.separator_options,
            "number_suffix_probability": self.number_suffix_probability,
            "preserve_domain_ratio": self.preserve_domain_ratio,
            "business_domain_ratio": self.business_domain_ratio,
            "detailed_metrics": self.detailed_metrics,
            "max_retries": self.max_retries,
        }
