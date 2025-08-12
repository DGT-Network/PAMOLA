"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor` for analyzing and validating email
addresses, including format validation, domain analysis, and public vs. business address classification.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Format Analysis Profiling Operations 
--------------------------------   
It includes the following capabilities:  
- Format compliance statistics
- Breakdown by domain
- Classification of public vs. business emails
- Domain whitelist/blacklist compliance
- Invalid email analysis
- Top domains statistics
- Missing/null email statistics
- Domain collection for further analysis

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import dns.resolver

from pamola_core.common.regex.patterns import Patterns
from pamola_core.profiling.format.base import BaseDataFormatProfilingProcessor


class EmailFormatProfilingProcessor(BaseDataFormatProfilingProcessor):
    """
    Processor for analyzing and validating email addresses, including format validation, domain
    analysis, and public vs. business address classification.
    """
    
    def __init__(
        self,
        domain_whitelist: Optional[List[str]] = None, 
        domain_blacklist: Optional[List[str]] = None,
        public_domains: Optional[List[str]] = None,
        collect_domains: bool = True, 
        validate_mx_records: bool = False, 
        allow_nulls: bool = True,
        save_invalid_emails: bool = False,
        max_invalid_examples: int = 100,
    ):
        """
        Initializes the Email Format Profiling Processor.

        Parameters:
        -----------
        domain_whitelist : List[str], optional
            List of explicitly allowed domains (default=[]).
        domain_blacklist : List[str], optional
            List of explicitly forbidden domains (default=[]).
        public_domains : List[str], optional
            List of public email domains or path to file (default=None).
        collect_domains : bool, optional
            Whether to collect all domains (default=True).
        validate_mx_records : bool, optional
            Whether to validate domain MX record (default=False).
        allow_nulls: bool, optional
            Whether null/empty values are allowed (default=True).
        save_invalid_emails: bool, optional
            Whether to save invalid emails to a file (default=False).
        max_invalid_examples: int, optional
            Maximum invalid examples to include in report (default=100).

        """
        super().__init__()
        self.domain_whitelist = domain_whitelist
        self.domain_blacklist = domain_blacklist
        self.public_domains = public_domains
        self.collect_domains = collect_domains
        self.validate_mx_records = validate_mx_records
        self.allow_nulls = allow_nulls
        self.save_invalid_emails = save_invalid_emails
        self.max_invalid_examples = max_invalid_examples

    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform categorical pattern analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all categorical columns will be selected.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - `domain_whitelist` (List[str], default=self.domain_whitelist):
                List of explicitly allowed domains.
            - `domain_blacklist` (List[str], default=self.domain_blacklist):
                List of explicitly forbidden domains.
            - `public_domains` (List[str], default=self.public_domains):
                List of public email domains or path to file.
            - `collect_domains` (bool, default=self.collect_domains):
                Whether to collect all domains.
            - `validate_mx_records` (bool, default=self.validate_mx_records):
                Whether to validate domain MX record.
            - `allow_nulls` (bool, default=self.allow_nulls):
                Whether null/empty values are allowed.
            - `save_invalid_emails` (bool, default=self.save_invalid_emails):
                Whether to save invalid emails to a file.
            - `max_invalid_examples` (int, default=self.max_invalid_examples):
                Maximum invalid examples to include in report.
        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their pattern analysis results.
        """

        domain_whitelist = kwargs.get("domain_whitelist", self.domain_whitelist)
        domain_blacklist = kwargs.get("domain_blacklist", self.domain_blacklist)
        public_domains = kwargs.get("public_domains", self.public_domains)
        collect_domains = kwargs.get("collect_domains", self.collect_domains)
        validate_mx_records = kwargs.get("validate_mx_records", self.validate_mx_records)
        allow_nulls = kwargs.get("allow_nulls", self.allow_nulls)
        save_invalid_emails = kwargs.get("save_invalid_emails", self.save_invalid_emails)
        max_invalid_examples = kwargs.get("max_invalid_examples", self.max_invalid_examples)

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        results = {}

        for col in columns:
            series = df[col]

            if not allow_nulls:
                series = series.dropna()

            email_data = series.dropna().astype(str)
            match_mask = email_data.str.match(Patterns.EMAIL_REGEX, na=False)
            valid_emails = email_data[match_mask]
            invalid_emails = email_data[~match_mask]

            # Domain extraction
            domains = valid_emails.str.extract(Patterns.DOMAIN_REGEX)[0].dropna()
            domain_counts = domains.value_counts().to_dict()

            # Whitelist/Blacklist compliance
            whitelisted = domains.isin(domain_whitelist) if domain_whitelist else pd.Series([False] * len(domains))
            blacklisted = domains.isin(domain_blacklist) if domain_blacklist else pd.Series([False] * len(domains))
            public_emails = domains.apply(lambda d: d in public_domains)
            business_emails = ~public_emails

            collected_domains = list(domain_counts.keys()) if collect_domains else None

            mx_validity = None
            if validate_mx_records:
                mx_validity = {domain: self._is_validate_mx(domain) for domain in domain_counts.keys()}

            top_domains = list(domain_counts.items())

            invalid_examples = invalid_emails.head(max_invalid_examples).tolist()

            results[col] = {
                "total": len(email_data),
                "valid_count": len(valid_emails),
                "invalid_count": len(invalid_emails),
                "invalid_examples": invalid_examples,
                "missing_count": series.isna().sum(),
                "top_domains": top_domains,
                "whitelisted_count": whitelisted.sum(),
                "blacklisted_count": blacklisted.sum(),
                "public_email_count": public_emails.sum(),
                "business_email_count": business_emails.sum(),
                "collected_domains": collected_domains,
                "mx_validity": mx_validity,
            }

        return results
    
    def _is_validate_mx(self, domain: str) -> bool:
        """
        Check if a domain has a valid MX record.

        Parameters:
        -----------
        domain : str
            The domain name to check for MX (Mail Exchange) records.

        Returns:
        --------
        bool
            True if the domain has at least one valid MX record, False otherwise.
        """
        try:
            answers = dns.resolver.resolve(domain, 'MX')
            return len(answers) > 0
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.Timeout):
            return False