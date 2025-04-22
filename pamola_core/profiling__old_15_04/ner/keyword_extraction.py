"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor` that performs
detailed grouping using a tokenization-based approach to extract key terms from
semi-structured text, with support for synonyms, stop words, and frequency analysis.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

Licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Named Entity Recognition Profiling Operations
--------------------------------   
It includes the following capabilities:
- Extracted keywords with frequencies
- Token distribution statistics
- Synonyms replacement tracking
- Stop words filtering statistics
- Top keywords list
- Token co-occurrence patterns
- Visualization of keyword distribution

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from itertools import combinations
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import pandas as pd
from scipy.stats import entropy
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from pamola_core.profiling.base import BaseProfilingProcessor

# Configure logging
logger = logging.getLogger(__name__)


class KeywordExtractionProfilingProcessor(BaseProfilingProcessor):
    """
    Processor that performs detailed grouping using a tokenization-based approach
    to extract key terms from semi-structured text, with support for synonyms,
    stop words, and frequency analysis.
    """
    
    def __init__(
        self,
        source_field: str,
        stop_words_file: str = "",
        synonyms_file: str = "",
        synonyms_format: List[str] = ["main_key", "synonym_list", "alias"],
        min_frequency_threshold: float = 0.01,
        max_frequency_threshold: float = 0.9,
        token_separators: List[str] = [" ", "(", ")", "-", "/", ":"], 
        top_n: int = 30, 
        ignore_case: bool = True,
        export_tokens: bool = True,
    ):
        """
        Initializes the Keyword Extraction Profiling Processor.

        Parameters:
        -----------
        source_field : str, required
            Field to analyze.
        stop_words_file : str, optional
            Path to file with stop words (default="").
        synonyms_file : str, optional
            Path to CSV with synonyms (default="")
        synonyms_format : List[str], optional
            Format of synonyms CSV (default=["main_key", "synonym_list", "alias"]).
        min_frequency_threshold : float, optional
            Minimum frequency ratio to keep tokens (default=0.01).
        max_frequency_threshold : float, optional
            Maximum frequency ratio to keep tokens (default=0.9).
        token_separators : List[str], optional
            Characters used for tokenization (default=[" ", "(", ")", "-", "/", ":"]).
        top_n : int, optional
            Number of top tokens to report (default=30).
        ignore_case: bool, optional
            Whether to ignore case differences (default=True).
        export_tokens: bool, optional
            Whether to export top tokens to file (default=True).
        """
        super().__init__()
        self.source_field = source_field
        self.stop_words_file = stop_words_file
        self.synonyms_file = synonyms_file
        self.synonyms_format = synonyms_format
        self.min_frequency_threshold = min_frequency_threshold
        self.max_frequency_threshold = max_frequency_threshold
        self.token_separators = token_separators
        self.top_n = top_n
        self.ignore_case = ignore_case
        self.export_tokens = export_tokens
    
    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform keyword/token extraction and profiling on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all categorical columns will be selected.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - `source_field` (str, default=self.source_field):
                Field to analyze.
            - `stop_words_file` (str, default=self.stop_words_file):
                Path to file with stop words.
            - `synonyms_file` (str, default=self.synonyms_file):
                Path to CSV with synonyms.
            - `synonyms_format` (str, default=self.synonyms_format):
                Format of synonyms CSV.
            - `min_frequency_threshold` (float, default=self.min_frequency_threshold):
                Minimum frequency ratio to keep tokens.
            - `max_frequency_threshold` (float, default=self.max_frequency_threshold):
                Maximum frequency ratio to keep token.
            - `token_separators` (List[str], default=self.token_separators):
                Characters used for tokenization.
            - `top_n` (int, default=self.top_n):
                Number of top tokens to report.
            - `ignore_case` (bool, default=self.ignore_case):
                Whether to ignore case differences.
            - `export_tokens` (bool, default=self.export_tokens):
                Whether to aggregate similar entities.
        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their pattern analysis results.
        """
        source_field = kwargs.get("source_field", self.source_field)
        stop_words_file = kwargs.get("stop_words_file", self.stop_words_file)
        synonyms_file = kwargs.get("synonyms_file", self.synonyms_file)
        synonyms_format = kwargs.get("synonyms_format", self.synonyms_format)
        min_frequency_threshold = kwargs.get("min_frequency_threshold", self.min_frequency_threshold)
        max_frequency_threshold = kwargs.get("max_frequency_threshold", self.max_frequency_threshold)
        token_separators = kwargs.get("token_separators", self.token_separators)
        top_n = kwargs.get("top_n", self.top_n)
        ignore_case = kwargs.get("ignore_case", self.ignore_case)
        export_tokens = kwargs.get("export_tokens", self.export_tokens)

        stop_words = self._load_stop_words(stop_words_file)
        synonym_map = self._load_synonyms(synonyms_file, synonyms_format)

        result = {}
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            series = df[col].dropna().astype(str)

            result[col] = self._process_column(
                series=series,
                stop_words=stop_words,
                synonyms=synonym_map,
                separators=token_separators,
                ignore_case=ignore_case,
                min_freq=min_frequency_threshold,
                max_freq=max_frequency_threshold,
                top_n=top_n,
                export=export_tokens,
                column_name=source_field,
            )
        
        return result
    
    def _process_column(
        self,
        series: pd.Series,
        stop_words: set,
        synonyms: Dict[str, str],
        separators: List[str],
        ignore_case: bool,
        min_freq: float,
        max_freq: float,
        top_n: int,
        export: bool,
        column_name: str,
    ) -> Dict[str, Any]:
        """
        Processes a single column for token extraction and profiling.

        Parameters:
        -----------
        series : pd.Series
            The input text series.
        stop_words : set
            A set of stop words to filter out.
        synonyms : Dict[str, str]
            A dictionary mapping synonyms to their main keys.
        separators : List[str]
            A list of separators for tokenization.
        ignore_case : bool
            Whether to ignore case during tokenization.
        min_freq : float
            Minimum frequency ratio to keep tokens.
        max_freq : float
            Maximum frequency ratio to keep tokens.
        top_n : int
            Number of top tokens to report.
        export : bool
            Whether to export the results to a CSV file.
        column_name : str
            The name of the column being processed.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the profiling results.
        """
        token_series, synonym_tracking, total_tokens = self._tokenize_and_count(
            series, stop_words, synonyms, separators, ignore_case
        )

        filtered_tokens = self._filter_tokens(token_series, total_tokens, min_freq, max_freq)
        top_tokens = filtered_tokens.head(top_n).to_dict()
        distribution = self._calculate_distribution(token_series, total_tokens, filtered_tokens)
        
        # Calculate token co-occurrence patterns
        cooccurrence_patterns = self._calculate_cooccurrence(series, separators, ignore_case)
        result = {
            "top_keywords": top_tokens,
            "token_distribution": distribution,
            "synonym_replacements": {k: list(set(v)) for k, v in synonym_tracking.items()},
            "cooccurrence_patterns": cooccurrence_patterns,
        }

        if export:
            output_file = Path(f"top_tokens_{column_name}.csv")
            pd.DataFrame.from_dict(top_tokens, orient="index", columns=["count"]).to_csv(output_file)
            result["exported_file"] = str(output_file)

        self._visualize_keyword_distribution(top_tokens, column_name, export)

        return result

    def _tokenize_and_count(
        self,
        series: pd.Series,
        stop_words: set,
        synonyms: Dict[str, str],
        separators: List[str],
        ignore_case: bool,
    ) -> Tuple[pd.Series, defaultdict, int]:
        """
        Tokenizes the text, applies stop word filtering, and counts token frequencies.

        Parameters:
        -----------
        series : pd.Series
            The input text series.
        stop_words : set
            A set of stop words to filter out.
        synonyms : Dict[str, str]
            A dictionary mapping synonyms to their main keys.
        separators : List[str]
            A list of separators for tokenization.
        ignore_case : bool
            Whether to ignore case during tokenization.

        Returns:
        --------
        Tuple[pd.Series, defaultdict, int]
            - A pandas Series containing token counts.
            - A defaultdict tracking synonym replacements.
            - Total number of tokens.
        """
        synonym_tracking = defaultdict(list)

        # Tokenize the series using pandas' string methods
        pattern = '|'.join(map(re.escape, separators))
        if ignore_case:
            series = series.str.lower()
        tokens = series.str.split(pattern, expand=False).explode()

        # Filter out stop words and apply synonyms
        tokens = tokens[~tokens.isin(stop_words)]
        tokens = tokens.dropna()

        # Apply synonym mapping
        def map_synonyms(token):
            if token in synonyms:
                synonym_tracking[synonyms[token]].append(token)
                return synonyms[token]
            return token

        tokens = tokens.map(map_synonyms)

        # Count token frequencies
        token_counts = tokens.value_counts()
        total_tokens = token_counts.sum()

        return token_counts, synonym_tracking, total_tokens

    def _filter_tokens(
        self, token_series: pd.Series, total_tokens: int, min_freq: float, max_freq: float
    ) -> pd.Series:
        """
        Filters tokens based on frequency thresholds.

        Parameters:
        -----------
        token_series : pd.Series
            A pandas Series containing token counts.
        total_tokens : int
            Total number of tokens.
        min_freq : float
            Minimum frequency ratio to keep tokens.
        max_freq : float
            Maximum frequency ratio to keep tokens.

        Returns:
        --------
        pd.Series
            A Series containing filtered tokens and their counts.
        """
        # Calculate relative frequencies
        relative_frequencies = token_series / total_tokens

        # Filter tokens based on frequency thresholds
        filtered_tokens = token_series[
            (relative_frequencies >= min_freq) & (relative_frequencies <= max_freq)
        ]

        return filtered_tokens

    def _calculate_distribution(
        self, token_series: pd.Series, total_tokens: int, filtered_tokens: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculates token distribution statistics.

        Parameters:
        -----------
        token_series : pd.Series
            A pandas Series containing all tokens.
        total_tokens : int
            Total number of tokens.
        filtered_tokens : pd.Series
            A Series containing filtered tokens and their counts.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing token distribution statistics.
        """
        # Calculate relative frequencies for all tokens
        relative_frequencies = token_series / total_tokens

        # Use numpy for entropy calculation
        token_entropy = entropy(relative_frequencies.values) if total_tokens > 0 else 0.0

        return {
            "total_documents": len(token_series),
            "total_tokens": total_tokens,
            "unique_tokens": len(filtered_tokens),
            "entropy": token_entropy,
        }

    def _load_stop_words(self, file_path: str) -> set:
        if not file_path or not Path(file_path).exists():
            return set()
        with open(file_path, "r", encoding="utf-8") as f:
            return set(word.strip().lower() for word in f if word.strip())

    def _load_synonyms(self, file_path: str, synonyms_format: List[str]) -> Dict[str, str]:
        """
        Load synonyms from a CSV file and return a mapping from synonym to main keyword.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing synonyms.
        synonyms_format : List[str]
            Format of the CSV file. Must contain at least two elements:
            [main_key_column, synonym_list_column].

        Returns:
        --------
        Dict[str, str]
            A dictionary mapping each synonym to its main keyword.
        """
        if not file_path or not Path(file_path).exists():
            logger.warning(f"Synonyms file '{file_path}' does not exist.")
            return {}

        if len(synonyms_format) < 2:
            raise ValueError("synonyms_format must have at least 2 elements: [main_key_column, synonym_list_column]")

        main_key_col, synonym_list_col = synonyms_format[:2]
        synonym_map = {}

        try:
            # Load the CSV file using pandas
            df = pd.read_csv(file_path, encoding="utf-8")

            # Validate required columns
            if main_key_col not in df.columns or synonym_list_col not in df.columns:
                raise ValueError(f"CSV file must contain columns '{main_key_col}' and '{synonym_list_col}'.")

            # Process each row to build the synonym map
            for _, row in df.iterrows():
                main_key = str(row[main_key_col]).strip().lower()
                synonyms = row[synonym_list_col]

                if not main_key or pd.isna(synonyms):
                    continue

                # Split synonyms by delimiter and map them to the main key
                for synonym in str(synonyms).split(";"):
                    synonym_clean = synonym.strip().lower()
                    if synonym_clean:
                        synonym_map[synonym_clean] = main_key

                # Map the main key to itself
                synonym_map[main_key] = main_key

        except pd.errors.EmptyDataError:
            logger.error(f"Synonyms file '{file_path}' is empty or invalid.")
        except Exception as e:
            logger.error(f"Error loading synonyms file '{file_path}': {e}")

        return synonym_map

    def _tokenize(self, text: str, separators: List[str], ignore_case: bool = True) -> List[str]:
        """
        Tokenizes a given text into a list of tokens based on specified separators.

        Parameters:
        -----------
        text : str
            The input text to tokenize.
        separators : List[str]
            A list of characters or strings used as delimiters for tokenization.
        ignore_case : bool, optional
            Whether to convert the text to lowercase before tokenization (default=True).

        Returns:
        --------
        List[str]
            A list of tokens extracted from the input text.
        """
        # Convert text to lowercase if ignore_case is True
        if ignore_case:
            text = text.lower()

        # Create a regex pattern from the list of separators
        pattern = '|'.join(map(re.escape, separators))

        # Split the text into tokens using the regex pattern
        tokens = re.split(pattern, text)

        # Remove empty tokens and strip whitespace from each token
        return [t.strip() for t in tokens if t.strip()]
    

    def _calculate_cooccurrence(self, series: pd.Series, separators: List[str], ignore_case: bool) -> Dict[str, int]:
        """
        Calculates token co-occurrence patterns.

        Parameters:
        -----------
        series : pd.Series
            A pandas Series containing text data.
        separators : List[str]
            A list of separators for tokenization.
        ignore_case : bool
            Whether to ignore case during tokenization.

        Returns:
        --------
        Dict[str, int]
            A dictionary where keys are token pairs (as strings) and values are their co-occurrence counts.
        """
        cooccurrence_counter = Counter()

        # Tokenize each document and calculate co-occurrences
        for text in series.dropna():
            tokens = self._tokenize(text, separators, ignore_case)
            unique_tokens = set(tokens)  # Use unique tokens per document
            for token_pair in combinations(sorted(unique_tokens), 2):  # Generate all unique pairs
                cooccurrence_counter[token_pair] += 1

        # Convert tuple keys to strings
        cooccurrence_patterns = {f"{pair[0]}|{pair[1]}": count for pair, count in cooccurrence_counter.items()}
        return cooccurrence_patterns
    
    def _visualize_keyword_distribution(self, top_keywords: Dict[str, int], column_name: str, export: bool = True) -> None:
        """
        Visualizes the keyword distribution as a bar chart.

        Parameters:
        -----------
        top_keywords : Dict[str, int]
            A dictionary of top keywords and their frequencies.
        column_name : str
            The name of the column being processed.
        export : bool, optional
            Whether to save the visualization as a file (default=True).
        """
        if not top_keywords:
            logger.warning(f"No keywords to visualize for column '{column_name}'.")
            return

        # Prepare data for visualization
        keywords, frequencies = zip(*top_keywords.items())
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(frequencies), y=list(keywords), palette="viridis")
        plt.xlabel("Frequency")
        plt.ylabel("Keywords")
        plt.title(f"Keyword Distribution for Column: {column_name}")
        plt.tight_layout()

        # Export visualization if required
        if export:
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"keyword_distribution_{column_name}.png"
            plt.savefig(output_file, dpi=300)
            logger.info(f"Keyword distribution saved to {output_file.resolve()}")
