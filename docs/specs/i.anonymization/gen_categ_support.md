"""
Category analysis and manipulation utilities.

This module provides statistical analysis, distribution metrics,
and grouping strategies for categorical data.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def analyze_category_distribution(series: pd.Series,
                                 top_n: int = 20,
                                 min_frequency: int = 1,
                                 calculate_entropy: bool = True,
                                 calculate_gini: bool = True,
                                 calculate_concentration: bool = True) -> Dict[str, Any]:
    """
    Comprehensive analysis of category distribution.
    
    Parameters:
    -----------
    series : pd.Series
        Categorical data to analyze
    top_n : int
        Number of top categories to detail
    min_frequency : int
        Minimum frequency to include in analysis
    calculate_entropy : bool
        Whether to calculate Shannon entropy
    calculate_gini : bool
        Whether to calculate Gini coefficient
    calculate_concentration : bool
        Whether to calculate concentration metrics
        
    Returns:
    --------
    Dict[str, Any]
        - total_categories: int
        - total_records: int
        - frequency_distribution: Dict[str, int]
        - percentage_distribution: Dict[str, float]
        - top_n_categories: List[Tuple[str, int, float]]
        - rare_categories: List[str]
        - entropy: float (if calculated)
        - gini_coefficient: float (if calculated)
        - concentration_ratio_5: float (if calculated)
        - concentration_ratio_10: float (if calculated)
        - coverage_90_percentile: int (categories needed for 90% coverage)
    """
    
def identify_rare_categories(series: pd.Series,
                           count_threshold: int = 10,
                           percent_threshold: float = 0.01,
                           combined_criteria: bool = True) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    """
    Identify rare categories based on multiple criteria.
    
    Parameters:
    -----------
    series : pd.Series
        Categorical data
    count_threshold : int
        Minimum count threshold
    percent_threshold : float
        Minimum percentage threshold
    combined_criteria : bool
        Whether both criteria must be met (AND) or either (OR)
        
    Returns:
    --------
    Tuple[Set[str], Dict[str, Dict[str, Any]]]
        (rare_categories, detailed_info)
        
    Where detailed_info contains:
        - count: int
        - percentage: float
        - rank: int
    """
    
def group_rare_categories(series: pd.Series,
                         grouping_strategy: str = "single_other",
                         threshold: Union[int, float] = 10,
                         max_groups: int = 10,
                         group_prefix: str = "GROUP_",
                         preserve_top_n: Optional[int] = None,
                         similarity_threshold: float = 0.8) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Group rare categories using various strategies.
    
    Parameters:
    -----------
    series : pd.Series
        Categorical data
    grouping_strategy : str
        Strategy for grouping:
        - "single_other": All rare → "OTHER"
        - "numbered": "GROUP_001", "GROUP_002", etc.
        - "frequency_bands": Group by frequency ranges
        - "similarity": Group by string similarity
        - "alphabetical": Group by first letter/prefix
        - "length": Group by value length
    threshold : Union[int, float]
        Count or percentage threshold
    max_groups : int
        Maximum number of groups to create
    group_prefix : str
        Prefix for group names
    preserve_top_n : Optional[int]
        Number of top categories to preserve
    similarity_threshold : float
        Threshold for similarity grouping
        
    Returns:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        (grouped_series, grouping_info)
        
    Where grouping_info contains:
        - groups_created: int
        - group_mapping: Dict[str, str]
        - group_sizes: Dict[str, int]
        - reduction_ratio: float
    """
    
def calculate_category_entropy(series: pd.Series,
                             base: float = 2.0,
                             normalize: bool = True) -> float:
    """
    Calculate Shannon entropy of categorical distribution.
    
    Parameters:
    -----------
    series : pd.Series
        Categorical data
    base : float
        Logarithm base (2 for bits, e for nats)
    normalize : bool
        Whether to normalize by maximum entropy
        
    Returns:
    --------
    float
        Entropy value
    """
    
def calculate_gini_coefficient(series: pd.Series) -> float:
    """
    Calculate Gini coefficient for category distribution.
    
    Returns:
    --------
    float
        Gini coefficient [0, 1] where 0 = perfect equality
    """
    
def calculate_concentration_metrics(series: pd.Series,
                                  top_k: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate concentration ratios (CR-k).
    
    Parameters:
    -----------
    series : pd.Series
        Categorical data
    top_k : List[int]
        K values for concentration ratios
        
    Returns:
    --------
    Dict[str, float]
        Concentration ratios for each k
    """
    
def find_category_clusters(categories: List[str],
                         similarity_threshold: float = 0.8,
                         min_cluster_size: int = 2,
                         max_clusters: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Find clusters of similar categories.
    
    Parameters:
    -----------
    categories : List[str]
        Categories to cluster
    similarity_threshold : float
        Minimum similarity for clustering
    min_cluster_size : int
        Minimum cluster size
    max_clusters : Optional[int]
        Maximum number of clusters
        
    Returns:
    --------
    Dict[str, List[str]]
        Cluster ID to categories mapping
    """
    
def suggest_grouping_strategy(series: pd.Series,
                            target_categories: Optional[int] = None,
                            min_group_size: int = 10) -> Dict[str, Any]:
    """
    Suggest optimal grouping strategy based on distribution.
    
    Returns:
    --------
    Dict[str, Any]
        - recommended_strategy: str
        - reasoning: List[str]
        - expected_groups: int
        - expected_reduction: float
        - parameters: Dict[str, Any]
    """
    
def create_frequency_bands(series: pd.Series,
                         n_bands: int = 5,
                         method: str = "equal_frequency") -> Dict[str, str]:
    """
    Create frequency-based grouping bands.
    
    Parameters:
    -----------
    series : pd.Series
        Categorical data
    n_bands : int
        Number of frequency bands
    method : str
        Banding method:
        - "equal_frequency": Equal number of categories per band
        - "equal_width": Equal frequency range per band
        - "logarithmic": Logarithmic frequency bands
        - "custom": Custom band boundaries
        
    Returns:
    --------
    Dict[str, str]
        Category to band mapping
    """
    
def validate_category_mapping(original: pd.Series,
                            mapped: pd.Series,
                            mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate category mapping consistency.
    
    Returns:
    --------
    Dict[str, Any]
        - is_valid: bool
        - unmapped_categories: List[str]
        - inconsistent_mappings: List[Tuple[str, str, str]]
        - coverage: float
        - reduction_ratio: float
    """
    
def calculate_semantic_diversity(categories: List[str],
                               method: str = "token_overlap") -> float:
    """
    Calculate semantic diversity of categories.
    
    Parameters:
    -----------
    categories : List[str]
        Category list
    method : str
        Diversity calculation method:
        - "token_overlap": Based on shared tokens
        - "edit_distance": Based on string similarity
        - "length_variance": Based on length distribution
        
    Returns:
    --------
    float
        Diversity score [0, 1]
    """