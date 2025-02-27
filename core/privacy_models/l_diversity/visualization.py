"""
PAMOLA.CORE - L-Diversity Visualization Module

Provides visualization capabilities for l-diversity
anonymization techniques with integration with caching system.

Key Features:
- Diversity distribution plots using cached data
- Risk visualization integrated with privacy assessment
- Attribute distribution visualization
- Cache-aware operation to avoid redundant calculations

This module works seamlessly with the L-Diversity processor's
centralized cache mechanism for efficient visualization generation.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import file saving utility
from core.utils.file_io import save_plot

# Configure logging
logger = logging.getLogger(__name__)


class LDiversityVisualizer:
    """
    Generates visualizations for l-diversity analysis with caching support

    This class integrates with the L-Diversity processor to utilize
    its cached calculations for efficient visualization generation.
    """

    def __init__(self, processor=None):
        """
        Initialize visualizer with processor for cache access

        Parameters:
        -----------
        processor : object, optional
            L-Diversity processor instance for cached calculations
        """
        self.processor = processor
        self.logger = logging.getLogger(__name__)

        # Default visualization configuration
        self.config = {
            'figsize': (10, 6),
            'style': 'whitegrid',
            'palette': 'viridis',
            'save_format': 'png',
            'dpi': 300
        }

    def visualize_l_distribution(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            diversity_type: Optional[str] = None,
            l_threshold: Optional[int] = None,
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Visualize l-diversity distribution using cached calculations if available

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        diversity_type : str, optional
            Type of diversity to visualize (uses processor's type if None)
        l_threshold : int, optional
            L-threshold to show in visualization (uses processor's value if None)
        save_path : str or Path, optional
            Path to save visualization
        **kwargs : dict
            Additional visualization parameters

        Returns:
        --------
        Tuple[plt.Figure, Optional[str]]
            Figure and optional saved path
        """
        try:
            # Get configuration by merging defaults with kwargs
            config = {**self.config, **kwargs}

            # Use processor's diversity type if not specified
            if diversity_type is None and self.processor:
                diversity_type = getattr(self.processor, 'diversity_type', 'distinct')
            elif diversity_type is None:
                diversity_type = 'distinct'

            # Use processor's l-threshold if not specified
            if l_threshold is None and self.processor:
                l_threshold = getattr(self.processor, 'l', 3)
            elif l_threshold is None:
                l_threshold = 3

            # Get diversity data from processor's cache if available
            group_diversity = None
            if self.processor:
                try:
                    # Use the processor's cache directly
                    cache_key = (
                        tuple(quasi_identifiers),
                        tuple(sensitive_attributes),
                        diversity_type
                    )

                    if hasattr(self.processor, '_results_cache') and cache_key in self.processor._results_cache:
                        group_diversity = self.processor._results_cache[cache_key]
                    else:
                        # Calculate diversity if not in cache
                        group_diversity = self.processor.calculate_group_diversity(
                            data, quasi_identifiers, sensitive_attributes
                        )
                except Exception as e:
                    self.logger.warning(f"Error accessing processor's cache: {e}")
                    group_diversity = None

            # If we couldn't get data from processor's cache, calculate directly
            if group_diversity is None:
                self.logger.info("Cache not available, calculating diversity directly")
                group_diversity = self._calculate_group_diversity(
                    data, quasi_identifiers, sensitive_attributes, diversity_type
                )

            # Prepare l-values based on diversity type
            l_values = []
            for _, row in group_diversity.iterrows():
                for sa in sensitive_attributes:
                    if diversity_type == 'entropy':
                        col_name = f"{sa}_entropy"
                        if col_name in row:
                            # For entropy, convert to effective l
                            entropy = row[col_name]
                            effective_l = np.exp(entropy) if entropy > 0 else 1
                            l_values.append(effective_l)
                    else:
                        # For distinct and recursive
                        col_name = f"{sa}_distinct"
                        if col_name in row:
                            l_values.append(row[col_name])

            # Create figure
            plt.figure(figsize=config['figsize'])
            plt.style.use(config['style'])

            # Create distribution plot
            ax = sns.histplot(
                l_values,
                kde=True,
                color=config.get('color', 'blue'),
                bins=config.get('bins', 10)
            )

            # Set title and labels based on diversity type
            diversity_name = diversity_type.capitalize()
            plt.title(f'Distribution of {diversity_name} L-Diversity Values')
            plt.xlabel('L-Value')
            plt.ylabel('Frequency')

            # Add reference line for l threshold
            plt.axvline(
                x=l_threshold,
                color='red',
                linestyle='--',
                label=f'L-Threshold ({l_threshold})'
            )
            plt.legend()

            # Add summary statistics as text
            if l_values:
                min_l = min(l_values)
                max_l = max(l_values)
                mean_l = np.mean(l_values)
                median_l = np.median(l_values)

                stats_text = (
                    f"Min: {min_l:.2f}\n"
                    f"Max: {max_l:.2f}\n"
                    f"Mean: {mean_l:.2f}\n"
                    f"Median: {median_l:.2f}"
                )

                # Add text in top right corner
                plt.annotate(
                    stats_text,
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    ha='right',
                    va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )

            # Get figure and save if path provided
            fig = plt.gcf()
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"l_diversity_distribution_{diversity_type}_{timestamp}.{config['save_format']}"
                    save_path = save_path / filename

                # Use centralized save_plot utility
                saved_path = save_plot(
                    fig,
                    save_path,
                    save_format=config['save_format'],
                    dpi=config['dpi'],
                    bbox_inches="tight"
                )

                if saved_path:
                    self.logger.info(f"L-distribution plot saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save L-distribution plot")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during l-diversity visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def visualize_attribute_distribution(
            self,
            data: pd.DataFrame,
            sensitive_attribute: str,
            quasi_identifiers: Optional[List[str]] = None,
            top_n: int = 10,
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Visualize distribution of sensitive attribute values

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        sensitive_attribute : str
            Sensitive attribute column to visualize
        quasi_identifiers : List[str], optional
            Columns to group by (if None, shows overall distribution)
        top_n : int, optional
            Number of top groups to display
        save_path : str or Path, optional
            Path to save visualization
        **kwargs : dict
            Additional visualization parameters

        Returns:
        --------
        Tuple[plt.Figure, Optional[str]]
            Figure and optional saved path
        """
        try:
            # Get configuration by merging defaults with kwargs
            config = {**self.config, **kwargs}

            # Create figure
            plt.figure(figsize=config.get('figsize', (12, 8)))
            plt.style.use(config.get('style', 'whitegrid'))

            # If quasi-identifiers provided, show distribution by groups
            if quasi_identifiers:
                # Check if we can reuse cached group data
                grouped_data = None
                if self.processor:
                    try:
                        # Try to access grouped data from processor
                        cache_key = (tuple(quasi_identifiers), (sensitive_attribute,), 'distinct')
                        if hasattr(self.processor, '_results_cache') and cache_key in self.processor._results_cache:
                            # We have cached group diversity, but need original groups
                            pass  # Can't directly use cache for this visualization
                    except Exception as e:
                        self.logger.warning(f"Error accessing processor's cache: {e}")

                # Group data (can't easily use cache for this)
                # Create a composite group key for display
                if len(quasi_identifiers) > 1:
                    data = data.copy()
                    data['_group_key'] = data[quasi_identifiers].apply(
                        lambda row: ' | '.join(str(val) for val in row),
                        axis=1
                    )
                    group_col = '_group_key'
                else:
                    group_col = quasi_identifiers[0]

                # Get top N groups by size
                top_groups = data[group_col].value_counts().nlargest(top_n).index.tolist()
                filtered_data = data[data[group_col].isin(top_groups)]

                # Check data type of sensitive attribute for appropriate plot
                if pd.api.types.is_numeric_dtype(data[sensitive_attribute]):
                    # For numeric data, use boxplot
                    ax = sns.boxplot(
                        x=group_col,
                        y=sensitive_attribute,
                        data=filtered_data,
                        palette=config.get('palette', 'viridis')
                    )
                    plt.title(f'Distribution of {sensitive_attribute} by {", ".join(quasi_identifiers)}')
                    plt.xlabel('Quasi-identifier Groups')
                    plt.ylabel(sensitive_attribute)
                    plt.xticks(rotation=45, ha='right')
                else:
                    # For categorical data, use heatmap of counts
                    # Create cross-tabulation
                    cross_tab = pd.crosstab(
                        filtered_data[group_col],
                        filtered_data[sensitive_attribute],
                        normalize='index'
                    )

                    # Plot heatmap
                    ax = sns.heatmap(
                        cross_tab,
                        annot=True,
                        cmap=config.get('cmap', 'YlGnBu'),
                        fmt='.2f',
                        cbar_kws={'label': 'Proportion'}
                    )
                    plt.title(f'Distribution of {sensitive_attribute} by {", ".join(quasi_identifiers)}')
                    plt.tight_layout()
            else:
                # Show overall distribution
                if pd.api.types.is_numeric_dtype(data[sensitive_attribute]):
                    # For numeric data, use histogram
                    ax = sns.histplot(
                        data[sensitive_attribute],
                        kde=True,
                        color=config.get('color', 'green')
                    )
                    plt.title(f'Distribution of {sensitive_attribute}')
                    plt.xlabel(sensitive_attribute)
                    plt.ylabel('Frequency')
                else:
                    # For categorical data, use bar plot
                    value_counts = data[sensitive_attribute].value_counts().nlargest(top_n)
                    ax = sns.barplot(
                        x=value_counts.index,
                        y=value_counts.values,
                        palette=config.get('palette', 'viridis')
                    )
                    plt.title(f'Distribution of {sensitive_attribute}')
                    plt.xlabel(sensitive_attribute)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')

            # Get figure and save if path provided
            fig = plt.gcf()
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    group_suffix = '_grouped' if quasi_identifiers else ''
                    filename = f"{sensitive_attribute}_distribution{group_suffix}_{timestamp}.{config['save_format']}"
                    save_path = save_path / filename

                # Use centralized save_plot utility
                saved_path = save_plot(
                    fig,
                    save_path,
                    save_format=config['save_format'],
                    dpi=config['dpi'],
                    bbox_inches="tight"
                )

                if saved_path:
                    self.logger.info(f"Attribute distribution plot saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save attribute distribution plot")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during attribute distribution visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def visualize_risk_heatmap(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Create risk heatmap visualization using cached risk assessments

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Quasi-identifier columns
        sensitive_attributes : List[str]
            Sensitive attribute columns
        save_path : str or Path, optional
            Path to save visualization
        **kwargs : dict
            Additional visualization parameters

        Returns:
        --------
        Tuple[plt.Figure, Optional[str]]
            Figure and optional saved path
        """
        try:
            # Get configuration by merging defaults with kwargs
            config = {**self.config, **kwargs}

            # Try to use privacy risk assessor from the processor
            risk_metrics = None
            if self.processor:
                # If processor has evaluate_privacy method, use it
                if hasattr(self.processor, 'evaluate_privacy'):
                    try:
                        # Get risk metrics from processor
                        risk_metrics = self.processor.evaluate_privacy(
                            data, quasi_identifiers, sensitive_attributes,
                            **kwargs.get('risk_kwargs', {})
                        )
                    except Exception as e:
                        self.logger.warning(f"Error using processor's privacy evaluation: {e}")

                # Alternative: check if processor has a risk_assessor attribute
                elif hasattr(self.processor, 'risk_assessor'):
                    try:
                        risk_assessor = self.processor.risk_assessor
                        risk_metrics = risk_assessor.assess_privacy_risks(
                            data, quasi_identifiers, sensitive_attributes,
                            **kwargs.get('risk_kwargs', {})
                        )
                    except Exception as e:
                        self.logger.warning(f"Error using processor's risk assessor: {e}")

            # If no risk metrics available or error occurred, create a basic risk matrix
            if risk_metrics is None:
                self.logger.info("No risk assessor available, calculating basic risk matrix")

                # Create a basic risk matrix
                risk_matrix = pd.DataFrame(index=quasi_identifiers, columns=sensitive_attributes)

                # Identify high-risk groups (basic k=2 criteria)
                high_risk_groups = data.groupby(quasi_identifiers).filter(
                    lambda x: len(x) < kwargs.get('k_threshold', 2)
                )

                # Fill risk matrix with simple risk metrics
                for qi in quasi_identifiers:
                    for sa in sensitive_attributes:
                        # Calculate risk as proportion of high-risk groups for this QI-SA pair
                        if not high_risk_groups.empty:
                            qi_values = high_risk_groups[qi].unique()
                            risk = len(qi_values) / len(data[qi].unique()) * 100 if len(data[qi].unique()) > 0 else 0
                        else:
                            risk = 0
                        risk_matrix.loc[qi, sa] = risk
            else:
                # Use risk metrics to create a more accurate risk matrix
                risk_matrix = pd.DataFrame(index=quasi_identifiers, columns=sensitive_attributes)

                # Extract attribute risks from metrics
                attribute_risks = risk_metrics.get('attribute_risks', {})

                # Fill the risk matrix using attribute risks
                for qi in quasi_identifiers:
                    for sa in sensitive_attributes:
                        if sa in attribute_risks:
                            # Get QI-specific risk if available
                            sa_risk = attribute_risks[sa]
                            # Try to get QI-specific risk
                            qi_risk = sa_risk.get('group_metrics', {}).get(qi, {}).get('risk', 0)
                            if qi_risk:
                                risk_matrix.loc[qi, sa] = qi_risk
                            else:
                                # Fall back to overall risk
                                risk_matrix.loc[qi, sa] = sa_risk.get('risk_percentage', 0)
                        else:
                            risk_matrix.loc[qi, sa] = 0

            # Create figure for heatmap
            plt.figure(figsize=config.get('figsize', (10, 8)))

            # Create heatmap
            ax = sns.heatmap(
                risk_matrix,
                annot=True,
                cmap=config.get('cmap', 'YlOrRd'),
                fmt='.2f',
                cbar_kws={'label': 'Risk Percentage'}
            )

            plt.title('Privacy Risk Heatmap')
            plt.xlabel('Sensitive Attributes')
            plt.ylabel('Quasi-Identifiers')

            # Get figure and save if path provided
            fig = plt.gcf()
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"privacy_risk_heatmap_{timestamp}.{config['save_format']}"
                    save_path = save_path / filename

                # Use centralized save_plot utility
                saved_path = save_plot(
                    fig,
                    save_path,
                    save_format=config['save_format'],
                    dpi=config['dpi'],
                    bbox_inches="tight"
                )

                if saved_path:
                    self.logger.info(f"Risk heatmap saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save risk heatmap")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during risk heatmap visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def visualize_disclosure_risk(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Visualize disclosure risk for sensitive attributes

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Quasi-identifier columns
        sensitive_attributes : List[str]
            Sensitive attribute columns
        save_path : str or Path, optional
            Path to save visualization
        **kwargs : dict
            Additional visualization parameters

        Returns:
        --------
        Tuple[plt.Figure, Optional[str]]
            Figure and optional saved path
        """
        try:
            # Get configuration by merging defaults with kwargs
            config = {**self.config, **kwargs}

            # Try to use privacy risk assessor from the processor
            risk_metrics = None
            if self.processor:
                # If processor has evaluate_privacy method, use it
                if hasattr(self.processor, 'evaluate_privacy'):
                    try:
                        # Get risk metrics from processor
                        risk_metrics = self.processor.evaluate_privacy(
                            data, quasi_identifiers, sensitive_attributes,
                            **kwargs.get('risk_kwargs', {})
                        )
                    except Exception as e:
                        self.logger.warning(f"Error using processor's privacy evaluation: {e}")

                # Alternative: check if processor has a risk_assessor attribute
                elif hasattr(self.processor, 'risk_assessor'):
                    try:
                        risk_assessor = self.processor.risk_assessor
                        risk_metrics = risk_assessor.assess_privacy_risks(
                            data, quasi_identifiers, sensitive_attributes,
                            **kwargs.get('risk_kwargs', {})
                        )
                    except Exception as e:
                        self.logger.warning(f"Error using processor's risk assessor: {e}")

            # Create figure with 2 subplots (bar chart and pie chart)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get('figsize', (16, 6)))

            # Process risk metrics if available
            if risk_metrics:
                # Extract attack model risks
                attack_models = risk_metrics.get('attack_models', {})
                prosecutor_risk = attack_models.get('prosecutor_risk', 0)
                journalist_risk = attack_models.get('journalist_risk', 0)
                marketer_risk = attack_models.get('marketer_risk', 0)

                # Create bar chart of attack model risks
                risk_df = pd.DataFrame({
                    'Attack Model': ['Prosecutor', 'Journalist', 'Marketer'],
                    'Risk (%)': [prosecutor_risk, journalist_risk, marketer_risk]
                })

                sns.barplot(
                    x='Attack Model',
                    y='Risk (%)',
                    data=risk_df,
                    palette=['#ff9999', '#66b3ff', '#99ff99'],
                    ax=ax1
                )

                ax1.axhline(y=20, color='orange', linestyle='--', label='Moderate Risk Threshold')
                ax1.axhline(y=50, color='red', linestyle='--', label='High Risk Threshold')
                ax1.legend()
                ax1.set_title('Disclosure Risk by Attack Model')
                ax1.set_ylim(0, 100)

                # Pie chart of attribute risks
                attribute_risks = risk_metrics.get('attribute_risks', {})
                if attribute_risks:
                    # Get risk percentages for attributes
                    attr_names = []
                    attr_risks = []

                    for sa, metrics in attribute_risks.items():
                        attr_names.append(sa)
                        attr_risks.append(metrics.get('risk_percentage', 0))

                    # Create pie chart
                    ax2.pie(
                        attr_risks,
                        labels=attr_names,
                        autopct='%1.1f%%',
                        colors=sns.color_palette('Set3', len(attr_names)),
                        startangle=90
                    )
                    ax2.set_title('Contribution to Overall Risk by Attribute')
                    ax2.axis('equal')  # Equal aspect ratio ensures circular pie
                else:
                    # If no attribute risks available
                    ax2.text(
                        0.5, 0.5,
                        "No attribute risk data available",
                        ha='center', va='center',
                        fontsize=12
                    )
                    ax2.axis('off')

                # Add overall risk as text
                overall_risk = risk_metrics.get('overall_risk', {})
                min_diversity = overall_risk.get('min_diversity', 0)
                overall_compliant = overall_risk.get('overall_compliant', False)

                status_color = 'green' if overall_compliant else 'red'
                status_text = 'COMPLIANT' if overall_compliant else 'NON-COMPLIANT'

                plt.figtext(
                    0.5, 0.01,
                    f"Overall Status: {status_text} | Min Diversity: {min_diversity:.2f} | "
                    f"Prosecutor Risk: {prosecutor_risk:.2f}%",
                    ha='center',
                    fontsize=12,
                    bbox=dict(facecolor=status_color, alpha=0.2, boxstyle='round,pad=0.5')
                )
            else:
                # If no risk metrics available, create a basic visualization
                self.logger.info("No risk metrics available, creating placeholder visualization")

                # Create dummy risk data
                ax1.text(
                    0.5, 0.5,
                    "No risk data available.\nUse a LDiversityRiskAssessor for detailed metrics.",
                    ha='center', va='center',
                    fontsize=12
                )
                ax1.axis('off')

                ax2.text(
                    0.5, 0.5,
                    "No attribute risk data available",
                    ha='center', va='center',
                    fontsize=12
                )
                ax2.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for figtext

            # Save if path provided
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"disclosure_risk_visualization_{timestamp}.{config['save_format']}"
                    save_path = save_path / filename

                # Use centralized save_plot utility
                saved_path = save_plot(
                    fig,
                    save_path,
                    save_format=config['save_format'],
                    dpi=config['dpi'],
                    bbox_inches="tight"
                )

                if saved_path:
                    self.logger.info(f"Disclosure risk visualization saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save disclosure risk visualization")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during disclosure risk visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def _calculate_group_diversity(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            diversity_type: str = 'distinct'
    ) -> pd.DataFrame:
        """
        Calculate group diversity metrics directly when processor cache unavailable

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        diversity_type : str, optional
            Type of diversity to calculate

        Returns:
        --------
        pd.DataFrame
            Group diversity metrics
        """
        # Group data by quasi-identifiers
        grouped = data.groupby(quasi_identifiers)
        diversity_metrics = []

        # Process each group
        for group_name, group_data in grouped:
            # Initialize group metrics
            group_metrics = {}

            # Add quasi-identifier values
            if isinstance(group_name, tuple):
                for i, qi in enumerate(quasi_identifiers):
                    group_metrics[qi] = group_name[i]
            else:
                if quasi_identifiers:
                    group_metrics[quasi_identifiers[0]] = group_name

            # Calculate diversity metrics for each sensitive attribute
            for sa in sensitive_attributes:
                # Skip if attribute not in dataset
                if sa not in group_data.columns:
                    continue

                # Get unique values
                sa_values = group_data[sa].values

                # Calculate distinct count
                distinct_values = len(np.unique(sa_values))
                group_metrics[f"{sa}_distinct"] = distinct_values

                # Entropy calculation if needed
                if diversity_type == 'entropy':
                    unique_values, counts = np.unique(sa_values, return_counts=True)
                    probabilities = counts / len(sa_values)
                    # Calculate entropy (add small epsilon to avoid log(0))
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                    group_metrics[f"{sa}_entropy"] = entropy

            # Add group size
            group_metrics['group_size'] = len(group_data)
            diversity_metrics.append(group_metrics)

        # Convert to DataFrame
        return pd.DataFrame(diversity_metrics)


# Utility functions for standalone usage

def visualize_l_diversity(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        diversity_type: str = 'distinct',
        l_threshold: int = 3,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Quick utility function for l-diversity visualization without requiring a processor

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns
    diversity_type : str, optional
        Type of diversity to visualize
    l_threshold : int, optional
        L-threshold to show in visualization
    save_path : str or Path, optional
        Path to save visualization
    **kwargs : dict
        Additional visualization parameters

    Returns:
    --------
    Tuple[plt.Figure, Optional[str]]
        Figure and optional saved path
    """
    visualizer = LDiversityVisualizer()

    return visualizer.visualize_l_distribution(
        data,
        quasi_identifiers,
        sensitive_attributes,
        diversity_type=diversity_type,
        l_threshold=l_threshold,
        save_path=save_path,
        **kwargs
    )

def visualize_attribute_distributions(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> List[Tuple[plt.Figure, Optional[str]]]:
    """
    Generate visualizations for all sensitive attributes

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns
    save_path : str or Path, optional
        Directory to save visualizations
    **kwargs : dict
        Additional visualization parameters

    Returns:
    --------
    List[Tuple[plt.Figure, Optional[str]]]
        List of figure and saved path tuples
    """
    visualizer = LDiversityVisualizer()
    results = []

    for sa in sensitive_attributes:
        # Create a specific save path for this attribute if directory provided
        attr_save_path = None
        if save_path:
            save_dir = Path(save_path)
            if save_dir.is_dir():
                attr_save_path = save_dir

        # Generate visualization
        fig, saved_path = visualizer.visualize_attribute_distribution(
            data,
            sa,
            quasi_identifiers=quasi_identifiers,
            save_path=attr_save_path,
            **kwargs
        )

        results.append((fig, saved_path))

    return results

def visualize_risk_dashboard(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        processor=None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Create a comprehensive risk dashboard with multiple visualizations

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns
    processor : object, optional
        L-Diversity processor instance for cached calculations
    save_path : str or Path, optional
        Path to save visualization
    **kwargs : dict
        Additional visualization parameters

    Returns:
    --------
    Tuple[plt.Figure, Optional[str]]
        Figure and optional saved path
    """
    # Get configuration
    config = {
        'figsize': (18, 10),
        'style': 'whitegrid',
        'save_format': 'png',
        'dpi': 300
    }
    config.update(kwargs)

    # Create visualizer with processor if provided
    visualizer = LDiversityVisualizer(processor)

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=config['figsize'])

    try:
        # 1. L-Diversity Distribution (top-left)
        l_fig, _ = visualizer.visualize_l_distribution(
            data, quasi_identifiers, sensitive_attributes,
            save_path=None  # Don't save individual plots
        )
        if l_fig:
            # Copy the content to our dashboard
            l_ax = l_fig.get_axes()[0]
            for artist in l_ax.get_children():
                if hasattr(artist, 'get_xydata'):  # For lines
                    xy_data = artist.get_xydata()
                    if len(xy_data) > 0:
                        axes[0, 0].plot(xy_data[:, 0], xy_data[:, 1], color=artist.get_color())
                elif hasattr(artist, 'get_height'):  # For bar plots
                    try:
                        axes[0, 0].bar(
                            artist.get_x() + artist.get_width()/2,
                            artist.get_height(),
                            width=artist.get_width(),
                            color=artist.get_facecolor()
                        )
                    except:
                        pass
            axes[0, 0].set_title('L-Diversity Distribution')
            axes[0, 0].set_xlabel('L-Value')
            axes[0, 0].set_ylabel('Frequency')
            plt.close(l_fig)  # Close the original figure
        else:
            axes[0, 0].text(0.5, 0.5, "Could not generate L-distribution",
                            ha='center', va='center')
            axes[0, 0].axis('off')

        # 2. Risk Heatmap (top-right)
        risk_fig, _ = visualizer.visualize_risk_heatmap(
            data, quasi_identifiers, sensitive_attributes,
            save_path=None  # Don't save individual plots
        )
        if risk_fig:
            # Copy content to our dashboard
            risk_ax = risk_fig.get_axes()[0]
            # For heatmaps, we need to recalculate
            try:
                # Extract the risk matrix data
                risk_matrix = None
                for collection in risk_ax.collections:
                    if hasattr(collection, 'get_array') and collection.get_array() is not None:
                        data_values = collection.get_array()
                        if len(data_values) > 0:
                            # Get dimensions from original axis
                            rows = len(risk_ax.get_yticklabels())
                            cols = len(risk_ax.get_xticklabels())
                            if rows * cols == len(data_values):
                                risk_matrix = data_values.reshape(rows, cols)
                                break

                if risk_matrix is not None:
                    # Create new heatmap in our dashboard
                    sns.heatmap(
                        risk_matrix,
                        annot=True,
                        cmap='YlOrRd',
                        fmt='.2f',
                        cbar_kws={'label': 'Risk Percentage'},
                        ax=axes[0, 1]
                    )
                    # Copy labels
                    if hasattr(risk_ax, 'get_xticklabels') and hasattr(risk_ax, 'get_yticklabels'):
                        x_labels = [label.get_text() for label in risk_ax.get_xticklabels()]
                        y_labels = [label.get_text() for label in risk_ax.get_yticklabels()]
                        axes[0, 1].set_xticklabels(x_labels)
                        axes[0, 1].set_yticklabels(y_labels)
                else:
                    # Fall back to text if extraction failed
                    axes[0, 1].text(0.5, 0.5, "Risk heatmap data extraction failed",
                                    ha='center', va='center')
                    axes[0, 1].axis('off')
            except Exception as e:
                logger.warning(f"Error copying risk heatmap: {e}")
                axes[0, 1].text(0.5, 0.5, "Error copying risk heatmap",
                                ha='center', va='center')
                axes[0, 1].axis('off')

            axes[0, 1].set_title('Privacy Risk Heatmap')
            plt.close(risk_fig)  # Close the original figure
        else:
            axes[0, 1].text(0.5, 0.5, "Could not generate risk heatmap",
                           ha='center', va='center')
            axes[0, 1].axis('off')

        # 3. Attribute Distribution (bottom-left)
        if sensitive_attributes:
            # Pick the first sensitive attribute for visualization
            attr_fig, _ = visualizer.visualize_attribute_distribution(
                data, sensitive_attributes[0], quasi_identifiers=quasi_identifiers,
                save_path=None  # Don't save individual plots
            )
            if attr_fig:
                # Copy content to our dashboard
                attr_ax = attr_fig.get_axes()[0]
                # This is complex to copy directly, so just add a note
                axes[1, 0].text(0.5, 0.5,
                                f"Distribution of {sensitive_attributes[0]}\n"
                                "See separate attribute distribution plots for details",
                                ha='center', va='center')
                axes[1, 0].axis('off')
                plt.close(attr_fig)  # Close the original figure
            else:
                axes[1, 0].text(0.5, 0.5, "Could not generate attribute distribution",
                               ha='center', va='center')
                axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, "No sensitive attributes specified",
                           ha='center', va='center')
            axes[1, 0].axis('off')

        # 4. Disclosure Risk (bottom-right)
        disc_fig, _ = visualizer.visualize_disclosure_risk(
            data, quasi_identifiers, sensitive_attributes,
            save_path=None  # Don't save individual plots
        )
        if disc_fig:
            # Copy content to our dashboard is complex for multi-subplot figures
            # Just add a note
            axes[1, 1].text(0.5, 0.5,
                           "Disclosure Risk Analysis\n"
                           "See separate disclosure risk plot for details",
                           ha='center', va='center')
            axes[1, 1].axis('off')
            plt.close(disc_fig)  # Close the original figure
        else:
            axes[1, 1].text(0.5, 0.5, "Could not generate disclosure risk visualization",
                           ha='center', va='center')
            axes[1, 1].axis('off')

        # Add overall title
        plt.suptitle('L-Diversity Privacy Analysis Dashboard', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save if path provided
        saved_path = None
        if save_path:
            # Process save_path
            save_path = Path(save_path)
            if save_path.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"privacy_dashboard_{timestamp}.{config['save_format']}"
                save_path = save_path / filename

            # Use centralized save_plot utility
            saved_path = save_plot(
                fig,
                save_path,
                save_format=config['save_format'],
                dpi=config['dpi'],
                bbox_inches="tight"
            )

            if saved_path:
                logger.info(f"Privacy dashboard saved to {saved_path}")
            else:
                logger.warning("Failed to save privacy dashboard")

        return fig, saved_path

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        for ax in axes.flatten():
            ax.text(0.5, 0.5, "Error generating dashboard",
                   ha='center', va='center')
            ax.axis('off')
        return fig, None