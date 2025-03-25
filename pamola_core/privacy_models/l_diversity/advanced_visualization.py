"""
PAMOLA.CORE - L-Diversity Advanced Visualization Module

Provides advanced visualization capabilities for l-diversity
anonymization techniques. This module extends the basic
visualization capabilities with more complex visualizations.

Key Features:
- Diversity comparison across groups
- Attack model visualization
- Attribute correlation heatmaps
- Disclosure risk analysis
- Regulatory compliance visualization

This module works seamlessly with the L-Diversity processor's
centralized cache mechanism for efficient visualization generation.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Import file saving utility
from pamola.pamola_core.utils.file_io import save_plot

# Configure logging
logger = logging.getLogger(__name__)


class LDiversityAdvancedVisualizer:
    """
    Generates advanced visualizations for l-diversity analysis with caching support

    This class extends the basic visualization capabilities with more complex
    visualizations for in-depth analysis of l-diversity anonymization.
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
            'figsize': (12, 8),
            'style': 'whitegrid',
            'palette': 'viridis',
            'save_format': 'png',
            'dpi': 300
        }

    def visualize_diversity_comparison(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attribute: str,
            diversity_type: str = 'distinct',
            top_n: int = 10,
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Create a visualization comparing diversity across different equivalence classes

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attribute : str
            Sensitive attribute column to analyze
        diversity_type : str, optional
            Type of diversity to compare ('distinct', 'entropy', 'recursive')
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

            # Retrieve l-threshold from processor or use default
            l_threshold = kwargs.get('l_threshold', None)
            if l_threshold is None and self.processor:
                l_threshold = getattr(self.processor, 'l', 3)
            elif l_threshold is None:
                l_threshold = 3

            # Try to get group diversity from cache
            group_diversity = None
            if self.processor:
                try:
                    # Create cache key
                    cache_key = (tuple(quasi_identifiers), (sensitive_attribute,), diversity_type)

                    # Check processor's cache
                    if hasattr(self.processor, '_results_cache') and cache_key in self.processor._results_cache:
                        group_diversity = self.processor._results_cache[cache_key]
                    else:
                        # Calculate diversity if not in cache
                        group_diversity = self.processor.calculate_group_diversity(
                            data, quasi_identifiers, [sensitive_attribute], diversity_type
                        )
                except Exception as e:
                    self.logger.warning(f"Error accessing processor's cache: {e}")
                    group_diversity = None

            # If cache access failed, calculate diversity directly
            if group_diversity is None:
                self.logger.info("Cache not available, calculating diversity directly")
                group_diversity = self._calculate_group_diversity(
                    data, quasi_identifiers, [sensitive_attribute], diversity_type
                )

            # Prepare diversity metrics for visualization
            # Create column name based on diversity type
            if diversity_type == 'entropy':
                metric_column = f"{sensitive_attribute}_entropy"
                y_label = "Entropy"
            else:
                # For distinct and recursive
                metric_column = f"{sensitive_attribute}_distinct"
                y_label = "Distinct Values"

            # Check if the metric column exists
            if metric_column not in group_diversity.columns:
                raise ValueError(f"Metric column '{metric_column}' not found in diversity data")

            # Create a composite group key for display
            group_diversity = group_diversity.copy()
            if len(quasi_identifiers) > 1:
                group_keys = []
                for _, row in group_diversity.iterrows():
                    # Create a composite key from quasi-identifiers
                    key_parts = []
                    for qi in quasi_identifiers:
                        if qi in row:
                            key_parts.append(str(row[qi]))
                    group_keys.append(' | '.join(key_parts))
                group_diversity['group_key'] = group_keys
            else:
                # Use the single quasi-identifier as key
                group_diversity['group_key'] = group_diversity[quasi_identifiers[0]].astype(str)

            # Sort by diversity and get top N groups
            if metric_column in group_diversity.columns:
                group_diversity = group_diversity.sort_values(metric_column, ascending=False).head(top_n)

            # Create figure
            plt.figure(figsize=config['figsize'])
            plt.style.use(config['style'])

            # Create bar plot
            ax = sns.barplot(
                x='group_key',
                y=metric_column,
                data=group_diversity,
                palette=config.get('palette', 'viridis')
            )

            # Add group size as text on bars if group_size column available
            if 'group_size' in group_diversity.columns:
                for i, bar in enumerate(ax.patches):
                    if i < len(group_diversity):
                        group_size = group_diversity['group_size'].iloc[i]
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f"n={group_size}",
                            ha='center', va='bottom',
                            fontsize=8
                        )

            # Add reference line for l-threshold
            if diversity_type == 'entropy':
                # For entropy, the threshold is log(l)
                threshold = np.log(l_threshold)
                threshold_label = f"log({l_threshold}) = {threshold:.2f}"
            else:
                # For distinct and recursive
                threshold = l_threshold
                threshold_label = f"l = {l_threshold}"

            plt.axhline(
                y=threshold,
                color='red',
                linestyle='--',
                label=threshold_label
            )
            plt.legend()

            # Set title and labels
            plt.title(f"{diversity_type.capitalize()} Diversity of '{sensitive_attribute}' across Equivalence Classes")
            plt.xlabel("Equivalence Classes (Groups with same Quasi-identifiers)")
            plt.ylabel(y_label)

            # Rotate x-labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Adjust layout
            plt.tight_layout()

            # Get figure and save if path provided
            fig = plt.gcf()
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"diversity_comparison_{diversity_type}_{sensitive_attribute}_{timestamp}.{config['save_format']}"
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
                    self.logger.info(f"Diversity comparison plot saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save diversity comparison plot")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during diversity comparison visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def visualize_attribute_correlation(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Create a correlation heatmap between quasi-identifiers and sensitive attributes

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

            # Create a copy of the dataframe to avoid modifying the original
            df = data.copy()

            # Convert categorical columns to numeric for correlation calculation
            for col in quasi_identifiers + sensitive_attributes:
                if col in df.columns:
                    if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                        # Convert to category codes
                        df[col] = pd.Categorical(df[col]).codes

            # Calculate correlation matrix
            # Only include columns that exist in the dataframe
            valid_qi = [qi for qi in quasi_identifiers if qi in df.columns]
            valid_sa = [sa for sa in sensitive_attributes if sa in df.columns]

            if not valid_qi or not valid_sa:
                raise ValueError("No valid quasi-identifiers or sensitive attributes found")

            # Calculate correlation matrix for valid columns
            correlation = df[valid_qi + valid_sa].corr()

            # Extract only QI-SA correlations
            qi_sa_corr = correlation.loc[valid_qi, valid_sa]

            # Create figure
            plt.figure(figsize=config['figsize'])
            plt.style.use(config['style'])

            # Create heatmap
            ax = sns.heatmap(
                qi_sa_corr,
                annot=True,
                cmap=config.get('cmap', 'coolwarm'),
                vmin=-1,
                vmax=1,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'}
            )

            # Set title and labels
            plt.title('Correlation between Quasi-identifiers and Sensitive Attributes')

            # Adjust layout
            plt.tight_layout()

            # Get figure and save if path provided
            fig = plt.gcf()
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"attribute_correlation_{timestamp}.{config['save_format']}"
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
                    self.logger.info(f"Attribute correlation plot saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save attribute correlation plot")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during attribute correlation visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def visualize_attack_models(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Visualize disclosure risk under different attack models

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

            # Try to get risk metrics from the processor
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

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config['figsize'])

            # Process risk metrics if available
            if risk_metrics and 'attack_models' in risk_metrics:
                # Extract attack model risks
                attack_models = risk_metrics['attack_models']
                prosecutor_risk = attack_models.get('prosecutor_risk', 0)
                journalist_risk = attack_models.get('journalist_risk', 0)
                marketer_risk = attack_models.get('marketer_risk', 0)

                # Create comprehensive visualization

                # 1. Bar chart of attack model risks (left subplot)
                model_df = pd.DataFrame({
                    'Attack Model': ['Prosecutor', 'Journalist', 'Marketer'],
                    'Risk (%)': [prosecutor_risk, journalist_risk, marketer_risk]
                })

                # Create bar chart with custom colors
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                sns.barplot(
                    x='Attack Model',
                    y='Risk (%)',
                    data=model_df,
                    palette=colors,
                    ax=ax1
                )

                # Add thresholds for risk levels
                ax1.axhline(y=15, color='green', linestyle='--', alpha=0.7, linewidth=1)
                ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.7, linewidth=1)
                ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=1)

                # Add risk level indicators
                ax1.fill_between([-0.5, 2.5], 0, 15, color='green', alpha=0.1)
                ax1.fill_between([-0.5, 2.5], 15, 30, color='yellow', alpha=0.1)
                ax1.fill_between([-0.5, 2.5], 30, 50, color='orange', alpha=0.1)
                ax1.fill_between([-0.5, 2.5], 50, 100, color='red', alpha=0.1)

                # Add text annotations to the bars
                for i, risk in enumerate([prosecutor_risk, journalist_risk, marketer_risk]):
                    ax1.text(i, risk + 2, f"{risk:.1f}%", ha='center')

                # Add legend for risk levels
                patches = [
                    mpatches.Patch(color='green', alpha=0.3, label='Low Risk (<15%)'),
                    mpatches.Patch(color='yellow', alpha=0.3, label='Medium Risk (15-30%)'),
                    mpatches.Patch(color='orange', alpha=0.3, label='High Risk (30-50%)'),
                    mpatches.Patch(color='red', alpha=0.3, label='Very High Risk (>50%)')
                ]
                ax1.legend(handles=patches, loc='upper right')

                ax1.set_title('Risk by Attack Model')
                ax1.set_ylim(0, 100)  # Fixed scale for risk percentage

                # 2. Risk interpretation (right subplot)
                # Clear the axes for text
                ax2.axis('off')

                # Add attack model descriptions
                model_descriptions = {
                    'Prosecutor': (
                        "PROSECUTOR MODEL:\n"
                        "Assumes the attacker knows their target is in the dataset.\n"
                        f"Risk: {prosecutor_risk:.1f}%\n"
                        f"{self._interpret_risk_level(prosecutor_risk)}"
                    ),
                    'Journalist': (
                        "JOURNALIST MODEL:\n"
                        "Assumes the attacker doesn't know if target is in dataset.\n"
                        f"Risk: {journalist_risk:.1f}%\n"
                        f"{self._interpret_risk_level(journalist_risk)}"
                    ),
                    'Marketer': (
                        "MARKETER MODEL:\n"
                        "Focuses on the fraction of records that can be re-identified.\n"
                        f"Risk: {marketer_risk:.1f}%\n"
                        f"{self._interpret_risk_level(marketer_risk)}"
                    )
                }

                # Add text boxes with interpretations
                y_positions = [0.75, 0.5, 0.25]
                for i, (model, description) in enumerate(model_descriptions.items()):
                    ax2.text(
                        0.5, y_positions[i],
                        description,
                        ha='center', va='center',
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor=self._get_risk_color(
                                [prosecutor_risk, journalist_risk, marketer_risk][i]
                            ),
                            alpha=0.2
                        ),
                        fontsize=10
                    )

                # Add overall risk assessment
                overall_risk = max(prosecutor_risk, journalist_risk, marketer_risk)

                ax2.text(
                    0.5, 0.95,
                    f"OVERALL PRIVACY RISK ASSESSMENT:\n{self._interpret_risk_level(overall_risk)}",
                    ha='center', va='top',
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor=self._get_risk_color(overall_risk),
                        alpha=0.3
                    ),
                    fontsize=12,
                    weight='bold'
                )

                # Add recommendations
                recommendations = self._get_risk_recommendations(
                    overall_risk, prosecutor_risk, risk_metrics
                )

                ax2.text(
                    0.5, 0.05,
                    f"RECOMMENDATIONS:\n{recommendations}",
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.3),
                    fontsize=10,
                    wrap=True
                )

            else:
                # If no risk metrics available, create a simple visualization
                self.logger.info("No risk metrics available, creating placeholder visualization")

                # Create dummy risk data with all three attack models
                dummy_risks = {
                    'Prosecutor': 50,
                    'Journalist': 30,
                    'Marketer': 20
                }

                # Plot dummy data
                ax1.bar(
                    dummy_risks.keys(),
                    dummy_risks.values(),
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )

                ax1.set_title('Attack Models (Example)')
                ax1.set_ylabel('Risk Level (%)')
                ax1.set_ylim(0, 100)

                # Add note about missing risk metrics
                ax1.text(
                    1.5, 75,
                    "NOTE: No actual risk metrics available.\nThis is an example visualization only.",
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                    fontsize=10,
                    color='red'
                )

                # Add explanation text
                ax2.axis('off')
                ax2.text(
                    0.5, 0.5,
                    "Attack Models Explanation:\n\n"
                    "Prosecutor Model: Assumes attacker knows target is in dataset\n\n"
                    "Journalist Model: Assumes attacker doesn't know if target is in dataset\n\n"
                    "Marketer Model: Focuses on fraction of records that can be re-identified\n\n"
                    "\nUse a proper risk assessor to get accurate metrics.",
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                    fontsize=10
                )

            # Set main title
            plt.suptitle('Privacy Disclosure Risk by Attack Model', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

            # Save figure if path provided
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"attack_models_{timestamp}.{config['save_format']}"
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
                    self.logger.info(f"Attack models visualization saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save attack models visualization")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during attack models visualization: {e}", exc_info=True)
            # Create an error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, None

    def visualize_regulatory_compliance(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            regulation: str = 'GDPR',
            save_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Tuple[plt.Figure, Optional[str]]:
        """
        Visualize compliance with a specific privacy regulation

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Quasi-identifier columns
        sensitive_attributes : List[str]
            Sensitive attribute columns
        regulation : str, optional
            Regulation to assess compliance with (default: 'GDPR')
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

            # Define regulatory requirements
            regulatory_requirements = {
                'GDPR': {
                    'l_threshold': 3,
                    'diversity_type': 'distinct',
                    'description': 'General Data Protection Regulation (EU)',
                    'key_articles': ['Art. 5', 'Art. 25', 'Art. 35']
                },
                'HIPAA': {
                    'l_threshold': 4,
                    'diversity_type': 'recursive',
                    'c_value': 0.5,
                    'description': 'Health Insurance Portability and Accountability Act (US)',
                    'key_articles': ['§164.514(b)']
                },
                'CCPA': {
                    'l_threshold': 3,
                    'diversity_type': 'entropy',
                    'description': 'California Consumer Privacy Act (US)',
                    'key_articles': ['1798.140', '1798.100']
                },
                'PIPEDA': {
                    'l_threshold': 3,
                    'diversity_type': 'distinct',
                    'description': 'Personal Information Protection and Electronic Documents Act (Canada)',
                    'key_articles': ['Principle 4.5', 'Principle 4.7']
                }
            }

            # Get requirements for the specified regulation
            if regulation not in regulatory_requirements:
                self.logger.warning(f"Unknown regulation: {regulation}, using GDPR defaults")
                regulation = 'GDPR'

            req = regulatory_requirements[regulation]

            # Try to get risk metrics from the processor
            risk_metrics = None
            if self.processor:
                # If processor has evaluate_privacy method, use it
                if hasattr(self.processor, 'evaluate_privacy'):
                    try:
                        # Extract regulatory parameters
                        risk_kwargs = kwargs.get('risk_kwargs', {})
                        risk_kwargs['diversity_type'] = req['diversity_type']
                        risk_kwargs['l_threshold'] = req['l_threshold']
                        if 'c_value' in req:
                            risk_kwargs['c_value'] = req['c_value']

                        # Get risk metrics from processor with regulatory parameters
                        risk_metrics = self.processor.evaluate_privacy(
                            data, quasi_identifiers, sensitive_attributes,
                            **risk_kwargs
                        )
                    except Exception as e:
                        self.logger.warning(f"Error using processor's privacy evaluation: {e}")

                # Alternative: check if processor has a risk_assessor attribute
                elif hasattr(self.processor, 'risk_assessor'):
                    try:
                        risk_assessor = self.processor.risk_assessor

                        # Extract regulatory parameters
                        risk_kwargs = kwargs.get('risk_kwargs', {})
                        risk_kwargs['diversity_type'] = req['diversity_type']
                        risk_kwargs['l_threshold'] = req['l_threshold']
                        if 'c_value' in req:
                            risk_kwargs['c_value'] = req['c_value']

                        risk_metrics = risk_assessor.assess_privacy_risks(
                            data, quasi_identifiers, sensitive_attributes,
                            **risk_kwargs
                        )
                    except Exception as e:
                        self.logger.warning(f"Error using processor's risk assessor: {e}")

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config['figsize'],
                                           gridspec_kw={'width_ratios': [2, 3]})

            # Process compliance data
            compliance_data = {}
            overall_compliant = False

            if risk_metrics:
                # Extract compliance data
                overall_risk = risk_metrics.get('overall_risk', {})
                overall_compliant = overall_risk.get('overall_compliant', False)
                min_diversity = overall_risk.get('min_diversity', 0)

                # Extract attribute-specific compliance
                attribute_risks = risk_metrics.get('attribute_risks', {})

                for sa in sensitive_attributes:
                    if sa in attribute_risks:
                        sa_risk = attribute_risks[sa]
                        compliance_data[sa] = {
                            'compliant': sa_risk.get('compliant', False),
                            'min_diversity': sa_risk.get('min_diversity', 0),
                            'risk_percentage': sa_risk.get('risk_percentage', 0)
                        }
            else:
                # If no risk metrics, create dummy compliance data
                self.logger.info("No risk metrics available, creating dummy compliance data")

                # Create dummy data
                overall_compliant = False
                min_diversity = 1  # Below most thresholds

                for sa in sensitive_attributes:
                    compliance_data[sa] = {
                        'compliant': False,
                        'min_diversity': 1,  # Below most thresholds
                        'risk_percentage': 70  # High risk
                    }

            # 1. Compliance gauge (left subplot)
            # Clear the axes for custom drawing
            ax1.clear()
            ax1.axis('off')

            # Draw a circular gauge
            center = (0.5, 0.5)
            radius = 0.35

            # Draw gauge background
            compliance_circle = plt.Circle(
                center, radius, color='#f0f0f0', fill=True, alpha=0.5
            )
            ax1.add_patch(compliance_circle)

            # Determine compliance score (0-100)
            if risk_metrics:
                if req['diversity_type'] == 'entropy':
                    # For entropy l-diversity
                    threshold = np.log(req['l_threshold'])
                    compliance_score = min(100, (min_diversity / threshold) * 100)
                else:
                    # For distinct and recursive l-diversity
                    threshold = req['l_threshold']
                    compliance_score = min(100, (min_diversity / threshold) * 100)
            else:
                compliance_score = 30  # Default for dummy data

            # Determine gauge color based on compliance score
            if compliance_score >= 100:
                gauge_color = 'green'
                compliance_status = 'COMPLIANT'
            elif compliance_score >= 75:
                gauge_color = 'yellowgreen'
                compliance_status = 'MOSTLY COMPLIANT'
            elif compliance_score >= 50:
                gauge_color = 'gold'
                compliance_status = 'PARTIALLY COMPLIANT'
            elif compliance_score >= 25:
                gauge_color = 'orange'
                compliance_status = 'MOSTLY NON-COMPLIANT'
            else:
                gauge_color = 'red'
                compliance_status = 'NON-COMPLIANT'

            # Draw filled portion of gauge
            start_angle = -180  # Start at 6 o'clock position
            end_angle = start_angle + (compliance_score / 100) * 180

            # Draw arc
            for angle in range(int(start_angle), int(end_angle), 1):
                rad_angle = math.radians(angle)
                x = center[0] + radius * math.cos(rad_angle)
                y = center[1] + radius * math.sin(rad_angle)

                # Draw a small circle at this point
                ax1.add_patch(plt.Circle((x, y), 0.01, color=gauge_color, alpha=0.8))

            # Add needle
            needle_angle = math.radians(start_angle + (compliance_score / 100) * 180)
            needle_length = radius * 0.9
            needle_x = center[0] + needle_length * math.cos(needle_angle)
            needle_y = center[1] + needle_length * math.sin(needle_angle)

            ax1.plot(
                [center[0], needle_x],
                [center[1], needle_y],
                color='black',
                linewidth=2
            )

            # Add gauge labels
            ax1.text(
                center[0], center[1] - 0.15,
                f"{compliance_score:.1f}%",
                ha='center', va='center',
                fontsize=14,
                weight='bold'
            )

            ax1.text(
                center[0], center[1] - 0.25,
                compliance_status,
                ha='center', va='center',
                fontsize=12,
                color=gauge_color,
                weight='bold'
            )

            # Add regulation info
            ax1.text(
                center[0], center[1] + 0.6,
                f"{regulation}",
                ha='center', va='center',
                fontsize=16,
                weight='bold'
            )

            ax1.text(
                center[0], center[1] + 0.5,
                req['description'],
                ha='center', va='center',
                fontsize=10
            )

            # Add threshold info
            threshold_text = f"Required l-diversity: {req['l_threshold']} ({req['diversity_type']})"
            if 'c_value' in req:
                threshold_text += f", c={req['c_value']}"

            ax1.text(
                center[0], center[1] - 0.6,
                threshold_text,
                ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
            )

            # 2. Compliance details (right subplot)
            ax2.clear()
            ax2.axis('off')

            # Create a table with compliance details
            table_data = []
            table_colors = []

            # Header row
            table_data.append(['Attribute', 'Status', 'Min-L', 'Risk'])
            table_colors.append(['#f0f0f0', '#f0f0f0', '#f0f0f0', '#f0f0f0'])

            # Data rows
            for sa, metrics in compliance_data.items():
                status = 'Compliant' if metrics['compliant'] else 'Non-Compliant'
                status_color = 'green' if metrics['compliant'] else 'red'

                # Format min diversity value
                min_div = metrics['min_diversity']
                if req['diversity_type'] == 'entropy':
                    min_div_text = f"{min_div:.2f}"
                else:
                    min_div_text = f"{min_div:.1f}"

                risk = metrics['risk_percentage']

                table_data.append([sa, status, min_div_text, f"{risk:.1f}%"])

                # Set row color based on compliance
                if metrics['compliant']:
                    cell_color = '#e6ffe6'  # Light green
                else:
                    cell_color = '#ffe6e6'  # Light red

                table_colors.append([cell_color, cell_color, cell_color, cell_color])

            # Create the table
            table = ax2.table(
                cellText=table_data,
                cellColours=table_colors,
                loc='center',
                cellLoc='center'
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Add recommendations
            if risk_metrics:
                # Generate recommendations based on compliance
                if overall_compliant:
                    recommendations = (
                        f"Dataset is compliant with {regulation} requirements. "
                        "Maintain current privacy protections."
                    )
                else:
                    # Find non-compliant attributes
                    non_compliant = [
                        sa for sa, metrics in compliance_data.items()
                        if not metrics['compliant']
                    ]

                    if non_compliant:
                        recommendations = (
                            f"Dataset fails to meet {regulation} requirements for attributes: "
                            f"{', '.join(non_compliant)}. "
                            "Consider additional generalization or suppression."
                        )
                    else:
                        recommendations = (
                            f"Dataset does not fully meet {regulation} requirements. "
                            "Review privacy protections."
                        )
            else:
                # Default recommendation
                recommendations = (
                    "Cannot assess compliance without proper risk metrics. "
                    "Use a risk assessor for accurate evaluation."
                )

            # Add recommendations text
            ax2.text(
                0.5, 0.05,
                f"RECOMMENDATIONS:\n{recommendations}",
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5),
                fontsize=10,
                transform=ax2.transAxes,
                wrap=True
            )

            # Set main title
            plt.suptitle(f'Regulatory Compliance: {regulation}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

            # Save figure if path provided
            saved_path = None

            if save_path:
                # Process save_path
                save_path = Path(save_path)
                if save_path.is_dir():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"regulatory_compliance_{regulation}_{timestamp}.{config['save_format']}"
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
                    self.logger.info(f"Regulatory compliance visualization saved to {saved_path}")
                else:
                    self.logger.warning("Failed to save regulatory compliance visualization")

            return fig, saved_path

        except Exception as e:
            self.logger.error(f"Error during regulatory compliance visualization: {e}", exc_info=True)
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

    def _interpret_risk_level(self, risk_value: float) -> str:
        """
        Provide a human-readable interpretation of a risk value

        Parameters:
        -----------
        risk_value : float
            Risk percentage (0-100)

        Returns:
        --------
        str
            Human-readable interpretation
        """
        if risk_value < 15:
            return "Very Low Risk - Excellent privacy protection"
        elif risk_value < 30:
            return "Low Risk - Good privacy protection"
        elif risk_value < 50:
            return "Moderate Risk - Acceptable for many scenarios"
        elif risk_value < 75:
            return "High Risk - Significant privacy concerns"
        else:
            return "Very High Risk - Severe privacy vulnerabilities"

    def _get_risk_color(self, risk_value: float) -> str:
        """
        Get a color corresponding to a risk level

        Parameters:
        -----------
        risk_value : float
            Risk percentage (0-100)

        Returns:
        --------
        str
            Color code
        """
        if risk_value < 15:
            return '#4CAF50'  # Green
        elif risk_value < 30:
            return '#8BC34A'  # Light Green
        elif risk_value < 50:
            return '#FFC107'  # Amber
        elif risk_value < 75:
            return '#FF9800'  # Orange
        else:
            return '#F44336'  # Red

    def _get_risk_recommendations(
            self,
            overall_risk: float,
            prosecutor_risk: float,
            risk_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get recommendations based on risk assessment

        Parameters:
        -----------
        overall_risk : float
            Overall risk percentage
        prosecutor_risk : float
            Prosecutor risk percentage
        risk_metrics : Dict[str, Any], optional
            Complete risk metrics if available

        Returns:
        --------
        str
            Risk recommendations
        """
        # General recommendations based on risk level
        if overall_risk < 15:
            recommendations = "Current privacy protection is strong. Data is suitable for most uses."
        elif overall_risk < 30:
            recommendations = "Consider additional anonymization for highly sensitive data."
        elif overall_risk < 50:
            recommendations = (
                "Apply stronger generalization to quasi-identifiers or reduce granularity "
                "of sensitive attributes."
            )
        elif overall_risk < 75:
            recommendations = (
                "Not suitable for public release without further anonymization. "
                "Use stronger privacy techniques."
            )
        else:
            recommendations = (
                "Data presents serious privacy risks. Substantial additional anonymization required "
                "before any sharing or release."
            )

        # Add specific recommendations if risk metrics available
        if risk_metrics and 'attribute_risks' in risk_metrics:
            attribute_risks = risk_metrics['attribute_risks']
            high_risk_attrs = []

            for attr, metrics in attribute_risks.items():
                if metrics.get('risk_percentage', 0) > 50:
                    high_risk_attrs.append(attr)

            if high_risk_attrs:
                recommendations += f" Pay special attention to high-risk attributes: {', '.join(high_risk_attrs)}."

        return recommendations


# Utility functions for standalone usage

def visualize_attribute_correlation(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Create a correlation heatmap between quasi-identifiers and sensitive attributes

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
    visualizer = LDiversityAdvancedVisualizer()

    return visualizer.visualize_attribute_correlation(
        data,
        quasi_identifiers,
        sensitive_attributes,
        save_path=save_path,
        **kwargs
    )


def visualize_attack_models(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        processor=None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Visualize disclosure risk under different attack models

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
    visualizer = LDiversityAdvancedVisualizer(processor)

    return visualizer.visualize_attack_models(
        data,
        quasi_identifiers,
        sensitive_attributes,
        save_path=save_path,
        **kwargs
    )


def visualize_regulatory_compliance(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        processor=None,
        regulation: str = 'GDPR',
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Visualize compliance with a specific privacy regulation

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
    regulation : str, optional
        Regulation to assess compliance with (default: 'GDPR')
    save_path : str or Path, optional
        Path to save visualization
    **kwargs : dict
        Additional visualization parameters

    Returns:
    --------
    Tuple[plt.Figure, Optional[str]]
        Figure and optional saved path
    """
    visualizer = LDiversityAdvancedVisualizer(processor)

    return visualizer.visualize_regulatory_compliance(
        data,
        quasi_identifiers,
        sensitive_attributes,
        regulation=regulation,
        save_path=save_path,
        **kwargs
    )


def create_comprehensive_report(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        processor=None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
) -> Dict[str, Optional[str]]:
    """
    Generate a comprehensive set of visualizations for l-diversity analysis

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
    output_dir : str or Path, optional
        Directory to save visualizations
    **kwargs : dict
        Additional visualization parameters

    Returns:
    --------
    Dict[str, Optional[str]]
        Dictionary mapping visualization names to their saved paths
    """
    visualizer = LDiversityAdvancedVisualizer(processor)
    basic_visualizer = __import__('visualization').LDiversityVisualizer(processor)

    # Create output directory if provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Record saved paths
    saved_paths = {}

    # 1. L-Diversity Distribution
    _, saved_path = basic_visualizer.visualize_l_distribution(
        data, quasi_identifiers, sensitive_attributes,
        save_path=output_dir,
        **kwargs
    )
    saved_paths['l_distribution'] = saved_path

    # 2. Risk Heatmap
    _, saved_path = basic_visualizer.visualize_risk_heatmap(
        data, quasi_identifiers, sensitive_attributes,
        save_path=output_dir,
        **kwargs
    )
    saved_paths['risk_heatmap'] = saved_path

    # 3. Attack Models
    _, saved_path = visualizer.visualize_attack_models(
        data, quasi_identifiers, sensitive_attributes,
        save_path=output_dir,
        **kwargs
    )
    saved_paths['attack_models'] = saved_path

    # 4. Attribute Correlation
    _, saved_path = visualizer.visualize_attribute_correlation(
        data, quasi_identifiers, sensitive_attributes,
        save_path=output_dir,
        **kwargs
    )
    saved_paths['attribute_correlation'] = saved_path

    # 5. Regulatory Compliance (GDPR)
    _, saved_path = visualizer.visualize_regulatory_compliance(
        data, quasi_identifiers, sensitive_attributes,
        regulation='GDPR',
        save_path=output_dir,
        **kwargs
    )
    saved_paths['gdpr_compliance'] = saved_path

    # 6. Attribute-specific visualizations
    attribute_paths = {}

    for sa in sensitive_attributes:
        # Attribute distribution
        _, saved_path = basic_visualizer.visualize_attribute_distribution(
            data, sa, quasi_identifiers=quasi_identifiers,
            save_path=output_dir,
            **kwargs
        )
        attribute_paths[f"{sa}_distribution"] = saved_path

        # Diversity comparison
        _, saved_path = visualizer.visualize_diversity_comparison(
            data, quasi_identifiers, sa,
            save_path=output_dir,
            **kwargs
        )
        attribute_paths[f"{sa}_diversity_comparison"] = saved_path

    saved_paths['attribute_visualizations'] = attribute_paths

    # Return all saved paths
    return saved_paths