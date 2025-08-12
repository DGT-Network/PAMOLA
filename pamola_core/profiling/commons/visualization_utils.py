"""
Visualization utilities for the profiling package.

This module provides specialized utility functions for creating various types of plots
and visualizations used specifically in profiling operations, especially for correlation
analysis and other relationship visualizations.

Core components:
- create_scatter_plot: Create scatter plots for numeric-numeric correlations
- create_boxplot: Create boxplots for numeric-categorical relationships
- create_heatmap: Create heatmaps for categorical-categorical relationships
- create_correlation_matrix_plot: Create visualization of correlation matrices
"""

import logging
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)


def create_scatter_plot(plot_data: Dict[str, Any],
                        field1: str,
                        field2: str,
                        correlation: float,
                        method: str) -> plt.Figure:
    """
    Create a scatter plot for numeric-numeric correlation.

    Parameters:
    -----------
    plot_data : Dict[str, Any]
        Plot data configuration
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field
    correlation : float
        Correlation coefficient
    method : str
        Correlation method used

    Returns:
    --------
    plt.Figure
        Matplotlib figure with the scatter plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot
    x_values = plot_data['x_values']
    y_values = plot_data['y_values']
    scatter = ax.scatter(x_values, y_values, alpha=0.5, edgecolors='w', linewidth=0.5)

    # Add regression line
    if len(x_values) > 1 and len(y_values) > 1:
        x_line = np.array([min(x_values), max(x_values)])
        if method == 'pearson':
            # For Pearson, we can use a linear regression line
            coefficients = np.polyfit(x_values, y_values, 1)
            polynomial = np.poly1d(coefficients)
            ax.plot(x_line, polynomial(x_line), color='red', linewidth=2)

    # Add labels and title
    ax.set_xlabel(plot_data.get('x_label', field1))
    ax.set_ylabel(plot_data.get('y_label', field2))
    ax.set_title(f"Correlation between {field1} and {field2}")

    # Add correlation annotation
    corr_text = f"{method.capitalize()} correlation: {correlation:.3f}"
    ax.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def create_boxplot(plot_data: Dict[str, Any],
                   correlation: float,
                   method: str) -> plt.Figure:
    """
    Create a boxplot for numeric-categorical correlation.

    Parameters:
    -----------
    plot_data : Dict[str, Any]
        Plot data configuration
    correlation : float
        Correlation coefficient
    method : str
        Correlation method used

    Returns:
    --------
    plt.Figure
        Matplotlib figure with the boxplot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert data to DataFrame for seaborn
    df_plot = pd.DataFrame({
        'category': plot_data['categories'],
        'value': plot_data['values']
    })

    # Create boxplot
    sns.boxplot(x='category', y='value', data=df_plot, ax=ax)

    # Add individual points for better visibility
    sns.stripplot(x='category', y='value', data=df_plot,
                  size=4, color='.3', linewidth=0, ax=ax)

    # Add labels and title
    ax.set_xlabel(plot_data.get('x_label', 'Category'))
    ax.set_ylabel(plot_data.get('y_label', 'Value'))
    ax.set_title(f"Relationship between {plot_data.get('x_label')} and {plot_data.get('y_label')}")

    # Add correlation annotation
    corr_text = f"{method.capitalize()}: {correlation:.3f}"
    ax.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def create_heatmap(plot_data: Dict[str, Any],
                   field1: str,
                   field2: str,
                   correlation: float,
                   method: str) -> plt.Figure:
    """
    Create a heatmap for categorical-categorical correlation.

    Parameters:
    -----------
    plot_data : Dict[str, Any]
        Plot data configuration
    field1 : str
        Name of the first field
    field2 : str
        Name of the second field
    correlation : float
        Correlation coefficient
    method : str
        Correlation method used

    Returns:
    --------
    plt.Figure
        Matplotlib figure with the heatmap
    """
    # Convert matrix dict to DataFrame
    matrix_data = plot_data['matrix']
    matrix_df = pd.DataFrame(matrix_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(matrix_df, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=.5, ax=ax, cbar_kws={'label': 'Proportion'})

    # Add labels and title
    ax.set_xlabel(plot_data.get('x_label', field2))
    ax.set_ylabel(plot_data.get('y_label', field1))
    ax.set_title(f"Relationship between {field1} and {field2}")

    # Add correlation annotation
    corr_text = f"{method.capitalize()}: {correlation:.3f}"
    ax.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_correlation_matrix_plot(matrix_data: Dict[str, Any],
                                   title: str = "Correlation Matrix") -> plt.Figure:
    """
    Create a visualization of a correlation matrix.

    Parameters:
    -----------
    matrix_data : Dict[str, Any]
        Dictionary containing correlation matrix data
    title : str
        Title for the plot

    Returns:
    --------
    plt.Figure
        Matplotlib figure with the correlation matrix plot
    """
    try:
        # Convert dict to DataFrame
        if isinstance(matrix_data, dict) and 'correlation_matrix' in matrix_data:
            corr_matrix = pd.DataFrame(matrix_data['correlation_matrix'])
        else:
            corr_matrix = pd.DataFrame(matrix_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create heatmap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    ax=ax)

        # Add title
        ax.set_title(title)

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating correlation matrix plot: {e}", exc_info=True)
        # Create error plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error creating correlation matrix plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax.set_title(f"Correlation Matrix Plot Error")
        ax.axis('off')
        return fig


def create_error_plot(error_message: str, title: str = "Error") -> plt.Figure:
    """
    Create a plot displaying an error message.

    Parameters:
    -----------
    error_message : str
        Error message to display
    title : str
        Title for the plot

    Returns:
    --------
    plt.Figure
        Matplotlib figure with the error message
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, error_message,
            horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.set_title(title)
    ax.axis('off')
    return fig