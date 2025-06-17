import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unittest import mock
from pamola_core.profiling.commons import visualization_utils


def test_create_scatter_plot_valid_case():
    plot_data = {
        'x_values': [1, 2, 3, 4, 5],
        'y_values': [2, 4, 6, 8, 10],
        'x_label': 'X',
        'y_label': 'Y',
    }
    fig = visualization_utils.create_scatter_plot(plot_data, 'X', 'Y', 1.0, 'pearson')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_scatter_plot_empty_data():
    plot_data = {
        'x_values': [],
        'y_values': [],
        'x_label': 'X',
        'y_label': 'Y',
    }
    fig = visualization_utils.create_scatter_plot(plot_data, 'X', 'Y', 0.0, 'pearson')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_scatter_plot_invalid_data():
    plot_data = {
        'x_values': [1, 2, 3],
        'y_values': [],
    }
    with pytest.raises(ValueError):
        visualization_utils.create_scatter_plot(plot_data, 'X', 'Y', 0.0, 'pearson')


def test_create_boxplot_valid_case():
    plot_data = {
        'categories': ['A', 'A', 'B', 'B'],
        'values': [1, 2, 3, 4],
        'x_label': 'Category',
        'y_label': 'Value',
    }
    fig = visualization_utils.create_boxplot(plot_data, 0.5, 'anova')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_boxplot_empty_data():
    plot_data = {
        'categories': [],
        'values': [],
        'x_label': 'Category',
        'y_label': 'Value',
    }
    fig = visualization_utils.create_boxplot(plot_data, 0.0, 'anova')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_boxplot_invalid_data():
    plot_data = {
        'categories': [1, 2, 3],
        'values': 'not_a_list',
    }
    fig = visualization_utils.create_boxplot(plot_data, 0.0, 'anova')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_heatmap_valid_case():
    plot_data = {
        'matrix': [[0.1, 0.2], [0.3, 0.4]],
        'x_label': 'Col',
        'y_label': 'Row',
    }
    fig = visualization_utils.create_heatmap(plot_data, 'Row', 'Col', 0.7, 'cramers_v')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_heatmap_empty_matrix():
    plot_data = {
        'matrix': [],
        'x_label': 'Col',
        'y_label': 'Row',
    }
    with pytest.raises(ValueError):
        visualization_utils.create_heatmap(plot_data, 'Row', 'Col', 0.0, 'cramers_v')


def test_create_heatmap_invalid_matrix():
    plot_data = {
        'matrix': 'not_a_matrix',
    }
    with pytest.raises(Exception):
        visualization_utils.create_heatmap(plot_data, 'Row', 'Col', 0.0, 'cramers_v')


def test_create_correlation_matrix_plot_valid_case():
    matrix_data = {
        'correlation_matrix': [[1.0, 0.5], [0.5, 1.0]]
    }
    fig = visualization_utils.create_correlation_matrix_plot(matrix_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_correlation_matrix_plot_dict_matrix():
    matrix_data = [[1.0, -0.5], [-0.5, 1.0]]
    fig = visualization_utils.create_correlation_matrix_plot(matrix_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_correlation_matrix_plot_invalid_matrix():
    matrix_data = 'not_a_matrix'
    fig = visualization_utils.create_correlation_matrix_plot(matrix_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_correlation_matrix_plot_exception(monkeypatch):
    def raise_exception(*args, **kwargs):
        raise ValueError('Test error')
    monkeypatch.setattr(pd, 'DataFrame', raise_exception)
    matrix_data = {'correlation_matrix': [[1.0, 0.5], [0.5, 1.0]]}
    fig = visualization_utils.create_correlation_matrix_plot(matrix_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_error_plot_valid_case():
    fig = visualization_utils.create_error_plot('This is an error', 'Error Title')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_error_plot_empty_message():
    fig = visualization_utils.create_error_plot('', '')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_error_plot_long_message():
    long_message = 'A' * 1000
    fig = visualization_utils.create_error_plot(long_message, 'Long Error')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

if __name__ == "__main__":
    pytest.main()
