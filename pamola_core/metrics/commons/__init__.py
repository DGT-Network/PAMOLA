"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Package: pamola_core.metrics.commons
Type: Internal (Non-Public API)

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pamola_core.metrics.commons.schema_manager import SchemaManager
    from pamola_core.metrics.commons.quality_scoring import (
        DataQualityCalculator,
        QualityWeights,
    )

__all__ = [
    # schema_manager.py
    "SchemaManager",
    # quality_scoring.py
    "DataQualityCalculator",
    "QualityWeights",
    # convenience functions
    "create_quality_calculator",
    "create_schema_from_dataframe",
    "calculate_quality_with_rules",
    # risk_scoring.py
    "calculate_provisional_risk",
    # predicted_utility_scoring.py
    "calculate_predicted_utility",
    # validation_rules.py
    "RuleCode",
]

from pamola_core.metrics.commons.schema_manager import SchemaManager

from pamola_core.metrics.commons.quality_scoring import DataQualityCalculator
from pamola_core.metrics.commons.quality_scoring import QualityWeights

from pamola_core.metrics.commons.risk_scoring import calculate_provisional_risk

from pamola_core.metrics.commons.predicted_utility_scoring import calculate_predicted_utility

from pamola_core.metrics.commons.validation_rules import RuleCode

def create_quality_calculator(weights: dict = None) -> DataQualityCalculator:
    """
    Convenience function to create a DataQualityCalculator with custom weights.

    Parameters
    ----------
    weights : dict, optional
        Dictionary with 'completeness', 'validity', 'diversity' keys.
        Defaults to standard weights (0.5, 0.3, 0.2).

    Returns
    -------
    DataQualityCalculator
        Configured quality calculator instance

    Examples
    --------
    >>> calculator = create_quality_calculator()
    >>> calculator = create_quality_calculator({
    ...     'completeness': 0.6,
    ...     'validity': 0.3,
    ...     'diversity': 0.1
    ... })
    """
    from pamola_core.metrics.commons.quality_scoring import (
        DataQualityCalculator,
        QualityWeights,
    )

    if weights is None:
        return DataQualityCalculator()

    quality_weights = QualityWeights(
        completeness=weights.get("completeness", 0.5),
        validity=weights.get("validity", 0.3),
        diversity=weights.get("diversity", 0.2),
    )

    return DataQualityCalculator(quality_weights)

def create_schema_from_dataframe(df, auto_detect: bool = True) -> SchemaManager:
    """
    Convenience function to create a SchemaManager from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze
    auto_detect : bool, default True
        Whether to auto-detect field definitions

    Returns
    -------
    SchemaManager
        Schema manager with field definitions

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
    >>> schema = create_schema_from_dataframe(df)
    """
    from pamola_core.metrics.commons.schema_manager import SchemaManager

    schema = SchemaManager()
    if auto_detect:
        schema.auto_detect_schema(df)
    return schema

def calculate_quality_with_rules(
    df,
    schema: SchemaManager = None,
    weights: dict = None,
    analyze_scope: str = "dataset",
    columns: list = None,
) -> dict:
    """
    Convenience function to calculate quality metrics with validation rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze
    schema : SchemaManager, optional
        Schema with field definitions and rules. If None, auto-detects schema.
    weights : dict, optional
        Custom quality weights. If None, uses default weights.
    analyze_scope : str, optional
        "dataset" (default) to compute full metrics, or "column" to compute only specified columns.
    columns : list, optional
        Column names to analyze when analyze_scope="column".

    Returns
    -------
    dict
        Comprehensive quality metrics including UI-friendly outputs

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
    >>> results = calculate_quality_with_rules(df)
    >>> print(f"Quality: {results['dataset_card']['overall_quality']:.1f}%")
    """
    if schema is None:
        schema = create_schema_from_dataframe(df)

    calculator = create_quality_calculator(weights)
    return calculator.calculate_quality(
        df, schema, analyze_scope=analyze_scope, columns=columns
    )
