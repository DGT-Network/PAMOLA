from .aggregation import (
    aggregate_column_metrics,
    create_composite_score,
    create_value_dictionary,
)
from .normalize import (
    normalize_metric_value,
    normalize_array_np,
    normalize_array_sklearn,
    round_metric_values,
)
from .validation import validate_dataset_compatibility, validate_metric_inputs
from .safe_instantiate import safe_instantiate
from .preprocessing import prepare_data_for_distance_metrics
from .risk_scoring import calculate_provisional_risk
from .predicted_utility_scoring import calculate_predicted_utility

# Import main components for easy access
from .validation_rules import (
    ValidationRule,
    ValidationResult,
    RequiredRule,
    UniqueRule,
    DatatypeRule,
    MinMaxRule,
    ValidValuesRule,
    RegexRule,
    FormatRule,
    FormatValidator,
    EmailValidator,
    PhoneValidator,
    URLValidator,
    IPValidator,
    CreditCardValidator,
    PostalCodeValidator,
    SSNValidator,
    UUIDValidator,
    ValidationRuleRegistry,
    rule_registry,
    RuleType,
    RuleCode,
)

from .schema_manager import FieldDefinition, SchemaManager

from .quality_scoring import (
    QualityWeights,
    ColumnQualityMetrics,
    DatasetQualityMetrics,
    DataQualityCalculator,
)

# Version information
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__status__ = "stable"

# Main exports for public API
__all__ = [
    "aggregate_column_metrics",
    "create_composite_score",
    "create_value_dictionary",
    "normalize_metric_value",
    "normalize_array_np",
    "normalize_array_sklearn",
    "round_metric_values",
    "validate_dataset_compatibility",
    "validate_metric_inputs",
    "safe_instantiate",
    "prepare_data_for_distance_metrics",
    "calculate_provisional_risk",
    "calculate_predicted_utility",
    # Validation Rules
    "ValidationRule",
    "ValidationResult",
    "RequiredRule",
    "UniqueRule",
    "DatatypeRule",
    "MinMaxRule",
    "ValidValuesRule",
    "RegexRule",
    "FormatRule",
    "ValidationRuleRegistry",
    "rule_registry",
    "RuleType",
    "RuleCode",
    # Format validators
    "FormatValidator",
    "EmailValidator",
    "PhoneValidator",
    "URLValidator",
    "IPValidator",
    "CreditCardValidator",
    "PostalCodeValidator",
    "SSNValidator",
    "UUIDValidator",
    # Schema Management
    "FieldDefinition",
    "SchemaManager",
    # Quality Calculation
    "QualityWeights",
    "ColumnQualityMetrics",
    "DatasetQualityMetrics",
    "DataQualityCalculator",
    # Version info
    "__version__",
    "__author__",
    "__status__",
]


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
    >>> # Use default weights
    >>> calculator = create_quality_calculator()

    >>> # Use custom weights
    >>> calculator = create_quality_calculator({
    ...     'completeness': 0.6,
    ...     'validity': 0.3,
    ...     'diversity': 0.1
    ... })
    """
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


# Add convenience functions to __all__
__all__.extend(
    [
        "create_quality_calculator",
        "create_schema_from_dataframe",
        "calculate_quality_with_rules",
    ]
)
