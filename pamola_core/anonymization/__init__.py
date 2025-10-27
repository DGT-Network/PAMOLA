"""
Initialize the fake data operations package.
"""

# Import operations to register them
from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation

# Make operations available at package level
__all__ = [
    'NumericGeneralizationOperation',
    'DateTimeGeneralizationOperation',
    'CategoricalGeneralizationOperation',
    ]