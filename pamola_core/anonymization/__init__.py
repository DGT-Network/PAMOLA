"""
Initialize the fake data operations package.
"""

# Import operations to register them
from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation
from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation

from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
# from pamola_core.anonymization.pseudonymization.hash_based_op import HashBasedPseudonymizationOperation
# from pamola_core.anonymization.pseudonymization.mapping_op import ConsistentMappingPseudonymizationOperation

from pamola_core.anonymization.noise.uniform_numeric_op import UniformNumericNoiseOperation
from pamola_core.anonymization.noise.uniform_temporal_op import UniformTemporalNoiseOperation

# Make operations available at package level
__all__ = [
    'AttributeSuppressionOperation',
    'RecordSuppressionOperation',
    'CellSuppressionOperation',
    'NumericGeneralizationOperation',
    'DateTimeGeneralizationOperation',
    'CategoricalGeneralizationOperation',
    # 'HashBasedPseudonymizationOperation',
    # 'ConsistentMappingPseudonymizationOperation',
    'UniformNumericNoiseOperation',
    'UniformTemporalNoiseOperation',
    ]