"""
Core package for data profiling in the anonymization project.

This package provides tools for profiling data sources, analyzing field characteristics,
and generating reports to support data anonymization decisions.
"""

from pamola_core.profiling.commons.data_types import DataType, AnalysisType, ResultType, ArtifactType, OperationStatus, PrivacyLevel

__version__ = "1.0.0"

# First export the type checking functions which are the most fundamental
from pamola_core.profiling.commons.dtype_helpers import (
    is_numeric_dtype, is_bool_dtype, is_object_dtype, is_string_dtype,
    is_datetime64_dtype, is_categorical_dtype, is_integer_dtype,
    is_float_dtype, is_list_like, is_dict_like
)


# Import operations to register them
from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation
from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.profiling.analyzers.correlation import CorrelationOperation, CorrelationMatrixOperation
from pamola_core.profiling.analyzers.date import DateOperation
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.profiling.analyzers.group import GroupAnalyzerOperation
from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
from pamola_core.profiling.analyzers.mvf import MVFOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.profiling.analyzers.phone import PhoneOperation
from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation  
from pamola_core.profiling.analyzers.currency import CurrencyOperation

# Make operations available at package level
__all__ = [
    'KAnonymityProfilerOperation', 
    'DataAttributeProfilerOperation', 
    'CategoricalOperation', 
    'CorrelationOperation',
    'CorrelationMatrixOperation', 
    'DateOperation', 
    'EmailOperation', 
    'GroupAnalyzerOperation',
    'IdentityAnalysisOperation', 
    'MVFOperation', 
    'NumericOperation', 
    'PhoneOperation', 
    'TextSemanticCategorizerOperation',
    'CurrencyOperation'
    ]