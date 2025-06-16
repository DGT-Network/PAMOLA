"""
Core package for data profiling in the HHR anonymization project.

This package provides tools for profiling data sources, analyzing field characteristics,
and generating reports to support data anonymization decisions.
"""

from pamola_core.profiling.commons.data_types import DataType, AnalysisType, ResultType, ArtifactType, OperationStatus, PrivacyLevel
from pamola_core.profiling.commons.base import BaseAnalyzer, BaseOperation, AnalysisResult

__version__ = "1.0.0"


# Import operations to register them
from pamola_core.profiling.analyzers.anonymity import PreKAnonymityProfilingOperation
from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.profiling.analyzers.correlation import CorrelationOperation, CorrelationMatrixOperation
from pamola_core.profiling.analyzers.date import DateOperation
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.profiling.analyzers.group import GroupOperation
from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
from pamola_core.profiling.analyzers.mvf import MVFOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.profiling.analyzers.phone import PhoneOperation
from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation  

# Make operations available at package level
__all__ = [
    'PreKAnonymityProfilingOperation', 
    'DataAttributeProfilerOperation', 
    'CategoricalOperation', 
    'CorrelationOperation',
    'CorrelationMatrixOperation', 
    'DateOperation', 
    'EmailOperation', 
    'GroupOperation',
    'IdentityAnalysisOperation', 
    'MVFOperation', 
    'NumericOperation', 
    'PhoneOperation', 
    'TextSemanticCategorizerOperation'
    ]
