"""
Core package for data profiling in the HHR anonymization project.

This package provides tools for profiling data sources, analyzing field characteristics,
and generating reports to support data anonymization decisions.
"""

from pamola_core.profiling.commons.data_types import DataType, AnalysisType, ResultType, ArtifactType, OperationStatus, PrivacyLevel
from pamola_core.profiling.commons.base import BaseAnalyzer, BaseOperation, AnalysisResult

__version__ = "1.0.0"