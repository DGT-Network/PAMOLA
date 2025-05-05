"""
Base classes and interfaces for the profiling package.

This module defines the core abstractions used throughout the profiling system:
- BaseAnalyzer: Abstract base class for all data analyzers
- BaseOperation: Abstract base class for profiling operations
- AnalysisResult: Class representing results of analysis operations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class AnalysisResult:
    """
    Class representing the results of a profiling analysis operation.

    This class encapsulates the results of an analysis operation, including
    statistics, metadata, and paths to generated artifacts.
    """

    def __init__(self,
                 stats: Dict[str, Any],
                 field_name: Optional[str] = None,
                 data_type: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new analysis result.

        Parameters:
        -----------
        stats : Dict[str, Any]
            The actual statistics and analysis results
        field_name : str, optional
            The name of the analyzed field
        data_type : str, optional
            The data type of the analyzed field
        metadata : Dict[str, Any], optional
            Additional metadata about the analysis
        """
        self.stats = stats
        self.field_name = field_name
        self.data_type = data_type
        self.metadata = metadata or {}
        self.artifacts = []

    def add_artifact(self,
                     artifact_type: str,
                     artifact_path: str,
                     description: str = "") -> None:
        """
        Add an artifact to the analysis result.

        Parameters:
        -----------
        artifact_type : str
            Type of the artifact (json, csv, png, etc.)
        artifact_path : str
            Path to the artifact
        description : str
            Description of the artifact
        """
        self.artifacts.append({
            "type": artifact_type,
            "path": artifact_path,
            "description": description
        })

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the analysis result to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            Dictionary representation of the analysis result
        """
        return {
            "stats": self.stats,
            "field_name": self.field_name,
            "data_type": self.data_type,
            "metadata": self.metadata,
            "artifacts": self.artifacts
        }


class BaseAnalyzer(ABC):
    """
    Abstract base class for all data analyzers.

    This class defines the interface for data analyzers that perform
    various types of analysis on data fields.
    """

    @abstractmethod
    def analyze(self,
                df: pd.DataFrame,
                field_name: str,
                **kwargs) -> AnalysisResult:
        """
        Analyze a field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        AnalysisResult
            The results of the analysis
        """
        pass


class BaseMultiFieldAnalyzer(ABC):
    """
    Abstract base class for analyzers that work with multiple fields.

    This class defines the interface for analyzers that analyze relationships
    between multiple fields, such as correlation analysis.
    """

    @abstractmethod
    def analyze_fields(self,
                       df: pd.DataFrame,
                       field_names: List[str],
                       **kwargs) -> AnalysisResult:
        """
        Analyze multiple fields in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_names : List[str]
            The names of the fields to analyze
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        AnalysisResult
            The results of the analysis
        """
        pass


class BaseOperation(ABC):
    """
    Abstract base class for profiling operations.

    This class defines the interface for operations that can be executed
    as part of a profiling task.
    """

    @abstractmethod
    def execute(self,
                df: pd.DataFrame,
                reporter: Any,
                profile_type: str,
                **kwargs) -> None:
        """
        Execute the operation.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to operate on
        reporter : Any
            The reporter object for tracking progress and artifacts
        profile_type : str
            The type of profiling being performed
        **kwargs : dict
            Additional parameters for the operation
        """
        pass


class DataFrameProfiler:
    """
    Base class for profiling an entire DataFrame.

    This class provides methods for profiling all columns in a DataFrame
    and generating comprehensive reports.
    """

    def __init__(self, analyzers: Optional[Dict[str, BaseAnalyzer]] = None):
        """
        Initialize a new DataFrame profiler.

        Parameters:
        -----------
        analyzers : Dict[str, BaseAnalyzer], optional
            Dictionary mapping data types to appropriate analyzers
        """
        self.analyzers = analyzers or {}

    def profile_dataframe(self,
                          df: pd.DataFrame,
                          include_fields: Optional[List[str]] = None,
                          exclude_fields: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, AnalysisResult]:
        """
        Profile all columns in the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to profile
        include_fields : List[str], optional
            List of fields to include in profiling. If None, all fields are included.
        exclude_fields : List[str], optional
            List of fields to exclude from profiling
        **kwargs : dict
            Additional parameters for the profiling

        Returns:
        --------
        Dict[str, AnalysisResult]
            Dictionary mapping field names to analysis results
        """
        results = {}

        # Determine fields to profile
        fields_to_profile = include_fields if include_fields else df.columns
        if exclude_fields:
            fields_to_profile = [f for f in fields_to_profile if f not in exclude_fields]

        # Profile each field
        for field in fields_to_profile:
            try:
                data_type = self._infer_data_type(df[field])
                analyzer = self._get_analyzer_for_type(data_type)
                if analyzer:
                    results[field] = analyzer.analyze(df, field, **kwargs)
                else:
                    logger.warning(f"No analyzer found for field {field} with type {data_type}")
            except Exception as e:
                logger.error(f"Error profiling field {field}: {e}", exc_info=True)

        return results

    def _infer_data_type(self, series: pd.Series) -> str:
        """
        Infer the data type of a series.

        This is a simplistic implementation. The actual implementation should use
        the more sophisticated infer_data_type function from helpers.py.

        Parameters:
        -----------
        series : pd.Series
            The series to analyze

        Returns:
        --------
        str
            The inferred data type
        """
        from core.profiling.commons.helpers import infer_data_type
        return infer_data_type(series)

    def _get_analyzer_for_type(self, data_type: str) -> Optional[BaseAnalyzer]:
        """
        Get the appropriate analyzer for a data type.

        Parameters:
        -----------
        data_type : str
            The data type

        Returns:
        --------
        BaseAnalyzer or None
            The appropriate analyzer, or None if no analyzer is found
        """
        return self.analyzers.get(data_type)


class ProfileOperation:
    """
    Base class for a profiling operation that combines multiple analyzers.

    This class provides a way to execute a sequence of operations
    on a DataFrame and collect the results.
    """

    def __init__(self, name: str, operations: Optional[List[BaseOperation]] = None):
        """
        Initialize a new profile operation.

        Parameters:
        -----------
        name : str
            The name of the profile operation
        operations : List[BaseOperation], optional
            List of operations to execute
        """
        self.name = name
        self.operations = operations or []

    def add_operation(self, operation: BaseOperation) -> None:
        """
        Add an operation to the profile operation.

        Parameters:
        -----------
        operation : BaseOperation
            The operation to add
        """
        self.operations.append(operation)

    def execute(self,
                df: pd.DataFrame,
                reporter: Any,
                profile_type: str,
                **kwargs) -> Dict[str, Any]:
        """
        Execute all operations.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to operate on
        reporter : Any
            The reporter object for tracking progress and artifacts
        profile_type : str
            The type of profiling being performed
        **kwargs : dict
            Additional parameters for the operations

        Returns:
        --------
        Dict[str, Any]
            Dictionary mapping operation names to results
        """
        results = {}

        reporter.add_operation(f"Starting profile operation: {self.name}", details={
            "description": f"Executing {len(self.operations)} operations"
        })

        for i, operation in enumerate(self.operations):
            try:
                operation_name = getattr(operation, 'name', f"Operation {i + 1}")
                reporter.add_operation(f"Executing {operation_name}", details={
                    "operation_index": i + 1,
                    "total_operations": len(self.operations)
                })

                result = operation.execute(df, reporter, profile_type, **kwargs)
                results[operation_name] = result

                reporter.add_operation(f"Completed {operation_name}", status="success")

            except Exception as e:
                logger.error(f"Error executing operation {i + 1}: {e}", exc_info=True)
                reporter.add_operation(f"Operation {i + 1}", status="error",
                                       details={"error": str(e)})

        reporter.add_operation(f"Completed profile operation: {self.name}", details={
            "operations_completed": len(results),
            "operations_failed": len(self.operations) - len(results)
        })

        return results