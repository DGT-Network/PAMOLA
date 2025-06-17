"""
Common utilities and base classes for the profiling package.

This subpackage contains pamola_core abstractions and utilities used throughout the profiling system.
"""

# First, import our type checking functions so they're available everywhere
from pamola_core.profiling.commons.dtype_helpers import (
    is_numeric_dtype, is_bool_dtype, is_object_dtype, is_string_dtype,
    is_datetime64_dtype, is_categorical_dtype, is_integer_dtype,
    is_float_dtype, is_list_like, is_dict_like
)

# Import data_types now, as it has no internal dependencies
from pamola_core.profiling.commons.data_types import DataType, AnalysisType, ResultType, ArtifactType, OperationStatus, PrivacyLevel, ProfilerConfig, FieldCategory

# Import base classes
from pamola_core.profiling.commons.base import BaseAnalyzer, BaseOperation, AnalysisResult, BaseMultiFieldAnalyzer, DataFrameProfiler, ProfileOperation

# Finally, import helpers that depend on data_types
from pamola_core.profiling.commons.helpers import (
    infer_data_type, prepare_field_for_analysis, parse_multi_valued_field,
    detect_json_field, parse_json_field, detect_array_field, parse_array_field,
    is_valid_email, extract_email_domain, is_phone_number_format, parse_phone_number,
    convert_numpy_types
)