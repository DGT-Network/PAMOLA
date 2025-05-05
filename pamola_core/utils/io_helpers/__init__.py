"""
Helper modules for the HHR I/O utilities.

This package contains helper modules that provide specialized functionality
for the main I/O module. These helpers handle integration with optional
dependencies like Dask, encryption/decryption, and support for various
file formats.
"""

# Import directory management utilities
from pamola_core.utils.io_helpers.directory_utils import (
    ensure_directory,
    get_timestamped_filename,
    get_file_stats,
    list_directory_contents,
    clear_directory,
    make_unique_path
)

# Import CSV utilities
from pamola_core.utils.io_helpers.csv_utils import (
    estimate_csv_size,
    prepare_csv_reader_options,
    prepare_csv_writer_options,
    count_csv_lines,
    validate_csv_structure,
    monitor_csv_operation,
    report_memory_usage
)

# Import JSON utilities
from pamola_core.utils.io_helpers.json_utils import (
    convert_numpy_types,
    validate_json_structure,
    merge_json_objects_in_memory,
    prettify_json,
    detect_array_or_object,
    extract_json_subset,
    prepare_json_writer_options
)

# Import format utilities
from pamola_core.utils.io_helpers.format_utils import (
    detect_format_from_extension,
    is_format_supported,
    get_format_extension,
    check_pyarrow_available,
    check_openpyxl_available,
    check_matplotlib_available,
    convert_dataframe_to_json,
    get_pandas_dtypes_info,
    get_dataframe_stats
)

# Import crypto utilities
from pamola_core.utils.io_helpers.crypto_utils import (
    encrypt_file_content,
    decrypt_file_content,
    decrypt_file,
    encrypt_file,
    encrypt_content_to_file,
    is_encrypted_data,
    get_encryption_metadata,
    safe_remove_temp_file
)

# Import Dask utilities
from pamola_core.utils.io_helpers.dask_utils import (
    is_dask_available,
    read_csv_in_chunks,
    read_full_csv,
    write_dataframe_to_csv,
    compute_dask_stats
)

# Import image utilities
from pamola_core.utils.io_helpers.image_utils import (
    save_plot,
    get_figure_format,
    prepare_figure_options,
    get_optimal_figure_size
)

# Define version
__version__ = '1.0.0'

# Define public API
__all__ = [
    # Directory utilities
    'ensure_directory',
    'get_timestamped_filename',
    'get_file_stats',
    'list_directory_contents',
    'clear_directory',
    'make_unique_path',

    # CSV utilities
    'estimate_csv_size',
    'prepare_csv_reader_options',
    'prepare_csv_writer_options',
    'count_csv_lines',
    'validate_csv_structure',
    'monitor_csv_operation',
    'report_memory_usage',

    # JSON utilities
    'convert_numpy_types',
    'validate_json_structure',
    'merge_json_objects_in_memory',
    'prettify_json',
    'detect_array_or_object',
    'extract_json_subset',
    'prepare_json_writer_options',

    # Format utilities
    'detect_format_from_extension',
    'is_format_supported',
    'get_format_extension',
    'check_pyarrow_available',
    'check_openpyxl_available',
    'check_matplotlib_available',
    'convert_dataframe_to_json',
    'get_pandas_dtypes_info',
    'get_dataframe_stats',

    # Crypto utilities
    'encrypt_file_content',
    'decrypt_file_content',
    'decrypt_file',
    'encrypt_file',
    'encrypt_content_to_file',
    'is_encrypted_data',
    'get_encryption_metadata',
    'safe_remove_temp_file',

    # Dask utilities
    'is_dask_available',
    'read_csv_in_chunks',
    'read_full_csv',
    'write_dataframe_to_csv',
    'compute_dask_stats',

    # Image utilities
    'save_plot',
    'get_figure_format',
    'prepare_figure_options',
    'get_optimal_figure_size'
]