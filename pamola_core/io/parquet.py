"""
    PAMOLA Core - IO Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all IO operations in the PAMOLA Core
    library. It defines the standard contracts that all format-specific handlers must implement.

    Key features:
    - Implementation of DataReader protocol for standardized Parquet reading
    - Implementation of DataWriter protocol for standardized Parquet writing

    This module serves as the foundation for all IO operations in PAMOLA Core, ensuring
    consistency across different data formats and storage methods.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:
        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from typing import Any
from .base import data_read, data_write

class DataParquet:
    def __init__(self): ...

    def read(self, path: str, **kwargs) -> Any:
        return pd.read_parquet(path = path
                               , engine = kwargs.get('engine') if 'engine' in kwargs else 'auto'
                               , columns = kwargs.get('columns')
                               , storage_options = kwargs.get('storage_options')
                               , dtype_backend=kwargs.get('dtype_backend') if 'dtype_backend' in kwargs else 'numpy_nullable'
                               , filesystem = kwargs.get('filesystem')
                               , filters = kwargs.get('filters')
                               )

    def write(self, data: Any, path: str, **kwargs) -> Any:
        return data.to_parquet(path = path
                               , engine = kwargs.get('engine') if 'engine' in kwargs else 'auto'
                               , compression = kwargs.get('compression') if 'compression' in kwargs else 'snappy'
                               , index = kwargs.get('index')
                               , partition_cols = kwargs.get('partition_cols')
                               , storage_options = kwargs.get('storage_options')
                               )

def read_parquet(path: str, **kwargs) -> Any:
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like object implementing
        a binary ``read()`` function. The string could be a URL. Valid URL schemes include http,
        ftp, s3, gs, and file.

    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option ``io.parquet.engine`` is used. The default
        ``io.parquet.engine`` behavior is to try 'pyarrow', falling back to 'fastparquet' if 'pyarrow'
        is unavailable.
        When using the ``'pyarrow'`` engine and no storage options are provided and a filesystem is
        implemented by both ``pyarrow.fs`` and ``fsspec`` (e.g. "s3://"), then the ``pyarrow.fs``
        filesystem is attempted first. Use the filesystem keyword with an instantiated fsspec
        filesystem if you wish to use its implementation.

    columns : list, default=None
        If not None, only these columns will be read from the file.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`. Behaviour is as follows:
        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame` (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype` DataFrame.

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented for ``engine="pyarrow"``.

    filters : List[Tuple] or List[List[Tuple]], default None
        To filter out data. Filter syntax: [[(column, op, val), ...],...] where op is [==, =, >, >=, <,
        <=, !=, in, not in] The innermost tuples are transposed into a set of filters applied through an
        `AND` operation. The outer list combines these sets of filters through an `OR` operation. A single
        list of tuples can also be used, meaning that no `OR` operation between set of filters is to be
        conducted.

    Returns
    -------
    DataFrame
    """
    return data_read(DataParquet(), path, **kwargs)

def write_parquet(df: pd.DataFrame, path: str, **kwargs) -> Any:
    """
    Write a DataFrame to the binary parquet format.

    This function writes the dataframe as a `parquet file`. You can choose different parquet
    backends, and have the option of compression.

    Parameters
    ----------
    df : DataFrame
        The input data to be processed.

    path : str, path object, file-like object, or None, default None
        String, path object (implementing ``os.PathLike[str]``), or file-like object implementing
        a binary ``write()`` function. If None, the result is returned as bytes. If a string or path,
        it will be used as Root Directory path when writing a partitioned dataset.

    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option ``io.parquet.engine`` is used. The default
        ``io.parquet.engine`` behavior is to try 'pyarrow', falling back to 'fastparquet' if 'pyarrow'
        is unavailable.

    compression : str or None, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression. Supported options: 'snappy',
        'gzip', 'brotli', 'lz4', 'zstd'.

    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If ``False``, they will not
        be written to the file. If ``None``, similar to ``True`` the dataframe's index(es) will be saved.
        However, instead of being saved as values, the RangeIndex will be stored as a range in the
        metadata so it doesn't require much space and is faster. Other indexes will be included as
        columns in the file output.

    partition_cols : list, optional, default None
        Column names by which to partition the dataset. Columns are partitioned in the order they are
        given. Must be None if path is not a string.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    Returns
    -------
    bytes if no path argument is provided else None
    """
    return data_write(DataParquet(), df, path, **kwargs)
