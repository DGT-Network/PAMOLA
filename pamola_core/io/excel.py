"""
    PAMOLA Core - IO Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all IO operations in the PAMOLA Core
    library. It defines the standard contracts that all format-specific handlers must implement.

    Key features:
    - Implementation of DataReader protocol for standardized Excel reading
    - Implementation of DataWriter protocol for standardized Excel writing

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

class DataExcel:
    def __init__(self): ...

    def read(self, path: str, **kwargs) -> Any:
        return pd.read_excel(io = path
                             , sheet_name = kwargs.get('sheet_name') if 'sheet_name' in kwargs else 0
                             , header=kwargs.get('header') if 'header' in kwargs else 0
                             , names = kwargs.get('names')
                             , index_col = kwargs.get('index_col')
                             , usecols = kwargs.get('usecols')
                             , dtype = kwargs.get('dtype')
                             , engine = kwargs.get('engine')
                             , converters = kwargs.get('converters')
                             , true_values = kwargs.get('true_values')
                             , false_values = kwargs.get('false_values')
                             , skiprows = kwargs.get('skiprows')
                             , nrows = kwargs.get('nrows')
                             , na_values = kwargs.get('na_values')
                             , keep_default_na = kwargs.get('keep_default_na') if 'keep_default_na' in kwargs else True
                             , na_filter = kwargs.get('na_filter') if 'na_filter' in kwargs else True
                             , verbose=kwargs.get('verbose') if 'verbose' in kwargs else False
                             , parse_dates=kwargs.get('parse_dates') if 'parse_dates' in kwargs else False
                             , date_format = kwargs.get('date_format')
                             , thousands = kwargs.get('thousands')
                             , decimal = kwargs.get('decimal') if 'decimal' in kwargs else '.'
                             , comment = kwargs.get('comment')
                             , skipfooter = kwargs.get('skipfooter') if 'skipfooter' in kwargs else 0
                             , storage_options = kwargs.get('storage_options')
                             , dtype_backend = kwargs.get('dtype_backend') if 'dtype_backend' in kwargs else 'numpy_nullable'
                             )

    def write(self, data: Any, path: str, **kwargs) -> Any:
        data.to_excel(excel_writer = path
                      , sheet_name = kwargs.get('sheet_name') if 'sheet_name' in kwargs else 'Sheet1'
                      , na_rep = kwargs.get('na_rep') if 'na_rep' in kwargs else ''
                      , float_format = kwargs.get('float_format')
                      , columns = kwargs.get('columns')
                      , header = kwargs.get('header') if 'header' in kwargs else True
                      , index = kwargs.get('index') if 'index' in kwargs else True
                      , index_label = kwargs.get('index_label')
                      , startrow = kwargs.get('startrow') if 'startrow' in kwargs else 0
                      , startcol = kwargs.get('startcol') if 'startcol' in kwargs else 0
                      , engine = kwargs.get('engine')
                      , merge_cells = kwargs.get('merge_cells') if 'merge_cells' in kwargs else True
                      , inf_rep = kwargs.get('inf_rep') if 'inf_rep' in kwargs else 'inf'
                      , freeze_panes = kwargs.get('freeze_panes')
                      , storage_options = kwargs.get('storage_options')
                      )

def read_excel(path: str, **kwargs) -> Any:
    """
    Read an Excel file into a ``pandas`` ``DataFrame``.

    Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions read from
    a local filesystem or URL. Supports an option to read a single sheet or a list of sheets.

    Parameters
    ----------
    path : str, ExcelFile, xlrd.Book, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, and file.

    sheet_name : str, int, list, or None, default 0
        Strings are used for sheet names. Integers are used in zero-indexed sheet positions (chart
        sheets do not count as a sheet position). Lists of strings/integers are used to request
        multiple sheets. When ``None``, will return a dictionary containing DataFrames for each sheet.
        Available cases:
        * Defaults to ``0``: 1st sheet as a `DataFrame`
        * ``1``: 2nd sheet as a `DataFrame`
        * ``"Sheet1"``: Load sheet with name "Sheet1"
        * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5" as a dict of `DataFrame`
        * ``None``: Returns a dictionary containing DataFrames for each sheet..

    header : int, list of int, default 0
        Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is
        passed those row positions will be combined into a ``MultiIndex``. Use None if there is no header.

    names : array-like, default None
        List of column names to use. If file contains no header row, then you should explicitly pass
        header=None.

    index_col : int, str, list of int, default None
        Column (0-indexed) to use as the row labels of the DataFrame. Pass None if there is no such column.
        If a list is passed, those columns will be combined into a ``MultiIndex``.  If a subset of data is
        selected with ``usecols``, index_col is based on the subset.
        Missing values will be forward filled to allow roundtripping with ``to_excel`` for
        ``merged_cells=True``. To avoid forward filling the missing values use ``set_index`` after reading
        the data instead of ``index_col``.

    usecols : str, list-like, or callable, default None
        * If None, then parse all columns.
        * If str, then indicates comma separated list of Excel column letters and column ranges (e.g.
          "A:E" or "A,C,E:F"). Ranges are inclusive of both sides.
        * If list of int, then indicates list of column numbers to be parsed (0-indexed).
        * If list of string, then indicates list of column names to be parsed.
        * If callable, then evaluate each column name against it and parse the column if the callable
          returns ``True``.
        Returns a subset of the columns according to behavior above.

    dtype : Type name or dict of column -> type, default None
        Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32}} Use ``object`` to preserve
        data as stored in Excel and not interpret dtype, which will necessarily result in ``object`` dtype.
        If converters are specified, they will be applied INSTEAD of dtype conversion. If you use ``None``,
        it will infer the dtype of each column based on the data.

    engine : {{'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}}, default None
        If io is not a buffer or path, this must be set to identify io. Engine compatibility :
        - ``openpyxl`` supports newer Excel file formats.
        - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb) and OpenDocument (.ods) file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.
        - ``xlrd`` supports old-style Excel files (.xls).
        When ``engine=None``, the following logic will be used to determine the engine:
        - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt), then `odf` will be used.
        - Otherwise if ``path_or_buffer`` is an xls format, ``xlrd`` will be used.
        - Otherwise if ``path_or_buffer`` is in xlsb format, ``pyxlsb`` will be used.
        - Otherwise ``openpyxl`` will be used.

    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can either be integers or column
        labels, values are functions that take one input argument, the Excel cell content, and return the
        transformed content.

    true_values : list, default None
        Values to consider as True.

    false_values : list, default None
        Values to consider as False.

    skiprows : list-like, int, or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. If
        callable, the callable function will be evaluated against the row indices, returning True if the
        row should be skipped and False otherwise. An example of a valid callable argument would be
        ``lambda x: x in [0, 2]``.

    nrows : int, default None
        Number of rows to parse.

    na_values : scalar, str, list-like, or dict, default None
        Additional strings to recognize as NA/NaN. If dict passed, specific per-column NA values. By default
        the following values are interpreted as NaN: ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
        ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘None’, ‘n/a’, ‘nan’, ‘null’.

    keep_default_na : bool, default True
        Whether or not to include the default NaN values when parsing the data. Depending on whether
        ``na_values`` is passed in, the behavior is as follows:
        * If ``keep_default_na`` is True, and ``na_values`` are specified, ``na_values`` is appended to
          the default NaN values used for parsing.
        * If ``keep_default_na`` is True, and ``na_values`` are not specified, only the default NaN values
          are used for parsing.
        * If ``keep_default_na`` is False, and ``na_values`` are specified, only the NaN values specified
          ``na_values`` are used for parsing.
        * If ``keep_default_na`` is False, and ``na_values`` are not specified, no strings will be parsed
          as NaN.
        Note that if `na_filter` is passed in as False, the ``keep_default_na`` and ``na_values`` parameters
        will be ignored.

    na_filter : bool, default True
        Detect missing value markers (empty strings and the value of na_values). In data without any NAs,
        passing ``na_filter=False`` can improve the performance of reading a large file.

    verbose : bool, default False
        Indicate number of NA values placed in non-numeric columns.

    parse_dates : bool, list-like, or dict, default False
        The behavior is as follows:
        * ``bool``. If True -> try parsing the index.
        * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate
          date column.
        * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as a single date
          column.
        * ``dict``, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call result 'foo'
        If a column or index contains an unparsable date, the entire column or index will be returned
        unaltered as an object data type. If you don`t want to parse some cells as date just change their
        type in Excel to "Text". For non-standard datetime parsing, use ``pd.to_datetime`` after
        ``pd.read_excel``.
        Note: A fast-path exists for iso8601-formatted dates.

    date_format : str or dict of column -> format, default ``None``
       If used in conjunction with ``parse_dates``, will parse dates according to this format. For anything
       more complex, please read in as ``object`` and then apply :func:`to_datetime` as-needed.
    
    thousands : str, default None
        Thousands separator for parsing string columns to numeric.  Note that this parameter is only necessary
        for columns stored as TEXT in Excel, any numeric columns will automatically be parsed, regardless of
        display format.

    decimal : str, default '.'
        Character to recognize as decimal point for parsing string columns to numeric. Note that this parameter
        is only necessary for columns stored as TEXT in Excel, any numeric columns will automatically be parsed,
        regardless of display format.(e.g. use ',' for European data).

    comment : str, default None
        Comments out remainder of line. Pass a character or characters to this argument to indicate comments in
        the input file. Any data between the comment string and the end of the current line is ignored.

    skipfooter : int, default 0
        Rows at the end to skip (0-indexed).

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default ‘numpy_nullable’
        Back-end data type applied to the resultant :class:`DataFrame`. If not specified, the default
        behavior is to not use nullable data types. If specified, the behavior is as follows:
        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype` :class:`DataFrame`

    Returns
    -------
    DataFrame or dict of DataFrames
        DataFrame from the passed in Excel file.
    """
    return data_read(DataExcel(), path, **kwargs)

def write_excel(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Write object to an Excel sheet.

    To write a single object to an Excel .xlsx file it is only necessary to specify a target file name.
    To write to multiple sheets it is necessary to create an `ExcelWriter` object with a target file name,
    and specify a sheet in the file to write to.

    Multiple sheets may be written to by specifying unique `sheet_name`. With all data written to the file
    it is necessary to save the changes. Note that creating an `ExcelWriter` object with a file name that
    already exists will result in the contents of the existing file being erased.

    Parameters
    ----------
    df : DataFrame
        The input data to be processed.

    path : path-like, file-like, or ExcelWriter object
        File path or existing ExcelWriter.

    sheet_name : str, default 'Sheet1'
        Name of sheet which will contain DataFrame.

    na_rep : str, default ''
        Missing data representation.

    float_format : str, optional
        Format string for floating point numbers. For example ``float_format="%.2f"`` will format 0.1234
        to 0.12.

    columns : sequence or list of str, optional
        Columns to write.

    header : bool or list of str, default True
        Write out the column names. If a list of string is given it is assumed to be aliases for the
        column names.

    index : bool, default True
        Write row names (index).

    index_label : str or sequence, optional
        Column label for index column(s) if desired. If not specified, and `header` and `index` are True,
        then the index names are used. A sequence should be given if the DataFrame uses MultiIndex.

    startrow : int, default 0
        Upper left cell row to dump data frame.

    startcol : int, default 0
        Upper left cell column to dump data frame.

    engine : str, optional
        Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this via the options
        ``io.excel.xlsx.writer`` or ``io.excel.xlsm.writer``.

    merge_cells : bool or 'columns', default False
        If True, write MultiIndex index and columns as merged cells.
        If 'columns', merge MultiIndex column cells only.

    inf_rep : str, default 'inf'
        Representation for infinity (there is no native representation for infinity in Excel).

    freeze_panes : tuple of int (length 2), optional
        Specifies the one-based bottommost row and rightmost column that is to be frozen.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.
    """
    data_write(DataExcel(), df, path, **kwargs)
