"""
    PAMOLA.CORE - I/O File Utilities
    --------------------------------------
    This file is part of the PAMOLA ecosystem, a comprehensive suite for
    privacy-enhancing technologies. PAMOLA.CORE serves as the open-source
    foundation for secure, structured, and scalable file operations.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:

        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Module: Comma-separated values (csv)
    --------------------------


    NOTE: This module requires `pandas`.

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd

def read(
    filepath_or_buffer,
    sep=",",
    usecols=None,
    skiprows=None,
    nrows=None
):
    """
        Read a comma-separated values (csv) file into DataFrame

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            Input data.

        sep : str, default ‘,’
            Input data.

        usecols : Sequence of Hashable or Callable, optional
            Input data.

        skiprows : int, list of int or Callable, optional
            Input data.

        nrows : int, optional
            Input data.

        Returns
        -------
        DataFrame or TextFileReader
    """
    return pd.read_csv(filepath_or_buffer=filepath_or_buffer,sep=sep,usecols=usecols,skiprows=skiprows,nrows=nrows)

def write(
    df,
    path_or_buf=None,
    sep=",",
    columns=None
):
    """
        Write object to a comma-separated values (csv) file

        Parameters
        ----------
        df : DataFrame
            Input data.

        path_or_buf : str, path object, file-like object, or None, default None
            Input data.

        sep : str, default ‘,’
            Input data.

        columns : sequence, optional
            Input data.

        Returns
        -------
        None or str
    """
    return df.to_csv(path_or_buf=path_or_buf,sep=sep,columns=columns,index=False)
