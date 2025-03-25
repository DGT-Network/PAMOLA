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

    Module: JavaScript Object Notation (json)
    --------------------------


    NOTE: This module requires `pandas`.

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd

def read(
    path_or_buf,
    orient=None,
    nrows=None
):
    """
        Convert a JSON string to pandas object

        Parameters
        ----------
        path_or_buf : a valid JSON str, path object or file-like object
            Input data.

        orient : str, optional, [split,records,index,columns,values,table]
            Input data.

        nrows : int, optional
            Input data.

        Returns
        -------
        DataFrame or TextFileReader
    """
    return pd.read_json(path_or_buf=path_or_buf,orient=orient,nrows=nrows,lines=False)

def write(
    df,
    path_or_buf=None,
    orient=None,
    date_format=None,
    double_precision=10,
    lines=False
):
    """
        Convert the object to a JSON string

        Parameters
        ----------
        df : DataFrame
            Input data.

        path_or_buf : str, path object, file-like object, or None, default None
            Input data.

        orient : str, optional, [split,records,index,columns,values,table]
            Input data.

        date_format : {None, ‘epoch’, ‘iso’}
            Input data.

        double_precision : int, default 10
            Input data.

        lines : bool, default False
            Input data.

        Returns
        -------
        None or str
    """
    return df.to_json(path_or_buf=path_or_buf,orient=orient,date_format=date_format,double_precision=double_precision,lines=lines)
