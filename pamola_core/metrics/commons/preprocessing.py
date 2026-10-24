"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
Module:        Metric Preprocessing Utilities
Package:       pamola_core.metrics.commons.preprocessing
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides preprocessing utilities for transforming input datasets into numeric format
  suitable for distance-based metric calculations. Handles boolean, datetime, string,
  and categorical columns, ensuring compatibility for privacy, utility, and fidelity metrics.

Key Features:
  - Convert boolean columns to integers
  - Convert datetime columns to numeric timestamps
  - Ordinal encoding for string and categorical columns
  - Drop rows with missing values for robust metric computation

Design Principles:
  - Consistent and interpretable input formats for metrics
  - Ready for use in distance, nearest neighbor, and privacy metric calculations
  - Modular and reusable preprocessing logic

Dependencies:
  - pandas - DataFrame operations
  - scikit-learn - OrdinalEncoder for categorical encoding
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.exceptions import DataError


def encode_pair_with_dummies(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    exclude: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode two datasets into the SAME feature space.

    ``pd.get_dummies`` applied to each dataset separately creates a column per
    category *present in that dataset*, so a transformed frame that lost (or gained)
    a category gets a different set of columns, and a model trained on one is
    scored on a misaligned other. Here the categories are those of both datasets
    together: both results have the same columns in the same order. A category
    absent from one dataset is an all-False column there; there is no unseen
    category. Columns named in ``exclude`` (the target) are passed through
    unencoded. Row labels of each frame are preserved.
    """
    keep = [c for c in exclude if c in original_df.columns and c in transformed_df.columns]
    n_orig = len(original_df)
    combined = pd.concat(
        [original_df.reset_index(drop=True), transformed_df.reset_index(drop=True)],
        ignore_index=True,
    )
    encoded = pd.get_dummies(combined.drop(columns=keep))
    for col in keep:
        encoded[col] = combined[col]
    original_out = encoded.iloc[:n_orig].copy()
    transformed_out = encoded.iloc[n_orig:].copy()
    original_out.index = original_df.index
    transformed_out.index = transformed_df.index
    return original_out, transformed_out


def _is_numeric_like(s: pd.Series) -> bool:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return _is_numeric_like(pd.Series(s.cat.categories))
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _plain(s: pd.Series) -> pd.Series:
    """The values of a column without a categorical wrapper."""
    return s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s


def _as_datetime(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        try:
            return pd.to_datetime(_plain(s), errors="raise")
        except Exception:  # noqa: BLE001
            return None
    return None


def _column_kind(a: pd.Series, b: pd.Series) -> str:
    """ONE representation for a column, decided from BOTH inputs.

    ``datetime`` if both sides are (or parse as) datetimes, ``numeric`` if both are numbers
    (a categorical of numbers counts as numbers), else ``categorical`` - and then both sides
    are compared as text, so ``10`` in an integer column and ``10`` in a categorical column
    are the same value in the same space.
    """
    if _as_datetime(a) is not None and _as_datetime(b) is not None and not (
        _is_numeric_like(a) or _is_numeric_like(b)
    ):
        return "datetime"
    if _is_numeric_like(a) and _is_numeric_like(b):
        return "numeric"
    # a number column against text that is all numbers: numbers
    for num, other in ((a, b), (b, a)):
        if _is_numeric_like(num) and not _is_numeric_like(other):
            try:
                pd.to_numeric(_plain(other), errors="raise")
                return "numeric"
            except Exception:  # noqa: BLE001
                pass
    return "categorical"


def prepare_pair_for_distance_metrics(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert two datasets to numeric frames in ONE shared schema.

    The treatment of every column - numeric, datetime or categorical, and the code of every
    category - is decided from both inputs together *before* either is encoded. Deciding per
    input made identical records look different (``[10, 20]`` categorical in one input and
    integer in the other became ``[0, 1]`` against ``[10, 20]``) and unrelated category sets
    look identical. The two inputs must have the same columns (any order); otherwise a
    ``ValidationError`` names the difference. Rows with a missing numeric / datetime value are
    dropped, as in ``prepare_data_for_distance_metrics``.
    """
    from pamola_core.errors.exceptions import ValidationError

    for df in (original_df, transformed_df):
        if df.empty:
            raise DataError(message="Input DataFrame is empty", error_code=ErrorCode.DATA_EMPTY)
    only_a = [c for c in original_df.columns if c not in set(transformed_df.columns)]
    only_b = [c for c in transformed_df.columns if c not in set(original_df.columns)]
    if only_a or only_b:
        raise ValidationError(
            "The two datasets have different columns: only in the original "
            f"{only_a}, only in the transformed {only_b}. Distance metrics need the same "
            "features on both sides."
        )
    cols = list(original_df.columns)
    a_out, b_out = {}, {}
    for col in cols:
        sa, sb = original_df[col], transformed_df[col]
        kind = _column_kind(sa, sb)
        if kind == "datetime":
            da, db = _as_datetime(sa), _as_datetime(sb)
            a_out[col] = da.astype("int64").where(~da.isna()).astype(float)
            b_out[col] = db.astype("int64").where(~db.isna()).astype(float)
        elif kind == "numeric":
            a_out[col] = pd.to_numeric(_plain(sa).astype(float) if sa.dtype == bool else _plain(sa), errors="coerce").astype(float)
            b_out[col] = pd.to_numeric(_plain(sb).astype(float) if sb.dtype == bool else _plain(sb), errors="coerce").astype(float)
        else:
            ta = _plain(sa).astype(str).str.strip()
            tb = _plain(sb).astype(str).str.strip()
            levels = sorted(set(ta.unique()) | set(tb.unique()))
            index = {v: float(i) for i, v in enumerate(levels)}
            a_out[col] = ta.map(index).astype(float)
            b_out[col] = tb.map(index).astype(float)
    out_a = pd.DataFrame(a_out, index=original_df.index)[cols].dropna()
    out_b = pd.DataFrame(b_out, index=transformed_df.index)[cols].dropna()
    return out_a, out_b


def prepare_data_for_distance_metrics(
    df: pd.DataFrame, reference: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Preprocess the input DataFrame for distance-based metric calculations.
    Converts boolean, datetime, string, and categorical columns to numeric.

    Distances between two datasets are only meaningful if both are encoded in the
    same space. Encoding each one on its own maps *unrelated* category sets to the
    same codes (``['alice', 'bob']`` and ``['xavier', 'zoe']`` both become 0 and 1),
    so records that share nothing look identical and a distance metric reports 0.
    Pass the other dataset as ``reference``: each categorical column is then encoded
    from the union of the categories of both datasets (sorted, so the same category
    gets the same code in both), and calling this function once per dataset with
    the other as ``reference`` yields comparable frames. There is no unseen-category
    case: every category of either dataset is in the encoding.

    Parameters
    -----------
    df : pd.DataFrame
        Input dataset. Each row is a record, columns are features.
    reference : pd.DataFrame, optional
        The dataset ``df`` will be compared with. Only its categorical values are
        read, to complete the encoding.

    Returns
    --------
    pd.DataFrame
        Preprocessed DataFrame with all features converted to numeric types.
    """
    if reference is not None:
        # a reference means "compare with it": one shared schema for both
        return prepare_pair_for_distance_metrics(df, reference)[0]

    if df.empty:
        raise DataError(
            message="Input DataFrame is empty",
            error_code=ErrorCode.DATA_EMPTY,
        )

    df_converted = df.copy()

    # 1. Convert boolean columns to integers
    bool_cols = df_converted.select_dtypes(include=["bool"]).columns.tolist()
    df_converted[bool_cols] = df_converted[bool_cols].astype(int)

    # 2. Convert datetime columns (including convertible object columns) to int64 (nanoseconds)
    datetime_cols = []
    for col in df_converted.columns:
        if pd.api.types.is_datetime64_any_dtype(df_converted[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_object_dtype(df_converted[col]):
            try:
                converted = pd.to_datetime(df_converted[col], errors="raise")
                df_converted[col] = converted
                datetime_cols.append(col)
            except Exception:
                continue

    for col in datetime_cols:
        df_converted[col] = df_converted[col].astype(
            "int64"
        )  # timestamp in nanoseconds

    # 3. Convert string and categorical columns to ordinal encoded numeric values
    string_cols = df_converted.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in string_cols:
        df_converted[col] = df_converted[col].astype(str)
        df_converted[col] = df_converted[col].str.strip()

    if string_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if reference is not None:
            # Declare the categories as the union of both datasets (sorted, as
            # OrdinalEncoder does) so a category has one code in both.
            categories = []
            for col in string_cols:
                values = set(df_converted[col].unique())
                if col in reference.columns and (
                    pd.api.types.is_object_dtype(reference[col])
                    or isinstance(reference[col].dtype, pd.CategoricalDtype)
                    or pd.api.types.is_string_dtype(reference[col])
                ):
                    values |= set(reference[col].astype(str).str.strip().unique())
                categories.append(sorted(values))
            encoder = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df_converted[string_cols] = encoder.fit_transform(df_converted[string_cols])
        else:
            df_converted[string_cols] = encoder.fit_transform(df_converted[string_cols])

    # 4. Drop rows with any NaN values
    df_converted = df_converted.dropna()

    numeric_cols = df_converted.select_dtypes(include=[np.number]).columns
    df_converted[numeric_cols] = df_converted[numeric_cols].fillna(
        df_converted[numeric_cols].median()
    )

    return df_converted
