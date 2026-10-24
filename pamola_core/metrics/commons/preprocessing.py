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
  - One-hot encoding of nominal categorical columns (ordinal only when declared)
  - Drop rows with missing values for robust metric computation

Design Principles:
  - Consistent and interpretable input formats for metrics
  - Ready for use in distance, nearest neighbor, and privacy metric calculations
  - Modular and reusable preprocessing logic

Dependencies:
  - pandas - DataFrame operations
  - scikit-learn - StandardScaler for the shared standardization
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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


def _semantic_type(s: pd.Series) -> str:
    """``nominal`` | ``numeric`` | ``datetime`` | ``text`` - decided from the dtype FIRST.

    A pandas categorical and a boolean are nominal whatever their labels look like (``1, 2, 10``
    are names, not measurements). An object column holding Python numbers is numeric (never
    parsed as dates).
    """
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
        return "nominal"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    inferred = pd.api.types.infer_dtype(s, skipna=True)
    if inferred == "boolean":
        return "nominal"
    if inferred in ("integer", "floating", "mixed-integer-float", "decimal"):
        return "numeric"
    if inferred in ("datetime", "datetime64", "date"):
        return "datetime"
    return "text"


def _plain(s: pd.Series) -> pd.Series:
    """The values of a column without a categorical wrapper."""
    return s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s


def _as_datetime(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if _semantic_type(s) in ("datetime", "text"):
        try:
            return pd.to_datetime(_plain(s), errors="raise")
        except Exception:  # noqa: BLE001
            return None
    return None


def _column_kind(a: pd.Series, b: pd.Series) -> str:
    """ONE representation for a column, decided from BOTH inputs.

    The dtype decides first: a pandas categorical or a boolean on either side makes the column
    ``categorical`` (nominal) even when its labels are numbers or dates. Otherwise ``datetime``
    if both sides are (or parse as) datetimes, ``numeric`` if both are numbers (or one is a
    number and the other text that is all numbers), else ``categorical``.
    """
    ta, tb = _semantic_type(a), _semantic_type(b)
    if "nominal" in (ta, tb):
        return "categorical"
    if ta == tb == "numeric":
        return "numeric"
    if {ta, tb} == {"numeric", "text"}:
        text = b if ta == "numeric" else a
        try:
            pd.to_numeric(text, errors="raise")
            return "numeric"
        except Exception:  # noqa: BLE001
            return "categorical"
    if ta in ("datetime", "text") and tb in ("datetime", "text") and "datetime" in (ta, tb) or ta == tb == "text":
        if _as_datetime(a) is not None and _as_datetime(b) is not None:
            return "datetime"
    return "categorical"


#: Weight of every one-hot column of a nominal categorical feature. Two records that differ in
#: such a feature differ in exactly two one-hot columns (the category of one and the category of
#: the other), so with this weight the Euclidean distance contributed by a mismatch is
#: ``sqrt(2 * (1/sqrt(2))**2) = 1``, whatever the categories are and however many there are.
#: A numeric feature standardized to unit variance contributes 1 per standard deviation, so a
#: mismatch of a nominal feature weighs as much as a one-standard-deviation gap of a numeric one.
NOMINAL_ONE_HOT_WEIGHT = 1.0 / float(np.sqrt(2.0))

#: Ceiling on the number of one-hot columns of all nominal features together; beyond it the
#: encoding would need more memory than the distance computation can bound.
MAX_ONE_HOT_FEATURES = 1000


def _format_label(value: object) -> str:
    """The label of one non-missing categorical value: the same text for ``10``, ``10.0`` and
    ``"10"``-typed numbers, ``True`` / ``False`` for booleans, stripped text otherwise."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value).strip()


def _labels(s: pd.Series) -> Tuple[pd.Series, np.ndarray]:
    """(labels, missing mask). The mask is taken BEFORE any string conversion, so a missing
    value never becomes the text ``'None'`` / ``'nan'`` and cannot equal a real label."""
    missing = s.isna().to_numpy()
    plain = _plain(s)
    labels = plain.where(~missing, other="").map(_format_label)
    return labels, missing


def _validate_ordinal_declaration(columns: Sequence[str], ordinal_columns: Optional[object]) -> dict:
    """``{column: declared order or None}``; rejects unknown names, duplicate levels, bad types."""
    from pamola_core.errors.exceptions import ValidationError

    if ordinal_columns is None:
        return {}
    if isinstance(ordinal_columns, str):
        raise ValidationError(
            "ordinal_columns must be a sequence of column names or a mapping "
            "{column: [categories in increasing order]}, not a single string."
        )
    if isinstance(ordinal_columns, dict):
        declared = {str(k) if not isinstance(k, str) else k: v for k, v in ordinal_columns.items()}
    else:
        declared = {name: None for name in ordinal_columns}
    unknown = [c for c in declared if c not in set(columns)]
    if unknown:
        raise ValidationError(
            f"ordinal_columns names unknown column(s) {unknown}; the data has {list(columns)}."
        )
    out = {}
    for name, order in declared.items():
        if order is None:
            out[name] = None
            continue
        if isinstance(order, str) or not hasattr(order, "__iter__"):
            raise ValidationError(f"Ordinal column '{name}': the order must be a list of categories.")
        norm = [_format_label(v) for v in order]
        dup = sorted({v for v in norm if norm.count(v) > 1})
        if not norm or dup:
            raise ValidationError(
                f"Ordinal column '{name}': the declared order "
                + ("is empty." if not norm else f"repeats {dup}; each category may appear once.")
            )
        out[name] = norm
    return out


def prepare_pair_for_distance_metrics(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    ordinal_columns: Optional[object] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert two datasets to numeric frames in ONE shared schema.

    The treatment of every column - numeric, datetime or categorical, and the code of every
    category - is decided from both inputs together *before* either is encoded. Deciding per
    input made identical records look different (``[10, 20]`` categorical in one input and
    integer in the other became ``[0, 1]`` against ``[10, 20]``) and unrelated category sets
    look identical. The two inputs must have the same, unique columns (any order); otherwise a
    ``ValidationError`` names the difference. Rows with a missing numeric / datetime value are
    dropped, as in ``prepare_data_for_distance_metrics``.

    Treatment of categorical columns
    --------------------------------
    The dtype decides first: a pandas categorical or a boolean column is categorical even if its
    labels are numbers or dates. A categorical column is **nominal** unless the caller declares
    it ordinal: it is one-hot encoded over the union of the categories of both inputs (one 0/1
    column per category, named ``<column>=<category>``), so every pair of different categories
    is equally far apart. Integer codes (``A=0, B=1, C=2``) would make ``d(A, C) = 2 * d(A, B)``
    although nominal values have no order. Missing values (``None``, ``NaN``, ``pd.NA``) form
    ONE extra category (``<column>=<missing>``) that is disjoint from every real label,
    including the texts ``'None'`` and ``'nan'``.

    The one-hot columns are recorded by POSITION in ``frame.attrs["nominal_positions"]`` (and
    by name in ``attrs["nominal_columns"]`` for reading); ``scale_pair_for_distance`` gives
    them the weight ``NOMINAL_ONE_HOT_WEIGHT`` so that one mismatch is one unit of Euclidean
    distance, commensurate with one standard deviation of a standardized numeric feature.
    Generated names never collide with an input column name.

    ``ordinal_columns`` declares columns whose categories have an order: a sequence of column
    names (the order is the numeric order of the labels if all are numbers, otherwise the
    alphabetical order) or a mapping ``{column: [categories in increasing order]}``. An ordinal
    column is encoded as the rank of its category, a numeric feature (a missing value drops the
    row, as for a numeric column). The whole declaration is validated before any encoding:
    unknown column names, repeated or missing categories, and columns that are numeric or
    datetime (already ordered) raise ``ValidationError``.

    Raises ``ResourceError`` when the nominal columns would need more than
    ``MAX_ONE_HOT_FEATURES`` one-hot columns together.
    """
    from pamola_core.errors.exceptions import ResourceError, ValidationError

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
    if len(set(cols)) != len(cols) or transformed_df.columns.duplicated().any():
        raise ValidationError("Distance metrics need unique column names in both datasets.")
    declared = _validate_ordinal_declaration(cols, ordinal_columns)

    # pass 1: the kind of every column, and the whole ordinal declaration checked against them
    kinds = {c: _column_kind(original_df[c], transformed_df[c]) for c in cols}
    for name in declared:
        if kinds[name] in ("numeric", "datetime"):
            raise ValidationError(
                f"Ordinal column '{name}' is {kinds[name]}: its values are already ordered. "
                "Declare only categorical columns as ordinal."
            )

    a_out: list = []
    b_out: list = []
    reserved = set(cols)  # generated names must never equal an input column name
    nominal_names: list = []
    width = 0
    for col in cols:
        sa, sb = original_df[col], transformed_df[col]
        kind = kinds[col]
        if kind == "datetime":
            da, db = _as_datetime(sa), _as_datetime(sb)
            a_out.append((col, da.astype("int64").where(~da.isna()).astype(float)))
            b_out.append((col, db.astype("int64").where(~db.isna()).astype(float)))
        elif kind == "numeric":
            a_out.append((col, pd.to_numeric(_plain(sa).astype(float) if sa.dtype == bool else _plain(sa), errors="coerce").astype(float)))
            b_out.append((col, pd.to_numeric(_plain(sb).astype(float) if sb.dtype == bool else _plain(sb), errors="coerce").astype(float)))
        else:
            la, ma = _labels(sa)
            lb, mb = _labels(sb)
            levels = sorted(set(la[~ma]) | set(lb[~mb]))
            if col in declared:
                order = declared[col]
                if order is None:  # a bare name: numbers sort as numbers, text alphabetically
                    try:
                        order = sorted(levels, key=float)
                    except (TypeError, ValueError):
                        order = list(levels)
                missing_levels = [v for v in levels if v not in set(order)]
                if missing_levels:
                    raise ValidationError(
                        f"Ordinal column '{col}': the declared order does not list "
                        f"{len(missing_levels)} value(s) present in the data."
                    )
                rank = {v: float(i) for i, v in enumerate(order)}
                ra = la.map(rank).astype(float).where(~ma)
                rb = lb.map(rank).astype(float).where(~mb)
                a_out.append((col, ra))
                b_out.append((col, rb))
                continue
            has_missing = bool(ma.any() or mb.any())
            width += len(levels) + int(has_missing)
            if width > MAX_ONE_HOT_FEATURES:
                raise ResourceError(
                    operation="distance metrics",
                    reason=(
                        f"The nominal columns need more than {MAX_ONE_HOT_FEATURES} one-hot "
                        f"features (column '{col}' has {len(levels)} categories). Drop or "
                        "generalize high-cardinality columns, or declare the column ordinal "
                        "with `ordinal_columns` if its categories have an order."
                    ),
                )
            entries = [(f"{col}={lv}", (la == lv) & ~ma, (lb == lv) & ~mb) for lv in levels]
            if has_missing:
                entries.append((f"{col}=<missing>", pd.Series(ma, index=la.index), pd.Series(mb, index=lb.index)))
            for k, (name, ea, eb) in enumerate(entries):
                unique_name = name
                while unique_name in reserved:
                    unique_name = f"{name}#{k}" if unique_name == name else f"{unique_name}#"
                reserved.add(unique_name)
                nominal_names.append(unique_name)
                a_out.append((unique_name, ea.astype(float)))
                b_out.append((unique_name, eb.astype(float)))
    out_a = pd.concat([s.rename(n) for n, s in a_out], axis=1)
    out_b = pd.concat([s.rename(n) for n, s in b_out], axis=1)
    out_a.index, out_b.index = original_df.index, transformed_df.index
    out_a, out_b = out_a.dropna(), out_b.dropna()
    positions = [i for i, (n, _) in enumerate(a_out) if n in set(nominal_names)]
    for frame in (out_a, out_b):
        frame.attrs["nominal_columns"] = list(nominal_names)
        frame.attrs["nominal_positions"] = list(positions)
    return out_a, out_b


def scale_pair_for_distance(
    original: pd.DataFrame, transformed: pd.DataFrame, normalize_features: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn the frames of ``prepare_pair_for_distance_metrics`` into the arrays the distance
    backends use.

    Numeric, datetime and ordinal columns are standardized with ONE ``StandardScaler`` fitted on
    the original (when ``normalize_features``), or used raw. One-hot columns of nominal
    features (identified by POSITION, never by name) are not standardized (their variance
    depends on category frequency, which would make a mismatch of a rare category count more
    than a mismatch of a common one); they are multiplied by ``NOMINAL_ONE_HOT_WEIGHT``, so
    each mismatch is one unit of Euclidean distance. All backends (exact, block-wise, FAISS)
    receive the same arrays.
    """
    from sklearn.preprocessing import StandardScaler
    from pamola_core.errors.exceptions import ValidationError

    if len(set(original.columns)) != len(original.columns):
        raise ValidationError("Feature names are not unique; the feature types cannot be told apart.")
    nominal = set(original.attrs.get("nominal_positions", []))
    n_cols = original.shape[1]
    numeric_idx = [i for i in range(n_cols) if i not in nominal]
    nominal_idx = sorted(nominal)
    a = np.zeros((len(original), n_cols), dtype=float)
    b = np.zeros((len(transformed), n_cols), dtype=float)
    if numeric_idx:
        num_a = original.iloc[:, numeric_idx].values.astype(float)
        num_b = transformed.iloc[:, numeric_idx].values.astype(float)
        if normalize_features:
            scaler = StandardScaler().fit(num_a)
            num_a, num_b = scaler.transform(num_a), scaler.transform(num_b)
        a[:, numeric_idx], b[:, numeric_idx] = num_a, num_b
    if nominal_idx:
        a[:, nominal_idx] = original.iloc[:, nominal_idx].values * NOMINAL_ONE_HOT_WEIGHT
        b[:, nominal_idx] = transformed.iloc[:, nominal_idx].values * NOMINAL_ONE_HOT_WEIGHT
    return a, b


def mahalanobis_inverse_covariance(original_features: np.ndarray) -> np.ndarray:
    """``VI`` for the Mahalanobis distance: the pseudo-inverse of the covariance of the
    ORIGINAL's prepared features. One-hot columns make the covariance singular (the columns of
    one feature sum to a constant), so a plain inverse does not exist; the Moore-Penrose
    pseudo-inverse is the distance in the span of the data."""
    x = np.asarray(original_features, dtype=float)
    cov = np.atleast_2d(np.cov(x, rowvar=False)) if x.shape[0] > 1 else np.zeros((x.shape[1], x.shape[1]))
    return np.linalg.pinv(cov, hermitian=True)


def prepare_data_for_distance_metrics(
    df: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
    ordinal_columns: Optional[object] = None,
) -> pd.DataFrame:
    """
    Preprocess a DataFrame for distance-based metric calculations, in the same nominal-aware
    schema as ``prepare_pair_for_distance_metrics`` (nominal categoricals one-hot encoded,
    ordinal only when declared, missing categories as one tagged category).

    Distances between two datasets are only meaningful if both are encoded in the same space,
    so pass the other dataset as ``reference``: the categories are then those of both datasets
    and calling this function once per dataset with the other as ``reference`` yields
    comparable frames. Without a reference the dataset is encoded with its own categories.
    Apply ``scale_pair_for_distance`` to obtain the weighted arrays.

    Returns
    --------
    pd.DataFrame
        Numeric frame; ``frame.attrs["nominal_positions"]`` marks the one-hot columns.
    """
    if reference is not None:
        return prepare_pair_for_distance_metrics(df, reference, ordinal_columns)[0]
    return prepare_pair_for_distance_metrics(df, df, ordinal_columns)[0]
