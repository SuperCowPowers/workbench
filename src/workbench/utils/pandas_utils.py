"""Utility/helper methods for Pandas dataframe operations"""

import pandas as pd
from pandas.errors import ParserError
import numpy as np
import json
from io import StringIO
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union

# Workbench Logger
log = logging.getLogger("workbench")


class DataFrameBuilder:
    def __init__(self):
        self.rows = []

    def add_row(self, row_data: dict):
        """Adds a new row to the DataFrame.

        Parameters:
            row_data (dict): Key-value pairs representing column names and their values for the row.
        """
        self.rows.append(row_data)

    def build(self) -> pd.DataFrame:
        """Constructs the DataFrame from the accumulated rows.

        Returns:
            A pandas DataFrame containing all the added rows.
        """
        return pd.DataFrame(self.rows)


def df_hash(df: pd.DataFrame) -> str:
    """Calculate a hash for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to hash.

    Returns:
        str: Hash of the DataFrame.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def convert_categorical_types(input_df: pd.DataFrame, category_mappings: dict) -> pd.DataFrame:
    """
    Converts appropriate columns to categorical type with consistent mappings.

    Args:
        input_df (pd.DataFrame): The DataFrame to process.
        category_mappings (dict): Existing category mappings.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to categorical type.
    """
    df = input_df.copy()
    for col, categories in category_mappings.items():
        if col in df.columns:
            log.info(f"Applying categorical mapping for {col}")
            df[col] = pd.Categorical(df[col], categories=categories)
    return df


def dataframe_delta(func_that_returns_df, previous_hash: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """Generalized method to compute the delta for any function that returns a DataFrame.

    Args:
        func_that_returns_df (callable): A function that returns a DataFrame.
        previous_hash (str): Optional hash of the previous DataFrame for change detection.

    Returns:
        tuple: (DataFrame, current_hash)
               - DataFrame is None if there are no changes, otherwise the updated DataFrame.
               - current_hash is the hash of the current DataFrame.
    """
    # Get the current DataFrame
    df = func_that_returns_df()

    # Compute the hash for the DataFrame
    current_hash = df_hash(df)

    # Compare hashes
    if current_hash == previous_hash:
        return None, current_hash

    # Return the DataFrame and the new hash if changed
    return df, current_hash


@dataclass
class KeyedComparison:
    """The result of `compare_keyed`: every row of both frames lands in exactly one bucket.

    Attributes:
        on (List[str]): The key column(s) the frames were compared on.
        values (List[str]): The column(s) checked for agreement.
        scope (Dict[str, list]): Per scope column, the values both sides hold, which is what
            the comparison was limited to. Empty when no scope was given.
        rtol (float): The relative tolerance used.
        atol (float): The absolute tolerance used.
        repeats (str): How repeated keys were judged, "exact" or "values".
        pairs (pd.DataFrame): One row per key held by exactly one row on each side. Columns
            are the keys, `left_index` / `right_index` (each side's original index label),
            then `{value}_left`, `{value}_right`, `{value}_agree` per compared column, and
            `agree` (True when every compared column agrees).
        repeated (pd.DataFrame): Keys on both sides where either side holds more than one
            row, with `left_rows` / `right_rows`, each side's values as text in `{value}_left` /
            `{value}_right`, and `values_match`. Rows under a repeated key cannot be paired
            one to one, so they are judged as a set: matching sets agree, anything else
            needs a look.
        left_only (pd.DataFrame): Rows of left whose key is not on the right (a NaN in any
            key column matches nothing).
        right_only (pd.DataFrame): Rows of right whose key is not on the left.
        left_out_of_scope (pd.DataFrame): Rows of left carrying a `scope` value the right
            never holds, with `scope_reason` naming it (e.g. "matrix = blood"). Empty unless
            `scope` is given.
        right_out_of_scope (pd.DataFrame): Rows of right carrying a `scope` value the left
            never holds.
    """

    on: List[str]
    values: List[str]
    scope: Dict[str, list]
    rtol: float
    atol: float
    repeats: str
    pairs: pd.DataFrame
    repeated: pd.DataFrame
    left_only: pd.DataFrame
    right_only: pd.DataFrame
    left_out_of_scope: pd.DataFrame
    right_out_of_scope: pd.DataFrame

    @property
    def mismatched(self) -> pd.DataFrame:
        """The uniquely-keyed pairs where at least one compared column disagrees."""
        return self.pairs[~self.pairs["agree"]]

    @property
    def repeated_differ(self) -> pd.DataFrame:
        """The repeated keys whose two sides do not hold the same values."""
        return self.repeated[~self.repeated["values_match"]]

    def summary(self) -> pd.Series:
        """Counts per bucket.

        `agree` and `mismatched` count pairs, and `pairs` is their sum; the left rows under a
        repeated key whose values match count as agreeing pairs. `repeated_differ` counts
        keys. The rest count rows.
        """
        matched = int(self.repeated.loc[self.repeated["values_match"], "left_rows"].sum())
        agree = int(self.pairs["agree"].sum()) + matched
        return pd.Series(
            {
                "pairs": agree + len(self.mismatched),
                "agree": agree,
                "mismatched": len(self.mismatched),
                "repeated_differ": len(self.repeated_differ),
                "left_only": len(self.left_only),
                "right_only": len(self.right_only),
                "left_out_of_scope": len(self.left_out_of_scope),
                "right_out_of_scope": len(self.right_out_of_scope),
            },
            name="rows",
        )


def _values_agree(a: pd.Series, b: pd.Series, rtol: float, atol: float) -> np.ndarray:
    """Element-wise agreement of two aligned columns: within tolerance for numbers, equality otherwise."""
    numeric = all(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) for s in (a, b))
    if numeric and (rtol or atol):
        a_num = a.to_numpy(dtype=float, na_value=np.nan)
        b_num = b.to_numpy(dtype=float, na_value=np.nan)
        return np.isclose(a_num, b_num, rtol=rtol, atol=atol, equal_nan=True)
    return ((a == b).fillna(False).astype(bool) | (a.isna() & b.isna())).to_numpy()


def _rows_pair_up(left: pd.DataFrame, right: pd.DataFrame, values: List[str], rtol: float, atol: float) -> bool:
    """Whether two groups of rows can be paired one to one with every pair agreeing on every value."""
    from scipy.optimize import linear_sum_assignment

    n = len(left)
    if n != len(right) or n > MAX_MATCH_ROWS:
        return False
    agree = np.ones(n * n, dtype=bool)
    for value in values:  # every left row against every right row
        a = left[value].iloc[np.repeat(np.arange(n), n)].reset_index(drop=True)
        b = right[value].iloc[np.tile(np.arange(n), n)].reset_index(drop=True)
        agree &= _values_agree(a, b, rtol, atol)
    agree = agree.reshape(n, n)
    rows, columns = linear_sum_assignment(~agree)
    return bool(agree[rows, columns].all())


def _sortable(values: pd.Series) -> pd.Series:
    """A column as given when it is numeric, as text otherwise, so mixed values still sort."""
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    return values if numeric else values.astype(str)


def _as_text(values: pd.Series) -> str:
    """A group's values on one line, e.g. "12.4; 18.5; 221", naming how many more past the first few."""
    shown = [f"{v:g}" if isinstance(v, (float, np.floating)) else str(v) for v in values.iloc[:MAX_VALUES_AS_TEXT]]
    more = len(values) - len(shown)
    return "; ".join(shown) + (f"; and {more:,} more" if more else "")


def compare_keyed(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    values: Optional[List[str]] = None,
    rtol: float = 0.0,
    atol: float = 0.0,
    scope: Optional[List[str]] = None,
    repeats: str = "exact",
) -> KeyedComparison:
    """Compare two DataFrames by key: which keys each side holds, and whether the values agree.

    Only keys held by exactly one row on each side are scored row against row. A key repeated
    on either side joins many-to-many, and counting those joined rows as disagreements
    inflates the mismatch count. Repeated keys are judged as a set instead, by `repeats`,
    and are returned in `repeated` either way.

    Keys, scope columns and values are compared as given, so fold spellings (case,
    padding, units) into one form on both sides before calling.

    Args:
        left (pd.DataFrame): First DataFrame.
        right (pd.DataFrame): Second DataFrame.
        on (Union[str, List[str]]): Key column(s), named the same on both sides.
        values (Optional[List[str]]): Columns to check for agreement. Defaults to every
            column the two share outside `on` and `scope`.
        rtol (float): Relative tolerance for numeric values, as a fraction of the right
            value. Defaults to 0.0.
        atol (float): Absolute tolerance for numeric values. Defaults to 0.0. With both
            tolerances zero, values are compared exactly in their own dtype.
        scope (Optional[List[str]]): Columns whose values define what each side is meant to
            carry. A row holding a value the other side never has in that column is out of
            scope rather than missing, and is set aside before keys are compared.
        repeats (str): When a repeated key agrees. "exact" (the default) needs the same rows
            on both sides: the same count carrying the same values, in any order. "values"
            needs only that every row has its values on the other side, however many rows
            carry them, for loosely processed data where a result may be recorded more than once.
            Under "values", a key holding too many distinct values to compare as a set (a sign
            the key is not really a key) is reported as differing, with a warning.

    Returns:
        KeyedComparison: The pairs, repeated keys, one-sided rows and out-of-scope rows.

    Raises:
        ValueError: If a key, scope or value column is missing from either side, or
            `repeats` is not "exact" or "values".
    """
    if repeats not in ("exact", "values"):
        raise ValueError(f"repeats must be 'exact' or 'values', not '{repeats}'")
    on = [on] if isinstance(on, str) else list(on)
    scope = list(scope or [])
    if values is None:
        values = [c for c in left.columns if c in right.columns and c not in on + scope]
    for side, df in (("left", left), ("right", right)):
        missing = [c for c in on + scope + values if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in the {side} DataFrame")

    # Scope: set aside rows carrying a value the other side never holds, and say which
    def scope_reasons(df: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
        reasons = np.full(len(df), "", dtype=object)
        for col in scope:
            unheld = ~df[col].isin(other[col].dropna().unique()).to_numpy()
            named = (col + " = " + df[col].astype(str)).where(df[col].notna(), f"{col} is missing").to_numpy()
            reasons = np.where(unheld, np.where(reasons == "", named, reasons + "; " + named), reasons)
        return reasons

    left_why, right_why = scope_reasons(left, right), scope_reasons(right, left)
    lf, rf = left[left_why == ""], right[right_why == ""]
    left_out = left[left_why != ""].assign(scope_reason=left_why[left_why != ""])
    right_out = right[right_why != ""].assign(scope_reason=right_why[right_why != ""])

    # Rows per key on each side (groupby drops keys holding a NaN)
    left_n = lf.groupby(on, as_index=False, observed=True).size().rename(columns={"size": "left_rows"})
    right_n = rf.groupby(on, as_index=False, observed=True).size().rename(columns={"size": "right_rows"})
    shared = left_n.merge(right_n, on=on)
    one_to_one = (shared["left_rows"] == 1) & (shared["right_rows"] == 1)

    def one_sided(df: pd.DataFrame) -> pd.DataFrame:
        hit = df[on].merge(shared[on].assign(_hit=True), on=on, how="left")["_hit"]
        return df[hit.isna().to_numpy()]

    # One-to-one keys are scored row against row
    unique = shared.loc[one_to_one, on]
    lp = lf[on + values].assign(left_index=lf.index).merge(unique, on=on)
    rp = rf[on + values].assign(right_index=rf.index).merge(unique, on=on)
    pairs = lp.merge(rp, on=on, suffixes=("_left", "_right"), validate="one_to_one")
    for value in values:
        pairs[f"{value}_agree"] = _values_agree(pairs[f"{value}_left"], pairs[f"{value}_right"], rtol, atol)
    pairs["agree"] = pairs[[f"{v}_agree" for v in values]].all(axis=1)

    # Repeated keys are judged as a set: every row on each side needs its values on the other
    repeated = shared[~one_to_one].reset_index(drop=True)

    def under_repeated_keys(df: pd.DataFrame, side: str) -> pd.DataFrame:
        rows = df[on + values].merge(repeated[on], on=on).sort_values(on + values, key=_sortable, kind="stable")
        return rows.assign(**{f"_{side}": np.arange(len(rows))})

    left_rep, right_rep = under_repeated_keys(lf, "left"), under_repeated_keys(rf, "right")

    if repeats == "exact":  # the same rows: each side in value order, compared rank by rank
        ranks = [rows.assign(_rank=rows.groupby(on, observed=True).cumcount()) for rows in (left_rep, right_rep)]
        by_rank = ranks[0].merge(ranks[1], on=on + ["_rank"], how="outer", suffixes=("_left", "_right"))
        same = by_rank["_left"].notna().to_numpy() & by_rank["_right"].notna().to_numpy()
        for value in values:
            same &= _values_agree(by_rank[f"{value}_left"], by_rank[f"{value}_right"], rtol, atol)
        failed = by_rank.loc[~same, on].drop_duplicates()
        if (rtol or atol) and len(failed):
            # Within a tolerance, value order need not line matching rows up, so a key that
            # failed by rank gets a proper one-to-one matching of its rows
            failed = failed.assign(_key=np.arange(len(failed)))
            sides = [dict(iter(rows.merge(failed, on=on).groupby("_key"))) for rows in (left_rep, right_rep)]
            matched = [
                k for k in sides[0] if k in sides[1] and _rows_pair_up(sides[0][k], sides[1][k], values, rtol, atol)
            ]
            failed = failed[~failed["_key"].isin(matched)]
        unmet = [failed[on]]
    else:
        # Whether a row's values are on the other side depends only on the distinct values, so
        # those are what get joined. A key with too many of them to join is left as differing.
        distinct = [rows.drop_duplicates(on + values) for rows in (left_rep, right_rep)]
        sizes = [rows.groupby(on, as_index=False, observed=True).size() for rows in distinct]
        load = sizes[0].merge(sizes[1], on=on, suffixes=("_left", "_right"))
        too_many = load.loc[load["size_left"] * load["size_right"] > MAX_SET_PAIRS, on]
        if len(too_many):
            log.warning(f"compare_keyed: {len(too_many)} repeated keys hold too many distinct values to compare")
        joinable = [rows.merge(too_many.assign(_skip=True), on=on, how="left") for rows in distinct]
        joinable = [rows[rows["_skip"].isna()].drop(columns="_skip") for rows in joinable]
        every = joinable[0].merge(joinable[1], on=on, suffixes=("_left", "_right"))
        every["_agree"] = True
        for value in values:
            every["_agree"] &= _values_agree(every[f"{value}_left"], every[f"{value}_right"], rtol, atol)
        unmet = [too_many]
        for side, rows in (("left", joinable[0]), ("right", joinable[1])):
            met = every.groupby(f"_{side}")["_agree"].any()
            unmet.append(rows.loc[~rows[f"_{side}"].map(met).to_numpy(dtype=bool), on])
    differing = pd.concat(unmet).drop_duplicates().assign(_differs=True)
    repeated["values_match"] = repeated[on].merge(differing, on=on, how="left")["_differs"].isna().to_numpy()
    for value in values:
        for side, rows in (("left", left_rep), ("right", right_rep)):
            text = pd.DataFrame(
                [(*key, _as_text(group[value])) for key, group in rows.groupby(on, observed=True, sort=False)],
                columns=on + [f"{value}_{side}"],
            )
            repeated = (
                repeated.merge(text, on=on, how="left") if len(text) else repeated.assign(**{text.columns[-1]: None})
            )

    columns = on + ["left_index", "right_index"]
    columns += [f"{v}_{part}" for v in values for part in ("left", "right", "agree")]
    return KeyedComparison(
        on=on,
        values=values,
        scope={col: sorted(set(left[col].dropna()) & set(right[col].dropna()), key=str) for col in scope},
        rtol=rtol,
        atol=atol,
        repeats=repeats,
        pairs=pairs[columns + ["agree"]],
        repeated=repeated,
        left_only=one_sided(lf),
        right_only=one_sided(rf),
        left_out_of_scope=left_out,
        right_out_of_scope=right_out,
    )


def binary_accuracy(conf_matrix: pd.DataFrame, positive_classes: list) -> float:
    """
    Computes binary accuracy from a confusion matrix.

    Parameters:
        conf_matrix (pd.DataFrame): Confusion matrix in the standard form -- a "labels"
            column (the actual class) plus one count column per predicted class.
        positive_classes (list): List of class labels considered as 'positive'.

    Returns:
        float: Binary accuracy score.
    """
    # Rows are the actual class, columns the predicted. Every cell is either a hit or a
    # miss, so correct is the two agreeing blocks (positive/positive, negative/negative)
    # over the whole matrix.
    counts = conf_matrix.drop(columns="labels")
    pos_rows = conf_matrix["labels"].isin(positive_classes)
    pos_cols = counts.columns.isin(positive_classes)

    correct = counts.loc[pos_rows, pos_cols].values.sum() + counts.loc[~pos_rows, ~pos_cols].values.sum()
    return correct / counts.values.sum()


def max_proba(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'max_proba' column with the max value among '_proba' columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with a new 'max_proba' column (in-place).
    """
    # Identify probability columns
    proba_cols = [col for col in df.columns if col.endswith("_proba")]

    # Compute max probability per row
    df["max_proba"] = df[proba_cols].max(axis=1)
    return df


def proba_to_conf(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'confidence' column by normalizing max_proba to [0, 1].

    Maps the maximum class probability from [1/n_classes, 1.0] to [0.0, 1.0],
    where 1/n_classes (random guess) becomes 0 and 1.0 (certain) stays 1.
    Requires a 'max_proba' column (see :func:`max_proba`).

    Args:
        df (pd.DataFrame): DataFrame with a 'max_proba' column and one or more
            columns ending in '_proba'.

    Returns:
        pd.DataFrame: The DataFrame with a new 'confidence' column (in-place).
    """
    if "max_proba" not in df.columns:
        return df
    n_classes = len([c for c in df.columns if c.endswith("_proba") and c != "max_proba"])
    baseline = 1.0 / max(n_classes, 2)
    df["confidence"] = (df["max_proba"] - baseline) / (1.0 - baseline)
    return df


def compute_confusion(df: pd.DataFrame, n_classes: int = None) -> pd.DataFrame:
    """Add a 'confusion' column combining residual and confidence.

    Each residual level gets its own non-overlapping band with small (0.1)
    gaps between them. Band width = (1 - (n-1)*gap) / n. Confidence scales
    within each band.

    For 3 classes the bands are:
        residual=0: [0.000, 0.267]  (correct predictions)
        gap:        (0.267, 0.367)
        residual=1: [0.367, 0.633]  (off-by-one)
        gap:        (0.633, 0.733)
        residual=2: [0.733, 1.000]  (maximally wrong)

    Requires 'residual' and 'confidence' columns (no-op if either is missing).

    Args:
        df (pd.DataFrame): DataFrame with 'residual' and 'confidence' columns.
        n_classes (int): Number of classes. If None, inferred from '_proba' columns.

    Returns:
        pd.DataFrame: The DataFrame with a 'confusion' column (in-place).
    """
    if "residual" not in df.columns or "confidence" not in df.columns:
        return df
    if n_classes is None:
        n_classes = len([c for c in df.columns if c.endswith("_proba") and c != "max_proba"])
    n_classes = max(n_classes, 2)

    # Each residual level r gets its own band with small gaps between them.
    # band_width = (1.0 - (n-1)*gap) / n, base = r * (band_width + gap)
    gap = 0.0
    band_width = (1.0 - (n_classes - 1) * gap) / n_classes
    residual = df["residual"]
    confidence = df["confidence"]
    correct = residual == 0
    base = residual * (band_width + gap)
    df["confusion"] = 0.0
    # Correct predictions: high confidence → 0 confusion, low confidence → top of band
    df.loc[correct, "confusion"] = (1 - confidence[correct]) * band_width
    # Wrong predictions: high confidence → top of band, low confidence → bottom
    df.loc[~correct, "confusion"] = base[~correct] + confidence[~correct] * band_width
    return df


def temporal_split(df: pd.DataFrame, date_column: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test sets based on a date boundary.

    Rows with date > end_date go to test; every other row goes to train, including
    rows whose date is null or unparseable (date-less rows carry no temporal
    information, so they are never held out).

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_column (str): The name of the datetime column.
        end_date (str): Rows strictly after this date become the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    df = df.copy()
    raw_values = df[date_column].copy()
    # format="mixed" handles columns that contain both 'YYYY-MM-DD' and
    # 'YYYY-MM-DD HH:MM:SS' style strings (which can happen when a sentinel
    # Timestamp is filled into a column that otherwise holds date-only strings).
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce", format="mixed")

    # Log diagnostic info about any unparseable dates. A null/unparseable date is
    # expected for date-less rows (e.g. external/public sources merged in via
    # SMILES) — those simply fall to the training side below — so this is a
    # once-per-call warning, not an error.
    nat_count = df[date_column].isna().sum()
    if nat_count > 0:
        nat_mask = df[date_column].isna()
        sample_raw = raw_values[nat_mask].head(5).tolist()
        log.warning(
            f"temporal_split: {nat_count}/{len(df)} '{date_column}' values could not be parsed as dates "
            f"(treated as training). Sample raw values: {sample_raw}, dtypes: {raw_values.dtype}"
        )

    # NaT compares False against the cutoff either way, so the holdout mask is the
    # only side that can be expressed directly -- train is its complement.
    cutoff = pd.to_datetime(end_date)
    holdout_mask = df[date_column] > cutoff
    return df[~holdout_mask].copy(), df[holdout_mask].copy()


def get_percent_nan(df):
    log.info("DataFrame ({:d} rows)".format(len(df)))
    s = df.isna().mean().round(3) * 100.0
    s.name = "percent_nan"
    return s


def unique(df):
    s = df.nunique()
    s.name = "num_unique"
    return s


def column_dtypes(df):
    s = df.dtypes
    s.name = "dtype"
    return s


def examples(df):
    first_n = [df[c].unique()[:5].tolist() for c in df.columns]
    first_n = [", ".join([str(x) for x in _list]) for _list in first_n]
    s = pd.Series(first_n, df.columns)
    s.name = "examples"
    return s


def info(df):
    # Get the number of unique values for each column
    s0 = column_dtypes(df)
    s1 = df.count()
    s1.name = "count"
    s2 = get_percent_nan(df)
    s3 = unique(df)

    # Remove all the numeric columns from the original dataframe
    # non_numeric_columns = df.select_dtypes(exclude="number").columns.tolist()
    s4 = examples(df)

    # Concatenate the series together
    return pd.concat([s0, s1, s2, s3, s4], axis=1)


def numeric_stats(df):
    """Simple function to get the numeric stats for a dataframe"""
    return df.describe().round(2).T.drop("count", axis=1)


def remove_rows_with_nans(input_df: pd.DataFrame, how: str = "any", subset: list = None) -> pd.DataFrame:
    """
    Removes rows with NaN or INF values from the DataFrame.

    Args:
        input_df (pd.DataFrame): Input data frame.
        how (str): 'any' to drop rows where any value is NaN/INF, 'all' to drop rows where all values are NaN/INF.
        subset (list): Specific subset of columns to check for NaN/INF.

    Returns:
        pd.DataFrame: DataFrame with rows containing NaN/INF removed.
    """
    log.info("Replacing INF/-INF values with NaN.")
    input_df = input_df.replace([np.inf, -np.inf], np.nan)

    orig_num_rows = len(input_df)
    log.info(f"Checking for rows with NaN values using '{how}' condition.")

    input_df = input_df.dropna(axis=0, how=how, subset=subset)
    dropped_rows = orig_num_rows - len(input_df)

    if dropped_rows > 0:
        log.warning(f"Dropped {dropped_rows} rows with NaN or INF values.")
    else:
        log.info("No rows with NaN or INF values found.")

    input_df.reset_index(drop=True, inplace=True)
    return input_df


def detect_drift(current_df, new_df, drift_percentage=0.1):
    """
    Detects drift between two dataframes by checking if values in the new dataframe
    fall outside the acceptable range based on the current dataframe.

    Args:
        current_df: DataFrame containing the baseline data
        new_df: DataFrame containing the new data to check for drift
        drift_percentage: Maximum allowed percentage change (0.1 = 0.1%)
                          relative to each column's min-max range

    Returns:
        tuple: (has_drift, details_dict)
            - has_drift: Boolean indicating if any drift was detected
            - details_dict: Dictionary containing:
                - 'columns_with_drift': List of column names with drift
                - 'drift_examples': DataFrame showing drift details

    Scale invariance:
        For each column, the allowable drift is calculated as a percentage
        of that column's total value range, making the comparison scale-invariant.
    """

    # Ensure we're only comparing columns present in both dataframes
    common_columns = list(set(current_df.columns) & set(new_df.columns))

    if not common_columns:
        return False, {"columns_with_drift": [], "drift_examples": pd.DataFrame()}

    drift_cols = []
    drift_data = []

    for col in common_columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(current_df[col]) or not pd.api.types.is_numeric_dtype(new_df[col]):
            continue

        # Calculate min/max for the allowable range
        min_val = min(current_df[col].min(), new_df[col].min())
        max_val = max(current_df[col].max(), new_df[col].max())

        # Avoid division by zero for constant columns
        if max_val == min_val:
            continue

        # Calculate the column range and the drift allowance
        column_range = max_val - min_val
        column_drift = column_range * (drift_percentage / 100)

        # Calculate drift boundaries
        drift_min = current_df[col] - column_drift
        drift_max = current_df[col] + column_drift

        # Check which values in new_df are outside the drift boundaries
        out_of_bounds = (new_df[col] < drift_min) | (new_df[col] > drift_max)

        if out_of_bounds.any():
            drift_cols.append(col)

            # Get indices where drift occurred
            drift_indices = out_of_bounds[out_of_bounds].index

            # Create example data (limited to first 5 instances)
            for idx in drift_indices[:5]:
                drift_data.append(
                    {
                        "column": col,
                        "row_index": idx,
                        "current_value": current_df.loc[idx, col],
                        "new_value": new_df.loc[idx, col],
                        "min_bound": drift_min[idx],
                        "max_bound": drift_max[idx],
                        "column_range": column_range,
                        "allowed_drift": column_drift,
                    }
                )

    has_drift = len(drift_cols) > 0
    details = {
        "columns_with_drift": drift_cols,
        "drift_examples": pd.DataFrame(drift_data) if drift_data else pd.DataFrame(),
    }

    return has_drift, details


def drop_outliers_iqr(input_df: pd.DataFrame, scale: float = 1.5) -> pd.DataFrame:
    """Drop outliers from a dataframe
    Args:
        input_df (pd.DataFrame): Input DataFrame
        scale (float, optional): Scale to use for IQR. Defaults to 1.5.
    Returns:
        pd.DataFrame: DataFrame with outliers dropped
    """

    # Just the numeric columns
    numeric_df = input_df.select_dtypes(include="number")

    # Drop Outliers using IQR
    q1 = numeric_df.quantile(0.25, numeric_only=True)
    q3 = numeric_df.quantile(0.75, numeric_only=True)
    iqr_scale = (q3 - q1) * scale
    output_df = input_df[~((numeric_df < (q1 - iqr_scale)) | (numeric_df > (q3 + iqr_scale))).any(axis=1)]
    if input_df.shape[0] != output_df.shape[0]:
        log.info(f"Dropped {input_df.shape[0] - output_df.shape[0]} outlier rows")
    return output_df


MIN_PILEUP_ROWS = 3


def target_health(df: pd.DataFrame, target: str, pileup_percent: float = 2.0) -> pd.DataFrame:
    """Check a regression target column for the pathologies that quietly cap model performance.

    Assay data arrives censored (values clipped at a detection limit), discretized
    (rounded to the reporting precision), and skewed (concentrations and clearance
    rates are log-normal). None of these are visible in a scatter plot of predictions
    vs actuals, and all of them change what a model can learn.

    Args:
        df (pd.DataFrame): Input DataFrame
        target (str): The target column to check
        pileup_percent (float, optional): Percent of rows on a single value that counts
            as a pileup. Defaults to 2.0.

    Returns:
        pd.DataFrame: One row per check with columns: check, severity (ok/info/warn),
            value, detail.
    """
    findings = []

    def add(check, severity, value, detail):
        findings.append({"check": check, "severity": severity, "value": value, "detail": detail})

    values = df[target]
    n_rows = len(values)
    n_nan = int(values.isna().sum())
    clean = values.dropna()

    add(
        "missing",
        "warn" if n_nan else "ok",
        f"{n_nan} ({100 * n_nan / n_rows:.1f}%)" if n_rows else "0",
        "rows with no target value" if n_nan else "no missing targets",
    )
    if clean.empty:
        return pd.DataFrame(findings)

    # Discretization: a target rounded to the assay's reporting precision can't be
    # predicted more finely than that grid, which puts a floor under achievable RMSE.
    n_unique = int(clean.nunique())
    unique_ratio = n_unique / len(clean)
    add(
        "discretization",
        "warn" if unique_ratio < 0.05 else "info" if unique_ratio < 0.25 else "ok",
        f"{n_unique} unique / {len(clean)} rows",
        f"target resolves to {n_unique} distinct values — model error can't go below the grid spacing",
    )

    # Censoring. The sibling "{target}_lt" / "{target}_gt" booleans are the convention
    # for a bound the data already declares, and chemprop reads that same pair under
    # bounded_loss. Declared rows are reported and sit out the pileup heuristic below,
    # which exists only to catch censoring nobody labelled ("> 300" recorded as 300).
    counts = clean.value_counts()
    top_value, top_count = counts.index[0], int(counts.iloc[0])
    top_percent = 100 * top_count / len(clean)
    lo, hi = clean.min(), clean.max()

    bound_columns = [f"{target}_lt", f"{target}_gt"]
    flags = df.reindex(columns=bound_columns).reindex(clean.index).eq(True)
    declared = {c: int(flags[c].sum()) for c in bound_columns if c in df.columns and flags[c].any()}
    unlabelled = clean[~flags.any(axis=1)]

    boundary = {v: int((unlabelled == v).sum()) for v in (lo, hi)}
    # A couple of rows on the boundary is just the extremes of a distribution — a
    # pileup needs both a real share and enough rows to be a stack.
    boundary_hits = {
        v: c for v, c in boundary.items() if c >= MIN_PILEUP_ROWS and 100 * c / len(clean) >= pileup_percent
    }

    declared_note = ", ".join(f"{c}: {n} rows" for c, n in declared.items())
    pileup_note = ", ".join(f"{v:g}: {c} rows" for v, c in boundary_hits.items())
    add(
        "censoring",
        "warn" if boundary_hits else "info" if declared else "ok",
        "; ".join(n for n in (declared_note, pileup_note) if n) or "none",
        (
            "rows stacked on the min/max — censored assay values, consider dropping or modeling as bounded"
            if boundary_hits
            else (
                "bounds declared — train with bounded_loss=True to honor them"
                if declared
                else "no pileup at the target's min or max"
            )
        ),
    )
    add(
        "pileup",
        "info" if top_count >= MIN_PILEUP_ROWS and top_percent >= pileup_percent else "ok",
        f"{top_value:g}: {top_count} rows ({top_percent:.1f}%)",
        "most repeated single value",
    )

    # Skew: log-normal targets (solubility, clearance, IC50) trained untransformed let
    # a handful of large values dominate the loss.
    skew = float(clean.skew())
    n_non_positive = int((clean <= 0).sum())
    add(
        "skew",
        "warn" if abs(skew) > 2 else "info" if abs(skew) > 1 else "ok",
        f"{skew:.2f}",
        (
            f"consider a log transform ({n_non_positive} non-positive values need an offset)"
            if abs(skew) > 2 and n_non_positive
            else "consider a log transform" if abs(skew) > 2 else "acceptable skew"
        ),
    )
    add("range", "ok", f"[{lo:g}, {hi:g}]", f"mean={clean.mean():g} median={clean.median():g}")

    return pd.DataFrame(findings)


# A column missing in at least this fraction of rows is a warning rather than a note
MISSING_WARN_FRACTION = 0.05

# A text column where at least this fraction of entries parse as numbers is a numeric
# column carrying stray text ("<0.1", "BLQ"), and the entries that do not parse are flagged
MOSTLY_NUMERIC_FRACTION = 0.5

# compare_keyed: the most distinct-value pairs joined under one repeated key, and the most of a
# repeated key's values written out as text
MAX_SET_PAIRS, MAX_VALUES_AS_TEXT = 250_000, 20

# compare_keyed: the most rows under one repeated key that are matched one to one within a tolerance
MAX_MATCH_ROWS = 400

# A positive column skewed beyond this is treated as log-distributed when looking for outliers
LOG_FENCE_SKEW = 2.0

SEVERITY_ORDER = {"warn": 0, "info": 1, "ok": 2}


@dataclass
class DataQuality:
    """The result of `data_quality`: what is wrong, and which rows.

    Attributes:
        findings (pd.DataFrame): One row per finding with columns: check, column,
            severity (warn/info/ok), count (rows flagged, where the check counts rows),
            value, detail. Worst first.
        rows (pd.DataFrame): One row per offending cell: the key column(s) (or `index`
            when no key was given), check, column, and the offending value as text.
            Joins back to the source frame on the key.
    """

    findings: pd.DataFrame
    rows: pd.DataFrame


def data_quality(
    df: pd.DataFrame,
    key: Union[str, List[str], None] = None,
    target: Optional[str] = None,
    outlier_scale: float = 3.0,
    max_rows: int = 1000,
) -> DataQuality:
    """Check a DataFrame for the problems that corrupt whatever is built on it.

    Per column: `missing` values, `non_finite` (infinite) numbers, `unparseable` text in a
    mostly numeric column, `outliers` beyond the quartiles, and `constant` columns. Per
    frame: `duplicate_key` and `duplicate_rows`. With a `target`, the `target_health`
    checks (censoring, discretization, pileup, skew, range) are appended.

    Only problems are reported, so an empty `findings` frame is a clean bill.

    Args:
        df (pd.DataFrame): Input DataFrame.
        key (Union[str, List[str], None]): Column(s) identifying a row. Offending rows are
            listed by key so they can be looked up; without one the index is used.
        target (Optional[str]): Regression target column to run `target_health` on.
        outlier_scale (float): Multiple of the interquartile range beyond the quartiles
            that counts as an outlier. Defaults to 3.0. A positive, strongly skewed column
            is fenced on its log10 values.
        max_rows (int): Most rows listed per finding. Defaults to 1000.

    Returns:
        DataQuality: The findings and the rows behind them.
    """
    key_cols = [key] if isinstance(key, str) else list(key or [])
    if not key_cols:
        log.warning("data_quality: no key given, offending rows are listed by index")
    ids = df[key_cols] if key_cols else pd.DataFrame({"index": df.index}, index=df.index)
    findings, row_frames = [], []

    def add(check: str, column: str, severity: str, mask: pd.Series, detail: str, values: pd.Series = None):
        count = int(mask.sum())
        if count > max_rows:
            detail += f" (first {max_rows} rows listed)"
        value = f"{count} ({100 * count / len(df):.1f}%)"
        finding = {"check": check, "column": column, "severity": severity, "count": count}
        findings.append({**finding, "value": value, "detail": detail})
        hit = ids[mask.to_numpy()].head(max_rows).assign(check=check, column=column, value=None)
        if values is not None:
            hit["value"] = values[mask.to_numpy()].head(max_rows).astype(str).to_numpy()
        row_frames.append(hit)

    hashable_cols = []
    for col in df.columns:
        values = df[col]
        missing = values.isna()
        if missing.any():
            serious = col in key_cols or missing.mean() >= MISSING_WARN_FRACTION
            add("missing", col, "warn" if serious else "info", missing, "rows with no value")
        present = values[~missing]
        if present.map(pd.api.types.is_hashable).all():
            hashable_cols.append(col)
        else:
            continue
        if col in key_cols:
            continue

        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            numbers = pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=values.index)
            infinite = np.isinf(numbers)
            if infinite.any():
                add("non_finite", col, "warn", infinite, "infinite values", values)
            finite = numbers[~infinite & ~missing]
            if not finite.empty:
                # A positive, strongly skewed column (a concentration, a clearance) is fenced on
                # its log10 values; on the raw values its whole upper tail reads as outliers
                on_log = bool((finite > 0).all() and abs(finite.skew()) > LOG_FENCE_SKEW)
                q1, q3 = np.percentile(np.log10(finite) if on_log else finite, [25, 75])
                low, high = q1 - outlier_scale * (q3 - q1), q3 + outlier_scale * (q3 - q1)
                low, high = (10**low, 10**high) if on_log else (low, high)
                outliers = ~infinite & ((numbers < low) | (numbers > high))
                if q3 > q1 and outliers.any():
                    scale = " of the log10 values" if on_log else ""
                    detail = f"outside [{low:g}, {high:g}], {outlier_scale:g} x IQR beyond the quartiles{scale}"
                    add("outliers", col, "info", outliers, detail, values)
        elif not pd.api.types.is_bool_dtype(values) and not pd.api.types.is_datetime64_any_dtype(values):
            unparseable = pd.to_numeric(values, errors="coerce").isna() & ~missing
            if unparseable.any() and 1 - unparseable.sum() / len(present) >= MOSTLY_NUMERIC_FRACTION:
                add("unparseable", col, "warn", unparseable, "text in a mostly numeric column", values)

        if len(present) > 1 and present.nunique() == 1:
            finding = {"check": "constant", "column": col, "severity": "info", "value": str(present.iloc[0])}
            findings.append({**finding, "detail": "every row holds the same value"})

    if key_cols:
        repeated = df.duplicated(key_cols, keep=False) & df[key_cols].notna().all(axis=1)
        if repeated.any():
            add("duplicate_key", ", ".join(key_cols), "warn", repeated, "key held by more than one row")
    # Whole-row identity needs every column comparable
    comparable = len(hashable_cols) == len(df.columns) and len(hashable_cols) > 0
    identical = df.duplicated(keep=False) if comparable else pd.Series(False, index=df.index)
    if identical.any():
        add("duplicate_rows", None, "warn", identical, "rows identical in every column")

    findings_df = pd.DataFrame(findings, columns=["check", "column", "severity", "count", "value", "detail"])
    if target:
        health = target_health(df, target)
        health = health[health["check"] != "missing"].assign(column=target, count=pd.NA)
        findings_df = pd.concat([findings_df, health[findings_df.columns]], ignore_index=True)
    findings_df["count"] = findings_df["count"].astype("Int64")
    order = findings_df["severity"].map(SEVERITY_ORDER)
    findings_df = findings_df.loc[order.sort_values(kind="stable").index].reset_index(drop=True)

    row_columns = list(ids.columns) + ["check", "column", "value"]
    rows = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame(columns=row_columns)
    return DataQuality(findings=findings_df, rows=rows[row_columns])


def shorten_values(df: pd.DataFrame, max_length: int = 100) -> pd.DataFrame:
    """This method should be used with caution"""

    def truncate_element(element):
        if isinstance(element, str) and len(element) > max_length:
            return element[:max_length] + "..."  # Add ellipsis to indicate truncation
        return element

    return df.map(truncate_element)


def subplot_positions(df: pd.DataFrame, num_columns: int = 2) -> Dict[str, Tuple[int, int]]:
    positions = {}
    for i, col in enumerate(df.columns):
        row = i // num_columns + 1
        col_pos = i % num_columns + 1
        positions[col] = (row, col_pos)
    return positions


def displayable_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """Experimental: Create a displayable dataframe from FeatureSet data
    Args:
        input_df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with displayable columns
    """
    exclude_columns = ["write_time", "api_invocation_time", "is_deleted", "training"]
    df = input_df[input_df.columns.difference(exclude_columns)].copy()
    dummy_cols = get_dummy_cols(df)

    # Okay, so this is a bit of a hack, but we need to replace all but the last underscore
    # run the from_dummies method, then change the column names back to the original
    un_dummy = undummify(df[dummy_cols])
    return pd.concat([df.drop(dummy_cols, axis=1), un_dummy], axis=1)


def undummify(df, prefix_sep="_"):
    """Experimental: Undummify a dataframe"""
    cols2collapse = {prefix_sep.join(item.split(prefix_sep)[:-1]): (prefix_sep in item) for item in df.columns}
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = df.filter(like=col).idxmax(axis=1).apply(lambda x: x.split(prefix_sep)[-1]).rename(col)
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def get_dummy_cols(df: pd.DataFrame) -> list:
    """Determines a list of dummy columns for the given DataFrame
    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        list: List of dummy columns
    """
    dum_cols = list(df.select_dtypes(include=["int", "bool"]).columns)
    underscore_cols = [col for col in df.columns if "_" in col and col in dum_cols]
    dummy_cols = []
    for col in underscore_cols:
        # Just columns with 0 and 1
        if set(df[col].unique()).issubset([0, 1]):
            dummy_cols.append(col)
    return dummy_cols


def athena_to_pandas_types(df: pd.DataFrame, column_athena_types: dict) -> pd.DataFrame:
    """Converts a dataframe into the proper Pandas types
    Args:
        df (pd.DataFrame): The DataFrame we want to convert types for
        column_athena_types (dict): A dictionary of Athena types for each column
    Returns:
        pd.DataFrame: The DataFrame with the proper types
    """

    # Sanity check
    if df.empty:
        return df

    # Convert the Athena types to Pandas types with this mapper
    athena_to_pandas_mapper = {
        "tinyint": "Int8",
        "smallint": "Int16",
        "int": "Int32",
        "integer": "Int32",
        "bigint": "Int64",
        "boolean": "boolean",
        "float": "float32",
        "double": "float64",
        "real": "float64",
        "decimal": "float64",
        "numeric": "float64",
        "char": "string",
        "varchar": "string",
        "string": "string",
        "date": "datetime64[ns]",
        "timestamp": "datetime64[ns]",
        "binary": "object",
        "array": "object",
        "map": "object",
        "struct": "object",
        "uniontype": "object",
    }

    # Map the Athena types to Pandas types
    pandas_column_types = {
        col: athena_to_pandas_mapper[athena_type] for col, athena_type in column_athena_types.items()
    }

    # Convert the DataFrame columns to the mapped Pandas types
    df = df.astype(pandas_column_types)
    return df


def convert_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to automatically convert object columns to something more concrete"""
    for c in df.columns[df.dtypes == "object"]:  # Look at the object columns
        # Try to convert object to datetime
        if "date" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c])
            except (ParserError, ValueError, TypeError):
                log.debug(f"Column {c} could not be converted to datetime...")

        # Try to convert object to string
        else:
            try:
                df[c] = df[c].astype(str)
            except (ParserError, ValueError, TypeError):
                log.debug(f"Column {c} could not be converted to string...")
    return df


def serialize_aws_metadata(metadata):
    """
    Serializes a dictionary of DataFrames to a JSON-formatted string.
    Args:
        metadata (dict): Dictionary of DataFrames
    Returns:
        str: JSON-formatted string
    """
    serialized_dict = {key: df.to_json() for key, df in metadata.items()}
    return json.dumps(serialized_dict)


def deserialize_aws_metadata(serialized_data):
    """
    Deserializes a JSON-formatted string to a dictionary of DataFrames.
    Args:
        serialized_data (str): JSON-formatted string
    Returns:
        dict: Dictionary of DataFrames
    """
    deserialized_dict = json.loads(serialized_data)
    return {key: pd.read_json(StringIO(df_json)) for key, df_json in deserialized_dict.items()}


def expand_proba_column(df: pd.DataFrame, class_labels: List[str]) -> pd.DataFrame:
    """
    Expands a column in a DataFrame containing a list of probabilities into separate columns.

    Args:
        df (pd.DataFrame): DataFrame containing a "pred_proba" column
        class_labels (List[str]): List of class labels

    Returns:
        pd.DataFrame: DataFrame with the "pred_proba" expanded into separate columns
    """

    # Sanity check
    proba_column = "pred_proba"
    if proba_column not in df.columns:
        raise ValueError('DataFrame does not contain a "pred_proba" column')

    # Construct new column names with '_proba' suffix
    new_col_names = [f"{label}_proba" for label in class_labels]

    # Expand the proba_column into separate columns for each probability
    proba_df = pd.DataFrame(df[proba_column].tolist(), columns=new_col_names)

    # Drop the original proba_column and reset the index in prep for the concat
    df = df.drop(columns=[proba_column])
    df = df.reset_index(drop=True)

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, proba_df], axis=1)
    return df


def stratified_split(df, column_name, test_size=0.2, random_state=42):
    """
    Stratified train-test split based on a categorical column, including handling NaNs as a separate category.

    Args:
        df (pd.DataFrame): DataFrame to split
        column_name (str): Column name to stratify the split on
        test_size (float): Proportion of the test set
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame, pd.DataFrame: Train and Test DataFrames

    """
    df_temp = df.copy()
    df_temp[column_name] = df_temp[column_name].fillna("NaN_PLACEHOLDER")

    # Ensure at least 1 sample per group in test
    test_counts = (df_temp[column_name].value_counts() * test_size).clip(lower=1).astype(int)

    test_indices = (
        df_temp.groupby(column_name)
        .apply(lambda x: x.sample(n=test_counts[x.name], random_state=random_state))
        .index.get_level_values(1)
    )

    train_df = df_temp.drop(test_indices)
    test_df = df_temp.loc[test_indices]

    # Restore NaNs
    for df_split in [train_df, test_df]:
        df_split[column_name] = df_split[column_name].replace("NaN_PLACEHOLDER", np.nan)

    return train_df, test_df


if __name__ == "__main__":
    """Exercise the Pandas Utility Methods"""
    from workbench.utils.synthetic_data_generator import SyntheticDataGenerator

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Generate some test data
    test_data = SyntheticDataGenerator()
    test_df = test_data.person_data()

    # Get the info about this dataframe
    info_df = info(test_df)

    # Show the info dataframe
    print(info_df)

    # Get the hash of the DataFrame
    hash_value = df_hash(test_df)
    print(hash_value)

    # Get min/max/mean/median/std for numeric columns
    stats_df = numeric_stats(test_df)
    print(stats_df)

    # Clean the DataFrame
    clean_df = remove_rows_with_nans(test_df)
    log.info(clean_df)

    # Drop Outliers
    norm_df = drop_outliers_iqr(clean_df)
    log.info(norm_df)

    # Compare two dataframes by key
    my_df1 = test_data.person_data()
    my_df2 = my_df1.copy()
    my_df2.loc[6, "Height"] = 99
    my_df2.loc[48, "Salary"] = 99000
    comparison = compare_keyed(my_df1, my_df2.drop(index=82), on="Id", values=["Height", "Salary", "Age"])
    print(comparison.summary())
    print(comparison.mismatched)
    print(comparison.left_only)

    # Example for Binary Accuracy
    conf_matrix = pd.DataFrame(
        {
            "labels": ["bad_low", "okay_low", "good", "okay_high", "bad_high"],
            "bad_low": [11, 1, 0, 0, 0],
            "okay_low": [19, 413, 64, 0, 0],
            "good": [0, 92, 1161, 111, 17],
            "okay_high": [0, 0, 42, 188, 34],
            "bad_high": [0, 0, 8, 18, 96],
        }
    )

    # Define Positive Classes
    positive_classes = ["okay_low", "good", "okay_high"]

    # Compute Binary Accuracy
    accuracy = binary_accuracy(conf_matrix, positive_classes)
    print(f"Binary Accuracy: {accuracy:.3f}")

    # A classifier prediction frame, for the checks below
    data = {
        "class": ["low", "low", "high", "med", "low", "high", "low", "high", "low", "high"],
        "prediction": ["low", "low", "high", "high", "low", "high", "low", "high", "low", "med"],
        "pred_proba": [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
            [0.6, 0.2, 0.2],
            [0.1, 0.3, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.3, 0.4],
            [0.6, 0.2, 0.2],
            [0.25, 0.5, 0.25],
        ],
    }
    df = pd.DataFrame(data)
    df = expand_proba_column(df, ["low", "med", "high"])

    # Add a datetime column to our df and test temporal_split
    df["date"] = pd.date_range(start="1/1/2020", periods=len(df))
    train_df, test_df = temporal_split(df, "date", end_date="2020-01-08")
    print("Train DataFrame:")
    print(train_df)
    print("\nTest DataFrame:")
    print(test_df)
