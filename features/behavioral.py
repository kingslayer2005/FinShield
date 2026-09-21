"""
features/behavioral.py
Computes leakage-safe behavioural features for each transaction.

"Leakage-safe" means: for every row, we only use information from
transactions by the SAME card that happened BEFORE this one (strictly
earlier TransactionDT).  This function is the single source of truth —
the online pipeline (Phase 3) must match these numbers exactly.
"""

import pandas as pd
import numpy as np
from typing import List


def make_card_id(df: pd.DataFrame, card_key_cols: List[str]) -> pd.Series:
    """Build a single string card identifier from one or more columns.

    Args:
        df: DataFrame containing the card key columns.
        card_key_cols: List of column names (e.g. ["card1", "addr1"]).

    Returns:
        A Series of string card IDs like "12345_456".
        If addr1 is NaN for a row, we fall back to card1 alone so that
        every row still gets an identifier.
    """
    # Start with the first key column, converted to string
    card_id = df[card_key_cols[0]].astype(str)

    # Append each additional key column, using "" for NaN values
    for col in card_key_cols[1:]:
        # fillna("") so NaN addr1 doesn't make the whole ID "nan"
        card_id = card_id + "_" + df[col].fillna("").astype(str)

    return card_id


def compute_behavioral_features(
    df: pd.DataFrame,
    card_key_cols: List[str],
    windows_sec: List[int],
) -> pd.DataFrame:
    """Add behavioural features using only each card's PAST transactions.

    For each transaction we compute (using only prior rows of the same card):
      - card_txn_count_{window}s   : number of transactions in the last N seconds
      - card_amt_sum_{window}s     : total TransactionAmt in the last N seconds
      - card_time_since_last       : seconds since this card's previous transaction

    Args:
        df: DataFrame that MUST contain TransactionDT, TransactionAmt,
            and the columns listed in card_key_cols.  It does NOT need
            to be sorted — we sort internally.
        card_key_cols: Columns identifying a unique card (e.g. ["card1","addr1"]).
        windows_sec: List of rolling-window sizes in seconds (e.g. [3600, 86400]).

    Returns:
        A copy of df (same row order as input) with new feature columns appended.
    """
    # -- 1. Build card ID and save original index so we can restore order later --
    result = df.copy()
    result["_card_id"] = make_card_id(result, card_key_cols)
    result["_orig_idx"] = result.index  # remember original row positions

    # -- 2. Sort by card then time so grouped rolling works correctly --
    result = result.sort_values(["_card_id", "TransactionDT"]).reset_index(drop=True)

    # -- 3. Time since this card's previous transaction --
    # shift(1) within each card gives the previous row's TransactionDT
    prev_dt = result.groupby("_card_id")["TransactionDT"].shift(1)
    # Subtract to get seconds gap; first transaction of a card gets -1
    result["card_time_since_last"] = (result["TransactionDT"] - prev_dt).fillna(-1)

    # -- 4. Rolling counts and sums for each time window --
    # We use a manual approach because pandas rolling("3600s") would
    # include the current row.  We need STRICTLY PAST transactions only.
    for window in windows_sec:
        # Human-readable suffix: 3600 → "3600s"
        suffix = f"{window}s"

        # For each card group, count how many prior transactions fall
        # within [current_dt - window, current_dt) — note open on the right.
        count_col = f"card_txn_count_{suffix}"
        sum_col = f"card_amt_sum_{suffix}"

        # Initialise with zeros
        result[count_col] = 0
        result[sum_col] = 0.0

        # Process each card separately — vectorised within each group
        for card_id, group in result.groupby("_card_id"):
            # Get the timestamps and amounts as numpy arrays for speed
            dts = group["TransactionDT"].values     # sorted ascending
            amts = group["TransactionAmt"].values

            n = len(dts)
            counts = np.zeros(n, dtype=np.int64)
            sums = np.zeros(n, dtype=np.float64)

            # Sliding window: left pointer moves forward as we advance
            left = 0
            running_sum = 0.0
            for i in range(n):
                # Current transaction's time
                curr_dt = dts[i]
                # Window start: anything at or after this time counts
                window_start = curr_dt - window

                # Move left pointer to exclude transactions older than the window
                while left < i and dts[left] < window_start:
                    running_sum -= amts[left]
                    left += 1

                # Count and sum exclude the current row (index i)
                # Everything from left to i-1 is in the window
                counts[i] = i - left
                sums[i] = running_sum

                # Add current transaction to the running sum for future rows
                running_sum += amts[i]

            # Write back into the result DataFrame
            result.loc[group.index, count_col] = counts
            result.loc[group.index, sum_col] = sums

    # -- 5. Restore original row order and drop helper columns --
    result = result.sort_values("_orig_idx").reset_index(drop=True)
    result = result.drop(columns=["_card_id", "_orig_idx"])

    return result
