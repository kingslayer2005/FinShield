"""
tests/fixtures/make_synthetic_data.py
Generates a fake dataset shaped like IEEE-CIS Fraud Detection for smoke tests.

Produces ~20,000 transactions with:
  - Same column names and dtypes the pipeline expects
  - Realistic NaN patterns (identity cols ~30% missing, V-cols ~40% missing)
  - ~3.5% fraud rate with learnable signal (higher amounts, late hours)
  - Many cards with multiple transactions over time
  - A few categories that only appear late (to test unknown-category handling)

Run from repo root:
    python tests/fixtures/make_synthetic_data.py

Output: data/synthetic/train_transaction.csv, data/synthetic/train_identity.csv
"""

import numpy as np
import pandas as pd
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TRANSACTIONS = 20_000               # total number of transactions
FRAUD_RATE = 0.035                     # ~3.5% fraud rate
N_CARDS = 3_000                        # number of unique card1 values
N_ADDRS = 500                          # number of unique addr1 values
SEED = 42                             # reproducibility
OUT_DIR = os.path.join("data", "synthetic")

# Categories that only appear in the last 20% of time (test unknown handling)
LATE_EMAIL_DOMAINS = ["newmail.com", "latecorp.org"]
LATE_DEVICE_TYPES = ["smartwatch"]


def main():
    """Generate synthetic IEEE-CIS-like data and save as CSVs."""
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating synthetic IEEE-CIS data for smoke tests")
    print("=" * 60)

    # -- 1. Generate TransactionDT (seconds from reference, spans ~30 days) --
    # Sort ascending to simulate chronological ordering
    dt_values = np.sort(np.random.randint(86400, 86400 * 30, size=N_TRANSACTIONS))

    # -- 2. Card identifiers: card1 (always present), addr1 (sometimes NaN) --
    card1 = np.random.randint(1000, 1000 + N_CARDS, size=N_TRANSACTIONS)
    addr1 = np.random.choice(
        list(range(100, 100 + N_ADDRS)) + [np.nan],  # ~10% NaN
        size=N_TRANSACTIONS,
        p=[0.9 / N_ADDRS] * N_ADDRS + [0.1],
    )

    # -- 3. TransactionAmt: log-normal distribution, fraud tends higher --
    base_amt = np.exp(np.random.normal(3.5, 1.2, size=N_TRANSACTIONS))
    base_amt = np.clip(base_amt, 0.5, 10000).astype(np.float32)

    # -- 4. Fraud labels with learnable signal --
    fraud_prob = np.full(N_TRANSACTIONS, FRAUD_RATE, dtype=np.float64)

    # Signal 1: high-amount transactions are more likely fraud
    high_amt_mask = base_amt > np.percentile(base_amt, 90)
    fraud_prob[high_amt_mask] *= 3.0  # 3x base rate for high amounts

    # Signal 2: late-night transactions (hours 0-5) are more likely fraud
    hour_of_day = (dt_values / 3600 % 24).astype(int)
    late_night_mask = hour_of_day < 6
    fraud_prob[late_night_mask] *= 2.0  # 2x for late night

    # Signal 3: certain cards are "compromised" — higher fraud rate
    compromised_cards = np.random.choice(
        range(1000, 1000 + N_CARDS), size=int(N_CARDS * 0.05), replace=False
    )
    compromised_mask = np.isin(card1, compromised_cards)
    fraud_prob[compromised_mask] *= 5.0

    # Clip probabilities and sample
    fraud_prob = np.clip(fraud_prob, 0, 0.95)
    is_fraud = (np.random.random(N_TRANSACTIONS) < fraud_prob).astype(int)

    # Adjust amounts for fraud: fraud transactions tend to be higher
    fraud_mask = is_fraud == 1
    base_amt[fraud_mask] *= np.random.uniform(1.5, 4.0, size=fraud_mask.sum())

    actual_fraud_rate = is_fraud.mean()
    print(f"  Fraud rate: {actual_fraud_rate * 100:.1f}% "
          f"({is_fraud.sum()} / {N_TRANSACTIONS})")

    # -- 5. Card features: card2-card6 --
    card2 = np.random.choice([111, 222, 333, 444, 555, np.nan],
                             size=N_TRANSACTIONS, p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
    card3 = np.random.choice([150, 185, 226], size=N_TRANSACTIONS)
    card4 = np.random.choice(["visa", "mastercard", "discover", "american express"],
                             size=N_TRANSACTIONS, p=[0.5, 0.3, 0.1, 0.1])
    card5 = np.random.choice([100, 117, 166, 224, 226], size=N_TRANSACTIONS)
    card6 = np.random.choice(["debit", "credit", "charge", "debit or credit"],
                             size=N_TRANSACTIONS, p=[0.4, 0.35, 0.05, 0.2])

    # -- 6. Email domains (P_emaildomain, R_emaildomain) --
    # Common domains that appear throughout the dataset
    common_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                      "aol.com", "protonmail.com"]

    # Determine which transactions are "late" (last 20% of time range)
    time_80pct = np.percentile(dt_values, 80)
    is_late = dt_values >= time_80pct

    # Build P_emaildomain: common domains everywhere, late domains only at end
    p_email = np.random.choice(common_domains, size=N_TRANSACTIONS)
    # Inject late-only domains into the last 20%
    late_indices = np.where(is_late)[0]
    n_late_inject = min(100, len(late_indices))  # inject ~100 late-domain rows
    inject_idx = np.random.choice(late_indices, size=n_late_inject, replace=False)
    p_email[inject_idx] = np.random.choice(LATE_EMAIL_DOMAINS, size=n_late_inject)
    # Add ~15% NaN
    nan_mask = np.random.random(N_TRANSACTIONS) < 0.15
    p_email_series = pd.array(p_email, dtype="object")
    p_email_series[nan_mask] = None

    r_email = np.random.choice(common_domains + [None], size=N_TRANSACTIONS,
                               p=[0.12, 0.12, 0.12, 0.12, 0.06, 0.06, 0.40])

    # -- 7. C columns (count-like features): C1-C14 --
    c_cols = {}
    for i in range(1, 15):
        # Integer counts with different distributions per column
        c_vals = np.random.poisson(lam=np.random.uniform(0.5, 5.0),
                                   size=N_TRANSACTIONS).astype(np.float64)
        # Inject ~5-20% NaN depending on column
        nan_frac = np.random.uniform(0.05, 0.20)
        c_nan_mask = np.random.random(N_TRANSACTIONS) < nan_frac
        c_vals[c_nan_mask] = np.nan
        c_cols[f"C{i}"] = c_vals

    # -- 8. D columns (time-delta features): D1-D10 --
    d_cols = {}
    for i in range(1, 11):
        d_vals = np.random.exponential(scale=50, size=N_TRANSACTIONS).astype(np.float64)
        # D columns have more missing data (~20-40%)
        nan_frac = np.random.uniform(0.20, 0.40)
        d_nan_mask = np.random.random(N_TRANSACTIONS) < nan_frac
        d_vals[d_nan_mask] = np.nan
        d_cols[f"D{i}"] = d_vals

    # -- 9. V columns (Vesta engineered features): V1-V30 --
    v_cols = {}
    for i in range(1, 31):
        v_vals = np.random.normal(0, 1, size=N_TRANSACTIONS).astype(np.float64)
        # V columns have high missing rates (~30-50%)
        nan_frac = np.random.uniform(0.30, 0.50)
        v_nan_mask = np.random.random(N_TRANSACTIONS) < nan_frac
        v_vals[v_nan_mask] = np.nan
        v_cols[f"V{i}"] = v_vals

    # -- 10. Product code and email matches --
    product_cd = np.random.choice(["W", "H", "C", "S", "R"],
                                  size=N_TRANSACTIONS, p=[0.7, 0.1, 0.1, 0.05, 0.05])

    # -- 11. Build transaction DataFrame --
    txn_data = {
        "TransactionID": np.arange(1, N_TRANSACTIONS + 1),
        "TransactionDT": dt_values,
        "TransactionAmt": base_amt,
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3.astype(np.float64),  # float to allow NaN in other datasets
        "card4": card4,
        "card5": card5.astype(np.float64),
        "card6": card6,
        "addr1": addr1,
        "addr2": np.random.choice([87.0, 60.0, 96.0, np.nan],
                                  size=N_TRANSACTIONS, p=[0.6, 0.2, 0.1, 0.1]),
        "P_emaildomain": p_email_series,
        "R_emaildomain": r_email,
        "isFraud": is_fraud,
    }

    # Add C, D, V columns
    txn_data.update(c_cols)
    txn_data.update(d_cols)
    txn_data.update(v_cols)

    txn_df = pd.DataFrame(txn_data)
    print(f"  Transaction table: {txn_df.shape}")

    # -- 12. Build identity DataFrame (~40% of transactions have identity) --
    n_ident = int(N_TRANSACTIONS * 0.40)
    ident_ids = np.sort(np.random.choice(
        txn_df["TransactionID"].values, size=n_ident, replace=False
    ))

    # DeviceType: common types everywhere, "smartwatch" only in late period
    device_types = ["desktop", "mobile"]
    ident_device = np.random.choice(device_types, size=n_ident, p=[0.6, 0.4])

    # Inject late-only device type for identity rows in the late period
    ident_dt = txn_df.set_index("TransactionID").loc[ident_ids, "TransactionDT"].values
    late_ident = ident_dt >= time_80pct
    late_ident_idx = np.where(late_ident)[0]
    if len(late_ident_idx) > 20:
        inject_dev = np.random.choice(late_ident_idx, size=20, replace=False)
        ident_device[inject_dev] = "smartwatch"

    device_info = np.random.choice(
        ["Windows", "MacOS", "iOS", "Android", "Linux", None],
        size=n_ident, p=[0.3, 0.15, 0.2, 0.2, 0.05, 0.1]
    )

    # id_01 to id_06: numeric identity features
    id_cols = {}
    for i in range(1, 7):
        id_vals = np.random.normal(0, 30, size=n_ident).astype(np.float64)
        nan_frac = np.random.uniform(0.1, 0.3)
        id_nan_mask = np.random.random(n_ident) < nan_frac
        id_vals[id_nan_mask] = np.nan
        id_cols[f"id_{i:02d}"] = id_vals

    ident_data = {
        "TransactionID": ident_ids,
        "DeviceType": ident_device,
        "DeviceInfo": device_info,
    }
    ident_data.update(id_cols)

    ident_df = pd.DataFrame(ident_data)
    print(f"  Identity table:    {ident_df.shape}")

    # -- 13. Save to CSV (matching Kaggle format) --
    txn_path = os.path.join(OUT_DIR, "train_transaction.csv")
    ident_path = os.path.join(OUT_DIR, "train_identity.csv")

    txn_df.to_csv(txn_path, index=False)
    ident_df.to_csv(ident_path, index=False)

    print(f"\n  Saved: {txn_path} ({os.path.getsize(txn_path) / 1024:.0f} KB)")
    print(f"  Saved: {ident_path} ({os.path.getsize(ident_path) / 1024:.0f} KB)")
    print(f"  Fraud rate: {is_fraud.mean() * 100:.2f}%")
    print(f"  Late-only email domains: {LATE_EMAIL_DOMAINS}")
    print(f"  Late-only device types: {LATE_DEVICE_TYPES}")
    print("=" * 60)


if __name__ == "__main__":
    main()
