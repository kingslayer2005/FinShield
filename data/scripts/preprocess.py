# data/scripts/preprocess.py
import pandas as pd
import numpy as np
import os

print("FinShield Data Preprocessing Started...")

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

print("Loading datasets...")
transactions = pd.read_csv(f"{RAW}/train_transaction.csv")
identity     = pd.read_csv(f"{RAW}/train_identity.csv")

print(f"   Transactions shape : {transactions.shape}")
print(f"   Identity shape     : {identity.shape}")

print("Merging datasets...")
df = transactions.merge(identity, on="TransactionID", how="left")
print(f"   Merged shape       : {df.shape}")


fraud_rate = df["isFraud"].mean() * 100
print(f"\nFraud rate: {fraud_rate:.2f}%")
print(f"   Fraud cases    : {df['isFraud'].sum():,}")
print(f"   Legit cases    : {(df['isFraud'] == 0).sum():,}")


print("\nDropping high-missing columns (>50% missing)...")
before = df.shape[1]
thresh = len(df) * 0.5
df = df.dropna(axis=1, thresh=thresh)
print(f"   Columns before : {before}")
print(f"   Columns after  : {df.shape[1]}")


print("Filling missing values...")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna("unknown")

print("Engineering features...")


df["Transaction_Hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)


df["Transaction_Day"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)


df["Is_High_Value"] = (df["TransactionAmt"] > 500).astype(int)


df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"])


df["Amt_vs_Mean"] = df.groupby("card1")["TransactionAmt"].transform(
    lambda x: x - x.mean()
)

print(f"   New features added : Transaction_Hour, Transaction_Day,")
print(f"                        Is_High_Value, TransactionAmt_Log, Amt_vs_Mean")


print("Encoding categorical columns...")
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes


print("\nSaving processed data...")
df.to_csv(f"{OUT}/finshield_processed.csv", index=False)


df.sample(n=10000, random_state=42).to_csv(
    f"{OUT}/finshield_sample.csv", index=False
)

print(f"\nDone!")
print(f"   Full dataset : data/processed/finshield_processed.csv")
print(f"   Sample       : data/processed/finshield_sample.csv")
print(f"   Final shape  : {df.shape}")
print(f"   Columns      : {list(df.columns[:10])} ...")