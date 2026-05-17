# data/scripts/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import os

print("FinShield EDA Starting...")

df = pd.read_csv("data/processed/finshield_processed.csv")
os.makedirs("docs", exist_ok=True)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("FinShield — Fraud Analysis", fontsize=14, fontweight="bold")


counts = df["isFraud"].value_counts()
axes[0].bar(["Legit", "Fraud"], counts.values, color=["#00ff88", "#ff4444"])
axes[0].set_title("Fraud vs Legit Transactions")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')


df[df["isFraud"]==0]["TransactionAmt"].clip(0,1000).hist(
    bins=50, ax=axes[1], color="#00ff88", alpha=0.7, label="Legit")
df[df["isFraud"]==1]["TransactionAmt"].clip(0,1000).hist(
    bins=50, ax=axes[1], color="#ff4444", alpha=0.7, label="Fraud")
axes[1].set_title("Transaction Amount Distribution")
axes[1].set_xlabel("Amount ($)")
axes[1].legend()


fraud_by_hour = df.groupby("Transaction_Hour")["isFraud"].mean() * 100
axes[2].plot(fraud_by_hour.index, fraud_by_hour.values,
             color="#ff4444", linewidth=2, marker="o", markersize=4)
axes[2].set_title("Fraud Rate by Hour of Day")
axes[2].set_xlabel("Hour")
axes[2].set_ylabel("Fraud Rate (%)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("docs/eda_analysis.png", dpi=150, bbox_inches="tight")
print("EDA chart saved to docs/eda_analysis.png")


print(f"\nKey Insights:")
print(f"   Total transactions  : {len(df):,}")
print(f"   Fraud rate          : {df['isFraud'].mean()*100:.2f}%")
print(f"   Avg fraud amount    : ${df[df['isFraud']==1]['TransactionAmt'].mean():.2f}")
print(f"   Avg legit amount    : ${df[df['isFraud']==0]['TransactionAmt'].mean():.2f}")
peak_hour = fraud_by_hour.idxmax()
print(f"   Peak fraud hour     : {peak_hour}:00 ({fraud_by_hour[peak_hour]:.2f}% fraud rate)")