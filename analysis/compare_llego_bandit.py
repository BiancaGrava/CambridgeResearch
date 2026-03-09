# analysis/compare_llego_bandit.py
import os, pandas as pd, matplotlib.pyplot as plt

csv_path = "results/all_results.csv"
if not os.path.exists(csv_path):
    raise SystemExit("No results/all_results.csv. Run aggregate_results.py first.")

df = pd.read_csv(csv_path)

# If only bandit files exist, just summarize by dataset.
is_bandit = df["filename"].str.startswith("bandit")
is_llego  = df["filename"].str.startswith("llego")

if is_llego.any() and is_bandit.any():
    bandit = df[is_bandit].copy()
    llego  = df[is_llego].copy()
    merged = bandit.merge(llego, on="dataset", suffixes=("_bandit","_llego"))

    # Balanced accuracy parity plot (if available)
    if {"balanced_accuracy_bandit","balanced_accuracy_llego"} <= set(merged.columns):
        plt.figure(figsize=(6,6))
        plt.scatter(merged["balanced_accuracy_llego"], merged["balanced_accuracy_bandit"], s=60)
        plt.plot([0.5,1],[0.5,1],"k--")
        plt.xlabel("LLEGO Balanced Accuracy")
        plt.ylabel("BanditLLEGGO Balanced Accuracy")
        plt.title("Balanced Accuracy Parity")
        plt.savefig("results/ba_parity.png")
        print("Saved results/ba_parity.png")
else:
    # Bandit-only summary
    keep_cols = [c for c in df.columns if c in {"dataset","balanced_accuracy","accuracy","ece","rmse","depth","n_params","runtime","llm_calls"}]
    summary = df[keep_cols].groupby("dataset").mean(numeric_only=True).reset_index()
    summary.to_csv("results/bandit_only_summary.csv", index=False)
    print("Wrote results/bandit_only_summary.csv")