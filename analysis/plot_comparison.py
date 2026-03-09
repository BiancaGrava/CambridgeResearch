import pandas as pd
import matplotlib.pyplot as plt

b = pd.read_csv("results/bandit_summary.csv")
l = pd.read_csv("results/llego_original.csv")

df = b.merge(l, on="dataset", suffixes=("_bandit", "_llego"))

plt.scatter(df["accuracy_llego"], df["accuracy_bandit"])
plt.plot([0.5,1.0],[0.5,1.0],"k--")
plt.xlabel("LLEGO Accuracy")
plt.ylabel("BanditLLEGGO Accuracy")
plt.title("Accuracy Comparison")
plt.savefig("results/accuracy_comparison.png")