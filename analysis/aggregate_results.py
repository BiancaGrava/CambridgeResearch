# analysis/aggregate_results.py
import os, json, pandas as pd

rows = []
os.makedirs("results", exist_ok=True)

for f in os.listdir("results"):
    if f.endswith("_metrics.json"):
        path = os.path.join("results", f)
        try:
            d = json.load(open(path))
            d["filename"] = f
            rows.append(d)
        except Exception:
            pass

df = pd.DataFrame(rows)
df.to_csv("results/all_results.csv", index=False)
print("Wrote results/all_results.csv")
print(df.head(12))