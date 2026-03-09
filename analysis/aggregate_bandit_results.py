import json, os, pandas as pd
rows = []

for file in os.listdir("results"):
    if file.endswith("_metrics.json"):
        with open(os.path.join("results", file)) as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows)
df.to_csv("results/bandit_summary.csv", index=False)
print(df)