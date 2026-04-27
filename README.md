# BanditLLEGGO — Two‑Phase Pipeline (Rules → Tree)

An MVP for my research paper designed as part of my past Cambridge PhD Application.

## What the two phases emulate

- **Phase‑1 (“rules”)**  
  The bandit is restricted to *new_rules* / *refine_rules*. GP inner tree steps are skipped, so we spend no GA budget on trees. We filter proposals by **support**, **utility**, **ECE**, and optional fairness. We then **freeze Top‑K** rules to form \(S^\*\) and save them to `results/<exp>_Sstar.json`.

- **Phase‑2 (“tree”)**  
  We **lock** the rule pool to \(S^\*\) and disable the LLM (pure GP). We Pittsburgh‑seed from \(S^\*\) and use \(S^\*\) in rule‑aware mutation/crossover to assemble the final decision tree \(T^\*\). Metrics are recorded: **Balanced Accuracy** / **ECE** for classification, **RMSE** for regression, plus **depth**, **n\_params**, and **llm\_calls** (which is 0 in Phase‑2).


1. Setup
Create and activate a new environment with conda (with Python 3.9 or newer).
conda create -n llego python=3.9
conda activate llego
Install the necesary requirements (for LLEGO).
pip install -e .
Install the external libraries (for the baselines).
bash install_external.sh
Our code allows for logging via wandb. If you want to use it, make sure it is correctly configured on your machine by following this guide.
2. Reproducing Results
Set up LLM credentials. Our implementation uses the OpenAI family of models via API queries. Add your credentials to configs/endpoint/default.yaml.

## Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .
pip install matplotlib
