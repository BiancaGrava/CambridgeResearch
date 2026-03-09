# run_all.ps1
# ============================
# RUN BANDIT-LLEGO
# ============================

Write-Host "=== Running BanditLLEGGO ==="

$datasets = Get-Content "./datasets.txt"

foreach ($ds in $datasets) {
    Write-Host "`n[Bandit] Running dataset = $ds"
    # NOTE: Removed "dataset=$ds" (Hydra group override).
    python experiments/exp_bandit_lleggo.py `
        alg.dataset.name=$ds `
        alg.exp_name="bandit_lleggo_$ds"
}

# ============================
# AGGREGATE + COMPARE
# ============================

Write-Host "`n=== Aggregating Results ==="
python analysis/aggregate_results.py

Write-Host "`n=== Producing Comparison Figures ==="
python analysis/compare_llego_bandit.py

Write-Host "`n=== DONE! Results in results/ folder ==="