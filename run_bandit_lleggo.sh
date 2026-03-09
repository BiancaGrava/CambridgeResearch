$datasets = "abalone","breast","cars","cholesterol","compas","credit-g","cutract","diabetes","heart-statlog","liver","maggic","seer","vehicle","wage","wine"

foreach ($ds in $datasets) {
    python experiments/exp_bandit_lleggo.py alg.dataset.name=$ds alg.exp_name="bandit_$ds"
}