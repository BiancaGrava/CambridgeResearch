
$datasets = "abalone","breast","cars","cholesterol","compas","credit-g","cutract","diabetes","heart-statlog","liver","maggic","seer","vehicle","wage","wine"

foreach ($ds in $datasets) {
    python experiments/exp_llego.py dataset=$ds training.outer_rounds=5 exp_name=llego_$ds
}
