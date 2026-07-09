#!/bin/bash
# Ruleaza tune_ml_hparams.py pt toate cele 4 seturi de date (lab, physics_aug,
# gan, combined) x 3 seturi de features (core, extended, experimental) = 24
# containere, in paralel cu  --output/--log-csv unic ca sa nu se suprascrie).
#
#  GridSearchCV pe ridge, elastic_net, decision_tree, random_forest
#     (spatii mici/discrete - grid gaseste mai rapid)
#  BayesSearchCV pe svr, xgboost (spatii mari/continue unde Bayes ajuta)
#
# FISIERE !!!
# in /home/soragi/pyocyanin/vectorized/:
#   gan_only_core.csv, gan_only_extended.csv, gan_only_experimental.csv
#   combined_all_core.csv, combined_all_extended.csv, combined_all_experimental.csv
#
# (pachete instalate separat ca sa nu se blocheze)
#   docker run --name soragi_setup --rm -u $(id -u):$(id -g) \
#       -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
#       -v /home/soragi/pyocyanin:/app -v /mnt/QNAP/soragi/.local_packages:/.local_packages \
#       -e USER=soragi -e PYTHONUNBUFFERED=1 -e PYTHONUSERBASE=/.local_packages \
#       -e PATH=/.local_packages/bin:$PATH -w /app \
#       nvcr.io/nvidia/pytorch:23.10-py3 bash -c \
#       "pip install --user --no-cache-dir scikit-learn>=1.8.0 xgboost>=3.2.0 scikit-optimize>=0.10.2 optuna>=4.0.0"
#
# Rulare: bash dgx_run_ml_sweep.sh

set -e

PARALLEL=4      
N_JOBS=4        
N_ITER=12       #redus de la 40

CONDITIONS="lab physics_aug gan combined"
SUITES="core extended experimental"
GRID_MODELS="ridge elastic_net decision_tree random_forest"
BAYES_MODELS="svr xgboost"

# combinatii "conditie_suita_metoda" sa fie sarite
# (l-am rulat manual pe DGX nu prin script)
SKIP_RUNS=" lab_core_bayes "

declare -A OUTER_CV
OUTER_CV[lab]="loo"
OUTER_CV[physics_aug]="kfold"
OUTER_CV[gan]="kfold"
OUTER_CV[combined]="kfold"

declare -A FEATURES_CSV
FEATURES_CSV[lab_core]="vectorized/core.csv"
FEATURES_CSV[lab_extended]="vectorized/extended.csv"
FEATURES_CSV[lab_experimental]="vectorized/experimental.csv"
FEATURES_CSV[physics_aug_core]="vectorized/full_augmented_core.csv"
FEATURES_CSV[physics_aug_extended]="vectorized/full_augmented_extended.csv"
FEATURES_CSV[physics_aug_experimental]="vectorized/full_augmented_experimental.csv"
FEATURES_CSV[gan_core]="vectorized/gan_only_core.csv"
FEATURES_CSV[gan_extended]="vectorized/gan_only_extended.csv"
FEATURES_CSV[gan_experimental]="vectorized/gan_only_experimental.csv"
FEATURES_CSV[combined_core]="vectorized/combined_all_core.csv"
FEATURES_CSV[combined_extended]="vectorized/combined_all_extended.csv"
FEATURES_CSV[combined_experimental]="vectorized/combined_all_experimental.csv"

running=()

wait_for_slot() {
    while [ "${#running[@]}" -ge "$PARALLEL" ]; do
        docker wait "${running[0]}" > /dev/null 2>&1 || true
        running=("${running[@]:1}")
    done
}

start_run() {
    local condition=$1 suite=$2 method=$3 models=$4 n_iter=$5
    local key="${condition}_${suite}_${method}"
    if [[ "$SKIP_RUNS" == *" ${key} "* ]]; then
        echo "Sar peste: soragi_ml_${key}  (deja ruleaza manual, nu-l ating)"
        return
    fi
    local outer_cv=${OUTER_CV[$condition]}
    local features=${FEATURES_CSV["${condition}_${suite}"]}
    local name="soragi_ml_${condition}_${suite}_${method}"
    local out_json="models/${condition}_${suite}_${method}_best_params.json"
    local log_csv="logs/${condition}_${suite}_${method}_nested_cv.csv"

    wait_for_slot
    echo "Pornesc: $name  (features=$features, outer-cv=$outer_cv, search=$method, models=$models)"
    docker rm -f "$name" > /dev/null 2>&1 || true   # curata un container ramas dintr-o rulare anterioara cu acelasi nume
    docker run --name "$name" -d -u "$(id -u):$(id -g)" \
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
        -v /home/soragi/pyocyanin:/app \
        -v /mnt/QNAP/soragi/models:/app/models \
        -v /mnt/QNAP/soragi/logs:/app/logs \
        -v /mnt/QNAP/soragi/.local_packages:/.local_packages \
        -e USER=soragi -e PYTHONUNBUFFERED=1 -e PYTHONUSERBASE=/.local_packages \
        -e PATH=/.local_packages/bin:$PATH -w /app \
        nvcr.io/nvidia/pytorch:23.10-py3 bash -c \
        "python tune_ml_hparams.py --features-csv ${features} \
             --models ${models} \
             --outer-cv ${outer_cv} --outer-splits 5 --inner-splits 3 \
             --search-method ${method} --n-iter ${n_iter} --n-jobs ${N_JOBS} \
             --output ${out_json} --log-csv ${log_csv}" > /dev/null
    running+=("$name")
}

for condition in $CONDITIONS; do
    for suite in $SUITES; do
        start_run "$condition" "$suite" "grid"  "$GRID_MODELS"  "$N_ITER"
        start_run "$condition" "$suite" "bayes" "$BAYES_MODELS" "$N_ITER"
    done
done

for name in "${running[@]}"; do
    docker wait "$name" > /dev/null 2>&1 || true
done

echo ""
echo "FINISHED."
echo "Results in /mnt/QNAP/soragi/models/ (fisiere *_best_params.json)"
echo " and /mnt/QNAP/soragi/logs/ (fisiere *_nested_cv.csv)."
echo ""