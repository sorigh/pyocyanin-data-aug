#!/bin/bash
# Run tune_dl_hparams.py (MLP + CNN) pentru toate cele 4 seturi de date.
# MLP se ruleaza o data per (conditie, suita de features) = 12 containere.
# CNN se ruleaza o data per conditie = 4 containere. 
# Total 16 containere, distribuite round-robin 
# (PARALLEL = numarul de GPU-uri, ca fiecare container sa aiba GPU-ul
# lui, nu sa il imparta cu altul).
#
# Fisiere pe server:
# /home/soragi/pyocyanin/vectorized/
# /home/soragi/pyocyanin/raw/:
# vectorized/gan_only_core.csv, gan_only_extended.csv, gan_only_experimental.csv
# vectorized/combined_all_core.csv, combined_all_extended.csv, combined_all_experimental.csv
# raw/gan_only_raw_signals.csv
# raw/combined_all_raw_signals.csv
#
# (pachetele trb instalate o singura data ca sa nu dea crash)
#
# Rulare: bash dgx_run_dl_sweep.sh

set -e

GPU_DEVICES=(1 2 3)   
PARALLEL=${#GPU_DEVICES[@]}     # containere paralele
N_TRIALS=30     # redus de la 100
MAX_EPOCHS=100  # redus de la 300 
EVAL_N_TRIALS=8 # redus de la 20 - cate incercari Optuna per "fold" la evaluarea nested

CONDITIONS="lab physics_aug gan combined"
SUITES="core extended experimental"

declare -A EVAL_CV
# "lab" foloseste kfold=5 aici, NU loo (40 fold-uri), (la ML lab foloseste loo)
# a DL, fiecare "fold" reface o cautare Optuna (bucla loo dureaza mult) 
EVAL_CV[lab]="kfold"
EVAL_CV[physics_aug]="kfold"
EVAL_CV[gan]="kfold"
EVAL_CV[combined]="kfold"

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

declare -A SIGNALS_CSV
SIGNALS_CSV[lab]="raw/raw_signals_real.csv"
SIGNALS_CSV[physics_aug]="raw/raw_signals_augmented.csv"
SIGNALS_CSV[gan]="raw/gan_only_raw_signals.csv"
SIGNALS_CSV[combined]="raw/combined_all_raw_signals.csv"

running=()
gpu_counter=0

wait_for_slot() {
    while [ "${#running[@]}" -ge "$PARALLEL" ]; do
        docker wait "${running[0]}" > /dev/null 2>&1 || true
        running=("${running[@]:1}")
    done
}

# kind: "mlp" (foloseste --features-csv) sau "cnn" (foloseste --signals-csv)
start_run() {
    local condition=$1 kind=$2 data_csv=$3 suite_suffix=$4
    local eval_cv=${EVAL_CV[$condition]}
    local name="soragi_dl_${condition}${suite_suffix}_${kind}"
    local out_json="models/${condition}${suite_suffix}_${kind}_best_params.json"
    local log_csv="logs/${condition}${suite_suffix}_${kind}_studies.csv"
    local models_dir="models/${condition}${suite_suffix}_${kind}"
    local data_flag="--features-csv ${data_csv}"
    if [ "$kind" == "cnn" ]; then
        data_flag="--signals-csv ${data_csv}"
    fi
    local gpu=${GPU_DEVICES[$((gpu_counter % ${#GPU_DEVICES[@]}))]}
    gpu_counter=$((gpu_counter + 1))

    wait_for_slot
    echo "Pornesc: $name  (data=$data_csv, eval-cv=$eval_cv, gpu=$gpu)"
    docker rm -f "$name" > /dev/null 2>&1 || true   # clean old container
    docker run --gpus all --name "$name" -d -u "$(id -u):$(id -g)" \
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
        -v /home/soragi/pyocyanin:/app \
        -v /mnt/QNAP/soragi/models:/app/models \
        -v /mnt/QNAP/soragi/logs:/app/logs \
        -v /mnt/QNAP/soragi/.local_packages:/.local_packages \
        -e CUDA_VISIBLE_DEVICES=${gpu} -e USER=soragi -e PYTHONUNBUFFERED=1 \
        -e PYTHONUSERBASE=/.local_packages -e PATH=/.local_packages/bin:$PATH -w /app \
        nvcr.io/nvidia/pytorch:23.10-py3 bash -c \
        "python tune_dl_hparams.py ${data_flag} --models ${kind} \
             --n-trials ${N_TRIALS} --max-epochs ${MAX_EPOCHS} \
             --eval-cv ${eval_cv} --eval-splits 5 --eval-n-trials ${EVAL_N_TRIALS} \
             --output ${out_json} --log-csv ${log_csv} --models-dir ${models_dir}" > /dev/null
    running+=("$name")
}

# --- MLP: o rulare per (conditie, suita) ---
for condition in $CONDITIONS; do
    for suite in $SUITES; do
        start_run "$condition" "mlp" "${FEATURES_CSV["${condition}_${suite}"]}" "_${suite}"
    done
done

# --- CNN: o rulare per conditie (nu depinde de suita) ---
for condition in $CONDITIONS; do
    start_run "$condition" "cnn" "${SIGNALS_CSV[$condition]}" ""
done

for name in "${running[@]}"; do
    docker wait "$name" > /dev/null 2>&1 || true
done

echo ""
echo "FINISHED."
echo "Results in /mnt/QNAP/soragi/models/ (fisiere *_best_params.json + checkpoint-uri .pt)"
echo "and /mnt/QNAP/soragi/logs/ (fisiere *_studies.csv)."
echo ""
