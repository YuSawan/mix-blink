#!/bin/bash
#SBATCH --job-name=MixBLINK_train
#SBATCH -p gpu_long
#SBATCH --gres=gpu:a6000:1

export WANDB_PROJECT=MixBLINK_EL

config_file=configs/config.yaml
output_dir=save_models/bert-base-uncased

for seed in 0 21 42; do
for measure in 'cos' 'ip' 'cos'; do

base_output_dir=${output_dir}/${seed}/${measure}
mkdir -p ${base_output_dir}

uv run python luke_el/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file ${config_file} \
    --measure ${measure} \
    --seed ${seed} \
    --output_dir ${base_output_dir}/first \
    --run_name ${base_output_dir}/first

uv run python luke_el/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --negative 'dense' \
    --config_file ${config_file} \
    --measure ${measure} \
    --seed ${seed} \
    --prev_path ${base_output_dir}/first \
    --output_dir ${base_output_dir}/second \
    --run_name ${base_output_dir}/second

done
done
