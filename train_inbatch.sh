#!/bin/bash
#SBATCH --job-name=MixBLINK_train
#SBATCH -p gpu_long
#SBATCH --gres=gpu:a6000:1

export WANDB_PROJECT=MixBLINK_EL

config_file=configs/config_inbatch.yaml
output_dir=save_models/bert-base-uncased
seed=0
measure='ip'

base_output_dir=${output_dir}/${seed}/${measure}
mkdir -p ${base_output_dir}

uv run python mix_blink/cli/run.py \
    --negative False \
    --config_file ${config_file} \
    --measure ${measure} \
    --seed ${seed} \
    --output_dir ${base_output_dir}/inbatch \
    --run_name ${base_output_dir}/inbatch
