#!/bin/bash
#SBATCH --job-name=MixBLINK_train
#SBATCH -p gpu_long
#SBATCH --gres=gpu:a6000:1

export WANDB_PROJECT=MixBLINK_EL

config_file=configs/config.yaml
output_dir=save_models/bert-base-uncased
seed=0
measure='ip'

base_output_dir=${output_dir}/${seed}/${measure}
mkdir -p ${base_output_dir}

uv run torchrun mix_blink/cli/train.py \
    --config_file ${config_file} \
    --measure ${measure} \
    --output_dir ${base_output_dir}/inbatch \
    --run_name ${base_output_dir}/inbatch \
    --seed ${seed}
