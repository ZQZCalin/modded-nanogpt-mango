#!/bin/bash

lr=0.05
momentum="0.85,0.95,300"
name="muon_lr${lr}_mom${momentum}"
project="QZ_test_grad_noise"

args=(
    # basic configs
    "--run_name ${name}"
    "--wandb_project ${project}"
    "--log_folder check_grad_noise"
    "--random_seed 42"
    # some unrelated configs for convenience
    "--compile_only False"  # turn on to warmup the node (for the first run)
    "--advanced_log False"  # turn on to log rms norms
    # optimizer configs
    "--optimizer muon"
    "--lr ${lr}"
    "--momentum ${momentum}"
)
torchrun --standalone --nproc_per_node=1 experiments/gradient_noise/train_gradient_noise_ckpt.py ${args[@]}