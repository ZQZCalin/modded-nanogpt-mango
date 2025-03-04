#!/bin/bash

# -----------------------------------------------------------------------------
# Experiment configs
# -> The following should be used as the muon baseline on single L40S gpu
# seeds=(42 21 1009 324 5646)
# seed=${seeds[0]}
# name="muon_seed${seed}"

# args=(
#     "--log_folder muon"
#     "--run_name ${name}"
#     "--random_seed ${seed}"
# )

# script="train"

# -> Reproducing 0201 record on single gpu (with specified random seeds)
# seeds=(42 21 1009 324 5646)
# seed=${seeds[4]}
# name="muon0201_seed${seed}"

# args=(
#     "--log_folder muon0201_record"
#     "--run_name ${name}"
#     "--random_seed ${seed}"
# )
# script="train_record_feb1"

# -----------------------------------------------------------------------------
# Alternative optimizers
script="train"

# precmuon (we've seen this is better on JAX)
# beta2=0.95
# # rms=True
# # lrs=(0.0018 0.003 0.001
# #     6e-4 3e-4 1e-4)
# rms=False
# lrs=(0.05 0.1 0.025
#     0.01 6e-3 3e-3 1e-3)
# lr=${lrs[2]}
# name="precmuon_rms-${rms}_lr${lr}"

# muon (as a baseline)
# beta2=0.0
# # rms=True
# # lrs=(0.0018 0.003 0.001)
# rms=False
# lrs=(0.05 0.1 0.025)
# lr=${lrs[2]}
# name="muon_rms-${rms}_lr${lr}"

# args=(
#     # basic configs
#     "--log_folder muon_rms"
#     "--random_seed 42"
#     "--run_name ${name}"
#     # optimizer configs
#     "--optimizer mango"
#     "--mango_mat_lr ${lr}"
#     "--mango_mat_beta2 ${beta2}"
#     "--mango_mat_scale_rms ${rms}"
#     "--mango_mat_precond_power 0.5"
# )

# different betas
betas=(0.0 0.9 0.95 0.99)
beta2=${betas[0]}
rms=True
lrs=(0.0018 0.003 0.001)
# rms=False
# lrs=(0.01 0.025 0.05 0.075 0.1)
# lr=${lrs[0]}
lr=7.5e-4
name="muon_rms-${rms}_beta2${beta2}_lr${lr}"

args=(
    # basic configs
    "--run_name ${name}"
    "--log_folder finetune_muon"
    "--random_seed 42"
    # optimizer configs
    "--optimizer mango"
    "--mango_mat_lr ${lr}"
    "--mango_mat_beta2 ${beta2}"
    "--mango_mat_scale_rms ${rms}"
    "--mango_mat_precond_power 0.5"
)


# -----------------------------------------------------------------------------
# Redirect SCC outputs.
BASE_DIR=/projectnb/aclab/qinziz/nanogpt-mango      # change your base path here
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE
mkdir -p $OUTPUT_PATH

# -----------------------------------------------------------------------------
# Submit job to SCC.
GPU=L40S
NODES=1
mode=0
# mode=1      # uncomment to run locally instead of submit to scc
# mode=2      # uncomment to run batch submits

submit_job() {
    local args=("$@")
    
    job_output=$(qsub <<EOF
#!/bin/bash -l
#$ -pe omp 8
#$ -l h="!scc-506"          # Blacklists bad nodes
#$ -l gpus=${NODES}
#$ -l gpu_type=${GPU}       # Specifies the gpu type
#$ -l h_rt=8:00:00          # Specifies the hard time limit for the job
#$ -N "$name".sh
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

source activate_env.sh
torchrun --standalone --nproc_per_node=${NODES} ${script}.py ${args[@]}
EOF
    )

    # Extract job id and log the submission.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $name"
}

if [[ $mode -eq 1 ]]; then
    torchrun --standalone --nproc_per_node=${NODES} ${script}.py ${args[@]}
elif [[ $mode -eq 0 ]]; then
    submit_job ${args[@]}
fi

# -----------------------------------------------------------------------------
# Batch submit if necessary

# betas=(0.0 0.9 0.95 0.99)
# lrs=(3e-3 2e-3 1e-3 7.5e-4 5e-4 2.5e-4)
# rms=True

# betas=(0.9)
# lrs=(0.01 0.05 0.75 0.1)
# rms=False

# for beta2 in "${betas[@]}"; do
#     for lr in "${lrs[@]}"; do
#         name="muon_rms-${rms}_beta2${beta2}_lr${lr}"
#         args=(
#             # basic configs
#             "--run_name ${name}"
#             "--log_folder finetune_muon"
#             "--random_seed 42"
#             # optimizer configs
#             "--optimizer mango"
#             "--mango_mat_lr ${lr}"
#             "--mango_mat_beta2 ${beta2}"
#             "--mango_mat_scale_rms ${rms}"
#             "--mango_mat_precond_power 0.5"
#         )
#         submit_job ${args[@]}
#     done
# done