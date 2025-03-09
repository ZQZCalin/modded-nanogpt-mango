#!/bin/bash

# -----------------------------------------------------------------------------
# Alternative optimizers
script="train"

lr=0.05
beta1="0.85,0.95,300"
beta2="0,0.95,300"
rms=False
grafting=True
cond=True
laprop=False
p_pre=0.5
p_post=0.0

name=test_mango

DATE=$(date +"%Y-%m-%d")
args=(
    # basic configs
    "--run_name ${name}"
    "--wandb_project nanogpt_speedrun"  # comment out this line to use default project name
    "--log_folder mango_${DATE}"
    "--random_seed 42"
    # optimizer configs
    "--optimizer mango"
    "--mango_mat_lr ${lr}"
    "--mango_mat_beta1 ${beta1}"
    "--mango_mat_beta2 ${beta2}"
    "--mango_mat_scale_rms ${rms}"
    "--mango_mat_grafting ${grafting}"
    "--mango_mat_use_cond ${cond}"
    "--mango_mat_laprop ${laprop}"
    "--mango_mat_precond_power ${p_pre}"
    "--mango_mat_postcond_power ${p_post}"
    # some unrelated configs for convenience
    "--compile_only False"  # turn on to warmup the node (for the first run)
    "--advanced_log False"  # turn on to log rms norms
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
mode=1      # uncomment to run locally instead of submit to scc
# mode=2      # uncomment to run batch submits

submit_job() {
    local args=("$@")
    
    job_output=$(qsub <<EOF
#!/bin/bash -l
#$ -pe omp 8
#$ -l h="!scc-506"          # Blacklists bad nodes
#$ -l gpus=${NODES}
#$ -l gpu_type=${GPU}       # Specifies the gpu type
#$ -l h_rt=2:00:00          # Specifies the hard time limit for the job
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
elif [[ $mode -ne 2 ]]; then
    exit    # quit if not for batch submit
fi

# -----------------------------------------------------------------------------
# Below is batch submit script

# Batch submit 1: pre-conditionings
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

# 2. Schedule-free muon
# list_momentum=(
#     "0.85,0.95"
# )
# list_nesterov=(
#     # "0.85,0.97"
#     # "0.9,0.97"
#     # "0.85,0.95"
#     # "0.85,0.9"
#     # "0.8,0.9"
#     # "0.85,0.85"
#     # "0.75,0.85"
#     "0.85,0.98"
#     "0.9,0.98"
#     "0.95,0.98"
#     "0.85,0.99"
#     "0.9,0.99"
#     "0.95,0.99"
#     "0.85,1"
#     "0.9,1"
#     "0.95,1"
#     "1,1"
# )
# list_warmup=(300)
# list_lr=(0.05 0.04 0.06 0.03 0.07)
# for mom in "${list_momentum[@]}"; do
#     for nes in "${list_nesterov[@]}"; do
#         for warmup in "${list_warmup[@]}"; do
#             for lr in "${list_lr[@]}"; do
#                 momentum="${mom},${warmup}"
#                 nesterov="${nes},${warmup}"
#                 name="sfmuon_mom${momentum}_nes${nesterov}_lr${lr}"
#                 args=(
#                     # basic configs
#                     "--run_name ${name}"
#                     "--wandb_project nanogpt_speedrun"  # comment out this line to use default project name
#                     "--log_folder sfmuon"
#                     "--random_seed 42"
#                     # optimizer configs
#                     "--optimizer sfmuon"
#                     "--sfmuon_lr ${lr}"
#                     "--sfmuon_momentum ${momentum}"
#                     "--sfmuon_nesterov_beta ${nesterov}"
#                 )
#                 # echo $name ${args[@]}
#                 submit_job ${args[@]}
#             done
#         done
#     done
# done