#!/bin/bash

script="experiments/baseline/train_baseline.py"

# -----------------------------------------------------------------------------
# Configs.
GPU=L40S
NODES=1
BASE_DIR=/projectnb/aclab/qinziz/nanogpt-mango      # change your base path here
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE
mkdir -p $OUTPUT_PATH

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

cd ${BASE_DIR}
source activate_env.sh
torchrun --standalone --nproc_per_node=${NODES} ${script} ${args[@]}
EOF
    )

    # Extract job id and log the submission.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $name"
}

# -----------------------------------------------------------------------------
# Experiments
mode=0      # 0 for local run; 1 for single job qsub; 2 for customized batch submit

lr=0.05
momentum="0.85,0.95,300"
name="muon_lr${lr}"
project="nanogpt_speedrun"

DATE=$(date +"%Y-%m-%d")
args=(
    # basic configs
    "--run_name ${name}"
    "--wandb_project ${project}"
    "--log_folder muon_${DATE}"
    "--random_seed 42"
    # some unrelated configs for convenience
    "--compile_only False"  # turn on to warmup the node (for the first run)
    "--advanced_log False"  # turn on to log rms norms
    # optimizer configs
    "--optimizer muon"
    "--lr ${lr}"
    "--momentum ${momentum}"
)

if [[ $mode -eq 0 ]]; then
    torchrun --standalone --nproc_per_node=${NODES} ${script} ${args[@]}
elif [[ $mode -eq 1 ]]; then
    submit_job ${args[@]}
elif [[ $mode -eq 2 ]]; then
    lrs=(1e-4 1e-3 1e-2 0.1 1 10)
    for lr in "${lrs[@]}"; do
        name="muon_lr${lr}"
        args=(
            "--run_name ${name}"
            "--wandb_project ${project}"
            "--log_folder muon_${DATE}"
            "--random_seed 42"
            "--optimizer muon"
            "--lr ${lr}"
            "--momentum ${momentum}"
        )
        submit_job ${args[@]}
    done
fi