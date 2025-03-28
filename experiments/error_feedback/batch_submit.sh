#!/bin/bash

script="experiments/error_feedback/train.py"

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
# Batch submitting

error_feedbacks=(0.0 0.01 0.05 0.1 1.0)
lrs=(0.05 5e-3 0.5)
nesterov=True

for lr in "${lrs[@]}"; do
    for err in "${error_feedbacks[@]}"; do
        name="muon-err-v2_lr${lr}_ef${err}"
        args=(
            # basic configs
            "--run_name ${name}"
            "--wandb_project nanogpt_speedrun"
            "--log_folder muon_err_${DATE}"
            "--random_seed 42"
            # optimizer configs
            "--optimizer muon_err"
            "--lr ${lr}"
            "--error_feedback ${err}"
            "--nesterov ${nesterov}"
        )
        submit_job ${args[@]}
    done
done