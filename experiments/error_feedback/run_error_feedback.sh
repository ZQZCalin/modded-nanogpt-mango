#!/bin/bash

# -----------------------------------------------------------------------------
# Experiment configs

script="experiments/error_feedback/train_error_feedback.py"

lr=0.05
err=0.05

name=muon_err

DATE=$(date +"%Y-%m-%d")
args=(
    # basic configs
    "--run_name ${name}_lr${lr}_ef${err}"
    "--wandb_project nanogpt_speedrun"
    "--log_folder mango_${DATE}"
    "--random_seed 42"
    # some unrelated configs for convenience
    "--compile_only False"  # turn on to warmup the node (for the first run)
    "--advanced_log False"  # turn on to log rms norms
    # optimizer configs
    "--optimizer muon_err"
    "--lr ${lr}"
    "--error_feedback ${err}"
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
#$ -l h_rt=8:00:00          # Specifies the hard time limit for the job
#$ -N "$name".sh
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

source activate_env.sh
torchrun --standalone --nproc_per_node=${NODES} ${script} ${args[@]}
EOF
    )

    # Extract job id and log the submission.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $name"
}

if [[ $mode -eq 1 ]]; then
    torchrun --standalone --nproc_per_node=${NODES} ${script} ${args[@]}
elif [[ $mode -eq 0 ]]; then
    submit_job ${args[@]}
fi