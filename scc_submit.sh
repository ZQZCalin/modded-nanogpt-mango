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

name="precmuon_rms"
seed=42
args=(
    # basic configs
    "--log_folder mango"
    "--run_name ${name}"
    "--random_seed ${seed}"
    # optimizer configs
    "--optimizer mango"
    "--mango_mat_beta2 0.95"
    "--mango_mat_scale_rms True"
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
test=0
# test=1      # uncomment to run locally instead of submit to scc

if [[ $test -eq 1 ]]; then
    torchrun --standalone --nproc_per_node=${NODES} ${script}.py ${args[@]}
else
    job_output=$(qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
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

    # -----------------------------------------------------------------------------
    # Save job id and associated name to local .txt
    # This is extremely helpful to manage a bunch of experiments.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"

    echo "Submitted job: $name"
fi