#!/bin/bash

#=========================================================================================
# --- 1. SLURM & JOB CONFIGURATION ---
#=========================================================================================

# SLURM resource configuration
TOTAL_NODES=8
GPUS_PER_NODE=4
CPUS_PER_GPU=10 # This will be used to calculate cpus-per-task

# Training parameters
EPOCHS=1.0
SAVE_STEPS=500
JOB_SUFFIX="dxy_full_finetune"
timestr=$(date +%y%m%d_%H%M%S)

# Calculate gradient accumulation dynamically.
# This value should be adjusted based on your desired global batch size.
# For this example, let's assume a desired global batch size of 128.
GLOBAL_BATCH_SIZE=16
GRADIENT_ACCUM=4


#=========================================================================================
# --- 2. EXPERIMENT DEFINITIONS ---
#=========================================================================================

declare -A configs=(
    ["gpt-oss-20b_dxy_5e-5_cosine"]="/scratch/gpfs/PLI/yong/gpt-oss-20b-bf16 formal 3.0e-5 cosine"
)


#=========================================================================================
# --- 3. SBATCH SCRIPT GENERATION FUNCTION ---
#=========================================================================================

create_train_script() {
    # --- Input parameters ---
    local model_path=$1
    local dataset=$2
    local lr=$3
    local scheduler=$4

    # --- Generate Job Name and Output Directory (more robust method) ---
    local model_name_base=$(basename "$model_path")
    # Sanitize names to be filesystem and job-name friendly
    local model_name_sanitized=$(echo "$model_name_base" | sed 's/[^a-zA-Z0-9_-]/-/g')
    local dataset_sanitized=$(echo "$dataset" | sed 's/[^a-zA-Z0-9_-]/-/g')

    local job_name="train_${dataset_sanitized}_${model_name_sanitized}_lr${lr}_${scheduler}_${JOB_SUFFIX}"
    local output_dir="/scratch/gpfs/CHIJ/yong/trained_models/${job_name}"
    local script_file_path="sbatch_jobs/${job_name}.sh"

    echo "==> Generating script for job: ${job_name}"

    # --- Create the sbatch script file using a heredoc ---
    cat << EOF > "${script_file_path}"
#!/bin/bash
#SBATCH --nodes=${TOTAL_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$((GPUS_PER_NODE * CPUS_PER_GPU))
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --mem=$((GPUS_PER_NODE * 100))G
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${timestr}_${job_name}.log
#SBATCH --error=logs/${timestr}_${job_name}.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lyubh22@gmail.com
#SBATCH --partition=pli-p          # Set partition based on your provided examples
#SBATCH --qos=pli-high             # Set QoS based on your provided examples
#SBATCH --account=goedel_prover_prio # Set account name based on your provided examples

# --- Environment Setup ---
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export PATH=\$CUDA_HOME/bin:\$PATH

# Workaround for libcurand if necessary
mkdir -p ~/local/cuda/lib64 && cd ~/local/cuda/lib64
ln -sf /usr/local/cuda-12.8/lib64/libcurand.so.10 libcurand.so
export LD_LIBRARY_PATH=~/local/cuda/lib64:\$LD_LIBRARY_PATH
export LIBRARY_PATH=~/local/cuda/lib64:\$LIBRARY_PATH

# Activate Conda environment
source /scratch/gpfs/yl7690/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate oss

# --- Job Information Logging ---
echo "Job started at \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node List: \$SLURM_JOB_NODELIST"
echo "Number of Nodes: \$SLURM_NNODES"
echo "GPUs per Node: \$SLURM_GPUS_ON_NODE"
echo "Master Addr: \$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)"

# --- Execute Distributed Training on All Nodes ---
srun bash -c '
set -x # Echo commands for easier debugging

cd /scratch/gpfs/yl7690/projects/LLaMA-Factory-new

# Set environment variables for training
export WANDB_MODE=disabled # Disable Weights & Biases logging
export DISABLE_VERSION_CHECK=1 # Disable version checking if needed

torchrun \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --nnodes=\$SLURM_NNODES \\
    --node_rank=\$SLURM_PROCID \\
    --master_addr=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \\
    --master_port=29500 \\
    src/train.py \\
    --model_name_or_path "${model_path}" \\
    --trust_remote_code true \\
    --stage sft \\
    --do_train \\
    --finetuning_type full \\
    --deepspeed /scratch/gpfs/yl7690/projects/LLaMA-Factory-new/examples/deepspeed/ds_z3_offload_config.json \\
    --dataset "${dataset}" \\
    --template gpt \\
    --cutoff_len 64000 \\
    --max_samples 100000000 \\
    --overwrite_cache true \\
    --preprocessing_num_workers 64 \\
    --dataloader_num_workers 64 \\
    --output_dir "${output_dir}" \\
    --logging_steps 10 \\
    --save_steps ${SAVE_STEPS} \\
    --plot_loss true \\
    --overwrite_output_dir true \\
    --save_only_model true \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \\
    --learning_rate ${lr} \\
    --num_train_epochs ${EPOCHS} \\
    --lr_scheduler_type "${scheduler}" \\
    --warmup_ratio 0.1 \\
    --bf16 true \\
    --ddp_timeout 180000000 \\
    --flash_attn fa2 \\
    --enable_liger_kernel true \\
    
' # End of srun bash -c command

echo "Job finished at \$(date)"
EOF

    chmod +x "${script_file_path}"
    echo "==> Script created at: ${script_file_path}"
}


#=========================================================================================
# --- 4. MAIN EXECUTION LOGIC ---
#=========================================================================================

# Create directories for logs and scripts if they don't exist
mkdir -p sbatch_jobs logs

# Iterate through the configs array to generate and submit a job for each
for key in "${!configs[@]}"; do
    IFS=' ' read -r model_path dataset lr scheduler <<< "${configs[$key]}"

    echo "--- Processing config: [${key}] ---"

    # Call the function to create the script.
    # The function now handles calculating the file path internally.
    create_train_script "$model_path" "$dataset" "$lr" "$scheduler"

    # To submit the job, we recalculate the script path here to ensure it's correct.
    # This is a more reliable pattern than capturing function output.
    model_name_base=$(basename "$model_path")
    model_name_sanitized=$(echo "$model_name_base" | sed 's/[^a-zA-Z0-9_-]/-/g')
    dataset_sanitized=$(echo "$dataset" | sed 's/[^a-zA-Z0-9_-]/-/g')
    job_name="train_${dataset_sanitized}_${model_name_sanitized}_lr${lr}_${scheduler}_${JOB_SUFFIX}"
    generated_script_path="sbatch_jobs/${job_name}.sh"

    if [[ -f "$generated_script_path" ]]; then
        echo "Submitting job: ${generated_script_path}"
        sbatch "$generated_script_path"
        echo "--- Job submitted for [${key}] ---"
    else
        echo "Error: Script file not found at ${generated_script_path}"
    fi
    echo "" # Add a blank line for readability
done

echo "All jobs have been submitted."
