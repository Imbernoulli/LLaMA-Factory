#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH --mem=800g
#SBATCH --partition=pli-c
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --job-name=dxy
#SBATCH --output=logs/dxy.log
#SBATCH --error=logs/dxy.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lyubh22@gmail.com

export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# CUDA lib workaround
mkdir -p ~/local/cuda/lib64
cd ~/local/cuda/lib64
ln -sf /usr/local/cuda-12.8/lib64/libcurand.so.10 libcurand.so
export LD_LIBRARY_PATH=~/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=~/local/cuda/lib64:$LIBRARY_PATH

export DISABLE_VERSION_CHECK=1

source /scratch/gpfs/yl7690/miniconda3/etc/profile.d/conda.sh
conda activate oss

cd /scratch/gpfs/yl7690/projects/LLaMA-Factory-new

torchrun --nproc_per_node=8 --master_port=29500 \
    src/train.py \
    --model_name_or_path /scratch/gpfs/PLI/yong/gpt-oss-20b-bf16 \
    --trust_remote_code true \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --dataset dxy \
    --template gpt \
    --cutoff_len 64000 \
    --max_samples 100000000 \
    --overwrite_cache true \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 64 \
    --output_dir /scratch/gpfs/CHIJ/yong/trained_models/dxy \
    --logging_steps 10 \
    --save_steps 100 \
    --plot_loss true \
    --overwrite_output_dir true \
    --save_only_model true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    # --val_size 0.01 \
    # --per_device_eval_batch_size 1 \
    # --eval_strategy steps \
    # --eval_steps 500 \
    --flash_attn fa2