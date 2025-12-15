#!/bin/bash

# Throughput script for launching hyperparameter sweep

# Slurm sbatch options
#SBATCH -o runs/deeplabv3/data_v1_4/sweep/task-{%a+18}/task.sh.log # outdir directory
#   #SBATCH -o runs/deeplabv3/data_v1_4/sweep/task-7/task.sh.log # outdir directory
# Create job array. Slurm will execute this same script
# in a 'throughput' parallel style
#SBATCH --array 1-3 # Create job array
#   #SBATCH --array 1-1# Create job array
#SBATCH --gres=gpu:volta:1 # number of GPUs; linear model faster on CPU only
# By default supercloud recommends 20 CPUs per GPU. 
# Each worker comes with 4GB of memory
#SBATCH --cpus-per-task 18 # number of workers. 
#   #SBATCH -t 16:00:00 # Deeplabv3+ trainning once took about 10hrs for 525 epochs
#SBATCH -t 38:00:00 # Deeplabv3+ trainning once took about 10hrs for 525 epochs

# Loading the required module
source /etc/profile # eofe: source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda deactivate
module load anaconda/2023a-pytorch # eofe: module load anaconda3/2020.11
conda activate hrmelt
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # sets GPU operations like gaussian_kernel to deterministic
export WANDB_MODE='offline'
export TRANSFORMERS_OFFLINE=1 # set hugginface model weights download offline
export HF_DATASETS_OFFLINE=1
echo "hello"

# Use these lines if wanting to run job in a specific task folder
# export SLURM_ARRAY_TASK_ID=19
# export SLURM_ARRAY_TASK_COUNT=19
export SLURM_ARRAY_TASK_OFFSET=18

# Run the script
python hrmelt/train.py \
--parallel \
--sweep \
--cfg_path 'runs/deeplabv3/data_v1_4/config/config.yaml' \
--task_id $SLURM_ARRAY_TASK_ID \
--num_tasks $SLURM_ARRAY_TASK_COUNT \
--task_id_offset $SLURM_ARRAY_TASK_OFFSET