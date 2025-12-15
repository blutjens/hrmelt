#!/bin/bash

# Throughput script for launching hyperparameter sweep

# Slurm sbatch options
#SBATCH -o runs/linear_dem/data_v1_4/sweep/task-%a/task.sh.log # outdir directory
#   #SBATCH -o runs/linear_dem/data_v1_4/sweep/task-7/task.sh.log # outdir directory
# Create job array. Slurm will execute this same script
# in a 'throughput' parallel style
#SBATCH --array 1-8 # Create job array
#   #SBATCH --array 1-1# Create job array
#SBATCH --gres=gpu:volta:1 # number of GPUs; linear model faster on CPU only
# By default supercloud recommends 20 CPUs per GPU. 
# Each worker comes with 4GB of memory
#SBATCH --cpus-per-task 18 # number of workers. 
#SBATCH -t 8:00:00

# Loading the required module
source /etc/profile # eofe: source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda deactivate
module load anaconda/2023a-pytorch # eofe: module load anaconda3/2020.11
conda activate hrmelt
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # sets GPU operations like gaussian_kernel to deterministic
export WANDB_MODE='offline'
echo "hello"

# Use these lines if wanting to run job in a specific task folder
# export SLURM_ARRAY_TASK_ID=7
# export SLURM_ARRAY_TASK_COUNT=7

# Run the script
python hrmelt/train.py \
--parallel \
--sweep \
--cfg_path 'runs/linear_dem/data_v1_4/config/config.yaml' \
--task_id $SLURM_ARRAY_TASK_ID \
--num_tasks $SLURM_ARRAY_TASK_COUNT