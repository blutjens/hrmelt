#!/bin/bash

# Throughput script for launching hyperparameter sweep

# Slurm sbatch options
#   #SBATCH -o runs/interpolate_mar/data_v1_2/sweep/task-%a/task.sh.log # outdir directory
#SBATCH -o runs/interpolate_mar/data_v1_2/sweep/task-1/task.sh.log # outdir directory
# Create job array. Slurm will execute this same script
# in a 'throughput' parallel style
#   #SBATCH --array 1-12 # Create job array
#SBATCH --array 1-1# Create job array
#SBATCH --gres=gpu:volta:1 # number of GPUs
# By default supercloud recommends 20 CPUs per GPU. 
# Each worker comes with 4GB of memory
#SBATCH --cpus-per-task 18 # number of workers. 

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
python hrmelt/models/interpolate_mar/sweep.py \
--cfg_path runs/interpolate_mar/data_v1_2/config/config.yaml \
--sweep_path runs/interpolate_mar/data_v1_2/config/sweep.yaml
