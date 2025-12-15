#!/bin/bash

# Script to launch deploy scrips for every mdoel

# Slurm sbatch options
#SBATCH -o runs/unet_smp/data_v1_4/deploy/deploy.sh.log # outdir directory
#SBATCH --gres=gpu:volta:1 # number of GPUs; linear model faster on CPU only
# By default supercloud recommends 20 CPUs per GPU. 
# Each worker comes with 4GB of memory
#SBATCH --cpus-per-task 18 # number of workers. 
#SBATCH -t 48:00:00 # unet_smp+ trainning once took about 10hrs for 525 epochs

# Loading the required module
source /etc/profile # eofe: source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda deactivate
module load anaconda/2023a-pytorch # eofe: module load anaconda3/2020.11
conda activate hrmelt
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # sets GPU operations like gaussian_kernel to deterministic
export WANDB_MODE='offline'
echo "hello"

# Generate unet_smp deploy predictions
python hrmelt/predict.py \
--parallel \
--cfg_path='runs/unet_smp/data_v1_4/config/config.yaml' \
--load='runs/unet_smp/data_v1_4/sweep/task-1/checkpoints/checkpoint_epoch1025.pth' \
--data_split='deploy' \
--path_time_interpolate_sar='interim/runs/time_interpolate_sar/data_v1_4/deploy/' \
--prediction_batch_size=32 \
--erode_size=24 \
--prediction_stride=58 \
--apply_landmask_to_predictions=True