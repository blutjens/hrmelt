#!/bin/bash

# Set the loop interval in seconds (30 minutes = 1800 seconds)
INTERVAL=3600

# Set temporary wandb directory to avoid issues with shared /tmp
export WANDB_DIR=./runs/ # unet/data_v1_4/
echo "Relative root path"
pwd

while true; do
  # Call the wandb sync command. 
  wandb sync --include-offline ./runs/unet_smp/data_v1_4/sweep/task-41/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4/sweep/task-42/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4/sweep/task-43/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4/sweep/task-44/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4/sweep/task-45/wandb/offline-*
  wandb sync --include-offline ./runs/unet_smp/data_v1_4/sweep/task-46/wandb/offline-*

  # Sleep for the interval before looping again
  echo 'sleeping'
  sleep $INTERVAL
done
