#!/bin/bash

#SBATCH --job-name=test_script
#SBATCH --output=logs/test_script_%j.out
#SBATCH --error=logs/test_script_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load devel/miniforge
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/
wandb agent ahmad-baki-karlsruhe-institute-of-technology/gaze_pred_training/ro1yp153
