#!/bin/bash

#SBATCH --job-name=train_gaze
#SBATCH --output=logs/train/train_gaze_%j.out
#SBATCH --error=logs/train/train_gaze_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load devel/miniforge
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/
python3 src/main.py dataset=gaze_dataset_workspace_cucumber_in_pot