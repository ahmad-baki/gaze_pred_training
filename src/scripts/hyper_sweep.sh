#!/bin/bash

#SBATCH --job-name=test_script
#SBATCH --output=logs_train/test_script_%j.out
#SBATCH --error=logs_train/test_script_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load devel/miniforge
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/
wandb agent --count 100 ahmad-baki-karlsruhe-institute-of-technology/test/kv5dvxgg 