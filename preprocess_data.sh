#!/bin/bash

#SBATCH --job-name=test_script
#SBATCH --output=logs/test_script_%j.out
#SBATCH --error=logs/test_script_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --mem=6gb
#SBATCH --partition=dev_cpu
#SBATCH --ntasks=1

module load devel/miniforge
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/helper
python3 preprocess_workspace.py