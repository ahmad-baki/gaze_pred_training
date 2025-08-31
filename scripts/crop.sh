#!/bin/bash

#SBATCH --job-name=crop_data
#SBATCH --output=logs/crop_data/crop_data_%j.out
#SBATCH --error=logs/crop_data/crop_data_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=200gb
#SBATCH --partition=cpu
#SBATCH --ntasks=1

module load devel/miniforge
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/src/helper
python3 -u crop_data2.py
