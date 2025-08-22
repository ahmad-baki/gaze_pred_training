#!/bin/bash

#SBATCH --job-name=generate_gif
#SBATCH --output=logs/generate_gif/generate_gif_%j.out
#SBATCH --error=logs/generate_gif/generate_gif_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load devel/miniforge 
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/src
python3 helper/gaze_gif.py