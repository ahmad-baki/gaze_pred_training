#!/bin/bash

#SBATCH --job-name=generate_gam_gif
#SBATCH --output=logs/generate_gam_gif/generate_gam_gif_%j.out
#SBATCH --error=logs/generate_gam_gif/generate_gam_gif_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load devel/miniforge 
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/src
python3 helper/gaze_and_model_gif.py