#!/bin/bash

#SBATCH --job-name=test_hyper_sweep
#SBATCH --output=logs/test_hyper_sweep/test_hyper_sweep_%j.out
#SBATCH --error=logs/test_hyper_sweep/test_hyper_sweep_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_il 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load devel/miniforge
conda activate gaze_pred_train
cd /home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/src
wandb agent ahmad-baki-karlsruhe-institute-of-technology/test/3ssrhnr5