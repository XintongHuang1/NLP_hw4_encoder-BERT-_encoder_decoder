#!/bin/bash

#SBATCH --job-name=resume_exp4

#SBATCH --output=logs/resume_%j.out

#SBATCH --error=logs/resume_%j.err

#SBATCH --time=04:00:00

#SBATCH --mem=16G

#SBATCH --cpus-per-task=4

#SBATCH --partition=a100_1,a100_2,rtx8000,h100_1

#SBATCH --gres=gpu:1



echo "========================================"

echo "Resuming exp5_lr3e4 Training"

echo "========================================"

echo "Job ID: $SLURM_JOB_ID"

echo "GPU: $CUDA_VISIBLE_DEVICES"

echo "Node: $HOSTNAME"

echo "Start: $(date)"

echo ""



# Activate conda environment

source ~/.bashrc

conda activate hw4-part-2-nlp



echo "Python: $(which python)"

echo "Conda env: $CONDA_DEFAULT_ENV"

echo ""



mkdir -p logs checkpoints results records



echo "Current best F1: $(cat best_f1.txt 2>/dev/null || echo 'N/A')"

echo ""



python train_t5.py --finetune --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 5 --batch_size 16 --experiment_name "exp5_lr3e4" --resume_from checkpoints/ft_experiments/exp4_lr3e4



echo ""

echo "========================================"

echo "Training Completed!"

echo "========================================"

echo "End: $(date)"

echo "Final best F1: $(cat best_f1.txt 2>/dev/null || echo 'N/A')"

echo ""

cat best_model_info.txt 2>/dev/null || echo "No best model info found"


