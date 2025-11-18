python train_t5.py --finetune --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 5 --batch_size 16 --experiment_name "exp4_lr3e4"

#!/bin/bash
#SBATCH --job-name=t5_multi_exp
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_1,a100_2,rtx8000,h100_1
#SBATCH --gres=gpu:1

echo "========================================"
echo "T5 Fine-tuning - Multi Experiments"
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

echo "========================================"
echo "Experiment 1: LR=3e-4, 50 epochs"
echo "========================================"
python train_t5.py --finetune --learning_rate 3e-4 --max_n_epochs 50 --patience_epochs 5 --batch_size 16 --experiment_name "exp4_lr3e4"

echo ""
echo "Exp4 Done! F1: $(cat best_f1.txt 2>/dev/null || echo 'N/A')"
echo ""
sleep 3

echo "========================================"
echo "Experiment 5: LR=3e-5, 20 epochs"
echo "========================================"
#python train_t5.py --finetune --learning_rate 1e-4 --max_n_epochs 20 --patience_epochs 5 --batch_size 16 --experiment_name "exp2_lr1e4"

echo ""
echo "Exp2 Done! F1: $(cat best_f1.txt 2>/dev/null || echo 'N/A')"
echo ""
sleep 3

echo "========================================"
echo "Experiment 3: LR=5e-5, 30 epochs"
echo "========================================"
#python train_t5.py --finetune --learning_rate 5e-5 --max_n_epochs 30 --patience_epochs 8 --batch_size 16 --experiment_name "exp3_long"

echo ""
echo "Exp3 Done! F1: $(cat best_f1.txt 2>/dev/null || echo 'N/A')"
echo ""

echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "End: $(date)"
echo "Best F1: $(cat best_f1.txt 2>/dev/null || echo 'N/A')"
echo ""
echo "Submit file: results/t5_ft_test.sql"
echo "Details: cat best_model_info.txt"
