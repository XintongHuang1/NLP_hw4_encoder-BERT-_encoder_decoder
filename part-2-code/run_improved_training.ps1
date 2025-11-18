# 改进的训练脚本 - 提升F1性能
# 关键改进点：
# 1. 更小的学习率 (1e-4)
# 2. 足够的训练轮数 (20 epochs)
# 3. Warmup (1 epoch)
# 4. Early stopping (5 epochs patience)
# 5. 权重衰减正则化 (0.01)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始优化的 T5 Fine-tuning 训练" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 方案1: 使用优化的默认参数（推荐）
Write-Host "方案1: 标准优化配置 (LR=1e-4)" -ForegroundColor Green
python train_t5.py --finetune --experiment_name "improved_v1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "如果效果还不够好，可以尝试以下方案：" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

Write-Host "方案2: 更小学习率 (LR=5e-5)" -ForegroundColor Yellow
Write-Host "python train_t5.py --finetune --learning_rate 5e-5 --experiment_name 'improved_v2'" -ForegroundColor Gray
Write-Host ""

Write-Host "方案3: 更长训练 (30 epochs)" -ForegroundColor Yellow
Write-Host "python train_t5.py --finetune --max_n_epochs 30 --patience_epochs 8 --experiment_name 'improved_v3'" -ForegroundColor Gray
Write-Host ""

Write-Host "方案4: 更大batch size (32)" -ForegroundColor Yellow
Write-Host "python train_t5.py --finetune --batch_size 32 --experiment_name 'improved_v4'" -ForegroundColor Gray
Write-Host ""

Write-Host "方案5: 组合优化" -ForegroundColor Yellow
Write-Host "python train_t5.py --finetune --learning_rate 5e-5 --max_n_epochs 30 --batch_size 32 --patience_epochs 8 --experiment_name 'improved_best'" -ForegroundColor Gray
