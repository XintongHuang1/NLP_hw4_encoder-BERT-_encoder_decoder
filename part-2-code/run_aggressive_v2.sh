#!/bin/bash

# 激进优化版本 - 针对低F1问题
# 关键改进：
# 1. 移除了可能有问题的实体对齐
# 2. 更好的prompt: "translate to SQL for flight database: "
# 3. 更强的生成参数: beam=10, max_tokens=200
# 4. 更大的学习率: 3e-4 (学习更快)

echo "========================================"
echo "🚀 激进优化训练 - V2"
echo "========================================"
echo ""
echo "改进点:"
echo "  ✅ 移除实体对齐 (可能引入错误)"
echo "  ✅ 改进prompt (更清晰)"
echo "  ✅ 增强生成 (beam=10, tokens=200)"
echo "  ✅ 提高学习率 (3e-4)"
echo ""

# 推荐配置
python train_t5.py --finetune --experiment_name "aggressive_v2" --max_n_epochs 25

echo ""
echo "========================================"
echo "备选方案（如果上面不够好）："
echo "========================================"
echo ""

echo "方案A: 更激进的学习率 (5e-4)"
echo "python train_t5.py --finetune --learning_rate 5e-4 --max_n_epochs 25 --experiment_name 'lr_5e4'"
echo ""

echo "方案B: 组合优化 + 更大batch"
echo "python train_t5.py --finetune --batch_size 32 --max_n_epochs 30 --experiment_name 'batch32_long'"
echo ""

echo "方案C: 尝试T5-small (更快，可能更稳定)"
echo "# 需要修改 t5_utils.py 中的 model_name = 'google-t5/t5-small'"
