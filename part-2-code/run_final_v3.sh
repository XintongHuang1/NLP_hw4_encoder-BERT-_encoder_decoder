#!/bin/bash

# 🎯 最终优化版本 - 适应复杂SQL生成任务
# 
# 关键发现：
# - 这是一个非常复杂的语义解析任务
# - SQL查询非常复杂(多表JOIN, 子查询, 复杂条件)
# - 需要学习数据库结构和查询模式
#
# 优化策略：
# 1. 移除所有额外的prompt (让模型从数据学习)
# 2. 使用更保守的学习率 (5e-5)
# 3. 更长的训练时间 (30+ epochs)
# 4. 更强的生成参数 (beam=10)

echo "========================================" 
echo "🚀 最终优化训练 - V3"
echo "========================================"
echo ""
echo "任务特点:"
echo "  - 非常复杂的SQL生成 (多表JOIN)"
echo "  - 需要学习数据库schema"
echo "  - 高错误率正常 (这是困难任务)"
echo ""
echo "优化策略:"
echo "  ✅ 无额外prompt (纯数据驱动)"
echo "  ✅ 保守学习率 5e-5"
echo "  ✅ 长训练 30 epochs"
echo "  ✅ 强生成 beam=10"
echo ""

# 核心配置 - 适合复杂任务
python train_t5.py \
    --finetune \
    --learning_rate 5e-5 \
    --max_n_epochs 30 \
    --patience_epochs 10 \
    --batch_size 16 \
    --experiment_name "final_v3"

echo ""
echo "========================================"
echo "预期性能:"
echo "========================================"
echo "  - 这个任务非常难!"
echo "  - F1 40-50% 就算不错"
echo "  - F1 55-60% 算很好"
echo "  - F1 65%+ 需要更复杂的方法"
echo ""
echo "如果F1还是很低(<30%)，可能需要:"
echo "  1. 使用T5-large (更大模型)"
echo "  2. 添加schema信息到prompt"
echo "  3. 使用更复杂的架构"
