#!/bin/bash
# The-Stack-v2 数据处理命令
# 生成时间: 2025-08-15 21:56:50

echo "🚀 开始处理The-Stack-v2数据..."

python3 client/the_stack_processor.py \
  --data-dir "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download" \
  --output-dir "examples/sample_processing_outputs" \
  --api-url "http://10.208.60.212:8000" \
  --max-concurrent 80 \
  --batch-size 500 \
  --resume

echo "✅ 处理完成!"
echo "结果保存在: examples/sample_processing_outputs"
echo "日志文件: examples/sample_processing_outputs/processing.log"
echo "检查点文件: examples/sample_processing_outputs/processing_checkpoint.json"
