#!/bin/bash
# 客户端测试脚本（在PyTorch镜像环境中运行）

echo "🚀 FastText Serving 客户端测试流程"
echo "=================================================="

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <FastText服务URL>"
    echo "例如: $0 http://10.1.1.100:8000"
    exit 1
fi

SERVICE_URL="$1"
echo "🎯 目标服务: $SERVICE_URL"

# 检查Python环境
echo "🐍 检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 未找到"
    exit 1
fi

# 安装依赖
echo "📦 安装依赖..."
pip install -r tests/requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️ 依赖安装可能有问题，继续执行..."
fi

# 配置服务URL
echo "🔧 配置服务URL..."
python3 tests/config_service_url.py "$SERVICE_URL"

echo ""
echo "🔍 第1步: 环境探测"
echo "================================"
python3 tests/01_environment_probe.py

echo ""
echo "🔍 第2步: 数据探索"  
echo "================================"
python3 tests/02_data_explorer.py

echo ""
echo "🔍 第3步: 模型验证"
echo "================================"
python3 tests/03_model_validator.py

echo ""
echo "🔍 第4步: 服务测试"
echo "================================"
python3 tests/04_service_test.py

echo ""
echo "✅ 测试流程完成!"
echo "================================"
echo "📋 检查生成的结果文件:"
ls -la *.json

echo ""
echo "🎯 如果所有测试通过，可以运行生产客户端:"
echo "python3 tests/05_production_client.py \\"
echo "  --data-dir /mnt/project/yifan/data/the-stack-v2-dedup_batched_download \\"
echo "  --output-dir ./output \\"
echo "  --service-url $SERVICE_URL \\"
echo "  --batch-size 100 \\"
echo "  --max-concurrent 10"
