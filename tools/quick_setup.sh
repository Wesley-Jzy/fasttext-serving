#!/bin/bash
set -e

# FastText Serving 快速设置脚本
# 帮助新用户快速开始使用

echo "🚀 FastText Serving 快速设置"
echo "=================================="

# 检查Python环境
echo "🐍 检查Python环境..."
python3 --version || {
    echo "❌ 需要Python 3.7+，请先安装Python"
    exit 1
}

# 检查依赖
echo "📦 检查依赖包..."
python3 -c "import pandas, aiohttp, pyarrow" 2>/dev/null || {
    echo "⚠️  缺少依赖包，正在安装..."
    pip3 install pandas aiohttp pyarrow
}

echo "✅ 依赖检查完成"

# 获取用户输入
echo ""
echo "📋 请提供以下信息:"

# 数据目录
read -p "The-Stack-v2数据目录路径: " DATA_DIR
if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    exit 1
fi

# 输出目录
read -p "结果输出目录路径: " OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

# API地址
read -p "FastText API地址 (默认: http://localhost:8000): " API_URL
API_URL=${API_URL:-http://localhost:8000}

# 测试API连接
echo ""
echo "🔗 测试API连接..."
curl -s -f "$API_URL/health" > /dev/null || {
    echo "❌ 无法连接到API服务: $API_URL"
    echo "请确保FastText服务正在运行"
    exit 1
}
echo "✅ API连接正常"

# 运行性能测试
echo ""
echo "⚡ 运行API性能测试..."
python3 tools/api_performance_tester.py \
    --api-url "$API_URL" \
    --quick \
    --output "$OUTPUT_DIR/performance_test.json"

# 提取推荐配置
RECOMMENDED_CONFIG=$(python3 -c "
import json
try:
    with open('$OUTPUT_DIR/performance_test.json', 'r') as f:
        data = json.load(f)
    best = data['best_configuration']['best_overall']
    print(f'--max-concurrent {best[\"concurrency\"]} --batch-size {best[\"batch_size\"]}')
except:
    print('--max-concurrent 50 --batch-size 200')
")

# 生成启动命令
echo ""
echo "🎯 生成处理命令..."

COMMAND="python3 client/the_stack_processor.py \\
  --data-dir \"$DATA_DIR\" \\
  --output-dir \"$OUTPUT_DIR\" \\
  --api-url \"$API_URL\" \\
  $RECOMMENDED_CONFIG \\
  --resume"

echo "命令已生成并保存到: $OUTPUT_DIR/run_processing.sh"

# 保存运行脚本
cat > "$OUTPUT_DIR/run_processing.sh" << EOF
#!/bin/bash
# The-Stack-v2 数据处理命令
# 生成时间: $(date)

echo "🚀 开始处理The-Stack-v2数据..."

$COMMAND

echo "✅ 处理完成!"
echo "结果保存在: $OUTPUT_DIR"
EOF

chmod +x "$OUTPUT_DIR/run_processing.sh"

# 最终说明
echo ""
echo "🎉 设置完成!"
echo ""
echo "📋 下一步操作:"
echo "1. 启动处理: $OUTPUT_DIR/run_processing.sh"
echo "2. 监控日志: tail -f $OUTPUT_DIR/processing.log"
echo "3. 查看进度: 命令行会显示实时进度"
echo ""
echo "💡 命令说明:"
echo "$COMMAND"
echo ""
echo "📊 性能测试结果已保存到: $OUTPUT_DIR/performance_test.json"
