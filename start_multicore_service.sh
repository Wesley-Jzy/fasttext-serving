#!/bin/bash
"""
FastText多核服务启动脚本
使用Gunicorn实现多进程，充分利用CPU
"""

set -e

# 默认配置
MODEL_PATH="/mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin"
PORT=8000
WORKERS=16  # 根据CPU数量调整，建议 CPU核心数 / 8
TIMEOUT=300

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model PATH      FastText模型文件路径"
            echo "  --port PORT       服务端口 (默认: 8000)"
            echo "  --workers NUM     Worker进程数 (默认: 16)"
            echo "  --timeout SEC     请求超时时间 (默认: 300)"
            echo "  --help            显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --model /path/to/model.bin --workers 8"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查模型文件
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查gunicorn是否安装
if ! command -v gunicorn &> /dev/null; then
    echo "⚠️ 警告: gunicorn未安装，正在安装..."
    pip install gunicorn
fi

# 停止已有服务
echo "🛑 停止已有服务..."
pkill -f "gunicorn.*fasttext" || true
sleep 2

# 设置环境变量
export FASTTEXT_MODEL_PATH="$MODEL_PATH"
export FASTTEXT_MAX_TEXT_LENGTH="10000000"
export FASTTEXT_DEFAULT_THRESHOLD="0.0"

echo "🚀 启动FastText多核服务"
echo "===================="
echo "模型路径: $MODEL_PATH"
echo "服务端口: $PORT"
echo "Worker进程数: $WORKERS"
echo "请求超时: ${TIMEOUT}秒"
echo "===================="

# 启动Gunicorn服务
nohup gunicorn \
    --workers $WORKERS \
    --bind 0.0.0.0:$PORT \
    --timeout $TIMEOUT \
    --worker-class sync \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    start_fasttext_gunicorn:application \
    > fasttext_multicore.log 2>&1 &

GUNICORN_PID=$!
echo $GUNICORN_PID > fasttext_multicore.pid

echo "✅ 服务启动成功"
echo "PID: $GUNICORN_PID"
echo "日志文件: fasttext_multicore.log"
echo "PID文件: fasttext_multicore.pid"

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 健康检查
echo "🔍 健康检查..."
if curl -s http://localhost:$PORT/health > /dev/null; then
    echo "✅ 服务健康检查通过"
    echo "🎯 API端点: http://localhost:$PORT"
    echo ""
    echo "停止服务命令:"
    echo "  pkill -f 'gunicorn.*fasttext' 或"
    echo "  kill \$(cat fasttext_multicore.pid)"
    echo ""
    echo "查看日志命令:"
    echo "  tail -f fasttext_multicore.log"
else
    echo "❌ 服务健康检查失败"
    echo "查看日志: tail fasttext_multicore.log"
    exit 1
fi
