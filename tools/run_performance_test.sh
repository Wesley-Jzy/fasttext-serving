#!/bin/bash
"""
性能测试便捷脚本
直接调用真实的数据处理客户端进行性能测试
"""

set -e

# 默认配置
API_URL="http://localhost:8000"
DATA_DIR="/mnt/project/yifan/data/the-stack-v2-dedup_batched_download"
OUTPUT_DIR="./performance_test_results"
TEST_FILES_LIMIT=2
TEST_SAMPLES_PER_FILE=1000

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test-files-limit)
            TEST_FILES_LIMIT="$2"
            shift 2
            ;;
        --test-samples-per-file)
            TEST_SAMPLES_PER_FILE="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --api-url URL                   API服务地址 (默认: $API_URL)"
            echo "  --data-dir DIR                  数据目录 (默认: $DATA_DIR)"
            echo "  --output-dir DIR                输出目录 (默认: $OUTPUT_DIR)"
            echo "  --test-files-limit NUM          测试文件数量 (默认: $TEST_FILES_LIMIT)"
            echo "  --test-samples-per-file NUM     每文件样本数 (默认: $TEST_SAMPLES_PER_FILE)"
            echo "  --help                          显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --api-url http://10.208.60.212:8000"
            echo "  $0 --test-files-limit 3 --test-samples-per-file 2000"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "🚀 开始性能测试"
echo "=================="
echo "API地址: $API_URL"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "测试文件数: $TEST_FILES_LIMIT"
echo "每文件样本数: $TEST_SAMPLES_PER_FILE"
echo "=================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 测试不同的配置组合
configs=(
    "10 50"     # 保守配置
    "20 100"    # 中等配置
    "50 200"    # 高并发配置
)

echo "📊 将测试 ${#configs[@]} 种配置..."

for config in "${configs[@]}"; do
    read -r concurrent batch_size <<< "$config"
    
    echo ""
    echo "🧪 测试配置: 并发=$concurrent, 批次=$batch_size"
    echo "----------------------------------------"
    
    # 创建此配置的子目录
    config_output_dir="$OUTPUT_DIR/config_${concurrent}c_${batch_size}b"
    mkdir -p "$config_output_dir"
    
    # 运行性能测试
    python3 client/the_stack_processor.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$config_output_dir" \
        --api-url "$API_URL" \
        --max-concurrent $concurrent \
        --batch-size $batch_size \
        --performance-test \
        --test-files-limit $TEST_FILES_LIMIT \
        --test-samples-per-file $TEST_SAMPLES_PER_FILE \
        --enable-monitoring \
        --monitoring-interval 2 \
        --log-level INFO
    
    echo "✅ 配置 ${concurrent}c_${batch_size}b 测试完成"
    
    # 短暂休息避免服务过载
    sleep 3
done

echo ""
echo "🎉 性能测试全部完成！"
echo "===================="

# 汇总结果
echo "📋 测试结果汇总:"
for config in "${configs[@]}"; do
    read -r concurrent batch_size <<< "$config"
    config_output_dir="$OUTPUT_DIR/config_${concurrent}c_${batch_size}b"
    report_file="$config_output_dir/performance_test_report.json"
    
    if [[ -f "$report_file" ]]; then
        throughput_sps=$(jq -r '.performance_results.throughput_sps' "$report_file" 2>/dev/null || echo "N/A")
        throughput_gbps=$(jq -r '.performance_results.throughput_gbps' "$report_file" 2>/dev/null || echo "N/A")
        success_rate=$(jq -r '.performance_results.success_rate' "$report_file" 2>/dev/null || echo "N/A")
        cpu_avg=$(jq -r '.system_monitoring.avg_cpu_percent // "N/A"' "$report_file" 2>/dev/null || echo "N/A")
        memory_avg=$(jq -r '.system_monitoring.avg_memory_percent // "N/A"' "$report_file" 2>/dev/null || echo "N/A")
        
        if [[ "$throughput_gbps" != "N/A" ]]; then
            throughput_gbps=$(printf "%.3f" "$throughput_gbps")
        fi
        
        echo "  配置 ${concurrent}c_${batch_size}b: ${throughput_gbps} GB/s (${throughput_sps} samples/sec), 成功率: ${success_rate}, CPU: ${cpu_avg}%, 内存: ${memory_avg}%"
    else
        echo "  配置 ${concurrent}c_${batch_size}b: 测试失败，无报告文件"
    fi
done

echo ""
echo "📁 详细报告位置: $OUTPUT_DIR"
echo "💡 查看具体配置的详细报告:"
echo "   cat $OUTPUT_DIR/config_*/performance_test_report.json | jq ."
