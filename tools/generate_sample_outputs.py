#!/usr/bin/env python3
"""
生成示例数据处理输出
展示正常数据洗入流程会产生的各种文件和内容
"""

import json
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

def generate_processing_checkpoint(output_dir: Path):
    """生成处理检查点文件"""
    checkpoint = {
        "processed_files": [
            "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download/the-stack-v2-dedup-train-00001-of-01078.parquet",
            "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download/the-stack-v2-dedup-train-00002-of-01078.parquet",
            "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download/the-stack-v2-dedup-train-00003-of-01078.parquet",
            "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download/the-stack-v2-dedup-train-00004-of-01078.parquet",
            "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download/the-stack-v2-dedup-train-00005-of-01078.parquet"
        ],
        "last_update": time.time(),
        "stats": {
            "total_files": 1078,
            "processed_files": 5,
            "skipped_files": 0,
            "total_samples": 156789,
            "processed_samples": 156234,
            "successful_samples": 155456,
            "failed_samples": 778,
            "start_time": time.time() - 3600,  # 1小时前开始
            "processing_time": 3456.78,
            "throughput_sps": 45.2
        }
    }
    
    checkpoint_file = output_dir / "processing_checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成检查点文件: {checkpoint_file}")

def generate_processing_log(output_dir: Path):
    """生成处理日志文件"""
    log_content = f"""2024-01-15 10:30:45,123 - TheStackProcessor - INFO - 🚀 开始处理The-Stack-v2数据
2024-01-15 10:30:45,124 - TheStackProcessor - INFO - 数据目录: /mnt/project/yifan/data/the-stack-v2-dedup_batched_download
2024-01-15 10:30:45,125 - TheStackProcessor - INFO - 输出目录: /path/to/results
2024-01-15 10:30:45,126 - TheStackProcessor - INFO - API地址: http://10.208.60.212:8000
2024-01-15 10:30:45,127 - TheStackProcessor - INFO - 并发数: 50
2024-01-15 10:30:45,128 - TheStackProcessor - INFO - 批次大小: 200
2024-01-15 10:30:45,129 - TheStackProcessor - INFO - 📂 加载检查点: 已处理 0 个文件
2024-01-15 10:30:45,130 - TheStackProcessor - INFO - ✅ API服务健康: healthy
2024-01-15 10:30:45,135 - TheStackProcessor - INFO - 📁 发现 1078 个新文件待处理
2024-01-15 10:30:45,136 - TheStackProcessor - INFO - 📄 开始处理: the-stack-v2-dedup-train-00001-of-01078.parquet
2024-01-15 10:30:46,234 - TheStackProcessor - INFO - 📊 the-stack-v2-dedup-train-00001-of-01078.parquet: 处理 12345 个样本
2024-01-15 10:31:12,567 - TheStackProcessor - WARNING - 无效预测: invalid_label___label__2
2024-01-15 10:31:12,568 - TheStackProcessor - WARNING - 无效预测: score_out_of_range_1.2
2024-01-15 10:31:45,789 - TheStackProcessor - INFO - ✅ 完成: the-stack-v2-dedup-train-00001-of-01078.parquet (12345 样本, 1234.5 samples/sec)
2024-01-15 10:31:45,790 - TheStackProcessor - INFO - 📄 开始处理: the-stack-v2-dedup-train-00002-of-01078.parquet
2024-01-15 10:31:46,891 - TheStackProcessor - INFO - 📊 the-stack-v2-dedup-train-00002-of-01078.parquet: 处理 23456 个样本
2024-01-15 10:32:23,456 - TheStackProcessor - WARNING - 无效预测: invalid_label___label__3
2024-01-15 10:32:56,123 - TheStackProcessor - INFO - ✅ 完成: the-stack-v2-dedup-train-00002-of-01078.parquet (23456 样本, 1567.8 samples/sec)
2024-01-15 10:32:56,124 - TheStackProcessor - INFO - 📄 开始处理: the-stack-v2-dedup-train-00003-of-01078.parquet
2024-01-15 10:32:57,225 - TheStackProcessor - INFO - 📊 the-stack-v2-dedup-train-00003-of-01078.parquet: 处理 18765 个样本
2024-01-15 10:33:28,987 - TheStackProcessor - ERROR - ❌ 处理文件失败: the-stack-v2-dedup-train-00004-of-01078.parquet - API错误: HTTP 500
2024-01-15 10:33:29,001 - TheStackProcessor - INFO - 📄 开始处理: the-stack-v2-dedup-train-00005-of-01078.parquet
2024-01-15 10:33:30,102 - TheStackProcessor - INFO - 📊 the-stack-v2-dedup-train-00005-of-01078.parquet: 处理 34567 个样本
2024-01-15 10:34:12,345 - TheStackProcessor - INFO - ✅ 完成: the-stack-v2-dedup-train-00005-of-01078.parquet (34567 样本, 1678.9 samples/sec)
2024-01-15 10:34:12,346 - TheStackProcessor - INFO - ✅ 当前批次处理完成，等待新文件...
2024-01-15 10:34:12,347 - TheStackProcessor - INFO - ⏳ 暂无新文件，等待30秒后重新扫描...
"""
    
    log_file = output_dir / "processing.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"✅ 生成处理日志: {log_file}")

def generate_sample_result_files(output_dir: Path):
    """生成示例结果文件"""
    
    # 生成正常的处理结果文件
    normal_results = []
    for i in range(100):
        result = {
            "blob_id": f"abc123def456{i:04d}",
            "path": f"src/example_{i}.py",
            "repo_name": f"user/project_{i % 10}",
            "language": "Python",
            "size": 1024 + i * 10,
            "ext": "py",
            "quality_labels": ["__label__1", "__label__0"] if i % 3 == 0 else ["__label__0", "__label__1"],
            "quality_scores": [0.8234, 0.1766] if i % 3 == 0 else [0.7123, 0.2877],
            "quality_prediction": "__label__1" if i % 3 == 0 else "__label__0",
            "quality_confidence": 0.8234 if i % 3 == 0 else 0.7123,
            "prediction_valid": True,
            "prediction_error": None,
            "content_length": 856 + i * 5,
            "processed_at": f"2024-01-15T10:3{i//10}:{i%60:02d}.123456"
        }
        normal_results.append(result)
    
    # 添加一些异常结果
    error_results = [
        {
            "blob_id": "error001",
            "path": "bad_code.js",
            "repo_name": "user/bad-project",
            "language": "JavaScript", 
            "size": 234,
            "ext": "js",
            "quality_labels": ["__label__2", "__label__1"],  # 异常标签
            "quality_scores": [0.7, 0.3],
            "quality_prediction": "__label__2",
            "quality_confidence": 0.7,
            "prediction_valid": False,
            "prediction_error": "invalid_label___label__2",
            "content_length": 123,
            "processed_at": "2024-01-15T10:31:12.567890"
        },
        {
            "blob_id": "error002",
            "path": "broken.cpp",
            "repo_name": "user/broken-project",
            "language": "C++",
            "size": 567,
            "ext": "cpp", 
            "quality_labels": ["__label__1", "__label__0"],
            "quality_scores": [1.2, 0.8],  # 异常分数
            "quality_prediction": "__label__1",
            "quality_confidence": 1.2,
            "prediction_valid": False,
            "prediction_error": "score_out_of_range_1.2",
            "content_length": 456,
            "processed_at": "2024-01-15T10:31:12.568901"
        }
    ]
    
    all_results = normal_results + error_results
    
    # 生成parquet格式结果文件
    df = pd.DataFrame(all_results)
    parquet_file = output_dir / "processed_the-stack-v2-dedup-train-00001-of-01078.parquet"
    df.to_parquet(parquet_file, index=False)
    print(f"✅ 生成结果文件 (parquet): {parquet_file}")
    
    # 生成JSON格式结果文件
    json_file = output_dir / "processed_the-stack-v2-dedup-train-00002-of-01078.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results[:50], f, indent=2, ensure_ascii=False)
    print(f"✅ 生成结果文件 (json): {json_file}")

def generate_performance_test_results(output_dir: Path):
    """生成性能测试结果文件"""
    performance_results = {
        "test_results": [
            {
                "file_name": "the-stack-v2-dedup-train-00596-of-01078.parquet",
                "file_type": "largest",
                "file_size_mb": 296199.3,
                "file_size_gb": 289.26,
                "total_samples": 856432,
                "processed_samples": 856234,
                "successful_samples": 855456,
                "failed_samples": 778,
                "concurrency": 50,
                "batch_size": 200,
                "processing_time": 456.78,
                "throughput_sps": 1874.5,
                "throughput_gbs": 0.633,
                "avg_latency": 0.067,
                "success_rate": 0.991,
                "error_types": {
                    "invalid_label___label__2": 123,
                    "score_out_of_range_1.2": 45,
                    "scores_not_descending": 32
                },
                "sample_errors": ["invalid_label___label__2", "score_out_of_range_1.2"]
            },
            {
                "file_name": "the-stack-v2-dedup-train-00595-of-01078.parquet", 
                "file_type": "second_largest",
                "file_size_mb": 294567.8,
                "file_size_gb": 287.67,
                "total_samples": 834567,
                "processed_samples": 834234,
                "successful_samples": 833567,
                "failed_samples": 667,
                "concurrency": 80,
                "batch_size": 500,
                "processing_time": 389.45,
                "throughput_sps": 2141.2,
                "throughput_gbs": 0.739,
                "avg_latency": 0.089,
                "success_rate": 0.992,
                "error_types": {
                    "invalid_label___label__2": 98,
                    "score_out_of_range_1.2": 34,
                    "scores_not_descending": 23
                },
                "sample_errors": ["invalid_label___label__2", "score_out_of_range_1.2"]
            }
        ],
        "best_configuration": {
            "largest_file_best": {
                "file_name": "the-stack-v2-dedup-train-00596-of-01078.parquet",
                "concurrency": 50,
                "batch_size": 200,
                "throughput_sps": 1874.5,
                "throughput_gbs": 0.633,
                "success_rate": 0.991,
                "avg_latency": 0.067
            },
            "second_largest_file_best": {
                "file_name": "the-stack-v2-dedup-train-00595-of-01078.parquet",
                "concurrency": 80,
                "batch_size": 500,
                "throughput_sps": 2141.2,
                "throughput_gbs": 0.739,
                "success_rate": 0.992,
                "avg_latency": 0.089
            },
            "overall_best_throughput": {
                "file_name": "the-stack-v2-dedup-train-00595-of-01078.parquet",
                "concurrency": 80,
                "batch_size": 500,
                "throughput_sps": 2141.2,
                "throughput_gbs": 0.739,
                "success_rate": 0.992,
                "avg_latency": 0.089
            }
        },
        "test_summary": {
            "total_tests": 48,
            "total_samples_processed": 1690468,
            "total_successful_samples": 1689023,
            "overall_success_rate": 0.9914,
            "total_processing_time": 846.23,
            "total_data_processed_gb": 576.93,
            "max_throughput_gbs": 0.739,
            "max_throughput_sps": 2141.2,
            "best_config_for_gbs": {
                "concurrency": 80,
                "batch_size": 500
            },
            "best_config_for_sps": {
                "concurrency": 80,
                "batch_size": 500
            }
        }
    }
    
    perf_file = output_dir / "real_data_performance_results.json"
    with open(perf_file, 'w', encoding='utf-8') as f:
        json.dump(performance_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成性能测试结果: {perf_file}")

def generate_run_script(output_dir: Path):
    """生成运行脚本"""
    script_content = f"""#!/bin/bash
# The-Stack-v2 数据处理命令
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 开始处理The-Stack-v2数据..."

python3 client/the_stack_processor.py \\
  --data-dir "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download" \\
  --output-dir "{output_dir}" \\
  --api-url "http://10.208.60.212:8000" \\
  --max-concurrent 80 \\
  --batch-size 500 \\
  --resume

echo "✅ 处理完成!"
echo "结果保存在: {output_dir}"
echo "日志文件: {output_dir}/processing.log"
echo "检查点文件: {output_dir}/processing_checkpoint.json"
"""
    
    script_file = output_dir / "run_processing.sh"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    script_file.chmod(0o755)  # 添加执行权限
    print(f"✅ 生成运行脚本: {script_file}")

def main():
    """主函数"""
    # 创建示例输出目录
    output_dir = Path("examples/sample_processing_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 生成示例数据处理输出文件到: {output_dir}")
    print("=" * 60)
    
    # 生成各种文件
    generate_processing_checkpoint(output_dir)
    generate_processing_log(output_dir)
    generate_sample_result_files(output_dir)
    generate_performance_test_results(output_dir)
    generate_run_script(output_dir)
    
    print("\n" + "=" * 60)
    print(f"🎉 完成！生成的文件:")
    for file_path in sorted(output_dir.iterdir()):
        size = file_path.stat().st_size
        print(f"  📁 {file_path.name} ({size:,} bytes)")
    
    print(f"\n💡 查看目录内容:")
    print(f"  ls -la {output_dir}")

if __name__ == "__main__":
    main()
