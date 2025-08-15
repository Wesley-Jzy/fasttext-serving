#!/usr/bin/env python3
"""
测试内存优化效果的脚本
"""

import psutil
import time
import os
from pathlib import Path

def get_memory_usage():
    """获取当前进程内存使用量(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_old_method(file_path):
    """测试旧方法：一次性读取整个文件"""
    print("🧪 测试旧方法（一次性读取）...")
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        import pandas as pd
        df = pd.read_parquet(file_path)
        peak_memory = get_memory_usage()
        elapsed = time.time() - start_time
        
        print(f"  ✅ 成功读取 {len(df):,} 行")
        print(f"  📊 内存使用: {start_memory:.1f} → {peak_memory:.1f} MB (增加 {peak_memory - start_memory:.1f} MB)")
        print(f"  ⏱️  耗时: {elapsed:.2f} 秒")
        
        return peak_memory - start_memory
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return None

def test_new_method(file_path):
    """测试新方法：流式读取"""
    print("🚀 测试新方法（流式读取）...")
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        # 模拟处理配置
        cache_size = 1000  # 小缓存测试
        processed_rows = 0
        max_memory = start_memory
        
        for batch in parquet_file.iter_batches(batch_size=cache_size):
            df_chunk = batch.to_pandas()
            processed_rows += len(df_chunk)
            
            current_memory = get_memory_usage()
            max_memory = max(max_memory, current_memory)
            
            # 模拟处理时间
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        
        print(f"  ✅ 成功处理 {processed_rows:,} 行 (总计 {total_rows:,})")
        print(f"  📊 内存使用: {start_memory:.1f} → {max_memory:.1f} MB (峰值增加 {max_memory - start_memory:.1f} MB)")
        print(f"  ⏱️  耗时: {elapsed:.2f} 秒")
        
        return max_memory - start_memory
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return None

def main():
    """主测试函数"""
    print("🔍 内存优化效果测试")
    print("=" * 50)
    
    # 查找测试文件
    data_dir = Path("/mnt/project/yifan/data/the-stack-v2-dedup_batched_download")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("❌ 未找到parquet文件")
        return
    
    # 选择一个较大的文件进行测试
    test_file = max(parquet_files, key=lambda f: f.stat().st_size)
    file_size_mb = test_file.stat().st_size / (1024 * 1024)
    
    print(f"🎯 测试文件: {test_file.name}")
    print(f"📁 文件大小: {file_size_mb:.1f} MB")
    print()
    
    # 测试新方法（安全）
    new_memory = test_new_method(test_file)
    print()
    
    # 只有在文件不太大时才测试旧方法
    if file_size_mb < 500:  # 限制在500MB以下
        old_memory = test_old_method(test_file)
        print()
        
        if old_memory and new_memory:
            improvement = ((old_memory - new_memory) / old_memory) * 100
            print(f"🎉 内存优化效果: 减少 {improvement:.1f}% 内存使用")
    else:
        print("⚠️  文件过大，跳过旧方法测试（避免内存爆炸）")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()
