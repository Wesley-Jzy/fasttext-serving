#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import sys

def explore_data_directory():
    data_dir = '/mnt/project/yifan/data/the-stack-v2-dedup_batched_download'
    
    print("🔍 探索 The Stack v2 数据目录")
    print("=" * 50)
    
    # 1. 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    print(f"✅ 数据目录存在: {data_dir}")
    
    # 2. 统计parquet文件
    parquet_files = list(Path(data_dir).glob("*.parquet"))
    print(f"📊 Parquet文件数量: {len(parquet_files)}")
    
    if len(parquet_files) == 0:
        print("❌ 未找到parquet文件")
        return
    
    # 3. 分析文件大小
    file_sizes = []
    for i, f in enumerate(parquet_files[:10]):  # 只查看前10个文件
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
            file_sizes.append(size_mb)
            print(f"📁 {f.name}: {size_mb:.1f} MB")
        except Exception as e:
            print(f"❌ 无法读取文件 {f.name}: {e}")
        
        if i >= 9:  # 限制输出
            break
    
    if file_sizes:
        print(f"📈 文件大小统计 (前10个):")
        print(f"   平均: {sum(file_sizes)/len(file_sizes):.1f} MB")
        print(f"   最小: {min(file_sizes):.1f} MB")
        print(f"   最大: {max(file_sizes):.1f} MB")
    
    # 4. 分析parquet内容结构
    print("\n🔍 分析Parquet文件结构")
    print("-" * 30)
    
    successful_reads = 0
    for i, parquet_file in enumerate(parquet_files[:5]):  # 只尝试前5个文件
        try:
            print(f"\n📄 文件 {i+1}: {parquet_file.name}")
            df = pd.read_parquet(parquet_file)
            
            print(f"   📊 形状: {df.shape}")
            print(f"   📋 列名: {list(df.columns)}")
            
            # 检查content列
            if 'content' in df.columns:
                content_sample = df['content'].iloc[0] if len(df) > 0 else None
                if content_sample:
                    content_str = str(content_sample)
                    print(f"   📝 Content样例:")
                    print(f"      长度: {len(content_str)} 字符")
                    print(f"      前200字符: {content_str[:200]}")
                    if len(content_str) > 200:
                        print("      ...")
                    
                    # 统计内容长度分布
                    lengths = df['content'].astype(str).str.len()
                    print(f"   📈 Content长度统计:")
                    print(f"      平均: {lengths.mean():.0f} 字符")
                    print(f"      中位数: {lengths.median():.0f} 字符")
                    print(f"      最大: {lengths.max():.0f} 字符")
                    print(f"      最小: {lengths.min():.0f} 字符")
            else:
                print("   ❌ 未找到 'content' 列")
            
            successful_reads += 1
            
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            continue
    
    print(f"\n✅ 成功读取 {successful_reads}/{min(5, len(parquet_files))} 个文件")

if __name__ == "__main__":
    explore_data_directory()
