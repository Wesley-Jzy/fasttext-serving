#!/usr/bin/env python3
"""
数据探索脚本
深入分析the-stack-v2数据的结构、内容分布、样本特征
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

def analyze_parquet_files(data_dir: str, max_files: int = 5) -> Dict[str, Any]:
    """分析parquet文件的基本信息"""
    print(f"📁 分析数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("❌ 数据目录不存在")
        return {}
    
    # 获取所有parquet文件
    files = list(Path(data_dir).glob("*.parquet"))
    total_files = len(files)
    
    print(f"📊 总文件数: {total_files}")
    
    if total_files == 0:
        print("⚠️ 未找到parquet文件")
        return {"total_files": 0}
    
    # 分析前几个文件
    analyze_files = files[:max_files]
    print(f"🔍 详细分析前 {len(analyze_files)} 个文件")
    
    file_info = []
    total_rows = 0
    corrupted_files = []
    
    for i, file_path in enumerate(analyze_files):
        print(f"\n--- 文件 {i+1}: {file_path.name} ---")
        
        try:
            # 获取文件基本信息
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"文件大小: {size_mb:.1f}MB")
            
            # 尝试读取parquet文件
            df = pd.read_parquet(file_path, engine='pyarrow')
            rows = len(df)
            cols = list(df.columns)
            
            print(f"数据行数: {rows:,}")
            print(f"列名: {cols}")
            
            file_info.append({
                "filename": file_path.name,
                "size_mb": round(size_mb, 1),
                "rows": rows,
                "columns": cols
            })
            
            total_rows += rows
            
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            corrupted_files.append(file_path.name)
    
    result = {
        "total_files": total_files,
        "analyzed_files": len(file_info),
        "corrupted_files": corrupted_files,
        "file_details": file_info,
        "estimated_total_rows": int(total_rows * total_files / len(analyze_files)) if file_info else 0
    }
    
    print(f"\n📈 汇总信息:")
    print(f"  成功分析: {len(file_info)}/{len(analyze_files)} 个文件")
    print(f"  损坏文件: {len(corrupted_files)} 个")
    print(f"  预估总行数: {result['estimated_total_rows']:,}")
    
    return result

def analyze_content_column(data_dir: str, sample_size: int = 100) -> Dict[str, Any]:
    """专门分析content列的特征"""
    print(f"\n📝 分析content列特征 (样本数: {sample_size})")
    
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        return {}
    
    content_samples = []
    
    # 从多个文件采样
    for file_path in files[:3]:  # 最多从3个文件采样
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            if 'content' not in df.columns:
                print(f"⚠️ {file_path.name} 没有content列")
                continue
            
            # 随机采样
            sample_df = df.sample(n=min(sample_size//3, len(df)), random_state=42)
            content_samples.extend(sample_df['content'].tolist())
            
            if len(content_samples) >= sample_size:
                break
                
        except Exception as e:
            print(f"⚠️ 采样 {file_path.name} 失败: {e}")
    
    if not content_samples:
        print("❌ 无法获取content样本")
        return {}
    
    # 分析content特征
    lengths = [len(str(content)) for content in content_samples]
    
    analysis = {
        "sample_count": len(content_samples),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths)
        }
    }
    
    print(f"📊 Content长度统计:")
    print(f"  样本数: {analysis['sample_count']}")
    print(f"  最短: {analysis['length_stats']['min']:,} 字符")
    print(f"  最长: {analysis['length_stats']['max']:,} 字符")
    print(f"  平均: {analysis['length_stats']['mean']:.0f} 字符")
    print(f"  中位数: {analysis['length_stats']['median']:.0f} 字符")
    
    # 显示几个样本
    print(f"\n📄 Content样本预览:")
    for i, content in enumerate(content_samples[:3]):
        content_str = str(content)
        preview = content_str[:200].replace('\n', '\\n').replace('\t', '\\t')
        print(f"  样本 {i+1} (长度: {len(content_str)}): {preview}...")
    
    return analysis

def detect_file_patterns(data_dir: str) -> Dict[str, Any]:
    """检测文件的命名模式和时间特征"""
    print(f"\n🔍 检测文件模式")
    
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        return {}
    
    # 分析文件名模式
    filenames = [f.name for f in files]
    
    # 按文件名排序
    filenames.sort()
    
    print(f"📋 文件名模式 (前10个):")
    for name in filenames[:10]:
        print(f"  {name}")
    
    if len(filenames) > 10:
        print(f"  ... 还有 {len(filenames)-10} 个文件")
    
    # 检查是否有时间戳模式
    patterns = {
        "total_files": len(filenames),
        "first_file": filenames[0] if filenames else "",
        "last_file": filenames[-1] if filenames else "",
        "sample_names": filenames[:5]
    }
    
    return patterns

def check_processing_feasibility(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """评估处理可行性"""
    print(f"\n⚖️ 处理可行性评估")
    
    if not analysis_result:
        return {}
    
    total_rows = analysis_result.get("estimated_total_rows", 0)
    content_stats = analysis_result.get("content_stats", {})
    
    # 估算处理时间和资源需求
    if total_rows > 0:
        # 假设处理速度: 1000 samples/sec
        estimated_seconds = total_rows / 1000
        estimated_hours = estimated_seconds / 3600
        
        print(f"📊 处理估算:")
        print(f"  预估样本数: {total_rows:,}")
        print(f"  预估处理时间 (1K/sec): {estimated_hours:.1f} 小时")
        
        if content_stats and "length_stats" in content_stats:
            avg_length = content_stats["length_stats"]["mean"]
            max_length = content_stats["length_stats"]["max"]
            
            print(f"📝 文本特征:")
            print(f"  平均长度: {avg_length:.0f} 字符")
            print(f"  最大长度: {max_length:,} 字符")
            
            # 检查是否需要特殊处理
            if max_length > 100000:
                print("⚠️ 发现超长文本，可能需要分块处理")
            
            if avg_length > 10000:
                print("⚠️ 平均文本较长，建议增加处理超时时间")
    
    feasibility = {
        "total_samples": total_rows,
        "estimated_hours": estimated_hours if total_rows > 0 else 0,
        "needs_chunking": content_stats.get("length_stats", {}).get("max", 0) > 100000,
        "avg_text_length": content_stats.get("length_stats", {}).get("mean", 0)
    }
    
    return feasibility

def main():
    """主函数"""
    print("🔍 The-Stack-v2 数据探索")
    print("=" * 50)
    
    data_dir = "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download"
    
    # 1. 分析parquet文件
    file_analysis = analyze_parquet_files(data_dir)
    
    # 2. 分析content列
    content_analysis = analyze_content_column(data_dir)
    
    # 3. 检测文件模式
    pattern_analysis = detect_file_patterns(data_dir)
    
    # 4. 评估处理可行性
    full_analysis = {
        "file_info": file_analysis,
        "content_stats": content_analysis,
        "file_patterns": pattern_analysis
    }
    
    feasibility = check_processing_feasibility(full_analysis)
    
    # 保存分析结果
    output_file = "data_analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            **full_analysis,
            "feasibility": feasibility
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据探索完成!")
    print(f"📄 详细结果已保存到: {output_file}")
    print(f"\n📋 下一步:")
    print(f"  1. 检查分析结果文件")
    print(f"  2. 执行模型验证: python3 tests/03_model_validator.py")

if __name__ == "__main__":
    main()
