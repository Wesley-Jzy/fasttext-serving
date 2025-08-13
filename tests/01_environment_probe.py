#!/usr/bin/env python3
"""
马来环境探测脚本
检查数据路径、模型文件、Python环境等基础信息
"""
import os
import sys
import platform
from pathlib import Path
import json

def check_python_environment():
    """检查Python环境"""
    print("🐍 Python环境信息:")
    print(f"  版本: {sys.version}")
    print(f"  可执行文件: {sys.executable}")
    print(f"  平台: {platform.platform()}")
    print()

def check_required_packages():
    """检查必要的Python包"""
    print("📦 Python包检查:")
    required_packages = [
        'pandas', 'pyarrow', 'aiohttp', 'requests', 'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}: 已安装")
        except ImportError:
            print(f"  ❌ {package}: 未安装")
    
    # 特别检查fasttext
    try:
        import fasttext
        print(f"  ✅ fasttext: 已安装 (版本: {fasttext.__version__ if hasattr(fasttext, '__version__') else '未知'})")
    except ImportError:
        print(f"  ❌ fasttext: 未安装")
    print()

def check_data_paths():
    """检查数据路径"""
    print("📁 数据路径检查:")
    
    # 检查数据目录
    data_path = "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download"
    print(f"  数据路径: {data_path}")
    
    if os.path.exists(data_path):
        print(f"  ✅ 路径存在")
        try:
            files = list(Path(data_path).glob("*.parquet"))
            print(f"  📊 发现 {len(files)} 个parquet文件")
            if files:
                # 显示前几个文件名和大小
                print("  前5个文件:")
                for f in files[:5]:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"    - {f.name} ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"  ⚠️ 无法列出文件: {e}")
    else:
        print(f"  ❌ 路径不存在")
    
    print()

def check_model_file():
    """检查模型文件"""
    print("🤖 模型文件检查:")
    
    model_path = "/mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin"
    print(f"  模型路径: {model_path}")
    
    if os.path.exists(model_path):
        print(f"  ✅ 文件存在")
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"  📏 文件大小: {size_mb:.1f}MB")
    else:
        print(f"  ❌ 文件不存在")
    
    print()

def check_sample_data():
    """检查样本数据结构"""
    print("🔍 样本数据结构检查:")
    
    data_path = "/mnt/project/yifan/data/the-stack-v2-dedup_batched_download"
    
    if not os.path.exists(data_path):
        print("  ⚠️ 数据路径不存在，跳过")
        return
    
    try:
        import pandas as pd
        
        # 找到第一个parquet文件
        files = list(Path(data_path).glob("*.parquet"))
        if not files:
            print("  ⚠️ 未找到parquet文件")
            return
        
        first_file = files[0]
        print(f"  正在分析文件: {first_file.name}")
        
        # 尝试读取文件头部
        try:
            df = pd.read_parquet(first_file, engine='pyarrow')
            print(f"  ✅ 成功读取，共 {len(df)} 行")
            print(f"  📋 列名: {list(df.columns)}")
            
            if 'content' in df.columns:
                # 检查content列
                content_sample = df['content'].iloc[0] if len(df) > 0 else None
                if content_sample:
                    print(f"  📝 content样本长度: {len(str(content_sample))} 字符")
                    print(f"  📝 content前100字符: {str(content_sample)[:100]}...")
            else:
                print("  ⚠️ 未找到'content'列")
                
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            # 尝试使用其他方法
            try:
                df_head = pd.read_parquet(first_file, engine='pyarrow', nrows=1)
                print(f"  📋 (部分读取)列名: {list(df_head.columns)}")
            except Exception as e2:
                print(f"  ❌ 完全无法读取: {e2}")
    
    except ImportError:
        print("  ⚠️ pandas未安装，无法检查数据")
    
    print()

def check_network_connectivity():
    """检查网络连接"""
    print("🌐 网络连接检查:")
    
    # 检查是否可以访问外网
    try:
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        print(f"  ✅ 外网连接正常 (状态码: {response.status_code})")
    except Exception as e:
        print(f"  ⚠️ 外网连接异常: {e}")
    
    print()

def main():
    """主函数"""
    print("🔍 FastText Serving 马来环境探测")
    print("=" * 50)
    print()
    
    check_python_environment()
    check_required_packages()
    check_data_paths()
    check_model_file()
    check_sample_data()
    check_network_connectivity()
    
    print("✅ 环境探测完成!")
    print("\n📋 下一步:")
    print("  1. 如果有缺失的包，请安装: pip install -r tests/requirements.txt")
    print("  2. 执行数据探索脚本: python3 tests/02_data_explorer.py")

if __name__ == "__main__":
    main()
