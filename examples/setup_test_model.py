#!/usr/bin/env python3
"""
设置 FastText 测试模型
用于代码质量分类测试
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess


def create_sample_training_data():
    """创建样本训练数据"""
    sample_data = [
        # 高质量代码示例
        "__label__high_quality def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "__label__high_quality class DataProcessor:\n    def __init__(self, config):\n        self.config = config\n    def process(self, data):\n        return self._validate(data)",
        "__label__high_quality import logging\nlogger = logging.getLogger(__name__)\n\ndef calculate_metrics(data):\n    \"\"\"Calculate performance metrics\"\"\"\n    try:\n        return sum(data) / len(data)\n    except ZeroDivisionError:\n        logger.warning(\"Empty data provided\")\n        return 0.0",
        "__label__high_quality from typing import List, Dict, Optional\n\ndef process_items(items: List[Dict]) -> Optional[List[Dict]]:\n    \"\"\"Process a list of items with proper type hints\"\"\"\n    if not items:\n        return None\n    return [item for item in items if item.get('valid')]",
        
        # 低质量代码示例  
        "__label__low_quality x=1;y=2;print(x+y)",
        "__label__low_quality def func():pass",
        "__label__low_quality a=b=c=1\nprint(a,b,c)",
        "__label__low_quality for i in range(10):exec('print(i)')",
        "__label__low_quality import *\nfrom os import *",
        "__label__low_quality x=eval(input())",
        
        # 中等质量代码示例
        "__label__medium_quality def add(a, b):\n    return a + b\n\nresult = add(1, 2)\nprint(result)",
        "__label__medium_quality data = [1, 2, 3]\nfor item in data:\n    print(item * 2)",
        "__label__medium_quality def get_user_info(user_id):\n    if user_id > 0:\n        return {'id': user_id, 'name': 'user'}\n    return None",
    ]
    
    # 增加数据量以提高模型质量
    extended_data = []
    for _ in range(100):  # 重复100次
        extended_data.extend(sample_data)
    
    return extended_data


def train_fasttext_model(output_path: str):
    """训练 FastText 模型"""
    print("Creating sample training data...")
    training_data = create_sample_training_data()
    
    # 写入临时训练文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for line in training_data:
            f.write(line + '\n')
        train_file = f.name
    
    try:
        print(f"Training FastText model with {len(training_data)} samples...")
        
        # 使用 fasttext 命令行工具训练模型
        cmd = [
            'fasttext', 'supervised',
            '-input', train_file,
            '-output', output_path.replace('.bin', ''),
            '-epoch', '25',
            '-lr', '0.1',
            '-wordNgrams', '2',
            '-dim', '100',
            '-minCount', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"FastText model trained successfully: {output_path}")
            return True
        else:
            print(f"FastText training failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: fasttext command not found. Please install FastText first.")
        print("Installation: pip install fasttext")
        return False
    except Exception as e:
        print(f"Training failed: {e}")
        return False
    finally:
        # 清理临时文件
        os.unlink(train_file)


def download_pretrained_model(output_path: str):
    """下载预训练模型（备用方案）"""
    try:
        import requests
        
        # 使用一个小的预训练模型进行测试
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        
        print(f"Downloading pretrained model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded pretrained model: {output_path}")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup FastText test model")
    parser.add_argument("--output", "-o", default="models/test_model.bin", 
                       help="Output model path")
    parser.add_argument("--method", choices=["train", "download"], default="train",
                       help="Setup method: train new model or download pretrained")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing model")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not args.force:
        print(f"Model already exists: {output_path}")
        print("Use --force to overwrite")
        return
    
    success = False
    if args.method == "train":
        success = train_fasttext_model(str(output_path))
        
        # 如果训练失败，尝试下载预训练模型
        if not success:
            print("Training failed, trying to download pretrained model...")
            success = download_pretrained_model(str(output_path))
    else:
        success = download_pretrained_model(str(output_path))
    
    if success:
        print(f"\n✅ Model setup completed: {output_path}")
        print(f"Model size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print("\nYou can now test the FastText serving with:")
        print(f"fasttext-serving --model {output_path}")
    else:
        print("\n❌ Model setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 