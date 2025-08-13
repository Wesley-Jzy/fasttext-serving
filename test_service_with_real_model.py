#!/usr/bin/env python3
import requests
import json
import time

def test_fasttext_service():
    base_url = "http://localhost:8000"
    
    print("🔍 测试 FastText 服务 (真实模型)")
    print("=" * 45)
    
    # 1. 健康检查
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务健康检查通过")
            print(f"   响应: {response.json()}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接服务: {e}")
        return
    
    # 2. 测试基础预测
    test_cases = [
        "print('hello world')",
        "def function(): pass",
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
        "// This is a comment\nint main() { return 0; }",
        "SELECT * FROM users WHERE id = 1;"
    ]
    
    print(f"\n🔍 测试单个预测:")
    for i, test_text in enumerate(test_cases):
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=[test_text],
                params={"k": 2, "threshold": 0.0},
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                labels, scores = results[0]
                
                print(f"\n   测试 {i+1}:")
                print(f"   输入: {test_text[:50]}{'...' if len(test_text) > 50 else ''}")
                print(f"   标签: {labels}")
                print(f"   分数: {[f'{s:.4f}' for s in scores]}")
            else:
                print(f"   ❌ 预测失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ 请求异常: {e}")
    
    # 3. 测试批量预测
    print(f"\n🎯 测试批量预测:")
    try:
        batch_texts = test_cases[:3]
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=batch_texts,
            params={"k": 2, "threshold": 0.0},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ 批量预测成功")
            print(f"   批量大小: {len(batch_texts)}")
            print(f"   处理时间: {(end_time - start_time):.3f} 秒")
            print(f"   平均延迟: {(end_time - start_time) / len(batch_texts) * 1000:.1f} ms/样本")
            
            for i, (labels, scores) in enumerate(results):
                print(f"   结果 {i+1}: {labels[0]} ({scores[0]:.4f})")
        else:
            print(f"   ❌ 批量预测失败: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ 批量预测异常: {e}")
    
    # 4. 测试大文本处理
    print(f"\n📄 测试大文本处理:")
    try:
        # 创建一个较大的代码文本
        large_code = "import numpy as np\nimport pandas as pd\n\n" * 1000
        large_code += "def process_data():\n    # This is a large function\n" * 500
        
        print(f"   大文本长度: {len(large_code)} 字符")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=[large_code],
            params={"k": 2, "threshold": 0.0},
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            labels, scores = results[0]
            print(f"   ✅ 大文本处理成功")
            print(f"   处理时间: {(end_time - start_time):.3f} 秒")
            print(f"   预测结果: {labels[0]} ({scores[0]:.4f})")
        else:
            print(f"   ❌ 大文本处理失败: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ 大文本处理异常: {e}")

if __name__ == "__main__":
    test_fasttext_service()
