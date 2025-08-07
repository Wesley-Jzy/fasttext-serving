#!/usr/bin/env python3
"""
FastText Serving 服务测试脚本
用于快速验证部署的服务是否正常工作
"""

import sys
import requests
import json
import time


def test_health(base_url):
    """测试健康检查"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data}")
            return True
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False


def test_prediction(base_url):
    """测试预测功能"""
    test_texts = [
        "Which baking dish is best to bake a banana bread?",
        "Why not put knives in the dishwasher?",
        "How do I make the perfect chocolate chip cookies?",
        "What temperature should I use for roasting vegetables?"
    ]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=test_texts,
            params={"k": 2, "threshold": 0.1},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ 预测测试通过 (耗时: {end_time - start_time:.2f}秒)")
            print(f"   处理了 {len(test_texts)} 个文本")
            
            print("\n📊 预测结果示例:")
            for i, (text, result) in enumerate(zip(test_texts, results)):
                labels, scores = result
                print(f"   {i+1}. \"{text[:40]}...\"")
                print(f"      → {labels[0]} (置信度: {scores[0]:.3f})")
            
            return True
        else:
            print(f"❌ 预测测试失败: HTTP {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 预测测试异常: {e}")
        return False


def test_large_batch(base_url):
    """测试大批量处理"""
    print("\n🚀 测试大批量处理...")
    
    # 创建1000个测试样本
    base_texts = [
        "Which baking dish is best to bake a banana bread?",
        "Why not put knives in the dishwasher?",
        "How do I make the perfect chocolate chip cookies?",
        "What temperature should I use for roasting vegetables?",
        "How long should I marinate chicken for grilling?"
    ]
    
    large_batch = []
    for i in range(200):  # 200 * 5 = 1000个样本
        for text in base_texts:
            large_batch.append(f"{text} (sample {i})")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=large_batch,
            timeout=120  # 2分钟超时
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            throughput = len(large_batch) / (end_time - start_time)
            print(f"✅ 大批量测试通过")
            print(f"   处理样本数: {len(large_batch)}")
            print(f"   总耗时: {end_time - start_time:.2f}秒")
            print(f"   吞吐量: {throughput:.1f} 样本/秒")
            
            # 统计分类结果
            categories = {}
            for result in results:
                labels, scores = result
                if labels:
                    category = labels[0]
                    categories[category] = categories.get(category, 0) + 1
            
            print("\n📈 分类统计:")
            for category, count in sorted(categories.items()):
                percentage = (count / len(results)) * 100
                print(f"   {category}: {count} ({percentage:.1f}%)")
            
            return True
        else:
            print(f"❌ 大批量测试失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 大批量测试异常: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_service.py <service-url>")
        print("Example: python test_service.py http://your-k8s-service:8000")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    print(f"🧪 开始测试 FastText Serving: {base_url}")
    print("=" * 60)
    
    # 1. 健康检查
    print("1. 健康检查测试...")
    if not test_health(base_url):
        print("\n❌ 健康检查失败，停止测试")
        sys.exit(1)
    
    # 2. 基础预测测试
    print("\n2. 基础预测测试...")
    if not test_prediction(base_url):
        print("\n❌ 基础预测测试失败")
        sys.exit(1)
    
    # 3. 大批量测试
    if not test_large_batch(base_url):
        print("\n❌ 大批量测试失败")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！FastText Serving 服务运行正常")
    print("\n💡 接下来你可以:")
    print(f"   - 使用 client.py 进行自定义测试")
    print(f"   - 部署到生产环境处理真实数据")
    print(f"   - 根据性能需求调整并发和批处理参数")


if __name__ == "__main__":
    main() 