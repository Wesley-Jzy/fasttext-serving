#!/usr/bin/env python3
"""
模型验证脚本
通过HTTP API测试FastText服务中的模型功能、二分类能力、性能表现
"""
import os
import sys
import time
import json
import requests
from typing import List, Dict, Any

def test_http_client():
    """测试HTTP客户端库"""
    print("📦 测试HTTP客户端库")
    
    try:
        import requests
        import aiohttp
        print(f"✅ HTTP客户端库导入成功")
        print(f"  requests: 已安装")
        print(f"  aiohttp: 已安装")
        return True
    except ImportError as e:
        print(f"❌ HTTP客户端库导入失败: {e}")
        return False

def check_model_file(model_path: str):
    """检查模型文件是否存在（在马来环境中）"""
    print(f"🤖 检查模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在")
        return False
    
    try:
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"✅ 模型文件存在")
        print(f"  文件大小: {size_mb:.1f}MB")
        return True
    except Exception as e:
        print(f"❌ 无法访问模型文件: {e}")
        return False

def test_service_with_model(service_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """通过HTTP API测试服务中的模型"""
    print(f"🌐 测试服务模型功能: {service_url}")
    
    # 1. 健康检查
    try:
        health_response = requests.get(f"{service_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ 服务健康状态正常")
        else:
            print(f"⚠️ 服务健康状态异常: {health_response.status_code}")
            return {"status": "service_unhealthy"}
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return {"status": "connection_failed", "error": str(e)}
    
    # 2. 测试基本预测功能
    test_samples = [
        "def hello_world():\n    print('Hello, World!')\n",
        "import numpy as np\narray = np.zeros((10, 10))\nprint(array.shape)\n"
    ]
    
    try:
        predict_url = f"{service_url}/predict"
        params = {"k": 2, "threshold": 0.0}
        
        response = requests.post(
            predict_url,
            json=test_samples,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ 模型预测功能正常")
            
            # 分析结果格式
            if isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, list) and len(first_result) == 2:
                    labels, scores = first_result
                    print(f"📋 预测标签: {labels}")
                    print(f"📊 预测分数: {scores}")
                    
                    # 检查是否是二分类
                    is_binary = len(labels) == 2
                    print(f"🎯 二分类检查: {'✅' if is_binary else '❌'}")
                    
                    # 检查标签格式
                    label_format_ok = all(isinstance(label, str) for label in labels)
                    print(f"🏷️ 标签格式: {'✅' if label_format_ok else '❌'}")
                    
                    return {
                        "status": "success",
                        "is_binary": is_binary,
                        "labels": labels,
                        "sample_scores": scores,
                        "label_format_ok": label_format_ok
                    }
                else:
                    print(f"⚠️ 意外的结果格式: {first_result}")
                    return {"status": "unexpected_format", "result": first_result}
            else:
                print(f"⚠️ 空结果或格式错误: {results}")
                return {"status": "empty_result"}
        else:
            print(f"❌ 预测请求失败: {response.status_code}")
            return {"status": "prediction_failed", "status_code": response.status_code}
            
    except Exception as e:
        print(f"❌ 预测测试异常: {e}")
        return {"status": "prediction_error", "error": str(e)}

def test_binary_classification_behavior(service_url: str) -> Dict[str, Any]:
    """专门测试二分类行为"""
    print(f"\n🎯 测试二分类行为")
    
    # 准备不同类型的代码样本，期望得到不同的分类结果
    test_cases = [
        {
            "name": "简单函数",
            "code": "def add(a, b):\n    return a + b\n"
        },
        {
            "name": "复杂类定义",
            "code": "class ComplexProcessor:\n    def __init__(self, config):\n        self.config = config\n        self.results = []\n    \n    def process(self, data):\n        for item in data:\n            result = self.transform(item)\n            self.results.append(result)\n        return self.results\n"
        },
        {
            "name": "错误代码",
            "code": "def broken_function(\n    # 缺少参数定义\n    return undefined_variable\n"
        },
        {
            "name": "注释块",
            "code": "# This is just a comment\n# TODO: implement feature\n# FIXME: handle edge case\n"
        },
        {
            "name": "SQL查询",
            "code": "SELECT users.name, orders.amount\nFROM users\nJOIN orders ON users.id = orders.user_id\nWHERE orders.created_at > '2024-01-01';\n"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n--- 测试: {test_case['name']} ---")
        
        try:
            response = requests.post(
                f"{service_url}/predict",
                json=[test_case['code']],
                params={"k": 2, "threshold": 0.0},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()[0]  # 第一个样本的结果
                labels, scores = result
                
                prediction = labels[0] if labels else "unknown"
                confidence = scores[0] if scores else 0.0
                
                print(f"  预测: {prediction} (置信度: {confidence:.3f})")
                print(f"  所有标签: {labels}")
                print(f"  所有分数: {[f'{s:.3f}' for s in scores]}")
                
                results.append({
                    "test_name": test_case['name'],
                    "prediction": prediction,
                    "confidence": confidence,
                    "all_labels": labels,
                    "all_scores": scores
                })
            else:
                print(f"  ❌ 请求失败: {response.status_code}")
                results.append({
                    "test_name": test_case['name'],
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            results.append({
                "test_name": test_case['name'],
                "error": str(e)
            })
    
    return {"binary_classification_tests": results}

def test_performance_via_api(service_url: str, sample_count: int = 50) -> Dict[str, Any]:
    """通过API测试性能"""
    print(f"\n⚡ API性能测试 (样本数: {sample_count})")
    
    # 生成测试样本
    base_code = "def test_function():\n    return 'test'\n"
    test_samples = [f"# Sample {i}\n{base_code}" for i in range(sample_count)]
    
    # 测试不同批次大小的性能
    batch_sizes = [1, 10, 25, 50] if sample_count >= 50 else [1, min(10, sample_count)]
    
    performance_results = []
    
    for batch_size in batch_sizes:
        if batch_size > len(test_samples):
            continue
            
        batch_samples = test_samples[:batch_size]
        
        print(f"📊 测试批次大小: {batch_size}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{service_url}/predict",
                json=batch_samples,
                params={"k": 2, "threshold": 0.0},
                timeout=60
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json()
                throughput = len(batch_samples) / elapsed if elapsed > 0 else 0
                
                print(f"  ✅ 成功处理 {len(batch_samples)} 样本")
                print(f"  耗时: {elapsed:.3f}秒")
                print(f"  吞吐量: {throughput:.0f} samples/sec")
                
                performance_results.append({
                    "batch_size": batch_size,
                    "samples": len(batch_samples),
                    "time_seconds": elapsed,
                    "throughput": throughput,
                    "success": True
                })
            else:
                print(f"  ❌ 失败: HTTP {response.status_code}")
                performance_results.append({
                    "batch_size": batch_size,
                    "success": False,
                    "status_code": response.status_code
                })
                
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            performance_results.append({
                "batch_size": batch_size,
                "success": False,
                "error": str(e)
            })
    
    return {"performance_tests": performance_results}

def main():
    """主函数"""
    print("🤖 FastText模型验证 (通过HTTP API)")
    print("=" * 50)
    
    model_path = "/mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin"
    # 修改为实际的FastText服务地址
    service_url = "http://localhost:8000"  # TODO: 修改为实际服务地址，如 http://服务IP:8000
    
    # 1. 测试HTTP客户端
    if not test_http_client():
        print("❌ HTTP客户端库有问题，停止测试")
        return
    
    # 2. 检查模型文件（可选，因为服务可能在不同机器上）
    model_exists = check_model_file(model_path)
    if not model_exists:
        print("⚠️ 本地模型文件不存在，但服务可能在其他位置有模型")
    
    # 3. 测试服务中的模型
    service_test = test_service_with_model(service_url)
    if service_test.get("status") not in ["success"]:
        print("❌ 服务模型测试失败，检查服务是否运行")
        print("💡 提示：如果服务在其他地址，请修改 service_url 变量")
        
        # 保存基础结果
        basic_results = {
            "model_file_exists": model_exists,
            "service_test": service_test
        }
        
        output_file = "model_validation_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(basic_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 基础结果已保存到: {output_file}")
        return
    
    # 4. 二分类行为测试
    binary_test = test_binary_classification_behavior(service_url)
    
    # 5. 性能测试
    performance_test = test_performance_via_api(service_url, sample_count=30)
    
    # 汇总结果
    validation_results = {
        "model_file_exists": model_exists,
        "service_test": service_test,
        "binary_classification": binary_test,
        "performance": performance_test
    }
    
    # 保存结果
    output_file = "model_validation_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 模型验证完成!")
    print(f"📄 详细结果已保存到: {output_file}")
    print(f"\n📋 下一步:")
    print(f"  1. 检查验证结果文件")
    print(f"  2. 如果服务在其他地址，修改脚本中的service_url")
    print(f"  3. 执行服务测试: python3 tests/04_service_test.py")

if __name__ == "__main__":
    main()