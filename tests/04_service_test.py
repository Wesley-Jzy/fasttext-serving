#!/usr/bin/env python3
"""
服务连通性测试脚本
测试FastText Serving服务的连接、API调用、性能表现
"""
import asyncio
import aiohttp
import requests
import json
import time
from typing import List, Dict, Any, Optional

def test_service_health(base_url: str) -> Dict[str, Any]:
    """测试服务健康状态"""
    print(f"🏥 测试服务健康状态: {base_url}")
    
    health_url = f"{base_url}/health"
    
    try:
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 服务健康正常")
            print(f"  响应: {health_data}")
            return {"status": "healthy", "response": health_data}
        else:
            print(f"⚠️ 服务响应异常: {response.status_code}")
            return {"status": "unhealthy", "status_code": response.status_code}
            
    except Exception as e:
        print(f"❌ 服务连接失败: {e}")
        return {"status": "error", "error": str(e)}

def test_basic_prediction(base_url: str) -> Dict[str, Any]:
    """测试基本预测功能"""
    print(f"\n🧪 测试基本预测功能")
    
    predict_url = f"{base_url}/predict"
    
    # 测试样本
    test_samples = [
        "def hello():\n    print('Hello World')\n",
        "import numpy as np\narray = np.array([1, 2, 3])\n",
        "SELECT * FROM users;\n"
    ]
    
    results = []
    
    for i, text in enumerate(test_samples):
        print(f"\n--- 测试样本 {i+1} ---")
        print(f"输入: {repr(text[:30])}...")
        
        try:
            payload = [text]  # 发送单个样本的数组
            params = {"k": 2, "threshold": 0.0}
            
            response = requests.post(
                predict_url, 
                json=payload, 
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 预测成功")
                print(f"  响应格式: {type(result)}")
                
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]  # 第一个样本的结果
                    if isinstance(prediction, list) and len(prediction) == 2:
                        labels, scores = prediction
                        print(f"  标签: {labels}")
                        print(f"  分数: {scores}")
                        
                        results.append({
                            "input": text[:50],
                            "success": True,
                            "labels": labels,
                            "scores": scores
                        })
                    else:
                        print(f"  意外的响应格式: {prediction}")
                        results.append({
                            "input": text[:50],
                            "success": False,
                            "error": f"Unexpected format: {prediction}"
                        })
                else:
                    print(f"  空响应或格式错误: {result}")
                    results.append({
                        "input": text[:50],
                        "success": False,
                        "error": f"Empty or invalid response"
                    })
            else:
                error_text = response.text
                print(f"❌ 请求失败: {response.status_code}")
                print(f"  错误: {error_text}")
                results.append({
                    "input": text[:50],
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_text
                })
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            results.append({
                "input": text[:50],
                "success": False,
                "error": str(e)
            })
    
    return {"basic_prediction_results": results}

async def test_async_performance(base_url: str, concurrent_requests: int = 5, samples_per_request: int = 10) -> Dict[str, Any]:
    """测试异步并发性能"""
    print(f"\n⚡ 异步并发性能测试")
    print(f"  并发请求数: {concurrent_requests}")
    print(f"  每请求样本数: {samples_per_request}")
    
    # 生成测试样本
    base_samples = [
        "def func():\n    return 42\n",
        "import sys\nprint(sys.version)\n",
        "class Test:\n    pass\n",
        "try:\n    x = 1/0\nexcept:\n    pass\n"
    ]
    
    # 为每个请求准备样本
    all_requests = []
    for req_id in range(concurrent_requests):
        request_samples = []
        for i in range(samples_per_request):
            base = base_samples[i % len(base_samples)]
            sample = f"# Request {req_id} Sample {i}\n{base}"
            request_samples.append(sample)
        all_requests.append(request_samples)
    
    async def send_request(session: aiohttp.ClientSession, request_id: int, samples: List[str]) -> Dict[str, Any]:
        """发送单个异步请求"""
        url = f"{base_url}/predict"
        params = {"k": 2, "threshold": 0.0}
        
        try:
            start_time = time.time()
            async with session.post(url, json=samples, params=params) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "request_id": request_id,
                        "success": True,
                        "samples_count": len(samples),
                        "response_time": elapsed,
                        "throughput": len(samples) / elapsed if elapsed > 0 else 0
                    }
                else:
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "response_time": elapsed
                    }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    # 执行并发请求
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        start_time = time.time()
        
        tasks = [
            send_request(session, i, samples) 
            for i, samples in enumerate(all_requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_elapsed = time.time() - start_time
    
    # 分析结果
    successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
    failed_results = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]
    
    total_samples = sum(r.get("samples_count", 0) for r in successful_results)
    total_throughput = total_samples / total_elapsed if total_elapsed > 0 else 0
    
    print(f"📊 并发测试结果:")
    print(f"  成功请求: {len(successful_results)}/{len(results)}")
    print(f"  总处理样本: {total_samples}")
    print(f"  总耗时: {total_elapsed:.2f}秒")
    print(f"  总吞吐量: {total_throughput:.0f} samples/sec")
    
    if successful_results:
        avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
        print(f"  平均响应时间: {avg_response_time:.3f}秒")
    
    return {
        "concurrent_requests": concurrent_requests,
        "samples_per_request": samples_per_request,
        "total_samples": total_samples,
        "total_time": total_elapsed,
        "total_throughput": total_throughput,
        "successful_requests": len(successful_results),
        "failed_requests": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0
    }

def test_large_batch(base_url: str, batch_sizes: List[int] = [10, 50, 100, 500]) -> Dict[str, Any]:
    """测试大批量处理"""
    print(f"\n📦 大批量处理测试")
    
    batch_results = []
    
    # 准备基础样本
    base_code = "def sample_function():\n    return 'test'\n"
    
    for batch_size in batch_sizes:
        print(f"\n--- 批量大小: {batch_size} ---")
        
        # 生成批量样本
        samples = []
        for i in range(batch_size):
            sample = f"# Sample {i}\n{base_code}"
            samples.append(sample)
        
        try:
            url = f"{base_url}/predict"
            params = {"k": 2, "threshold": 0.0}
            
            start_time = time.time()
            response = requests.post(
                url, 
                json=samples, 
                params=params,
                timeout=120  # 增加超时时间
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                throughput = batch_size / elapsed if elapsed > 0 else 0
                
                print(f"  ✅ 成功处理 {batch_size} 样本")
                print(f"  耗时: {elapsed:.2f}秒")
                print(f"  吞吐量: {throughput:.0f} samples/sec")
                
                batch_results.append({
                    "batch_size": batch_size,
                    "success": True,
                    "response_time": elapsed,
                    "throughput": throughput
                })
            else:
                print(f"  ❌ 失败: HTTP {response.status_code}")
                batch_results.append({
                    "batch_size": batch_size,
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text[:200]
                })
                
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            batch_results.append({
                "batch_size": batch_size,
                "success": False,
                "error": str(e)
            })
    
    return {"batch_test_results": batch_results}

async def main():
    """主函数"""
    print("🌐 FastText Serving 服务测试")
    print("=" * 50)
    
    # 这里需要用户提供实际的服务URL
    # base_url = "http://localhost:8000"  # 默认本地
    base_url = "http://fasttext-serving-4nodes.serving.va-mlp.anuttacon.com"  # 用户提供的URL
    
    print(f"🎯 测试目标: {base_url}")
    
    # 1. 健康检查
    health_result = test_service_health(base_url)
    
    if health_result.get("status") != "healthy":
        print("❌ 服务不健康，停止测试")
        return
    
    # 2. 基本预测测试
    basic_result = test_basic_prediction(base_url)
    
    # 3. 异步并发测试
    async_result = await test_async_performance(base_url, concurrent_requests=3, samples_per_request=5)
    
    # 4. 大批量测试
    batch_result = test_large_batch(base_url, batch_sizes=[10, 50, 100])
    
    # 汇总结果
    test_results = {
        "service_url": base_url,
        "health_check": health_result,
        "basic_prediction": basic_result,
        "async_performance": async_result,
        "batch_processing": batch_result
    }
    
    # 保存结果
    output_file = "service_test_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 服务测试完成!")
    print(f"📄 详细结果已保存到: {output_file}")
    print(f"\n📋 下一步:")
    print(f"  1. 检查服务测试结果")
    print(f"  2. 开发生产客户端: python3 tests/05_production_client.py")

if __name__ == "__main__":
    asyncio.run(main())
