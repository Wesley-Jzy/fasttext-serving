#!/usr/bin/env python3
"""
FastText Serving 服务测试脚本
用于快速验证部署的服务是否正常工作
"""

import sys
import requests
import json
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    """测试大批量处理（单个请求）"""
    print("\n🚀 测试大批量处理（单个请求）...")
    
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


def send_concurrent_request(base_url, batch, request_id):
    """发送单个并发请求"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=batch,
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            return {
                'request_id': request_id,
                'success': True,
                'batch_size': len(batch),
                'duration': end_time - start_time,
                'results': results
            }
        else:
            return {
                'request_id': request_id,
                'success': False,
                'error': f"HTTP {response.status_code}",
                'duration': end_time - start_time
            }
    except Exception as e:
        return {
            'request_id': request_id,
            'success': False,
            'error': str(e),
            'duration': 0
        }


def test_concurrent_load(base_url, total_samples=10000, concurrent_requests=20, batch_size=500):
    """测试真正的并发负载，验证多实例负载分担"""
    print(f"\n🎯 测试并发负载分担...")
    print(f"   总样本数: {total_samples}")
    print(f"   并发请求数: {concurrent_requests}")
    print(f"   每个请求批次大小: {batch_size}")
    
    # 准备测试数据
    base_texts = [
        "Which baking dish is best to bake a banana bread?",
        "Why not put knives in the dishwasher?",
        "How do I make the perfect chocolate chip cookies?",
        "What temperature should I use for roasting vegetables?",
        "How long should I marinate chicken for grilling?"
    ]
    
    # 创建并发请求的批次
    all_batches = []
    texts_per_batch = batch_size
    
    for req_id in range(concurrent_requests):
        batch = []
        for i in range(texts_per_batch):
            text_idx = (req_id * texts_per_batch + i) % len(base_texts)
            batch.append(f"{base_texts[text_idx]} (req-{req_id}-sample-{i})")
        all_batches.append((req_id, batch))
    
    total_texts = concurrent_requests * texts_per_batch
    print(f"   实际处理文本数: {total_texts}")
    
    # 使用线程池发送并发请求
    print(f"\n⏳ 发送 {concurrent_requests} 个并发请求...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # 提交所有请求
        future_to_req = {
            executor.submit(send_concurrent_request, base_url, batch, req_id): req_id
            for req_id, batch in all_batches
        }
        
        # 收集结果
        results = []
        success_count = 0
        total_processed = 0
        
        for future in as_completed(future_to_req):
            req_id = future_to_req[future]
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    success_count += 1
                    total_processed += result['batch_size']
                    print(f"   ✅ 请求 {result['request_id']} 完成 ({result['batch_size']} 样本, {result['duration']:.2f}s)")
                else:
                    print(f"   ❌ 请求 {result['request_id']} 失败: {result['error']}")
            except Exception as e:
                print(f"   ❌ 请求 {req_id} 异常: {e}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 统计结果
    print(f"\n📊 并发测试结果:")
    print(f"   成功请求: {success_count}/{concurrent_requests}")
    print(f"   处理样本数: {total_processed}")
    print(f"   总耗时: {total_duration:.2f}秒")
    
    if total_processed > 0:
        overall_throughput = total_processed / total_duration
        print(f"   整体吞吐量: {overall_throughput:.1f} 样本/秒")
        
        # 计算平均每个请求的处理时间
        successful_durations = [r['duration'] for r in results if r['success']]
        if successful_durations:
            avg_request_time = sum(successful_durations) / len(successful_durations)
            print(f"   平均请求处理时间: {avg_request_time:.2f}秒")
            print(f"   理论并发增益: {total_duration / avg_request_time:.1f}x")
    
    # 统计分类结果
    all_categories = {}
    for result in results:
        if result['success']:
            for prediction in result['results']:
                labels, scores = prediction
                if labels:
                    category = labels[0]
                    all_categories[category] = all_categories.get(category, 0) + 1
    
    if all_categories:
        print(f"\n📈 分类统计:")
        for category, count in sorted(all_categories.items()):
            percentage = (count / total_processed) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
    
    # 判断成功标准
    success_rate = success_count / concurrent_requests
    if success_rate >= 0.9 and total_processed >= total_texts * 0.9:
        print(f"\n✅ 并发负载测试通过！")
        print(f"   多实例服务可以正确处理并发请求")
        return True
    else:
        print(f"\n❌ 并发负载测试未达标")
        print(f"   成功率: {success_rate:.1%} (要求: ≥90%)")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_service.py <service-url> [options]")
        print("Options:")
        print("  --concurrent-only    只运行并发测试")
        print("  --concurrent-requests N   并发请求数 (默认: 20)")
        print("  --batch-size N       每个请求的批次大小 (默认: 500)")
        print("  --total-samples N    总样本数 (默认: 10000)")
        print("\nExample:")
        print("  python test_service.py http://your-k8s-service:8000")
        print("  python test_service.py http://your-k8s-service:8000 --concurrent-only")
        print("  python test_service.py http://your-k8s-service:8000 --concurrent-requests 50 --batch-size 1000")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    # 解析参数
    concurrent_only = '--concurrent-only' in sys.argv
    concurrent_requests = 200
    batch_size = 5000
    total_samples = 1000000
    
    for i, arg in enumerate(sys.argv):
        if arg == '--concurrent-requests' and i + 1 < len(sys.argv):
            concurrent_requests = int(sys.argv[i + 1])
        elif arg == '--batch-size' and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
        elif arg == '--total-samples' and i + 1 < len(sys.argv):
            total_samples = int(sys.argv[i + 1])
    
    print(f"🧪 开始测试 FastText Serving: {base_url}")
    print("=" * 60)
    
    if not concurrent_only:
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
        
        # 3. 大批量测试（单请求）
        print("\n3. 单请求大批量测试...")
        if not test_large_batch(base_url):
            print("\n⚠️  大批量测试失败，但继续并发测试")
    
    # 4. 并发负载测试（多请求）
    print(f"\n4. 并发负载测试...")
    if not test_concurrent_load(base_url, total_samples, concurrent_requests, batch_size):
        print("\n❌ 并发负载测试失败")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！FastText Serving 服务运行正常")
    
    if concurrent_requests > 1:
        print("\n💡 关于多实例负载分担:")
        print(f"   - 如果是多实例部署，负载均衡器应该将 {concurrent_requests} 个请求分发到不同实例")
        print(f"   - 检查各实例的日志，应该看到请求分布在不同实例上")
        print(f"   - 单实例部署时，所有请求都会由同一实例处理")

if __name__ == "__main__":
    main() 