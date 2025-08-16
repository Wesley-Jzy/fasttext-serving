#!/usr/bin/env python3
"""
测试服务端真实处理能力
直接压测API来验证是单个服务还是客户端的瓶颈
"""

import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor
import threading

async def test_api_capacity(api_url, concurrent_requests=50, duration=30):
    """测试API服务的真实容量"""
    
    # 测试数据
    test_payload = {
        "texts": ["print('hello world')" * 10] * 30  # 30个样本，模拟真实batch
    }
    
    # 统计数据
    stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_samples": 0,
        "start_time": time.time(),
        "response_times": []
    }
    
    async def make_request(session, request_id):
        try:
            start = time.time()
            async with session.post(
                f"{api_url}/predict",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    stats["successful_requests"] += 1
                    stats["total_samples"] += len(test_payload["texts"])
                else:
                    stats["failed_requests"] += 1
                
                stats["response_times"].append(time.time() - start)
                stats["total_requests"] += 1
                
        except Exception as e:
            stats["failed_requests"] += 1
            stats["total_requests"] += 1
            print(f"请求 {request_id} 失败: {e}")
    
    # 创建HTTP会话
    connector = aiohttp.TCPConnector(
        limit=concurrent_requests * 2,
        limit_per_host=concurrent_requests * 2,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"🚀 开始压测: {concurrent_requests}并发, 持续{duration}秒")
        print(f"🎯 目标API: {api_url}")
        
        end_time = time.time() + duration
        request_id = 0
        
        while time.time() < end_time:
            # 启动一批并发请求
            tasks = []
            for _ in range(concurrent_requests):
                task = asyncio.create_task(make_request(session, request_id))
                tasks.append(task)
                request_id += 1
            
            # 等待这批请求完成
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 打印实时统计
            elapsed = time.time() - stats["start_time"]
            if elapsed > 0:
                rps = stats["total_requests"] / elapsed
                sps = stats["total_samples"] / elapsed
                avg_response_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
                
                print(f"⏱️  {elapsed:.1f}s: {stats['total_requests']} 请求, "
                      f"{rps:.1f} RPS, {sps:.1f} samples/sec, "
                      f"平均响应: {avg_response_time:.3f}s")
            
            # 短暂休息避免过度压测
            await asyncio.sleep(0.1)
    
    # 最终统计
    total_time = time.time() - stats["start_time"]
    final_rps = stats["total_requests"] / total_time
    final_sps = stats["total_samples"] / total_time
    success_rate = stats["successful_requests"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
    avg_response_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
    
    print(f"\n📊 最终结果:")
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功率: {success_rate:.1%}")
    print(f"请求速率: {final_rps:.1f} RPS")
    print(f"样本处理速率: {final_sps:.1f} samples/sec")
    print(f"平均响应时间: {avg_response_time:.3f}s")
    
    return {
        "requests_per_second": final_rps,
        "samples_per_second": final_sps,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time
    }

async def compare_service_configs(api_url):
    """比较不同并发配置下的服务性能"""
    print("🧪 测试不同并发级别下的API性能")
    
    configs = [
        {"concurrent": 20, "duration": 20},
        {"concurrent": 50, "duration": 20}, 
        {"concurrent": 100, "duration": 20},
        {"concurrent": 200, "duration": 20},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['concurrent']}并发")
        
        result = await test_api_capacity(
            api_url, 
            concurrent_requests=config['concurrent'],
            duration=config['duration']
        )
        
        result['config'] = config
        results.append(result)
        
        # 休息一下避免影响下次测试
        await asyncio.sleep(5)
    
    # 输出对比结果
    print(f"\n📋 性能对比结果:")
    print(f"{'并发数':<8} {'RPS':<10} {'Samples/s':<12} {'响应时间':<10} {'成功率':<8}")
    print("-" * 50)
    
    for result in results:
        config = result['config']
        print(f"{config['concurrent']:<8} "
              f"{result['requests_per_second']:<10.1f} "
              f"{result['samples_per_second']:<12.1f} "
              f"{result['avg_response_time']:<10.3f} "
              f"{result['success_rate']:<8.1%}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python3 test_service_capacity.py <API_URL>")
        print("示例: python3 test_service_capacity.py http://localhost:8000")
        sys.exit(1)
    
    api_url = sys.argv[1]
    
    # 运行测试
    asyncio.run(compare_service_configs(api_url))
