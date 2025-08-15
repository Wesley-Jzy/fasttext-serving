#!/usr/bin/env python3
"""
FastText API真实数据性能测试工具
使用真实The-Stack-v2数据测试不同并发数和批次大小下的API性能表现
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
import pandas as pd
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestConfig:
    """测试配置"""
    api_url: str
    data_dir: str
    concurrency_levels: List[int] = None
    batch_sizes: List[int] = None

@dataclass
class FileTestResult:
    """单文件测试结果"""
    file_name: str
    file_size_mb: float
    file_size_gb: float
    total_samples: int
    processed_samples: int
    successful_samples: int
    failed_samples: int
    concurrency: int
    batch_size: int
    processing_time: float
    throughput_sps: float  # samples per second
    throughput_gbs: float  # GB per second (原始数据)
    avg_latency: float
    success_rate: float
    error_types: Dict[str, int]
    sample_errors: List[str]  # 前几个错误样本

class APIPerformanceTester:
    """API性能测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session: aiohttp.ClientSession = None
        
    async def __aenter__(self):
        # 配置连接池
        connector = aiohttp.TCPConnector(
            limit=1000,  # 总连接池大小
            limit_per_host=500,  # 每个主机连接数
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_test_texts(self, count: int = 1000) -> List[str]:
        """生成测试文本"""
        # 基础代码样本
        base_samples = [
            "def hello_world():\n    print('Hello, World!')\n    return True",
            "import numpy as np\ndef calculate_mean(data):\n    return np.mean(data)",
            "class DataProcessor:\n    def __init__(self):\n        self.data = []\n    def process(self):\n        pass",
            "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
            "from typing import List, Dict\ndef merge_dicts(dicts: List[Dict]) -> Dict:\n    result = {}\n    for d in dicts:\n        result.update(d)\n    return result",
            "SELECT users.name, COUNT(orders.id) as order_count\nFROM users\nLEFT JOIN orders ON users.id = orders.user_id\nGROUP BY users.name",
            "function calculateTotal(items) {\n    return items.reduce((sum, item) => sum + item.price, 0);\n}",
            "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"Hello, Go!\")\n}",
            "#include <stdio.h>\nint main() {\n    printf(\"Hello, C!\\n\");\n    return 0;\n}",
            "pub fn fibonacci(n: u32) -> u32 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2)\n    }\n}"
        ]
        
        # 生成指定数量的测试文本
        test_texts = []
        for i in range(count):
            base_idx = i % len(base_samples)
            # 添加一些变化使文本略有不同
            text = f"// Sample {i+1}\n{base_samples[base_idx]}\n// End of sample"
            test_texts.append(text)
        
        return test_texts
    
    async def single_request(self, texts: List[str]) -> Tuple[bool, float, str]:
        """执行单次API请求"""
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.config.api_url}/predict",
                json=texts,
                params={"k": 2, "threshold": 0.0}
            ) as response:
                latency = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    # 验证返回结果格式
                    if isinstance(result, list) and len(result) == len(texts):
                        return True, latency, "success"
                    else:
                        return False, latency, f"invalid_response_format"
                else:
                    error_text = await response.text()
                    return False, latency, f"http_{response.status}"
                    
        except asyncio.TimeoutError:
            latency = time.time() - start_time
            return False, latency, "timeout"
        except Exception as e:
            latency = time.time() - start_time
            return False, latency, f"exception_{type(e).__name__}"
    
    async def run_load_test(self, concurrency: int, batch_size: int) -> TestResult:
        """运行负载测试"""
        print(f"\n🧪 测试配置: 并发={concurrency}, 批次大小={batch_size}")
        
        # 准备测试数据
        test_texts = self.generate_test_texts(1000)
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        
        # 统计数据
        results = []
        error_types = {}
        start_test_time = time.time()
        
        async def worker():
            """工作协程"""
            while time.time() - start_test_time < self.config.test_duration:
                # 随机选择批次文本
                batch_start = (len(results) * batch_size) % (len(test_texts) - batch_size)
                batch_texts = test_texts[batch_start:batch_start + batch_size]
                
                async with semaphore:
                    success, latency, error_type = await self.single_request(batch_texts)
                    
                    results.append({
                        'success': success,
                        'latency': latency,
                        'error_type': error_type,
                        'batch_size': batch_size
                    })
                    
                    if not success:
                        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # 预热阶段
        print(f"  🔥 预热 {self.config.warmup_duration} 秒...")
        warmup_tasks = [worker() for _ in range(min(concurrency, 10))]
        await asyncio.sleep(self.config.warmup_duration)
        
        # 取消预热任务
        for task in warmup_tasks:
            task.cancel()
        
        # 清理预热结果
        results.clear()
        error_types.clear()
        
        # 正式测试阶段
        print(f"  ⚡ 正式测试 {self.config.test_duration} 秒...")
        start_test_time = time.time()
        
        # 启动工作协程
        workers = [worker() for _ in range(concurrency)]
        
        # 等待测试完成
        await asyncio.sleep(self.config.test_duration)
        
        # 停止所有工作协程
        for worker_task in workers:
            worker_task.cancel()
        
        # 等待所有协程完成
        await asyncio.gather(*workers, return_exceptions=True)
        
        actual_duration = time.time() - start_test_time
        
        # 统计结果
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        total_samples = sum(r['batch_size'] for r in results)
        
        if total_requests > 0:
            latencies = [r['latency'] for r in results if r['success']]
            avg_latency = statistics.mean(latencies) if latencies else 0
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0
            throughput_rps = successful_requests / actual_duration
            throughput_sps = sum(r['batch_size'] for r in results if r['success']) / actual_duration
            success_rate = successful_requests / total_requests
        else:
            avg_latency = p95_latency = p99_latency = 0
            throughput_rps = throughput_sps = success_rate = 0
        
        result = TestResult(
            concurrency=concurrency,
            batch_size=batch_size,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_samples=total_samples,
            test_duration=actual_duration,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput_rps=throughput_rps,
            throughput_sps=throughput_sps,
            success_rate=success_rate,
            error_types=error_types
        )
        
        # 输出结果
        print(f"  📊 结果: {successful_requests}/{total_requests} 成功, "
              f"{throughput_sps:.1f} samples/sec, "
              f"{avg_latency*1000:.1f}ms 平均延迟")
        
        return result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面性能测试"""
        print(f"🚀 开始API性能测试")
        print(f"目标URL: {self.config.api_url}")
        print(f"测试时长: {self.config.test_duration}秒/配置")
        
        # 健康检查
        try:
            async with self.session.get(f"{self.config.api_url}/health") as response:
                if response.status == 200:
                    health_info = await response.json()
                    print(f"✅ 服务健康检查通过: {health_info.get('status', 'unknown')}")
                else:
                    print(f"⚠️  服务健康检查异常: HTTP {response.status}")
        except Exception as e:
            print(f"❌ 无法连接到服务: {e}")
            return {}
        
        # 执行测试矩阵
        all_results = []
        
        for concurrency in self.config.concurrency_levels:
            for batch_size in self.config.batch_sizes:
                try:
                    result = await self.run_load_test(concurrency, batch_size)
                    all_results.append(result)
                    
                    # 短暂休息避免服务过载
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"❌ 测试失败 (并发={concurrency}, 批次={batch_size}): {e}")
        
        # 分析最优配置
        best_config = self.analyze_results(all_results)
        
        return {
            "test_results": all_results,
            "best_configuration": best_config,
            "test_summary": self.generate_summary(all_results)
        }
    
    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """分析测试结果，找出最优配置"""
        if not results:
            return {}
        
        # 过滤出成功率高的结果
        valid_results = [r for r in results if r.success_rate >= 0.95]
        
        if not valid_results:
            print("⚠️  没有找到成功率>=95%的配置")
            valid_results = results
        
        # 按吞吐量排序
        best_throughput = max(valid_results, key=lambda x: x.throughput_sps)
        
        # 按延迟排序（最低延迟）
        best_latency = min(valid_results, key=lambda x: x.avg_latency)
        
        # 综合评分（吞吐量 * 成功率 / 延迟）
        def score(r: TestResult) -> float:
            return (r.throughput_sps * r.success_rate) / max(r.avg_latency, 0.001)
        
        best_overall = max(valid_results, key=score)
        
        return {
            "best_throughput": {
                "concurrency": best_throughput.concurrency,
                "batch_size": best_throughput.batch_size,
                "throughput_sps": best_throughput.throughput_sps,
                "success_rate": best_throughput.success_rate,
                "avg_latency": best_throughput.avg_latency
            },
            "best_latency": {
                "concurrency": best_latency.concurrency,
                "batch_size": best_latency.batch_size,
                "throughput_sps": best_latency.throughput_sps,
                "success_rate": best_latency.success_rate,
                "avg_latency": best_latency.avg_latency
            },
            "best_overall": {
                "concurrency": best_overall.concurrency,
                "batch_size": best_overall.batch_size,
                "throughput_sps": best_overall.throughput_sps,
                "success_rate": best_overall.success_rate,
                "avg_latency": best_overall.avg_latency,
                "score": score(best_overall)
            }
        }
    
    def generate_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """生成测试摘要"""
        if not results:
            return {}
        
        max_throughput = max(results, key=lambda x: x.throughput_sps)
        min_latency = min(results, key=lambda x: x.avg_latency)
        
        return {
            "total_tests": len(results),
            "max_throughput": max_throughput.throughput_sps,
            "min_latency": min_latency.avg_latency,
            "avg_success_rate": statistics.mean([r.success_rate for r in results]),
            "total_samples_processed": sum([r.total_samples for r in results])
        }

def save_results(results: Dict[str, Any], output_file: str):
    """保存测试结果"""
    # 转换结果为可序列化格式
    serializable_results = {}
    
    for key, value in results.items():
        if key == "test_results":
            serializable_results[key] = [
                {
                    "concurrency": r.concurrency,
                    "batch_size": r.batch_size,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "total_samples": r.total_samples,
                    "test_duration": r.test_duration,
                    "avg_latency": r.avg_latency,
                    "p95_latency": r.p95_latency,
                    "p99_latency": r.p99_latency,
                    "throughput_rps": r.throughput_rps,
                    "throughput_sps": r.throughput_sps,
                    "success_rate": r.success_rate,
                    "error_types": r.error_types
                }
                for r in value
            ]
        else:
            serializable_results[key] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存到: {output_file}")

async def main():
    parser = argparse.ArgumentParser(description='FastText API性能测试工具')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API服务地址')
    parser.add_argument('--test-duration', type=int, default=30, help='每个配置的测试时长(秒)')
    parser.add_argument('--warmup-duration', type=int, default=5, help='预热时长(秒)')
    parser.add_argument('--output', default='api_performance_results.json', help='结果输出文件')
    parser.add_argument('--quick', action='store_true', help='快速测试模式(少量配置)')
    
    args = parser.parse_args()
    
    # 配置测试参数
    if args.quick:
        concurrency_levels = [10, 30, 50]
        batch_sizes = [50, 200]
    else:
        concurrency_levels = [5, 10, 20, 30, 50, 80, 100, 150]
        batch_sizes = [10, 50, 100, 200, 500, 1000]
    
    config = TestConfig(
        api_url=args.api_url,
        test_duration=args.test_duration,
        warmup_duration=args.warmup_duration,
        concurrency_levels=concurrency_levels,
        batch_sizes=batch_sizes
    )
    
    # 运行测试
    async with APIPerformanceTester(config) as tester:
        results = await tester.run_comprehensive_test()
        
        if results:
            # 输出最优配置建议
            best_config = results.get("best_configuration", {})
            if "best_overall" in best_config:
                best = best_config["best_overall"]
                print(f"\n🏆 推荐配置:")
                print(f"  并发数: {best['concurrency']}")
                print(f"  批次大小: {best['batch_size']}")
                print(f"  预期吞吐量: {best['throughput_sps']:.1f} samples/sec")
                print(f"  平均延迟: {best['avg_latency']*1000:.1f}ms")
                print(f"  成功率: {best['success_rate']*100:.1f}%")
                print(f"\n📋 CLI命令建议:")
                print(f"  --max-concurrent {best['concurrency']} --batch-size {best['batch_size']}")
            
            # 保存结果
            save_results(results, args.output)
        else:
            print("❌ 测试失败，无结果生成")

if __name__ == "__main__":
    asyncio.run(main())
