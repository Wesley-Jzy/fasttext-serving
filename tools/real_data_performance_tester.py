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
from dataclasses import dataclass, asdict
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

class RealDataPerformanceTester:
    """真实数据API性能测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session: aiohttp.ClientSession = None
        
    async def __aenter__(self):
        # 配置连接池
        connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=500,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=60)  # 增加超时时间
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def find_test_files(self) -> Tuple[Path, Path]:
        """找到最大和最小的parquet文件"""
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        parquet_files = list(data_dir.glob("*.parquet"))
        if len(parquet_files) < 2:
            raise ValueError(f"数据目录中parquet文件数量不足: {len(parquet_files)}")
        
        # 按文件大小排序
        files_with_size = [(f, f.stat().st_size) for f in parquet_files]
        files_with_size.sort(key=lambda x: x[1])
        
        smallest_file = files_with_size[0][0]
        largest_file = files_with_size[-1][0]
        
        print(f"📁 选中测试文件:")
        print(f"  最小文件: {smallest_file.name} ({files_with_size[0][1] / 1024**2:.1f} MB)")
        print(f"  最大文件: {largest_file.name} ({files_with_size[-1][1] / 1024**2:.1f} MB)")
        
        return smallest_file, largest_file
    
    def load_file_data(self, file_path: Path) -> Tuple[List[str], int, float]:
        """加载文件数据并预处理"""
        print(f"📖 加载文件: {file_path.name}")
        
        # 获取文件大小
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 ** 2)
        file_size_gb = file_size_bytes / (1024 ** 3)
        
        try:
            # 读取parquet文件
            df = pd.read_parquet(file_path)
            
            if 'content' not in df.columns:
                raise ValueError(f"文件缺少content列: {file_path.name}")
            
            # 预处理内容
            valid_texts = []
            for content in df['content']:
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    valid_texts.append(content.strip())
            
            print(f"  总样本数: {len(df):,}")
            print(f"  有效样本数: {len(valid_texts):,}")
            print(f"  文件大小: {file_size_mb:.1f} MB ({file_size_gb:.3f} GB)")
            
            return valid_texts, len(df), file_size_gb
            
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {file_path.name} - {e}")
    
    async def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        try:
            async with self.session.post(
                f"{self.config.api_url}/predict",
                json=texts,
                params={"k": 2, "threshold": 0.0}
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    
                    # 转换为标准格式
                    formatted_results = []
                    for result in results:
                        if isinstance(result, dict) and "labels" in result:
                            # 新格式: {"labels": [...], "scores": [...]}
                            formatted_results.append({
                                "labels": result["labels"],
                                "scores": result["scores"],
                                "prediction": result["labels"][0] if result["labels"] else "__label__0",
                                "confidence": result["scores"][0] if result["scores"] else 0.0
                            })
                        elif isinstance(result, (list, tuple)) and len(result) == 2:
                            # 旧格式: [labels, scores]
                            labels, scores = result
                            formatted_results.append({
                                "labels": labels,
                                "scores": scores,
                                "prediction": labels[0] if labels else "__label__0",
                                "confidence": scores[0] if scores else 0.0
                            })
                        else:
                            # 异常格式，使用默认值
                            formatted_results.append({
                                "labels": ["__label__0"],
                                "scores": [0.0],
                                "prediction": "__label__0",
                                "confidence": 0.0
                            })
                    
                    return formatted_results
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"API错误: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            # 返回默认结果，但记录错误
            return [{
                "labels": ["__label__0"],
                "scores": [0.0],
                "prediction": "__label__0",
                "confidence": 0.0,
                "error": str(e)
            } for _ in texts]
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """验证预测结果的有效性"""
        try:
            if "error" in prediction:
                return False, f"api_error_{prediction['error'][:50]}"
            
            labels = prediction.get("labels", [])
            scores = prediction.get("scores", [])
            
            # 基本格式检查
            if not isinstance(labels, list) or not isinstance(scores, list):
                return False, "invalid_format_not_list"
            
            if len(labels) != len(scores):
                return False, "length_mismatch"
            
            if len(labels) == 0:
                return False, "empty_prediction"
            
            # 检查标签格式
            valid_labels = {"__label__0", "__label__1"}
            for label in labels:
                if label not in valid_labels:
                    return False, f"invalid_label_{label}"
            
            # 检查是否包含预期的标签组合
            label_set = set(labels)
            if len(label_set) > 2:
                return False, f"too_many_unique_labels_{len(label_set)}"
            
            # 检查分数是否按降序排列
            if scores != sorted(scores, reverse=True):
                return False, "scores_not_descending"
            
            return True, None
            
        except Exception as e:
            return False, f"validation_exception_{str(e)}"
    
    async def test_file_with_config(self, file_path: Path, concurrency: int, batch_size: int) -> FileTestResult:
        """使用指定配置测试单个文件"""
        print(f"\n🧪 测试文件: {file_path.name}")
        print(f"   配置: 并发={concurrency}, 批次大小={batch_size}")
        
        # 加载数据
        texts, total_samples, file_size_gb = self.load_file_data(file_path)
        file_size_mb = file_size_gb * 1024
        
        if not texts:
            raise ValueError(f"文件中没有有效的文本内容: {file_path.name}")
        
        # 处理统计
        start_time = time.time()
        processed_samples = 0
        successful_samples = 0
        failed_samples = 0
        error_types = {}
        sample_errors = []
        latencies = []
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_batch(batch_texts: List[str]) -> Tuple[int, int, List[str], List[float]]:
            """处理单个批次"""
            async with semaphore:
                batch_start = time.time()
                predictions = await self.predict_batch(batch_texts)
                batch_latency = time.time() - batch_start
                
                batch_successful = 0
                batch_failed = 0
                batch_errors = []
                
                for prediction in predictions:
                    is_valid, error_reason = self.validate_prediction(prediction)
                    if is_valid:
                        batch_successful += 1
                    else:
                        batch_failed += 1
                        if error_reason:
                            error_types[error_reason] = error_types.get(error_reason, 0) + 1
                            if len(batch_errors) < 5:  # 只保留前几个错误样本
                                batch_errors.append(error_reason)
                
                return batch_successful, batch_failed, batch_errors, [batch_latency]
        
        # 分批处理
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tasks.append(process_batch(batch_texts))
        
        print(f"   开始处理 {len(tasks)} 个批次...")
        
        # 并发执行所有批次
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        # 汇总结果
        for result in results:
            if isinstance(result, Exception):
                print(f"   ❌ 批次处理异常: {result}")
                continue
            
            batch_successful, batch_failed, batch_errors, batch_latencies = result
            successful_samples += batch_successful
            failed_samples += batch_failed
            sample_errors.extend(batch_errors)
            latencies.extend(batch_latencies)
        
        processed_samples = successful_samples + failed_samples
        
        # 计算性能指标
        throughput_sps = processed_samples / processing_time if processing_time > 0 else 0
        throughput_gbs = file_size_gb / processing_time if processing_time > 0 else 0
        avg_latency = statistics.mean(latencies) if latencies else 0
        success_rate = successful_samples / processed_samples if processed_samples > 0 else 0
        
        print(f"   ✅ 完成: {processed_samples}/{total_samples} 样本")
        print(f"   📊 吞吐量: {throughput_sps:.1f} samples/sec, {throughput_gbs:.3f} GB/sec")
        print(f"   🎯 成功率: {success_rate:.1%}")
        
        return FileTestResult(
            file_name=file_path.name,
            file_size_mb=file_size_mb,
            file_size_gb=file_size_gb,
            total_samples=total_samples,
            processed_samples=processed_samples,
            successful_samples=successful_samples,
            failed_samples=failed_samples,
            concurrency=concurrency,
            batch_size=batch_size,
            processing_time=processing_time,
            throughput_sps=throughput_sps,
            throughput_gbs=throughput_gbs,
            avg_latency=avg_latency,
            success_rate=success_rate,
            error_types=error_types,
            sample_errors=sample_errors[:10]  # 只保留前10个错误
        )
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面的真实数据性能测试"""
        print(f"🚀 开始真实数据API性能测试")
        print(f"目标URL: {self.config.api_url}")
        print(f"数据目录: {self.config.data_dir}")
        
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
        
        # 选择测试文件
        try:
            smallest_file, largest_file = self.find_test_files()
        except Exception as e:
            print(f"❌ 选择测试文件失败: {e}")
            return {}
        
        # 测试结果
        all_results = []
        
        # 测试两个文件，每个文件用不同配置
        test_files = [
            ("smallest", smallest_file),
            ("largest", largest_file)
        ]
        
        for file_type, file_path in test_files:
            print(f"\n📂 开始测试 {file_type} 文件: {file_path.name}")
            
            for concurrency in self.config.concurrency_levels:
                for batch_size in self.config.batch_sizes:
                    try:
                        result = await self.test_file_with_config(file_path, concurrency, batch_size)
                        result_dict = asdict(result)
                        result_dict['file_type'] = file_type
                        all_results.append(result_dict)
                        
                        # 短暂休息避免服务过载
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        print(f"❌ 测试失败 ({file_type}, 并发={concurrency}, 批次={batch_size}): {e}")
        
        # 分析最佳配置
        best_config = self.analyze_results(all_results)
        
        return {
            "test_results": all_results,
            "best_configuration": best_config,
            "test_summary": self.generate_summary(all_results)
        }
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析测试结果，找出最优配置"""
        if not results:
            return {}
        
        # 按文件类型分组
        smallest_results = [r for r in results if r['file_type'] == 'smallest']
        largest_results = [r for r in results if r['file_type'] == 'largest']
        
        def find_best(file_results: List[Dict], criteria: str = "throughput_gbs") -> Dict:
            """找到最佳配置"""
            if not file_results:
                return {}
            
            # 过滤成功率高的结果
            valid_results = [r for r in file_results if r['success_rate'] >= 0.95]
            if not valid_results:
                valid_results = file_results
            
            # 按指定标准排序
            best_result = max(valid_results, key=lambda x: x[criteria])
            return {
                "file_name": best_result["file_name"],
                "concurrency": best_result["concurrency"],
                "batch_size": best_result["batch_size"],
                "throughput_sps": best_result["throughput_sps"],
                "throughput_gbs": best_result["throughput_gbs"],
                "success_rate": best_result["success_rate"],
                "avg_latency": best_result["avg_latency"]
            }
        
        return {
            "smallest_file_best": find_best(smallest_results),
            "largest_file_best": find_best(largest_results),
            "overall_best_throughput": find_best(results, "throughput_gbs"),
            "overall_best_sps": find_best(results, "throughput_sps")
        }
    
    def generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """生成测试摘要"""
        if not results:
            return {}
        
        total_processed = sum(r['processed_samples'] for r in results)
        total_successful = sum(r['successful_samples'] for r in results)
        total_time = sum(r['processing_time'] for r in results)
        total_data_gb = sum(r['file_size_gb'] for r in results)
        
        max_throughput_gbs = max(results, key=lambda x: x['throughput_gbs'])
        max_throughput_sps = max(results, key=lambda x: x['throughput_sps'])
        
        return {
            "total_tests": len(results),
            "total_samples_processed": total_processed,
            "total_successful_samples": total_successful,
            "overall_success_rate": total_successful / total_processed if total_processed > 0 else 0,
            "total_processing_time": total_time,
            "total_data_processed_gb": total_data_gb,
            "max_throughput_gbs": max_throughput_gbs['throughput_gbs'],
            "max_throughput_sps": max_throughput_sps['throughput_sps'],
            "best_config_for_gbs": {
                "concurrency": max_throughput_gbs['concurrency'],
                "batch_size": max_throughput_gbs['batch_size']
            },
            "best_config_for_sps": {
                "concurrency": max_throughput_sps['concurrency'],
                "batch_size": max_throughput_sps['batch_size']
            }
        }

def save_results(results: Dict[str, Any], output_file: str):
    """保存测试结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存到: {output_file}")

def print_summary(results: Dict[str, Any]):
    """打印测试摘要"""
    if not results:
        return
    
    summary = results.get("test_summary", {})
    best_config = results.get("best_configuration", {})
    
    print(f"\n📊 测试摘要:")
    print(f"="*60)
    print(f"总测试数: {summary.get('total_tests', 0)}")
    print(f"总样本数: {summary.get('total_samples_processed', 0):,}")
    print(f"成功率: {summary.get('overall_success_rate', 0):.1%}")
    print(f"总数据量: {summary.get('total_data_processed_gb', 0):.3f} GB")
    print(f"总处理时间: {summary.get('total_processing_time', 0):.1f} 秒")
    
    print(f"\n🏆 性能峰值:")
    print(f"最高数据吞吐量: {summary.get('max_throughput_gbs', 0):.3f} GB/sec")
    print(f"最高样本吞吐量: {summary.get('max_throughput_sps', 0):.1f} samples/sec")
    
    if "overall_best_throughput" in best_config:
        best = best_config["overall_best_throughput"]
        print(f"\n💡 推荐配置 (基于数据吞吐量):")
        print(f"  并发数: {best.get('concurrency', 0)}")
        print(f"  批次大小: {best.get('batch_size', 0)}")
        print(f"  预期性能: {best.get('throughput_gbs', 0):.3f} GB/sec")
        print(f"  预期样本率: {best.get('throughput_sps', 0):.1f} samples/sec")
        print(f"\n📋 CLI命令建议:")
        print(f"  --max-concurrent {best.get('concurrency', 50)} --batch-size {best.get('batch_size', 200)}")

async def main():
    parser = argparse.ArgumentParser(
        description='FastText API真实数据性能测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--api-url', required=True, help='API服务地址')
    parser.add_argument('--data-dir', required=True, help='The-Stack-v2数据目录')
    parser.add_argument('--output', default='real_data_performance_results.json', help='结果输出文件')
    parser.add_argument('--quick', action='store_true', help='快速测试模式(少量配置)')
    
    args = parser.parse_args()
    
    # 配置测试参数
    if args.quick:
        concurrency_levels = [20, 50]
        batch_sizes = [100, 200]
    else:
        concurrency_levels = [10, 20, 30, 50, 80, 100]
        batch_sizes = [50, 100, 200, 500]
    
    config = TestConfig(
        api_url=args.api_url,
        data_dir=args.data_dir,
        concurrency_levels=concurrency_levels,
        batch_sizes=batch_sizes
    )
    
    # 运行测试
    async with RealDataPerformanceTester(config) as tester:
        results = await tester.run_comprehensive_test()
        
        if results:
            # 保存结果
            save_results(results, args.output)
            
            # 打印摘要
            print_summary(results)
        else:
            print("❌ 测试失败，无结果生成")

if __name__ == "__main__":
    asyncio.run(main())
