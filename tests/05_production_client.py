#!/usr/bin/env python3
"""
生产级数据清洗客户端
专门用于处理the-stack-v2数据的大规模FastText推理
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple
import argparse
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """处理配置"""
    data_dir: str
    output_dir: str
    service_url: str
    batch_size: int = 100
    max_concurrent: int = 10
    max_text_length: int = 100000  # 最大文本长度
    timeout: int = 120
    skip_corrupted: bool = True
    save_format: str = "parquet"  # parquet 或 json

class TheStackV2Processor:
    """The-Stack-v2 数据处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent * 2)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def discover_parquet_files(self) -> List[Path]:
        """发现所有parquet文件"""
        print(f"🔍 扫描数据目录: {self.config.data_dir}")
        
        data_path = Path(self.config.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.config.data_dir}")
        
        parquet_files = list(data_path.glob("*.parquet"))
        print(f"📁 发现 {len(parquet_files)} 个parquet文件")
        
        # 按文件名排序，确保处理顺序一致
        parquet_files.sort()
        
        return parquet_files
    
    def load_parquet_safe(self, file_path: Path) -> Optional[pd.DataFrame]:
        """安全加载parquet文件"""
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            # 检查必要的列
            if 'content' not in df.columns:
                print(f"⚠️ {file_path.name}: 缺少'content'列")
                return None
            
            print(f"✅ {file_path.name}: 成功加载 {len(df)} 行")
            return df
            
        except Exception as e:
            if self.config.skip_corrupted:
                print(f"⚠️ {file_path.name}: 跳过损坏文件 - {e}")
                return None
            else:
                raise e
    
    def preprocess_content(self, content: str) -> str:
        """预处理content内容"""
        if pd.isna(content):
            return ""
        
        content_str = str(content)
        
        # 截断过长的文本
        if len(content_str) > self.config.max_text_length:
            print(f"⚠️ 截断长文本: {len(content_str)} -> {self.config.max_text_length}")
            content_str = content_str[:self.config.max_text_length]
        
        return content_str
    
    async def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        url = f"{self.config.service_url}/predict"
        params = {"k": 2, "threshold": 0.0}
        
        try:
            async with self.session.post(url, json=texts, params=params) as response:
                if response.status == 200:
                    results = await response.json()
                    
                    # 转换格式 [(labels, scores)] -> [{"labels": [], "scores": [], ...}]
                    formatted_results = []
                    for labels, scores in results:
                        formatted_results.append({
                            "labels": labels,
                            "scores": scores,
                            "prediction": labels[0] if labels else "unknown",
                            "confidence": scores[0] if scores else 0.0
                        })
                    
                    return formatted_results
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            print(f"❌ 批量预测失败: {e}")
            # 返回默认结果
            return [
                {
                    "labels": ["error"],
                    "scores": [0.0],
                    "prediction": "error",
                    "confidence": 0.0
                } for _ in texts
            ]
    
    def postprocess_results(self, original_data: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """后处理结果"""
        results = []
        
        for original, prediction in zip(original_data, predictions):
            result = original.copy()
            result.update({
                "quality_prediction": prediction["prediction"],
                "quality_confidence": prediction["confidence"],
                "quality_labels": prediction["labels"],
                "quality_scores": prediction["scores"]
            })
            results.append(result)
        
        return results
    
    async def process_file(self, file_path: Path, semaphore: asyncio.Semaphore) -> Tuple[int, int]:
        """处理单个文件"""
        async with semaphore:
            print(f"\n📄 处理文件: {file_path.name}")
            
            # 加载数据
            df = self.load_parquet_safe(file_path)
            if df is None:
                return 0, 0
            
            if len(df) == 0:
                print(f"⚠️ {file_path.name}: 空文件")
                return 0, 0
            
            # 提取和预处理内容
            original_data = []
            texts_for_prediction = []
            
            for idx, row in df.iterrows():
                content = self.preprocess_content(row['content'])
                if content:
                    # 保存原始数据
                    row_dict = row.to_dict()
                    original_data.append(row_dict)
                    texts_for_prediction.append(content)
            
            if not texts_for_prediction:
                print(f"⚠️ {file_path.name}: 没有有效内容")
                return 0, 0
            
            print(f"📊 {file_path.name}: 准备处理 {len(texts_for_prediction)} 个样本")
            
            # 分批处理
            all_results = []
            batch_count = 0
            
            for i in range(0, len(texts_for_prediction), self.config.batch_size):
                batch_texts = texts_for_prediction[i:i + self.config.batch_size]
                batch_original = original_data[i:i + self.config.batch_size]
                
                batch_count += 1
                print(f"  批次 {batch_count}: {len(batch_texts)} 样本")
                
                # 预测
                predictions = await self.predict_batch(batch_texts)
                
                # 后处理
                batch_results = self.postprocess_results(batch_original, predictions)
                all_results.extend(batch_results)
            
            # 保存结果
            success_count = await self.save_results(file_path, all_results)
            
            print(f"✅ {file_path.name}: 完成 {success_count} 个样本")
            return success_count, len(texts_for_prediction) - success_count
    
    async def save_results(self, source_file: Path, results: List[Dict]) -> int:
        """保存处理结果"""
        if not results:
            return 0
        
        # 生成输出文件名
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = source_file.stem  # 不含扩展名
        
        if self.config.save_format == "parquet":
            output_file = output_dir / f"{base_name}_processed.parquet"
            try:
                df = pd.DataFrame(results)
                df.to_parquet(output_file, engine='pyarrow', index=False)
                print(f"💾 保存到: {output_file}")
                return len(results)
            except Exception as e:
                print(f"❌ 保存parquet失败: {e}")
                return 0
        
        elif self.config.save_format == "json":
            output_file = output_dir / f"{base_name}_processed.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 保存到: {output_file}")
                return len(results)
            except Exception as e:
                print(f"❌ 保存json失败: {e}")
                return 0
        
        else:
            print(f"❌ 不支持的保存格式: {self.config.save_format}")
            return 0
    
    async def process_all_files(self) -> Dict[str, Any]:
        """处理所有文件"""
        print("🚀 开始大规模数据处理")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # 发现文件
        parquet_files = self.discover_parquet_files()
        
        if not parquet_files:
            print("❌ 没有找到parquet文件")
            return {"status": "no_files"}
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # 并发处理文件
        print(f"⚡ 开始并发处理 {len(parquet_files)} 个文件")
        print(f"📊 并发数: {self.config.max_concurrent}")
        print(f"📦 批次大小: {self.config.batch_size}")
        
        tasks = [
            self.process_file(file_path, semaphore)
            for file_path in parquet_files
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        total_success = 0
        total_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ 文件 {parquet_files[i].name} 处理异常: {result}")
                total_errors += 1
            else:
                success, errors = result
                total_success += success
                total_errors += errors
        
        elapsed = time.time() - self.start_time
        throughput = total_success / elapsed if elapsed > 0 else 0
        
        # 输出统计
        print(f"\n📈 处理完成统计:")
        print(f"  总文件数: {len(parquet_files)}")
        print(f"  成功样本: {total_success:,}")
        print(f"  失败样本: {total_errors:,}")
        print(f"  总耗时: {elapsed:.1f} 秒")
        print(f"  吞吐量: {throughput:.0f} samples/sec")
        
        return {
            "status": "completed",
            "total_files": len(parquet_files),
            "successful_samples": total_success,
            "failed_samples": total_errors,
            "total_time": elapsed,
            "throughput": throughput
        }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="The-Stack-v2 数据清洗客户端")
    
    parser.add_argument("--data-dir", required=True,
                       help="数据目录路径")
    parser.add_argument("--output-dir", required=True,
                       help="输出目录路径")
    parser.add_argument("--service-url", required=True,
                       help="FastText服务URL")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="批次大小 (默认: 100)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="最大并发数 (默认: 10)")
    parser.add_argument("--max-text-length", type=int, default=100000,
                       help="最大文本长度 (默认: 100000)")
    parser.add_argument("--timeout", type=int, default=120,
                       help="请求超时时间 (默认: 120秒)")
    parser.add_argument("--save-format", choices=["parquet", "json"], default="parquet",
                       help="保存格式 (默认: parquet)")
    parser.add_argument("--no-skip-corrupted", action="store_true",
                       help="不跳过损坏的文件")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_arguments()
    
    config = ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        service_url=args.service_url,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        max_text_length=args.max_text_length,
        timeout=args.timeout,
        skip_corrupted=not args.no_skip_corrupted,
        save_format=args.save_format
    )
    
    print("🎯 生产级数据清洗客户端")
    print("=" * 50)
    print(f"数据目录: {config.data_dir}")
    print(f"输出目录: {config.output_dir}")
    print(f"服务地址: {config.service_url}")
    print(f"批次大小: {config.batch_size}")
    print(f"并发数: {config.max_concurrent}")
    print(f"保存格式: {config.save_format}")
    print()
    
    try:
        async with TheStackV2Processor(config) as processor:
            result = await processor.process_all_files()
            
            # 保存处理报告
            report_file = Path(config.output_dir) / "processing_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 处理报告已保存到: {report_file}")
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
