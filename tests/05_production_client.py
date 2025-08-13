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
    resume_from_checkpoint: bool = True  # 是否从断点继续

class TheStackV2Processor:
    """The-Stack-v2 数据处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.progress_file = Path(config.output_dir) / "progress.json"
        
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
        
        # 按文件大小排序：小文件优先，便于快速验证和负载均衡
        parquet_files.sort(key=lambda f: f.stat().st_size)
        print(f"📊 按文件大小排序（小到大）")
        
        return parquet_files
    
    def load_progress(self) -> Dict[str, Any]:
        """加载处理进度"""
        if not self.progress_file.exists():
            return {"completed_files": [], "start_time": None}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"📂 发现断点文件，已完成 {len(progress.get('completed_files', []))} 个文件")
            return progress
        except Exception as e:
            print(f"⚠️ 断点文件损坏，从头开始: {e}")
            return {"completed_files": [], "start_time": None}
    
    def save_progress(self, completed_files: List[str], stats: Dict[str, Any] = None):
        """保存处理进度"""
        try:
            # 确保输出目录存在
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            progress = {
                "completed_files": completed_files,
                "start_time": pd.Timestamp.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "last_update": pd.Timestamp.now().isoformat(),
                "total_processed": self.processed_count,
                "total_errors": self.error_count,
                "stats": stats or {}
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ 保存进度失败: {e}")
    
    def filter_remaining_files(self, all_files: List[Path], test_mode: bool = False) -> List[Path]:
        """过滤出剩余未处理的文件"""
        # 测试模式：选择已知的好文件进行测试
        if test_mode:
            # 寻找之前测试过的好文件
            known_good_files = [
                "the-stack-v2-dedup-train-00516-of-01078.parquet",
                "the-stack-v2-dedup-train-00341-of-01078.parquet", 
                "the-stack-v2-dedup-train-00195-of-01078.parquet"
            ]
            
            test_files = []
            for file_path in all_files:
                if file_path.name in known_good_files:
                    test_files.append(file_path)
                    if len(test_files) >= 2:  # 只要2个文件测试
                        break
            
            if not test_files:
                # 如果找不到已知好文件，选择中等大小的文件（避免最小的损坏文件）
                mid_point = len(all_files) // 2
                test_files = all_files[mid_point:mid_point+2]
                print(f"🧪 测试模式：未找到已知好文件，选择中等大小的文件")
            else:
                print(f"🧪 测试模式：选择已知的好文件进行测试")
            
            print(f"📋 测试文件: {[f.name for f in test_files]}")
            return test_files
        
        if not self.config.resume_from_checkpoint:
            return all_files
        
        progress = self.load_progress()
        completed_files = set(progress.get("completed_files", []))
        
        remaining_files = [f for f in all_files if f.name not in completed_files]
        
        if len(remaining_files) < len(all_files):
            skipped_count = len(all_files) - len(remaining_files)
            print(f"🔄 断点继续：跳过已完成的 {skipped_count} 个文件，剩余 {len(remaining_files)} 个")
        
        return remaining_files
    
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
        
        # 可选：保留原始长度用于后续分析
        # 不进行截断，让模型处理完整内容
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
            # 检查预测结果的有效性
            is_valid, invalid_reason = self.validate_prediction(prediction)
            
            # 构建输出记录：保留关键字段 + 预测结果
            result = {
                # 源文件关联字段（用于后续匹配）
                "blob_id": original.get("blob_id"),
                "path": original.get("path"),
                "repo_name": original.get("repo_name"),
                "language": original.get("language"),
                "filename": original.get("filename"),
                "length_bytes": original.get("length_bytes"),
                
                # 预测结果字段
                "quality_labels": prediction["labels"],
                "quality_scores": prediction["scores"], 
                "primary_prediction": prediction["prediction"],  # 主要预测类别
                "primary_confidence": prediction["confidence"],  # 主要置信度
                
                # 数据有效性字段
                "valid": is_valid,
                "invalid_reason": invalid_reason if not is_valid else None,
                
                # 处理元信息
                "content_length": len(str(original.get("content", ""))),
                "processed_at": pd.Timestamp.now().isoformat()
            }
            
            results.append(result)
        
        return results
    
    def validate_prediction(self, prediction: Dict) -> Tuple[bool, Optional[str]]:
        """验证预测结果的有效性"""
        try:
            labels = prediction.get("labels", [])
            scores = prediction.get("scores", [])
            
            # 检查标签数量是否正确（应该是2个：'0'和'1'）
            if len(labels) != 2:
                return False, f"unexpected_label_count:{len(labels)}"
            
            # 检查分数数量是否匹配
            if len(scores) != len(labels):
                return False, f"label_score_mismatch:{len(labels)}vs{len(scores)}"
            
            # 检查是否有预期的标签
            expected_labels = {"0", "1"}
            if set(labels) != expected_labels:
                return False, f"unexpected_labels:{labels}"
            
            # 检查分数是否合理
            if not all(0 <= score <= 1 for score in scores):
                return False, f"invalid_scores:{scores}"
            
            return True, None
            
        except Exception as e:
            return False, f"validation_error:{str(e)}"
    
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
        
        # 统计处理结果
        total_count = len(results)
        valid_count = sum(1 for r in results if r.get("valid", True))
        invalid_count = total_count - valid_count
        
        # 生成统计信息
        stats = {
            "source_file": source_file.name,
            "total_samples": total_count,
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "invalid_rate": invalid_count / total_count if total_count > 0 else 0,
            "processing_time": pd.Timestamp.now().isoformat()
        }
        
        if self.config.save_format == "parquet":
            # 主要结果文件
            output_file = output_dir / f"{base_name}_quality_predictions.parquet"
            try:
                df = pd.DataFrame(results)
                df.to_parquet(output_file, engine='pyarrow', index=False)
                print(f"💾 预测结果: {output_file}")
                
                # 可选：如果需要快速查看统计，取消注释下面几行
                # stats_file = output_dir / f"{base_name}_stats.json"
                # with open(stats_file, 'w', encoding='utf-8') as f:
                #     json.dump(stats, f, indent=2, ensure_ascii=False)
                
                print(f"📊 有效样本: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                
                return valid_count
            except Exception as e:
                print(f"❌ 保存parquet失败: {e}")
                return 0
        
        elif self.config.save_format == "json":
            output_file = output_dir / f"{base_name}_quality_predictions.json"
            try:
                output_data = {
                    "stats": stats,
                    "predictions": results
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"💾 保存到: {output_file}")
                return valid_count
            except Exception as e:
                print(f"❌ 保存json失败: {e}")
                return 0
        
        else:
            print(f"❌ 不支持的保存格式: {self.config.save_format}")
            return 0
    
    async def process_all_files(self, test_mode: bool = False) -> Dict[str, Any]:
        """处理所有文件"""
        print("🚀 开始大规模数据处理")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # 发现文件
        all_files = self.discover_parquet_files()
        
        if not all_files:
            print("❌ 没有找到parquet文件")
            return {"status": "no_files"}
        
        # 过滤出剩余未处理的文件
        remaining_files = self.filter_remaining_files(all_files, test_mode)
        
        if not remaining_files:
            print("✅ 所有文件已处理完成！")
            return {"status": "already_completed"}
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # 处理文件
        print(f"⚡ 开始处理 {len(remaining_files)} 个文件（共{len(all_files)}个）")
        print(f"📊 并发数: {self.config.max_concurrent}")
        print(f"📦 批次大小: {self.config.batch_size}")
        
        completed_files = []
        total_success = 0
        total_errors = 0
        
        # 逐个处理文件以支持断点保存
        for i, file_path in enumerate(remaining_files):
            print(f"\n📊 进度: {i+1}/{len(remaining_files)} - {file_path.name}")
            
            try:
                result = await self.process_file(file_path, semaphore)
                if isinstance(result, Exception):
                    print(f"❌ 文件处理异常: {result}")
                    total_errors += 1
                else:
                    success, errors = result
                    total_success += success
                    total_errors += errors
                    
                    # 标记为已完成
                    completed_files.append(file_path.name)
                    
                    # 保存进度
                    self.save_progress(completed_files, {
                        "current_file": i + 1,
                        "total_files": len(remaining_files),
                        "success_samples": total_success,
                        "error_samples": total_errors
                    })
                    
                    print(f"✅ 已完成: {len(completed_files)}/{len(remaining_files)}")
                    
            except Exception as e:
                print(f"❌ 处理文件 {file_path.name} 时异常: {e}")
                total_errors += 1
                # 继续处理下一个文件
                continue
        
        elapsed = time.time() - self.start_time
        throughput = total_success / elapsed if elapsed > 0 else 0
        
        # 输出最终统计
        print(f"\n📈 处理完成统计:")
        print(f"  处理文件数: {len(completed_files)}/{len(remaining_files)}")
        print(f"  成功样本: {total_success:,}")
        print(f"  失败样本: {total_errors:,}")
        print(f"  总耗时: {elapsed:.1f} 秒")
        print(f"  吞吐量: {throughput:.0f} samples/sec")
        
        return {
            "status": "completed",
            "processed_files": len(completed_files),
            "total_files": len(all_files),
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
    parser.add_argument("--no-resume", action="store_true",
                       help="不从断点继续，重新开始处理")
    parser.add_argument("--test-mode", action="store_true", 
                       help="测试模式：只处理前2个最小文件")
    
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
        save_format=args.save_format,
        resume_from_checkpoint=not args.no_resume
    )
    
    print("🎯 生产级数据清洗客户端")
    print("=" * 50)
    print(f"数据目录: {config.data_dir}")
    print(f"输出目录: {config.output_dir}")
    print(f"服务地址: {config.service_url}")
    print(f"批次大小: {config.batch_size}")
    print(f"并发数: {config.max_concurrent}")
    print(f"保存格式: {config.save_format}")
    print(f"断点继续: {config.resume_from_checkpoint}")
    if args.test_mode:
        print("🧪 测试模式: 只处理前2个最小文件")
    print()
    
    try:
        async with TheStackV2Processor(config) as processor:
            result = await processor.process_all_files(test_mode=args.test_mode)
            
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
