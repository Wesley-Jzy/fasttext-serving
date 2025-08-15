#!/usr/bin/env python3
"""
The-Stack-v2 大规模代码质量分类客户端
"""

import asyncio
import aiohttp
import pandas as pd
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import signal
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# 导入文件检测器
sys.path.append(str(Path(__file__).parent.parent))
from implementations.python.file_detector import IncrementalFileDetector

@dataclass
class ProcessingConfig:
    """处理配置"""
    data_dir: str
    output_dir: str
    api_url: str
    max_concurrent: int = 50
    batch_size: int = 200
    timeout: int = 30
    stability_window: int = 30  # 文件稳定性检测窗口
    resume: bool = True  # 断点续传
    save_format: str = "parquet"  # 输出格式: parquet 或 json
    log_level: str = "INFO"

@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    total_samples: int = 0
    processed_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    start_time: float = 0
    current_file: str = ""
    processing_time: float = 0
    throughput_sps: float = 0  # samples per second

class TheStackProcessor:
    """The-Stack-v2数据处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.session: Optional[aiohttp.ClientSession] = None
        self.file_detector = IncrementalFileDetector(config.stability_window)
        self.shutdown_event = asyncio.Event()
        self.checkpoint_file = Path(config.output_dir) / "processing_checkpoint.json"
        self.processed_files = set()
        
        # 设置日志
        self.setup_logging()
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载检查点
        if config.resume:
            self.load_checkpoint()
    
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 文件处理器
        log_file = Path(self.config.output_dir) / "processing.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 配置logger
        self.logger = logging.getLogger('TheStackProcessor')
        self.logger.setLevel(log_level)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def load_checkpoint(self):
        """加载处理检查点"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    self.processed_files = set(checkpoint.get('processed_files', []))
                    self.logger.info(f"📂 加载检查点: 已处理 {len(self.processed_files)} 个文件")
            except Exception as e:
                self.logger.warning(f"⚠️ 加载检查点失败: {e}")
    
    def save_checkpoint(self):
        """保存处理检查点"""
        try:
            checkpoint = {
                'processed_files': list(self.processed_files),
                'last_update': time.time(),
                'stats': asdict(self.stats)
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"❌ 保存检查点失败: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 配置HTTP会话
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent * 2,
            limit_per_host=self.config.max_concurrent,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        # 健康检查
        await self.health_check()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def health_check(self):
        """API健康检查"""
        try:
            async with self.session.get(f"{self.config.api_url}/health") as response:
                if response.status == 200:
                    health_info = await response.json()
                    self.logger.info(f"✅ API服务健康: {health_info.get('status', 'unknown')}")
                else:
                    self.logger.warning(f"⚠️ API健康检查异常: HTTP {response.status}")
        except Exception as e:
            self.logger.error(f"❌ 无法连接到API服务: {e}")
            raise
    
    def preprocess_content(self, content: str) -> Optional[str]:
        """预处理代码内容"""
        if not content or not isinstance(content, str):
            return None
        
        # 基本清理
        content = content.strip()
        
        # 长度检查
        if len(content) == 0:
            return None
        
        # 这里可以添加更多预处理逻辑
        return content
    
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
                    self.logger.error(f"API错误: HTTP {response.status} - {error_text}")
                    # 返回默认结果
                    return [self._get_default_prediction() for _ in texts]
                    
        except Exception as e:
            self.logger.error(f"预测请求异常: {e}")
            return [self._get_default_prediction() for _ in texts]
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """获取默认预测结果"""
        return {
            "labels": ["__label__0"],
            "scores": [0.0],
            "prediction": "__label__0",
            "confidence": 0.0
        }
    
    def postprocess_results(self, original_data: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """后处理结果，生成最终输出"""
        results = []
        
        for original, prediction in zip(original_data, predictions):
            # 验证预测结果的有效性
            is_valid, error_reason = self.validate_prediction(prediction)
            
            # 构建输出记录
            result = {
                # 保留关键原始字段
                "blob_id": original.get("blob_id"),
                "path": original.get("path"),
                "repo_name": original.get("repo_name"),
                "language": original.get("language"),
                "size": original.get("size"),
                "ext": original.get("ext"),
                
                # 添加质量分类结果
                "quality_labels": prediction["labels"],
                "quality_scores": prediction["scores"],
                "quality_prediction": prediction["prediction"],
                "quality_confidence": prediction["confidence"],
                
                # 数据有效性标记
                "prediction_valid": is_valid,
                "prediction_error": error_reason if not is_valid else None,
                
                # 处理元信息
                "content_length": len(str(original.get("content", ""))),
                "processed_at": pd.Timestamp.now().isoformat()
            }
            
            results.append(result)
        
        return results
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """验证预测结果的有效性"""
        try:
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
    
    async def process_file(self, file_path: Path, semaphore: asyncio.Semaphore) -> bool:
        """处理单个文件"""
        async with semaphore:
            start_time = time.time()
            
            # 检查是否已处理
            if self.config.resume and str(file_path) in self.processed_files:
                self.logger.info(f"⏭️ 跳过已处理文件: {file_path.name}")
                self.stats.skipped_files += 1
                return True
            
            self.stats.current_file = file_path.name
            self.logger.info(f"📄 开始处理: {file_path.name}")
            
            try:
                # 加载数据
                df = pd.read_parquet(file_path)
                
                if len(df) == 0:
                    self.logger.warning(f"⚠️ 空文件: {file_path.name}")
                    self.processed_files.add(str(file_path))
                    return True
                
                # 检查必需列
                if 'content' not in df.columns:
                    self.logger.error(f"❌ 缺少content列: {file_path.name}")
                    return False
                
                # 预处理数据
                valid_data = []
                valid_texts = []
                
                for idx, row in df.iterrows():
                    content = self.preprocess_content(row.get('content', ''))
                    if content:
                        valid_data.append(row.to_dict())
                        valid_texts.append(content)
                
                if not valid_texts:
                    self.logger.warning(f"⚠️ 无有效内容: {file_path.name}")
                    self.processed_files.add(str(file_path))
                    return True
                
                self.logger.info(f"📊 {file_path.name}: 处理 {len(valid_texts)} 个样本")
                self.stats.total_samples += len(valid_texts)
                
                # 分批处理
                all_results = []
                batch_count = 0
                
                for i in range(0, len(valid_texts), self.config.batch_size):
                    # 检查是否需要关闭
                    if self.shutdown_event.is_set():
                        self.logger.info("🛑 收到关闭信号，停止处理")
                        return False
                    
                    batch_texts = valid_texts[i:i + self.config.batch_size]
                    batch_data = valid_data[i:i + self.config.batch_size]
                    
                    batch_count += 1
                    self.logger.debug(f"  批次 {batch_count}: {len(batch_texts)} 样本")
                    
                    # API预测
                    predictions = await self.predict_batch(batch_texts)
                    
                    # 后处理
                    batch_results = self.postprocess_results(batch_data, predictions)
                    all_results.extend(batch_results)
                    
                    # 更新统计
                    self.stats.processed_samples += len(batch_texts)
                    
                    # 统计有效预测和无效预测
                    valid_count = 0
                    for result in batch_results:
                        if result.get("prediction_valid", True):
                            valid_count += 1
                        else:
                            # 记录无效预测
                            self.logger.warning(f"无效预测: {result.get('prediction_error', 'unknown_error')}")
                    
                    self.stats.successful_samples += valid_count
                    self.stats.failed_samples += len(batch_texts) - valid_count
                
                # 保存结果
                success = await self.save_file_results(file_path, all_results)
                
                if success:
                    # 更新统计和检查点
                    self.processed_files.add(str(file_path))
                    self.stats.processed_files += 1
                    
                    processing_time = time.time() - start_time
                    self.stats.processing_time += processing_time
                    
                    throughput = len(valid_texts) / processing_time
                    self.logger.info(f"✅ 完成: {file_path.name} "
                                   f"({len(valid_texts)} 样本, {throughput:.1f} samples/sec)")
                    
                    # 定期保存检查点
                    if self.stats.processed_files % 5 == 0:
                        self.save_checkpoint()
                    
                    return True
                else:
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ 处理文件失败: {file_path.name} - {e}")
                return False
    
    async def save_file_results(self, original_file: Path, results: List[Dict]) -> bool:
        """保存单个文件的处理结果"""
        try:
            # 生成输出文件名
            output_name = f"processed_{original_file.stem}.{self.config.save_format}"
            output_path = Path(self.config.output_dir) / output_name
            
            if self.config.save_format == "parquet":
                df = pd.DataFrame(results)
                df.to_parquet(output_path, index=False)
            else:  # json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"💾 保存结果: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 保存结果失败: {e}")
            return False
    
    def update_progress_display(self):
        """更新进度显示"""
        if self.stats.total_files > 0:
            file_progress = (self.stats.processed_files / self.stats.total_files) * 100
        else:
            file_progress = 0
        
        if self.stats.processing_time > 0:
            current_throughput = self.stats.processed_samples / self.stats.processing_time
            self.stats.throughput_sps = current_throughput
        else:
            current_throughput = 0
        
        elapsed_time = time.time() - self.stats.start_time
        
        print(f"\r📊 进度: {self.stats.processed_files}/{self.stats.total_files} 文件 "
              f"({file_progress:.1f}%) | "
              f"{self.stats.processed_samples} 样本 | "
              f"{current_throughput:.1f} samples/sec | "
              f"用时: {elapsed_time:.0f}s", end="", flush=True)
    
    async def run(self):
        """运行主处理流程"""
        self.logger.info(f"🚀 开始处理The-Stack-v2数据")
        self.logger.info(f"数据目录: {self.config.data_dir}")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"API地址: {self.config.api_url}")
        self.logger.info(f"并发数: {self.config.max_concurrent}")
        self.logger.info(f"批次大小: {self.config.batch_size}")
        
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            self.logger.error(f"❌ 数据目录不存在: {data_dir}")
            return
        
        self.stats.start_time = time.time()
        
        # 设置信号处理
        def signal_handler(signum, frame):
            self.logger.info(f"收到信号 {signum}，准备优雅关闭...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            while not self.shutdown_event.is_set():
                # 扫描可处理的文件
                ready_files = self.file_detector.scan_ready_files(data_dir)
                
                # 过滤已处理的文件
                if self.config.resume:
                    new_files = [f for f in ready_files if str(f) not in self.processed_files]
                else:
                    new_files = ready_files
                
                if not new_files:
                    self.logger.info("⏳ 暂无新文件，等待30秒后重新扫描...")
                    await asyncio.sleep(30)
                    continue
                
                self.stats.total_files += len(new_files)
                self.logger.info(f"📁 发现 {len(new_files)} 个新文件待处理")
                
                # 逐个处理文件（文件级串行）
                for file_path in new_files:
                    if self.shutdown_event.is_set():
                        break
                    
                    # 更新进度显示
                    self.update_progress_display()
                    
                    # 处理文件
                    success = await self.process_file(file_path, semaphore)
                    
                    if not success:
                        self.logger.error(f"❌ 文件处理失败，跳过: {file_path.name}")
                
                # 清理文件状态缓存
                self.file_detector.cleanup_states(data_dir)
                
                # 如果没有更多文件，等待新文件
                if not self.shutdown_event.is_set():
                    self.logger.info("✅ 当前批次处理完成，等待新文件...")
                    await asyncio.sleep(60)  # 等待1分钟再次扫描
        
        except Exception as e:
            self.logger.error(f"❌ 处理过程中发生异常: {e}")
        
        finally:
            # 最终保存检查点
            self.save_checkpoint()
            
            # 输出最终统计
            total_time = time.time() - self.stats.start_time
            print(f"\n\n📊 处理完成统计:")
            print(f"  总处理时间: {total_time:.1f} 秒")
            print(f"  处理文件数: {self.stats.processed_files}")
            print(f"  跳过文件数: {self.stats.skipped_files}")
            print(f"  处理样本数: {self.stats.processed_samples}")
            print(f"  成功样本数: {self.stats.successful_samples}")
            print(f"  失败样本数: {self.stats.failed_samples}")
            if self.stats.processed_samples > 0:
                success_rate = (self.stats.successful_samples / self.stats.processed_samples) * 100
                print(f"  成功率: {success_rate:.1f}%")
            if total_time > 0:
                overall_throughput = self.stats.processed_samples / total_time
                print(f"  平均吞吐量: {overall_throughput:.1f} samples/sec")

async def main():
    parser = argparse.ArgumentParser(
        description='The-Stack-v2 大规模代码质量分类处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python3 the_stack_processor.py \\
    --data-dir /path/to/the-stack-v2 \\
    --output-dir /path/to/results \\
    --api-url http://fasttext-api:8000

  # 高性能配置
  python3 the_stack_processor.py \\
    --data-dir /path/to/the-stack-v2 \\
    --output-dir /path/to/results \\
    --api-url http://fasttext-api:8000 \\
    --max-concurrent 80 \\
    --batch-size 200

  # 从断点继续处理
  python3 the_stack_processor.py \\
    --data-dir /path/to/the-stack-v2 \\
    --output-dir /path/to/results \\
    --api-url http://fasttext-api:8000 \\
    --resume
        """
    )
    
    parser.add_argument('--data-dir', required=True,
                        help='The-Stack-v2数据目录路径')
    parser.add_argument('--output-dir', required=True,
                        help='处理结果输出目录')
    parser.add_argument('--api-url', required=True,
                        help='FastText API服务地址')
    parser.add_argument('--max-concurrent', type=int, default=50,
                        help='最大并发请求数 (默认: 50)')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='批处理大小 (默认: 200)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='请求超时时间(秒) (默认: 30)')
    parser.add_argument('--stability-window', type=int, default=30,
                        help='文件稳定性检测窗口(秒) (默认: 30)')
    parser.add_argument('--resume', action='store_true',
                        help='从断点继续处理')
    parser.add_argument('--save-format', choices=['parquet', 'json'], default='parquet',
                        help='输出文件格式 (默认: parquet)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        api_url=args.api_url,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        timeout=args.timeout,
        stability_window=args.stability_window,
        resume=args.resume,
        save_format=args.save_format,
        log_level=args.log_level
    )
    
    # 运行处理器
    async with TheStackProcessor(config) as processor:
        await processor.run()

if __name__ == "__main__":
    asyncio.run(main())
