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
    max_workers: int = None  # 最大工作进程数，None表示自动检测
    
    # 性能测试相关配置
    performance_test: bool = False  # 是否启用性能测试模式
    test_files_limit: int = 2  # 性能测试时最大文件数
    test_samples_per_file: int = 1000  # 每个文件最大样本数
    enable_monitoring: bool = False  # 是否启用系统监控
    monitoring_interval: int = 5  # 监控间隔(秒)

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
    
    # 数据量统计
    total_file_bytes: int = 0  # 总文件大小（字节）
    processed_content_bytes: int = 0  # 已处理的实际content字节数
    
    # 性能指标
    throughput_sps: float = 0  # samples per second
    throughput_gbps: float = 0  # GB per second (基于content)

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
        
        # 性能测试相关
        self.performance_results = []
        self.test_start_time = None
        self.system_monitor = None
        
        # 设置工作进程数
        if config.max_workers is None:
            # 根据CPU数量和内存情况智能设置
            cpu_count = os.cpu_count()
            if config.performance_test:
                # 性能测试时可以更激进
                self.max_workers = min(cpu_count // 2, 16)
            else:
                # 生产环境保守设置，避免内存问题
                self.max_workers = min(cpu_count // 4, 8)
        else:
            self.max_workers = config.max_workers
        
        # 设置日志
        self.setup_logging()
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载检查点（性能测试时跳过）
        if config.resume and not config.performance_test:
            self.load_checkpoint()
            
        # 启动系统监控
        if config.enable_monitoring:
            self.start_system_monitoring()
            
        # 初始化时扫描文件获取总数据量信息
        self._initial_scan_completed = False
    
    def start_system_monitoring(self):
        """启动系统监控"""
        try:
            import psutil
            self.system_monitor = {
                'enabled': True,
                'interval': self.config.monitoring_interval,
                'history': []
            }
            self.logger.info(f"✅ 启用系统监控 (间隔: {self.config.monitoring_interval}秒)")
        except ImportError:
            self.logger.warning("⚠️ 无法启用系统监控: 缺少psutil库")
            self.system_monitor = None
    
    def record_system_stats(self):
        """记录系统状态"""
        if not self.system_monitor or not self.system_monitor['enabled']:
            return
        
        try:
            import psutil
            
            stats = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'processed_samples': self.stats.processed_samples,
                'throughput_sps': self.stats.throughput_sps
            }
            
            self.system_monitor['history'].append(stats)
            
            # 只保留最近的监控数据
            if len(self.system_monitor['history']) > 1000:
                self.system_monitor['history'] = self.system_monitor['history'][-1000:]
                
        except Exception as e:
            self.logger.warning(f"⚠️ 系统监控记录失败: {e}")
    
    def scan_data_directory(self, data_dir: Path):
        """扫描数据目录，计算总文件数和总数据量"""
        if self._initial_scan_completed:
            return
            
        self.logger.info("🔍 扫描数据目录，计算总量...")
        
        parquet_files = list(data_dir.glob("*.parquet"))
        ready_files = []
        total_content_bytes = 0
        
        for file_path in parquet_files:
            # 跳过已处理的文件
            if str(file_path) in self.processed_files:
                continue
                
            # 检查文件完整性
            if not self.file_detector.is_file_ready(file_path):
                self.logger.debug(f"⏳ 文件未就绪，跳过: {file_path.name}")
                continue
                
            ready_files.append(file_path)
            
            # 直接使用文件大小，简单快速
            file_size_bytes = self.get_file_size_bytes(file_path)
            total_content_bytes += file_size_bytes
        
        # 性能测试模式下限制文件数
        if self.config.performance_test:
            ready_files = ready_files[:self.config.test_files_limit]
            # 重新计算限制后的总数据量
            total_content_bytes = 0
            for file_path in ready_files:
                file_size_bytes = self.get_file_size_bytes(file_path)
                total_content_bytes += file_size_bytes
        
        self.stats.total_files = len(ready_files)
        self.stats.total_file_bytes = total_content_bytes
        self._initial_scan_completed = True
        
        self.logger.info(f"📊 扫描完成: {len(ready_files)} 个就绪文件, 总计 {total_content_bytes / (1024**3):.2f} GB")
    
    def get_file_size_bytes(self, file_path: Path) -> int:
        """获取文件大小（字节）"""
        try:
            return file_path.stat().st_size
        except Exception as e:
            self.logger.warning(f"⚠️ 获取文件大小失败 {file_path.name}: {e}")
            return 0
    
    async def generate_performance_report(self):
        """生成性能测试报告"""
        if not self.config.performance_test or not self.test_start_time:
            return
        
        total_time = time.time() - self.test_start_time
        
        # 基础性能指标
        performance_data = {
            "test_configuration": {
                "max_concurrent": self.config.max_concurrent,
                "batch_size": self.config.batch_size,
                "max_workers": self.max_workers,
                "test_files_limit": self.config.test_files_limit,
                "test_samples_per_file": self.config.test_samples_per_file
            },
            "performance_results": {
                "total_processing_time": total_time,
                "total_files_processed": self.stats.processed_files,
                "total_samples_processed": self.stats.processed_samples,
                "successful_samples": self.stats.successful_samples,
                "failed_samples": self.stats.failed_samples,
                "success_rate": self.stats.successful_samples / max(self.stats.processed_samples, 1),
                "throughput_sps": self.stats.processed_samples / max(total_time, 1),
                "throughput_gbps": (self.stats.processed_content_bytes / (1024**3)) / max(total_time, 1),
                "cpu_cores": os.cpu_count(),
                "estimated_daily_capacity_samples": (self.stats.processed_samples / max(total_time, 1)) * 86400,  # 24小时
                "estimated_daily_capacity_gb": ((self.stats.processed_content_bytes / (1024**3)) / max(total_time, 1)) * 86400
            }
        }
        
        # 系统监控数据
        if self.system_monitor and self.system_monitor['history']:
            cpu_usage = [s['cpu_percent'] for s in self.system_monitor['history']]
            memory_usage = [s['memory_percent'] for s in self.system_monitor['history']]
            
            performance_data["system_monitoring"] = {
                "avg_cpu_percent": sum(cpu_usage) / len(cpu_usage),
                "max_cpu_percent": max(cpu_usage),
                "avg_memory_percent": sum(memory_usage) / len(memory_usage),
                "max_memory_percent": max(memory_usage),
                "monitoring_samples": len(self.system_monitor['history'])
            }
        
        # 估算不同数据量的处理时间
        if self.stats.processed_samples > 0:
            samples_per_sec = self.stats.processed_samples / total_time
            
            # 估算处理大规模数据的时间
            estimates = {
                "1M_samples": {"time_hours": 1_000_000 / samples_per_sec / 3600},
                "10M_samples": {"time_hours": 10_000_000 / samples_per_sec / 3600},
                "100M_samples": {"time_hours": 100_000_000 / samples_per_sec / 3600}
            }
            
            performance_data["scale_estimates"] = estimates
        
        # 保存报告
        report_file = Path(self.config.output_dir) / "performance_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
        
        # 输出摘要
        self.logger.info(f"\n" + "="*60)
        self.logger.info(f"🏆 性能测试报告")
        self.logger.info(f"="*60)
        self.logger.info(f"测试配置: 并发={self.config.max_concurrent}, 批次={self.config.batch_size}")
        self.logger.info(f"处理时间: {total_time:.1f} 秒")
        self.logger.info(f"处理样本: {self.stats.processed_samples:,}")
        self.logger.info(f"处理数据: {self.stats.processed_content_bytes / (1024**3):.2f} GB")
        self.logger.info(f"成功率: {performance_data['performance_results']['success_rate']:.1%}")
        self.logger.info(f"吞吐量: {performance_data['performance_results']['throughput_sps']:.1f} samples/sec")
        self.logger.info(f"🎯 内容处理速度: {performance_data['performance_results']['throughput_gbps']:.3f} GB/s")
        self.logger.info(f"预估日处理量: {performance_data['performance_results']['estimated_daily_capacity_samples']:,.0f} 样本 | {performance_data['performance_results']['estimated_daily_capacity_gb']:.1f} GB")
        
        if "system_monitoring" in performance_data:
            mon = performance_data["system_monitoring"]
            self.logger.info(f"平均CPU使用: {mon['avg_cpu_percent']:.1f}%")
            self.logger.info(f"平均内存使用: {mon['avg_memory_percent']:.1f}%")
        
        if "scale_estimates" in performance_data:
            est = performance_data["scale_estimates"]
            self.logger.info(f"\n📊 大规模处理预估:")
            self.logger.info(f"  100万样本: {est['1M_samples']['time_hours']:.1f} 小时")
            self.logger.info(f"  1000万样本: {est['10M_samples']['time_hours']:.1f} 小时")
            self.logger.info(f"  1亿样本: {est['100M_samples']['time_hours']:.1f} 小时")
        
        self.logger.info(f"\n💾 详细报告保存到: {report_file}")
        self.logger.info(f"="*60)
    
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
                # 使用流式分批读取，避免大文件内存爆炸
                import pyarrow.parquet as pq
                
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                
                if total_rows == 0:
                    self.logger.warning(f"⚠️ 空文件: {file_path.name}")
                    self.processed_files.add(str(file_path))
                    return True
                
                # 检查必需列
                schema = parquet_file.schema_arrow
                if 'content' not in schema.names:
                    self.logger.error(f"❌ 缺少content列: {file_path.name}")
                    return False
                
                # 计算动态缓存大小：基于当前并发配置
                # cache_size = 并发数 * 批次大小 * 缓冲倍数
                cache_size = self.config.max_concurrent * self.config.batch_size * 3
                
                # 性能测试模式下限制样本数
                max_samples = None
                if self.config.performance_test:
                    max_samples = self.config.test_samples_per_file
                    self.logger.info(f"🧪 性能测试模式: 限制样本数 {max_samples}")
                
                self.logger.info(f"📊 {file_path.name}: 总行数 {total_rows:,}, 缓存大小 {cache_size:,}")
                
                # 流式处理：分批读取和处理
                all_results = []
                file_content_bytes = 0
                processed_count = 0
                
                for batch in parquet_file.iter_batches(batch_size=cache_size):
                    # 转换为pandas DataFrame进行处理
                    df_chunk = batch.to_pandas()
                    
                    # 预处理这个chunk的数据
                    chunk_data = []
                    chunk_texts = []
                    
                    for idx, row in df_chunk.iterrows():
                        # 检查是否达到样本限制
                        if max_samples and processed_count >= max_samples:
                            break
                            
                        content = self.preprocess_content(row.get('content', ''))
                        if content:
                            chunk_data.append(row.to_dict())
                            chunk_texts.append(content)
                            file_content_bytes += len(content.encode('utf-8'))
                            processed_count += 1
                    
                    if not chunk_texts:
                        continue
                    
                    # 分批处理这个chunk的数据
                    chunk_results = []
                    for i in range(0, len(chunk_texts), self.config.batch_size):
                        batch_texts = chunk_texts[i:i + self.config.batch_size]
                        batch_data = chunk_data[i:i + self.config.batch_size]
                        
                        if not batch_texts:
                            continue
                        
                        # 异步推理（信号量控制在外层）
                        async with semaphore:
                            predictions = await self.predict_batch(batch_texts)
                        
                        # 后处理
                        batch_results = self.postprocess_results(batch_data, predictions)
                        chunk_results.extend(batch_results)
                        
                        # 更新统计
                        valid_count = sum(1 for r in batch_results if r.get('prediction_valid', False))
                        self.stats.successful_samples += valid_count
                        self.stats.failed_samples += len(batch_results) - valid_count
                    
                    all_results.extend(chunk_results)
                    
                    # 检查是否达到样本限制
                    if max_samples and processed_count >= max_samples:
                        break
                
                if not all_results:
                    self.logger.warning(f"⚠️ 无有效内容: {file_path.name}")
                    self.processed_files.add(str(file_path))
                    return True
                
                self.logger.info(f"📊 {file_path.name}: 流式处理完成 {processed_count} 个样本, {file_content_bytes / (1024**2):.1f} MB 内容")
                self.stats.total_samples += processed_count
                
                # 保存结果
                success = await self.save_file_results(file_path, all_results)
                
                if success:
                    # 更新统计和检查点
                    self.processed_files.add(str(file_path))
                    self.stats.processed_files += 1
                    self.stats.processed_content_bytes += file_content_bytes
                    
                    processing_time = time.time() - start_time
                    self.stats.processing_time += processing_time
                    
                    throughput = processed_count / processing_time
                    # 计算GB/s处理速度
                    content_gb = file_content_bytes / (1024**3)
                    throughput_gbps = content_gb / processing_time
                    self.logger.info(f"✅ 完成: {file_path.name} "
                                   f"({processed_count} 样本, {throughput:.1f} samples/sec, "
                                   f"{throughput_gbps:.3f} GB/s)")
                    
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
        elapsed_time = time.time() - self.stats.start_time
        
        # 文件进度
        if self.stats.total_files > 0:
            file_progress_pct = (self.stats.processed_files / self.stats.total_files) * 100
            file_progress_str = f"{self.stats.processed_files}/{self.stats.total_files} ({file_progress_pct:.1f}%)"
        else:
            file_progress_str = f"{self.stats.processed_files}"
        
        # 数据量进度
        processed_content_gb = self.stats.processed_content_bytes / (1024**3)
        if self.stats.total_file_bytes > 0:
            total_file_gb = self.stats.total_file_bytes / (1024**3)
            # 这里不做百分比，因为content字节数和文件大小不能直接比较
            data_progress_str = f"{processed_content_gb:.2f}GB内容 / {total_file_gb:.2f}GB文件"
        else:
            data_progress_str = f"{processed_content_gb:.2f}GB内容"
        
        # 性能指标
        if elapsed_time > 0:
            current_throughput = self.stats.processed_samples / elapsed_time
            current_throughput_gbps = processed_content_gb / elapsed_time
            self.stats.throughput_sps = current_throughput
            self.stats.throughput_gbps = current_throughput_gbps
        else:
            current_throughput = 0
            current_throughput_gbps = 0
        
        print(f"\r📊 进度: 文件 {file_progress_str} | "
              f"数据 {data_progress_str} | "
              f"{current_throughput:.1f} samples/sec | "
              f"{current_throughput_gbps:.3f} GB/s | "
              f"用时: {elapsed_time:.0f}s", end="", flush=True)
    
    async def run(self):
        """运行主处理流程"""
        mode = "性能测试" if self.config.performance_test else "生产处理"
        self.logger.info(f"🚀 开始处理The-Stack-v2数据 ({mode})")
        self.logger.info(f"数据目录: {self.config.data_dir}")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"API地址: {self.config.api_url}")
        self.logger.info(f"API并发数: {self.config.max_concurrent}")
        self.logger.info(f"批次大小: {self.config.batch_size}")
        self.logger.info(f"CPU核心数: {os.cpu_count()}")
        self.logger.info(f"工作进程数: {self.max_workers}")
        
        if self.config.performance_test:
            self.logger.info(f"🧪 性能测试配置:")
            self.logger.info(f"  最大文件数: {self.config.test_files_limit}")
            self.logger.info(f"  每文件样本数: {self.config.test_samples_per_file}")
            self.test_start_time = time.time()
        
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            self.logger.error(f"❌ 数据目录不存在: {data_dir}")
            return
        
        # 初始扫描数据目录
        self.scan_data_directory(data_dir)
        
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
            
            # 扫描所有可处理的文件
            all_parquet_files = list(data_dir.glob("*.parquet"))
            self.logger.info(f"🔍 发现 {len(all_parquet_files)} 个parquet文件")
            
            all_ready_files = []
            for file_path in all_parquet_files:
                is_ready = self.file_detector.is_file_ready(file_path)
                if is_ready:
                    all_ready_files.append(file_path)
                    self.logger.debug(f"✅ 就绪: {file_path.name}")
                else:
                    self.logger.info(f"⏳ 未就绪: {file_path.name}")
            
            self.logger.info(f"📊 其中 {len(all_ready_files)} 个文件就绪可处理")
            
            # 过滤已处理的文件
            if self.config.resume and not self.config.performance_test:
                files_to_process = [f for f in all_ready_files if str(f) not in self.processed_files]
                self.logger.info(f"📋 过滤后剩余 {len(files_to_process)} 个未处理文件")
            else:
                files_to_process = all_ready_files
            
            # 性能测试模式下限制文件数量
            if self.config.performance_test:
                files_to_process = files_to_process[:self.config.test_files_limit]
                self.logger.info(f"🧪 性能测试模式: 限制处理 {len(files_to_process)} 个文件")
            
            if not files_to_process:
                if self.config.performance_test:
                    self.logger.error("❌ 性能测试: 没有可处理的文件")
                    return
                else:
                    self.logger.info("⏳ 暂无可处理文件，进入监控模式...")
            else:
                self.logger.info(f"📁 找到 {len(files_to_process)} 个文件待处理")
                
                # 逐个处理文件（文件级串行）
                files_processed = 0
                for file_path in files_to_process:
                    if self.shutdown_event.is_set():
                        break
                    
                    # 更新进度显示
                    self.update_progress_display()
                    
                    # 记录系统状态
                    self.record_system_stats()
                    
                    # 处理文件
                    success = await self.process_file(file_path, semaphore)
                    
                    if success:
                        files_processed += 1
                        self.logger.info(f"✅ 完成文件 {files_processed}/{len(files_to_process)}: {file_path.name}")
                    else:
                        self.logger.error(f"❌ 文件处理失败，跳过: {file_path.name}")
                
                # 清理文件状态缓存
                self.file_detector.cleanup_states(data_dir)
                
                # 性能测试模式：处理完就退出
                if self.config.performance_test:
                    self.logger.info(f"🎉 性能测试完成！处理了 {files_processed} 个文件")
                    await self.generate_performance_report()
                    return
                
                self.logger.info("✅ 当前批次处理完成")
            
            # 生产模式：监控新文件（仅在非性能测试模式）
            if not self.config.performance_test:
                self.logger.info("🔄 进入增量监控模式，等待新文件...")
                while not self.shutdown_event.is_set():
                    await asyncio.sleep(60)  # 等待1分钟
                    
                    # 重新扫描新文件
                    current_ready_files = self.file_detector.scan_ready_files(data_dir)
                    new_files = [f for f in current_ready_files if str(f) not in self.processed_files]
                    
                    if new_files:
                        self.logger.info(f"📁 发现 {len(new_files)} 个新文件，开始处理...")
                        for file_path in new_files:
                            if self.shutdown_event.is_set():
                                break
                            
                            success = await self.process_file(file_path, semaphore)
                            if success:
                                self.logger.info(f"✅ 新文件处理完成: {file_path.name}")
                            else:
                                self.logger.error(f"❌ 新文件处理失败: {file_path.name}")
                    else:
                        self.logger.debug("⏳ 暂无新文件...")
        
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
    parser.add_argument('--max-workers', type=int,
                        help='最大工作进程数 (默认: 自动检测)')
    
    # 性能测试相关参数
    parser.add_argument('--performance-test', action='store_true',
                        help='启用性能测试模式')
    parser.add_argument('--test-files-limit', type=int, default=2,
                        help='性能测试时最大文件数 (默认: 2)')
    parser.add_argument('--test-samples-per-file', type=int, default=1000,
                        help='性能测试时每文件最大样本数 (默认: 1000)')
    parser.add_argument('--enable-monitoring', action='store_true',
                        help='启用系统资源监控')
    parser.add_argument('--monitoring-interval', type=int, default=5,
                        help='系统监控间隔(秒) (默认: 5)')
    
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
        log_level=args.log_level,
        max_workers=args.max_workers,
        performance_test=args.performance_test,
        test_files_limit=args.test_files_limit,
        test_samples_per_file=args.test_samples_per_file,
        enable_monitoring=args.enable_monitoring,
        monitoring_interval=args.monitoring_interval
    )
    
    # 运行处理器
    async with TheStackProcessor(config) as processor:
        await processor.run()

if __name__ == "__main__":
    asyncio.run(main())
