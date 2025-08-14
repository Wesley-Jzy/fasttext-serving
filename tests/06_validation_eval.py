#!/usr/bin/env python3
"""
FastText服务验证集评估脚本
对比我们的服务预测结果与真实标签，计算性能指标
"""

import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Dict, Any
import argparse

class FastTextValidator:
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def predict_batch(self, texts: List[str], batch_size: int = 50) -> List[Dict[str, Any]]:
        """批量预测文本"""
        if not texts:
            return []
        
        # 限制批次大小避免超时
        batch_texts = texts[:batch_size]
        
        try:
            async with self.session.post(
                f"{self.service_url}/predict",
                json=batch_texts,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # 调试：打印第一个批次的返回格式
                    if hasattr(self, '_first_batch_logged') is False:
                        print(f"🔍 服务返回格式示例: {result[:2]}")
                        self._first_batch_logged = True
                    return result
                else:
                    print(f"❌ 请求失败: {response.status}")
                    return [{"labels": ["__label__0"], "scores": [0.5]} for _ in batch_texts]
        except Exception as e:
            print(f"❌ 预测异常: {e}")
            return [{"labels": ["__label__0"], "scores": [0.5]} for _ in batch_texts]
    
    def convert_prediction(self, pred_result) -> str:
        """将预测结果转换为标签"""
        # 处理不同的返回格式
        if isinstance(pred_result, dict):
            labels = pred_result.get("labels", ["__label__0"])
            scores = pred_result.get("scores", [0.5])
        elif isinstance(pred_result, list) and len(pred_result) >= 2:
            # 如果是列表格式 [labels, scores]
            labels = pred_result[0] if len(pred_result) > 0 else ["__label__0"]
            scores = pred_result[1] if len(pred_result) > 1 else [0.5]
        else:
            print(f"⚠️ 未知预测格式: {pred_result}")
            return "__label__0"
        
        if not labels or not scores:
            return "__label__0"
        
        # 取得分最高的标签
        max_idx = np.argmax(scores)
        raw_label = labels[max_idx]
        
        # 转换为FastText格式
        if raw_label.startswith('__label__'):
            return raw_label
        else:
            # 如果是 '0', '1' 格式，转换为 '__label__0', '__label__1'
            return f"__label__{raw_label}"
    
    async def evaluate_validation_set(self, val_file: str, max_samples: int = None):
        """评估验证集"""
        print(f"📖 读取验证集: {val_file}")
        
        # 读取验证集
        try:
            df = pd.read_parquet(val_file)
        except Exception as e:
            print(f"❌ 读取验证集失败: {e}")
            return
        
        print(f"📊 数据概览:")
        print(f"  总样本数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        
        # 检查必要列是否存在
        required_cols = ['content', 'label', 'FT_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少必要列: {missing_cols}")
            return
        
        # 样本统计
        print(f"📈 标签分布:")
        print(df['label'].value_counts())
        print(f"📈 FastText标签分布:")
        print(df['FT_label'].value_counts())
        
        # 限制样本数量用于测试
        if max_samples:
            if hasattr(self, 'random_sampling') and self.random_sampling:
                df = df.sample(n=max_samples, random_state=42)
                print(f"🔬 测试模式: 随机抽样 {len(df)} 个样本")
            else:
                df = df.head(max_samples)
                print(f"🔬 测试模式: 只处理前 {len(df)} 个样本")
        
        # 获取内容和真实标签
        contents = df['content'].tolist()
        true_labels = df['FT_label'].tolist()  # 使用FastText格式的标签
        
        print(f"🚀 开始预测 {len(contents)} 个样本...")
        start_time = time.time()
        
        # 分批预测
        batch_size = 50
        all_predictions = []
        
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_predictions = await self.predict_batch(batch_contents, batch_size)
            
            # 转换预测结果
            batch_labels = [self.convert_prediction(pred) for pred in batch_predictions]
            all_predictions.extend(batch_labels)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  已处理: {i + len(batch_contents)}/{len(contents)} 样本")
        
        total_time = time.time() - start_time
        print(f"✅ 预测完成，耗时: {total_time:.2f}秒")
        print(f"📊 吞吐量: {len(contents) / total_time:.1f} samples/sec")
        
        # 计算评估指标
        self.calculate_metrics(true_labels, all_predictions)
        
        # 保存结果
        result_df = df.copy()
        result_df['predicted_label'] = all_predictions
        result_df['correct'] = result_df['FT_label'] == result_df['predicted_label']
        
        output_file = f"validation_results_{int(time.time())}.parquet"
        result_df.to_parquet(output_file)
        print(f"💾 结果已保存: {output_file}")
        
        return {
            'total_samples': len(contents),
            'predictions': all_predictions,
            'true_labels': true_labels,
            'processing_time': total_time,
            'throughput': len(contents) / total_time
        }
    
    def calculate_metrics(self, true_labels: List[str], pred_labels: List[str]):
        """计算评估指标"""
        print(f"\n📊 评估结果:")
        print("=" * 80)
        
        # 准确率
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"整体准确率: {accuracy:.4f}")
        
        # 分类报告
        report = classification_report(
            true_labels, 
            pred_labels, 
            target_names=['__label__0', '__label__1'],
            digits=4
        )
        print(f"\n详细分类报告:")
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        print(f"\n混淆矩阵:")
        print(f"             预测")
        print(f"真实    __label__0  __label__1")
        print(f"__label__0    {cm[0,0]:8d}    {cm[0,1]:8d}")
        print(f"__label__1    {cm[1,0]:8d}    {cm[1,1]:8d}")
        
        # 与基准对比
        print(f"\n🎯 与基准性能对比:")
        print(f"基准 __label__0: P=0.9902, R=0.8902, F1=0.9375")
        print(f"基准 __label__1: P=0.6466, R=0.9579, F1=0.7721")
        print(f"基准整体: P=0.9023, R=0.9018, F1=0.9021")

async def main():
    parser = argparse.ArgumentParser(description='FastText服务验证集评估')
    parser.add_argument('--val-file', 
                        default='/mnt/project/yifan/data/code/the-stack-v2_fasttext_val_with_gt.parquet',
                        help='验证集文件路径')
    parser.add_argument('--service-url', 
                        default='http://localhost:8000',
                        help='FastText服务URL')
    parser.add_argument('--max-samples', 
                        type=int,
                        help='最大样本数（测试用）')
    
    args = parser.parse_args()
    
    async with FastTextValidator(args.service_url) as validator:
        await validator.evaluate_validation_set(args.val_file, args.max_samples)

if __name__ == "__main__":
    asyncio.run(main())
