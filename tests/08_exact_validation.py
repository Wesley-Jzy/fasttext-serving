#!/usr/bin/env python3
"""
标签验证脚本：验证FastText推理框架的分类正确性
遍历整个验证集，对比预测标签与真实标签，输出P/R/F1指标
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Dict, Any

class LabelValidator:
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
                    return result
                else:
                    print(f"❌ 请求失败: {response.status}")
                    return [None for _ in batch_texts]
        except Exception as e:
            print(f"❌ 预测异常: {e}")
            return [None for _ in batch_texts]
    
    def parse_prediction(self, pred_result) -> str:
        """解析预测结果，返回标签"""
        if not pred_result:
            return "__label__0"  # 默认标签
            
        try:
            if isinstance(pred_result, list) and len(pred_result) >= 2:
                labels = pred_result[0]
                scores = pred_result[1]
                
                if labels and scores:
                    # 取得分最高的
                    max_idx = np.argmax(scores)
                    raw_label = labels[max_idx]
                    
                    # 转换标签格式
                    if raw_label.startswith('__label__'):
                        return raw_label
                    else:
                        return f"__label__{raw_label}"
        except Exception as e:
            print(f"❌ 解析预测结果失败: {e}")
        
        return "__label__0"  # 默认标签
    
    async def validate_all_labels(self, val_file: str, max_samples: int = None):
        """验证所有样本的标签分类正确性"""
        print(f"📖 读取验证集: {val_file}")
        
        try:
            df = pd.read_parquet(val_file)
        except Exception as e:
            print(f"❌ 读取验证集失败: {e}")
            return
        
        print(f"📊 验证集信息:")
        print(f"  总样本数: {len(df):,}")
        print(f"  标签分布: {df['FT_label'].value_counts().to_dict()}")
        
        # 限制样本数（用于测试）
        if max_samples:
            df = df.head(max_samples)
            print(f"🔬 测试模式: 只验证前 {len(df)} 个样本")
        
        # 获取内容和真实标签
        contents = df['clean_content'].tolist()
        true_labels = df['FT_label'].tolist()
        
        print(f"\n🚀 开始批量预测 {len(contents)} 个样本...")
        start_time = time.time()
        
        # 分批预测
        batch_size = 50
        all_predictions = []
        
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_predictions = await self.predict_batch(batch_contents, batch_size)
            
            # 转换预测结果
            batch_labels = [self.parse_prediction(pred) for pred in batch_predictions]
            all_predictions.extend(batch_labels)
            
            if (i // batch_size + 1) % 20 == 0:
                print(f"  已处理: {i + len(batch_contents)}/{len(contents)} 样本")
        
        total_time = time.time() - start_time
        print(f"✅ 预测完成，耗时: {total_time:.2f}秒")
        print(f"📊 吞吐量: {len(contents) / total_time:.1f} samples/sec")
        
        # 显示详细对比信息
        self.show_detailed_comparison(true_labels, all_predictions)
        
        # 计算评估指标
        self.calculate_metrics(true_labels, all_predictions)
        
        # 保存详细结果
        result_df = df.copy()
        result_df['predicted_label'] = all_predictions
        result_df['correct'] = result_df['FT_label'] == result_df['predicted_label']
        
        # 显示样本级别详细分析
        self.show_sample_analysis(result_df)
        
        # 保存结果
        output_file = f"label_validation_results_{int(time.time())}.parquet"
        result_df.to_parquet(output_file)
        print(f"💾 详细结果已保存: {output_file}")
        
        return {
            'total_samples': len(contents),
            'correct_predictions': sum(result_df['correct']),
            'accuracy': sum(result_df['correct']) / len(contents),
            'processing_time': total_time,
            'throughput': len(contents) / total_time
        }
    
    def show_detailed_comparison(self, true_labels: List[str], pred_labels: List[str]):
        """显示详细的标签对比信息"""
        print(f"\n🔍 标签格式对比:")
        print("=" * 80)
        
        # 显示真实标签格式
        true_unique = sorted(set(true_labels))
        print(f"算法验证集标签格式: {true_unique}")
        
        # 显示我们的预测格式  
        pred_unique = sorted(set(pred_labels))
        print(f"我们服务预测格式: {pred_unique}")
        
        # 显示标签分布对比
        from collections import Counter
        true_counts = Counter(true_labels)
        pred_counts = Counter(pred_labels)
        
        print(f"\n📊 标签分布对比:")
        print(f"{'标签':<15} {'算法验证集':<10} {'我们预测':<10} {'差异':<10}")
        print("-" * 50)
        for label in true_unique:
            true_count = true_counts.get(label, 0)
            pred_count = pred_counts.get(label, 0)
            diff = pred_count - true_count
            print(f"{label:<15} {true_count:<10} {pred_count:<10} {diff:+<10}")
        
        # 显示前几个样本的详细对比
        print(f"\n👀 前10个样本对比:")
        print(f"{'样本':<6} {'算法标签':<15} {'我们预测':<15} {'匹配':<6}")
        print("-" * 50)
        for i in range(min(10, len(true_labels))):
            match = "✅" if true_labels[i] == pred_labels[i] else "❌"
            print(f"{i+1:<6} {true_labels[i]:<15} {pred_labels[i]:<15} {match:<6}")
    
    def show_sample_analysis(self, result_df: pd.DataFrame):
        """显示样本级别的详细分析"""
        print(f"\n📈 错误样本详细分析:")
        print("=" * 80)
        
        # 错误类型统计
        error_samples = result_df[~result_df['correct']]
        if len(error_samples) > 0:
            print(f"总错误样本: {len(error_samples)}/{len(result_df)} ({len(error_samples)/len(result_df)*100:.1f}%)")
            
            error_stats = error_samples.groupby(['FT_label', 'predicted_label']).size()
            print(f"\n错误类型分布:")
            for (true_label, pred_label), count in error_stats.items():
                pct = count / len(error_samples) * 100
                print(f"  {true_label} → {pred_label}: {count} 个样本 ({pct:.1f}%)")
            
            # 显示几个具体的错误样本
            print(f"\n🔍 错误样本示例 (前5个):")
            error_examples = error_samples.head(5)
            for idx, row in error_examples.iterrows():
                content_preview = row['content'][:100] + "..." if len(row['content']) > 100 else row['content']
                print(f"\n样本 {idx}:")
                print(f"  内容: {content_preview}")
                print(f"  真实标签: {row['FT_label']}")
                print(f"  预测标签: {row['predicted_label']}")
                print(f"  内容长度: {len(row['content'])} 字符")
        else:
            print(f"🎉 所有样本都预测正确！")
    
    def calculate_metrics(self, true_labels: List[str], pred_labels: List[str]):
        """计算评估指标"""
        print(f"\n📊 标签验证结果:")
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
        
        # 问题诊断提示
        print(f"\n🔧 问题诊断提示:")
        if accuracy < 0.5:
            print(f"  准确率 < 50%，可能是标签映射问题")
        elif accuracy < 0.85:
            print(f"  准确率 < 85%，可能是模型或预处理问题")
        else:
            print(f"  准确率 > 85%，框架基本正常")
        
        print(f"\n💡 与算法同学讨论要点:")
        print(f"  1. 确认服务返回的 '0'/'1' 与 '__label__0'/'__label__1' 的对应关系")
        print(f"  2. 确认模型文件是否与验证集匹配")
        print(f"  3. 确认文本预处理是否一致")
        print(f"  4. 确认验证集的标签定义 (__label__0=低质量? __label__1=高质量?)")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='FastText标签验证脚本')
    parser.add_argument('--service-url', 
                        default='http://localhost:8000',
                        help='FastText服务URL')
    parser.add_argument('--val-file',
                        default='/mnt/project/yifan/data/code/the-stack-v2_fasttext_val_with_gt.parquet',
                        help='验证集文件')
    parser.add_argument('--max-samples', 
                        type=int,
                        help='最大验证样本数（不指定则验证全部）')
    
    args = parser.parse_args()
    
    async with LabelValidator(args.service_url) as validator:
        await validator.validate_all_labels(args.val_file, args.max_samples)

if __name__ == "__main__":
    asyncio.run(main())
