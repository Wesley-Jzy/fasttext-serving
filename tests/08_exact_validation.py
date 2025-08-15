#!/usr/bin/env python3
"""
简化标签验证脚本：只对比预测标签与真实标签是否一致
"""

import pandas as pd
import asyncio
import aiohttp
import time
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
    
    def parse_prediction(self, pred_result) -> tuple:
        """解析预测结果，返回(标签, 得分)"""
        if not pred_result:
            return ("__label__0", 0.0)  # 默认标签和得分
            
        try:
            # 正确的API格式：{"labels": [...], "scores": [...]}
            if isinstance(pred_result, dict):
                labels = pred_result.get("labels", [])
                scores = pred_result.get("scores", [])
                
                if labels and scores:
                    # 取得分最高的（第一个就是最高分）
                    raw_label = labels[0]
                    score = scores[0]
                    
                    # 确保标签格式正确
                    if raw_label.startswith('__label__'):
                        return (raw_label, score)
                    else:
                        return (f"__label__{raw_label}", score)
                        
        except Exception as e:
            print(f"❌ 解析预测结果失败: {e}")
        
        return ("__label__0", 0.0)  # 默认标签和得分
    
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
        
        # 记录所有样本的详细信息
        all_samples = []
        
        # 分批预测
        batch_size = 50
        
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_predictions = await self.predict_batch(batch_contents, batch_size)
            
            # 记录每个样本的详细信息
            for j, (content, true_label, pred_result) in enumerate(zip(batch_contents, true_labels[i:i+batch_size], batch_predictions)):
                pred_label, pred_score = self.parse_prediction(pred_result)
                
                sample_info = {
                    'index': i + j,
                    'content': content,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'pred_score': pred_score,
                    'matched': true_label == pred_label
                }
                all_samples.append(sample_info)
            
            if (i // batch_size + 1) % 20 == 0:
                print(f"  已处理: {i + len(batch_contents)}/{len(contents)} 样本")
        
        total_time = time.time() - start_time
        print(f"✅ 预测完成，耗时: {total_time:.2f}秒")
        print(f"📊 吞吐量: {len(contents) / total_time:.1f} samples/sec")
        
        # 统一分析所有样本数据
        self.analyze_results(all_samples, total_time)
        
        # 返回结果
        correct_count = sum(1 for sample in all_samples if sample['matched'])
        return {
            'total_samples': len(all_samples),
            'correct_predictions': correct_count,
            'accuracy': correct_count / len(all_samples),
            'processing_time': total_time,
            'throughput': len(all_samples) / total_time
        }
    

    

    


    def analyze_results(self, all_samples: List[Dict], total_time: float):
        """统一分析所有样本的结果"""
        from collections import Counter
        
        # 提取标签信息
        true_labels = [sample['true_label'] for sample in all_samples]
        pred_labels = [sample['pred_label'] for sample in all_samples]
        
        # 1. FT_label统计信息（真实标签分布）
        true_counts = Counter(true_labels)
        print(f"\n📊 FT_label统计信息（验证集真实标签）:")
        for label, count in sorted(true_counts.items()):
            pct = count / len(true_labels) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        # 2. 我们的预测统计信息
        pred_counts = Counter(pred_labels)
        print(f"\n📊 我们的预测统计信息:")
        for label, count in sorted(pred_counts.items()):
            pct = count / len(pred_labels) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        # 3. 匹配率
        correct_count = sum(1 for sample in all_samples if sample['matched'])
        accuracy = correct_count / len(all_samples)
        print(f"\n🎯 验证结果:")
        print(f"总样本数: {len(all_samples):,}")
        print(f"匹配样本数: {correct_count:,}")
        print(f"不匹配样本数: {len(all_samples) - correct_count:,}")
        print(f"匹配率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 4. 不匹配样本详细信息（包含分数）
        mismatched_samples = [sample for sample in all_samples if not sample['matched']]
        
        if mismatched_samples:
            print(f"\n❌ 不匹配样本详情（前15个）:")
            print(f"{'样本':<6} {'真实标签':<15} {'预测标签':<15} {'预测分数':<10} {'内容预览':<30}")
            print("-" * 80)
            
            for sample in mismatched_samples[:15]:
                content_preview = sample['content'][:30].replace('\n', ' ') + "..." if len(sample['content']) > 30 else sample['content'].replace('\n', ' ')
                print(f"{sample['index']+1:<6} {sample['true_label']:<15} {sample['pred_label']:<15} {sample['pred_score']:<10.4f} {content_preview:<30}")
            
            if len(mismatched_samples) > 15:
                print(f"\n... 还有 {len(mismatched_samples) - 15} 个不匹配样本")
            
            # 分析不匹配的类型分布
            mismatch_types = Counter([(sample['true_label'], sample['pred_label']) for sample in mismatched_samples])
            print(f"\n📈 不匹配类型分布:")
            for (true_label, pred_label), count in mismatch_types.most_common():
                pct = count / len(mismatched_samples) * 100
                print(f"  {true_label} → {pred_label}: {count} 个样本 ({pct:.1f}%)")
        else:
            print(f"\n🎉 所有标签都匹配！")

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
