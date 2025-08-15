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
    
    def parse_prediction(self, pred_result) -> str:
        """解析预测结果，返回标签"""
        if not pred_result:
            return "__label__0"  # 默认标签
            
        try:
            # 正确的API格式：{"labels": [...], "scores": [...]}
            if isinstance(pred_result, dict):
                labels = pred_result.get("labels", [])
                scores = pred_result.get("scores", [])
                
                if labels and scores:
                    # 取得分最高的（第一个就是最高分）
                    raw_label = labels[0]
                    
                    # 确保标签格式正确
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
        
        # 计算匹配情况
        correct_count = sum(1 for true, pred in zip(true_labels, all_predictions) if true == pred)
        accuracy = correct_count / len(true_labels)
        
        print(f"\n🎯 验证结果:")
        print(f"总样本数: {len(true_labels)}")
        print(f"匹配样本数: {correct_count}")
        print(f"不匹配样本数: {len(true_labels) - correct_count}")
        print(f"匹配率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 显示前20个样本的对比
        print(f"\n📋 前20个样本对比:")
        print(f"{'样本':<4} {'真实标签':<15} {'预测标签':<15} {'匹配':<4}")
        print("-" * 50)
        for i in range(min(20, len(true_labels))):
            match = "✅" if true_labels[i] == all_predictions[i] else "❌"
            print(f"{i+1:<4} {true_labels[i]:<15} {all_predictions[i]:<15} {match:<4}")
        
        # 如果有不匹配的，显示前几个
        mismatched = [(i, true, pred) for i, (true, pred) in enumerate(zip(true_labels, all_predictions)) if true != pred]
        if mismatched:
            print(f"\n❌ 前10个不匹配样本:")
            print(f"{'样本':<4} {'真实标签':<15} {'预测标签':<15}")
            print("-" * 40)
            for i, true, pred in mismatched[:10]:
                print(f"{i+1:<4} {true:<15} {pred:<15}")
        else:
            print(f"\n🎉 所有标签都匹配！")
        
        return {
            'total_samples': len(true_labels),
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'processing_time': total_time,
            'throughput': len(true_labels) / total_time
        }
    

    

    


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
