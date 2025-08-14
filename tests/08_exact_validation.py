#!/usr/bin/env python3
"""
精确验证脚本：逐个对比每个样本的预测结果
验证我们的推理框架与算法训练时的结果是否完全一致
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
from typing import List, Dict, Any

class ExactValidator:
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def predict_single(self, text: str) -> Dict[str, Any]:
        """单个文本预测"""
        try:
            async with self.session.post(
                f"{self.service_url}/predict",
                json=[text],
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result[0] if result else None
                else:
                    print(f"❌ 请求失败: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ 预测异常: {e}")
            return None
    
    def parse_prediction(self, pred_result) -> tuple:
        """解析预测结果，返回(标签, 分数)"""
        if not pred_result:
            return None, None
            
        try:
            if isinstance(pred_result, list) and len(pred_result) >= 2:
                labels = pred_result[0]
                scores = pred_result[1]
                
                if labels and scores:
                    # 取得分最高的
                    max_idx = np.argmax(scores)
                    raw_label = labels[max_idx]
                    score = scores[max_idx]
                    
                    # 转换标签格式
                    if raw_label.startswith('__label__'):
                        label = raw_label
                    else:
                        label = f"__label__{raw_label}"
                    
                    return label, score
        except Exception as e:
            print(f"❌ 解析预测结果失败: {e}")
        
        return None, None
    
    async def validate_samples(self, val_file: str, max_samples: int = 10):
        """逐个验证样本"""
        print(f"📖 读取验证集: {val_file}")
        
        try:
            df = pd.read_parquet(val_file)
        except Exception as e:
            print(f"❌ 读取验证集失败: {e}")
            return
        
        print(f"📊 验证集信息:")
        print(f"  总样本数: {len(df):,}")
        print(f"  包含分数列: {[col for col in df.columns if 'score' in col.lower()]}")
        
        # 限制样本数
        if max_samples:
            df = df.head(max_samples)
            print(f"🔬 验证前 {len(df)} 个样本")
        
        print(f"\n🚀 开始逐个验证...")
        print("=" * 100)
        
        exact_matches = 0
        label_matches = 0
        total_samples = 0
        score_diffs = []
        
        for idx, row in df.iterrows():
            total_samples += 1
            content = row['content']
            true_label = row['FT_label']
            
            # 检查是否有真实分数
            true_score = None
            if 'score' in row:
                true_score = row['score']
            elif 'score_fancreation' in row:
                true_score = row['score_fancreation']
            
            print(f"\n样本 {total_samples}/{len(df)}:")
            print(f"  内容长度: {len(content)} 字符")
            print(f"  真实标签: {true_label}")
            if true_score is not None:
                print(f"  真实分数: {true_score:.6f}")
            
            # 获取我们的预测
            pred_result = await self.predict_single(content)
            pred_label, pred_score = self.parse_prediction(pred_result)
            
            if pred_label is None:
                print(f"  ❌ 预测失败")
                continue
            
            print(f"  我们标签: {pred_label}")
            print(f"  我们分数: {pred_score:.6f}")
            
            # 标签对比
            label_match = (pred_label == true_label)
            if label_match:
                label_matches += 1
                print(f"  ✅ 标签匹配")
            else:
                print(f"  ❌ 标签不匹配: {pred_label} != {true_label}")
            
            # 分数对比
            score_match = False
            if true_score is not None:
                score_diff = abs(pred_score - true_score)
                score_diffs.append(score_diff)
                score_match = score_diff < 0.001  # 允许0.1%的误差
                
                print(f"  分数差异: {score_diff:.6f}")
                if score_match:
                    print(f"  ✅ 分数匹配")
                else:
                    print(f"  ❌ 分数差异过大")
            
            # 完全匹配
            if label_match and (true_score is None or score_match):
                exact_matches += 1
                print(f"  🎯 完全匹配")
            
            print("-" * 50)
        
        # 总结
        print(f"\n📊 验证总结:")
        print("=" * 50)
        print(f"验证样本数: {total_samples}")
        print(f"标签匹配: {label_matches}/{total_samples} ({label_matches/total_samples*100:.1f}%)")
        print(f"完全匹配: {exact_matches}/{total_samples} ({exact_matches/total_samples*100:.1f}%)")
        
        if score_diffs:
            print(f"\n分数差异统计:")
            print(f"  平均差异: {np.mean(score_diffs):.6f}")
            print(f"  最大差异: {np.max(score_diffs):.6f}")
            print(f"  分数匹配: {sum(1 for d in score_diffs if d < 0.001)}/{len(score_diffs)}")
        
        # 判断框架是否正确
        if label_matches / total_samples > 0.95:
            print(f"\n✅ 框架验证通过！标签匹配率 > 95%")
        else:
            print(f"\n❌ 框架验证失败！标签匹配率只有 {label_matches/total_samples*100:.1f}%")
            print(f"   可能原因：标签映射错误、模型文件不同、预处理差异")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='精确验证FastText推理框架')
    parser.add_argument('--service-url', 
                        default='http://localhost:8000',
                        help='FastText服务URL')
    parser.add_argument('--val-file',
                        default='/mnt/project/yifan/data/code/the-stack-v2_fasttext_val_with_gt.parquet',
                        help='验证集文件')
    parser.add_argument('--max-samples', 
                        type=int, 
                        default=10,
                        help='最大验证样本数')
    
    args = parser.parse_args()
    
    async with ExactValidator(args.service_url) as validator:
        await validator.validate_samples(args.val_file, args.max_samples)

if __name__ == "__main__":
    asyncio.run(main())
