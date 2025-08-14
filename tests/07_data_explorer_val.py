#!/usr/bin/env python3
"""
验证集数据结构探索脚本
了解验证集的数据格式和分布
"""

import pandas as pd
import numpy as np

def explore_validation_data():
    """探索验证集数据结构"""
    val_file = '/mnt/project/yifan/data/code/the-stack-v2_fasttext_val_with_gt.parquet'
    
    print(f"🔍 探索验证集数据: {val_file}")
    print("=" * 80)
    
    try:
        # 读取数据
        df = pd.read_parquet(val_file)
        print(f"✅ 成功读取数据")
        
        # 基本信息
        print(f"\n📊 基本信息:")
        print(f"  样本总数: {len(df):,}")
        print(f"  列数: {len(df.columns)}")
        print(f"  内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # 列信息
        print(f"\n📋 列信息:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 检查必要列
        required_cols = ['content', 'label', 'FT_label']
        print(f"\n🔍 必要列检查:")
        for col in required_cols:
            if col in df.columns:
                print(f"  ✅ {col}: 存在")
            else:
                print(f"  ❌ {col}: 缺失")
        
        # 标签分布
        if 'label' in df.columns:
            print(f"\n📈 'label' 列分布:")
            print(df['label'].value_counts().sort_index())
            print(f"  唯一值: {df['label'].nunique()}")
        
        if 'FT_label' in df.columns:
            print(f"\n📈 'FT_label' 列分布:")
            print(df['FT_label'].value_counts().sort_index())
            print(f"  唯一值: {df['FT_label'].nunique()}")
        
        # 内容长度统计
        if 'content' in df.columns:
            content_lengths = df['content'].str.len()
            print(f"\n📝 内容长度统计:")
            print(f"  平均长度: {content_lengths.mean():.1f} 字符")
            print(f"  中位数长度: {content_lengths.median():.1f} 字符")
            print(f"  最小长度: {content_lengths.min()} 字符")
            print(f"  最大长度: {content_lengths.max():,} 字符")
            print(f"  长度分位数:")
            for p in [25, 50, 75, 90, 95, 99]:
                print(f"    {p}%: {content_lengths.quantile(p/100):,.0f} 字符")
        
        # 检查是否有预测分数列
        score_cols = [col for col in df.columns if 'score' in col.lower() or 'prob' in col.lower() or 'conf' in col.lower()]
        if score_cols:
            print(f"\n📊 发现分数相关列: {score_cols}")
            for col in score_cols:
                print(f"  {col}: {df[col].dtype}, 样本值: {df[col].head(3).tolist()}")
        
        # 样本预览
        print(f"\n👀 数据样本预览:")
        display_cols = ['label', 'FT_label']
        if 'content' in df.columns:
            display_cols.append('content')
        # 添加分数列到预览
        if score_cols:
            display_cols.extend(score_cols[:2])  # 只显示前2个分数列
        
        for col in display_cols:
            if col in df.columns:
                print(f"\n  {col} 列前3个样本:")
                for i, val in enumerate(df[col].head(3)):
                    if col == 'content':
                        # 截断长文本
                        val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                    else:
                        val_str = str(val)
                    print(f"    [{i+1}] {val_str}")
        
        # 标签一致性检查
        if 'label' in df.columns and 'FT_label' in df.columns:
            print(f"\n🔄 标签一致性检查:")
            # 假设 label 0/1 对应 __label__0/__label__1
            df_check = df.copy()
            df_check['label_converted'] = df_check['label'].apply(lambda x: f'__label__{x}')
            consistent = (df_check['label_converted'] == df_check['FT_label']).sum()
            total = len(df_check)
            print(f"  一致样本: {consistent:,}/{total:,} ({consistent/total*100:.1f}%)")
            
            if consistent < total:
                print(f"  不一致样本分布:")
                inconsistent = df_check[df_check['label_converted'] != df_check['FT_label']]
                print(inconsistent[['label', 'FT_label', 'label_converted']].value_counts())
        
        return True
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {val_file}")
        return False
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return False

if __name__ == "__main__":
    explore_validation_data()
