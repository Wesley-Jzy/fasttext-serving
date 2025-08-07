#!/usr/bin/env python3
"""
生成cooking相关的二分类测试数据
分类：cake vs baking (简洁版本)
"""

import pandas as pd
from pathlib import Path

def create_cooking_binary_classification_data(samples_per_class=1000):
    """创建简洁的cake vs baking二分类数据"""
    
    # 正类：蛋糕制作 (cake)
    cake_template = "Mix eggs, flour, sugar, and butter to make a delicious chocolate cake. Bake at 350°F for 30 minutes until fluffy."
    
    # 负类：一般烘焙 (baking) 
    baking_template = "Preheat oven to 375°F. Mix flour, yeast, and salt to make fresh bread. Knead dough and let rise for 1 hour."
    
    # 创建数据 - 只保留核心字段
    data = []
    
    # 添加正类数据（cakes）
    for i in range(samples_per_class):
        data.append({
            "id": f"cake_{i+1}",
            "text": f"{cake_template} (sample {i+1})",
            "category": "cake"
        })
    
    # 添加负类数据（baking）
    for i in range(samples_per_class):
        data.append({
            "id": f"baking_{i+1}",
            "text": f"{baking_template} (sample {i+1})",
            "category": "baking"
        })
    
    return data

def main():
    print("🎂 生成简洁的cake vs baking二分类测试数据...")
    
    # 每类生成1000个样本
    samples_per_class = 1000
    
    # 生成数据
    data = create_cooking_binary_classification_data(samples_per_class)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 打乱数据顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 统计信息
    print(f"📊 数据统计:")
    print(f"   总样本数: {len(df)}")
    print(f"   每类样本数: {samples_per_class}")
    print(f"   类别分布:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"     {category}: {count} ({percentage:.1f}%)")
    
    # 保存为parquet文件
    output_path = Path("cooking_binary_test.parquet")
    df.to_parquet(output_path, index=False)
    print(f"✅ 数据已保存到: {output_path}")
    print(f"📁 文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    
    # 显示数据样例
    print(f"\n📋 简洁的数据结构:")
    print("Cake样例:")
    cake_sample = df[df['category'] == 'cake'].iloc[0]
    print(f"   ID: {cake_sample['id']}")
    print(f"   Text: {cake_sample['text'][:60]}...")
    print(f"   Category: {cake_sample['category']}")
    
    print("Baking样例:")
    baking_sample = df[df['category'] == 'baking'].iloc[0]
    print(f"   ID: {baking_sample['id']}")
    print(f"   Text: {baking_sample['text'][:60]}...")
    print(f"   Category: {baking_sample['category']}")
    
    return output_path

if __name__ == "__main__":
    main() 