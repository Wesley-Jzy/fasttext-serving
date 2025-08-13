#!/usr/bin/env python3
import os
import sys

def test_fasttext_model():
    model_path = '/mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin'
    
    print("🔍 测试 FastText 模型")
    print("=" * 40)
    
    # 1. 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ 模型文件存在: {model_path}")
    print(f"📊 文件大小: {file_size_mb:.1f} MB")
    
    # 2. 尝试加载模型
    try:
        import fasttext
        print("📦 导入FastText库成功")
        
        print("⏳ 加载模型中...")
        model = fasttext.load_model(model_path)
        print("✅ 模型加载成功")
        
        # 3. 获取模型信息
        print(f"\n📊 模型信息:")
        print(f"   向量维度: {model.get_dimension()}")
        
        labels = model.get_labels()
        print(f"   标签数量: {len(labels)}")
        print(f"   标签列表: {labels}")
        
        # 4. 测试预测
        test_cases = [
            "print('hello world')",
            "def function(): pass",
            "import pandas as pd\ndf = pd.read_csv('data.csv')",
            "// This is a comment\nint main() { return 0; }"
        ]
        
        print(f"\n🔍 测试预测 (k=2):")
        for i, test_text in enumerate(test_cases):
            try:
                predictions = model.predict(test_text, k=2)
                labels_pred, scores_pred = predictions
                
                print(f"\n   测试 {i+1}:")
                print(f"   输入: {test_text[:50]}{'...' if len(test_text) > 50 else ''}")
                print(f"   标签: {labels_pred}")
                print(f"   分数: {[f'{s:.4f}' for s in scores_pred]}")
                
            except Exception as e:
                print(f"   ❌ 预测失败: {e}")
        
        # 5. 测试predict-prob风格输出
        print(f"\n🎯 测试 predict-prob 风格:")
        test_text = "print('hello world')"
        try:
            # 获取所有标签的概率
            predictions = model.predict(test_text, k=len(labels))
            labels_pred, scores_pred = predictions
            
            print(f"   输入: {test_text}")
            for label, score in zip(labels_pred, scores_pred):
                clean_label = label.replace('__label__', '')
                print(f"   {clean_label}: {score:.6f}")
                
        except Exception as e:
            print(f"   ❌ predict-prob测试失败: {e}")
            
    except ImportError:
        print("❌ FastText库未安装，请先安装: pip install fasttext")
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")

if __name__ == "__main__":
    test_fasttext_model()
