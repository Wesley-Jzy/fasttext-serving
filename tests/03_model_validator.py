#!/usr/bin/env python3
"""
模型验证脚本
测试FastText模型加载、二分类功能、性能表现
"""
import os
import sys
import time
import json
from typing import List, Tuple, Dict, Any

def test_fasttext_import():
    """测试FastText库导入"""
    print("📦 测试FastText库")
    
    try:
        import fasttext
        print(f"✅ FastText导入成功")
        if hasattr(fasttext, '__version__'):
            print(f"  版本: {fasttext.__version__}")
        return True
    except ImportError as e:
        print(f"❌ FastText导入失败: {e}")
        return False

def load_model(model_path: str):
    """加载FastText模型"""
    print(f"🤖 加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在")
        return None
    
    try:
        import fasttext
        start_time = time.time()
        model = fasttext.load_model(model_path)
        load_time = time.time() - start_time
        
        print(f"✅ 模型加载成功 ({load_time:.2f}秒)")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def analyze_model_info(model):
    """分析模型基本信息"""
    print("🔍 模型信息分析")
    
    if model is None:
        return {}
    
    try:
        # 获取标签
        labels = model.get_labels()
        print(f"📋 模型标签: {labels}")
        print(f"  标签数量: {len(labels)}")
        
        # 检查是否是二分类
        is_binary = len(labels) == 2
        has_label_format = all(label.startswith('__label__') for label in labels)
        
        print(f"🎯 二分类检查:")
        print(f"  是否二分类: {is_binary}")
        print(f"  标签格式正确: {has_label_format}")
        
        if is_binary and has_label_format:
            label0 = labels[0].replace('__label__', '')
            label1 = labels[1].replace('__label__', '')
            print(f"  类别0: {label0}")
            print(f"  类别1: {label1}")
        
        # 获取词汇表大小
        try:
            words = model.get_words()
            print(f"📚 词汇表大小: {len(words)}")
        except:
            print(f"⚠️ 无法获取词汇表信息")
        
        model_info = {
            "labels": labels,
            "is_binary": is_binary,
            "has_correct_format": has_label_format,
            "vocab_size": len(words) if 'words' in locals() else None
        }
        
        return model_info
        
    except Exception as e:
        print(f"❌ 模型信息分析失败: {e}")
        return {}

def test_prediction_basic(model) -> Dict[str, Any]:
    """测试基本预测功能"""
    print("\n🧪 基本预测测试")
    
    if model is None:
        return {}
    
    # 测试样本（代码片段）
    test_samples = [
        "def hello_world():\n    print('Hello, World!')\n",
        "import numpy as np\narray = np.zeros((10, 10))\nprint(array.shape)\n",
        "function calculateSum(a, b) {\n    return a + b;\n}\nconsole.log(calculateSum(5, 3));\n",
        "# This is a comment\n# TODO: implement feature\npass\n",
        "SELECT * FROM users WHERE age > 18 ORDER BY name;\n"
    ]
    
    results = []
    
    for i, text in enumerate(test_samples):
        print(f"\n--- 样本 {i+1} ---")
        print(f"输入: {repr(text[:50])}...")
        
        try:
            # 使用predict方法（获取最可能的标签）
            labels, probs = model.predict(text, k=2)  # 获取top-2结果
            
            print(f"预测结果:")
            for label, prob in zip(labels, probs):
                clean_label = label.replace('__label__', '')
                print(f"  {clean_label}: {prob:.4f}")
            
            results.append({
                "input": text[:100],
                "labels": [l.replace('__label__', '') for l in labels],
                "probabilities": [float(p) for p in probs]
            })
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            results.append({
                "input": text[:100],
                "error": str(e)
            })
    
    return {"test_results": results}

def test_prediction_performance(model, sample_count: int = 100) -> Dict[str, Any]:
    """测试预测性能"""
    print(f"\n⚡ 性能测试 (样本数: {sample_count})")
    
    if model is None:
        return {}
    
    # 生成测试样本
    base_samples = [
        "def function():\n    return True\n",
        "import os\nprint(os.getcwd())\n",
        "class MyClass:\n    def __init__(self):\n        pass\n",
        "try:\n    result = process()\nexcept Exception as e:\n    print(e)\n"
    ]
    
    # 扩展到指定数量
    test_samples = []
    for i in range(sample_count):
        base = base_samples[i % len(base_samples)]
        # 添加一些变化使样本不完全相同
        sample = f"# Sample {i}\n{base}"
        test_samples.append(sample)
    
    print(f"生成 {len(test_samples)} 个测试样本")
    
    # 单次预测性能测试
    print("📊 单次预测测试...")
    single_times = []
    
    for i in range(min(10, sample_count)):
        start_time = time.time()
        try:
            labels, probs = model.predict(test_samples[i], k=2)
            elapsed = time.time() - start_time
            single_times.append(elapsed)
        except Exception as e:
            print(f"⚠️ 单次预测失败: {e}")
    
    if single_times:
        avg_single_time = sum(single_times) / len(single_times)
        print(f"  平均单次预测时间: {avg_single_time*1000:.2f}ms")
        print(f"  预估单次处理速度: {1/avg_single_time:.0f} samples/sec")
    
    # 批量预测性能测试
    print("📊 批量预测测试...")
    batch_sizes = [1, 10, 100] if sample_count >= 100 else [1, min(10, sample_count)]
    
    performance_results = {
        "single_prediction_ms": avg_single_time * 1000 if single_times else None,
        "batch_results": []
    }
    
    for batch_size in batch_sizes:
        if batch_size > len(test_samples):
            continue
            
        batch_samples = test_samples[:batch_size]
        
        start_time = time.time()
        success_count = 0
        
        for sample in batch_samples:
            try:
                labels, probs = model.predict(sample, k=2)
                success_count += 1
            except Exception as e:
                print(f"⚠️ 批量预测中出错: {e}")
        
        elapsed = time.time() - start_time
        
        if elapsed > 0:
            throughput = success_count / elapsed
            print(f"  批量大小 {batch_size}: {throughput:.0f} samples/sec")
            
            performance_results["batch_results"].append({
                "batch_size": batch_size,
                "throughput_per_sec": throughput,
                "success_rate": success_count / batch_size
            })
    
    return performance_results

def test_long_text_handling(model) -> Dict[str, Any]:
    """测试长文本处理能力"""
    print(f"\n📏 长文本处理测试")
    
    if model is None:
        return {}
    
    # 创建不同长度的测试文本
    base_code = """
def complex_function(data):
    '''
    This is a complex function that processes data
    and returns meaningful results.
    '''
    import numpy as np
    import pandas as pd
    
    # Data preprocessing
    cleaned_data = []
    for item in data:
        if item is not None:
            processed = str(item).strip().lower()
            if len(processed) > 0:
                cleaned_data.append(processed)
    
    # Statistical analysis
    if len(cleaned_data) > 0:
        result = {
            'count': len(cleaned_data),
            'unique': len(set(cleaned_data)),
            'avg_length': sum(len(x) for x in cleaned_data) / len(cleaned_data)
        }
        return result
    else:
        return {'error': 'No valid data found'}

# Usage example
sample_data = ['hello', 'world', None, '', 'python', 'fasttext']
result = complex_function(sample_data)
print(result)
"""
    
    length_tests = []
    
    # 测试不同长度
    for multiplier in [1, 5, 10, 50, 100]:
        long_text = base_code * multiplier
        length = len(long_text)
        
        print(f"测试长度: {length:,} 字符 (x{multiplier})")
        
        try:
            start_time = time.time()
            labels, probs = model.predict(long_text, k=2)
            elapsed = time.time() - start_time
            
            print(f"  ✅ 成功 - 耗时: {elapsed:.3f}s")
            print(f"  预测: {labels[0].replace('__label__', '')} ({probs[0]:.3f})")
            
            length_tests.append({
                "length": length,
                "multiplier": multiplier,
                "success": True,
                "time_seconds": elapsed,
                "prediction": labels[0].replace('__label__', ''),
                "confidence": float(probs[0])
            })
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            length_tests.append({
                "length": length,
                "multiplier": multiplier,
                "success": False,
                "error": str(e)
            })
    
    return {"length_tests": length_tests}

def main():
    """主函数"""
    print("🤖 FastText模型验证")
    print("=" * 50)
    
    model_path = "/mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin"
    
    # 1. 测试FastText导入
    if not test_fasttext_import():
        print("❌ FastText库有问题，停止测试")
        return
    
    # 2. 加载模型
    model = load_model(model_path)
    if model is None:
        print("❌ 模型加载失败，停止测试")
        return
    
    # 3. 分析模型信息
    model_info = analyze_model_info(model)
    
    # 4. 基本预测测试
    basic_results = test_prediction_basic(model)
    
    # 5. 性能测试
    performance_results = test_prediction_performance(model, sample_count=50)
    
    # 6. 长文本测试
    long_text_results = test_long_text_handling(model)
    
    # 汇总结果
    validation_results = {
        "model_info": model_info,
        "basic_prediction": basic_results,
        "performance": performance_results,
        "long_text_handling": long_text_results
    }
    
    # 保存结果
    output_file = "model_validation_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 模型验证完成!")
    print(f"📄 详细结果已保存到: {output_file}")
    print(f"\n📋 下一步:")
    print(f"  1. 检查验证结果文件")
    print(f"  2. 执行服务测试: python3 tests/04_service_test.py")

if __name__ == "__main__":
    main()
