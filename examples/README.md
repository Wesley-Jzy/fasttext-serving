# FastText Serving Client 测试

FastText 服务的高并发客户端测试工具，用于验证模型推理服务的性能和准确性。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成测试数据
```bash
python3 create_test_data.py
```

### 3. 运行测试
```bash
python3 client.py \
  --url http://fasttext-serving-4nodes.serving.va-mlp.anuttacon.com \
  --input cooking_binary_test.parquet \
  --output cake_baking_results.json \
  --batch-size 500 \
  --max-concurrent 20
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `client.py` | 高并发FastText客户端，支持批量处理和异步请求 |
| `create_test_data.py` | 生成cake vs baking二分类测试数据 |
| `requirements.txt` | Python依赖库 |
| `cooking_binary_test.parquet` | 测试数据文件（2000条样本） |
| `test_service.py` | 服务验证脚本，用于快速检查服务状态 |

## 🎯 测试数据

- **类别**: cake vs baking 二分类
- **样本数**: 2000条（每类1000条）
- **数据结构**: 
  ```json
  {
    "id": "cake_1",
    "text": "Mix eggs, flour, sugar...",
    "category": "cake"
  }
  ```

## ⚙️ 配置参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--url` | FastText服务地址 | localhost:8000 | 你的K8s服务URL |
| `--input` | 输入数据文件 | - | cooking_binary_test.parquet |
| `--output` | 输出结果文件 | - | 结果文件路径 |
| `--batch-size` | 批次大小 | 1000 | 500-2000 |
| `--max-concurrent` | 并发数 | 5 | 10-50 |

## 📊 预期结果

### 性能指标
- **吞吐量**: 300K+ samples/sec
- **处理时间**: 2000样本约1-3秒
- **成功率**: 100%

### 输出格式
```json
[
  {
    "id": "cake_1",
    "text": "Mix eggs, flour, sugar...",
    "category": "cake",
    "category_prediction": "baking",
    "category_confidence": 0.85,
    "category_labels": ["baking", "cooking"],
    "category_scores": [0.85, 0.15]
  }
]
```

## 🔧 扩展开发

### 前处理扩展
在 `client.py` 的 `preprocess()` 方法中添加文本清理逻辑：
```python
def preprocess(self, text: str) -> str:
    # 添加你的前处理逻辑
    text = text.lower().strip()
    return text
```

### 后处理扩展
在 `client.py` 的 `postprocess()` 方法中添加业务逻辑：
```python
def postprocess(self, prediction_result, original_data):
    result = original_data.copy()
    result.update({
        "category_prediction": prediction_result["prediction"],
        "category_confidence": prediction_result["confidence"],
        "category_labels": prediction_result["labels"],
        "category_scores": prediction_result["scores"],
        # 添加你的业务字段
        "quality_score": self.calculate_quality(prediction_result),
        "is_high_confidence": prediction_result["confidence"] > 0.8
    })
    return result
```
