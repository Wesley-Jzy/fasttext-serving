# FastText Serving 客户端示例

这个客户端演示了如何使用 FastText serving 进行大规模文本分类的数据处理。当前使用cooking问题分类模型进行测试。

## 功能特性

- 🚀 **并发批处理**: 支持高并发的批量推理请求
- 📊 **多种数据格式**: 支持 Parquet 和 JSON 格式的输入输出
- 🔧 **可扩展框架**: 提供前处理和后处理的扩展点
- 🛡️ **错误容错**: 单个批次失败不影响整体处理
- 📈 **性能监控**: 提供处理时间和吞吐量统计

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方式

### 1. 使用测试数据（cooking分类）

```bash
python client.py --use-test-data --output cooking_results.json
```

### 2. 处理 Parquet 文件

```bash
python client.py --input data.parquet --output results.parquet --batch-size 2000
```

### 3. 自定义配置

```bash
python client.py \
    --url http://your-service-url:8000 \
    --input large_dataset.parquet \
    --output processed_results.parquet \
    --batch-size 5000 \
    --max-concurrent 10
```

## 参数说明

- `--url`: FastText serving 服务地址 (默认: http://localhost:8000)
- `--input`: 输入 Parquet 文件路径
- `--output`: 输出文件路径 (支持 .json 和 .parquet)
- `--batch-size`: 批处理大小 (默认: 1000)
- `--max-concurrent`: 最大并发请求数 (默认: 5)
- `--use-test-data`: 使用内置测试数据而不是文件

## 数据格式

### 输入数据格式（cooking问题）
```json
{
  "id": 1,
  "text": "Which baking dish is best to bake a banana bread?",
  "source": "cooking_forum",
  "author": "home_baker"
}
```

### 输出数据格式
```json
{
  "id": 1,
  "text": "Which baking dish is best to bake a banana bread?",
  "source": "cooking_forum", 
  "author": "home_baker",
  "category_prediction": "baking",
  "category_confidence": 0.85,
  "category_labels": ["baking", "equipment"],
  "category_scores": [0.85, 0.15]
}
```

## Cooking模型分类

当前cooking模型可以识别的问题类型：
- **baking**: 烘焙相关问题
- **equipment**: 厨具设备问题  
- **food-safety**: 食品安全问题
- **preparation**: 食材准备问题
- **cooking-method**: 烹饪方法问题

## 性能优化建议

1. **批处理大小**: 根据文本长度调整，建议 1000-5000
2. **并发数**: 根据服务器性能，建议不超过 CPU 核数的 2 倍
3. **网络超时**: 大批次处理时适当增加超时时间

## 快速测试

```bash
# 基础功能测试
python client.py --use-test-data

# 性能测试
time python client.py --use-test-data --batch-size 2000 --max-concurrent 4

# 查看分类结果
python client.py --use-test-data --output test.json
cat test.json | jq '.[] | {text: .text[0:50], prediction: .category_prediction, confidence: .category_confidence}' | head -5
```

## 扩展开发

### 自定义前处理
在 `DataProcessor.preprocess()` 方法中添加文本清理逻辑：

```python
def preprocess(self, text: str) -> str:
    # 文本清理
    # 格式标准化
    # 特征提取
    return processed_text
```

### 自定义后处理
在 `DataProcessor.postprocess()` 方法中添加结果处理逻辑：

```python
def postprocess(self, prediction_result: Dict, original_data: Dict) -> Dict:
    # 添加置信度阈值过滤
    # 合并多个模型结果
    # 添加业务规则
    return enhanced_result
``` 