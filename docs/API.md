# FastText Serving API Documentation

## 🎯 设计原则

1. **一致性**: Rust和Python实现提供完全相同的API
2. **标准化**: 遵循RESTful API设计规范
3. **兼容性**: 保持向后兼容
4. **可扩展**: 易于添加新功能

## 📡 HTTP API

### 1. 预测接口

#### `POST /predict`

**功能**: 对文本进行分类预测

**请求**:
```json
{
  "url": "POST /predict?k=1&threshold=0.0",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": ["text1", "text2", "text3"]
}
```

**参数**:
- `k` (可选): 返回top-k个标签，默认1
- `threshold` (可选): 预测阈值，默认0.0

**响应**:
```json
[
  {
    "labels": ["__label__0"],
    "scores": [0.9987]
  },
  {
    "labels": ["__label__1"],
    "scores": [0.8234]
  }
]
```

**错误响应**:
```json
{
  "error": "Bad Request",
  "message": "Empty text input",
  "code": 400
}
```

### 2. 健康检查

#### `GET /health`

**功能**: 检查服务状态

**响应**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "implementation": "python",
  "version": "1.0.0",
  "model_path": "/path/to/model.bin"
}
```

### 3. 句向量接口

#### `POST /sentence-vector`

**功能**: 获取文本的向量表示

**请求**:
```json
["text1", "text2"]
```

**响应**:
```json
[
  [0.1, 0.2, 0.3, ...],
  [0.4, 0.5, 0.6, ...]
]
```

## 🔧 标签格式规范

### ⚠️ 重要变更：保持完整标签格式

**之前的错误处理**:
```python
# ❌ 错误：移除__label__前缀
labels.push(pred.label.trim_start_matches("__label__").to_string())
# 结果：["0", "1"] 
```

**正确的处理**:
```python
# ✅ 正确：保持完整格式
labels = ["__label__0", "__label__1"]
# 结果：["__label__0", "__label__1"]
```

## 📊 批处理规范

### 请求限制
- **最大批次大小**: 1000个文本
- **最大文本长度**: 10MB (可配置)
- **最大请求大小**: 500MB (可配置)

### 错误处理
- **单文本失败**: 返回错误标记，不影响其他文本
- **批次部分失败**: 继续处理其他文本
- **批次完全失败**: 返回400错误

## 🚀 性能指标

### 目标性能
- **Python实现**: 2000-3000 samples/sec
- **Rust实现**: 4000-5000 samples/sec
- **响应时间**: <100ms (单文本)
- **内存使用**: <2GB

### 监控指标
- 请求数/秒
- 平均响应时间
- 错误率
- 内存使用率
- CPU使用率

## 🔒 错误码定义

| 错误码 | 说明 | 示例 |
|--------|------|------|
| 400 | 请求错误 | 空文本、格式错误 |
| 413 | 请求过大 | 超出大小限制 |
| 500 | 服务器错误 | 模型加载失败 |
| 503 | 服务不可用 | 模型未加载 |

## 📝 客户端示例

### Python客户端
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json=['print("hello world")'],
    params={'k': 1, 'threshold': 0.0}
)
result = response.json()
print(result[0]['labels'][0])  # __label__1
```

### curl示例
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '["def hello(): return \"world\""]'
```

## 🧪 测试用例

### 标准测试用例
1. **单文本预测**: 确保基本功能正常
2. **批量预测**: 测试并发处理能力
3. **边界条件**: 空文本、超长文本、特殊字符
4. **错误处理**: 各种异常情况的处理
5. **性能测试**: 吞吐量和延迟测试

### 兼容性测试
1. **API一致性**: Rust vs Python实现
2. **结果一致性**: 相同输入产生相同输出
3. **错误一致性**: 相同错误产生相同响应
