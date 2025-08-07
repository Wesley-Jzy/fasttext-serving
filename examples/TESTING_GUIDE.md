# FastText Serving 测试指南

本指南将帮你完成从构建到测试的完整流程，验证cooking问题分类功能。

## 🚀 快速开始

### 1. 准备模型

使用现有的cooking模型：

```bash
# 检查现有模型
ls -la models/
file models/cooking.model.bin

# 确保模型文件存在和可读
chmod 644 models/cooking.model.bin
```

### 2. 构建和运行 FastText Serving

#### 方法 A: Docker 方式（推荐）

```bash
# 构建 Docker 镜像
docker build -f Dockerfile.test -t fasttext-serving:test .

# 运行服务（挂载模型）
docker run -d \
  --name fasttext-serving \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  fasttext-serving:test \
  --model /app/models/cooking.model.bin \
  --address 0.0.0.0 \
  --port 8000 \
  --max-request-size 500

# 检查服务状态
docker logs fasttext-serving
curl http://localhost:8000/health
```

#### 方法 B: 本地编译方式

```bash
# 编译（需要 Rust 环境）
cargo build --release

# 运行服务
./target/release/fasttext-serving \
  --model models/cooking.model.bin \
  --address 127.0.0.1 \
  --port 8000 \
  --max-request-size 500 \
  --workers 4
```

### 3. 测试客户端

```bash
cd examples

# 使用测试数据快速验证
python client.py --use-test-data --output cooking_test_results.json

# 查看结果
cat cooking_test_results.json | jq '.[] | {id, text, category_prediction, category_confidence}' | head -10
```

## 📊 性能测试

### 批处理能力测试

```bash
# 测试大批量处理（600个样本）
python client.py \
  --use-test-data \
  --batch-size 5000 \
  --max-concurrent 8 \
  --output large_batch_cooking_results.parquet

# 监控处理性能
time python client.py \
  --use-test-data \
  --batch-size 2000 \
  --max-concurrent 4 \
  --output performance_cooking_test.json
```

### 并发压力测试

```bash
# 多进程并发测试
for i in {1..5}; do
  python client.py \
    --use-test-data \
    --batch-size 1000 \
    --max-concurrent 2 \
    --output "concurrent_cooking_test_$i.json" &
done
wait

# 检查所有结果
ls -la concurrent_cooking_test_*.json
```

## 🔧 配置优化

### 服务端配置

根据你的硬件资源调整参数：

```bash
# 高性能配置（多核服务器）
fasttext-serving \
  --model models/cooking.model.bin \
  --workers 16 \
  --max-request-size 1000 \
  --address 0.0.0.0 \
  --port 8000

# 内存受限配置
fasttext-serving \
  --model models/cooking.model.bin \
  --workers 4 \
  --max-request-size 200 \
  --address 127.0.0.1 \
  --port 8000
```

### 客户端配置

```bash
# 高吞吐量配置
python client.py \
  --batch-size 10000 \
  --max-concurrent 16 \
  --use-test-data

# 稳定性优先配置  
python client.py \
  --batch-size 1000 \
  --max-concurrent 4 \
  --use-test-data
```

## 🐛 问题排查

### 常见问题

#### 1. 模型加载失败

```bash
# 检查模型文件
ls -la models/
file models/cooking.model.bin

# 检查文件权限
chmod 644 models/cooking.model.bin
```

#### 2. 请求超时

```bash
# 增加客户端超时时间
# 修改 client.py 中的 timeout 参数
timeout = aiohttp.ClientTimeout(total=600)  # 10分钟
```

#### 3. 内存不足

```bash
# 监控内存使用
docker stats fasttext-serving

# 减少批处理大小
python client.py --batch-size 500 --max-concurrent 2
```

#### 4. JSON 解析错误

```bash
# 检查请求大小
python -c "
import json
data = ['test'] * 100000
print(f'Size: {len(json.dumps(data)) / 1024 / 1024:.1f} MB')
"

# 调整服务端限制
fasttext-serving --max-request-size 1000  # 1GB
```

### 日志调试

```bash
# 启用详细日志
export RUST_LOG=fasttext_serving=debug
fasttext-serving --model models/cooking.model.bin

# Docker 日志
docker logs -f fasttext-serving

# 客户端详细输出
python client.py --use-test-data -v
```

## 📈 性能基准

### 预期性能指标（cooking模型）

| 配置 | 批处理大小 | 并发数 | 预期吞吐量 | 内存使用 |
|------|-----------|--------|------------|----------|
| 单核 | 1000 | 2 | 500 texts/sec | < 1GB |
| 4核 | 5000 | 8 | 2000 texts/sec | < 2GB |
| 8核 | 10000 | 16 | 5000 texts/sec | < 4GB |

### 优化建议

1. **批处理大小**: 
   - 短文本问题: 5000-10000
   - 长文本问题: 1000-2000

2. **并发配置**:
   - 客户端并发 = CPU核数 × 2
   - 服务端工作线程 = CPU核数

3. **内存管理**:
   - 单个文本限制: 1MB
   - 请求大小限制: 500MB-1GB
   - 预留系统内存: 20%

## 🔄 自动化测试

创建自动化测试脚本：

```bash
#!/bin/bash
# test_cooking_automation.sh

echo "🚀 Starting FastText Serving cooking test..."

# 1. 构建和启动服务
docker build -f Dockerfile.test -t fasttext-serving:test .
docker run -d --name test-serving -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  fasttext-serving:test \
  --model /app/models/cooking.model.bin

# 2. 等待服务启动
sleep 10

# 3. 运行测试
cd examples
python client.py --use-test-data --output auto_cooking_test_results.json

# 4. 验证结果
if [ -f "auto_cooking_test_results.json" ]; then
  echo "✅ Test completed successfully"
  python -c "
import json
with open('auto_cooking_test_results.json') as f:
    data = json.load(f)
    print(f'Processed {len(data)} samples')
    predictions = [item['category_prediction'] for item in data]
    print(f'Categories found: {set(predictions)}')
"
else
  echo "❌ Test failed"
  exit 1
fi

# 5. 清理
docker stop test-serving
docker rm test-serving

echo "🎉 Cooking automation test completed!"
```

使用方法：

```bash
chmod +x test_cooking_automation.sh
./test_cooking_automation.sh
```

## 🚢 生产部署准备

### K8s 部署配置

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fasttext-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fasttext-serving
  template:
    metadata:
      labels:
        app: fasttext-serving
    spec:
      containers:
      - name: fasttext-serving
        image: hub.anuttacon.com/infra/fasttext-serving:latest
        ports:
        - containerPort: 8000
        args:
          - "--model"
          - "/app/models/cooking.model.bin"
          - "--address"
          - "0.0.0.0"
          - "--workers"
          - "4"
          - "--max-request-size"
          - "500"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: fasttext-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: fasttext-serving-service
spec:
  selector:
    app: fasttext-serving
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 示例测试数据

cooking模型可以分类的问题类型包括：
- **baking**: 烘焙相关问题
- **equipment**: 厨具设备问题  
- **food-safety**: 食品安全问题
- **preparation**: 食材准备问题
- **cooking-method**: 烹饪方法问题

这个测试指南专门针对cooking模型进行了优化，现在你可以按照指南逐步验证 FastText serving 的功能。 